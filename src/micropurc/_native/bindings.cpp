// nanobind bindings and the streaming batch estimator.
//
// Composes the building blocks from common.hpp, activeset.hpp, and schur.hpp
// into three stages, run per OD group with work stealing across std::threads:
//   1. solve_od_group          — forward QP for one OD pair
//   2. build_active_projection — dual-slack active set, acyclicity check,
//                                grounding, dense incidence extraction
//   3. accumulate_trip_sparse  — each trip's Schur contribution into H, g, u
//
// The stages stream: a group's flow is consumed by its trips as soon as it is
// solved, so no per-trip (N x L) flow array is ever materialized and adding
// trips to an OD already in the batch costs no extra memory.

#include <unordered_map>

#include "activeset.hpp"
#include "common.hpp"
#include "route_sampler.hpp"
#include "schur.hpp"

#ifdef HAVE_PIQP
struct ODSolveResult {
    bool ok;
    // IPM iterations on success; the negated PIQP status code on failure.
    int qp_iters;
};

// Solves the forward QP for one OD pair on a solver instance reused across the
// thread's groups. b_vec is carried between calls and only the two entries of
// the previous OD are cleared, so the setup cost is paid once per thread and
// each further group costs an update(c, b).

static ODSolveResult solve_od_group(PIQPSolver &solver, bool &first_solve,
                                    const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &P,
                                    const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &A_eq,
                                    const Eigen::VectorXd &x_lo, Eigen::VectorXd &b_vec, int o,
                                    int d, int &prev_o, int &prev_d, const double *c_var, int L,
                                    double *x_out, double *lam_out, double *zbl_out, int V,
                                    bool use_warm, const double *x_warm, const double *y_warm,
                                    const double *zbl_warm) {
    if (prev_o >= 0) {
        b_vec(prev_o) = 0.0;
        b_vec(prev_d) = 0.0;
    }
    b_vec(o) = -1.0;
    b_vec(d) = 1.0;
    prev_o = o;
    prev_d = d;

    Eigen::Map<const Eigen::VectorXd> c_map(c_var, L);
    if (first_solve) {
        solver.setup(P, c_map, A_eq, b_vec, piqp::nullopt, piqp::nullopt, piqp::nullopt, x_lo,
                     piqp::nullopt);
        first_solve = false;
    } else {
        solver.update(piqp::nullopt, c_map, piqp::nullopt, b_vec);
    }

    // Warm-start the IPM from this group's previous-iteration primal-dual point.
    // Strict convexity guarantees the same minimizer as a cold solve; only the
    // iteration count changes. Groups with no prior point solve cold.
    solver.settings().warm_start = use_warm;
    if (use_warm) {
        Eigen::Map<const Eigen::VectorXd> xw(x_warm, L), yw(y_warm, V), zw(zbl_warm, L);
        solver.set_warm_start(xw, yw, piqp::nullopt, piqp::nullopt, zw, piqp::nullopt);
    }

    piqp::Status status = solver.solve();
    bool solve_ok = (status == piqp::PIQP_SOLVED);
    if (solve_ok) {
        // The primal is clamped at its own lower bound: the IPM leaves inactive
        // links at tiny negative values, and downstream code treats a flow as a
        // probability.
        const auto &res = solver.result();
        for (int j = 0; j < L; j++) x_out[j] = std::max(0.0, res.x(j));
        for (int i = 0; i < V; i++) lam_out[i] = res.y(i);
        for (int j = 0; j < L; j++) zbl_out[j] = res.z_bl(j);
        return {true, (int)res.info.iter};
    }
    return {false, -(int)status};
}
#endif

// Turns one solved group into everything the Schur step needs: the active link
// list, the grounded node rows, the dense incidence A_local over those rows and
// columns, and (when Z_v is given) the active design rows.
//
// A link is active when its dual slack c_j + (A^T lam)_j is below the cutoff,
// which is 0 and tightens to -slack_tol for a positive tolerance. Indices come
// out ascending, which is the canonical order hash_active_set assumes.
//
// Returns valid = false with drop_reason set when the group cannot contribute:
// no active links (1), a single active link (2), a cyclic active subgraph (3),
// or an empty grounding (4). The buffers are caller-owned and reused across
// groups on a thread; their contents after a drop are unspecified.
static ActiveProjection build_active_projection(
    const double *lam_od, const double *c_var, int L, int V, int K, int dest,
    const int64_t *A_indptr, const int64_t *A_indices, const double *A_data, const int *tail,
    const int *head, const double *Z_v, double slack_tol, std::vector<int> &active_idx,
    std::vector<int> &kept_rows, std::vector<double> &A_local_buf, std::vector<double> &Z_act_buf) {
    active_idx.clear();
    for (int j = 0; j < L; j++) {
        double at_lam_j = 0.0;
        for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
            at_lam_j += A_data[p] * lam_od[A_indices[p]];
        double slack_j = c_var[j] + at_lam_j;
        if (slack_j < (slack_tol > 0.0 ? -slack_tol : 0.0)) active_idx.push_back(j);
    }

    int n_active = (int)active_idx.size();
    if (n_active <= 1) return {n_active, 0, false, n_active == 0 ? 1 : 2};

    bool is_dag = false, is_connected = false;
    check_dag_connectivity(tail, head, n_active, active_idx.data(), dest, V, is_dag, is_connected);
    if (!is_dag) return {n_active, 0, false, 3};

    int n_components = 0;
    find_kept_rows_robust(tail, head, active_idx.data(), n_active, dest, V, A_indptr, A_indices,
                          kept_rows, n_components);

    int m_rows = (int)kept_rows.size();
    if (m_rows == 0) return {n_active, 0, false, 4};

    A_local_buf.resize(m_rows * n_active);
    extract_dense_submatrix(A_indptr, A_indices, A_data, active_idx.data(), n_active,
                            kept_rows.data(), m_rows, A_local_buf.data());

    if (Z_v) {
        Z_act_buf.resize(n_active * K);
        for (int i = 0; i < n_active; i++)
            for (int k = 0; k < K; k++) Z_act_buf[i * K + k] = Z_v[active_idx[i] * K + k];
    } else {
        Z_act_buf.clear();
    }

    return {n_active, m_rows, true};
}

static double active_threshold_native(nb::ndarray<double, nb::shape<-1>, nb::c_contig> x_arr,
                                      double threshold_factor, double threshold_min) {
    int n = (int)x_arr.shape(0);
    const double *x = x_arr.data();
    if (n == 0) return threshold_min;
    double max_x = x[0];
    for (int i = 1; i < n; i++) max_x = std::max(max_x, x[i]);
    return std::max(threshold_min, threshold_factor * max_x);
}

static nb::tuple check_dag_connectivity_native(
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> tail_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> head_arr, int dest, int V) {
    int n = (int)tail_arr.shape(0);
    const int64_t *tail64 = tail_arr.data();
    const int64_t *head64 = head_arr.data();
    std::vector<int> tail(n), head(n);
    for (int i = 0; i < n; i++) {
        tail[i] = (int)tail64[i];
        head[i] = (int)head64[i];
    }
    bool is_dag = false, is_connected = false;
    check_dag_connectivity_edges(tail.data(), head.data(), n, dest, V, is_dag, is_connected);
    return nb::make_tuple(is_dag, is_connected);
}

static nb::tuple find_kept_rows_native(nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> tail_arr,
                                       nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> head_arr,
                                       int dest, int V, const std::string &projection_mode) {
    int n = (int)tail_arr.shape(0);
    const int64_t *tail64 = tail_arr.data();
    const int64_t *head64 = head_arr.data();
    std::vector<int> tail(n), head(n), kept_rows;
    for (int i = 0; i < n; i++) {
        tail[i] = (int)tail64[i];
        head[i] = (int)head64[i];
    }

    int n_components = 0;
    if (projection_mode != "robust_pinv")
        throw std::invalid_argument(
            "projection_mode must be 'robust_pinv' (legacy_drop was removed)");
    find_kept_rows_robust_edges(tail.data(), head.data(), n, dest, V, kept_rows, n_components);
    return nb::make_tuple(make_int64_array(kept_rows), n_components, !kept_rows.empty());
}

static nb::dict determine_active_set_native(
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indptr_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indices_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> A_data_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> tail_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> head_arr, int V, int L, int dest,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> c_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> lam_eq_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> scale_m_arr, double slack_tol,
    const std::string &projection_mode) {
    const int64_t *A_indptr = A_indptr_arr.data();
    const int64_t *A_indices = A_indices_arr.data();
    const double *A_data = A_data_arr.data();
    const int64_t *tail64 = tail_arr.data();
    const int64_t *head64 = head_arr.data();
    const double *c = c_arr.data();
    const double *lam = lam_eq_arr.data();
    const double *scale_m = scale_m_arr.data();

    std::vector<int> tail(L), head(L), active_idx, kept_rows;
    for (int i = 0; i < L; i++) {
        tail[i] = (int)tail64[i];
        head[i] = (int)head64[i];
    }

    // Raw dual slack s_j = c_j + (A^T lam)_j. A link is active (carries flow)
    // iff s_j < cutoff, which is 0 and tightens to -slack_tol for a positive
    // tolerance. The full slack vector is returned so callers can form the
    // active-set margin |s_j| / ||Z_j|| themselves; that is done in Python,
    // where the per-variant design rows Z_j are known (see core.activeset).
    // scale_m is accepted for signature compatibility with that caller and is
    // unused: the active set is decided by the sign of the slack alone.
    (void)scale_m;
    std::vector<double> slack(L);
    double cutoff = slack_tol > 0.0 ? -slack_tol : 0.0;
    for (int j = 0; j < L; j++) {
        double at_lam_j = 0.0;
        for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
            at_lam_j += A_data[p] * lam[A_indices[p]];
        double slack_j = c[j] + at_lam_j;
        slack[j] = slack_j;
        if (slack_j < cutoff) active_idx.push_back(j);
    }

    nb::dict out;
    out["indices"] = make_int64_array(active_idx);
    out["slack"] = make_double_array(slack);
    out["kept_node_rows"] = make_int64_array(kept_rows);
    out["num_components"] = 0;
    out["valid"] = false;
    out["drop_reason"] = "";

    if (active_idx.empty()) {
        out["drop_reason"] = "no_active_links";
        return out;
    }
    if (active_idx.size() <= 1) {
        out["drop_reason"] = "trivial_flow";
        return out;
    }

    bool is_dag = false, is_connected = false;
    check_dag_connectivity(tail.data(), head.data(), (int)active_idx.size(), active_idx.data(),
                           dest, V, is_dag, is_connected);

    if (projection_mode != "robust_pinv")
        throw std::invalid_argument(
            "projection_mode must be 'robust_pinv' (legacy_drop was removed)");
    int n_components = 0;
    if (!is_dag) {
        out["drop_reason"] = "not_dag";
        return out;
    }
    find_kept_rows_robust(tail.data(), head.data(), active_idx.data(), (int)active_idx.size(), dest,
                          V, A_indptr, A_indices, kept_rows, n_components);
    if (kept_rows.empty()) {
        out["drop_reason"] = "no_kept_rows_robust";
        return out;
    }

    out["kept_node_rows"] = make_int64_array(kept_rows);
    out["num_components"] = n_components;
    out["valid"] = true;
    return out;
}

static nb::tuple schur_contribution_native(
    nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> Z_arr,
    nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> A_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> y_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> scale_arr, nb::object x_hat_obj) {
    int e = (int)Z_arr.shape(0);
    int K = (int)Z_arr.shape(1);
    int m = (int)A_arr.shape(0);
    const double *Z = Z_arr.data();
    const double *A = A_arr.data();
    const double *y = y_arr.data();
    const double *scale = scale_arr.data();

    std::vector<double> w_y(e), w_r;
    for (int i = 0; i < e; i++) w_y[i] = scale[i] * y[i];

    double *w_r_ptr = nullptr;
    if (!x_hat_obj.is_none()) {
        auto x_hat_arr = nb::cast<nb::ndarray<double, nb::shape<-1>, nb::c_contig>>(x_hat_obj);
        const double *x_hat = x_hat_arr.data();
        w_r.resize(e);
        for (int i = 0; i < e; i++) w_r[i] = scale[i] * (y[i] - x_hat[i]);
        w_r_ptr = w_r.data();
    }

    double *H = new double[(int64_t)K * K]();
    double *g = new double[K]();
    double *u = w_r_ptr ? new double[K]() : nullptr;

    int status = schur_contribution(Z, A, w_y.data(), w_r_ptr, e, m, K, H, g, u);
    if (status != 0) {
        delete[] H;
        delete[] g;
        if (u) delete[] u;
        throw std::runtime_error("Schur contribution Cholesky failed");
    }

    nb::capsule H_owner(H, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule g_owner(g, [](void *p) noexcept { delete[] (double *)p; });
    size_t H_shape[2] = {(size_t)K, (size_t)K};
    size_t g_shape[1] = {(size_t)K};
    auto H_out = nb::ndarray<nb::numpy, double>(H, 2, H_shape, H_owner);
    auto g_out = nb::ndarray<nb::numpy, double>(g, 1, g_shape, g_owner);

    nb::object u_out = nb::none();
    if (u) {
        nb::capsule u_owner(u, [](void *p) noexcept { delete[] (double *)p; });
        size_t u_shape[1] = {(size_t)K};
        u_out = nb::cast(nb::ndarray<nb::numpy, double>(u, 1, u_shape, u_owner));
    }
    return nb::make_tuple(H_out, g_out, u_out);
}

#ifdef HAVE_PIQP
// General term opcodes for the lazy design. A term evaluates to a per-link,
// per-variant regressor that is linear in the parameter it maps to.
enum LazyTermCode {
    TERM_COLUMN = 0,              // link attribute column `a`
    TERM_PRODUCT = 1,             // product of columns `a` and `b` (interaction)
    TERM_COVARIATE_INDICATOR = 2  // column `a` gated by 1[covariate `cov` `op` thr]
};

// Comparison used by TERM_COVARIATE_INDICATOR.
enum CovOp { COV_LE = 0, COV_GE = 1, COV_EQ = 2, COV_LT = 3, COV_GT = 4 };

class NativePIQPContext {
   public:
    int V, L;
    std::vector<int64_t> A_indptr, A_indices;
    std::vector<double> A_data, scale_m;
    std::vector<int> tail, head;
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> P_eigen;
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_eigen;
    Eigen::VectorXd x_lower;

    // Warm-start state, persisted across KM iterations.
    // The forward QP is strictly convex (P = diag(scale), scale > 0), so its
    // minimizer is unique: warm-starting the IPM leaves the solution alone and
    // moves only the iteration count. Each OD-group's primal-dual point
    // (x, y_eq, z_bl) is stored under its group key and re-injected via
    // set_warm_start on the next KM iteration, whose c = Zβ is nearby. These are
    // members of the persistent context, so the (n_groups × (2L + V)) state
    // never crosses the Python boundary.
    std::vector<int64_t> warm_keys_;  // sorted group keys from the previous call
    std::vector<double> warm_x_;      // n_prev × L  primal flows
    std::vector<double> warm_y_;      // n_prev × V  equality duals (node potentials)
    std::vector<double> warm_zbl_;    // n_prev × L  lower-bound duals
    // Persistent write buffers, ping-ponged with warm_*_ each Phi call. The large
    // n_unique×L / n_unique×V arrays are allocated and zero-filled once per fit,
    // and again only when the group count changes; on a large network the
    // per-call zero-fill runs to tens of GB and, being memory-bound, dominated
    // the cost of an iteration whose solves were all certified.
    // A slot left unwritten by a dropped group therefore carries a stale point
    // from an earlier iteration. That is safe: such slots are read only as
    // warm starts, which leave the strictly convex QP's optimum unchanged.
    std::vector<double> warm_x_next_, warm_y_next_, warm_zbl_next_;
    bool warm_enabled_ = true;

    // Certified Tiered Solve (CTS).
    // Tier 1 guesses that a KM step left this group's active set unchanged. It
    // re-solves on the cached active set through the grounded active-subgraph
    // Laplacian (cached LLT), extends the dual with the cached potentials, and
    // then checks an exact full KKT certificate: x_A > tau_p on the active links
    // and slack > tau_d on all L inactive ones.
    // Strict convexity makes those conditions sufficient, so a pass identifies
    // the unique global optimum and the IPM is skipped; a failure falls back to
    // the warm IPM. The certificate is checked in full and independently of how
    // the candidate was built, so a defect in the construction costs spurious
    // fallbacks and can never yield a wrong answer.
    struct CTSCache {
        bool valid = false;
        std::vector<int> act;             // active link ids (sorted)
        std::vector<int> touched;         // touched node ids (sorted)
        std::vector<int> keep;            // touched-local indices kept (ground dropped)
        int ground_node = -1;             // grounded node (single component only)
        Eigen::LLT<Eigen::MatrixXd> llt;  // grounded Laplacian factor
        std::vector<int> bnd;             // boundary arcs (exactly one endpoint touched)
        std::vector<char> bnd_tail_touched;
    };
    std::vector<CTSCache> cts_prev_;  // indexed like warm_keys_
    bool cts_enabled_ = true;         // certified, so it only skips IPMs; answers are fixed
    double cts_tau_p_ = 1e-8, cts_tau_d_ = 1e-8;
    // Sign convention of A relative to the tail/head arrays: sigma = A[tail_j, j],
    // so (A'y)_j = sigma * (y[tail_j] - y[head_j]). Detected and validated in the
    // constructor. Incidence builders disagree on which endpoint carries the
    // positive entry, and a dataset loader may label tail/head opposite to
    // Network.from_csv/grid, so this MUST be read off the matrix: a hard-coded
    // sigma mis-signs the (A'y)_j term of every slack the repair step reasons
    // about. A network whose incidence is anything other than
    // a uniform +-1 pattern leaves cts_conv_ok_ = false, tier 1 never runs, and
    // every solve takes the warm IPM path.
    double cts_sign_ = 1.0;
    bool cts_conv_ok_ = false;
    // node -> incident arc ids (both endpoints), CSR over V with 2L entries
    std::vector<int64_t> nadj_indptr_;
    std::vector<int> nadj_arcs_;

    // Clear persisted warm-start state (e.g. at the start of a fresh fit()).
    void reset_warm_start() {
        warm_keys_.clear();
        warm_x_.clear();
        warm_y_.clear();
        warm_zbl_.clear();
        warm_x_next_.clear();
        warm_y_next_.clear();
        warm_zbl_next_.clear();
        cts_prev_.clear();
    }

    NativePIQPContext(nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indptr_arr,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indices_arr,
                      nb::ndarray<double, nb::shape<-1>, nb::c_contig> A_data_arr,
                      nb::ndarray<double, nb::shape<-1>, nb::c_contig> scale_m_arr,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> tail_arr,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> head_arr, int V_in,
                      int L_in)
        : V(V_in), L(L_in), P_eigen(L_in, L_in), A_eigen(V_in, L_in), x_lower(L_in) {
        A_indptr.assign(A_indptr_arr.data(), A_indptr_arr.data() + A_indptr_arr.shape(0));
        A_indices.assign(A_indices_arr.data(), A_indices_arr.data() + A_indices_arr.shape(0));
        A_data.assign(A_data_arr.data(), A_data_arr.data() + A_data_arr.shape(0));
        scale_m.assign(scale_m_arr.data(), scale_m_arr.data() + scale_m_arr.shape(0));

        tail.resize(L);
        head.resize(L);
        const int64_t *tail64 = tail_arr.data();
        const int64_t *head64 = head_arr.data();
        for (int i = 0; i < L; i++) {
            tail[i] = (int)tail64[i];
            head[i] = (int)head64[i];
        }

        std::vector<Eigen::Triplet<double>> p_trips;
        p_trips.reserve(L);
        for (int j = 0; j < L; j++) p_trips.push_back({j, j, scale_m[j]});
        P_eigen.setFromTriplets(p_trips.begin(), p_trips.end());

        int nnz = (int)A_indptr[L];
        std::vector<Eigen::Triplet<double>> a_trips;
        a_trips.reserve(nnz);
        for (int j = 0; j < L; j++)
            for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
                a_trips.push_back({(int)A_indices[p], j, A_data[p]});
        A_eigen.setFromTriplets(a_trips.begin(), a_trips.end());
        x_lower.setZero();

        // Detect the incidence sign convention (see cts_sign_). Every link must
        // have exactly two +-1 entries, one at tail[j] and one at head[j], with
        // the same sign at the tail across ALL links. Any deviation disables CTS
        // and sends every solve down the IPM path. Self-loops are excluded here
        // too, since tail[j] == head[j] makes the two entries indistinguishable.
        cts_conv_ok_ = (L > 0);
        for (int j = 0; j < L && cts_conv_ok_; j++) {
            if (A_indptr[j + 1] - A_indptr[j] != 2 || tail[j] == head[j]) {
                cts_conv_ok_ = false;
                break;
            }
            double a_tail = 0.0, a_head = 0.0;
            for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++) {
                if ((int)A_indices[p] == tail[j]) a_tail = A_data[p];
                if ((int)A_indices[p] == head[j]) a_head = A_data[p];
            }
            if (std::abs(a_tail) != 1.0 || a_head != -a_tail) {
                cts_conv_ok_ = false;
                break;
            }
            if (j == 0)
                cts_sign_ = a_tail;
            else if (a_tail != cts_sign_) {
                cts_conv_ok_ = false;
                break;
            }
        }

        // node -> incident arcs adjacency, CSR over V with 2L entries, so the CTS
        // repair worklist can enqueue every arc touching a node it just moved.
        nadj_indptr_.assign(V + 1, 0);
        for (int j = 0; j < L; j++) {
            nadj_indptr_[tail[j] + 1]++;
            nadj_indptr_[head[j] + 1]++;
        }
        for (int v = 0; v < V; v++) nadj_indptr_[v + 1] += nadj_indptr_[v];
        nadj_arcs_.resize(2 * (size_t)L);
        std::vector<int64_t> fill(nadj_indptr_.begin(), nadj_indptr_.end() - 1);
        for (int j = 0; j < L; j++) {
            nadj_arcs_[fill[tail[j]]++] = j;
            nadj_arcs_[fill[head[j]]++] = j;
        }
    }

    // Build the CTS cache for one group from the just-solved active set.
    // Requires a single-component active subgraph, recognised by exactly one
    // touched node having been grounded away (m - m_rows == 1); multi-component
    // groups leave the cache invalid and always take the IPM path.
    // A_local is the dense (m_rows x n_active) grounded incidence extracted by
    // build_active_projection, whose rows follow kept_rows. `keep` maps each of
    // its rows to a touched-local index, which is what lets the tier-1 solve
    // address the factorization by node.
    // Leaving early with cc.valid still false is always safe: it only means the
    // next iteration solves this group with the IPM.
    void cts_build(CTSCache &cc, const std::vector<int> &active_idx, int n_active,
                   const std::vector<int> &kept_rows, const double *A_local, int m_rows,
                   std::vector<int> &stamp, std::vector<int> &pos, int &stamp_id) {
        cc.valid = false;
        if (n_active <= 1) return;
        // touched nodes = endpoints of active links
        stamp_id++;
        std::vector<int> touched;
        for (int i = 0; i < n_active; i++) {
            int j = active_idx[i];
            for (int e : {tail[j], head[j]})
                if (stamp[e] != stamp_id) {
                    stamp[e] = stamp_id;
                    touched.push_back(e);
                }
        }
        std::sort(touched.begin(), touched.end());
        int m = (int)touched.size();
        if (m - m_rows != 1) return;  // single grounded component only
        // The ground node is the one touched node missing from kept_rows; both
        // lists are sorted, so a merge walk finds it in one pass.
        std::vector<int> kept_sorted(kept_rows.begin(), kept_rows.begin() + m_rows);
        std::sort(kept_sorted.begin(), kept_sorted.end());
        int ground = -1;
        {
            size_t a = 0, b = 0;
            while (a < touched.size()) {
                if (b < kept_sorted.size() && touched[a] == kept_sorted[b]) {
                    a++;
                    b++;
                } else {
                    ground = touched[a];
                    break;
                }
            }
        }
        if (ground < 0) return;
        // keep[i] = touched-local index of kept_rows[i] (A_local row order!)
        stamp_id++;
        for (int i = 0; i < m; i++) {
            stamp[touched[i]] = stamp_id;
            pos[touched[i]] = i;
        }
        std::vector<int> keep(m_rows);
        for (int r = 0; r < m_rows; r++) keep[r] = pos[kept_rows[r]];
        // Grounded Laplacian A_local diag(1/scale_act) A_local'. On the active set
        // the stationarity condition gives x_A = -(c_A + A_A' y) / scale_A, and
        // substituting it into A_A x_A = b leaves this system for the potentials.
        // Grounding one node per component is what makes it invertible.
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Al(
            A_local, m_rows, n_active);
        Eigen::VectorXd w(n_active);
        for (int i = 0; i < n_active; i++) w(i) = 1.0 / scale_m[active_idx[i]];
        Eigen::MatrixXd Lap = Al * w.asDiagonal() * Al.transpose();
        cc.llt.compute(Lap);
        if (cc.llt.info() != Eigen::Success) return;
        // Boundary arcs have exactly one touched endpoint (the stamp still marks
        // the touched set), so they are the arcs whose slack a rigid shift of the
        // touched potentials moves. Caching them lets the repair step evaluate
        // the admissible shift interval without rescanning all L arcs.
        cc.bnd.clear();
        cc.bnd_tail_touched.clear();
        for (int j = 0; j < L; j++) {
            bool tt = (stamp[tail[j]] == stamp_id), th = (stamp[head[j]] == stamp_id);
            if (tt != th) {
                cc.bnd.push_back(j);
                cc.bnd_tail_touched.push_back((char)tt);
            }
        }
        cc.act.assign(active_idx.begin(), active_idx.begin() + n_active);
        cc.touched = std::move(touched);
        cc.keep = std::move(keep);
        cc.ground_node = ground;
        cc.valid = true;
    }

    // Tier-1 failure-reason counters, summed into stats. They are the tuning
    // signal: which one dominates says whether the cache is stale (f_od, f_tt),
    // numerically unusable (f_primal), or under-budgeted (f_budget).
    std::atomic<long> cts_f_od{0}, cts_f_primal{0}, cts_f_tt{0}, cts_f_budget{0}, cts_f_final{0};
    // Total repair worklist pops, successful and wasted alike; calibrates the
    // budget against the work actually spent.
    std::atomic<long> cts_repair_spent_{0};
    // Repair budget = factor * (touched nodes + initial violations) + 512.
    double cts_repair_budget_factor_ = 64.0;

    // Tier-1 attempt. On success fills x_slot (full-L primal), lam_slot (full-V
    // dual) and zbl_slot (full-L bound duals) and returns true; otherwise leaves
    // them unspecified and returns false, and the caller runs the IPM. Every
    // scratch buffer is caller-owned and thread-local, so attempts on different
    // groups are independent.
    bool cts_attempt(const CTSCache &cc, const double *c_var, int o, int d,
                     const double *y_warm_row, double *x_slot, double *lam_slot, double *zbl_slot,
                     std::vector<double> &slack, std::vector<int> &stamp, std::vector<int> &pos,
                     int &stamp_id, std::vector<int> &arc_stamp, int &arc_stamp_id,
                     std::vector<int> &wl, bool &used_repair) {
        const int n_a = (int)cc.act.size(), m = (int)cc.touched.size();
        const int mk = (int)cc.keep.size();
        used_repair = false;
        // Stamp the touched set and record each node's local index. An OD whose
        // endpoints fall outside it cannot be served by this cache at all.
        stamp_id++;
        for (int i = 0; i < m; i++) {
            stamp[cc.touched[i]] = stamp_id;
            pos[cc.touched[i]] = i;
        }
        if (stamp[o] != stamp_id || stamp[d] != stamp_id) {
            cts_f_od++;
            return false;
        }
        // Right-hand side over the touched nodes: -A_act (c_A / scale_A) - b.
        std::vector<double> rhs(m, 0.0);
        for (int i = 0; i < n_a; i++) {
            int j = cc.act[i];
            double cd = c_var[j] / scale_m[j];
            for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++) {
                int r = (int)A_indices[p];
                if (stamp[r] == stamp_id) rhs[pos[r]] -= A_data[p] * cd;
            }
        }
        rhs[pos[o]] += 1.0;  // -= b_o = -(-1)
        rhs[pos[d]] -= 1.0;  // -= b_d = -(+1)
        Eigen::VectorXd rk(mk);
        for (int r = 0; r < mk; r++) rk(r) = rhs[cc.keep[r]];
        Eigen::VectorXd yk = cc.llt.solve(rk);
        // Full dual candidate: the warm row, overwritten on the touched nodes by
        // the freshly solved potentials. The reduced solve pins its grounded node
        // at zero, so the whole block is shifted by that node's cached value; the
        // untouched potentials then stay on the same scale as the touched ones,
        // which is what makes the boundary slacks below meaningful.
        std::memcpy(lam_slot, y_warm_row, (size_t)V * sizeof(double));
        double shift = y_warm_row[cc.ground_node];
        for (int i = 0; i < m; i++) lam_slot[cc.touched[i]] = shift;
        for (int r = 0; r < mk; r++) lam_slot[cc.touched[cc.keep[r]]] = yk(r) + shift;
        // Primal on the active set, from stationarity: x_j = -(c_j + (A'y)_j)/scale_j.
        // A non-positive entry means this link has left the active set, so the
        // cache is stale and the IPM has to resolve the group.
        std::vector<double> x_a(n_a);
        for (int i = 0; i < n_a; i++) {
            int j = cc.act[i];
            double aty = 0.0;
            for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
                aty += A_data[p] * lam_slot[A_indices[p]];
            x_a[i] = -(c_var[j] + aty) / scale_m[j];
            if (x_a[i] < cts_tau_p_) {
                cts_f_primal++;
                return false;
            }
        }
        // Numerical soundness. The tau margins above mean nothing if the reduced
        // solve itself is inaccurate, and route Laplacians can be ill-conditioned
        // with 1/scale weights spanning orders of magnitude. So check the
        // candidate's own primal KKT residual, ||A x_hat - b||_inf over the
        // touched nodes, against a tolerance well inside tau_p. A failure sends
        // the group to the IPM.
        {
            std::vector<double> resid(m, 0.0);
            for (int i = 0; i < n_a; i++) {
                int j = cc.act[i];
                for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
                    resid[pos[(int)A_indices[p]]] += A_data[p] * x_a[i];
            }
            resid[pos[o]] += 1.0;
            resid[pos[d]] -= 1.0;
            for (int i = 0; i < m; i++)
                if (std::abs(resid[i]) > 1e-9) {
                    cts_f_primal++;
                    return false;
                }
        }
        // Mark the active arcs, whose slack is unconstrained by the KKT system.
        arc_stamp_id++;
        for (int i = 0; i < n_a; i++) arc_stamp[cc.act[i]] = arc_stamp_id;
        // Full slack scan, collecting the inactive arcs that violate dual
        // feasibility. Dual slack: c_j + (A'y)_j = c_j + sg*(y[tail_j] - y[head_j]),
        // evaluated from tail/head because the uniform incidence check passed.
        const double sg = cts_sign_;
        wl.clear();
        for (int j = 0; j < L; j++) {
            slack[j] = c_var[j] + sg * (lam_slot[tail[j]] - lam_slot[head[j]]);
            if (arc_stamp[j] != arc_stamp_id && slack[j] < cts_tau_d_) wl.push_back(j);
        }
        if (!wl.empty()) {
            used_repair = true;
            // The violations come from potentials outside the touched block being
            // stale, so repair moves those, leaving the certified interior alone.
            //
            // (1) Rigid shift of the touched potentials into the boundary interval.
            // Shifting all touched potentials by s changes a boundary arc's slack
            // by +sg*s when its touched endpoint is the TAIL, and by -sg*s when it
            // is the head. Interior arcs, having both endpoints touched, are
            // unaffected, so a shift cannot undo what the reduced solve certified.
            double lb = -std::numeric_limits<double>::infinity(), ub = -lb;
            for (size_t i = 0; i < cc.bnd.size(); i++) {
                int j = cc.bnd[i];
                bool pos_coef = (cc.bnd_tail_touched[i] != 0) == (sg > 0.0);
                if (pos_coef)
                    lb = std::max(lb, cts_tau_d_ - slack[j]);
                else
                    ub = std::min(ub, slack[j] - cts_tau_d_);
            }
            double s = 0.0;
            if (lb > 0.0 || ub < 0.0) s = (lb <= ub) ? (lb > 0.0 ? lb : ub) : lb;
            if (s != 0.0) {
                for (int i = 0; i < m; i++) lam_slot[cc.touched[i]] += s;
                for (size_t i = 0; i < cc.bnd.size(); i++) {
                    int j = cc.bnd[i];
                    bool pos_coef = (cc.bnd_tail_touched[i] != 0) == (sg > 0.0);
                    slack[j] += pos_coef ? s : -s;
                    if (arc_stamp[j] != arc_stamp_id && slack[j] < cts_tau_d_) wl.push_back(j);
                }
            }
            // (2) Worklist adjustment, restricted to untouched nodes: move the
            // untouched endpoint's potential so that the arc lands at
            // slack_j = c_j + sg*(y_tail - y_head) = tgt. Moving that node can
            // break its other incident arcs, so they are pushed back on the
            // worklist.
            //
            // Head-first: adjusting the downstream endpoint cascades far less in
            // practice. Measured on a large real-network fit, tail-first exhausted
            // the budget 13x more often, costing 1643 s against roughly 930 s of
            // wall time. This ordering is a performance choice only; the final
            // full re-scan below remains the acceptance gate either way.
            //
            // Adaptive effort cap: a legitimate staleness repair is local, a
            // median of 1-5 links adjacent to the route, so the chance of success
            // saturates at a small multiple of the candidate's own size, while a
            // runaway cascade is the signature of a true active-set change that
            // only the IPM can resolve. The budget is therefore scaled to
            // (touched nodes + initial violations); anything past the success
            // tail is spent work that delays the honest fallback.
            const double tgt = 2.0 * cts_tau_d_;
            size_t qh = 0;
            const long budget0 = (long)(cts_repair_budget_factor_ * (double)(m + wl.size())) + 512;
            long budget = budget0;
            while (qh < wl.size() && budget-- > 0) {
                int j = wl[qh++];
                if (arc_stamp[j] == arc_stamp_id) continue;
                double sj = c_var[j] + sg * (lam_slot[tail[j]] - lam_slot[head[j]]);
                if (sj >= cts_tau_d_) continue;
                int adj_node;
                if (stamp[head[j]] != stamp_id) {  // adjust untouched head
                    lam_slot[head[j]] = lam_slot[tail[j]] - sg * (tgt - c_var[j]);
                    adj_node = head[j];
                } else if (stamp[tail[j]] != stamp_id) {  // adjust untouched tail
                    lam_slot[tail[j]] = lam_slot[head[j]] + sg * (tgt - c_var[j]);
                    adj_node = tail[j];
                } else {
                    // Both endpoints touched: the violation is inside the block
                    // the reduced solve fixed, so no admissible move repairs it.
                    cts_f_tt++;
                    return false;
                }
                for (int64_t p = nadj_indptr_[adj_node]; p < nadj_indptr_[adj_node + 1]; p++)
                    wl.push_back(nadj_arcs_[p]);
            }
            cts_repair_spent_ += budget0 - (budget > 0 ? budget : 0);
            if (budget <= 0) {
                cts_f_budget++;
                return false;
            }
            // Final full re-scan over all L arcs. The repair only ever proposed a
            // candidate; this scan is the acceptance gate, and it re-derives every
            // slack from the final potentials, so the incremental updates above
            // carry no weight in the verdict.
            for (int j = 0; j < L; j++) {
                slack[j] = c_var[j] + sg * (lam_slot[tail[j]] - lam_slot[head[j]]);
                if (arc_stamp[j] != arc_stamp_id && slack[j] < cts_tau_d_) {
                    cts_f_final++;
                    return false;
                }
            }
        }
        // Certified. Emit the primal and the bound duals in the same layout the
        // IPM path writes, so the caller cannot tell which tier produced them.
        // Only the active links of x_slot are written: the certificate leaves
        // every other arc with strictly positive slack, and complementary
        // slackness puts its flow at zero.
        for (int i = 0; i < n_a; i++) x_slot[cc.act[i]] = x_a[i];
        for (int j = 0; j < L; j++) zbl_slot[j] = slack[j] > 0.0 ? slack[j] : 0.0;
        return true;
    }

    // Evaluate one design term at (link, variant), which is the entry the
    // explicit design would have stored at Z[variant][link][k]. Evaluating on
    // demand is what keeps the (N, L, K) design out of memory.
    //
    // Layout: link attributes in link_attrs[attr_variant][link][column],
    // per-variant covariates in variant_cov[variant][covariate], and
    // variant_attr mapping a variant to its attribute-variant row.
    //
    // Every opcode must stay linear in the parameter it multiplies, since the
    // estimator's design is linear in beta by construction.
    double term_value(int code, int a, int b, int cov, int op, double thr, int link, int var,
                      const double *link_attrs, int n_attr_cols, const int64_t *variant_attr,
                      const double *variant_cov, int n_cov) const {
        int av = (int)variant_attr[var];
        auto col = [&](int c) -> double {
            return link_attrs[(((int64_t)av * L) + link) * n_attr_cols + c];
        };
        switch (code) {
            case TERM_COLUMN:
                return col(a);
            case TERM_PRODUCT:
                return col(a) * col(b);
            case TERM_COVARIATE_INDICATOR: {
                double cv = variant_cov[(int64_t)var * n_cov + cov];
                bool keep;
                switch (op) {
                    case COV_LE:
                        keep = cv <= thr;
                        break;
                    case COV_GE:
                        keep = cv >= thr;
                        break;
                    case COV_EQ:
                        keep = cv == thr;
                        break;
                    case COV_LT:
                        keep = cv < thr;
                        break;
                    case COV_GT:
                        keep = cv > thr;
                        break;
                    default:
                        keep = false;
                }
                return keep ? col(a) : 0.0;
            }
            default:
                throw std::invalid_argument("unsupported lazy term code");
        }
    }

    nb::tuple batch_lazy_terms(
        nb::ndarray<double, nb::shape<-1, -1, -1>, nb::c_contig> link_attrs_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> variant_attr_arr,
        nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> variant_cov_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_code_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_a_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_b_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_cov_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_op_arr,
        nb::ndarray<double, nb::shape<-1>, nb::c_contig> term_thr_arr,
        nb::ndarray<double, nb::shape<-1>, nb::c_contig> beta_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> origins_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> dests_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> y_indptr_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> y_indices_arr,
        nb::ndarray<double, nb::shape<-1>, nb::c_contig> y_data_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> trip_variant_arr,
        const std::string &projection_mode, bool return_u, double slack_tol, double qp_eps_abs,
        double qp_eps_duality_gap_abs, double qp_eps_rel, double qp_eps_duality_gap_rel,
        int qp_max_iter) {
        int N = (int)origins_arr.shape(0);
        int n_var = (int)variant_attr_arr.shape(0);
        int K = (int)beta_arr.shape(0);
        int n_attr_cols = (int)link_attrs_arr.shape(2);
        int n_cov = (int)variant_cov_arr.shape(1);
        if (projection_mode != "robust_pinv")
            throw std::invalid_argument(
                "projection_mode must be 'robust_pinv' (legacy_drop was removed)");

        const double *link_attrs = link_attrs_arr.data();
        const int64_t *variant_attr = variant_attr_arr.data();
        const double *variant_cov = variant_cov_arr.data();
        const int64_t *term_code64 = term_code_arr.data();
        const int64_t *term_a64 = term_a_arr.data();
        const int64_t *term_b64 = term_b_arr.data();
        const int64_t *term_cov64 = term_cov_arr.data();
        const int64_t *term_op64 = term_op_arr.data();
        const double *term_thr = term_thr_arr.data();
        const double *beta = beta_arr.data();
        const int64_t *origins64 = origins_arr.data();
        const int64_t *dests64 = dests_arr.data();
        const int64_t *y_indptr = y_indptr_arr.data();
        const int64_t *y_indices = y_indices_arr.data();
        const double *y_data = y_data_arr.data();
        const int64_t *trip_variant64 = trip_variant_arr.data();

        std::vector<int> origins(N), dests(N), trip_var(N);
        for (int n = 0; n < N; n++) {
            origins[n] = (int)origins64[n];
            dests[n] = (int)dests64[n];
            trip_var[n] = (int)trip_variant64[n];
        }
        std::vector<int> term_code(K), term_a(K), term_b(K), term_cov(K), term_op(K);
        for (int k = 0; k < K; k++) {
            term_code[k] = (int)term_code64[k];
            term_a[k] = (int)term_a64[k];
            term_b[k] = (int)term_b64[k];
            term_cov[k] = (int)term_cov64[k];
            term_op[k] = (int)term_op64[k];
        }

        std::vector<int64_t> group_keys(N);
        for (int n = 0; n < N; n++) {
            int64_t od_key = (int64_t)origins[n] * V + dests[n];
            group_keys[n] = od_key * n_var + trip_var[n];
        }
        std::vector<int64_t> sorted_keys = group_keys;
        std::sort(sorted_keys.begin(), sorted_keys.end());
        sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()), sorted_keys.end());
        int n_unique = (int)sorted_keys.size();

        auto key_to_uid = [&](int64_t k) -> int {
            return (int)(std::lower_bound(sorted_keys.begin(), sorted_keys.end(), k) -
                         sorted_keys.begin());
        };
        std::vector<std::vector<int>> trips_for_group(n_unique);
        for (int n = 0; n < N; n++) trips_for_group[key_to_uid(group_keys[n])].push_back(n);

        std::vector<std::vector<int>> groups_for_variant(n_var);
        for (int u = 0; u < n_unique; u++) {
            int var = (int)(sorted_keys[u] % n_var);
            groups_for_variant[var].push_back(u);
        }

        // Look up each group's warm-start point in the persisted context state.
        // warm_keys_ was left sorted by the previous call, so this is a binary
        // search per group. has_warm[u] decides warm against cold; warm_uid[u]
        // indexes the previous call's warm_x_/warm_y_/warm_zbl_, whose row order
        // is the previous call's, and differs from u whenever the set of groups
        // changed.
        int n_prev = (int)warm_keys_.size();
        std::vector<char> has_warm(n_unique, 0);
        std::vector<int> warm_uid(n_unique, -1);
        if (warm_enabled_ && n_prev > 0) {
            for (int u = 0; u < n_unique; u++) {
                auto it = std::lower_bound(warm_keys_.begin(), warm_keys_.end(), sorted_keys[u]);
                if (it != warm_keys_.end() && *it == sorted_keys[u]) {
                    has_warm[u] = 1;
                    warm_uid[u] = (int)(it - warm_keys_.begin());
                }
            }
        }

        // Fresh primal-dual state written this call, persisted afterwards as the
        // warm state for the next KM iteration. new_y doubles as the per-group
        // node-potential buffer that build_active_projection reads to form the
        // dual slack, so the active set is derived from exactly the duals that
        // were just solved.
        const int64_t xsz = (int64_t)n_unique * L, ysz = (int64_t)n_unique * V;
        if ((int64_t)warm_x_next_.size() != xsz) {
            warm_x_next_.assign(xsz, 0.0);
            warm_zbl_next_.assign(xsz, 0.0);
            // Pre-size the read buffers as well. They are empty on the first Phi
            // after a reset and go unread there, because has_warm is false for
            // every group. Sizing them now means the buffer the second call
            // receives from the swap is already the right size, so only the first
            // Phi of a fit pays the zero-fill.
            if ((int64_t)warm_x_.size() != xsz) {
                warm_x_.assign(xsz, 0.0);
                warm_zbl_.assign(xsz, 0.0);
            }
        }
        if ((int64_t)warm_y_next_.size() != ysz) {
            warm_y_next_.assign(ysz, 0.0);
            if ((int64_t)warm_y_.size() != ysz) warm_y_.assign(ysz, 0.0);
        }
        double *new_x = warm_x_next_.data();
        double *new_y = warm_y_next_.data();
        double *new_zbl = warm_zbl_next_.data();

        int n_threads = 1;
        const char *env = std::getenv("OMP_NUM_THREADS");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) n_threads = v;
        } else {
            int hw = (int)std::thread::hardware_concurrency();
            if (hw > 1) n_threads = hw;
        }
        if (n_threads > n_unique) n_threads = n_unique;
        if (n_threads < 1) n_threads = 1;

        // Cost vector per variant, c_var[j] = sum_k beta_k * term_value(...).
        // Built once up front because every group of a variant reads the same
        // vector, and left serial: n_var * L * K term evaluations, against one QP
        // per group in the parallel section below.
        auto t_cost_pre0 = std::chrono::steady_clock::now();
        std::vector<double> c_all((int64_t)n_var * L);
        for (int var = 0; var < n_var; var++) {
            double *c_v = c_all.data() + (int64_t)var * L;
            for (int j = 0; j < L; j++) {
                double s = 0.0;
                for (int k = 0; k < K; k++) {
                    s += beta[k] * term_value(term_code[k], term_a[k], term_b[k], term_cov[k],
                                              term_op[k], term_thr[k], j, var, link_attrs,
                                              n_attr_cols, variant_attr, variant_cov, n_cov);
                }
                c_v[j] = s;
            }
        }

        auto t_cost_pre1 = std::chrono::steady_clock::now();
        double cost_precompute_time =
            std::chrono::duration<double>(t_cost_pre1 - t_cost_pre0).count();

        std::vector<std::vector<double>> H_locals(n_threads, std::vector<double>(K * K, 0.0));
        std::vector<std::vector<double>> g_locals(n_threads, std::vector<double>(K, 0.0));
        std::vector<int> dropped_local(n_threads, 0), valid_local(n_threads, 0),
            chol_fail_local(n_threads, 0), qp_iters(n_unique, 0);
        // Per-reason drop counters, aggregated into stats. A drop is charged to
        // every trip of the group, since the group's failure removes all of them.
        std::vector<int> drop_no_active_local(n_threads, 0), drop_trivial_local(n_threads, 0),
            drop_not_dag_local(n_threads, 0), drop_no_kept_local(n_threads, 0),
            drop_solve_fail_local(n_threads, 0);
        std::vector<double> solve_time_local(n_threads, 0.0), projection_time_local(n_threads, 0.0),
            schur_time_local(n_threads, 0.0);
        std::vector<std::vector<double>> u_per_trip;
        std::vector<char> u_valid;
        if (return_u) {
            u_per_trip.resize(N, std::vector<double>(K, 0.0));
            u_valid.resize(N, 0);
        }

        std::vector<int64_t> aset_hash_per_trip(N, 0);

        // CTS: caches are rebuilt fresh each call and indexed by this call's u,
        // while cts_prev_ stays indexed by warm_keys_; the two are bridged through
        // warm_uid. The assign below re-establishes that alignment after a reset
        // or a change in the group set.
        std::vector<CTSCache> new_cts(cts_enabled_ ? n_unique : 0);
        std::vector<int> cts_attempts_local(n_threads, 0), cts_pass_local(n_threads, 0),
            cts_repaired_local(n_threads, 0);
        if (cts_enabled_ && cts_prev_.size() != warm_keys_.size())
            cts_prev_.assign(warm_keys_.size(), CTSCache{});

        std::atomic<int> next_group{0};
        auto worker = [&](int tid) {
            PIQPSolver solver;
            solver.settings().verbose = false;
            solver.settings().compute_timings = false;
            solver.settings().eps_abs = qp_eps_abs;
            solver.settings().eps_duality_gap_abs = qp_eps_duality_gap_abs;
            solver.settings().eps_rel = qp_eps_rel;
            solver.settings().eps_duality_gap_rel = qp_eps_duality_gap_rel;
            solver.settings().max_iter = qp_max_iter;
            bool first_solve = true;
            int prev_o = -1, prev_d = -1;
            Eigen::VectorXd b_vec = Eigen::VectorXd::Zero(V);

            std::vector<int> active_idx, kept_rows, active_pos_map(L, -1);
            std::vector<double> A_local_buf, Z_act_buf, y_active_buf, w_y_buf, w_r_buf;
            std::vector<double> H_i(K * K), g_i(K), u_i(K);
            // CTS thread-local scratch
            std::vector<double> cts_slack(L);
            std::vector<int> cts_stamp(V, 0), cts_pos(V, 0), cts_arc_stamp(L, 0), cts_wl;
            int cts_stamp_id = 0, cts_arc_stamp_id = 0;

            while (true) {
                int u = next_group.fetch_add(1, std::memory_order_relaxed);
                if (u >= n_unique) break;

                int rep = trips_for_group[u][0];
                int o = origins[rep], d = dests[rep];
                int var = (int)(sorted_keys[u] % n_var);
                const double *c_var = c_all.data() + (int64_t)var * L;
                double *x_slot = new_x + (int64_t)u * L;
                double *lam_slot = new_y + (int64_t)u * V;
                double *zbl_slot = new_zbl + (int64_t)u * L;
                bool uw = has_warm[u];
                const double *xw = uw ? warm_x_.data() + (int64_t)warm_uid[u] * L : nullptr;
                const double *yw = uw ? warm_y_.data() + (int64_t)warm_uid[u] * V : nullptr;
                const double *zw = uw ? warm_zbl_.data() + (int64_t)warm_uid[u] * L : nullptr;

                auto t_solve0 = std::chrono::steady_clock::now();
                // Tier 1: certified reduced solve on the cached active set. It
                // needs a warm entry (the cache and the cached potentials both
                // live under the previous call's index) and a validated incidence
                // convention. On success the cache is carried forward unchanged,
                // since a certified pass means the active set did not move.
                bool tier1_ok = false;
                if (cts_enabled_ && cts_conv_ok_ && uw && cts_prev_[warm_uid[u]].valid) {
                    cts_attempts_local[tid]++;
                    bool used_repair = false;
                    tier1_ok =
                        cts_attempt(cts_prev_[warm_uid[u]], c_var, o, d, yw, x_slot, lam_slot,
                                    zbl_slot, cts_slack, cts_stamp, cts_pos, cts_stamp_id,
                                    cts_arc_stamp, cts_arc_stamp_id, cts_wl, used_repair);
                    if (tier1_ok) {
                        cts_pass_local[tid]++;
                        if (used_repair) cts_repaired_local[tid]++;
                        new_cts[u] = std::move(cts_prev_[warm_uid[u]]);
                        qp_iters[u] = 0;
                    }
                }
                if (!tier1_ok) {
                    ODSolveResult sr = solve_od_group(
                        solver, first_solve, P_eigen, A_eigen, x_lower, b_vec, o, d, prev_o, prev_d,
                        c_var, L, x_slot, lam_slot, zbl_slot, V, uw, xw, yw, zw);
                    qp_iters[u] = sr.qp_iters;
                    if (!sr.ok) {
                        auto t_solve1f = std::chrono::steady_clock::now();
                        solve_time_local[tid] +=
                            std::chrono::duration<double>(t_solve1f - t_solve0).count();
                        int gs = (int)trips_for_group[u].size();
                        dropped_local[tid] += gs;
                        drop_solve_fail_local[tid] += gs;
                        continue;
                    }
                }
                auto t_solve1 = std::chrono::steady_clock::now();
                solve_time_local[tid] += std::chrono::duration<double>(t_solve1 - t_solve0).count();

                auto t_proj0 = std::chrono::steady_clock::now();
                ActiveProjection proj = build_active_projection(
                    lam_slot, c_var, L, V, K, d, A_indptr.data(), A_indices.data(), A_data.data(),
                    tail.data(), head.data(), nullptr, slack_tol, active_idx, kept_rows,
                    A_local_buf, Z_act_buf);
                if (proj.valid) {
                    Z_act_buf.resize((int64_t)proj.n_active * K);
                    for (int i = 0; i < proj.n_active; i++) {
                        int edge = active_idx[i];
                        for (int k = 0; k < K; k++) {
                            Z_act_buf[i * K + k] =
                                term_value(term_code[k], term_a[k], term_b[k], term_cov[k],
                                           term_op[k], term_thr[k], edge, var, link_attrs,
                                           n_attr_cols, variant_attr, variant_cov, n_cov);
                        }
                    }
                }
                auto t_proj1 = std::chrono::steady_clock::now();
                projection_time_local[tid] +=
                    std::chrono::duration<double>(t_proj1 - t_proj0).count();

                if (!proj.valid) {
                    int gs = (int)trips_for_group[u].size();
                    dropped_local[tid] += gs;
                    switch (proj.drop_reason) {
                        case 1:
                            drop_no_active_local[tid] += gs;
                            break;
                        case 2:
                            drop_trivial_local[tid] += gs;
                            break;
                        case 3:
                            drop_not_dag_local[tid] += gs;
                            break;
                        case 4:
                            drop_no_kept_local[tid] += gs;
                            break;
                    }
                    continue;
                }

                // The IPM ran, so the active set may have moved: rebuild this
                // group's CTS cache from the set just detected.
                if (cts_enabled_ && !tier1_ok) {
                    cts_build(new_cts[u], active_idx, proj.n_active, kept_rows, A_local_buf.data(),
                              proj.m_rows, cts_stamp, cts_pos, cts_stamp_id);
                }

                int64_t ahash = hash_active_set(active_idx.data(), proj.n_active);

                for (int n : trips_for_group[u]) {
                    int64_t ys = y_indptr[n], ye = y_indptr[n + 1];
                    auto t_schur0 = std::chrono::steady_clock::now();
                    bool trip_ok = accumulate_trip_sparse(
                        active_idx.data(), proj.n_active, proj.m_rows, K, Z_act_buf.data(),
                        A_local_buf.data(), scale_m.data(), y_indices, y_data, ys, ye, x_slot,
                        return_u, active_pos_map, y_active_buf, w_y_buf, w_r_buf, H_i.data(),
                        g_i.data(), u_i.data());
                    auto t_schur1 = std::chrono::steady_clock::now();
                    schur_time_local[tid] +=
                        std::chrono::duration<double>(t_schur1 - t_schur0).count();

                    if (!trip_ok) {
                        chol_fail_local[tid]++;
                        dropped_local[tid]++;
                        continue;
                    }
                    for (int i = 0; i < K * K; i++) H_locals[tid][i] += H_i[i];
                    for (int k = 0; k < K; k++) g_locals[tid][k] += g_i[k];
                    valid_local[tid]++;
                    aset_hash_per_trip[n] = ahash;
                    if (return_u) {
                        for (int k = 0; k < K; k++) u_per_trip[n][k] = u_i[k];
                        u_valid[n] = 1;
                    }
                }
            }
        };

        if (n_threads <= 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(n_threads);
            for (int t = 0; t < n_threads; t++) threads.emplace_back(worker, t);
            for (auto &t : threads) t.join();
        }

        double *H_out = new double[K * K]();
        double *g_out = new double[K]();
        int total_dropped = 0, total_valid = 0, total_chol_fail = 0, total_qp_iters = 0;
        int td_no_active = 0, td_trivial = 0, td_not_dag = 0, td_no_kept = 0, td_solve_fail = 0;
        double total_solve_time = 0.0, total_projection_time = 0.0, total_schur_time = 0.0;
        for (int t = 0; t < n_threads; t++) {
            for (int i = 0; i < K * K; i++) H_out[i] += H_locals[t][i];
            for (int k = 0; k < K; k++) g_out[k] += g_locals[t][k];
            total_dropped += dropped_local[t];
            total_valid += valid_local[t];
            total_chol_fail += chol_fail_local[t];
            td_no_active += drop_no_active_local[t];
            td_trivial += drop_trivial_local[t];
            td_not_dag += drop_not_dag_local[t];
            td_no_kept += drop_no_kept_local[t];
            td_solve_fail += drop_solve_fail_local[t];
            total_solve_time += solve_time_local[t];
            total_projection_time += projection_time_local[t];
            total_schur_time += schur_time_local[t];
        }
        for (int u = 0; u < n_unique; u++) total_qp_iters += qp_iters[u];

        nb::capsule H_owner(H_out, [](void *p) noexcept { delete[] (double *)p; });
        nb::capsule g_owner(g_out, [](void *p) noexcept { delete[] (double *)p; });
        size_t H_shape[2] = {(size_t)K, (size_t)K};
        size_t g_shape[1] = {(size_t)K};
        auto H_arr_out = nb::ndarray<nb::numpy, double>(H_out, 2, H_shape, H_owner);
        auto g_arr_out = nb::ndarray<nb::numpy, double>(g_out, 1, g_shape, g_owner);

        nb::object u_out = nb::none();
        if (return_u) {
            int n_u = 0;
            for (int n = 0; n < N; n++)
                if (u_valid[n]) n_u++;
            if (n_u > 0) {
                double *u_data = new double[(int64_t)n_u * K];
                int idx = 0;
                for (int n = 0; n < N; n++) {
                    if (u_valid[n]) {
                        std::memcpy(u_data + (int64_t)idx * K, u_per_trip[n].data(),
                                    K * sizeof(double));
                        idx++;
                    }
                }
                nb::capsule u_owner(u_data, [](void *p) noexcept { delete[] (double *)p; });
                size_t u_shape[2] = {(size_t)n_u, (size_t)K};
                u_out = nb::cast(nb::ndarray<nb::numpy, double>(u_data, 2, u_shape, u_owner));
            }
        }

        // Persist this call's primal-dual state as the warm start for the next KM
        // iteration, inside the context, with no Python round-trip.
        warm_keys_ = std::move(sorted_keys);
        // Ping-pong: warm_*_ was read this call, warm_*_next_ was written. After
        // the swap the next call reads what was just written and writes into the
        // buffer that has now aged out, so neither is reallocated.
        std::swap(warm_x_, warm_x_next_);
        std::swap(warm_y_, warm_y_next_);
        std::swap(warm_zbl_, warm_zbl_next_);
        cts_prev_ = std::move(new_cts);  // empty when CTS disabled

        int cts_attempts = 0, cts_pass = 0, cts_repaired = 0;
        for (int t = 0; t < n_threads; t++) {
            cts_attempts += cts_attempts_local[t];
            cts_pass += cts_pass_local[t];
            cts_repaired += cts_repaired_local[t];
        }

        int total_warm = 0;
        for (int u = 0; u < n_unique; u++) total_warm += has_warm[u];

        nb::dict stats;
        stats["warm_groups"] = total_warm;
        stats["cts_attempts"] = cts_attempts;
        stats["cts_convention_ok"] = cts_conv_ok_;
        stats["cts_sign"] = cts_sign_;
        stats["cts_pass"] = cts_pass;
        stats["cts_repaired"] = cts_repaired;
        stats["cts_f_od"] = (long)cts_f_od;
        stats["cts_f_primal"] = (long)cts_f_primal;
        stats["cts_f_tt"] = (long)cts_f_tt;
        stats["cts_f_budget"] = (long)cts_f_budget;
        stats["cts_f_final"] = (long)cts_f_final;
        stats["cts_repair_spent"] = (long)cts_repair_spent_;
        stats["valid"] = total_valid;
        stats["dropped"] = total_dropped;
        stats["cholesky_failures"] = total_chol_fail;
        stats["drop_no_active_links"] = td_no_active;
        stats["drop_trivial_flow"] = td_trivial;
        stats["drop_not_dag"] = td_not_dag;
        stats["drop_no_kept_rows"] = td_no_kept;
        stats["drop_solve_failed"] = td_solve_fail;
        stats["n_threads"] = n_threads;
        stats["n_unique_groups"] = n_unique;
        stats["n_unique_variants"] = n_var;
        stats["total_qp_iters"] = total_qp_iters;
        stats["qp_eps_abs"] = qp_eps_abs;
        stats["qp_eps_duality_gap_abs"] = qp_eps_duality_gap_abs;
        stats["qp_eps_rel"] = qp_eps_rel;
        stats["qp_max_iter"] = qp_max_iter;
        stats["slack_tol"] = slack_tol;
        stats["time_cost_build_s"] = cost_precompute_time;
        stats["time_solve_s"] = total_solve_time;
        stats["time_projection_s"] = total_projection_time;
        stats["time_schur_s"] = total_schur_time;
        stats["lazy_design"] = true;

        int64_t *ahash_out = new int64_t[N];
        std::memcpy(ahash_out, aset_hash_per_trip.data(), N * sizeof(int64_t));
        nb::capsule ahash_owner(ahash_out, [](void *p) noexcept { delete[] (int64_t *)p; });
        size_t ahash_shape[1] = {(size_t)N};
        auto ahash_arr_out =
            nb::ndarray<nb::numpy, int64_t>(ahash_out, 1, ahash_shape, ahash_owner);

        return nb::make_tuple(H_arr_out, g_arr_out, u_out, stats, ahash_arr_out);
    }

    // One-shot parallel forward solve at a fixed beta, returning per-trip predicted
    // flows on the dual-slack active set as an (N x L) CSR (indptr, indices, data).
    // Independent of the estimation kernel above: no warm start, no CTS, no Schur
    // accumulation, and none of the context's persisted state is touched, so
    // prediction can run between fits without disturbing them.
    // Complementary slackness puts x at zero off the active set, so the sparse
    // active flows carry everything the prediction metrics need. x depends only on
    // (variant, o, d), so trips are grouped by that key and the groups are solved
    // across all threads; trips of one group then share its row values.
    nb::tuple batch_predict_flows(
        nb::ndarray<double, nb::shape<-1, -1, -1>, nb::c_contig> link_attrs_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> variant_attr_arr,
        nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> variant_cov_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_code_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_a_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_b_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_cov_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> term_op_arr,
        nb::ndarray<double, nb::shape<-1>, nb::c_contig> term_thr_arr,
        nb::ndarray<double, nb::shape<-1>, nb::c_contig> beta_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> origins_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> dests_arr,
        nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> trip_variant_arr, double slack_tol,
        double qp_eps_abs, double qp_eps_duality_gap_abs, double qp_eps_rel,
        double qp_eps_duality_gap_rel, int qp_max_iter) {
        int N = (int)origins_arr.shape(0);
        int n_var = (int)variant_attr_arr.shape(0);
        int K = (int)beta_arr.shape(0);
        int n_attr_cols = (int)link_attrs_arr.shape(2);
        int n_cov = (int)variant_cov_arr.shape(1);
        const double *link_attrs = link_attrs_arr.data();
        const int64_t *variant_attr = variant_attr_arr.data();
        const double *variant_cov = variant_cov_arr.data();
        const int64_t *term_code = term_code_arr.data();
        const int64_t *term_a = term_a_arr.data();
        const int64_t *term_b = term_b_arr.data();
        const int64_t *term_cov = term_cov_arr.data();
        const int64_t *term_op = term_op_arr.data();
        const double *term_thr = term_thr_arr.data();
        const double *beta = beta_arr.data();
        const int64_t *origins = origins_arr.data();
        const int64_t *dests = dests_arr.data();
        const int64_t *tvar = trip_variant_arr.data();

        // Cost vector per variant: c_var[j] = sum_k beta_k * term_value(...),
        // built for every variant so a group can index it by variant id.
        std::vector<double> c_all((int64_t)n_var * L);
        for (int var = 0; var < n_var; var++) {
            double *c_v = c_all.data() + (int64_t)var * L;
            for (int j = 0; j < L; j++) {
                double s = 0.0;
                for (int k = 0; k < K; k++)
                    s += beta[k] * term_value(term_code[k], term_a[k], term_b[k], term_cov[k],
                                              term_op[k], term_thr[k], j, var, link_attrs,
                                              n_attr_cols, variant_attr, variant_cov, n_cov);
                c_v[j] = s;
            }
        }

        // Unique (variant, origin, dest) groups; the flow depends only on the group.
        std::unordered_map<int64_t, int> key2g;
        key2g.reserve((size_t)N * 2);
        std::vector<int> g_var, g_o, g_d, trip_group(N);
        for (int i = 0; i < N; i++) {
            int var = (int)tvar[i], o = (int)origins[i], d = (int)dests[i];
            int64_t key = ((int64_t)var * V + o) * V + d;
            auto it = key2g.find(key);
            int g;
            if (it == key2g.end()) {
                g = (int)g_var.size();
                key2g.emplace(key, g);
                g_var.push_back(var);
                g_o.push_back(o);
                g_d.push_back(d);
            } else {
                g = it->second;
            }
            trip_group[i] = g;
        }
        int n_groups = (int)g_var.size();

        std::vector<std::vector<int>> g_idx(n_groups);
        std::vector<std::vector<double>> g_x(n_groups);
        double cutoff = slack_tol > 0.0 ? -slack_tol : 0.0;

        int n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 1;
        if (n_threads > n_groups) n_threads = n_groups;
        if (n_threads < 1) n_threads = 1;

        std::atomic<int> next_g{0};
        auto worker = [&]() {
            PIQPSolver solver;
            solver.settings().verbose = false;
            solver.settings().compute_timings = false;
            solver.settings().eps_abs = qp_eps_abs;
            solver.settings().eps_duality_gap_abs = qp_eps_duality_gap_abs;
            solver.settings().eps_rel = qp_eps_rel;
            solver.settings().eps_duality_gap_rel = qp_eps_duality_gap_rel;
            solver.settings().max_iter = qp_max_iter;
            bool first_solve = true;
            int prev_o = -1, prev_d = -1;
            Eigen::VectorXd b_vec = Eigen::VectorXd::Zero(V);
            std::vector<double> x_out(L), lam_out(V), zbl_out(L);
            while (true) {
                int g = next_g.fetch_add(1, std::memory_order_relaxed);
                if (g >= n_groups) break;
                int o = g_o[g], d = g_d[g];
                const double *c_var = c_all.data() + (int64_t)g_var[g] * L;
                ODSolveResult sr =
                    solve_od_group(solver, first_solve, P_eigen, A_eigen, x_lower, b_vec, o, d,
                                   prev_o, prev_d, c_var, L, x_out.data(), lam_out.data(),
                                   zbl_out.data(), V, false, nullptr, nullptr, nullptr);
                // A failed solve leaves g_idx[g] empty, so the group's trips come
                // back as empty CSR rows and the caller sees no prediction there.
                if (!sr.ok) continue;
                for (int j = 0; j < L; j++) {
                    double at = 0.0;
                    for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
                        at += A_data[p] * lam_out[A_indices[p]];
                    if (c_var[j] + at < cutoff) {  // dual-slack active set
                        g_idx[g].push_back(j);
                        g_x[g].push_back(x_out[j]);
                    }
                }
            }
        };
        if (n_threads <= 1) {
            worker();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(n_threads);
            for (int t = 0; t < n_threads; t++) threads.emplace_back(worker);
            for (auto &t : threads) t.join();
        }

        // Assemble the (N x L) CSR of predicted active flows. Trips of one group
        // hold identical rows, which are materialized per trip so the result is a
        // plain CSR the caller can hand to scipy without a mapping table.
        std::vector<int> indptr(N + 1, 0);
        for (int i = 0; i < N; i++) indptr[i + 1] = indptr[i] + (int)g_idx[trip_group[i]].size();
        int nnz = indptr[N];
        std::vector<int> indices(nnz);
        std::vector<double> data(nnz);
        for (int i = 0; i < N; i++) {
            const auto &gi = g_idx[trip_group[i]];
            const auto &gx = g_x[trip_group[i]];
            int off = indptr[i];
            for (size_t t = 0; t < gi.size(); t++) {
                indices[off + (int)t] = gi[t];
                data[off + (int)t] = gx[t];
            }
        }
        return nb::make_tuple(make_int64_array(indptr), make_int64_array(indices),
                              make_double_array(data));
    }
};
#endif

// Fused batch solve + accumulate over an explicit design Z.
//
// The interface is shaped to keep the per-trip footprint small: OD endpoints
// arrive as origins/dests (N) and the right-hand sides b (N x V) are rebuilt one
// column pair at a time; observed routes arrive as a CSR y, whose nnz is the
// number of links actually traversed. Solve and accumulation are fused per OD
// group, so no per-trip flow array is retained.
//
// Memory: O(n_unique * V) for the lam warm-start cache plus O(n_threads * L) of
// worker scratch.
static nb::tuple batch_solve_and_accumulate(
    // Network
    nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> Z_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indptr_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indices_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> A_data_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> scale_m_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> tail_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> head_arr, int V, int L, int K,
    // Cost vector
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> c_arr,
    // Trip data (memory-efficient)
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> origins_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> dests_arr,
    // Sparse y in CSR: y_indptr (N+1), y_indices (nnz), y_data (nnz)
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> y_indptr_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> y_indices_arr,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> y_data_arr,
    // Warm-start duals from previous iteration (n_unique_prev x V), or empty
    nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> lam_warm_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> od_keys_prev_arr,
    // Per-trip Z variant (empty array if single Z)
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> trip_variant_arr,
    int n_z_variants,  // 0 = single Z (backward compat), >0 = num variants
    // Config
    const std::string &projection_mode, bool return_u, double slack_tol, double qp_eps_abs,
    double qp_eps_duality_gap_abs, double qp_eps_rel, double qp_eps_duality_gap_rel,
    int qp_max_iter) {
    int N = (int)origins_arr.shape(0);
    if (projection_mode != "robust_pinv")
        throw std::invalid_argument(
            "projection_mode must be 'robust_pinv' (legacy_drop was removed)");

    const double *Z = Z_arr.data();
    const int64_t *A_indptr = A_indptr_arr.data();
    const int64_t *A_indices = A_indices_arr.data();
    const double *A_data = A_data_arr.data();
    const double *scale_m = scale_m_arr.data();
    const int64_t *tail64 = tail_arr.data();
    const int64_t *head64 = head_arr.data();
    const double *c = c_arr.data();
    const int64_t *origins64 = origins_arr.data();
    const int64_t *dests64 = dests_arr.data();
    const int64_t *y_indptr = y_indptr_arr.data();
    const int64_t *y_indices = y_indices_arr.data();
    const double *y_data = y_data_arr.data();

    int n_warm = (int)lam_warm_arr.shape(0);
    const double *lam_warm = lam_warm_arr.data();
    const int64_t *od_keys_prev = od_keys_prev_arr.data();

    std::vector<int> tail_i(L), head_i(L);
    for (int i = 0; i < L; i++) {
        tail_i[i] = (int)tail64[i];
        head_i[i] = (int)head64[i];
    }
    const int *tail = tail_i.data();
    const int *head = head_i.data();

    std::vector<int> origins(N), dests(N);
    for (int n = 0; n < N; n++) {
        origins[n] = (int)origins64[n];
        dests[n] = (int)dests64[n];
    }

    // n_var = 1 covers the single-Z case, where every trip shares one design and
    // trip_variant is unused.
    int n_var = (n_z_variants > 0) ? n_z_variants : 1;
    std::vector<int> trip_var(N, 0);
    if (n_z_variants > 0) {
        const int64_t *tv64 = trip_variant_arr.data();
        for (int n = 0; n < N; n++) trip_var[n] = (int)tv64[n];
    }

    // Group trips by (OD, variant): the forward flow depends on nothing else, so
    // one solve serves the whole group. group_key = od_key * n_var + variant is
    // injective because variant < n_var, and collapses to od_key when n_var == 1.
    std::vector<int64_t> od_keys(N);
    std::vector<int64_t> group_keys(N);
    for (int n = 0; n < N; n++) {
        od_keys[n] = (int64_t)origins[n] * V + dests[n];
        group_keys[n] = od_keys[n] * n_var + trip_var[n];
    }

    std::vector<int64_t> sorted_keys = group_keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()), sorted_keys.end());
    int n_unique = (int)sorted_keys.size();

    auto key_to_uid = [&](int64_t k) -> int {
        return (int)(std::lower_bound(sorted_keys.begin(), sorted_keys.end(), k) -
                     sorted_keys.begin());
    };

    std::vector<int> trip_to_uid(N);
    for (int n = 0; n < N; n++) trip_to_uid[n] = key_to_uid(group_keys[n]);

    // trips_for_od[u]: the trips of unique group u. Each is visited by exactly
    // one worker, which is what makes the per-trip outputs lock-free.
    std::vector<std::vector<int>> trips_for_od(n_unique);
    for (int n = 0; n < N; n++) trips_for_od[trip_to_uid[n]].push_back(n);

    // Unique OD keys are kept for stats only. Warm starts are keyed by group.
    std::vector<int64_t> unique_od_keys;
    {
        std::vector<int64_t> tmp = od_keys;
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        unique_od_keys = std::move(tmp);
    }
    int n_unique_ods = (int)unique_od_keys.size();

    // Warm-start lookup by group key. The caller's od_keys_prev arrives in its
    // own order, so sort a permutation alongside it and keep warm_key_idx to map
    // a hit back to the row of lam_warm it came from.
    std::vector<int64_t> warm_keys_sorted;
    std::vector<int> warm_key_idx;
    if (n_warm > 0) {
        warm_keys_sorted.resize(n_warm);
        warm_key_idx.resize(n_warm);
        std::iota(warm_key_idx.begin(), warm_key_idx.end(), 0);
        for (int i = 0; i < n_warm; i++) warm_keys_sorted[i] = od_keys_prev[i];
        std::sort(warm_key_idx.begin(), warm_key_idx.end(),
                  [&](int a, int bb) { return warm_keys_sorted[a] < warm_keys_sorted[bb]; });
        std::sort(warm_keys_sorted.begin(), warm_keys_sorted.end());
    }

    // lam cache, one row per unique group. Seeded from the caller's warm duals
    // where a key matches, then overwritten in place by this call's solve, and
    // returned whole so the caller can pass it back next iteration. A group whose
    // solve fails keeps its seeded row, which is the last known dual for that key.
    std::vector<double> lam_unique((int64_t)n_unique * V, 0.0);
    std::vector<int> qp_iters(n_unique, 0);

    for (int u = 0; u < n_unique; u++) {
        if (n_warm > 0) {
            auto it =
                std::lower_bound(warm_keys_sorted.begin(), warm_keys_sorted.end(), sorted_keys[u]);
            if (it != warm_keys_sorted.end() && *it == sorted_keys[u]) {
                int pos = (int)(it - warm_keys_sorted.begin());
                int orig_idx = warm_key_idx[pos];
                std::memcpy(lam_unique.data() + (int64_t)u * V, lam_warm + (int64_t)orig_idx * V,
                            V * sizeof(double));
            }
        }
    }

    int n_threads = 1;
    {
        const char *env = std::getenv("OMP_NUM_THREADS");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) n_threads = v;
        } else {
            int hw = (int)std::thread::hardware_concurrency();
            if (hw > 1) n_threads = hw;
        }
    }
    if (n_threads > n_unique) n_threads = n_unique;
    if (n_threads < 1) n_threads = 1;

    // Build Eigen matrices for PIQP (shared, read-only).
#ifdef HAVE_PIQP
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> P_eigen(L, L);
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(L);
        for (int j = 0; j < L; j++) trips.push_back({j, j, scale_m[j]});
        P_eigen.setFromTriplets(trips.begin(), trips.end());
    }

    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_eigen(V, L);
    {
        int nnz = (int)A_indptr[L];
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(nnz);
        for (int j = 0; j < L; j++)
            for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
                trips.push_back({(int)A_indices[p], j, A_data[p]});
        A_eigen.setFromTriplets(trips.begin(), trips.end());
    }

    Eigen::VectorXd x_lower = Eigen::VectorXd::Zero(L);
#endif

    // Thread-local accumulators, reduced after the join. Summing H and g per
    // thread avoids contention, and it also fixes the summation order for a given
    // thread count, which a shared atomic accumulator would not.
    std::vector<std::vector<double>> H_locals(n_threads, std::vector<double>(K * K, 0.0));
    std::vector<std::vector<double>> g_locals(n_threads, std::vector<double>(K, 0.0));
    std::vector<int> dropped_local(n_threads, 0);
    std::vector<int> valid_local(n_threads, 0);
    std::vector<int> chol_fail_local(n_threads, 0);
    std::vector<double> solve_time_local(n_threads, 0.0);
    std::vector<double> projection_time_local(n_threads, 0.0);
    std::vector<double> schur_time_local(n_threads, 0.0);

    // Records which groups had a warm dual available. Nothing reads the result:
    // this path solves every group cold (see fused_worker below), so the flag is
    // computed and discarded.
    std::vector<bool> has_warm(n_unique, false);
    for (int u = 0; u < n_unique; u++) {
        if (n_warm > 0) {
            auto it =
                std::lower_bound(warm_keys_sorted.begin(), warm_keys_sorted.end(), sorted_keys[u]);
            if (it != warm_keys_sorted.end() && *it == sorted_keys[u]) has_warm[u] = true;
        }
    }

    std::vector<std::vector<double>> u_per_trip;
    std::vector<char> u_valid;
    if (return_u) {
        u_per_trip.resize(N, std::vector<double>(K, 0.0));
        u_valid.resize(N, 0);
    }

    std::vector<int64_t> aset_hash_per_trip(N, 0);

    std::atomic<int> next_od{0};

    // Fused solve + accumulate worker.
#ifdef HAVE_PIQP
    auto fused_worker = [&](int tid) {
        PIQPSolver solver;
        solver.settings().verbose = false;
        solver.settings().compute_timings = false;
        solver.settings().eps_abs = qp_eps_abs;
        solver.settings().eps_duality_gap_abs = qp_eps_duality_gap_abs;
        solver.settings().eps_rel = qp_eps_rel;
        solver.settings().eps_duality_gap_rel = qp_eps_duality_gap_rel;
        solver.settings().max_iter = qp_max_iter;
        bool first_solve = true;
        int prev_o = -1, prev_d = -1;
        Eigen::VectorXd b_vec = Eigen::VectorXd::Zero(V);

        double *H_loc = H_locals[tid].data();
        double *g_loc = g_locals[tid].data();
        std::vector<int> active_idx, kept_rows;
        std::vector<double> A_local_buf, Z_act_buf, w_y_buf, w_r_buf;
        std::vector<int> active_pos_map(L, -1);
        std::vector<double> x_u(L, 0.0), y_active_buf, zbl_scratch(L, 0.0);
        std::vector<double> H_i(K * K), g_i(K), u_i(K);

        while (true) {
            int u = next_od.fetch_add(1, std::memory_order_relaxed);
            if (u >= n_unique) break;

            int rep = trips_for_od[u][0];
            int o = origins[rep], d = dests[rep];
            int64_t gk = sorted_keys[u];
            int var = (int)(gk % n_var);
            double *lam_slot = lam_unique.data() + (int64_t)u * V;
            const double *c_var = c + (int64_t)var * L;
            const double *Z_v = Z + (int64_t)var * L * K;

            auto t_solve0 = std::chrono::steady_clock::now();
            // Explicit-Z oracle path: cold solves only (no warm-start persistence).
            ODSolveResult sr = solve_od_group(
                solver, first_solve, P_eigen, A_eigen, x_lower, b_vec, o, d, prev_o, prev_d, c_var,
                L, x_u.data(), lam_slot, zbl_scratch.data(), V, false, nullptr, nullptr, nullptr);
            auto t_solve1 = std::chrono::steady_clock::now();
            solve_time_local[tid] += std::chrono::duration<double>(t_solve1 - t_solve0).count();

            qp_iters[u] = sr.qp_iters;
            if (!sr.ok) {
                dropped_local[tid] += (int)trips_for_od[u].size();
                continue;
            }

            // Detect the active set and build the projection workspace.
            auto t_proj0 = std::chrono::steady_clock::now();
            ActiveProjection proj = build_active_projection(
                lam_slot, c_var, L, V, K, d, A_indptr, A_indices, A_data, tail, head, Z_v,
                slack_tol, active_idx, kept_rows, A_local_buf, Z_act_buf);
            auto t_proj1 = std::chrono::steady_clock::now();
            projection_time_local[tid] += std::chrono::duration<double>(t_proj1 - t_proj0).count();

            if (!proj.valid) {
                dropped_local[tid] += (int)trips_for_od[u].size();
                continue;
            }

            int64_t ahash = hash_active_set(active_idx.data(), proj.n_active);

            // Accumulate every trip of the group against this one projection.
            for (int n : trips_for_od[u]) {
                int64_t ys = y_indptr[n], ye = y_indptr[n + 1];

                auto t_schur0 = std::chrono::steady_clock::now();
                bool trip_ok = accumulate_trip_sparse(
                    active_idx.data(), proj.n_active, proj.m_rows, K, Z_act_buf.data(),
                    A_local_buf.data(), scale_m, y_indices, y_data, ys, ye, x_u.data(), return_u,
                    active_pos_map, y_active_buf, w_y_buf, w_r_buf, H_i.data(), g_i.data(),
                    u_i.data());
                auto t_schur1 = std::chrono::steady_clock::now();
                schur_time_local[tid] += std::chrono::duration<double>(t_schur1 - t_schur0).count();

                if (!trip_ok) {
                    chol_fail_local[tid]++;
                    dropped_local[tid]++;
                    continue;
                }

                for (int i = 0; i < K * K; i++) H_loc[i] += H_i[i];
                for (int k = 0; k < K; k++) g_loc[k] += g_i[k];
                valid_local[tid]++;
                aset_hash_per_trip[n] = ahash;

                if (return_u) {
                    for (int k = 0; k < K; k++) u_per_trip[n][k] = u_i[k];
                    u_valid[n] = 1;
                }
            }
        }
    };
#else
    auto fused_worker = [&](int) {
        throw std::runtime_error("Native QP solver requires PIQP -- rebuild with HAVE_PIQP");
    };
#endif

    if (n_threads <= 1) {
        fused_worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int t = 0; t < n_threads; t++) threads.emplace_back(fused_worker, t);
        for (auto &t : threads) t.join();
    }

    // Build outputs.
    double *H_out = new double[K * K]();
    double *g_out = new double[K]();

    for (int t = 0; t < n_threads; t++) {
        for (int i = 0; i < K * K; i++) H_out[i] += H_locals[t][i];
        for (int k = 0; k < K; k++) g_out[k] += g_locals[t][k];
    }

    int total_dropped = 0, total_valid = 0, total_chol_fail = 0, total_qp_iters = 0;
    double total_solve_time = 0.0, total_projection_time = 0.0, total_schur_time = 0.0;
    for (int t = 0; t < n_threads; t++) {
        total_dropped += dropped_local[t];
        total_valid += valid_local[t];
        total_chol_fail += chol_fail_local[t];
        total_solve_time += solve_time_local[t];
        total_projection_time += projection_time_local[t];
        total_schur_time += schur_time_local[t];
    }
    for (int u = 0; u < n_unique; u++) total_qp_iters += qp_iters[u];

    nb::capsule H_owner(H_out, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule g_owner(g_out, [](void *p) noexcept { delete[] (double *)p; });

    size_t H_shape[2] = {(size_t)K, (size_t)K};
    size_t g_shape[1] = {(size_t)K};

    auto H_arr_out = nb::ndarray<nb::numpy, double>(H_out, 2, H_shape, H_owner);
    auto g_arr_out = nb::ndarray<nb::numpy, double>(g_out, 1, g_shape, g_owner);

    nb::object u_out = nb::none();
    if (return_u) {
        int n_u = 0;
        for (int n = 0; n < N; n++)
            if (u_valid[n]) n_u++;
        if (n_u > 0) {
            double *u_data = new double[n_u * K];
            int idx = 0;
            for (int n = 0; n < N; n++) {
                if (u_valid[n]) {
                    std::memcpy(u_data + idx * K, u_per_trip[n].data(), K * sizeof(double));
                    idx++;
                }
            }
            nb::capsule u_owner(u_data, [](void *p) noexcept { delete[] (double *)p; });
            size_t u_shape[2] = {(size_t)n_u, (size_t)K};
            u_out = nb::cast(nb::ndarray<nb::numpy, double>(u_data, 2, u_shape, u_owner));
        }
    }

    double *lam_out = new double[(int64_t)n_unique * V];
    std::memcpy(lam_out, lam_unique.data(), (int64_t)n_unique * V * sizeof(double));
    nb::capsule lam_owner(lam_out, [](void *p) noexcept { delete[] (double *)p; });
    size_t lam_shape[2] = {(size_t)n_unique, (size_t)V};
    auto lam_arr_out = nb::ndarray<nb::numpy, double>(lam_out, 2, lam_shape, lam_owner);

    int64_t *keys_out = new int64_t[n_unique];
    std::memcpy(keys_out, sorted_keys.data(), n_unique * sizeof(int64_t));
    nb::capsule keys_owner(keys_out, [](void *p) noexcept { delete[] (int64_t *)p; });
    size_t keys_shape[1] = {(size_t)n_unique};
    auto keys_arr_out = nb::ndarray<nb::numpy, int64_t>(keys_out, 1, keys_shape, keys_owner);

    nb::dict stats;
    stats["valid"] = total_valid;
    stats["dropped"] = total_dropped;
    stats["cholesky_failures"] = total_chol_fail;
    stats["n_threads"] = n_threads;
    stats["n_unique_ods"] = n_unique_ods;
    stats["n_unique_groups"] = n_unique;
    stats["total_qp_iters"] = total_qp_iters;
    stats["qp_eps_abs"] = qp_eps_abs;
    stats["qp_eps_duality_gap_abs"] = qp_eps_duality_gap_abs;
    stats["qp_eps_rel"] = qp_eps_rel;
    stats["qp_max_iter"] = qp_max_iter;
    stats["slack_tol"] = slack_tol;
    stats["time_solve_s"] = total_solve_time;
    stats["time_projection_s"] = total_projection_time;
    stats["time_schur_s"] = total_schur_time;
    stats["time_cost_build_s"] = 0.0;

    int64_t *ahash_out = new int64_t[N];
    std::memcpy(ahash_out, aset_hash_per_trip.data(), N * sizeof(int64_t));
    nb::capsule ahash_owner(ahash_out, [](void *p) noexcept { delete[] (int64_t *)p; });
    size_t ahash_shape[1] = {(size_t)N};
    auto ahash_arr_out = nb::ndarray<nb::numpy, int64_t>(ahash_out, 1, ahash_shape, ahash_owner);

    return nb::make_tuple(H_arr_out, g_arr_out, u_out, lam_arr_out, keys_arr_out, stats,
                          ahash_arr_out);
}

#ifdef HAVE_PIQP
// Termination settings, mapped one-to-one from the Python SolverConfig. Kept as
// a value type on the solver so a caller can read back exactly what the solver
// was configured with.
struct NativeSolverSettings {
    bool verbose;
    double eps_abs;
    double eps_duality_gap_abs;
    double eps_rel;
    double eps_duality_gap_rel;
    int max_iter;
};

// Persistent single-right-hand-side forward QP solver:
//
//   min_x  c'x + 0.5 x' diag(scale_m) x   s.t.   A x = b,   x >= 0,
//
// with no upper bound on x. The sparsity pattern of A and of diag(scale_m) is
// fixed at construction and PIQP is set up exactly once, so each solve(c, b)
// updates only those two vectors and reuses the symbolic factorization; the
// solver object carries its own state across calls.
//
// The upper bound is passed as +infinity rather than omitted, which is the same
// problem: PIQP stores only the entries of x_u strictly below its infinity
// marker, so an all-infinite x_u leaves zero upper-bound rows in the data.
//
// solve() releases the GIL, so one instance serves one thread at a time: the
// (c, b) buffers and PIQP's own workspace are per-instance mutable state.
// Concurrent callers need one instance each, which is what the parallel batch
// entry points do internally.
class NativeSparseSolver {
   public:
    NativeSparseSolver(nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indptr_arr,
                       nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indices_arr,
                       nb::ndarray<double, nb::shape<-1>, nb::c_contig> A_data_arr,
                       nb::ndarray<double, nb::shape<-1>, nb::c_contig> scale_m_arr, int V_in,
                       int L_in, bool verbose, double eps_abs, double eps_duality_gap_abs,
                       double eps_rel, double eps_duality_gap_rel, int max_iter)
        : V(V_in), L(L_in), P_eigen(L_in, L_in), A_eigen(V_in, L_in) {
        if (V <= 0 || L <= 0)
            throw std::invalid_argument("NativeSparseSolver needs V > 0 and L > 0");
        if ((int64_t)A_indptr_arr.shape(0) != (int64_t)L + 1)
            throw std::invalid_argument("A_indptr must have length L + 1");
        if ((int64_t)scale_m_arr.shape(0) != (int64_t)L)
            throw std::invalid_argument("scale_m must have length L");

        const int64_t *indptr = A_indptr_arr.data();
        const int64_t *indices = A_indices_arr.data();
        const double *data = A_data_arr.data();
        const double *scale_m = scale_m_arr.data();

        int64_t nnz = indptr[L];
        if (indptr[0] != 0)
            throw std::invalid_argument("A_indptr must start at 0 (CSC of an (V, L) matrix)");
        if ((int64_t)A_indices_arr.shape(0) != nnz || (int64_t)A_data_arr.shape(0) != nnz)
            throw std::invalid_argument("A_indices and A_data must have length A_indptr[L]");
        for (int64_t p = 0; p < nnz; p++)
            if (indices[p] < 0 || indices[p] >= V)
                throw std::invalid_argument("A_indices entry outside [0, V)");

        std::vector<Eigen::Triplet<double>> p_trips;
        p_trips.reserve(L);
        for (int j = 0; j < L; j++) p_trips.push_back({j, j, scale_m[j]});
        P_eigen.setFromTriplets(p_trips.begin(), p_trips.end());

        std::vector<Eigen::Triplet<double>> a_trips;
        a_trips.reserve((size_t)nnz);
        for (int j = 0; j < L; j++)
            for (int64_t p = indptr[j]; p < indptr[j + 1]; p++)
                a_trips.push_back({(int)indices[p], j, data[p]});
        A_eigen.setFromTriplets(a_trips.begin(), a_trips.end());

        settings_.verbose = verbose;
        settings_.eps_abs = eps_abs;
        settings_.eps_duality_gap_abs = eps_duality_gap_abs;
        settings_.eps_rel = eps_rel;
        settings_.eps_duality_gap_rel = eps_duality_gap_rel;
        settings_.max_iter = max_iter;

        solver_.settings().verbose = verbose;
        solver_.settings().eps_abs = eps_abs;
        solver_.settings().eps_duality_gap_abs = eps_duality_gap_abs;
        solver_.settings().eps_rel = eps_rel;
        solver_.settings().eps_duality_gap_rel = eps_duality_gap_rel;
        solver_.settings().max_iter = max_iter;
        // Timings are not part of the reported diagnostics, and false is also
        // PIQP's own default.
        solver_.settings().compute_timings = false;

        c_vec = Eigen::VectorXd::Zero(L);
        b_vec = Eigen::VectorXd::Zero(V);
        x_lower = Eigen::VectorXd::Zero(L);
        x_upper = Eigen::VectorXd::Constant(L, std::numeric_limits<double>::infinity());

        solver_.setup(P_eigen, c_vec, A_eigen, b_vec, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                      x_lower, x_upper);
    }

    // Update (c, b) on the set-up problem and solve it.
    // Returns (x (L,), y (V,), info), where y holds the equality-constraint
    // duals as PIQP reports them.
    nb::tuple solve(nb::ndarray<double, nb::shape<-1>, nb::c_contig> c_arr,
                    nb::ndarray<double, nb::shape<-1>, nb::c_contig> b_arr) {
        if ((int64_t)c_arr.shape(0) != (int64_t)L)
            throw std::invalid_argument("c must have length L");
        if ((int64_t)b_arr.shape(0) != (int64_t)V)
            throw std::invalid_argument("b must have length V");

        std::memcpy(c_vec.data(), c_arr.data(), sizeof(double) * (size_t)L);
        std::memcpy(b_vec.data(), b_arr.data(), sizeof(double) * (size_t)V);

        piqp::Status status;
        {
            // The solve touches no Python object, so other threads can run.
            nb::gil_scoped_release release;
            solver_.update(piqp::nullopt, c_vec, piqp::nullopt, b_vec);
            status = solver_.solve();
        }

        const auto &res = solver_.result();
        double *x = new double[L];
        double *y = new double[V];
        std::memcpy(x, res.x.data(), sizeof(double) * (size_t)L);
        std::memcpy(y, res.y.data(), sizeof(double) * (size_t)V);
        nb::capsule x_owner(x, [](void *p) noexcept { delete[] (double *)p; });
        nb::capsule y_owner(y, [](void *p) noexcept { delete[] (double *)p; });
        size_t x_shape[1] = {(size_t)L};
        size_t y_shape[1] = {(size_t)V};
        auto x_out = nb::ndarray<nb::numpy, double>(x, 1, x_shape, x_owner);
        auto y_out = nb::ndarray<nb::numpy, double>(y, 1, y_shape, y_owner);

        nb::dict info;
        info["status"] = status;
        info["iter"] = (int64_t)res.info.iter;
        info["primal_residual"] = res.info.primal_res;
        info["dual_residual"] = res.info.dual_res;
        info["duality_gap"] = res.info.duality_gap;

        return nb::make_tuple(x_out, y_out, info);
    }

    NativeSolverSettings settings_;

   private:
    int V, L;
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> P_eigen, A_eigen;
    Eigen::VectorXd c_vec, b_vec, x_lower, x_upper;
    PIQPSolver solver_;
};
#endif

// Parallel forward-QP batch solve (flows only).
// Solves U independent unit-demand OD flow problems in parallel and returns the
// (U, L) optimal-flow matrix. This is the forward solver the DGP uses to
// generate data: it takes a cost vector and OD endpoints, and returns flows.
// Each OD is a cold solve on a per-thread PIQP instance, reused across that
// thread's ODs via update(), so the result matches PIQPFlowSolver.solve on the
// same (c, A, scale_m). The equality duals stay inside the worker, so callers
// needing the dual-slack active set must solve through a path that returns them.
static nb::tuple batch_solve_flows(nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indptr_arr,
                                   nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> A_indices_arr,
                                   nb::ndarray<double, nb::shape<-1>, nb::c_contig> A_data_arr,
                                   nb::ndarray<double, nb::shape<-1>, nb::c_contig> scale_m_arr,
                                   int V, int L,
                                   nb::ndarray<double, nb::shape<-1>, nb::c_contig> c_arr,
                                   nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> origins_arr,
                                   nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> dests_arr,
                                   double qp_eps_abs, double qp_eps_duality_gap_abs,
                                   int qp_max_iter, int n_threads) {
#ifdef HAVE_PIQP
    int U = (int)origins_arr.shape(0);
    if ((int)dests_arr.shape(0) != U)
        throw std::invalid_argument("origins and dests must have the same length");

    const int64_t *A_indptr = A_indptr_arr.data();
    const int64_t *A_indices = A_indices_arr.data();
    const double *A_data = A_data_arr.data();
    const double *scale_m = scale_m_arr.data();
    const double *c = c_arr.data();
    const int64_t *origins64 = origins_arr.data();
    const int64_t *dests64 = dests_arr.data();

    std::vector<int> origins(U), dests(U);
    for (int u = 0; u < U; u++) {
        origins[u] = (int)origins64[u];
        dests[u] = (int)dests64[u];
    }

    if (n_threads <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        n_threads = hw > 1 ? hw : 1;
    }
    if (n_threads > U) n_threads = U;
    if (n_threads < 1) n_threads = 1;

    // Shared read-only Eigen matrices, built exactly as the estimator builds
    // them, so a DGP flow and an estimator flow come from the same problem.
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> P_eigen(L, L);
    {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(L);
        for (int j = 0; j < L; j++) trips.push_back({j, j, scale_m[j]});
        P_eigen.setFromTriplets(trips.begin(), trips.end());
    }
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_eigen(V, L);
    {
        int nnz = (int)A_indptr[L];
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(nnz);
        for (int j = 0; j < L; j++)
            for (int64_t p = A_indptr[j]; p < A_indptr[j + 1]; p++)
                trips.push_back({(int)A_indices[p], j, A_data[p]});
        A_eigen.setFromTriplets(trips.begin(), trips.end());
    }
    Eigen::VectorXd x_lower = Eigen::VectorXd::Zero(L);

    double *X = new double[(int64_t)U * L]();
    uint8_t *ok = new uint8_t[U]();

    std::atomic<int> next_u{0};
    auto worker = [&]() {
        PIQPSolver solver;
        solver.settings().verbose = false;
        solver.settings().compute_timings = false;
        solver.settings().eps_abs = qp_eps_abs;
        solver.settings().eps_duality_gap_abs = qp_eps_duality_gap_abs;
        solver.settings().max_iter = qp_max_iter;
        bool first_solve = true;
        int prev_o = -1, prev_d = -1;
        Eigen::VectorXd b_vec = Eigen::VectorXd::Zero(V);
        std::vector<double> lam_scratch(V), zbl_scratch(L);

        while (true) {
            int u = next_u.fetch_add(1, std::memory_order_relaxed);
            if (u >= U) break;
            ODSolveResult sr = solve_od_group(
                solver, first_solve, P_eigen, A_eigen, x_lower, b_vec, origins[u], dests[u], prev_o,
                prev_d, c, L, X + (int64_t)u * L, lam_scratch.data(), zbl_scratch.data(), V, false,
                nullptr, nullptr, nullptr);
            ok[u] = sr.ok ? 1 : 0;
        }
    };

    if (n_threads <= 1) {
        worker();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int t = 0; t < n_threads; t++) threads.emplace_back(worker);
        for (auto &th : threads) th.join();
    }

    nb::capsule X_owner(X, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule ok_owner(ok, [](void *p) noexcept { delete[] (uint8_t *)p; });
    size_t X_shape[2] = {(size_t)U, (size_t)L};
    size_t ok_shape[1] = {(size_t)U};
    auto X_out = nb::ndarray<nb::numpy, double>(X, 2, X_shape, X_owner);
    auto ok_out = nb::ndarray<nb::numpy, uint8_t>(ok, 1, ok_shape, ok_owner);
    return nb::make_tuple(X_out, ok_out);
#else
    (void)A_indptr_arr;
    (void)A_indices_arr;
    (void)A_data_arr;
    (void)scale_m_arr;
    (void)V;
    (void)L;
    (void)c_arr;
    (void)origins_arr;
    (void)dests_arr;
    (void)qp_eps_abs;
    (void)qp_eps_duality_gap_abs;
    (void)qp_max_iter;
    (void)n_threads;
    throw std::runtime_error("batch_solve_flows requires the PIQP backend (HAVE_PIQP).");
#endif
}

// Parallel Markovian route sampler (nanobind wrapper over
// batch_sample_routes_impl). Draws N routes, one per trip, from optimal link
// flows. Each trip's flow is `flows[flow_row[i]]` and each route is a
// flow-proportional simple path. flow_row is bounds-checked here, because the
// implementation indexes `flows` with it directly. Returns
// (Y (N,L) float64 indicator, valid (N,) uint8).
static nb::tuple batch_sample_routes_native(
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> tail_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> head_arr, int V,
    nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig> flows_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> flow_row_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> origins_arr,
    nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig> dests_arr,
    nb::ndarray<uint64_t, nb::shape<-1>, nb::c_contig> seeds_arr, double threshold_factor,
    double threshold_min, int max_steps, int n_threads) {
    int L = (int)tail_arr.shape(0);
    int M = (int)flows_arr.shape(0);
    int N = (int)flow_row_arr.shape(0);
    if ((int)flows_arr.shape(1) != L)
        throw std::invalid_argument("flows second dim must equal L (number of links)");
    if ((int)origins_arr.shape(0) != N || (int)dests_arr.shape(0) != N ||
        (int)seeds_arr.shape(0) != N)
        throw std::invalid_argument("origins/dests/seeds must have length N == len(flow_row)");

    const int64_t *tail64 = tail_arr.data();
    const int64_t *head64 = head_arr.data();
    const int64_t *frow64 = flow_row_arr.data();
    const int64_t *orig64 = origins_arr.data();
    const int64_t *dest64 = dests_arr.data();

    std::vector<int> tail(L), head(L), flow_row(N), origins(N), dests(N);
    for (int e = 0; e < L; ++e) {
        tail[e] = (int)tail64[e];
        head[e] = (int)head64[e];
    }
    for (int i = 0; i < N; ++i) {
        int m = (int)frow64[i];
        if (m < 0 || m >= M) throw std::invalid_argument("flow_row entry out of range [0, M)");
        flow_row[i] = m;
        origins[i] = (int)orig64[i];
        dests[i] = (int)dest64[i];
    }

    double *Y = new double[(int64_t)N * L]();
    uint8_t *valid = new uint8_t[N]();

    batch_sample_routes_impl(tail.data(), head.data(), L, V, flows_arr.data(), M, flow_row.data(),
                             N, origins.data(), dests.data(), seeds_arr.data(), threshold_factor,
                             threshold_min, max_steps, n_threads, Y, valid);

    nb::capsule Y_owner(Y, [](void *p) noexcept { delete[] (double *)p; });
    nb::capsule valid_owner(valid, [](void *p) noexcept { delete[] (uint8_t *)p; });
    size_t Y_shape[2] = {(size_t)N, (size_t)L};
    size_t valid_shape[1] = {(size_t)N};
    auto Y_out = nb::ndarray<nb::numpy, double>(Y, 2, Y_shape, Y_owner);
    auto valid_out = nb::ndarray<nb::numpy, uint8_t>(valid, 1, valid_shape, valid_owner);
    return nb::make_tuple(Y_out, valid_out);
}

NB_MODULE(_native, m) {
    m.doc() = "Native fused per-trip pipeline for micropurc estimation";

    m.def(
        "get_num_threads",
        []() -> int {
            const char *env = std::getenv("OMP_NUM_THREADS");
            if (env) {
                int v = std::atoi(env);
                if (v > 0) return v;
            }
            int hw = (int)std::thread::hardware_concurrency();
            return hw > 0 ? hw : 1;
        },
        "Return the number of worker threads available.");

    m.def("active_threshold", &active_threshold_native, nb::arg("x"), nb::arg("threshold_factor"),
          nb::arg("threshold_min"), "Compute the scale-aware active-flow threshold.");

    m.def("batch_solve_flows", &batch_solve_flows, nb::arg("A_indptr"), nb::arg("A_indices"),
          nb::arg("A_data"), nb::arg("scale_m"), nb::arg("V"), nb::arg("L"), nb::arg("c"),
          nb::arg("origins"), nb::arg("dests"), nb::arg("qp_eps_abs") = 1e-12,
          nb::arg("qp_eps_duality_gap_abs") = 1e-12, nb::arg("qp_max_iter") = 1000,
          nb::arg("n_threads") = -1,
          "Solve U unit-demand OD forward QPs in parallel (std::thread).\n\n"
          "Returns (X (U,L) float64 optimal flows, ok (U,) uint8 solve-success).");

    m.def("batch_sample_routes", &batch_sample_routes_native, nb::arg("tail"), nb::arg("head"),
          nb::arg("V"), nb::arg("flows"), nb::arg("flow_row"), nb::arg("origins"), nb::arg("dests"),
          nb::arg("seeds"), nb::arg("threshold_factor"), nb::arg("threshold_min"),
          nb::arg("max_steps") = -1, nb::arg("n_threads") = -1,
          "Sample N flow-proportional simple-path routes in parallel (std::thread).\n\n"
          "Each trip i walks flows[flow_row[i]] from origins[i] to dests[i] using a\n"
          "per-trip mt19937_64(seeds[i]). Returns (Y (N,L) float64, valid (N,) uint8).");

    m.def("check_dag_connectivity", &check_dag_connectivity_native, nb::arg("tail_active"),
          nb::arg("head_active"), nb::arg("dest"), nb::arg("V"),
          "Check DAG and destination reachability for an active edge list.");

    m.def("find_kept_rows", &find_kept_rows_native, nb::arg("tail_active"), nb::arg("head_active"),
          nb::arg("dest"), nb::arg("V"), nb::arg("projection_mode"),
          "Find grounded incidence rows for an active edge list.");

    m.def("determine_active_set", &determine_active_set_native, nb::arg("A_indptr"),
          nb::arg("A_indices"), nb::arg("A_data"), nb::arg("tail"), nb::arg("head"), nb::arg("V"),
          nb::arg("L"), nb::arg("dest"), nb::arg("c"), nb::arg("lam_eq"), nb::arg("scale_m"),
          nb::arg("slack_tol"), nb::arg("projection_mode"),
          "Determine active links and grounded rows via the native C++ implementation.");

    m.def("schur_contribution", &schur_contribution_native, nb::arg("Z_act"), nb::arg("A_local"),
          nb::arg("y_act"), nb::arg("scale_act"), nb::arg("x_hat_act") = nb::none(),
          "Compute one Schur contribution via the native C++ implementation.");

#ifdef HAVE_PIQP
    nb::enum_<piqp::Status>(m, "Status", nb::is_arithmetic(),
                            "Solver status codes, mirroring the PIQP C++ enum.")
        .value("PIQP_SOLVED", piqp::PIQP_SOLVED)
        .value("PIQP_MAX_ITER_REACHED", piqp::PIQP_MAX_ITER_REACHED)
        .value("PIQP_PRIMAL_INFEASIBLE", piqp::PIQP_PRIMAL_INFEASIBLE)
        .value("PIQP_DUAL_INFEASIBLE", piqp::PIQP_DUAL_INFEASIBLE)
        .value("PIQP_NUMERICS", piqp::PIQP_NUMERICS)
        .value("PIQP_UNSOLVED", piqp::PIQP_UNSOLVED)
        .value("PIQP_INVALID_SETTINGS", piqp::PIQP_INVALID_SETTINGS);

    nb::class_<NativeSolverSettings>(m, "NativeSolverSettings",
                                     "Termination settings a NativeSparseSolver was built with.")
        .def_ro("verbose", &NativeSolverSettings::verbose)
        .def_ro("eps_abs", &NativeSolverSettings::eps_abs)
        .def_ro("eps_duality_gap_abs", &NativeSolverSettings::eps_duality_gap_abs)
        .def_ro("eps_rel", &NativeSolverSettings::eps_rel)
        .def_ro("eps_duality_gap_rel", &NativeSolverSettings::eps_duality_gap_rel)
        .def_ro("max_iter", &NativeSolverSettings::max_iter);

    nb::class_<NativeSparseSolver>(m, "NativeSparseSolver")
        .def(nb::init<nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<double, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<double, nb::shape<-1>, nb::c_contig>, int, int, bool, double,
                      double, double, double, int>(),
             nb::arg("A_indptr"), nb::arg("A_indices"), nb::arg("A_data"), nb::arg("scale_m"),
             nb::arg("V"), nb::arg("L"), nb::arg("verbose"), nb::arg("eps_abs"),
             nb::arg("eps_duality_gap_abs"), nb::arg("eps_rel"), nb::arg("eps_duality_gap_rel"),
             nb::arg("max_iter"),
             "Set up one forward QP over the CSC equality matrix A (V x L) and the\n"
             "diagonal perturbation diag(scale_m):\n\n"
             "  min_x  c'x + 0.5 x' diag(scale_m) x   s.t.  A x = b,  x >= 0,\n\n"
             "with no upper bound on x. The setup runs once; solve(c, b) then only\n"
             "updates the two vectors.")
        .def("solve", &NativeSparseSolver::solve, nb::arg("c"), nb::arg("b"),
             "Update (c, b) and solve.\n\n"
             "Returns (x (L,) float64, y (V,) float64 equality duals, info dict with\n"
             "status, iter, primal_residual, dual_residual, duality_gap).")
        .def_ro("settings", &NativeSparseSolver::settings_,
                "Termination settings this solver was constructed with.");

    nb::class_<NativePIQPContext>(m, "NativePIQPContext")
        .def(nb::init<nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<double, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<double, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig>,
                      nb::ndarray<int64_t, nb::shape<-1>, nb::c_contig>, int, int>(),
             nb::arg("A_indptr"), nb::arg("A_indices"), nb::arg("A_data"), nb::arg("scale_m"),
             nb::arg("tail"), nb::arg("head"), nb::arg("V"), nb::arg("L"))
        .def("batch_lazy_terms", &NativePIQPContext::batch_lazy_terms, nb::arg("link_attrs"),
             nb::arg("variant_attr"), nb::arg("variant_cov"), nb::arg("term_code"),
             nb::arg("term_a"), nb::arg("term_b"), nb::arg("term_cov"), nb::arg("term_op"),
             nb::arg("term_thr"), nb::arg("beta"), nb::arg("origins"), nb::arg("dests"),
             nb::arg("y_indptr"), nb::arg("y_indices"), nb::arg("y_data"), nb::arg("trip_variant"),
             nb::arg("projection_mode"), nb::arg("return_u"), nb::arg("slack_tol") = 0.0,
             nb::arg("qp_eps_abs") = 1e-12, nb::arg("qp_eps_duality_gap_abs") = 1e-12,
             nb::arg("qp_eps_rel") = 0.0, nb::arg("qp_eps_duality_gap_rel") = 0.0,
             nb::arg("qp_max_iter") = 1000,
             "Lazy QP solve + active projection over a general term-coded design.\n\n"
             "Terms are evaluated per link and variant from link_attrs\n"
             "(attr_variant, L, A), per-variant covariates variant_cov (n_var, C),\n"
             "and variant_attr mapping each variant to its attribute-variant row.\n\n"
             "Warm-starts each OD-group's forward QP from its previous-iteration\n"
             "primal-dual point (persisted in this context). Solutions are unchanged\n"
             "(strictly convex QP); only IPM iteration counts drop.")
        .def("batch_predict_flows", &NativePIQPContext::batch_predict_flows, nb::arg("link_attrs"),
             nb::arg("variant_attr"), nb::arg("variant_cov"), nb::arg("term_code"),
             nb::arg("term_a"), nb::arg("term_b"), nb::arg("term_cov"), nb::arg("term_op"),
             nb::arg("term_thr"), nb::arg("beta"), nb::arg("origins"), nb::arg("dests"),
             nb::arg("trip_variant"), nb::arg("slack_tol") = 0.0, nb::arg("qp_eps_abs") = 1e-12,
             nb::arg("qp_eps_duality_gap_abs") = 1e-12, nb::arg("qp_eps_rel") = 0.0,
             nb::arg("qp_eps_duality_gap_rel") = 0.0, nb::arg("qp_max_iter") = 1000,
             "Parallel forward solve at a fixed beta; returns per-trip predicted flows on\n"
             "the dual-slack active set as (indptr, indices, data) of an (N, L) CSR.")
        .def("reset_warm_start", &NativePIQPContext::reset_warm_start,
             "Clear persisted warm-start state (call at the start of a fresh fit()).")
        .def_rw("warm_start_enabled", &NativePIQPContext::warm_enabled_,
                "If False, every forward QP solves cold (warm-start disabled).")
        .def_rw("cts_enabled", &NativePIQPContext::cts_enabled_,
                "Certified Tiered Solve: re-solve on the cached active set and skip the "
                "IPM when the exact full-KKT certificate passes. Sound by construction; "
                "default True (set False to force every solve through the IPM).")
        .def_rw("cts_tau_primal", &NativePIQPContext::cts_tau_p_)
        .def_rw("cts_tau_dual", &NativePIQPContext::cts_tau_d_)
        .def_rw("cts_repair_budget_factor", &NativePIQPContext::cts_repair_budget_factor_,
                "Repair effort cap = factor * (touched nodes + initial violations) "
                "+ 512 worklist pops. Sized to the candidate's own size: "
                "successful repairs are local, so budget beyond the success tail "
                "only delays the honest IPM fallback.");
#endif

    m.def("batch_solve_and_accumulate", &batch_solve_and_accumulate, nb::arg("Z"),
          nb::arg("A_indptr"), nb::arg("A_indices"), nb::arg("A_data"), nb::arg("scale_m"),
          nb::arg("tail"), nb::arg("head"), nb::arg("V"), nb::arg("L"), nb::arg("K"), nb::arg("c"),
          nb::arg("origins"), nb::arg("dests"), nb::arg("y_indptr"), nb::arg("y_indices"),
          nb::arg("y_data"), nb::arg("lam_warm"), nb::arg("od_keys_prev"), nb::arg("trip_variant"),
          nb::arg("n_z_variants"), nb::arg("projection_mode"), nb::arg("return_u"),
          nb::arg("slack_tol") = 0.0, nb::arg("qp_eps_abs") = 1e-12,
          nb::arg("qp_eps_duality_gap_abs") = 1e-12, nb::arg("qp_eps_rel") = 0.0,
          nb::arg("qp_eps_duality_gap_rel") = 0.0, nb::arg("qp_max_iter") = 1000,
          "Memory-optimized fused QP-solve + accumulate pipeline.\n\n"
          "Active-set detection via dual slack (KKT sign test).\n"
          "Groups by (OD, variant) for trip-specific Z support.\n"
          "Solver reused across groups via update(c, b).\n"
          "Warm-start keyed by group: OD for single Z, OD-variant otherwise.\n\n"
          "Returns (H, g, u_or_None, lam_cache, group_keys, stats_dict).");
}
