// Schur-complement building blocks: dense submatrix extraction, the per-trip
// contribution to H, g, and u, and GroupSchur, the same contribution split into
// what an OD group's active set determines (factorised once, sparsely) and what
// each trip's route adds. The projector P = I - A^T (A A^T)^{-1} A is never
// materialized; only the m×m system G = A A^T is factorized (see schur.cpp).
#pragma once

#include <cstdint>
#include <vector>

// Extract the dense submatrix A[kept_rows, active_idx] from a CSC matrix into
// the caller's row-major (n_kept x n_active) buffer. kept_rows must be
// ascending.
void extract_dense_submatrix(const int64_t *A_indptr, const int64_t *A_indices,
                             const double *A_data, const int *active_idx, int n_active,
                             const int *kept_rows, int n_kept, double *buf);

// One trip's Schur-complement contribution through the range-space projector.
// Writes H_i (K×K), g_i (K), and, when w_r is non-null, u_i (K). e is the
// active-link count and m the grounded row count. Returns 0 on success, or the
// Cholesky failure code of G = A A^T.
int schur_contribution(const double *Z_act, const double *A_local, const double *w_y,
                       const double *w_r, int e, int m, int K, double *H_i, double *g_i,
                       double *u_i);

// Accumulate one trip's contribution directly from its sparse route indicators
// (CSR slice y_indices/y_data over [y_start, y_end)), reusing the caller-provided
// scratch buffers so a worker thread allocates nothing per trip.
// active_pos_map must be an L-length buffer holding -1 everywhere; it is left in
// that state on return. Returns true on success.
bool accumulate_trip_sparse(const int *active_idx, int n_active, int m_rows, int K,
                            const double *Z_act, const double *A_local, const double *scale_m,
                            const int64_t *y_indices, const double *y_data, int64_t y_start,
                            int64_t y_end, const double *x, bool compute_u,
                            std::vector<int> &active_pos_map, std::vector<double> &y_active_buf,
                            std::vector<double> &w_y_buf, std::vector<double> &w_r_buf, double *H_i,
                            double *g_i, double *u_i);

// The part of the Schur step that depends only on an OD group's active set,
// computed once for all its trips (schur.cpp): the projected design R = P Z
// (e x K, row-major) and H = Z' P Z (K x K). prepare returns false when the
// grounded Laplacian G = A A' has no Cholesky factor, which drops every trip of
// the group, as each trip's own factorisation of the same G would.
struct GroupSchur {
    int e = 0, m = 0, K = 0;
    bool ok = false;
    std::vector<double> R;    // P Z, one row per active link
    std::vector<double> H;    // Z' P Z, every trip's H_i
    std::vector<int> pos;     // link -> its row of R, -1 off the active set (n_links)
    std::vector<int> linked;  // the links pos currently maps, cleared at the next prepare

    // From the sparse incidence: the active links' columns, restricted to
    // kept_rows (ascending). row_map_scratch is -1 everywhere and comes back so.
    bool prepare(const double *Z_act, const int64_t *A_indptr, const int64_t *A_indices,
                 const double *A_data, const int *active_idx, int n_active, const int *kept_rows,
                 int n_kept, int K, int n_links, std::vector<int> &row_map_scratch);
    // From the dense grounded incidence A_local (m x e, row-major).
    bool prepare_dense(const double *Z_act, const double *A_local, int e, int m, int K);
    // c = R' (scale . x_hat) over the active links, so that u_i = g_i - c.
    void scores_offset(const double *scale_m, const double *x, std::vector<double> &c) const;
    // One trip from its route, links off the active set ignored. u_i is
    // written when c is given.
    void trip(const double *scale_m, const int64_t *y_indices, const double *y_data,
              int64_t y_start, int64_t y_end, const double *c, double *H_i, double *g_i,
              double *u_i) const;
    // One trip from dense weight vectors over the active links (w_r may be null).
    void trip_dense(const double *w_y, const double *w_r, double *H_i, double *g_i,
                    double *u_i) const;
};
