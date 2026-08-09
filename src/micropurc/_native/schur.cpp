// Definitions for schur.hpp: the per-trip Schur-complement contribution.

#include "schur.hpp"

#include "common.hpp"

// Extract dense submatrix A[kept_rows, active_cols] from CSC.
// Output: row-major (m × e) stored in buf, zero-filled first so structural
// zeros of the sparse matrix come out as explicit zeros.
// kept_rows must be ascending: its last entry sizes the row map, and rows above
// that bound are skipped.
void extract_dense_submatrix(const int64_t *A_indptr, const int64_t *A_indices,
                             const double *A_data, const int *active_idx, int n_active,
                             const int *kept_rows, int n_kept, double *buf) {
    int m = n_kept, e = n_active;
    std::memset(buf, 0, (size_t)m * e * sizeof(double));

    // Row → dense index map for O(1) lookup; -1 marks a row that was grounded
    // out and must be skipped.
    int max_row = kept_rows[n_kept - 1] + 1;
    std::vector<int> row_map(max_row, -1);
    for (int i = 0; i < n_kept; i++) {
        row_map[kept_rows[i]] = i;
    }

    for (int j = 0; j < n_active; j++) {
        int col = active_idx[j];
        for (int64_t p = A_indptr[col]; p < A_indptr[col + 1]; p++) {
            int row = (int)A_indices[p];
            if (row < max_row && row_map[row] >= 0) {
                buf[row_map[row] * e + j] = A_data[p];
            }
        }
    }
}

// One trip's contribution through the range-space projector
//   P = I - A^T (A A^T)^{-1} A,
// the orthogonal projection onto the null space of the grounded active
// incidence A. The quantities wanted are
//   H_i = Z^T P Z,   g_i = Z^T P w_y,   u_i = Z^T P w_r,
// and P is never formed: with G = A A^T (m×m, m much smaller than E) each
// M in [Z | w_y | w_r] gives
//   Z^T P M = Z^T M - (A Z)^T G^{-1} (A M),
// so the only inverse needed is one m×m Cholesky solve shared by all columns.
// Grounding guarantees G is SPD, so a Cholesky failure means the active set is
// broken; it is surfaced to the caller as a dropped trip.
//
// All buffers are row-major and function-local, so concurrent calls on distinct
// outputs are independent.
//
// Returns 0 on success, nonzero if the Cholesky fails.
int schur_contribution(const double *Z_act,    // (e, K) row-major
                       const double *A_local,  // (m, e) row-major
                       const double *w_y,      // (e,)
                       const double *w_r,      // (e,) or nullptr
                       int e, int m, int K,
                       double *H_i,  // (K, K) output, row-major
                       double *g_i,  // (K,) output
                       double *u_i   // (K,) output, only if w_r != nullptr
) {
    int n_rhs = K + 1 + (w_r ? 1 : 0);

    // Build M (e × n_rhs) row-major = [Z_act | w_y | w_r]. Stacking the right-
    // hand sides lets one triangular solve serve H_i, g_i, and u_i.
    std::vector<double> M(e * n_rhs);
    for (int i = 0; i < e; i++) {
        for (int k = 0; k < K; k++) M[i * n_rhs + k] = Z_act[i * K + k];
        M[i * n_rhs + K] = w_y[i];
        if (w_r) M[i * n_rhs + K + 1] = w_r[i];
    }

    // G = A A^T  (m × m, row-major lower triangle sufficient)
    std::vector<double> G(m * m, 0.0);
#ifdef HAVE_LAPACK
    if (m > LAPACK_THRESHOLD) {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, m, e, 1.0, A_local, e, 0.0, G.data(),
                    m);
    } else
#endif
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                double s = 0.0;
                for (int p = 0; p < e; p++) s += A_local[i * e + p] * A_local[j * e + p];
                G[i * m + j] = s;
                G[j * m + i] = s;
            }
    }

    // Cholesky: G → L in lower triangle (row-major)
#ifdef HAVE_LAPACK
    if (m > LAPACK_THRESHOLD) {
        // LAPACK reads column-major, so a row-major buffer is its transpose.
        // G is symmetric, so asking dpotrf for the upper factor of the transpose
        // leaves L exactly where the row-major reader expects it: the lower
        // triangle. Keep 'U' paired with the CblasLower dsyrk above.
        char uplo = 'U';
        int info = 0;
        dpotrf_(&uplo, &m, G.data(), &m, &info);
        if (info != 0) return info;
    } else
#endif
    {
        int info = small_cholesky(G.data(), m);
        if (info != 0) return info;
    }

    // B = A @ M  (m × n_rhs, row-major)
    std::vector<double> B(m * n_rhs, 0.0);
#ifdef HAVE_LAPACK
    if (m > LAPACK_THRESHOLD) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n_rhs, e, 1.0, A_local, e,
                    M.data(), n_rhs, 0.0, B.data(), n_rhs);
    } else
#endif
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n_rhs; j++) {
                double s = 0.0;
                for (int p = 0; p < e; p++) s += A_local[i * e + p] * M[p * n_rhs + j];
                B[i * n_rhs + j] = s;
            }
    }

    // AZ = B[:, 0:K], saved before the in-place solve overwrites B. Both A Z and
    // G^{-1} A Z are needed for the K×K term.
    std::vector<double> AZ(m * K);
    for (int i = 0; i < m; i++)
        for (int k = 0; k < K; k++) AZ[i * K + k] = B[i * n_rhs + k];

    // T = G^{-1} B  (solve L L^T T = B, in-place on B)
    small_cho_solve(G.data(), B.data(), m, n_rhs);
    // B now contains T (m × n_rhs, row-major).

    // Z^T Z  (K × K, row-major); symmetric, so only the lower half is computed.
    std::vector<double> ZtZ(K * K, 0.0);
    for (int i = 0; i < K; i++)
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int p = 0; p < e; p++) s += Z_act[p * K + i] * Z_act[p * K + j];
            ZtZ[i * K + j] = s;
            ZtZ[j * K + i] = s;
        }

    // AZ^T @ T_Z  (K × K, row-major)
    // T_Z = B[:, 0:K]
    std::vector<double> AZt_TZ(K * K, 0.0);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) {
            double s = 0.0;
            for (int p = 0; p < m; p++) s += AZ[p * K + i] * B[p * n_rhs + j];
            AZt_TZ[i * K + j] = s;
        }

    // H_i = Z^T Z - AZ^T T_Z
    for (int i = 0; i < K * K; i++) H_i[i] = ZtZ[i] - AZt_TZ[i];

    // g_i = Z^T w_y - AZ^T T_y
    for (int k = 0; k < K; k++) {
        double s = 0.0;
        for (int i = 0; i < e; i++) s += Z_act[i * K + k] * w_y[i];
        g_i[k] = s;
    }
    for (int k = 0; k < K; k++) {
        double s = 0.0;
        for (int i = 0; i < m; i++) s += AZ[i * K + k] * B[i * n_rhs + K];
        g_i[k] -= s;
    }

    // u_i = Z^T w_r - AZ^T T_r
    if (w_r && u_i) {
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int i = 0; i < e; i++) s += Z_act[i * K + k] * w_r[i];
            u_i[k] = s;
        }
        for (int k = 0; k < K; k++) {
            double s = 0.0;
            for (int i = 0; i < m; i++) s += AZ[i * K + k] * B[i * n_rhs + K + 1];
            u_i[k] -= s;
        }
    }

    return 0;
}

// Gathers one trip's sparse route indicator onto the active set, forms the two
// weight vectors the Schur step consumes, and accumulates the contribution.
//
//   w_y[i] = scale_j * y_j          (the observed route, restricted to active j)
//   w_r[i] = scale_j * (y_j - x_j)  (the residual, only when compute_u)
//
// active_pos_map is a caller-owned L-length scatter table holding -1 everywhere.
// It is filled for the active links, read once per route entry, and restored to
// -1 on exactly those links, so the reset touches only the entries it dirtied
// and leaves the table clean for the next trip on this thread.
bool accumulate_trip_sparse(const int *active_idx, int n_active, int m_rows, int K,
                            const double *Z_act, const double *A_local, const double *scale_m,
                            const int64_t *y_indices, const double *y_data, int64_t y_start,
                            int64_t y_end, const double *x, bool compute_u,
                            std::vector<int> &active_pos_map, std::vector<double> &y_active_buf,
                            std::vector<double> &w_y_buf, std::vector<double> &w_r_buf, double *H_i,
                            double *g_i, double *u_i) {
    y_active_buf.assign(n_active, 0.0);
    for (int i = 0; i < n_active; i++) active_pos_map[active_idx[i]] = i;
    for (int64_t p = y_start; p < y_end; p++) {
        int edge = (int)y_indices[p];
        int pos = active_pos_map[edge];
        if (pos >= 0) y_active_buf[pos] += y_data[p];
    }

    w_y_buf.resize(n_active);
    for (int i = 0; i < n_active; i++) w_y_buf[i] = scale_m[active_idx[i]] * y_active_buf[i];

    double *w_r_ptr = nullptr;
    if (compute_u) {
        w_r_buf.resize(n_active);
        for (int i = 0; i < n_active; i++) {
            int idx = active_idx[i];
            w_r_buf[i] = scale_m[idx] * (y_active_buf[i] - x[idx]);
        }
        w_r_ptr = w_r_buf.data();
    }

    // O(n_active) restore of the scatter table; skipping it would corrupt the
    // next trip handled by this thread.
    for (int i = 0; i < n_active; i++) active_pos_map[active_idx[i]] = -1;

    std::fill(H_i, H_i + K * K, 0.0);
    std::fill(g_i, g_i + K, 0.0);
    if (u_i) std::fill(u_i, u_i + K, 0.0);

    return schur_contribution(Z_act, A_local, w_y_buf.data(), w_r_ptr, n_active, m_rows, K, H_i,
                              g_i, u_i) == 0;
}
