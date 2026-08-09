// Schur-complement building blocks: dense submatrix extraction and the per-trip
// contribution to H, g, and u. The projector P = I - A^T (A A^T)^{-1} A is never
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
