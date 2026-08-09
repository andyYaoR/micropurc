// Shared includes, small numerical helpers, and result types for the native
// estimation pipeline.
//
// The pipeline is split across translation units:
//   activeset.{hpp,cpp} — DAG/connectivity checks and kept-row grounding
//   schur.{hpp,cpp}      — dense submatrix extraction and Schur contributions
//   bindings.cpp         — PIQP solve, active-set projection, the solver
//                          context, the batch driver, and the nanobind module
//
// Build: scikit-build-core + nanobind. Uses std::thread (no OpenMP/libomp).
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef HAVE_LAPACK
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
extern "C" {
void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
}
#endif
static constexpr int LAPACK_THRESHOLD = 24;
#endif

#ifdef HAVE_PIQP
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "piqp/piqp.hpp"
#include "piqp/sparse/preconditioner.hpp"
using PIQPSolver =
    piqp::SparseSolver<double, int, piqp::sparse::IdentityPreconditioner<double, int>>;
#endif

namespace nb = nanobind;

inline nb::ndarray<nb::numpy, int64_t> make_int64_array(const std::vector<int> &vals) {
    int64_t *data = new int64_t[vals.size()];
    for (size_t i = 0; i < vals.size(); i++) data[i] = (int64_t)vals[i];
    nb::capsule owner(data, [](void *p) noexcept { delete[] (int64_t *)p; });
    size_t shape[1] = {vals.size()};
    return nb::ndarray<nb::numpy, int64_t>(data, 1, shape, owner);
}

inline nb::ndarray<nb::numpy, double> make_double_array(const std::vector<double> &vals) {
    double *data = new double[vals.size()];
    if (!vals.empty()) std::memcpy(data, vals.data(), vals.size() * sizeof(double));
    nb::capsule owner(data, [](void *p) noexcept { delete[] (double *)p; });
    size_t shape[1] = {vals.size()};
    return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
}

// Inline small-matrix routines, usable without a LAPACK dependency. The Schur
// path calls them for every grounded system with m <= LAPACK_THRESHOLD rows,
// where the BLAS/LAPACK call overhead dominates the arithmetic.

// In-place Cholesky: G[m×m] row-major -> lower-triangular L, with the strict
// upper triangle zeroed. Returns 0 on success, or the 1-based index of the
// first leading minor whose pivot is non-positive (G is not SPD).
inline int small_cholesky(double *G, int m) {
    for (int j = 0; j < m; j++) {
        double s = G[j * m + j];
        for (int k = 0; k < j; k++) s -= G[j * m + k] * G[j * m + k];
        if (s <= 0.0) return j + 1;
        G[j * m + j] = std::sqrt(s);
        double inv_ljj = 1.0 / G[j * m + j];
        for (int i = j + 1; i < m; i++) {
            s = G[i * m + j];
            for (int k = 0; k < j; k++) s -= G[i * m + k] * G[j * m + k];
            G[i * m + j] = s * inv_ljj;
        }
        for (int i = 0; i < j; i++) G[i * m + j] = 0.0;
    }
    return 0;
}

// Solve L L^T X = B in-place (B overwritten with X), a forward substitution
// followed by a backward one. L[m×m] lower-triangular row-major (as produced by
// small_cholesky), B[m×nrhs] row-major.
inline void small_cho_solve(const double *L, double *B, int m, int nrhs) {
    // Forward: L y = B
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < m; i++) {
            double s = B[i * nrhs + j];
            for (int k = 0; k < i; k++) s -= L[i * m + k] * B[k * nrhs + j];
            B[i * nrhs + j] = s / L[i * m + i];
        }
    }
    // Backward: L^T x = y
    for (int j = 0; j < nrhs; j++) {
        for (int i = m - 1; i >= 0; i--) {
            double s = B[i * nrhs + j];
            for (int k = i + 1; k < m; k++) s -= L[k * m + i] * B[k * nrhs + j];
            B[i * nrhs + j] = s / L[i * m + i];
        }
    }
}

// Union-Find over a dense 0..n-1 node range, used to split an active subgraph
// into connected components. Path halving in find() plus union by rank.
struct UnionFind {
    std::vector<int> parent, rank_;
    int n_components;

    UnionFind(int n) : parent(n), rank_(n, 0), n_components(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
        n_components--;
    }
};

// FNV-1a hash of an active-set index list, used to tell whether a trip's active
// set changed between iterations. Order-sensitive, so callers must pass the
// indices in a canonical order (build_active_projection emits them ascending).
// The result is remapped away from 0, which is reserved to mark "no active set
// recorded for this trip".
inline int64_t hash_active_set(const int *active_idx, int n_active) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < n_active; i++) {
        h ^= (uint64_t)(unsigned int)active_idx[i];
        h *= 1099511628211ULL;
    }
    int64_t out = (int64_t)h;
    return out == 0 ? 1 : out;
}

// Result of active-set detection + grounding + extraction for one trip.
// n_active and m_rows are the dimensions of the extracted grounded incidence
// A_local (m_rows x n_active); they are meaningful only when valid is true.
struct ActiveProjection {
    int n_active;
    int m_rows;
    bool valid;
    // Why a trip was dropped: 0=valid, 1=no_active_links, 2=trivial_flow,
    // 3=not_dag, 4=no_kept_rows. Aggregated into per-reason drop counters.
    int drop_reason = 0;
};
