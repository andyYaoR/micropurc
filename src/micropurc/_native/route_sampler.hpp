// Parallel Markovian route sampler.
//
// Draws routes from optimal link flows by a flow-proportional random walk:
// at each node v the next edge e is chosen with probability
//   p(e | v) = x_e / sum_{e' in out(v)} x_{e'},
// so E[y] = x* component-wise. Only simple paths are accepted; a walk that
// revisits a node is rejected (valid = false). The acceptance rules and the
// activity threshold match the reference Python sampler in
// simulation/route_sampler.py, so the two agree in distribution.
//
// Threading via std::thread + an atomic work-stealing counter (no OpenMP),
// each trip driven by its own std::mt19937_64 seeded from `seeds[i]`.
#pragma once

#include <cstdint>

// Samples N routes in parallel. All pointer arguments are caller-owned.
//
// Topology (shared, read-only):
//   tail, head : (L,) endpoint node indices of each directed link.
//   V, L       : node and link counts.
// Flows:
//   flows      : (M, L) row-major link-flow vectors (>= 0 after clamping).
//   M          : number of distinct flow rows.
//   flow_row   : (N,) index into `flows` giving each trip's flow row.
// Trips:
//   N              : number of trips (routes to sample).
//   origins, dests : (N,) OD endpoints.
//   seeds          : (N,) per-trip RNG seeds.
// Cleaning (must match ActiveSetConfig): a link is present iff its clamped
// flow >= max(threshold_min, threshold_factor * max_e flow_e).
//   max_steps : walk-length cap; if <= 0, defaults to 2*(V + L).
//   n_threads : worker count; if <= 0, uses hardware_concurrency().
// Outputs (caller-allocated):
//   Y     : (N, L) row-major, pre-zeroed; set to 1.0 on the sampled links.
//   valid : (N,) 1 if a simple o->d route was drawn, 0 otherwise.
void batch_sample_routes_impl(const int *tail, const int *head, int L, int V, const double *flows,
                              int M, const int *flow_row, int N, const int *origins,
                              const int *dests, const uint64_t *seeds, double threshold_factor,
                              double threshold_min, int max_steps, int n_threads, double *Y,
                              uint8_t *valid);
