// Implementation of the parallel Markovian route sampler (see header).

#include "route_sampler.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace {

// One route drawn on `flow` (length L). Writes the L-indicator into `y_out`
// (assumed pre-zeroed for this trip). Returns true iff a simple o->d path was
// found; on failure `y_out` still holds the links visited before the walk gave
// up, so callers must ignore the row unless the return value is true.
//
// `visited` is a V-length stamp buffer and `stamp` is bumped once per call, so
// "not yet visited on this walk" is `visited[v] != stamp`. The reset between
// trips is the increment alone (O(1)), which matters because V dwarfs the
// handful of nodes one route touches.
bool sample_one_route(const int *head, int V, const double *flow, const int *out_ptr,
                      const int *out_eidx, int origin, int dest, double eps, int max_steps,
                      std::mt19937_64 &gen, std::vector<int> &visited, int stamp, double *y_out) {
    (void)V;
    // Origin feasibility: clamped, thresholded outflow must sum to ~1.
    double origin_out = 0.0;
    bool origin_has_edge = false;
    for (int p = out_ptr[origin]; p < out_ptr[origin + 1]; ++p) {
        double f = flow[out_eidx[p]];
        if (f >= eps && f > 0.0) {
            origin_out += f;
            origin_has_edge = true;
        }
    }
    // Same test as np.isclose(origin_out, 1.0, atol=1e-5, rtol=1e-3) in the
    // Python sampler: the tolerance atol + rtol * |1.0| folds to the constant
    // below because the reference value is exactly 1.
    if (std::fabs(origin_out - 1.0) > 1e-5 + 1e-3) return false;
    if (!origin_has_edge) return false;

    std::uniform_real_distribution<double> unif(0.0, 1.0);

    visited[origin] = stamp;
    int current = origin;

    for (int step = 0; step < max_steps; ++step) {
        if (current == dest) break;

        int begin = out_ptr[current], end = out_ptr[current + 1];
        // First pass: total present flow at this node.
        double total = 0.0;
        for (int p = begin; p < end; ++p) {
            double f = flow[out_eidx[p]];
            if (f >= eps && f > 0.0) total += f;
        }
        if (total <= 0.0) return false;

        // Flow-proportional draw over the present edges by inverse CDF, which is
        // what makes E[y] = x* per link.
        double r = unif(gen) * total;
        double acc = 0.0;
        int e_chosen = -1;
        for (int p = begin; p < end; ++p) {
            int e = out_eidx[p];
            double f = flow[e];
            if (f >= eps && f > 0.0) {
                acc += f;
                if (r <= acc) {
                    e_chosen = e;
                    break;
                }
            }
        }
        if (e_chosen < 0) {
            // The scan can fall through when the running sum lands a rounding
            // step below r on the final edge. Take the last present edge, which
            // is the one the exact arithmetic would have selected.
            for (int p = end - 1; p >= begin; --p) {
                int e = out_eidx[p];
                double f = flow[e];
                if (f >= eps && f > 0.0) {
                    e_chosen = e;
                    break;
                }
            }
            if (e_chosen < 0) return false;
        }

        y_out[e_chosen] = 1.0;
        int next = head[e_chosen];
        if (visited[next] == stamp && next != dest) return false;  // revisit -> not simple
        visited[next] = stamp;
        current = next;
    }

    return current == dest;
}

}  // namespace

void batch_sample_routes_impl(const int *tail, const int *head, int L, int V, const double *flows,
                              int M, const int *flow_row, int N, const int *origins,
                              const int *dests, const uint64_t *seeds, double threshold_factor,
                              double threshold_min, int max_steps, int n_threads, double *Y,
                              uint8_t *valid) {
    if (N <= 0) return;

    // Shared CSR adjacency (edges grouped by tail node), built once and read by
    // every worker: out_ptr[v]..out_ptr[v+1] indexes v's outgoing edges in
    // out_eidx. Counting sort, so `cursor` starts as a copy of out_ptr and is
    // advanced in place while scattering.
    std::vector<int> out_ptr(V + 1, 0);
    for (int e = 0; e < L; ++e) out_ptr[tail[e] + 1]++;
    for (int v = 0; v < V; ++v) out_ptr[v + 1] += out_ptr[v];
    std::vector<int> out_eidx(L);
    std::vector<int> cursor(out_ptr.begin(), out_ptr.end());
    for (int e = 0; e < L; ++e) out_eidx[cursor[tail[e]]++] = e;

    if (max_steps <= 0) max_steps = 2 * (V + L);
    if (n_threads <= 0) {
        int hw = (int)std::thread::hardware_concurrency();
        n_threads = hw > 1 ? hw : 1;
    }
    if (n_threads > N) n_threads = N;

    // Per-flow activity threshold: eps_m = max(threshold_min, factor * max_e clamp(flow_e, 0)),
    // matching active_threshold on np.maximum(x, 0). Precomputed once per flow
    // row, since many trips may share a row via flow_row.
    std::vector<double> eps_by_row(M);
    for (int m = 0; m < M; ++m) {
        const double *fr = flows + (int64_t)m * L;
        double max_x = 0.0;
        for (int e = 0; e < L; ++e) max_x = std::max(max_x, fr[e]);
        eps_by_row[m] = std::max(threshold_min, threshold_factor * max_x);
    }

    // `next_trip` is the only shared mutable state: each worker claims trip i,
    // then writes only row i of Y and entry i of valid, so the outputs need no
    // locking. `visited` and the RNG are per-thread, which is also what keeps a
    // trip's route a function of seeds[i] alone.
    std::atomic<int> next_trip{0};
    auto worker = [&]() {
        std::vector<int> visited(V, -1);
        int stamp = 0;
        while (true) {
            int i = next_trip.fetch_add(1, std::memory_order_relaxed);
            if (i >= N) break;

            int m = flow_row[i];
            const double *flow = flows + (int64_t)m * L;
            double eps = eps_by_row[m];
            std::mt19937_64 gen(seeds[i]);
            ++stamp;
            bool ok =
                sample_one_route(head, V, flow, out_ptr.data(), out_eidx.data(), origins[i],
                                 dests[i], eps, max_steps, gen, visited, stamp, Y + (int64_t)i * L);
            valid[i] = ok ? 1 : 0;
        }
    };

    if (n_threads <= 1) {
        worker();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) threads.emplace_back(worker);
        for (auto &th : threads) th.join();
    }
}
