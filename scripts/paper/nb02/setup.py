"""Setup for nb02 (computational scalability).

Builds the grid/benchmark network cores, the timed-fit helper, and the scale
config shared by every nb02 section script. Each section script imports this
module; it has no entry point of its own.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig, SolverConfig
from micropurc.core.activeset import determine_active_set
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.demands import ODDistribution, od_from_csv
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.dgp import DGP, DGPConfig
from micropurc.simulation.route_sampler import MarkovSampler
from scripts.paper import common  # puts src/ on sys.path


def current_rss_mb() -> float:
    """Current resident set size in MiB, cross-platform, with no extra dependency.

    A per-fit memory delta needs the *current* RSS: ``ru_maxrss`` is a monotone
    process-wide high-water mark, so differencing it across fits in one process
    is wrong (a small fit after a large one reports 0). Reads /proc on Linux and
    the Win32 API on Windows; on macOS only the peak is available, so the delta
    there is a high-water difference.
    """
    if sys.platform == "linux":
        with open("/proc/self/statm", encoding="utf-8") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024**2
    if sys.platform == "win32":
        from micropurc.core.diagnostics import _windows_peak_rss_bytes

        return _windows_peak_rss_bytes() / 1024**2
    import resource  # macOS fallback: peak (bytes on darwin)

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

NB = "nb02"
FULL = common.full_scale()
DATA = common.DATA

# --- scale config (full run; --demo shrinks every axis) ----------------------
if FULL:
    R = 5  # timing reps (median + min-max band)
    N_LIST = [500, 1000, 2000, 5000, 10000, 20000, 50000]  # (1) sample-size axis
    GRID_K_LIST = [5, 8, 10, 15, 20, 30, 40, 50]  # (2) network-size axis: grid side k
    REAL_NETWORKS = ["SiouxFalls", "EMA", "Anaheim", "Barcelona", "Winnipeg", "ChicagoSketch"]
    K_SWEEP = [2, 3, 4, 5, 8]  # (3) parameter-count axis
    CORES = [1, 2, 4, 8, 16, 32, 64]  # (4) parallel-scaling axis
    GRID_N, GRID_KDIM = 5000, 3  # trips / #attributes held fixed in the size, K, core sweeps
    SWEEP_K_FIXED = 20  # grid side held fixed while sweeping N, K, cores
    N_REAL = 5000  # trips per benchmark network
    CORE_K, CORE_N = 30, 20000  # grid side and trips for the core-scaling sweep
else:
    R = 2
    N_LIST = [500, 1000, 2000, 5000]
    GRID_K_LIST = [5, 8, 10, 15]
    REAL_NETWORKS = ["SiouxFalls", "EMA", "Anaheim"]
    K_SWEEP = [2, 3, 5]
    CORES = [1, 2, 4]
    GRID_N, GRID_KDIM = 1000, 3
    SWEEP_K_FIXED = 10
    N_REAL = 1000
    CORE_K, CORE_N = 15, 3000

TAU, MAX_ITER = 1e-4, 200  # fp_tol coefficient; scaled by sqrt(K)
SEED_SIM = 0  # base RNG seed for the simulated datasets

est_cfg = EstimationConfig(
    max_iterations=MAX_ITER, fp_tol=TAU, scale_fp_tol_by_sqrt_k=True, nm_window=3
)


# --- builders and the timed fit ----------------------------------------------
def set_threads(c):
    """Pin the estimator's native hot loop (and, best-effort, BLAS) to c cores.

    ``OMP_NUM_THREADS`` drives the estimator's actual parallelism -- the native
    per-trip loop re-reads it on every call, so it is honored dynamically. The
    BLAS variables are set too, so that BLAS threads cannot deflate the ``c=1``
    baseline; that part is best-effort, since a BLAS library may fix its pool
    size at import, before this runs. BLAS is used only for the tiny K x K
    solves (K=3-20), so the baseline stays serial in practice either way.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(int(c))


def _make_solver(net):
    return PIQPFlowSolver(
        network=net,
        scale_m=net.Z[:, 0].copy(),
        config=SolverConfig(backend="piqp", eps_abs=1e-12, eps_duality_gap_abs=1e-12),
    )


def bounded_od(net, n_pairs, seed):
    """Uniform demand over a random set of OD pairs."""
    rng = np.random.default_rng(seed)
    V = net.num_nodes
    o = rng.integers(0, V, size=n_pairs)
    d = rng.integers(0, V, size=n_pairs)
    m = o != d
    return ODDistribution(origins=o[m], dests=d[m], weights=np.ones(int(m.sum())))


def build_grid_core(k, K, seed=0, n_pairs=None):
    """A k x k grid network (|L| = 4k(k-1)) with full-rank synthetic attributes."""
    net = Network.grid(k, k, K, rng_seed=seed + 1000)
    L = net.num_links
    rng = np.random.default_rng(seed)
    Z = np.zeros((L, K))
    Z[:, 0] = np.exp(0.3 * rng.standard_normal(L))  # log-normal 'length'
    for j in range(1, K):
        Z[:, j] = Z[:, 0] ** (0.5 + 0.5 * j) * np.exp(0.2 * rng.standard_normal(L))
    net.Z = Z
    beta_true = np.ones(K) * 0.5
    solver = _make_solver(net)
    active_filter = ActiveSetConfig()
    sampler = MarkovSampler(network=net, active_set_config=active_filter)
    od = bounded_od(net, n_pairs or max(4 * GRID_N, 4000), seed=seed + 7)
    dgp = DGP(
        network=net,
        beta_true=beta_true,
        forward_solver=solver,
        route_sampler=sampler,
        od_dist=od,
        rng=np.random.default_rng(seed),
        config=DGPConfig(max_tries_per_trip=30, require_multiple_paths=True),
        active_set_config=active_filter,
    )
    return dict(
        net=net, dgp=dgp, solver=solver, active_filter=active_filter, beta_true=beta_true, K=K
    )


def build_benchmark_core(name, seed=0):
    """A benchmark network with its observed OD pairs as demand."""
    net = Network.from_csv(
        str(DATA / f"{name}_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1"],
    )
    if name == "Anaheim":
        net.Z[:, 0] = net.Z[:, 0] / 5280.0  # feet -> miles for uniform scaling
    net.Z[:, 1] = np.maximum(net.Z[:, 1], 0.01)  # floor tiny travel times
    beta_true = np.array([0.5, 0.5, 0.5])
    solver = _make_solver(net)
    active_filter = ActiveSetConfig()
    sampler = MarkovSampler(network=net, active_set_config=active_filter)
    od = od_from_csv(
        net,
        str(DATA / f"{name}_trips.csv"),
        origin_col="origin",
        dest_col="destination",
        weight_col="sampling_weight",
    )
    dgp = DGP(
        network=net,
        beta_true=beta_true,
        forward_solver=solver,
        route_sampler=sampler,
        od_dist=od,
        rng=np.random.default_rng(seed),
        config=DGPConfig(max_tries_per_trip=30, require_multiple_paths=True),
        active_set_config=active_filter,
    )
    return dict(
        net=net, dgp=dgp, solver=solver, active_filter=active_filter, beta_true=beta_true, K=3
    )


def active_size_stats(core, data):
    """Per-trip active-set size distribution under the estimator's dual-slack rule.

    Counts links by the same native dual-slack (KKT sign) test the estimator
    applies, so the reported size is the one that drives the runtime. Each unique
    OD is re-solved serially to obtain the node potentials that sign test needs.
    Returning the mean together with the p50/p90/p99 quantiles shows the whole
    spread behind the "small, slowly growing fraction of links" regime.
    """
    net = core["net"]
    c = net.Z @ core["beta_true"]
    scale_m = net.Z[:, 0].copy()
    uniq = sorted(set(zip(data["origins"].tolist(), data["destinations"].tolist())))
    size_by_od = {}
    for o, d in uniq:
        res = core["solver"].solve(c, _od_b(net, o, d))
        aset = determine_active_set(
            res.x, int(d), net, core["active_filter"],
            lam_eq=res.lam_eq, c=c, scale_m=scale_m,
        )
        size_by_od[(o, d)] = int(aset.indices.size) if aset.valid else 0
    per_trip = np.array(
        [size_by_od[(int(o), int(d))] for o, d in zip(data["origins"], data["destinations"])],
        dtype=float,
    )
    q = np.percentile(per_trip, [50, 90, 99])
    return dict(mean=float(per_trip.mean()), p50=float(q[0]), p90=float(q[1]), p99=float(q[2]))


def _od_b(net, o, d):
    """Unit-demand right-hand side for one OD pair: -1 at the origin, +1 at the dest."""
    b = np.zeros(net.num_nodes)
    b[int(o)] = -1.0
    b[int(d)] = 1.0
    return b


def time_fit(core, N, rep_seed, want_active=False):
    """Sample N trips, fit, and return runtime / memory / convergence diagnostics."""
    core["dgp"].rng = np.random.default_rng(rep_seed)
    data = core["dgp"].sample_dataset(N)
    n_unique = len(set(zip(data["origins"].tolist(), data["destinations"].tolist())))
    est = MicroPURCEstimator(
        network=core["net"],
        forward_solver=core["solver"],
        estimation_config=est_cfg,
        active_set_config=core["active_filter"],
        projection_config=ProjectionConfig(mode="robust_pinv"),
    )
    rss_before = current_rss_mb()  # per-fit delta; see current_rss_mb for why it is current
    res = est.fit(data["y"], data["b"], beta_init=np.ones(core["K"]))
    rss_after = current_rss_mb()
    d = res["diagnostics"]
    out = dict(
        V=core["net"].num_nodes,
        L=core["net"].num_links,
        N=N,
        K=core["K"],
        n_unique_od=int(n_unique),
        converged=bool(d["converged"]),
        n_iter=int(d["num_iterations"]),
        time_s=float(d["total_wall_s"]),
        mem_mb=float(max(rss_after - rss_before, 0.0)),
        n_threads=int(d["n_threads"]),
    )
    stats = active_size_stats(core, data) if want_active else {}
    out["active"] = stats.get("mean", np.nan)
    out["active_p90"] = stats.get("p90", np.nan)
    out["active_p99"] = stats.get("p99", np.nan)
    return out


def band(records, key):
    """(median, min, max) of a metric across timing reps."""
    v = np.array([r[key] for r in records], float)
    return float(np.median(v)), float(v.min()), float(v.max())


print(
    f"[nb02 setup] FULL={FULL}  R={R}  grid links per side: "
    + ", ".join(f"k={k}->L={4 * k * (k - 1)}" for k in GRID_K_LIST),
    flush=True,
)
