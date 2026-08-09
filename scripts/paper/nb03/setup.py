"""Setup for nb03 (comparison with alternative estimators).

Builds the network, DGP, scale config, and the four estimators shared by every
nb03 section script. Each section script imports this module; it has no entry
point of its own.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp

from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig, SolverConfig
from micropurc.core.covariance import sandwich_covariance
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.core.projection import trip_contribution_schur  # package Schur projection (native)
from micropurc.demands import od_from_csv
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.dgp import DGP, DGPConfig
from micropurc.simulation.route_sampler import MarkovSampler
from scripts.paper import common  # puts src/ on sys.path

NB = "nb03"
FULL = common.full_scale()
DATA = common.DATA

BETA0 = np.array([0.5, 0.5, 0.5])
ATTR = ["length", "time", "sim"]

# --- scale config (full run; --demo shrinks every axis) ----------------------
if FULL:
    R = 100  # replications per sample size
    N_LIST = [200, 500, 1000, 2000, 5000, 10000, 20000]  # finite-sample sample sizes
    # init-ball: independent datasets, trips per dataset, random starts per (dataset, radius)
    INIT_DATASETS, INIT_N, INIT_M = 5, 10000, 100
else:
    R = 3
    N_LIST = [500, 1000, 2000]
    INIT_DATASETS, INIT_N, INIT_M = 2, 2000, 5

RADII = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]  # init-ball radii, in log-parameter space
TAU, MAX_ITER = 1e-4, 200  # fp_tol coefficient; scaled by sqrt(K)
MLE_MAXITER = 500  # Nelder-Mead budget for the MLE / direct-RV searches
SEED_SIM, SEED_INIT = 0, 1000  # seeds: datasets, initialization draws

ESTIMATORS = ["microPURC", "PURC-OLS", "MLE", "direct-RV"]

# --- network + DGP -----------------------------------------------------------
net = Network.from_csv(
    str(DATA / "EMA_net.csv"),
    tail_col="init_node",
    head_col="term_node",
    attribute_cols=["length", "tt", "attr_1"],
)
scale_m = net.Z[:, 0].copy()
Z = net.Z
L, K = Z.shape
V = net.num_nodes
tail = net.tail.astype(int)
head = net.head.astype(int)
A = net.A.tocsc()
# S_tail[e, tail[e]] = 1  ->  flows @ S_tail gives per-node outflow for each OD flow row
S_tail = sp.csr_matrix((np.ones(L), (np.arange(L), tail)), shape=(L, V))

od = od_from_csv(
    net,
    str(DATA / "EMA_trips.csv"),
    origin_col="origin",
    dest_col="destination",
    weight_col="sampling_weight",
)
solver = PIQPFlowSolver(
    network=net,
    scale_m=scale_m,
    config=SolverConfig(backend="piqp", eps_abs=1e-12, eps_duality_gap_abs=1e-12),
)
active_filter = ActiveSetConfig()
sampler = MarkovSampler(network=net, active_set_config=active_filter)
est_cfg = EstimationConfig(
    max_iterations=MAX_ITER, fp_tol=TAU, scale_fp_tol_by_sqrt_k=True, nm_window=3
)


def dataset_seed(n_trips: int, rep: int) -> int:
    """Independent per-(N, rep) dataset seed (A3): datasets are not nested across N."""
    return int(np.random.SeedSequence([SEED_SIM, int(n_trips), int(rep)]).generate_state(1)[0])


def make_dataset(seed, n_trips):
    dgp = DGP(
        network=net,
        beta_true=BETA0,
        forward_solver=solver,
        route_sampler=sampler,
        od_dist=od,
        rng=np.random.default_rng(seed),
        config=DGPConfig(max_tries_per_trip=30, require_multiple_paths=True),
        active_set_config=active_filter,
    )
    return dgp.sample_dataset(n_trips=n_trips)


def group_by_od(origins, dests):
    """Map trips to unique OD ids; return the unique origins/dests for solve_batch."""
    ods = list(zip(origins.tolist(), dests.tolist()))
    uniq = sorted(set(ods))
    idx = {od_: i for i, od_ in enumerate(uniq)}
    trip_od = np.array([idx[o] for o in ods], dtype=int)
    uo = np.array([o for o, d in uniq], dtype=np.int64)
    ud = np.array([d for o, d in uniq], dtype=np.int64)
    return uniq, trip_od, uo, ud


def order_route_edges(obs_links, origin, dest):
    """Order an observed link set into an origin->dest traversal (for the path likelihood)."""
    remaining = set(int(e) for e in obs_links)
    ordered, node = [], int(origin)
    while remaining:
        nxt = next((e for e in remaining if int(tail[e]) == node), None)
        if nxt is None:
            break
        ordered.append(nxt)
        remaining.discard(nxt)
        node = int(head[nxt])
        if node == int(dest):
            break
    return ordered


# --- the four estimators -----------------------------------------------------
def fit_micropurc(y, b, beta_init):
    est = MicroPURCEstimator(
        network=net,
        forward_solver=solver,
        estimation_config=est_cfg,
        active_set_config=active_filter,
        projection_config=ProjectionConfig(),
    )
    t0 = time.perf_counter()
    try:
        res = est.fit(y=y, b=b, beta_init=beta_init)
    except Exception:  # noqa: BLE001  -- degenerate dataset (e.g. H_N all zeros at small N);
        # return a non-finite estimate, which summarize() counts as a non-recovery and
        # which keeps the section running. fit_purc_ols and nb01's fit_one do the same,
        # so one bad draw costs the same in every estimator's column.
        return dict(
            beta_hat=np.full(3, np.nan), converged=False, time_s=time.perf_counter() - t0
        )
    diag = res["diagnostics"]
    return dict(
        beta_hat=np.asarray(res["beta_hat"], float),
        converged=bool(diag["converged"]),
        time_s=float(diag["total_wall_s"]),
    )


def fit_purc_ols(y, origins, dests):
    """PURC-OLS (CCP-type): OLS on the OD-aggregated, projected first-order condition."""
    t0 = time.perf_counter()
    uniq, trip_od, uo, ud = group_by_od(origins, dests)
    H = np.zeros((K, K))
    g = np.zeros(K)
    scores = []
    for i, (o, d) in enumerate(uniq):
        mask = trip_od == i
        ybar = y[mask].mean(axis=0)  # empirical OD flow
        active = np.where(ybar > 0)[0]
        if active.size <= 1:
            continue
        A_sub = A[:, active]
        nz = np.unique(A_sub.nonzero()[0])
        rows = nz[nz != d]  # ground: drop destination row
        if rows.size == 0:
            continue
        A_local = A_sub[rows].toarray()
        tc = trip_contribution_schur(Z[active], A_local, ybar[active], scale_m[active])
        H_i = np.asarray(tc.H_i)
        g_i = np.asarray(tc.g_i)
        n = int(mask.sum())
        H += n * H_i
        g += n * g_i
        scores.append((H_i, g_i, n))
    beta_hat = -np.linalg.lstsq(H, g, rcond=None)[0]
    u_vecs = np.array([n * (Hi @ beta_hat + gi) for (Hi, gi, n) in scores])
    try:
        _, se = sandwich_covariance(H, u_vecs)
    except Exception:
        se = np.full(K, np.nan)
    return dict(beta_hat=beta_hat, se_hat=np.asarray(se, float), time_s=time.perf_counter() - t0)


def _flatten_routes(y, trip_od, origins, dests):
    """Vectorize every observed route's ordered edges with its OD id."""
    flat_e, flat_od = [], []
    for n in range(y.shape[0]):
        ordered = order_route_edges(np.where(y[n] > 0)[0], int(origins[n]), int(dests[n]))
        flat_e.extend(ordered)
        flat_od.extend([trip_od[n]] * len(ordered))
    return np.array(flat_e, dtype=int), np.array(flat_od, dtype=int)


def _neg_loglik(beta, uo, ud, flat_e, flat_od):
    """Negative Markov path log-likelihood, paired with a zero-probability link count.

    A route link with zero predicted flow has transition probability 0, so the path
    log-likelihood is genuinely -inf there. The negative log-likelihood returned in
    that case is +inf, and the accompanying count says how many links caused it, so
    the degeneracy stays visible to the caller. Flooring log(0) at some finite value
    would hide it and hand the optimizer a fabricated objective.
    """
    flows, _ = solver.solve_batch(Z @ beta, uo, ud)  # (U, L) predicted flows per OD
    node_out = flows @ S_tail  # (U, V) outflow per node per OD
    xe = flows[flat_od, flat_e]
    den = node_out[flat_od, tail[flat_e]]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(den > 0, xe / den, 0.0)
    n_zero = int(np.sum(p <= 0))
    if n_zero > 0:
        return np.inf, n_zero  # objective undefined (observed route uses a zero-flow link)
    return -float(np.sum(np.log(p))), n_zero


def fit_mle(y, origins, dests, beta_init, maxiter=None):
    """Maximize the Markov path log-likelihood with Nelder-Mead.

    The log-likelihood is -inf whenever an observed route uses a zero-predicted-flow
    link, so the minimized objective (its negative) is +inf there. When that already
    holds at the start value, no finite optimum is reachable from it and the fit is
    reported as a failure with beta_hat = nan; ``n_zero_init`` carries the offending
    link count so the failure can be attributed. Flooring the objective would keep
    the optimizer running on a value the model never assigns.
    """
    maxiter = maxiter or MLE_MAXITER
    uniq, trip_od, uo, ud = group_by_od(origins, dests)
    flat_e, flat_od = _flatten_routes(y, trip_od, origins, dests)
    _, nz_init = _neg_loglik(beta_init, uo, ud, flat_e, flat_od)
    t0 = time.perf_counter()
    if nz_init > 0:  # degenerate from this start: objective is -inf, no finite optimum
        return dict(
            beta_hat=np.full(3, np.nan),
            converged=False,
            time_s=time.perf_counter() - t0,
            nfev=0,
            n_zero_init=int(nz_init),
            degenerate=True,
        )
    state = {"nz_max": 0}

    def obj(beta):
        nll, nz = _neg_loglik(beta, uo, ud, flat_e, flat_od)
        state["nz_max"] = max(state["nz_max"], nz)  # track if the search wanders into log(0)
        return nll

    try:
        r = opt.minimize(
            obj,
            beta_init,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
        )
    except Exception:  # noqa: BLE001  -- objective solve may fail; report a non-recovery.
        return dict(
            beta_hat=np.full(3, np.nan),
            converged=False,
            time_s=time.perf_counter() - t0,
            nfev=0,
            n_zero_init=int(nz_init),
            n_zero_max=int(state["nz_max"]),
            degenerate=False,
        )
    return dict(
        beta_hat=np.asarray(r.x, float),
        converged=bool(r.success),
        time_s=time.perf_counter() - t0,
        nfev=int(r.nfev),
        n_zero_init=int(nz_init),
        n_zero_max=int(state["nz_max"]),
        degenerate=False,
    )


def fit_rv(y, origins, dests, beta_init, maxiter=None):
    """Minimize the residual variance RV_N(beta) directly with Nelder-Mead.

    RV_N(beta) is the mean over trips of ||y_n - x_n(beta)||^2, where x_n(beta) is
    the forward-solved flow for trip n's OD pair.
    """
    maxiter = maxiter or MLE_MAXITER
    uniq, trip_od, uo, ud = group_by_od(origins, dests)

    def obj(beta):
        flows, _ = solver.solve_batch(Z @ beta, uo, ud)
        resid = y - flows[trip_od]
        return float(np.mean(np.sum(resid * resid, axis=1)))

    t0 = time.perf_counter()
    try:
        r = opt.minimize(
            obj,
            beta_init,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
        )
    except Exception:  # noqa: BLE001  -- solve_batch may fail on a degenerate dataset; report
        # a non-finite estimate, which summarize() counts as a non-recovery, and let the
        # section continue with the remaining replications.
        return dict(beta_hat=np.full(3, np.nan), converged=False, time_s=time.perf_counter() - t0)
    return dict(
        beta_hat=np.asarray(r.x, float),
        converged=bool(r.success),
        time_s=time.perf_counter() - t0,
        nfev=int(r.nfev),
    )


def sample_in_ball(rng, center, r, dim=3):
    """Draw a start inside the radius-r ball around ``center``.

    Direction uniform on the sphere, radius uniform on [0, r], so every starting
    distance up to r carries equal weight. The init-robustness table's row for
    radius r therefore summarises starts spread over the whole interval [0, r].
    Kept identical to nb01's sampler so the two init-ball results are comparable.
    """
    g = rng.normal(size=dim)
    g /= np.linalg.norm(g)  # uniform direction
    return center + r * rng.random() * g  # radius uniform on [0, r]


print(
    f"[nb03 setup] FULL={FULL}  net |L|={net.num_links} |V|={net.num_nodes} "
    f"K={net.num_attributes} {net.attribute_names}",
    flush=True,
)
