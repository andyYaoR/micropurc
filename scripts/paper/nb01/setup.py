"""Setup for nb01 (EMA simulation experiments).

Builds the network, DGP, estimators, and scale config shared by every nb01
section script. Each section script imports this module; it has no entry point
of its own.
"""

from __future__ import annotations

import time

import numpy as np

from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig, SolverConfig
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.demands import od_from_csv
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.dgp import DGP, DGPConfig
from micropurc.simulation.route_sampler import MarkovSampler
from scripts.paper import common  # puts src/ on sys.path

NB = "nb01"
FULL = common.full_scale()
DATA = common.DATA

BETA0 = np.array([0.5, 0.5, 0.5])
ATTR = ["length", "time", "sim"]

# --- scale config (full run; --demo shrinks every axis) ----------------------
if FULL:
    R = 500
    N_LIST = [200, 500, 1000, 2000, 5000, 10000, 20000]
    INIT_N, INIT_DATASETS, INIT_M = 10000, 10, 1000
    MARGIN_N, MARGIN_R = 500, 500
    SWITCH_N_LIST, SWITCH_R, SWITCH_MAXIT = [200, 1000, 10000], 500, 40
    NONCONV_N, NONCONV_R = 200, 500
else:
    R = 50
    N_LIST = [200, 500, 1000, 2000]
    INIT_N, INIT_DATASETS, INIT_M = 2000, 2, 50
    MARGIN_N, MARGIN_R = 500, 20
    SWITCH_N_LIST, SWITCH_R, SWITCH_MAXIT = [200, 1000], 20, 40
    NONCONV_N, NONCONV_R = 200, 50
RADII = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
TAU, MAX_ITER = 1e-4, 200  # fp_tol coefficient; scaled by sqrt(K)
SEED_ATTR, SEED_SIM, SEED_INIT = 0, 0, 1000
MARGIN_GRID = np.logspace(-8, 0, 81)

# --- network + DGP + estimators ----------------------------------------------
net = Network.from_csv(
    str(DATA / "EMA_net.csv"),
    tail_col="init_node",
    head_col="term_node",
    attribute_cols=["length", "tt", "attr_1"],
)
scale_m = net.Z[:, 0].copy()
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
switch_cfg = EstimationConfig(
    max_iterations=SWITCH_MAXIT, fp_tol=TAU, scale_fp_tol_by_sqrt_k=True, nm_window=3
)


def dataset_seed(n_trips: int, rep: int) -> int:
    """Independent per-(N, rep) dataset seed (A3).

    Deriving the seed from both N and the replication index via SeedSequence
    makes each (N, rep) an independent Monte-Carlo draw -- datasets are no longer
    nested across N (which correlated the sqrt(N)-RMSE points) and no longer
    coincide across sections that share a sample size.
    """
    return int(np.random.SeedSequence([SEED_SIM, int(n_trips), int(rep)]).generate_state(1)[0])


def make_dataset(seed, n_trips):
    """Simulate n_trips at BETA0, accepting only ODs whose active subgraph branches."""
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


def fit_one(y, b, beta_init):
    """One microPURC fit: convergence flag, estimate, SE, iteration count, runtime."""
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
        diag = res["diagnostics"]
        return dict(
            converged=bool(diag.get("converged", False)),
            beta_hat=np.asarray(res["beta_hat"], float),
            se_hat=np.asarray(res.get("se_hat"), float),
            n_iter=len(diag.get("Q_history", [])),
            time_s=time.perf_counter() - t0,
        )
    except Exception:  # noqa: BLE001 -- record a degenerate fit (e.g. singular H_N)
        # as a non-convergence with a non-finite estimate, so one bad draw cannot
        # abort a sweep of hundreds of replications.
        return dict(
            converged=False,
            beta_hat=np.full(3, np.nan),
            se_hat=np.full(3, np.nan),
            n_iter=0,
            time_s=time.perf_counter() - t0,
        )


def sample_in_ball(rng, center, r, dim=3):
    """Draw a start inside the radius-r ball around ``center``.

    The direction is uniform on the sphere and the radius is uniform on [0, r],
    so every starting distance up to r carries equal weight. A row of the
    init-ball table at radius r therefore summarises starts spread over the
    whole interval [0, r], and its "Starting distance" column reports the
    realised mean.
    """
    g = rng.normal(size=dim)
    g /= np.linalg.norm(g)  # uniform direction
    return center + r * rng.random() * g  # radius uniform on [0, r]


print(
    f"[nb01 setup] FULL={FULL}  net |L|={net.num_links} K={net.num_attributes} "
    f"{net.attribute_names}",
    flush=True,
)
