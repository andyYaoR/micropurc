"""Setup for nb04 (algorithm robustness to tuning parameters).

Builds the network, DGP primitives, estimators, and scale config shared by
every nb04 section script. Each section script imports this module; it has no
entry point of its own. Import stays cheap — datasets and the baseline fit are
built on demand by build_datasets/build_baseline — so the isolated section
runner pays for them only in the section that asks.
"""

from __future__ import annotations

import numpy as np

from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig, SolverConfig
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.demands import od_from_csv
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.dgp import DGP, DGPConfig
from micropurc.simulation.route_sampler import MarkovSampler
from scripts.paper import common  # puts src/ on sys.path

NB = "nb04"
FULL = common.full_scale()
DATA = common.DATA

BETA0 = np.array([0.5, 0.5, 0.5])

# --- scale config (full run; --demo shrinks every axis) ----------------------
if FULL:
    D, SWEEP_N = 20, 10000  # datasets; trips per dataset for the main sweeps
    HARD_N, HARD_M, HARD_R = 2000, 200, 5.0  # hard-start: sample; starts/window; log-radius
else:
    D, SWEEP_N = 5, 2000
    HARD_N, HARD_M, HARD_R = 2000, 40, 5.0

SEED_SIM, SEED_INIT = 0, 1000

# --- fixed-point iteration defaults used by every simulation section ---------
# The line search is purely nonmonotone: trial steps alpha = ls_tau^m for
# m = 0..ls_max_tries, accepted as soon as Q_N(beta_(t+1)) is at most the largest
# merit over the last nm_window iterates. No sufficient-decrease (Armijo) term.
DEF = dict(
    fp_tol=1e-4,
    scale_fp_tol_by_sqrt_k=True,
    max_iterations=200,
    nm_window=3,
    ls_tau=0.5,
    ls_max_tries=5,
)
DEF_EPS = 1e-12  # inner QP absolute tolerance

# --- one-at-a-time sweeps (default value is included in each list) ------------
# These five knobs are the fixed-point iteration's own tuning parameters: the
# convergence tolerance, the iteration cap, the nonmonotone window, the
# backtracking factor, and the line-search budget. The inner QP tolerance is a
# solver setting and is swept separately in s2_qp_tol.
SWEEPS = {
    "fp_tol": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5],
    "max_iterations": [50, 100, 200, 500],
    "nm_window": [1, 3, 5, 10],
    "ls_tau": [0.3, 0.5, 0.7],
    "ls_max_tries": [3, 5, 10],
}
# A4: the inner-QP-tolerance sweep reaches a loose end (1e-4, 1e-6) where a
# near-degenerate dual slack could actually flip the sign-test active set -- that
# is the only regime that stresses the invariance claim. make_solver also zeroes
# PIQP's relative criteria so eps_abs genuinely binds (PIQP OR-terminates, so at
# eps_rel~1e-9 the swept eps_abs would stay largely inert).
EPS_SWEEP = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]

# --- network + DGP primitives + estimators -----------------------------------
net = Network.from_csv(
    str(DATA / "EMA_net.csv"),
    tail_col="init_node",
    head_col="term_node",
    attribute_cols=["length", "tt", "attr_1"],
)
scale_m = net.Z[:, 0].copy()
active_filter = ActiveSetConfig()
sampler = MarkovSampler(network=net, active_set_config=active_filter)
od = od_from_csv(
    net,
    str(DATA / "EMA_trips.csv"),
    origin_col="origin",
    dest_col="destination",
    weight_col="sampling_weight",
)


def make_solver(eps_abs=DEF_EPS):
    # A4: zero the relative criteria so the absolute tolerance eps_abs is what
    # actually terminates the IPM (PIQP OR-terminates, so leaving eps_rel=1e-9
    # would make the swept eps_abs inert and the invariance test near-vacuous).
    return PIQPFlowSolver(
        network=net,
        scale_m=scale_m,
        config=SolverConfig(
            backend="piqp",
            eps_abs=eps_abs,
            eps_duality_gap_abs=eps_abs,
            eps_rel=0.0,
            eps_duality_gap_rel=0.0,
        ),
    )


base_solver = make_solver()


def make_dataset(seed, n_trips):
    dgp = DGP(
        network=net,
        beta_true=BETA0,
        forward_solver=base_solver,
        route_sampler=sampler,
        od_dist=od,
        rng=np.random.default_rng(seed),
        config=DGPConfig(max_tries_per_trip=30, require_multiple_paths=True),
        active_set_config=active_filter,
    )
    return dgp.sample_dataset(n_trips=n_trips)


def fit_cfg(data, params, solver, beta_init=None):
    """Fit with the given tuning parameters; return estimate and diagnostics."""
    cfg = EstimationConfig(**params)
    est = MicroPURCEstimator(
        network=net,
        forward_solver=solver,
        estimation_config=cfg,
        active_set_config=active_filter,
        projection_config=ProjectionConfig(mode="robust_pinv"),
    )
    b0 = np.ones(3) if beta_init is None else beta_init
    res = est.fit(y=data["y"], b=data["b"], beta_init=b0)
    d = res["diagnostics"]
    return dict(
        beta=np.asarray(res["beta_hat"], float),
        converged=bool(d["converged"]),
        n_iter=int(d["num_iterations"]),
        time_s=float(d["total_wall_s"]),
    )


def sample_in_ball(rng, center, r, dim=3):
    """Draw a start inside the radius-r ball around ``center``.

    Direction uniform on the sphere, radius uniform on [0, r]. Kept identical to
    nb01's and nb03's samplers so the hard-start results are comparable with the
    init-ball ones.
    """
    g = rng.normal(size=dim)
    g /= np.linalg.norm(g)  # uniform direction
    return center + r * rng.random() * g  # radius uniform on [0, r]


def build_datasets():
    """Build the D sweep datasets shared by the sections.

    Heavy: D full DGP runs at SWEEP_N trips each. Call it from inside a section's
    main(), so importing this module stays cheap.
    """
    return [make_dataset(SEED_SIM + d, SWEEP_N) for d in range(D)]


def build_baseline(datasets):
    """Reference estimate per dataset: default tuning, common start beta=1."""
    baseline = [fit_cfg(data, DEF, base_solver) for data in datasets]
    base_beta = [b["beta"] for b in baseline]
    return baseline, base_beta


print(
    f"[nb04 setup] FULL={FULL}  net |L|={net.num_links} |V|={net.num_nodes} "
    f"K={net.num_attributes} {net.attribute_names}",
    flush=True,
)
