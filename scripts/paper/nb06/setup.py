"""Setup for nb06 (robustness to repeated trips / violation of i.i.d.).

Motivation: in revealed-preference route data the same traveler contributes many
trips, which are not independent -- violating the i.i.d. assumption behind the
asymptotics. A simulation isolates the effect: it fixes the dependence structure
by construction, so the coverage loss can be attributed to repetition alone and
compared against a closed form -- which no single empirical dataset can do.

Scope note: whenever a dataset carries a traveler identifier, trips can be
grouped and a cluster-robust standard error is both computable and correct; this
simulation validates that correction against a known truth.

Mechanism (mean-preserving by construction): tastes are homogeneous -- every trip
uses the same beta_0, so the choice model itself is untouched. Each traveler is a
commuter with one fixed OD who takes the same habitual route on every trip
(rho = 1, the conservative worst case). Each trip's marginal law is still exactly
the beta_0 route law, so the estimator stays consistent for beta_0; what changes
is the within-traveler dependence, and through it the standard errors. At rho = 1
a traveler's T trips are identical, so the traveler contributes one independent
observation and the naive i.i.d. standard error understates the truth by exactly
sqrt(T) -- a closed form that a cluster-robust standard error corrects.

Design: fix N trips, sweep T = trips per traveler (so H = N/T travelers). T = 1 is
the i.i.d. anchor. Report, per coefficient: bias, Monte-Carlo SD, and 95% coverage
under the naive and the traveler(user)-clustered standard errors. This quantifies
both that the point estimates stay safe under repetition and how much coverage is
lost when clustering is omitted -- the loss that clustering by traveler recovers.

Each section script imports this module; it has no entry point of its own.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats as sstats

from micropurc.config import ProjectionConfig
from micropurc.core.covariance import cluster_robust_covariance
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.simulation.dgp import DGP, DGPConfig
from scripts.paper import common

# Reuse the exact EMA network / solver / sampler / OD distribution / configs the
# other simulation sections use, so nothing about the setup differs from nb01/nb03.
from scripts.paper.nb03.setup import (  # noqa: E402
    BETA0,
    active_filter,
    est_cfg,
    net,
    od,
    sampler,
    solver,
)

NB = "nb06"
FULL = common.full_scale()

ATTR = ["length", "time", "sim"]
K = len(ATTR)

# --- scale config (full run; --demo shrinks every axis) ----------------------
RHO = 1.0  # habit strength: 1 = identical repeated trips (conservative worst case)
if FULL:
    N = 10_000  # total trips, held fixed across T
    T_LIST = [1, 2, 5, 10, 20]  # trips per traveler; H = N/T = 10000..500 travelers
    R = 500  # replications per T
else:
    N = 2_000
    T_LIST = [1, 5, 20]  # H = 2000, 400, 100
    R = 20

SEED_BASE = 6060


def seed_for(T: int, rep: int) -> int:
    """Independent per-(T, rep) dataset seed."""
    return int(np.random.SeedSequence([SEED_BASE, int(T), int(rep)]).generate_state(1)[0])


def make_clustered(seed: int, n_trips: int, T: int, rho: float):
    """One clustered dataset: H = n_trips/T commuters, T habitual trips each."""
    if n_trips % T != 0:
        raise ValueError(f"N={n_trips} must be divisible by T={T}.")
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
    return dgp.sample_clustered_dataset(n_trips // T, T, rho)


def fit_clustered(d, beta_init=None):
    """Fit homogeneous microPURC; return beta_hat, naive SE, and the sandwich parts."""
    est = MicroPURCEstimator(
        network=net,
        forward_solver=solver,
        estimation_config=est_cfg,
        active_set_config=active_filter,
        projection_config=ProjectionConfig(),
    )
    beta_init = np.ones(K) if beta_init is None else beta_init
    t0 = time.perf_counter()
    try:
        res = est.fit(y=d["y"], b=d["b"], beta_init=beta_init)
    except Exception:  # noqa: BLE001 -- a degenerate dataset returns nan and the sweep goes on
        return dict(
            beta_hat=np.full(K, np.nan), se_naive=np.full(K, np.nan),
            u=None, H=None, converged=False, time_s=time.perf_counter() - t0,
        )
    se = res["se_hat"]
    return dict(
        beta_hat=np.asarray(res["beta_hat"], float),
        se_naive=np.asarray(se, float) if se is not None else np.full(K, np.nan),
        u=res["u_scores"],
        H=res["hessian"],
        converged=bool(res["diagnostics"]["converged"]),
        time_s=float(res["diagnostics"]["total_wall_s"]),
    )


def cluster_se(H, u, cluster_ids):
    """CR1 cluster-robust SE and the t critical value for its dof.

    Returns (se (K,), G, tcrit) where G is the number of clusters and tcrit is the
    0.975 quantile of t_{G-1}, which the Wald interval uses as its critical value.
    Non-finite on failure. cluster_robust_covariance already applies the CR1
    scaling G/(G-1).
    """
    nan = np.full(K, np.nan)
    if H is None or u is None:
        return nan, 0, np.nan
    G = int(np.unique(cluster_ids).size)
    if G <= 1:
        return nan, G, np.nan
    try:
        _, se0 = cluster_robust_covariance(H, u, cluster_ids)
    except np.linalg.LinAlgError:
        return nan, G, np.nan
    se = np.asarray(se0, float)
    tcrit = float(sstats.t.ppf(0.975, df=G - 1))
    return se, G, tcrit


print(
    f"[nb06 setup] FULL={FULL}  N={N} T_LIST={T_LIST} R={R} rho={RHO}  "
    f"net |L|={net.num_links} |V|={net.num_nodes} K={K}",
    flush=True,
)
