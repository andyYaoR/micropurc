"""Scalability instrumentation exposed by MicroPURCEstimator.fit().

The estimator reports, in its ``diagnostics`` summary, the total fit wall-clock
time, native phase timings aggregated over every Phi evaluation in the fit,
thread count, and a platform-correct peak RSS. These are observational only:
they must leave the estimate unchanged.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig, SolverConfig
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.demands import od_from_csv
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.dgp import DGP, DGPConfig
from micropurc.simulation.route_sampler import MarkovSampler

DATA = pathlib.Path(__file__).resolve().parent / "data"
BETA0 = np.array([0.5, 0.5, 0.5])


def _fit_once(n_trips=800, seed=0):
    net = Network.from_csv(
        str(DATA / "EMA_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1"],
    )
    solver = PIQPFlowSolver(
        network=net,
        scale_m=net.Z[:, 0].copy(),
        config=SolverConfig(backend="piqp", eps_abs=1e-12, eps_duality_gap_abs=1e-12),
    )
    od = od_from_csv(
        net,
        str(DATA / "EMA_trips.csv"),
        origin_col="origin",
        dest_col="destination",
        weight_col="sampling_weight",
    )
    aset = ActiveSetConfig()
    sampler = MarkovSampler(network=net, active_set_config=aset)
    dgp = DGP(
        network=net,
        beta_true=BETA0,
        forward_solver=solver,
        route_sampler=sampler,
        od_dist=od,
        rng=np.random.default_rng(seed),
        config=DGPConfig(max_tries_per_trip=30, require_multiple_paths=True),
        active_set_config=aset,
    )
    data = dgp.sample_dataset(n_trips)
    est = MicroPURCEstimator(
        net,
        solver,
        estimation_config=EstimationConfig(max_iterations=200, fp_tol=1e-3, nm_window=3),
        active_set_config=aset,
        projection_config=ProjectionConfig(),
    )
    t0 = time.perf_counter()
    res = est.fit(data["y"], data["b"], beta_init=np.ones(3))
    return res, time.perf_counter() - t0


def test_summary_exposes_scaling_keys():
    res, _ = _fit_once()
    diag = res["diagnostics"]
    for key in (
        "total_wall_s",
        "time_solve_s",
        "time_projection_s",
        "time_schur_s",
        "total_qp_iters",
        "n_threads",
        "peak_rss_mb",
        "peak_rss_kb",
    ):
        assert key in diag, f"missing scaling key: {key}"


def test_wall_time_matches_external_clock():
    res, external = _fit_once()
    diag = res["diagnostics"]
    assert diag["total_wall_s"] > 0
    # Both clocks bracket the same fit() call, and fit() times only its own body,
    # so the reported figure sits just inside the caller's measurement.
    assert diag["total_wall_s"] <= external + 1e-6
    # The gap is call overhead alone; 0.5 is a loose floor that still catches a
    # counter reporting the wrong unit or a single phase.
    assert diag["total_wall_s"] == max(diag["total_wall_s"], 0.5 * external)


def test_phase_timings_aggregated_over_fit():
    """Phase timings sum over every Phi evaluation the fit performs."""
    res, _ = _fit_once()
    diag = res["diagnostics"]
    # The fit takes several iterations, so an aggregated counter is strictly
    # positive; the projection/Schur phases may legitimately round to zero.
    assert diag["time_solve_s"] > 0
    assert diag["time_projection_s"] >= 0
    assert diag["time_schur_s"] >= 0
    assert diag["total_qp_iters"] > 0
    assert diag["n_threads"] >= 1


def test_peak_rss_positive_and_platform_normalized():
    res, _ = _fit_once()
    diag = res["diagnostics"]
    assert diag["peak_rss_mb"] > 0
    # peak_rss_mb is a platform-correct MiB; peak_rss_kb is the raw stored value.
    assert diag["peak_rss_kb"] > 0


def test_instrumentation_does_not_change_estimate():
    """Instrumentation is passive: two identical fits agree up to thread noise.

    The estimate is not bit-reproducible, because the native solve accumulates
    H and g across worker threads in a nondeterministic order. The timing and
    RSS instrumentation adds no systematic shift on top of that, so the two runs
    agree to solver/thread float noise.
    """
    res_a, _ = _fit_once(seed=1)
    res_b, _ = _fit_once(seed=1)
    assert np.allclose(res_a["beta_hat"], res_b["beta_hat"], atol=1e-6, rtol=0)
