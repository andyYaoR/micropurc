"""Tests for the native C++ kernel.

Verifies that the native extension is functional: Phi returns a valid
K-vector, H is symmetric, and the thread pool is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"

try:
    from micropurc._native import batch_solve_and_accumulate, get_num_threads  # noqa: F401

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="Native extension not built")


def _build_ema_test_data(N=50, seed=42):
    """Build EMA network and sample N trips for testing."""
    from micropurc.config import SolverConfig
    from micropurc.demands import od_from_csv
    from micropurc.forward.piqp_solver import PIQPFlowSolver
    from micropurc.network import Network
    from micropurc.simulation.dgp import DGP
    from micropurc.simulation.route_sampler import MarkovSampler

    data_dir = DATA_DIR
    if not (data_dir / "EMA_net.csv").exists():
        pytest.skip("EMA data files not found")

    net = Network.from_csv(
        str(data_dir / "EMA_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1", "attr_2"],
    )
    scale_m = net.Z[:, 0].copy()
    beta_true = np.array([0.5, 0.5, 0.5, 0.5])
    c = net.Z @ beta_true

    solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
    od_dist = od_from_csv(net, str(data_dir / "EMA_trips.csv"))
    sampler = MarkovSampler(network=net)
    dgp = DGP(net, beta_true, solver, sampler, od_dist, rng=np.random.default_rng(seed))

    data = dgp.sample_dataset(N)
    y, b = data["y"], data["b"]

    x_all = np.empty((N, net.num_links), dtype=np.float64)
    dests = np.empty(N, dtype=np.intc)
    for n in range(N):
        result = solver.solve(c, b[n])
        x_all[n] = result.x
        dests[n] = int(np.argmax(b[n]))

    return net, solver, scale_m, beta_true, y, b, x_all, dests


class TestNativeKernel:
    """Basic sanity checks for the native C++ extension."""

    @pytest.fixture(scope="class")
    def ema_data(self):
        return _build_ema_test_data(N=50, seed=42)

    def test_phi_matches(self, ema_data):
        """est.phi(beta, y, b) returns a valid K-vector."""
        from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig
        from micropurc.core.estimator import MicroPURCEstimator

        net, solver, scale_m, beta, y, b, _, _ = ema_data
        K = net.num_attributes

        est = MicroPURCEstimator(
            net,
            solver,
            estimation_config=EstimationConfig(),
            active_set_config=ActiveSetConfig(),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )
        phi = est.phi(beta, y, b)

        assert phi.shape == (K,), f"Phi shape {phi.shape}, expected ({K},)"
        assert np.all(np.isfinite(phi)), f"Phi contains NaN/Inf: {phi}"

    def test_H_symmetric(self, ema_data):
        """H from _accumulate should be symmetric."""
        from micropurc.config import ActiveSetConfig, EstimationConfig, ProjectionConfig
        from micropurc.core.estimator import MicroPURCEstimator

        net, solver, scale_m, beta, y, b, _, _ = ema_data

        est = MicroPURCEstimator(
            net,
            solver,
            estimation_config=EstimationConfig(),
            active_set_config=ActiveSetConfig(),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )
        H, g, _, dropped = est._accumulate(beta, y, b)

        np.testing.assert_allclose(H, H.T, atol=1e-12, err_msg="H is not symmetric")

    def test_native_module_info(self):
        """Native module reports at least 1 thread."""
        threads = get_num_threads()
        assert threads >= 1
