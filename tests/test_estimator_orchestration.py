"""Estimator orchestration tests for native settings, OD inputs, and KM policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from micropurc.config import (
    ActiveSetConfig,
    EstimationConfig,
    ProjectionConfig,
    SolverConfig,
)
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.core.lazy_design import LazyDesign
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network

DATA_DIR = Path(__file__).resolve().parent / "data"


class _DummySolver:
    def __init__(self, scale_m: np.ndarray) -> None:
        self._scale_m = scale_m
        self.config = SolverConfig()

    @property
    def scale_m(self) -> np.ndarray:
        return self._scale_m


class _LineSearchEstimator(MicroPURCEstimator):
    def __init__(self, cfg: EstimationConfig, trial_residual: float) -> None:
        net = Network.grid(1, 2, K=1)
        super().__init__(
            net,
            _DummySolver(net.Z[:, 0].copy()),
            estimation_config=cfg,
            active_set_config=ActiveSetConfig(),
            projection_config=ProjectionConfig(),
        )
        self.trial_residual = trial_residual
        self.phi_calls = 0

    def _phi_full(self, beta, y, b, **kwargs):
        self.phi_calls += 1
        if self.phi_calls == 1:
            return np.array([1.0]), np.eye(1), 0
        return np.asarray(beta, dtype=float) + self.trial_residual, np.eye(1), 0

    def _accumulate(self, beta, y, b, return_u=False, **kwargs):
        return np.eye(1), np.zeros(1), None, 0


class TestPaperExactLineSearch:
    """The nonmonotone line search: acceptance rule and backtracking fallback.

    Both cases drive a stub estimator whose Phi returns a prescribed residual,
    so the accepted step is determined by the line-search policy alone.
    """

    def test_default_accepts_pure_nonmonotone_equality(self):
        """With ls_max_tries=0 the unit step is the only trial, and it is accepted.

        The stub makes Q_trial exactly equal to the nonmonotone reference Q_ref,
        so acceptance here rides on the rule being ``Q_trial <= Q_ref``; a strict
        inequality would reject the step and fall through to the fallback.
        """
        cfg = EstimationConfig(max_iterations=1, ls_max_tries=0)
        est = _LineSearchEstimator(cfg, trial_residual=1.0)

        result = est.fit(np.zeros((1, 2)), np.array([[-1.0, 1.0]]), beta_init=[0.0])

        assert result["diagnostics"]["accepted_history"] == [True]
        assert result["diagnostics"]["step_sizes"] == [1.0]

    def test_fallback_step_is_tau_to_mmax(self):
        """A step that is never accepted falls back to ls_tau ** ls_max_tries.

        With ls_tau=0.5 and ls_max_tries=3 the trial sequence is 1, 0.5, 0.25,
        0.125 — four evaluations — and the last one is taken.
        """
        cfg = EstimationConfig(
            max_iterations=1,
            ls_tau=0.5,
            ls_max_tries=3,
        )
        est = _LineSearchEstimator(cfg, trial_residual=2.0)

        result = est.fit(np.zeros((1, 2)), np.array([[-1.0, 1.0]]), beta_init=[0.0])

        assert result["diagnostics"]["accepted_history"] == [False]
        assert result["diagnostics"]["ls_tries_history"] == [4]
        assert result["diagnostics"]["step_sizes"] == [0.125]
        np.testing.assert_allclose(result["diagnostics"]["beta_history"][0], [0.125])


class TestODInputs:
    @staticmethod
    def _estimator() -> MicroPURCEstimator:
        net = Network.grid(2, 2, K=2)
        solver = PIQPFlowSolver(net, net.Z[:, 0].copy(), SolverConfig(backend="piqp"))
        return MicroPURCEstimator(
            net,
            solver,
            estimation_config=EstimationConfig(),
            active_set_config=ActiveSetConfig(),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )

    def test_dense_y_with_explicit_od_and_no_b(self):
        est = self._estimator()
        y = np.zeros((2, est.network.num_links), dtype=float)
        origins = np.array([0, 0])
        dests = np.array([3, 3])

        H, g, _, _ = est._accumulate(np.array([0.5, 0.5]), y, None, origins=origins, dests=dests)

        assert H.shape == (2, 2)
        assert g.shape == (2,)

    def test_dense_and_sparse_y_match_with_explicit_od(self):
        est_dense = self._estimator()
        est_sparse = self._estimator()
        y = np.zeros((2, est_dense.network.num_links), dtype=float)
        origins = np.array([0, 0])
        dests = np.array([3, 3])
        beta = np.array([0.5, 0.5])

        H_dense, g_dense, _, _ = est_dense._accumulate(beta, y, None, origins=origins, dests=dests)
        H_sparse, g_sparse, _, _ = est_sparse._accumulate(
            beta, sp.csr_matrix(y), None, origins=origins, dests=dests
        )

        np.testing.assert_allclose(H_dense, H_sparse)
        np.testing.assert_allclose(g_dense, g_sparse)

    def test_partial_explicit_od_raises(self):
        est = self._estimator()
        y = np.zeros((1, est.network.num_links), dtype=float)

        with pytest.raises(ValueError, match="origins and dests"):
            est._accumulate(np.array([0.5, 0.5]), y, None, origins=np.array([0]))

    def test_missing_b_and_od_raises(self):
        est = self._estimator()
        y = np.zeros((1, est.network.num_links), dtype=float)

        with pytest.raises(ValueError, match="b is required"):
            est._accumulate(np.array([0.5, 0.5]), y, None)


def _load_ema_sample(N: int = 8):
    from micropurc.demands import od_from_csv
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
    beta = np.array([0.5, 0.5, 0.5, 0.5])
    solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
    od_dist = od_from_csv(net, str(data_dir / "EMA_trips.csv"))
    sampler = MarkovSampler(network=net)
    dgp = DGP(net, beta, solver, sampler, od_dist, rng=np.random.default_rng(123))
    data = dgp.sample_dataset(N)
    return net, scale_m, beta, data


class TestNativeVariantAndSolverSettings:
    def test_piqp_solver_stores_and_applies_max_iter(self):
        net = Network.grid(2, 2, K=1)
        cfg = SolverConfig(backend="piqp", max_iter=7)
        solver = PIQPFlowSolver(net, net.Z[:, 0].copy(), cfg)

        assert solver.config is cfg
        assert solver._solver.settings.max_iter == 7

    def test_native_receives_solver_settings(self):
        net, scale_m, beta, data = _load_ema_sample(N=4)
        cfg = SolverConfig(
            backend="piqp",
            eps_abs=1e-9,
            eps_duality_gap_abs=1e-8,
            max_iter=777,
        )
        solver = PIQPFlowSolver(net, scale_m, cfg)
        est = MicroPURCEstimator(net, solver, projection_config=ProjectionConfig())

        est._accumulate(beta, data["y"], data["b"])

        assert est._last_native_stats["qp_eps_abs"] == pytest.approx(1e-9)
        assert est._last_native_stats["qp_eps_duality_gap_abs"] == pytest.approx(1e-8)
        assert est._last_native_stats["qp_max_iter"] == 777

    def test_low_native_max_iter_drops_unconverged_groups(self):
        net, scale_m, beta, data = _load_ema_sample(N=6)
        solver = PIQPFlowSolver(
            net,
            scale_m,
            SolverConfig(backend="piqp", max_iter=1),
        )
        est = MicroPURCEstimator(net, solver, projection_config=ProjectionConfig())

        _, _, _, dropped = est._accumulate(beta, data["y"], data["b"])

        assert est._last_native_stats["qp_max_iter"] == 1
        assert dropped > 0

    def test_variant_warm_cache_is_group_keyed_and_deterministic(self, monkeypatch):
        net, scale_m, beta, data = _load_ema_sample(N=4)
        origin = int(np.argmin(data["b"][0]))
        dest = int(np.argmax(data["b"][0]))
        origins = np.full(4, origin, dtype=np.int64)
        dests = np.full(4, dest, dtype=np.int64)
        trip_variant = np.array([0, 1, 0, 1], dtype=np.int64)
        z_variants = np.stack([net.Z, net.Z * np.array([1.0, 1.02, 0.98, 1.01])])

        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        solver1 = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est1 = MicroPURCEstimator(net, solver1, projection_config=ProjectionConfig())
        H1, g1, _, _ = est1._accumulate(
            beta,
            data["y"][:4],
            None,
            origins=origins,
            dests=dests,
            z_variants=z_variants,
            trip_variant=trip_variant,
        )
        H1_repeat, g1_repeat, _, _ = est1._accumulate(
            beta,
            data["y"][:4],
            None,
            origins=origins,
            dests=dests,
            z_variants=z_variants,
            trip_variant=trip_variant,
        )

        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        solver2 = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est2 = MicroPURCEstimator(net, solver2, projection_config=ProjectionConfig())
        H2, g2, _, _ = est2._accumulate(
            beta,
            data["y"][:4],
            None,
            origins=origins,
            dests=dests,
            z_variants=z_variants,
            trip_variant=trip_variant,
        )

        od_key = origin * net.num_nodes + dest
        np.testing.assert_array_equal(est1._od_keys_prev, [od_key * 2, od_key * 2 + 1])
        assert est1._lam_warm.shape == (2, net.num_nodes)
        np.testing.assert_allclose(H1, H1_repeat)
        np.testing.assert_allclose(g1, g1_repeat)
        np.testing.assert_allclose(H1, H2)
        np.testing.assert_allclose(g1, g2)

    def test_variant_validation_rejects_bad_inputs(self):
        net, scale_m, beta, data = _load_ema_sample(N=2)
        solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est = MicroPURCEstimator(net, solver, projection_config=ProjectionConfig())
        origins = np.array([int(np.argmin(row)) for row in data["b"][:2]])
        dests = np.array([int(np.argmax(row)) for row in data["b"][:2]])
        z_variants = np.stack([net.Z, net.Z])

        with pytest.raises(ValueError, match="trip_variant must have length"):
            est._accumulate(
                beta,
                data["y"][:2],
                None,
                origins=origins,
                dests=dests,
                z_variants=z_variants,
                trip_variant=np.array([0]),
            )

        with pytest.raises(ValueError, match="entries must be"):
            est._accumulate(
                beta,
                data["y"][:2],
                None,
                origins=origins,
                dests=dests,
                z_variants=z_variants,
                trip_variant=np.array([0, 2]),
            )


def _plain_lazy_design(network_attr, trip_variant):
    """A LazyDesign with two plain-column terms over columns 0 and 1."""
    n_var = network_attr.shape[0]
    return LazyDesign(
        link_attrs=network_attr,
        variant_attr=np.arange(n_var),
        variant_covariates=np.zeros((n_var, 1)),
        trip_variant=trip_variant,
        term_code=np.array([0, 0]),  # TERM_COLUMN, TERM_COLUMN
        term_a=np.array([0, 1]),
        term_b=np.array([0, 0]),
        term_cov=np.array([0, 0]),
        term_op=np.array([0, 0]),
        term_thr=np.array([0.0, 0.0]),
    )


class TestLazyPIQPDesign:
    def test_lazy_design_matches_materialized_variants(self, monkeypatch):
        net = Network.grid(3, 3, K=2)
        L = net.num_links
        scale_m = np.ones(L)
        beta = np.array([0.5, 0.2])

        rng = np.random.default_rng(123)
        network_attr = np.zeros((2, L, 4), dtype=float)
        network_attr[0, :, 0] = 1.0 + rng.uniform(size=L)
        network_attr[0, :, 1] = 0.5 + rng.uniform(size=L)
        network_attr[1, :, 0] = 1.1 * network_attr[0, :, 0]
        network_attr[1, :, 1] = 0.9 * network_attr[0, :, 1]
        trip_variant = np.array([0, 1, 0, 1], dtype=np.int64)
        origins = np.array([0, 0, 0, 0], dtype=np.int64)
        dests = np.array([8, 8, 8, 8], dtype=np.int64)
        y = sp.csr_matrix((4, L), dtype=float)

        lazy_design = _plain_lazy_design(network_attr, trip_variant)
        z_variants = network_attr[:, :, :2]

        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        solver_mat = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est_mat = MicroPURCEstimator(
            net, solver_mat, projection_config=ProjectionConfig(mode="robust_pinv")
        )
        H_mat, g_mat, _, dropped_mat = est_mat._accumulate(
            beta,
            y,
            None,
            origins=origins,
            dests=dests,
            z_variants=z_variants,
            trip_variant=trip_variant,
        )

        solver_lazy = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est_lazy = MicroPURCEstimator(
            net, solver_lazy, projection_config=ProjectionConfig(mode="robust_pinv")
        )
        H_lazy, g_lazy, _, dropped_lazy = est_lazy._accumulate(
            beta,
            y,
            None,
            origins=origins,
            dests=dests,
            lazy_design=lazy_design,
        )

        np.testing.assert_allclose(H_lazy, H_mat, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(g_lazy, g_mat, atol=1e-10, rtol=1e-10)
        assert dropped_lazy == dropped_mat
        assert est_lazy._last_native_stats["lazy_design"] is True
        assert "time_cost_build_s" in est_lazy._last_native_stats

    def test_lazy_design_thread_deterministic(self, monkeypatch):
        net = Network.grid(3, 3, K=2)
        L = net.num_links
        scale_m = np.ones(L)
        beta = np.array([0.5, 0.2])
        network_attr = np.zeros((2, L, 4), dtype=float)
        network_attr[:, :, 0] = 1.0
        network_attr[0, :, 1] = 0.5
        network_attr[1, :, 1] = 0.7
        lazy_design = _plain_lazy_design(network_attr, np.array([0, 1, 0, 1], dtype=np.int64))
        origins = np.zeros(4, dtype=np.int64)
        dests = np.full(4, 8, dtype=np.int64)
        y = sp.csr_matrix((4, L), dtype=float)

        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        est1 = MicroPURCEstimator(
            net,
            PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp")),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )
        H1, g1, _, _ = est1._accumulate(
            beta, y, None, origins=origins, dests=dests, lazy_design=lazy_design
        )

        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        est2 = MicroPURCEstimator(
            net,
            PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp")),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )
        H2, g2, _, _ = est2._accumulate(
            beta, y, None, origins=origins, dests=dests, lazy_design=lazy_design
        )

        np.testing.assert_allclose(H1, H2)
        np.testing.assert_allclose(g1, g2)
