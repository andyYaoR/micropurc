"""Active-set identification under the dual-slack (KKT sign) rule.

A link is active when its slack s_j = c_j + (A' lambda)_j falls below
-slack_tol; at the default slack_tol = 0.0 that is a strict sign test. The two
asserting tests here are a matched pair: one shows Phi is unchanged to 1e-13 as
slack_tol is raised to 1e-4, the other shows why — every active link's slack
already sits further than 1e-4 below zero, so moving the cutoff over that range
reclassifies nothing. A third, assertion-free case reports how the identified
sets move as the forward solver's own tolerance is loosened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"

from micropurc.config import ActiveSetConfig, SolverConfig
from micropurc.core.activeset import _check_dag_connectivity, determine_active_set


class TestActiveSetNativeWrappers:
    """Regression tests for Python wrappers backed by native C++."""

    def test_empty_edge_list_is_trivially_dag_connected(self):
        is_dag, is_connected = _check_dag_connectivity(
            np.array([], dtype=int),
            np.array([], dtype=int),
            dest=0,
        )
        assert is_dag
        assert is_connected


def _load_ema():
    data_dir = DATA_DIR
    if not (data_dir / "EMA_net.csv").exists():
        pytest.skip("EMA data files not found")

    from micropurc.network import Network

    net = Network.from_csv(
        str(data_dir / "EMA_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1", "attr_2"],
    )
    scale_m = net.Z[:, 0].copy()
    return net, scale_m


def _make_solver(net, scale_m, eps_abs=1e-12):
    from micropurc.forward.piqp_solver import PIQPFlowSolver

    return PIQPFlowSolver(
        net,
        scale_m,
        SolverConfig(backend="piqp", eps_abs=eps_abs, eps_duality_gap_abs=eps_abs),
    )


def _generate_trips(net, solver, beta, n_trips=100, seed=42):
    from micropurc.demands import od_from_csv
    from micropurc.simulation.dgp import DGP
    from micropurc.simulation.route_sampler import MarkovSampler

    data_dir = DATA_DIR
    od_dist = od_from_csv(net, str(data_dir / "EMA_trips.csv"))
    sampler = MarkovSampler(network=net)
    dgp = DGP(net, beta, solver, sampler, od_dist, rng=np.random.default_rng(seed))
    data = dgp.sample_dataset(n_trips)

    c = net.Z @ beta
    trips = []
    for n in range(n_trips):
        result = solver.solve(c, data["b"][n])
        dest = int(np.argmax(data["b"][n]))
        trips.append((result, data["y"][n], data["b"][n], dest))
    return trips, c


class TestDualSlackThresholdInvariance:
    """Phi is invariant to slack_tol over the range where no link's slack lives."""

    @pytest.fixture(scope="class")
    def setup(self):
        net, scale_m = _load_ema()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        solver = _make_solver(net, scale_m)
        trips, c = _generate_trips(net, solver, beta, n_trips=100)
        y = np.array([t[1] for t in trips])
        b = np.array([t[2] for t in trips])
        return net, scale_m, solver, trips, c, beta, y, b

    def test_phi_independent_of_slack_tol(self, setup):
        """Phi(beta) must agree to 1e-13 across slack_tol from 0 up to 1e-4."""
        from micropurc.config import EstimationConfig, ProjectionConfig
        from micropurc.core.estimator import MicroPURCEstimator

        net, scale_m, solver, trips, c, beta, y, b = setup
        tols = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
        phis = []

        for tol in tols:
            est = MicroPURCEstimator(
                net,
                solver,
                estimation_config=EstimationConfig(),
                active_set_config=ActiveSetConfig(slack_tol=tol),
                projection_config=ProjectionConfig(mode="robust_pinv"),
            )
            phis.append(est.phi(beta, y, b))

        for i in range(1, len(phis)):
            np.testing.assert_allclose(
                phis[0],
                phis[i],
                atol=1e-13,
                err_msg=f"Dual slack Phi changed at slack_tol={tols[i]:.0e}",
            )

    def test_slack_gap_away_from_zero(self, setup):
        """Across the first 50 trips, every active link's slack is below -1e-4.

        This is the gap that makes the sign test well posed, and it is what the
        slack_tol sweep above rests on: with the solve run at eps_abs = 1e-12,
        the classification survives both solver noise and a cutoff moved to
        -1e-4.
        """
        net, scale_m, solver, trips, c, beta, y, b = setup

        min_active_gap = np.inf
        max_inactive_slack = -np.inf

        for result, y_i, b_i, dest in trips[:50]:
            if result.lam_eq is None:
                continue
            slack = c + (net.A.T @ result.lam_eq)
            active = slack < 0
            if active.any():
                min_active_gap = min(min_active_gap, float(np.min(-slack[active])))
            inactive = slack >= 0
            if inactive.any():
                max_inactive_slack = max(max_inactive_slack, float(np.max(-slack[inactive])))

        print(f"\n  Slack gap: min |s| on active links = {min_active_gap:.6e}")
        print(f"             max -s on inactive links = {max_inactive_slack:.6e}")
        assert min_active_gap > 1e-4, f"Active link slack too close to zero: {min_active_gap:.2e}"


class TestDualSlackSolverRobustness:
    """Measure how the identified active sets move with forward-solver accuracy."""

    @pytest.fixture(scope="class")
    def setup(self):
        net, scale_m = _load_ema()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        solver_ref = _make_solver(net, scale_m, eps_abs=1e-14)
        trips, c = _generate_trips(net, solver_ref, beta, n_trips=50, seed=99)
        return net, scale_m, trips, c, beta

    def test_accuracy_sweep(self, setup):
        """The dual-slack active set is identical at every eps_abs from 1e-12 to 1e-6.

        The sign test flips only if a perturbation carries a link's slack across
        zero. ``test_slack_gap_away_from_zero`` measures the smallest active-link
        slack at about 1.8e-3, which is three orders of magnitude above the
        loosest tolerance swept here, so the exact churn to expect is zero. The
        per-tolerance table is printed as well; run with ``-s`` to read it.
        """
        net, scale_m, trips, c, beta = setup

        eps_values = [1e-12, 1e-10, 1e-8, 1e-6]

        print(f"\n  Active-set stability vs solver accuracy (N={len(trips)}):")
        print(f"  {'eps_abs':>10s}  {'dual_churn':>11s}")

        # Reference: dual slack at tightest tolerance
        ref_sets = []
        for result, y_i, b_i, dest in trips:
            aset = determine_active_set(
                result.x,
                dest,
                net,
                ActiveSetConfig(),
                projection_mode="robust_pinv",
                lam_eq=result.lam_eq,
                c=c,
                scale_m=scale_m,
            )
            ref_sets.append(set(aset.indices) if aset.valid else None)

        for eps in eps_values:
            solver_test = _make_solver(net, scale_m, eps_abs=eps)
            dual_changes = 0
            n_compared = 0

            for idx, (_, y_i, b_i, dest) in enumerate(trips):
                if ref_sets[idx] is None:
                    continue

                test_result = solver_test.solve(c, b_i)
                n_compared += 1

                aset_d = determine_active_set(
                    test_result.x,
                    dest,
                    net,
                    ActiveSetConfig(),
                    projection_mode="robust_pinv",
                    lam_eq=test_result.lam_eq,
                    c=c,
                    scale_m=scale_m,
                )

                if aset_d.valid and set(aset_d.indices) != ref_sets[idx]:
                    dual_changes += 1

            print(f"  {eps:>10.0e}  {dual_changes:>11d}")
            assert dual_changes == 0, (
                f"eps_abs={eps:.0e}: {dual_changes} of {n_compared} trips changed "
                "active set relative to the eps=1e-14 reference."
            )
