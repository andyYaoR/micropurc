r"""The native forward solver, and its equivalence with the pip ``piqp`` oracle.

``PIQPFlowSolver`` drives PIQP through the package's own C++ extension, which
compiles the vendored PIQP headers in ``extern/piqp``. The pip ``piqp``
package is kept as a test-only dependency so that equivalence with an
independently built PIQP stays provable: every oracle test here rebuilds the
same problem

.. math::

    \min_x \; c^{\top} x + \tfrac12 x^{\top} \mathrm{diag}(s) x
    \quad \text{s.t.} \quad A x = b, \; x \ge 0

through ``piqp.SparseSolver`` with the identical settings and compares.

Two properties of the comparison shape every tolerance below.

1. The minimizer is unique (the objective is strongly convex with modulus
   ``sigma_min = min(scale_m)``), so any disagreement is solver inaccuracy. A
   point with duality gap ``g`` obeys ``f(x) - f(x*) <= g``, and strong
   convexity turns that into ``||x - x*|| <= sqrt(2 g / sigma_min)``:
   :func:`_agreement_tol` is exactly that bound for the two sides, and is a
   theorem rather than a tuned threshold.
2. The equality duals are NOT unique. ``A`` is a node-arc incidence matrix, so
   its rows sum to zero and adding a constant to the potentials of a connected
   component leaves ``A' lam`` unchanged; a node touched by no active link is
   its own component and its potential is pinned only to an interval. Dual
   comparisons therefore run per connected component of the active subgraph,
   after removing the component's mean.

The tolerances the two sides can reach also depend on ``SolverConfig``. PIQP
stops on the absolute OR the relative criterion, so at the shipped default
``eps_rel = 1e-9`` both implementations stop while still ``O(1e-5)`` away from
the exact minimizer (strong convexity converts a permitted ``1e-8`` objective
gap into a ``1e-4`` flow difference). The equivalence sweeps therefore run at
purely absolute tolerances, where the two agree to ``1e-10``, and one dedicated
test measures the default configuration against a high-accuracy reference and
checks that the native path is no further from it than the oracle is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from micropurc.config import (
    ActiveSetConfig,
    EstimationConfig,
    ProjectionConfig,
    SolverConfig,
)
from micropurc.core.estimator import MicroPURCEstimator
from micropurc.demands import od_from_csv, od_uniform_all_pairs
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.dgp import DGP
from micropurc.simulation.route_sampler import MarkovSampler

DATA_DIR = Path(__file__).resolve().parent / "data"

HAS_PIQP = importlib.util.find_spec("piqp") is not None
oracle_test = pytest.mark.oracle
requires_oracle = pytest.mark.skipif(
    not HAS_PIQP, reason="pip `piqp` (the equivalence oracle) is not installed"
)

# Purely absolute termination: PIQP stops on the absolute OR the relative
# criterion, and the shipped relative default leaves both implementations
# ~1e-5 from the minimizer, which no cross-implementation test can see through.
TIGHT = SolverConfig(
    backend="piqp",
    eps_abs=1e-12,
    eps_duality_gap_abs=1e-12,
    eps_rel=0.0,
    eps_duality_gap_rel=0.0,
)

# Flow agreement claimed on the EMA fixtures. The strong-convexity certificate
# at a 1e-12 duality gap and the EMA scale (scale_m = link length, min ~2) puts
# the guaranteed bound near 1e-6; the measured worst case over all 1113 OD
# pairs is 8.4e-11, so 1e-8 is a far tighter claim than the theory, met with a
# hundredfold margin.
EMA_X_ATOL = 1e-8
# Potentials, compared per active component after removing that component's
# mean. Measured worst case over the same 1113 pairs: 7.7e-10.
EMA_LAM_ATOL = 1e-7


def _agreement_tol(gap_a: float, gap_b: float, sigma_min: float, gap_floor: float) -> float:
    """Largest honest disagreement between two solutions of the same QP.

    Strong convexity with modulus ``sigma_min`` turns each side's duality gap
    into a distance to the unique minimizer, and the triangle inequality adds
    the two. A reported gap is floored at the configured
    ``eps_duality_gap_abs``: a solver that reports zero has certified its gap
    only down to the tolerance it was asked to reach, so nothing below that is
    known.
    """
    ga = max(abs(gap_a), gap_floor)
    gb = max(abs(gap_b), gap_floor)
    return np.sqrt(2.0 * ga / sigma_min) + np.sqrt(2.0 * gb / sigma_min)


class PipPIQPFlowSolver(PIQPFlowSolver):
    """Oracle: the same forward solver driven by the pip ``piqp`` package.

    Subclasses the production solver so that everything except :meth:`solve`
    (``scale_m``, ``config``, ``solve_batch``) is shared, and only the single
    cold solve is rerouted to an independently built PIQP.
    """

    def __init__(self, network, scale_m, config=None):
        """Set up the pip-``piqp`` solver alongside the native one."""
        super().__init__(network, scale_m, config)
        import piqp
        import scipy.sparse as sp

        cfg = self.config
        L = int(self._scale_m.shape[0])
        V = int(self._A_eq.shape[0])
        self._pip = piqp.SparseSolver()
        s = self._pip.settings
        s.verbose = cfg.verbose
        s.eps_abs = cfg.eps_abs
        s.eps_duality_gap_abs = cfg.eps_duality_gap_abs
        s.eps_rel = cfg.eps_rel
        s.eps_duality_gap_rel = cfg.eps_duality_gap_rel
        s.max_iter = cfg.max_iter
        self._pip.setup(
            sp.diags(self._scale_m, format="csc", dtype=np.float64),
            c=np.zeros((L, 1)),
            A=self._A_eq,
            b=np.zeros((V, 1)),
            G=None,
            h_l=None,
            h_u=None,
            x_l=np.zeros((L, 1)),
            x_u=np.full((L, 1), np.inf),
        )

    def solve(self, c, b, warm=None):
        """Solve one right-hand side through pip ``piqp``."""
        from micropurc.forward.base import FlowResult

        self._pip.update(
            c=np.asarray(c, dtype=np.float64).reshape(-1, 1),
            b=np.asarray(b, dtype=np.float64).reshape(-1, 1),
        )
        self._pip.solve()
        result = self._pip.result
        info = {
            "status": int(result.info.status),
            "iter": int(result.info.iter),
            "duality_gap": float(result.info.duality_gap),
        }
        return FlowResult(
            x=np.asarray(result.x, dtype=np.float64).ravel().copy(),
            lam_eq=np.asarray(result.y, dtype=np.float64).ravel().copy(),
            info=info,
        )


def _active_mask(net: Network, c: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Dual-slack active set: links whose slack ``c_j + (A' lam)_j`` is negative."""
    return (np.asarray(c, dtype=float) + net.A.T @ np.asarray(lam, dtype=float)) < 0.0


def _component_labels(net: Network, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label each node by its component in the active subgraph.

    Returns ``(labels, touched)``: the component representative per node, and
    whether the node carries at least one active link.
    """
    parent = np.arange(net.num_nodes)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    touched = np.zeros(net.num_nodes, dtype=bool)
    for j in np.flatnonzero(active):
        u, v = int(net.tail[j]), int(net.head[j])
        touched[u] = True
        touched[v] = True
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
    labels = np.array([find(v) for v in range(net.num_nodes)])
    return labels, touched


def _max_potential_spread(net: Network, active: np.ndarray, d_lam: np.ndarray) -> float:
    """Worst within-component spread of a potential difference.

    Two valid dual solutions may differ by a constant per connected component
    of the active subgraph, so only the spread inside a component carries
    information.
    """
    labels, touched = _component_labels(net, active)
    worst = 0.0
    for root in np.unique(labels[touched]):
        m = (labels == root) & touched
        if m.sum() >= 2:
            worst = max(worst, float(np.abs(d_lam[m] - d_lam[m].mean()).max()))
    return worst


def _load_ema() -> tuple[Network, np.ndarray]:
    if not (DATA_DIR / "EMA_net.csv").exists():
        pytest.skip("EMA data files not found")
    net = Network.from_csv(
        str(DATA_DIR / "EMA_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1", "attr_2"],
    )
    return net, net.Z[:, 0].copy()


def _ema_problems() -> tuple[Network, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Every OD pair in the EMA trip file, as unit-demand right-hand sides."""
    net, scale_m = _load_ema()
    c = net.Z @ np.array([0.5, 0.5, 0.5, 0.5])
    od = od_from_csv(net, str(DATA_DIR / "EMA_trips.csv"))
    rhs = []
    for o, d in zip(od.origins, od.dests):
        b = np.zeros(net.num_nodes)
        b[int(o)] = -1.0
        b[int(d)] = 1.0
        rhs.append(b)
    return net, scale_m, c, rhs


def _random_problem_suite(seed: int = 7):
    """Networks, perturbation scales, costs, and ODs spanning the hard cases.

    Sweeps grid sizes, uniform against six-orders-of-magnitude skewed
    ``scale_m``, and cost vectors that are positive, exactly zero, negative, or
    large in magnitude; ODs mix adjacent nodes with far-apart pairs.
    """
    rng = np.random.default_rng(seed)
    for rows, cols in [(3, 3), (5, 4), (8, 8)]:
        net = Network.grid(rows, cols, K=3, rng_seed=seed)
        L, V = net.num_links, net.num_nodes
        for scale_name, scale_m in [
            ("uniform", rng.uniform(0.5, 1.5, L)),
            ("skewed", 10.0 ** rng.uniform(-3.0, 3.0, L)),
        ]:
            for cost_name, c in [
                ("positive", rng.uniform(0.1, 2.0, L)),
                ("zeros", np.zeros(L)),
                ("negative", rng.uniform(-2.0, -0.1, L)),
                ("large", rng.uniform(-1e3, 1e3, L)),
            ]:
                rhs = []
                # Adjacent OD pairs (one link apart) alongside random far pairs.
                for j in rng.choice(L, size=4, replace=False):
                    b = np.zeros(V)
                    b[int(net.tail[j])] = -1.0
                    b[int(net.head[j])] = 1.0
                    rhs.append(b)
                for _ in range(8):
                    o, d = rng.choice(V, size=2, replace=False)
                    b = np.zeros(V)
                    b[int(o)] = -1.0
                    b[int(d)] = 1.0
                    rhs.append(b)
                yield f"grid{rows}x{cols}/{scale_name}/{cost_name}", net, scale_m, c, rhs


class TestNativeSolverContract:
    """The native solver's own contract, checked without the oracle."""

    def test_flow_result_shapes_and_info_keys(self):
        net = Network.grid(3, 3, K=2)
        scale_m = net.Z[:, 0] + 1.0
        solver = PIQPFlowSolver(net, scale_m, TIGHT)
        b = np.zeros(net.num_nodes)
        b[0] = -1.0
        b[-1] = 1.0
        res = solver.solve(net.Z @ np.array([1.0, 0.5]), b)

        assert res.x.shape == (net.num_links,)
        assert res.lam_eq.shape == (net.num_nodes,)
        assert res.x.dtype == np.float64 and res.lam_eq.dtype == np.float64
        for key in ("status", "iter", "primal_residual", "dual_residual", "duality_gap"):
            assert key in res.info
        assert int(res.info["status"]) == 1  # PIQP_SOLVED

    def test_settings_are_read_back_from_config(self):
        net = Network.grid(2, 2, K=1)
        cfg = SolverConfig(
            backend="piqp",
            eps_abs=1e-11,
            eps_duality_gap_abs=1e-10,
            eps_rel=1e-8,
            eps_duality_gap_rel=1e-7,
            max_iter=13,
            verbose=False,
        )
        s = PIQPFlowSolver(net, net.Z[:, 0].copy() + 1.0, cfg)._solver.settings

        assert s.eps_abs == 1e-11
        assert s.eps_duality_gap_abs == 1e-10
        assert s.eps_rel == 1e-8
        assert s.eps_duality_gap_rel == 1e-7
        assert s.max_iter == 13
        assert s.verbose is False

    def test_earlier_results_survive_later_solves(self):
        """Each solve returns its own arrays, so a held result is never rewritten."""
        net = Network.grid(3, 3, K=2)
        solver = PIQPFlowSolver(net, net.Z[:, 0] + 1.0, TIGHT)
        c = net.Z @ np.array([1.0, 0.5])
        b1 = np.zeros(net.num_nodes)
        b1[0] = -1.0
        b1[-1] = 1.0
        b2 = np.zeros(net.num_nodes)
        b2[1] = -1.0
        b2[-2] = 1.0

        r1 = solver.solve(c, b1)
        x1, lam1 = r1.x.copy(), r1.lam_eq.copy()
        solver.solve(c, b2)

        np.testing.assert_array_equal(r1.x, x1)
        np.testing.assert_array_equal(r1.lam_eq, lam1)

    def test_wrong_length_inputs_are_rejected(self):
        net = Network.grid(3, 3, K=2)
        solver = PIQPFlowSolver(net, net.Z[:, 0] + 1.0, TIGHT)
        c = np.zeros(net.num_links)
        b = np.zeros(net.num_nodes)

        with pytest.raises(ValueError, match="c must have length L"):
            solver.solve(np.zeros(net.num_links + 1), b)
        with pytest.raises(ValueError, match="b must have length V"):
            solver.solve(c, np.zeros(net.num_nodes + 1))

    def test_scale_m_length_is_validated(self):
        net = Network.grid(3, 3, K=2)
        with pytest.raises(ValueError, match="scale_m has length"):
            PIQPFlowSolver(net, np.ones(net.num_links + 1), TIGHT)

    def test_malformed_incidence_is_rejected(self):
        """The native constructor fails loudly on an inconsistent CSC triple."""
        from micropurc._native import NativeSparseSolver

        net = Network.grid(3, 3, K=2)
        A = net.A.astype(np.float64).tocsc()
        V, L = int(A.shape[0]), int(A.shape[1])
        good = dict(
            A_indptr=A.indptr.astype(np.int64),
            A_indices=A.indices.astype(np.int64),
            A_data=np.ascontiguousarray(A.data, dtype=np.float64),
            scale_m=np.ones(L),
            V=V,
            L=L,
            verbose=False,
            eps_abs=1e-12,
            eps_duality_gap_abs=1e-12,
            eps_rel=0.0,
            eps_duality_gap_rel=0.0,
            max_iter=1000,
        )
        with pytest.raises(ValueError, match="A_indptr must have length"):
            NativeSparseSolver(**{**good, "L": L - 1})
        with pytest.raises(ValueError, match="scale_m must have length"):
            NativeSparseSolver(**{**good, "scale_m": np.ones(L + 1)})
        with pytest.raises(ValueError, match="A_indices and A_data"):
            NativeSparseSolver(**{**good, "A_data": good["A_data"][:-1]})
        bad_indices = good["A_indices"].copy()
        bad_indices[0] = V
        with pytest.raises(ValueError, match="A_indices entry outside"):
            NativeSparseSolver(**{**good, "A_indices": bad_indices})

    def test_solve_matches_solve_batch(self):
        """The single and batch entry points run the same native solver stack.

        Both are checked at the default ``SolverConfig``, the one configuration
        under which their termination settings coincide: ``solve_batch``
        deliberately leaves the relative criteria at PIQP's own defaults, which
        are the values the default ``SolverConfig`` carries. The two then agree
        exactly on this fixture, and 1e-12 allows only for round-off from the
        batch path's work-stealing over threads.
        """
        net = Network.grid(4, 4, K=2)
        scale_m = net.Z[:, 0] + 1.0
        solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        c = net.Z @ np.array([1.0, 0.5])
        origins = np.array([0, 1, 2], dtype=np.int64)
        dests = np.array([net.num_nodes - 1, net.num_nodes - 2, 5], dtype=np.int64)

        flows, ok = solver.solve_batch(c, origins, dests)

        assert ok.all()
        for u, (o, d) in enumerate(zip(origins, dests)):
            b = np.zeros(net.num_nodes)
            b[o] = -1.0
            b[d] = 1.0
            res = solver.solve(c, b)
            # solve_batch clamps at the primal lower bound; solve reports the
            # IPM's raw primal, which can sit a rounding step below zero.
            np.testing.assert_allclose(flows[u], np.maximum(res.x, 0.0), rtol=0, atol=1e-12)


@oracle_test
@requires_oracle
class TestOracleAgreementEMA:
    """Native against pip ``piqp`` on the repository's EMA fixtures."""

    @pytest.fixture(scope="class")
    @staticmethod
    def solved():
        net, scale_m, c, rhs = _ema_problems()
        native = PIQPFlowSolver(net, scale_m, TIGHT)
        oracle = PipPIQPFlowSolver(net, scale_m, TIGHT)
        pairs = [(native.solve(c, b), oracle.solve(c, b)) for b in rhs]
        return net, scale_m, c, pairs

    def test_flows_agree(self, solved):
        net, scale_m, c, pairs = solved
        worst = max(float(np.abs(rn.x - ro.x).max()) for rn, ro in pairs)
        assert worst <= EMA_X_ATOL, f"worst flow disagreement over {len(pairs)} ODs: {worst:.3e}"

    def test_flows_agree_within_the_strong_convexity_bound(self, solved):
        net, scale_m, c, pairs = solved
        sigma_min = float(scale_m.min())
        for rn, ro in pairs:
            tol = _agreement_tol(
                rn.info["duality_gap"],
                ro.info["duality_gap"],
                sigma_min,
                TIGHT.eps_duality_gap_abs,
            )
            assert float(np.abs(rn.x - ro.x).max()) <= tol

    def test_status_agrees(self, solved):
        net, scale_m, c, pairs = solved
        assert all(int(rn.info["status"]) == int(ro.info["status"]) for rn, ro in pairs)
        assert all(int(rn.info["status"]) == 1 for rn, _ in pairs)

    def test_active_sets_agree(self, solved):
        net, scale_m, c, pairs = solved
        mismatches = 0
        for rn, ro in pairs:
            native_active = _active_mask(net, c, rn.lam_eq)
            oracle_active = _active_mask(net, c, ro.lam_eq)
            mismatches += int(not np.array_equal(native_active, oracle_active))
        assert mismatches == 0, f"{mismatches} of {len(pairs)} OD pairs disagree on the active set"

    def test_potentials_agree_up_to_the_dual_null_space(self, solved):
        net, scale_m, c, pairs = solved
        worst = 0.0
        for rn, ro in pairs:
            active = _active_mask(net, c, rn.lam_eq)
            worst = max(worst, _max_potential_spread(net, active, rn.lam_eq - ro.lam_eq))
        assert worst <= EMA_LAM_ATOL, f"worst within-component potential spread: {worst:.3e}"

    def test_dual_sign_convention_matches_the_oracle(self, solved):
        r"""``lam_eq`` enters stationarity as ``+A' lam`` on both sides.

        The KKT conditions of the forward QP read
        ``c + diag(scale) x + A' lam = z >= 0`` with ``x_j z_j = 0``. Both
        solvers must satisfy them with the same sign on ``A' lam``, and
        flipping that sign must break them, which is what pins the convention
        rather than merely allowing it.
        """
        net, scale_m, c, pairs = solved
        flipped_violation = 0.0
        for rn, ro in pairs[:200]:
            for res in (rn, ro):
                z = c + scale_m * res.x + net.A.T @ res.lam_eq
                assert z.min() >= -1e-8
                assert float(np.abs(res.x * z).max()) <= 1e-8
            z_flipped = c + scale_m * rn.x - net.A.T @ rn.lam_eq
            flipped_violation = max(flipped_violation, float(-z_flipped.min()))
        assert flipped_violation > 1e-3


@oracle_test
@requires_oracle
class TestOracleAgreementRandomProblems:
    """Native against pip ``piqp`` over a sweep of generated problems."""

    def test_sweep_agrees_within_the_strong_convexity_bound(self):
        worst_ratio = 0.0
        worst_label = ""
        n_cases = 0
        for label, net, scale_m, c, rhs in _random_problem_suite():
            native = PIQPFlowSolver(net, scale_m, TIGHT)
            oracle = PipPIQPFlowSolver(net, scale_m, TIGHT)
            sigma_min = float(scale_m.min())
            for b in rhs:
                rn = native.solve(c, b)
                ro = oracle.solve(c, b)
                n_cases += 1
                # Every problem here is feasible and bounded, so neither side
                # may report infeasibility; hitting the iteration cap is
                # allowed, since a 1e-12 absolute duality gap is out of reach
                # when |c| runs to 1e3.
                assert int(rn.info["status"]) in (1, -1)
                assert int(ro.info["status"]) in (1, -1)
                tol = _agreement_tol(
                    rn.info["duality_gap"],
                    ro.info["duality_gap"],
                    sigma_min,
                    TIGHT.eps_duality_gap_abs,
                )
                ratio = float(np.abs(rn.x - ro.x).max()) / tol
                if ratio > worst_ratio:
                    worst_ratio, worst_label = ratio, label
        assert n_cases >= 200
        assert worst_ratio <= 1.0, f"{worst_label} exceeds the certificate by {worst_ratio:.2f}x"

    def test_repeated_solves_on_one_object_track_the_oracle(self):
        """Statefulness parity: compare after every solve of a shared sequence.

        Both sides keep one solver object across the whole sequence, so any
        state either carries between calls shows up as a growing divergence.
        """
        net = Network.grid(5, 4, K=3, rng_seed=3)
        rng = np.random.default_rng(11)
        L, V = net.num_links, net.num_nodes
        scale_m = rng.uniform(0.5, 1.5, L)
        native = PIQPFlowSolver(net, scale_m, TIGHT)
        oracle = PipPIQPFlowSolver(net, scale_m, TIGHT)
        sigma_min = float(scale_m.min())

        sequence = []
        for _ in range(25):
            o, d = rng.choice(V, size=2, replace=False)
            b = np.zeros(V)
            b[int(o)] = -1.0
            b[int(d)] = 1.0
            sequence.append((rng.uniform(-1.0, 2.0, L), b))

        for step, (c, b) in enumerate(sequence):
            rn = native.solve(c, b)
            ro = oracle.solve(c, b)
            tol = _agreement_tol(
                rn.info["duality_gap"],
                ro.info["duality_gap"],
                sigma_min,
                TIGHT.eps_duality_gap_abs,
            )
            assert float(np.abs(rn.x - ro.x).max()) <= tol, f"diverged at step {step}"

    def test_solve_sequence_is_order_independent(self):
        """A fresh object replaying the sequence backwards reaches the same flows."""
        net = Network.grid(4, 4, K=2, rng_seed=5)
        rng = np.random.default_rng(13)
        L, V = net.num_links, net.num_nodes
        scale_m = rng.uniform(0.5, 1.5, L)
        sequence = []
        for _ in range(15):
            o, d = rng.choice(V, size=2, replace=False)
            b = np.zeros(V)
            b[int(o)] = -1.0
            b[int(d)] = 1.0
            sequence.append((rng.uniform(-1.0, 2.0, L), b))

        forward = PIQPFlowSolver(net, scale_m, TIGHT)
        xs_forward = [forward.solve(c, b).x for c, b in sequence]
        backward = PIQPFlowSolver(net, scale_m, TIGHT)
        xs_backward = [None] * len(sequence)
        for i in reversed(range(len(sequence))):
            c, b = sequence[i]
            xs_backward[i] = backward.solve(c, b).x

        for xf, xb in zip(xs_forward, xs_backward):
            np.testing.assert_array_equal(xf, xb)


@oracle_test
@requires_oracle
class TestDefaultConfigAccuracy:
    """What the shipped ``SolverConfig`` defaults buy, measured against both."""

    def test_native_is_no_further_from_the_exact_solution_than_the_oracle(self):
        """At the default relative tolerance, neither side is systematically worse.

        ``eps_rel = 1e-9`` lets PIQP stop at an objective gap around ``1e-8``,
        which strong convexity turns into flow differences around ``1e-5`` --
        far above any cross-implementation tolerance. The meaningful statement
        is therefore about distance to a high-accuracy reference: the native
        path's worst objective excess must be of the same order as the
        oracle's, which is what rules out the native path terminating early.
        """
        net, scale_m, c, rhs = _ema_problems()
        rhs = rhs[:300]
        default = SolverConfig(backend="piqp")
        native = PIQPFlowSolver(net, scale_m, default)
        oracle = PipPIQPFlowSolver(net, scale_m, default)
        reference = PipPIQPFlowSolver(
            net,
            scale_m,
            SolverConfig(
                backend="piqp",
                eps_abs=1e-14,
                eps_duality_gap_abs=1e-14,
                eps_rel=0.0,
                eps_duality_gap_rel=0.0,
                max_iter=10000,
            ),
        )

        def objective(x):
            return float(c @ x + 0.5 * x @ (scale_m * x))

        excess_native = 0.0
        excess_oracle = 0.0
        for b in rhs:
            f_ref = objective(reference.solve(c, b).x)
            excess_native = max(excess_native, objective(native.solve(c, b).x) - f_ref)
            excess_oracle = max(excess_oracle, objective(oracle.solve(c, b).x) - f_ref)

        assert excess_native >= -1e-12, "the native path cannot beat the reference"
        # Measured on this fixture: 9.2e-8 (native) against 9.7e-8 (oracle).
        assert excess_native <= 10.0 * excess_oracle + 1e-12, (
            f"native objective excess {excess_native:.3e} against oracle {excess_oracle:.3e}"
        )


class TestEstimatorEndToEnd:
    """Simulate then estimate through the native forward solver."""

    @staticmethod
    def _grid_with_positive_scale(rows=4, cols=4, K=2, seed=0):
        net = Network.grid(rows, cols, K=1, rng_seed=seed)
        rng = np.random.default_rng(seed)
        Z = np.empty((net.num_links, K))
        Z[:, 0] = 1.0 + rng.exponential(0.5, net.num_links)
        Z[:, 1] = Z[:, 0] / (0.5 + rng.uniform(size=net.num_links))
        net.Z = Z
        return net

    @staticmethod
    def _fit(net, solver, data, beta_init):
        est = MicroPURCEstimator(
            net,
            solver,
            estimation_config=EstimationConfig(max_iterations=100),
            active_set_config=ActiveSetConfig(),
            projection_config=ProjectionConfig(mode="robust_pinv"),
        )
        return est.fit(data["y"], data["b"], beta_init=beta_init)

    def test_recovery_on_a_grid_from_a_cold_start(self):
        beta_true = np.array([1.0, 0.5])
        net = self._grid_with_positive_scale()
        solver = PIQPFlowSolver(net, net.Z[:, 0].copy(), TIGHT)
        dgp = DGP(
            net,
            beta_true,
            solver,
            MarkovSampler(network=net),
            od_uniform_all_pairs(net),
            rng=np.random.default_rng(42),
        )
        data = dgp.sample_dataset(1000)

        result = self._fit(net, solver, data, beta_init=np.ones(2) * 2.0)

        assert result["diagnostics"]["converged"]
        assert np.linalg.norm(result["beta_hat"] - beta_true) < 0.3

    @oracle_test
    @requires_oracle
    def test_fit_matches_a_pip_backed_pipeline(self):
        """Same seeds, same data, same fit, with the oracle generating the flows."""
        beta_true = np.array([1.0, 0.5])
        net = self._grid_with_positive_scale()
        scale_m = net.Z[:, 0].copy()

        results = []
        for solver_cls in (PIQPFlowSolver, PipPIQPFlowSolver):
            solver = solver_cls(net, scale_m, TIGHT)
            dgp = DGP(
                net,
                beta_true,
                solver,
                MarkovSampler(network=net),
                od_uniform_all_pairs(net),
                rng=np.random.default_rng(42),
            )
            data = dgp.sample_dataset(600)
            results.append((data, self._fit(net, solver, data, beta_init=np.ones(2) * 2.0)))

        (data_native, fit_native), (data_oracle, fit_oracle) = results
        np.testing.assert_array_equal(data_native["y"], data_oracle["y"])
        np.testing.assert_array_equal(data_native["b"], data_oracle["b"])
        assert fit_native["diagnostics"]["converged"]
        assert fit_oracle["diagnostics"]["converged"]
        np.testing.assert_allclose(
            fit_native["beta_hat"], fit_oracle["beta_hat"], rtol=0, atol=1e-8
        )
