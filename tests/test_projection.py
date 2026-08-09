"""Tests for projection identities and Schur vs dense reference equivalence."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

from micropurc.core.projection import TripContribution, trip_contribution_schur
from micropurc.network import Network


def trip_contribution_dense_reference(
    Z_act: np.ndarray,
    A_local: np.ndarray,
    y_act: np.ndarray,
    scale_act: np.ndarray,
    x_hat_act: np.ndarray | None = None,
) -> TripContribution:
    """Dense test oracle for P = I - A'(AA')^{-1}A.

    Production code routes through native C++ and a Schur complement, so this
    independent dense implementation lives only in the tests, where it catches
    projection regressions the native path could otherwise hide.
    """
    inv_mat = A_local @ A_local.T
    solved = scipy.linalg.solve(inv_mat, A_local, assume_a="pos")
    P_proj = -A_local.T @ solved
    np.fill_diagonal(P_proj, P_proj.diagonal() + 1.0)

    common = Z_act.T @ P_proj
    H_i = common @ Z_act
    g_i = (common @ (scale_act * y_act).reshape(-1, 1)).ravel()

    u_i = None
    if x_hat_act is not None:
        u_i = (common @ (scale_act * (y_act - x_hat_act)).reshape(-1, 1)).ravel()

    return TripContribution(H_i=H_i, g_i=g_i, u_i=u_i)


def _random_dag_network(
    V: int, L: int, K: int, rng: np.random.Generator
) -> tuple[Network, np.ndarray, np.ndarray]:
    """Build a random DAG network for testing.

    Every edge runs from a lower to a higher node index, so the graph is acyclic
    by construction. Returns (network, tail, head).
    """
    edges_list = []
    for _ in range(L):
        u = rng.integers(0, V - 1)
        v = rng.integers(u + 1, V)
        edges_list.append((u, v))

    edges = np.array(edges_list, dtype=int)
    actual_L = edges.shape[0]

    row_idx = np.concatenate([edges[:, 0], edges[:, 1]])
    col_idx = np.concatenate([np.arange(actual_L), np.arange(actual_L)])
    data = np.concatenate([np.full(actual_L, -1.0), np.full(actual_L, 1.0)])
    A = sp.csc_matrix((data, (row_idx, col_idx)), shape=(V, actual_L))

    Z = rng.standard_normal((actual_L, K))
    nodes = np.arange(V, dtype=int)

    net = Network(nodes=nodes, edges=edges, Z=Z, A=A)
    return net, edges[:, 0], edges[:, 1]


def _build_test_inputs(
    V: int, L: int, K: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-trip inputs a projection call needs.

    A_local is the incidence matrix grounded by dropping the destination row and
    every all-zero row, which is what makes A_local A_local' invertible. Returns
    (Z_act, A_local, y_act, scale_act, x_hat_act).
    """
    rng = np.random.default_rng(seed)
    net, tail, head = _random_dag_network(V, L, K, rng)

    e = net.num_links
    dest = V - 1

    A_sub = net.A.toarray()
    non_zero_rows = np.where(np.any(A_sub != 0, axis=1))[0]
    rows_to_keep = non_zero_rows[non_zero_rows != dest]

    A_local = A_sub[rows_to_keep, :]
    Z_act = net.Z
    y_act = rng.choice([0.0, 1.0], size=e, p=[0.7, 0.3])
    scale_act = np.abs(rng.standard_normal(e)) + 0.1
    x_hat_act = np.abs(rng.standard_normal(e))

    return Z_act, A_local, y_act, scale_act, x_hat_act


class TestProjectionIdentities:
    """The projection P = I - A'(AA')^{-1}A satisfies the projector identities."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_P_times_At_is_zero(self, seed: int) -> None:
        """PA' = 0: P projects onto the null space of A."""
        Z_act, A_local, _, _, _ = _build_test_inputs(8, 15, 3, seed)
        m, e = A_local.shape

        G = A_local @ A_local.T
        solved = np.linalg.solve(G, A_local)
        P = np.eye(e) - A_local.T @ solved

        result = P @ A_local.T
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_P_idempotent(self, seed: int) -> None:
        """P² = P: P is a projector."""
        Z_act, A_local, _, _, _ = _build_test_inputs(8, 15, 3, seed)
        m, e = A_local.shape

        G = A_local @ A_local.T
        solved = np.linalg.solve(G, A_local)
        P = np.eye(e) - A_local.T @ solved

        np.testing.assert_allclose(P @ P, P, atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_P_symmetric(self, seed: int) -> None:
        """P = P': P is symmetric."""
        Z_act, A_local, _, _, _ = _build_test_inputs(8, 15, 3, seed)
        m, e = A_local.shape

        G = A_local @ A_local.T
        solved = np.linalg.solve(G, A_local)
        P = np.eye(e) - A_local.T @ solved

        np.testing.assert_allclose(P, P.T, atol=1e-10)


class TestSchurVsDense:
    """The Schur complement gives the same H_i, g_i, u_i as the dense reference."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    @pytest.mark.parametrize("K", [1, 3, 6])
    def test_H_g_match(self, seed: int, K: int) -> None:
        """Schur H_i, g_i match the dense reference to ~1e-10."""
        Z_act, A_local, y_act, scale_act, _ = _build_test_inputs(8, 15, K, seed)

        schur = trip_contribution_schur(Z_act, A_local, y_act, scale_act)
        dense = trip_contribution_dense_reference(Z_act, A_local, y_act, scale_act)

        np.testing.assert_allclose(schur.H_i, dense.H_i, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(schur.g_i, dense.g_i, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_u_match(self, seed: int) -> None:
        """Schur u_i matches dense reference when x_hat is provided."""
        Z_act, A_local, y_act, scale_act, x_hat_act = _build_test_inputs(8, 15, 3, seed)

        schur = trip_contribution_schur(Z_act, A_local, y_act, scale_act, x_hat_act=x_hat_act)
        dense = trip_contribution_dense_reference(
            Z_act, A_local, y_act, scale_act, x_hat_act=x_hat_act
        )

        np.testing.assert_allclose(schur.u_i, dense.u_i, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_S_rank_one(self, seed: int) -> None:
        """S_i = u_i u_i' is rank 1."""
        Z_act, A_local, y_act, scale_act, x_hat_act = _build_test_inputs(8, 15, 3, seed)

        schur = trip_contribution_schur(Z_act, A_local, y_act, scale_act, x_hat_act=x_hat_act)

        S_i = np.outer(schur.u_i, schur.u_i)
        rank = np.linalg.matrix_rank(S_i, tol=1e-10)
        assert rank <= 1


class TestPhiDeterministic:
    """Phi depends only on its arguments, so repeated evaluations agree exactly."""

    @staticmethod
    def _make_heterogeneous_grid(rows=4, cols=4, K=2, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        net = Network.grid(rows, cols, K=1, rng_seed=rng_seed)
        L = net.num_links
        Z = np.empty((L, K), dtype=float)
        Z[:, 0] = 1.0 + rng.exponential(scale=0.5, size=L)
        if K >= 2:
            Z[:, 1] = Z[:, 0] / (0.5 + rng.uniform(size=L))
        for k in range(2, K):
            Z[:, k] = rng.standard_normal(size=L) * 0.3 + 0.5
        net.Z = Z
        return net

    def test_phi_deterministic(self) -> None:
        """Two estimator instances built from the same config produce identical Phi."""
        from micropurc.config import EstimationConfig
        from micropurc.core.estimator import MicroPURCEstimator
        from micropurc.demands import od_uniform_all_pairs
        from micropurc.forward.piqp_solver import PIQPFlowSolver
        from micropurc.simulation.dgp import DGP
        from micropurc.simulation.route_sampler import MarkovSampler

        net = self._make_heterogeneous_grid(rows=4, cols=4, K=2)
        scale_m = net.Z[:, 0].copy()
        solver = PIQPFlowSolver(net, scale_m)

        beta = np.array([0.5, 0.5])
        od_dist = od_uniform_all_pairs(net)
        sampler = MarkovSampler(network=net)
        dgp = DGP(net, beta, solver, sampler, od_dist, rng=np.random.default_rng(42))
        data = dgp.sample_dataset(100)

        est1 = MicroPURCEstimator(net, solver, estimation_config=EstimationConfig())
        est2 = MicroPURCEstimator(net, solver, estimation_config=EstimationConfig())

        phi1 = est1.phi(beta, data["y"], data["b"])
        phi2 = est2.phi(beta, data["y"], data["b"])

        np.testing.assert_allclose(phi1, phi2, atol=1e-12)


class TestPseudoinverseOracle:
    """Verify that component-aware grounding matches numpy.linalg.pinv.

    The projector is defined as P = I - A'(AA')^+ A, with (AA')^+ the
    Moore-Penrose pseudoinverse. The implementation reaches the same projector by
    component-aware grounding: drop one node row per connected component so the
    grounded A_tilde has full row rank, then form
    P = I - A_tilde'(A_tilde A_tilde')^{-1} A_tilde with an ordinary inverse.

    These tests show the two agree on graphs with 1-4 connected components,
    cross-checked against both numpy.linalg.pinv and torch.linalg.pinv.
    """

    @staticmethod
    def _build_disconnected_incidence(components, K, rng):
        """Build an incidence matrix with multiple connected components.

        components: list of (n_nodes, list_of_edges) per component.
        Returns: A_full (V x E), Z (E x K), dest, node_offset_per_component.
        """
        node_offset = 0
        rows, cols, vals = [], [], []
        edge_idx = 0
        comp_offsets = []

        for n_nodes, edges in components:
            comp_offsets.append(node_offset)
            for u, v in edges:
                rows.extend([node_offset + u, node_offset + v])
                cols.extend([edge_idx, edge_idx])
                vals.extend([-1.0, 1.0])
                edge_idx += 1
            node_offset += n_nodes

        V = node_offset
        E = edge_idx
        A_full = sp.csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(V, E),
        ).toarray()

        Z = rng.standard_normal((E, K))
        dest = comp_offsets[0] + components[0][0] - 1

        return A_full, Z, V, E, dest, comp_offsets

    @staticmethod
    def _ground_component_aware(A_full, dest, components, comp_offsets):
        """Apply component-aware grounding: drop one row per component."""
        rows_to_drop = set()
        for i, (n_nodes, _) in enumerate(components):
            offset = comp_offsets[i]
            comp_nodes = set(range(offset, offset + n_nodes))
            if dest in comp_nodes:
                rows_to_drop.add(dest)
            else:
                rows_to_drop.add(min(comp_nodes))

        all_rows = set(range(A_full.shape[0]))
        non_zero_rows = set(np.where(np.any(A_full != 0, axis=1))[0])
        kept = sorted(non_zero_rows - rows_to_drop)
        return A_full[kept, :]

    @staticmethod
    def _projector_from_pinv(A_full):
        """P = I - A' (AA')^+ A using numpy pseudoinverse."""
        AAt = A_full @ A_full.T
        AAt_pinv = np.linalg.pinv(AAt)
        E = A_full.shape[1]
        return np.eye(E) - A_full.T @ AAt_pinv @ A_full

    @staticmethod
    def _projector_from_grounded(A_grounded):
        """P = I - A_tilde' (A_tilde A_tilde')^{-1} A_tilde (ordinary inverse)."""
        G = A_grounded @ A_grounded.T
        E = A_grounded.shape[1]
        solved = np.linalg.solve(G, A_grounded)
        return np.eye(E) - A_grounded.T @ solved

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_single_component_matches_pinv(self, seed):
        """Connected graph: grounded inverse == pseudoinverse."""
        rng = np.random.default_rng(seed)
        components = [(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 4)])]
        A, Z, V, E, dest, offsets = self._build_disconnected_incidence(
            components,
            K=3,
            rng=rng,
        )
        A_grounded = self._ground_component_aware(A, dest, components, offsets)
        P_pinv = self._projector_from_pinv(A)
        P_grounded = self._projector_from_grounded(A_grounded)

        np.testing.assert_allclose(
            P_grounded, P_pinv, atol=1e-10, err_msg="1-component: grounded != pinv"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_two_components_matches_pinv(self, seed):
        """Two disconnected components: grounded inverse == pseudoinverse."""
        rng = np.random.default_rng(seed)
        components = [
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
            (3, [(0, 1), (1, 2)]),
        ]
        A, Z, V, E, dest, offsets = self._build_disconnected_incidence(
            components,
            K=3,
            rng=rng,
        )
        A_grounded = self._ground_component_aware(A, dest, components, offsets)
        P_pinv = self._projector_from_pinv(A)
        P_grounded = self._projector_from_grounded(A_grounded)

        np.testing.assert_allclose(
            P_grounded, P_pinv, atol=1e-10, err_msg="2-component: grounded != pinv"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_four_components_matches_pinv(self, seed):
        """Four disconnected components: grounded inverse == pseudoinverse."""
        rng = np.random.default_rng(seed)
        components = [
            (3, [(0, 1), (1, 2)]),
            (2, [(0, 1)]),
            (3, [(0, 1), (0, 2), (1, 2)]),
            (2, [(0, 1)]),
        ]
        A, Z, V, E, dest, offsets = self._build_disconnected_incidence(
            components,
            K=3,
            rng=rng,
        )
        A_grounded = self._ground_component_aware(A, dest, components, offsets)
        P_pinv = self._projector_from_pinv(A)
        P_grounded = self._projector_from_grounded(A_grounded)

        np.testing.assert_allclose(
            P_grounded, P_pinv, atol=1e-10, err_msg="4-component: grounded != pinv"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_H_g_match_pinv_oracle(self, seed):
        """H_i and g_i from Schur (grounded) match pinv oracle on disconnected graph."""
        rng = np.random.default_rng(seed)
        components = [
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
            (3, [(0, 1), (1, 2)]),
        ]
        A, Z, V, E, dest, offsets = self._build_disconnected_incidence(
            components,
            K=3,
            rng=rng,
        )
        y_act = rng.choice([0.0, 1.0], size=E, p=[0.5, 0.5])
        scale_act = np.abs(rng.standard_normal(E)) + 0.1
        x_hat_act = np.abs(rng.standard_normal(E))

        # Oracle: full pinv projector
        P = self._projector_from_pinv(A)
        w_y = scale_act * y_act
        H_oracle = Z.T @ P @ Z
        g_oracle = Z.T @ P @ w_y

        # Under test: Schur with component-aware grounding
        A_grounded = self._ground_component_aware(A, dest, components, offsets)
        contrib = trip_contribution_schur(Z, A_grounded, y_act, scale_act)

        np.testing.assert_allclose(
            contrib.H_i, H_oracle, atol=1e-10, err_msg="H_i doesn't match pinv oracle"
        )
        np.testing.assert_allclose(
            contrib.g_i, g_oracle, atol=1e-10, err_msg="g_i doesn't match pinv oracle"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_grounded_P_identities(self, seed):
        """Grounded projector on multi-component graph satisfies P²=P, P=P', PA'=0."""
        rng = np.random.default_rng(seed)
        components = [
            (4, [(0, 1), (1, 2), (2, 3)]),
            (3, [(0, 1), (1, 2)]),
            (2, [(0, 1)]),
        ]
        A, _, V, E, dest, offsets = self._build_disconnected_incidence(
            components,
            K=2,
            rng=rng,
        )
        A_grounded = self._ground_component_aware(A, dest, components, offsets)
        P = self._projector_from_grounded(A_grounded)

        np.testing.assert_allclose(P @ P, P, atol=1e-10, err_msg="P not idempotent")
        np.testing.assert_allclose(P, P.T, atol=1e-10, err_msg="P not symmetric")
        np.testing.assert_allclose(
            P @ A.T, 0, atol=1e-10, err_msg="PA' != 0 (P doesn't null-project A)"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_torch_pinv_matches(self, seed):
        """Cross-check: torch.linalg.pinv agrees with the grounded projector.

        A second independent pseudoinverse implementation, so a shared bug in
        numpy's cannot make the grounding look correct.
        """
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(seed)
        components = [
            (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
            (3, [(0, 1), (1, 2)]),
        ]
        A, Z, V, E, dest, offsets = self._build_disconnected_incidence(
            components,
            K=3,
            rng=rng,
        )
        A_grounded = self._ground_component_aware(A, dest, components, offsets)
        P_grounded = self._projector_from_grounded(A_grounded)

        AAt_t = torch.from_numpy(A @ A.T)
        AAt_pinv_t = torch.linalg.pinv(AAt_t).numpy()
        P_torch = np.eye(E) - A.T @ AAt_pinv_t @ A

        np.testing.assert_allclose(
            P_grounded, P_torch, atol=1e-10, err_msg="grounded != torch.linalg.pinv"
        )
