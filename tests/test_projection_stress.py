"""Stress tests for the projection pipeline.

Covers four sources of numerical fragility:
  1. Conditioning of G = ÃÃ' -- near-singular Gram matrices
  2. Active-set stability under beta perturbation
  3. H_i eigenvalue spectrum
  4. Schur vs pinv oracle agreement on real trips

Several of these report distributions to stdout and assert only the properties
that must hold on any network (G positive definite, H_i PSD, H_N SPD).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

DATA_DIR = Path(__file__).resolve().parent / "data"

from micropurc.config import ActiveSetConfig
from micropurc.core.activeset import determine_active_set
from micropurc.core.projection import TripContribution, trip_contribution_schur
from micropurc.network import Network

# Helpers


def trip_contribution_dense_reference(
    Z_act: np.ndarray,
    A_local: np.ndarray,
    y_act: np.ndarray,
    scale_act: np.ndarray,
    x_hat_act: np.ndarray | None = None,
) -> TripContribution:
    """Dense test oracle for P = I - A'(AA')^{-1}A."""
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


def _load_ema():
    data_dir = DATA_DIR
    if not (data_dir / "EMA_net.csv").exists():
        pytest.skip("EMA data files not found")

    from micropurc.config import SolverConfig
    from micropurc.forward.piqp_solver import PIQPFlowSolver

    net = Network.from_csv(
        str(data_dir / "EMA_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1", "attr_2"],
    )
    scale_m = net.Z[:, 0].copy()
    solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
    return net, solver, scale_m


def _solve_trips(net, solver, beta, n_trips=100, seed=42):
    """Sample n_trips and re-solve each one, returning (flow, y, b, dest) per trip."""
    from micropurc.demands import od_from_csv
    from micropurc.simulation.dgp import DGP
    from micropurc.simulation.route_sampler import MarkovSampler

    data_dir = DATA_DIR
    od_dist = od_from_csv(net, str(data_dir / "EMA_trips.csv"))
    sampler = MarkovSampler(network=net)
    dgp = DGP(net, beta, solver, sampler, od_dist, rng=np.random.default_rng(seed))
    data = dgp.sample_dataset(n_trips)

    c = net.Z @ beta
    flows = []
    for n in range(n_trips):
        result = solver.solve(c, data["b"][n])
        dest = int(np.argmax(data["b"][n]))
        flows.append((result, data["y"][n], data["b"][n], dest))
    return flows


# 1. Gram matrix conditioning


class TestGramConditioning:
    """Measure cond(G = ÃÃ') across trips and detect near-singular cases."""

    @pytest.fixture(scope="class")
    @staticmethod
    def ema_trips():
        net, solver, scale_m = _load_ema()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        trips = _solve_trips(net, solver, beta, n_trips=100, seed=42)
        return net, scale_m, trips

    def test_condition_number_distribution(self, ema_trips):
        """Report the cond(G) distribution and require G to be SPD on every trip."""
        net, scale_m, trips = ema_trips
        cfg = ActiveSetConfig()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        c = net.Z @ beta

        cond_numbers = []
        min_eigenvalues = []
        aset_sizes = []

        for result, y_i, b_i, dest in trips:
            aset = determine_active_set(
                result.x,
                dest,
                net,
                cfg,
                projection_mode="robust_pinv",
                lam_eq=result.lam_eq,
                c=c,
                scale_m=scale_m,
            )
            if not aset.valid:
                continue

            A_sub = net.A[:, aset.indices]
            A_local = (
                A_sub[aset.kept_node_rows, :].toarray()
                if sp.issparse(A_sub)
                else A_sub[aset.kept_node_rows, :]
            )

            G = A_local @ A_local.T
            eigvals = np.linalg.eigvalsh(G)
            lam_min = eigvals[0]
            lam_max = eigvals[-1]

            if lam_min > 0:
                cond_numbers.append(lam_max / lam_min)
            else:
                cond_numbers.append(np.inf)
            min_eigenvalues.append(lam_min)
            aset_sizes.append(len(aset.indices))

        conds = np.array(cond_numbers)
        mins = np.array(min_eigenvalues)

        print(f"\n  Gram matrix G = ÃÃ' condition numbers (N={len(conds)} valid trips):")
        print(f"    median cond: {np.median(conds):.2f}")
        print(f"    max cond: {np.max(conds):.2f}")
        print(f"    min λ_min: {np.min(mins):.2e}")
        print(f"    median λ_min: {np.median(mins):.2e}")
        print(f"    trips with cond > 1e6: {np.sum(conds > 1e6)}")
        print(f"    trips with cond > 1e12: {np.sum(conds > 1e12)}")
        print(
            f"    active set sizes: min={np.min(aset_sizes)} "
            f"median={np.median(aset_sizes):.0f} max={np.max(aset_sizes)}"
        )

        # With correct grounding, G should always be SPD.
        assert np.all(mins > 0), f"G has non-positive eigenvalue: min λ_min = {np.min(mins):.2e}"

    def test_near_singular_G_schur_vs_dense(self, ema_trips):
        """For trips with high cond(G), verify Schur matches dense reference."""
        net, scale_m, trips = ema_trips
        cfg = ActiveSetConfig()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        c = net.Z @ beta

        worst_cond = 0.0
        worst_H_diff = 0.0
        tested = 0

        for result, y_i, b_i, dest in trips:
            aset = determine_active_set(
                result.x,
                dest,
                net,
                cfg,
                projection_mode="robust_pinv",
                lam_eq=result.lam_eq,
                c=c,
                scale_m=scale_m,
            )
            if not aset.valid:
                continue

            A_sub = net.A[:, aset.indices]
            A_local = (
                A_sub[aset.kept_node_rows, :].toarray()
                if sp.issparse(A_sub)
                else A_sub[aset.kept_node_rows, :]
            )

            G = A_local @ A_local.T
            cond = np.linalg.cond(G)

            if cond > 100:
                Z_act = net.Z[aset.indices, :]
                y_act = y_i[aset.indices]
                s_act = scale_m[aset.indices]
                x_act = result.x[aset.indices]

                schur = trip_contribution_schur(Z_act, A_local, y_act, s_act, x_act)
                dense = trip_contribution_dense_reference(Z_act, A_local, y_act, s_act, x_act)

                H_diff = np.max(np.abs(schur.H_i - dense.H_i))
                g_diff = np.max(np.abs(schur.g_i - dense.g_i))

                if cond > worst_cond:
                    worst_cond = cond
                    worst_H_diff = H_diff

                # Both paths solve the same SPD system, so the achievable agreement
                # is set by cond(G); the tolerance grows with it accordingly.
                tol = max(1e-8, 1e-14 * cond)
                np.testing.assert_allclose(
                    schur.H_i,
                    dense.H_i,
                    atol=tol,
                    err_msg=f"Schur != dense at cond(G) = {cond:.2e}",
                )
                np.testing.assert_allclose(
                    schur.g_i,
                    dense.g_i,
                    atol=tol,
                    err_msg=f"Schur g_i != dense at cond(G) = {cond:.2e}",
                )
                tested += 1

        print(f"\n  Tested {tested} trips with cond(G) > 100")
        print(f"  Worst cond(G): {worst_cond:.2e}")
        print(f"  Worst |H_schur - H_dense|: {worst_H_diff:.2e}")


# 2. Active-set stability


class TestActiveSetStability:
    """Probe active-set stability under small β perturbations.

    The fixed-point argument needs the active set to be locally constant in β.
    If small perturbations flip many links, the fixed-point map is discontinuous
    there and convergence is at risk.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ema_setup():
        net, solver, scale_m = _load_ema()
        return net, solver, scale_m

    def test_beta_perturbation_active_set_stability(self, ema_setup):
        """Perturb β by small amounts and measure active-set churn."""
        net, solver, scale_m = ema_setup
        beta_0 = np.array([0.5, 0.5, 0.5, 0.5])
        K = len(beta_0)
        cfg = ActiveSetConfig()

        from micropurc.demands import od_from_csv
        from micropurc.simulation.dgp import DGP
        from micropurc.simulation.route_sampler import MarkovSampler

        data_dir = DATA_DIR
        od_dist = od_from_csv(net, str(data_dir / "EMA_trips.csv"))
        sampler = MarkovSampler(network=net)
        dgp = DGP(net, beta_0, solver, sampler, od_dist, rng=np.random.default_rng(42))
        data = dgp.sample_dataset(50)

        c_0 = net.Z @ beta_0
        perturbation_sizes = [1e-6, 1e-4, 1e-2, 1e-1]
        rng = np.random.default_rng(123)

        print("\n  Active-set stability under β perturbation:")

        for eps in perturbation_sizes:
            n_changed = 0
            n_links_flipped = 0
            n_tested = 0
            n_valid_base = 0

            direction = rng.standard_normal(K)
            direction /= np.linalg.norm(direction)
            beta_pert = beta_0 + eps * direction
            c_pert = net.Z @ beta_pert

            for n in range(len(data["b"])):
                dest = int(np.argmax(data["b"][n]))

                result_0 = solver.solve(c_0, data["b"][n])
                aset_0 = determine_active_set(
                    result_0.x,
                    dest,
                    net,
                    cfg,
                    projection_mode="robust_pinv",
                    lam_eq=result_0.lam_eq,
                    c=c_0,
                    scale_m=scale_m,
                )
                if not aset_0.valid:
                    continue
                n_valid_base += 1

                result_p = solver.solve(c_pert, data["b"][n])
                aset_p = determine_active_set(
                    result_p.x,
                    dest,
                    net,
                    cfg,
                    projection_mode="robust_pinv",
                    lam_eq=result_p.lam_eq,
                    c=c_pert,
                    scale_m=scale_m,
                )

                set_0 = set(aset_0.indices)
                set_p = set(aset_p.indices) if aset_p.valid else set()

                if set_0 != set_p:
                    n_changed += 1
                    n_links_flipped += len(set_0.symmetric_difference(set_p))

                n_tested += 1

            frac = n_changed / max(n_tested, 1)
            avg_flipped = n_links_flipped / max(n_changed, 1)
            print(
                f"    ε={eps:.0e}: {n_changed}/{n_tested} trips changed "
                f"({100 * frac:.1f}%), avg {avg_flipped:.1f} links flipped"
            )

        # At ε=1e-6 only a handful of trips should change active set, which is the
        # local constancy the fixed-point argument relies on. How many flip depends
        # on the network, so the counts are printed and left unasserted.

    def test_near_kink_links(self, ema_setup):
        """Report the share of active links whose dual slack sits near zero.

        Links with small |slack| are the most likely to flip active/inactive
        across iterations, driving non-convergence. The share is network- and
        draw-specific, so it is printed and no threshold is asserted.
        """
        net, solver, scale_m = ema_setup
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        c = net.Z @ beta

        from micropurc.demands import od_from_csv
        from micropurc.simulation.dgp import DGP
        from micropurc.simulation.route_sampler import MarkovSampler

        data_dir = DATA_DIR
        od_dist = od_from_csv(net, str(data_dir / "EMA_trips.csv"))
        sampler = MarkovSampler(network=net)
        dgp = DGP(net, beta, solver, sampler, od_dist, rng=np.random.default_rng(42))
        data = dgp.sample_dataset(100)

        near_zero_count = 0
        total_active = 0
        slack_magnitudes = []

        for n in range(len(data["b"])):
            dest = int(np.argmax(data["b"][n]))
            result = solver.solve(c, data["b"][n])
            if result.lam_eq is None:
                continue

            slack = c + (net.A.T @ result.lam_eq)
            active = slack < 0
            total_active += np.sum(active)

            for i in np.where(active)[0]:
                mag = abs(slack[i])
                slack_magnitudes.append(mag)
                if mag < 1e-4:
                    near_zero_count += 1

        mags = np.array(slack_magnitudes)
        print(f"\n  Near-kink link analysis (100 trips, {total_active} active links):")
        print(
            f"    Links with |slack| < 1e-4: "
            f"{near_zero_count} ({100 * near_zero_count / max(total_active, 1):.1f}%)"
        )
        print(
            f"    |slack| distribution: "
            f"p10={np.percentile(mags, 10):.2e} "
            f"p25={np.percentile(mags, 25):.2e} "
            f"p50={np.percentile(mags, 50):.2e} "
            f"p75={np.percentile(mags, 75):.2e} "
            f"p90={np.percentile(mags, 90):.2e}"
        )


# 3. H_i eigenvalue spectrum


class TestHiSpectrum:
    """Analyze the per-trip H_i eigenvalue spectrum.

    Near-zero eigenvalues in individual H_i can accumulate into a
    near-singular H_N, causing the fixed-point solve to blow up.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ema_trips():
        net, solver, scale_m = _load_ema()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        trips = _solve_trips(net, solver, beta, n_trips=100, seed=42)
        return net, scale_m, trips

    def test_Hi_eigenvalues_nonneg(self, ema_trips):
        """Each H_i must be positive semidefinite (P is a projector)."""
        net, scale_m, trips = ema_trips
        cfg = ActiveSetConfig()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        c = net.Z @ beta

        min_eigval_seen = np.inf
        negative_count = 0

        for result, y_i, b_i, dest in trips:
            aset = determine_active_set(
                result.x,
                dest,
                net,
                cfg,
                projection_mode="robust_pinv",
                lam_eq=result.lam_eq,
                c=c,
                scale_m=scale_m,
            )
            if not aset.valid:
                continue

            A_sub = net.A[:, aset.indices]
            A_local = (
                A_sub[aset.kept_node_rows, :].toarray()
                if sp.issparse(A_sub)
                else A_sub[aset.kept_node_rows, :]
            )

            Z_act = net.Z[aset.indices, :]
            y_act = y_i[aset.indices]
            s_act = scale_m[aset.indices]

            contrib = trip_contribution_schur(Z_act, A_local, y_act, s_act)
            eigvals = np.linalg.eigvalsh(contrib.H_i)
            min_eigval_seen = min(min_eigval_seen, eigvals[0])
            if eigvals[0] < -1e-10:
                negative_count += 1

        print("\n  H_i eigenvalue spectrum:")
        print(f"    Smallest λ_min across all trips: {min_eigval_seen:.2e}")
        print(f"    Trips with negative eigenvalue: {negative_count}")

        assert negative_count == 0, (
            f"{negative_count} trips have H_i with negative eigenvalue "
            f"(min = {min_eigval_seen:.2e})"
        )

    def test_HN_conditioning(self, ema_trips):
        """Aggregate H_N = Σ H_i must be SPD; its condition number is reported."""
        net, scale_m, trips = ema_trips
        cfg = ActiveSetConfig()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        c = net.Z @ beta
        K = net.num_attributes

        H_total = np.zeros((K, K))
        n_valid = 0

        for result, y_i, b_i, dest in trips:
            aset = determine_active_set(
                result.x,
                dest,
                net,
                cfg,
                projection_mode="robust_pinv",
                lam_eq=result.lam_eq,
                c=c,
                scale_m=scale_m,
            )
            if not aset.valid:
                continue

            A_sub = net.A[:, aset.indices]
            A_local = (
                A_sub[aset.kept_node_rows, :].toarray()
                if sp.issparse(A_sub)
                else A_sub[aset.kept_node_rows, :]
            )

            Z_act = net.Z[aset.indices, :]
            y_act = y_i[aset.indices]
            s_act = scale_m[aset.indices]

            contrib = trip_contribution_schur(Z_act, A_local, y_act, s_act)
            H_total += contrib.H_i
            n_valid += 1

        eigvals = np.linalg.eigvalsh(H_total)
        cond = eigvals[-1] / max(eigvals[0], 1e-30)

        print(f"\n  Aggregate H_N = Σ H_i ({n_valid} trips):")
        print(f"    eigenvalues: {eigvals}")
        print(f"    cond(H_N): {cond:.2e}")
        print(f"    H_N SPD: {np.all(eigvals > 0)}")

        assert np.all(eigvals > 0), f"H_N is not SPD: eigenvalues = {eigvals}"


# 4. Schur vs pinv oracle on real trips


class TestSchurVsPinvOnRealTrips:
    """Verify Schur projection matches numpy.linalg.pinv on actual EMA trips.

    The inputs here are flow solutions from a real QP solve, so the active sets
    and incidence blocks carry the numerical noise of an interior-point run,
    which the synthetic fixtures in TestPseudoinverseOracle cannot produce.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ema_trips():
        net, solver, scale_m = _load_ema()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        trips = _solve_trips(net, solver, beta, n_trips=100, seed=42)
        return net, scale_m, trips

    def test_schur_matches_pinv_on_all_valid_trips(self, ema_trips):
        """On every valid trip, the Schur H_i and g_i must match the pinv oracle."""
        net, scale_m, trips = ema_trips
        cfg = ActiveSetConfig()
        beta = np.array([0.5, 0.5, 0.5, 0.5])
        c = net.Z @ beta

        worst_diff = 0.0
        tested = 0

        for result, y_i, b_i, dest in trips:
            aset = determine_active_set(
                result.x,
                dest,
                net,
                cfg,
                projection_mode="robust_pinv",
                lam_eq=result.lam_eq,
                c=c,
                scale_m=scale_m,
            )
            if not aset.valid:
                continue

            A_sub = net.A[:, aset.indices]
            A_local = (
                A_sub[aset.kept_node_rows, :].toarray()
                if sp.issparse(A_sub)
                else A_sub[aset.kept_node_rows, :]
            )

            # Full incidence on active edges (before grounding)
            A_full = (
                net.A[:, aset.indices].toarray() if sp.issparse(net.A) else net.A[:, aset.indices]
            )

            Z_act = net.Z[aset.indices, :]
            y_act = y_i[aset.indices]
            s_act = scale_m[aset.indices]

            # Under test: the Schur path
            contrib = trip_contribution_schur(Z_act, A_local, y_act, s_act)

            # Grounded oracle for the range-space projector. Do NOT "simplify"
            # this back to np.linalg.pinv, of either A or A A': on a trip whose
            # active subgraph sits near the boundary, A has singular values
            # right at pinv's rcond cutoff, so which side of the cutoff they
            # land on depends on the BLAS backend. The reference then stops
            # being symmetric and disagrees across platforms, while the Schur
            # path under test is bit-identical everywhere.
            #
            # Ground instead, the way the package realizes the pseudoinverse.
            # Each column of A holds one -1 (tail) and one +1 (head), so within
            # every connected component of the active subgraph the rows sum to
            # zero exactly -- the entries are 0 and +-1, so that cancellation is
            # exact in floating point. Dropping one row per component therefore
            # leaves the row space, and hence the projector, mathematically
            # unchanged, while making A_g full row rank: A_g A_g' is SPD and the
            # projector follows from a plain SPD solve, with no cutoff anywhere.
            E = A_full.shape[1]
            incident = sp.csr_matrix(A_full != 0)
            n_components, labels = connected_components(incident @ incident.T, directed=False)
            keep_rows = np.ones(A_full.shape[0], dtype=bool)
            for component in range(n_components):
                keep_rows[np.flatnonzero(labels == component)[0]] = False
            A_g = A_full[keep_rows, :]
            G_g = A_g @ A_g.T
            # A miscounted component would leave A_g rank deficient and yield a
            # silently wrong oracle; the Cholesky turns that into a loud failure.
            scipy.linalg.cholesky(G_g)
            P_pinv = np.eye(E) - A_g.T @ np.linalg.solve(G_g, A_g)

            w_y = s_act * y_act
            H_oracle = Z_act.T @ P_pinv @ Z_act
            g_oracle = Z_act.T @ P_pinv @ w_y

            H_diff = np.max(np.abs(contrib.H_i - H_oracle))
            g_diff = np.max(np.abs(contrib.g_i - g_oracle))
            worst_diff = max(worst_diff, H_diff, g_diff)

            np.testing.assert_allclose(
                contrib.H_i,
                H_oracle,
                atol=1e-8,
                err_msg=f"Schur H_i != pinv oracle (diff={H_diff:.2e})",
            )
            np.testing.assert_allclose(
                contrib.g_i,
                g_oracle,
                atol=1e-8,
                err_msg=f"Schur g_i != pinv oracle (diff={g_diff:.2e})",
            )
            tested += 1

        print(f"\n  Schur vs pinv oracle on {tested} real EMA trips:")
        print(f"    Worst |H_schur - H_pinv|: {worst_diff:.2e}")
