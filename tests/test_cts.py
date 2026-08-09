"""Certified Tiered Solve: convention-awareness and answer preservation.

CTS is only sound if its dual-slack certificate tests the true KKT quantity
c_j + (A'y)_j. The tail/head arrays' sign convention differs across network
builders (Network.from_csv/grid: A[tail] = -1; builders that label edges the
other way round: A[tail] = +1), so the certificate detects the convention at
run time. These tests exercise tier 1 on both conventions and require (a) the
certificate actually fires (pass > 0 at a repeated beta) and (b) accumulated
H, g match the CTS-off reference to solver precision. A sign-flipped
certificate fails (a) on
one convention and risks silent wrong answers on the other; either way these
tests catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from micropurc import MicroPURCEstimator, PIQPFlowSolver
from micropurc.config import ProjectionConfig, SolverConfig
from micropurc.core.lazy_design import LazyDesign
from micropurc.network import Network


def _lazy_design(net: Network, n_trips: int) -> LazyDesign:
    k = net.num_attributes
    return LazyDesign(
        link_attrs=net.Z[None, :, :],
        variant_attr=np.array([0]),
        variant_covariates=np.zeros((1, 1)),
        trip_variant=np.zeros(n_trips, dtype=np.int64),
        term_code=np.zeros(k, dtype=np.int64),
        term_a=np.arange(k, dtype=np.int64),
        term_b=np.zeros(k, dtype=np.int64),
        term_cov=np.zeros(k, dtype=np.int64),
        term_op=np.zeros(k, dtype=np.int64),
        term_thr=np.zeros(k, dtype=np.float64),
    )


def _trips(net: Network):
    """A handful of OD pairs on the grid with 2-link observed routes."""
    v = net.num_nodes
    origins = np.array([0, 1, 0, 3], dtype=np.int64)
    dests = np.array([v - 1, v - 1, v - 2, v - 1], dtype=np.int64)
    rows, cols = [0, 0, 1, 2, 3, 3], [0, 1, 2, 0, 1, 3]
    y = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(origins), net.num_links))
    return y, origins, dests


def _accumulate_twice(net, scale_m, beta, y, origins, dests, cts: bool):
    """Two same-beta accumulates; the second exercises tier 1 when cts=True."""
    solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
    est = MicroPURCEstimator(net, solver, projection_config=ProjectionConfig(mode="robust_pinv"))
    ctx = est._get_native_context()
    ctx.cts_enabled = cts
    lazy = _lazy_design(net, origins.shape[0])
    est._accumulate(beta, y, None, origins=origins, dests=dests, lazy_design=lazy)
    H, g, _, dropped = est._accumulate(
        beta, y, None, origins=origins, dests=dests, lazy_design=lazy
    )
    return H, g, dropped, dict(est._last_native_stats)


class TestCTSConventions:
    def test_standard_convention_pass_and_match(self):
        """Network.grid convention (A[tail] = -1): tier 1 must fire and agree."""
        net = Network.grid(4, 4, K=2)
        scale_m = np.ones(net.num_links)
        beta = np.array([0.5, 0.2])
        y, o, d = _trips(net)

        H1, g1, dr1, s1 = _accumulate_twice(net, scale_m, beta, y, o, d, cts=True)
        H0, g0, dr0, _ = _accumulate_twice(net, scale_m, beta, y, o, d, cts=False)

        assert bool(s1["cts_convention_ok"]), "grid must be a supported convention"
        assert s1["cts_sign"] == -1.0, "from_csv/grid convention is A[tail] = -1"
        assert s1["cts_attempts"] > 0
        assert s1["cts_pass"] == s1["cts_attempts"], (
            f"certificate must pass at a repeated beta: {s1['cts_pass']}"
            f"/{s1['cts_attempts']} (a sign-flipped slack scan makes this 0)"
        )
        np.testing.assert_allclose(H1, H0, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(g1, g0, atol=1e-9, rtol=1e-9)
        assert dr1 == dr0

    def test_swapped_labels_pass_and_match(self):
        """Swapped edge labels (tail/head swapped vs A): sigma=+1 branch."""
        base = Network.grid(4, 4, K=2)
        net = Network(
            nodes=base.nodes,
            edges=base.edges[:, ::-1].copy(),  # swap labels; A unchanged
            Z=base.Z,
            A=base.A,
        )
        scale_m = np.ones(net.num_links)
        beta = np.array([0.5, 0.2])
        y, o, d = _trips(net)

        H1, g1, dr1, s1 = _accumulate_twice(net, scale_m, beta, y, o, d, cts=True)
        H0, g0, dr0, _ = _accumulate_twice(net, scale_m, beta, y, o, d, cts=False)

        assert bool(s1["cts_convention_ok"])
        assert s1["cts_sign"] == 1.0, "swapped labels give A[tail] = +1"
        assert s1["cts_attempts"] > 0
        assert s1["cts_pass"] == s1["cts_attempts"]
        np.testing.assert_allclose(H1, H0, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(g1, g0, atol=1e-9, rtol=1e-9)
        assert dr1 == dr0

    def test_unsupported_convention_falls_back(self):
        """A mixed-sign incidence must disable CTS, never certify."""
        base = Network.grid(4, 4, K=2)
        A = base.A.tolil()
        A[:, 0] = -A[:, 0].toarray()  # one column flipped -> mixed convention
        net = Network(nodes=base.nodes, edges=base.edges, Z=base.Z, A=A.tocsc())
        scale_m = np.ones(net.num_links)
        beta = np.array([0.5, 0.2])
        y, o, d = _trips(net)

        H1, g1, dr1, s1 = _accumulate_twice(net, scale_m, beta, y, o, d, cts=True)
        H0, g0, dr0, _ = _accumulate_twice(net, scale_m, beta, y, o, d, cts=False)

        assert not bool(s1["cts_convention_ok"])
        assert s1["cts_attempts"] == 0, "tier 1 must never run on mixed conventions"
        np.testing.assert_allclose(H1, H0, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(g1, g0, atol=1e-9, rtol=1e-9)
        assert dr1 == dr0

    def test_perturbed_beta_stays_sound(self):
        """After a beta step, certified accepts must still match the IPM."""
        net = Network.grid(4, 4, K=2)
        scale_m = np.ones(net.num_links)
        y, o, d = _trips(net)
        b0 = np.array([0.5, 0.2])
        b1 = b0 * 1.05

        solver = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est = MicroPURCEstimator(
            net, solver, projection_config=ProjectionConfig(mode="robust_pinv")
        )
        est._get_native_context().cts_enabled = True
        lazy = _lazy_design(net, o.shape[0])
        est._accumulate(b0, y, None, origins=o, dests=d, lazy_design=lazy)
        H1, g1, _, _ = est._accumulate(b1, y, None, origins=o, dests=d, lazy_design=lazy)

        solver0 = PIQPFlowSolver(net, scale_m, SolverConfig(backend="piqp"))
        est0 = MicroPURCEstimator(
            net, solver0, projection_config=ProjectionConfig(mode="robust_pinv")
        )
        est0._get_native_context().cts_enabled = False
        est0._accumulate(b0, y, None, origins=o, dests=d, lazy_design=lazy)
        H0, g0, _, _ = est0._accumulate(b1, y, None, origins=o, dests=d, lazy_design=lazy)

        np.testing.assert_allclose(H1, H0, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(g1, g0, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
