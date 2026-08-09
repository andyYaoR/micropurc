"""Correctness of the native parallel route sampler (MarkovSampler.sample_routes_batch).

The batched native walk must reproduce the reference per-trip Python sampler in
distribution: E[y] = x* per link, simple paths only, and the same validity
rules. It draws from an independent mt19937_64, so the agreement is statistical;
given the same seeds, its own output is reproducible to the bit.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from micropurc.config import ActiveSetConfig, SolverConfig
from micropurc.core.activeset import _check_dag_connectivity, active_threshold
from micropurc.forward.piqp_solver import PIQPFlowSolver
from micropurc.network import Network
from micropurc.simulation.route_sampler import MarkovSampler

DATA = pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def branchy_case():
    """A validated multi-path OD flow on the EMA network.

    Searches OD pairs until one yields a flow that passes the DAG/connectivity
    check and splits at some node, so the sampler has a real branch to resolve.
    """
    net = Network.from_csv(
        str(DATA / "EMA_net.csv"),
        tail_col="init_node",
        head_col="term_node",
        attribute_cols=["length", "tt", "attr_1"],
    )
    scale_m = net.Z[:, 0].copy()
    solver = PIQPFlowSolver(
        network=net,
        scale_m=scale_m,
        config=SolverConfig(backend="piqp", eps_abs=1e-12, eps_duality_gap_abs=1e-12),
    )
    aset = ActiveSetConfig()
    sampler = MarkovSampler(network=net, active_set_config=aset)
    c = net.Z @ np.array([0.5, 0.5, 0.5])
    V = net.num_nodes
    rng = np.random.default_rng(7)

    for _ in range(500):
        o, d = int(rng.integers(V)), int(rng.integers(V))
        if o == d:
            continue
        b = np.zeros(V)
        b[o], b[d] = -1.0, 1.0
        x = np.maximum(solver.solve(c, b).x, 0.0)
        eps = active_threshold(x, aset)
        x[x < eps] = 0.0
        if np.count_nonzero(x) <= 1:
            continue
        is_dag, is_conn = _check_dag_connectivity(net.tail[x > 0], net.head[x > 0], d)
        if not (is_dag and is_conn):
            continue
        branchy = any(
            np.count_nonzero((net.tail == v) & (x > 0)) >= 2 for v in np.unique(net.tail[x > 0])
        )
        if branchy:
            return net, sampler, o, d, x
    pytest.skip("no branchy validated flow found")


def _route_key(y):
    return tuple(np.flatnonzero(y).tolist())


def test_expected_flow_recovered(branchy_case):
    """Empirical link frequencies equal the flow (E[y] = x*)."""
    net, sampler, o, d, x = branchy_case
    M = 40_000
    seeds = np.random.default_rng(1).integers(0, 2**63 - 1, size=M, dtype=np.uint64)
    Y, valid = sampler.sample_routes_batch(
        x[None, :], np.full(M, o), np.full(M, d), seeds, flow_row=np.zeros(M, dtype=np.int64)
    )

    assert valid.all()  # a validated DAG flow always yields a route
    active = x > 0
    assert not Y[:, ~active].any()  # never lights an inactive link
    ey = Y.mean(axis=0)
    sig = np.sqrt(np.maximum(ey * (1 - ey), 1e-12) / M)
    assert np.all(np.abs(ey[active] - x[active]) <= 5 * sig[active] + 1e-6)


def test_routes_are_simple_paths(branchy_case):
    """Every sampled route is a simple origin->destination path (flow conservation)."""
    net, sampler, o, d, x = branchy_case
    M = 5_000
    seeds = np.random.default_rng(2).integers(0, 2**63 - 1, size=M, dtype=np.uint64)
    Y, valid = sampler.sample_routes_batch(
        x[None, :], np.full(M, o), np.full(M, d), seeds, flow_row=np.zeros(M, dtype=np.int64)
    )
    assert valid.all()
    for key in {_route_key(Y[i]) for i in range(M)}:
        edges = np.array(key, dtype=int)
        tail, head = net.tail[edges], net.head[edges]
        nodes = np.unique(np.concatenate([tail, head]))
        assert len(edges) == len(nodes) - 1  # a chain, no repeated node
        for v in nodes:
            outd, ind = int(np.sum(tail == v)), int(np.sum(head == v))
            if v == o:
                assert (outd, ind) == (1, 0)
            elif v == d:
                assert (outd, ind) == (0, 1)
            else:
                assert (outd, ind) == (1, 1)


def test_matches_python_sampler_distribution(branchy_case):
    """Native path frequencies match the reference Python sampler (small TV)."""
    net, sampler, o, d, x = branchy_case
    M = 20_000
    rng = np.random.default_rng(3)
    seeds = rng.integers(0, 2**63 - 1, size=M, dtype=np.uint64)
    Y, _ = sampler.sample_routes_batch(
        x[None, :], np.full(M, o), np.full(M, d), seeds, flow_row=np.zeros(M, dtype=np.int64)
    )

    from collections import Counter

    nat = Counter(_route_key(Y[i]) for i in range(M))
    py = Counter()
    for _ in range(M):
        py[_route_key(sampler.sample_route(x, o, d, rng).y)] += 1
    keys = set(nat) | set(py)
    tv = 0.5 * sum(abs(nat[k] / M - py[k] / M) for k in keys)
    assert tv < 0.05


def test_deterministic_across_thread_counts(branchy_case):
    """Same seeds -> identical output regardless of worker count."""
    net, sampler, o, d, x = branchy_case
    M = 3_000
    seeds = np.random.default_rng(4).integers(0, 2**63 - 1, size=M, dtype=np.uint64)
    kw = dict(flow_row=np.zeros(M, dtype=np.int64))
    Y1, v1 = sampler.sample_routes_batch(
        x[None, :], np.full(M, o), np.full(M, d), seeds, n_threads=1, **kw
    )
    Y8, v8 = sampler.sample_routes_batch(
        x[None, :], np.full(M, o), np.full(M, d), seeds, n_threads=8, **kw
    )
    assert np.array_equal(Y1, Y8)
    assert np.array_equal(v1, v8)


def test_invalid_when_origin_outflow_not_unit(branchy_case):
    """A flow whose origin outflow != 1 is rejected (valid=False)."""
    net, sampler, o, d, x = branchy_case
    x_bad = x.copy()
    x_bad[(net.tail == o) & (x_bad > 0)] *= 0.5  # origin outflow now ~0.5
    _, valid = sampler.sample_routes_batch(
        x_bad[None, :],
        np.array([o]),
        np.array([d]),
        np.array([1], dtype=np.uint64),
        flow_row=np.array([0]),
    )
    assert not bool(valid[0])


def test_shared_flow_rows_via_flow_row_index(branchy_case):
    """Many trips can point at one flow row through flow_row."""
    net, sampler, o, d, x = branchy_case
    M = 1_000
    seeds = np.random.default_rng(5).integers(0, 2**63 - 1, size=M, dtype=np.uint64)
    Y, valid = sampler.sample_routes_batch(
        x[None, :], np.full(M, o), np.full(M, d), seeds, flow_row=np.zeros(M, dtype=np.int64)
    )  # M trips, 1 flow row
    assert Y.shape == (M, net.num_links)
    assert valid.all()


def test_python_fallback_matches_native_structure(branchy_case):
    """The pure-Python fallback also yields valid routes confined to active links."""
    net, sampler, o, d, x = branchy_case
    M = 200
    seeds = np.random.default_rng(6).integers(0, 2**63 - 1, size=M, dtype=np.uint64)
    Y, valid = sampler.sample_routes_batch(
        x[None, :],
        np.full(M, o),
        np.full(M, d),
        seeds,
        flow_row=np.zeros(M, dtype=np.int64),
        use_native=False,
    )
    assert valid.all()
    active = x > 0
    assert not Y[:, ~active].any()
