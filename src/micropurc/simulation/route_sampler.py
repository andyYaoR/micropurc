r"""Route sampling from optimal link flows via Markovian decomposition.

The forward QP returns a link flow :math:`x^{\star}` for a unit demand, whose
mass in general spreads over several origin-destination paths. This module
realises the corresponding path distribution: a walk that leaves each node along
an edge drawn in proportion to its flow reproduces :math:`x^{\star}` in
expectation, so a sampled route is an observation the model could have produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ActiveSetConfig
from ..core.activeset import active_threshold
from ..network import Network


@dataclass
class SampledRoute:
    """Result of sampling a route from optimal flows.

    Attributes:
        y: Link indicator vector, shape (L,). y[e]=1 iff edge e is on the path.
        nodes: Ordered node path from origin to destination.
        valid: Whether the sampling succeeded. When False, ``y`` is all zeros
            and ``nodes`` is empty.
    """

    y: np.ndarray
    nodes: list[int]
    valid: bool


class MarkovSampler:
    r"""Markovian route sampler with :math:`\mathbb{E}[y] = x^{\star}` per link.

    At node :math:`v` the walk leaves along outgoing edge :math:`e` with
    probability

    .. math::

        p(e \mid v) = \frac{x_e}{\sum_{e' \in \mathrm{out}(v)} x_{e'}} ,

    and follows that policy from origin to destination. Flow conservation at the
    interior nodes is what makes the per-link expectation come back to
    :math:`x^{\star}`.
    """

    def __init__(
        self,
        network: Network,
        active_set_config: ActiveSetConfig | None = None,
    ) -> None:
        """Initializes the sampler.

        Args:
            network: Network whose ``tail``/``head`` arrays define the walk.
            active_set_config: Supplies the threshold below which a link counts
                as carrying no flow, so numerical dust cannot be walked on.
        """
        self.network = network
        self.aset_cfg = active_set_config or ActiveSetConfig()
        self._tail = network.tail.astype(int)
        self._head = network.head.astype(int)

    def sample_route(
        self,
        x_star: np.ndarray,
        origin: int,
        destination: int,
        rng: np.random.Generator | None = None,
    ) -> SampledRoute:
        """Samples one random route consistent with the flow vector.

        The flow is clamped at zero and links below the configured activity
        threshold are dropped before the walk starts. Three conditions make the
        result invalid: an origin whose remaining outflow misses one unit, a node
        the walk reaches with no outgoing flow, and a revisit, since only simple
        paths are accepted. The step budget ``2 * (V + L)`` bounds the walk on a
        flow that carries a cycle.

        Args:
            x_star: Optimal link flows, shape (L,).
            origin: Origin node index.
            destination: Destination node index.
            rng: Generator for the per-node edge draws; defaults to a fresh
                unseeded generator.

        Returns:
            SampledRoute with the link indicator and node path.
        """
        if rng is None:
            rng = np.random.default_rng()

        L = self.network.num_links
        V = self.network.num_nodes
        x = np.maximum(np.asarray(x_star, dtype=float).ravel(), 0.0)

        eps = active_threshold(x, self.aset_cfg)
        x[x < eps] = 0.0

        _invalid = SampledRoute(y=np.zeros(L, dtype=float), nodes=[], valid=False)

        out_origin = np.where(self._tail == origin)[0]
        total_out = float(np.sum(x[out_origin]))
        if not np.isclose(total_out, 1.0, atol=1e-5, rtol=1e-3):
            return _invalid

        outgoing: list[list[int]] = [[] for _ in range(V)]
        for e_idx in range(L):
            if x[e_idx] > 0.0:
                outgoing[self._tail[e_idx]].append(e_idx)

        if not outgoing[origin]:
            return _invalid

        node_path = [origin]
        y = np.zeros(L, dtype=float)
        visited = {origin}
        current = origin
        max_steps = 2 * (V + L)

        for _ in range(max_steps):
            if current == destination:
                break

            out_edges = outgoing[current]
            if not out_edges:
                return _invalid

            flows = np.array([x[e] for e in out_edges], dtype=float)
            total_flow = float(np.sum(flows))
            if total_flow <= 0.0:
                return _invalid

            probs = flows / total_flow
            choice_idx = rng.choice(len(out_edges), p=probs)
            e_chosen = out_edges[choice_idx]

            y[e_chosen] = 1.0
            next_node = self._head[e_chosen]

            if next_node in visited and next_node != destination:
                return _invalid

            node_path.append(next_node)
            visited.add(next_node)
            current = next_node

        if current != destination:
            return _invalid

        return SampledRoute(y=y, nodes=node_path, valid=True)

    def sample_routes_batch(
        self,
        flows: np.ndarray,
        origins: np.ndarray,
        dests: np.ndarray,
        seeds: np.ndarray | None = None,
        *,
        flow_row: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        n_threads: int = -1,
        use_native: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Samples many routes in parallel, one per trip.

        Each trip ``i`` draws a flow-proportional simple path from ``origins[i]``
        to ``dests[i]`` on the flow vector ``flows[flow_row[i]]``, seeded by
        ``seeds[i]``. This is the batched, multi-threaded counterpart of
        :meth:`sample_route`: same transition law
        (:math:`\mathbb{E}[y] = x^{\star}` per link), same activity threshold,
        and the same validity rules (unit origin outflow, simple paths only).

        Each route is drawn from its own ``mt19937_64``. The two methods
        therefore agree in distribution, and a given trip's realised route
        differs between them. Fixing ``seeds`` makes this method reproducible on
        its own terms, for any thread count and any scheduling order.

        Args:
            flows: Link-flow vectors, shape (M, L). Rows need not be distinct;
                use ``flow_row`` to point several trips at one row.
            origins: Origin node indices, shape (N,).
            dests: Destination node indices, shape (N,).
            seeds: Per-trip RNG seeds, shape (N,), dtype uint64. If ``None``,
                seeds are drawn from ``rng`` (or a default generator).
            flow_row: Index into ``flows`` per trip, shape (N,). Defaults to
                ``arange(N)`` (requires ``M == N``).
            rng: Generator used to draw ``seeds`` when they are not supplied.
            n_threads: Worker threads; ``-1`` uses all hardware threads.
            use_native: If ``False``, fall back to a serial Python loop over
                :meth:`sample_route` (for environments without the extension).

        Returns:
            Tuple ``(Y, valid)`` with ``Y`` of shape (N, L) float64 (link
            indicators) and ``valid`` of shape (N,) bool.

        Raises:
            ValueError: If array shapes are inconsistent.
        """
        flows = np.ascontiguousarray(flows, dtype=np.float64)
        if flows.ndim != 2:
            raise ValueError("flows must be 2-D (M, L)")
        m_rows, n_links = flows.shape
        if n_links != self.network.num_links:
            raise ValueError(f"flows has L={n_links}, network has L={self.network.num_links}")

        origins = np.ascontiguousarray(origins, dtype=np.int64).ravel()
        dests = np.ascontiguousarray(dests, dtype=np.int64).ravel()
        n_trips = origins.shape[0]
        if dests.shape[0] != n_trips:
            raise ValueError("origins and dests must have the same length")

        if flow_row is None:
            if m_rows != n_trips:
                raise ValueError("flow_row is required when flows has M != N rows")
            flow_row = np.arange(n_trips, dtype=np.int64)
        else:
            flow_row = np.ascontiguousarray(flow_row, dtype=np.int64).ravel()
            if flow_row.shape[0] != n_trips:
                raise ValueError("flow_row must have length N == len(origins)")

        if seeds is None:
            gen = rng or np.random.default_rng()
            seeds = gen.integers(0, 2**63 - 1, size=n_trips, dtype=np.uint64)
        else:
            seeds = np.ascontiguousarray(seeds, dtype=np.uint64).ravel()
            if seeds.shape[0] != n_trips:
                raise ValueError("seeds must have length N == len(origins)")

        if not use_native:
            return self._sample_routes_batch_py(flows, origins, dests, seeds, flow_row)

        from .._native import batch_sample_routes

        Y, valid = batch_sample_routes(
            tail=np.ascontiguousarray(self._tail, dtype=np.int64),
            head=np.ascontiguousarray(self._head, dtype=np.int64),
            V=int(self.network.num_nodes),
            flows=flows,
            flow_row=flow_row,
            origins=origins,
            dests=dests,
            seeds=seeds,
            threshold_factor=float(self.aset_cfg.threshold_factor),
            threshold_min=float(self.aset_cfg.threshold_min),
            n_threads=int(n_threads),
        )
        return Y, valid.astype(bool)

    def _sample_routes_batch_py(
        self,
        flows: np.ndarray,
        origins: np.ndarray,
        dests: np.ndarray,
        seeds: np.ndarray,
        flow_row: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Serial Python fallback for :meth:`sample_routes_batch`."""
        n_trips = origins.shape[0]
        L = self.network.num_links
        Y = np.zeros((n_trips, L), dtype=np.float64)
        valid = np.zeros(n_trips, dtype=bool)
        for i in range(n_trips):
            route = self.sample_route(
                flows[flow_row[i]],
                int(origins[i]),
                int(dests[i]),
                np.random.default_rng(int(seeds[i])),
            )
            Y[i] = route.y
            valid[i] = route.valid
        return Y, valid
