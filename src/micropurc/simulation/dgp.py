r"""Data generating process for microPURC simulation experiments.

One trip is produced in two steps: the forward QP is solved at
:math:`\beta_{\mathrm{true}}` for a sampled OD pair, yielding the optimal link
flow :math:`x^{\star}`, and a route is then drawn from :math:`x^{\star}` by the
Markovian flow decomposition of
:class:`~micropurc.simulation.route_sampler.MarkovSampler`.

Every candidate OD passes the same acceptance filters the estimator applies, so
a generated dataset contains no trip the estimator would later drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from ..core.activeset import ActiveSetConfig, determine_active_set
from ..demands import ODDistribution
from ..forward.base import ForwardSolver
from ..network import Network
from .route_sampler import MarkovSampler


@dataclass
class DGPConfig:
    """Configuration for the data generating process.

    Attributes:
        max_tries_per_trip: OD draws allowed per requested trip; the total
            attempt budget is ``max_tries_per_trip * n_trips``. It also caps how
            often the batch path re-seeds a walk that failed to complete.
        require_multiple_paths: Accept an OD only when its active subgraph
            carries at least two simple origin-destination paths, so every
            observation records a genuine route choice.
        batch_sampling: Draw the route walks with the multi-threaded native
            batch sampler. False selects the serial per-trip walk, which
            reproduces datasets generated before the batch sampler existed
            bit for bit.
        sampler_threads: Worker threads for the batch sampler; -1 uses all
            hardware threads.
    """

    max_tries_per_trip: int = 20
    require_multiple_paths: bool = True
    batch_sampling: bool = True
    sampler_threads: int = -1


class DGP:
    """Data generating process for microPURC simulation.

    Samples datasets whose observations satisfy the acceptance filters the
    estimator applies to real data: a dual-slack active set with more than one
    link, an acyclic active subgraph, and a grounding with at least one retained
    node row. Two further filters are specific to generation and controlled by
    :class:`DGPConfig`: the multi-path requirement and the unit origin outflow
    that guarantees the route walk terminates.
    """

    def __init__(
        self,
        network: Network,
        beta_true: np.ndarray,
        forward_solver: ForwardSolver,
        route_sampler: MarkovSampler,
        od_dist: ODDistribution,
        rng: np.random.Generator | None = None,
        config: DGPConfig | None = None,
        active_set_config: ActiveSetConfig | None = None,
    ) -> None:
        r"""Initializes the DGP.

        Args:
            network: Network the trips are generated on.
            beta_true: True parameter vector :math:`\beta_{\mathrm{true}}`,
                shape (K,); link costs are :math:`c = Z \beta_{\mathrm{true}}`.
            forward_solver: Solver for the per-OD forward QP.
            route_sampler: Sampler that decomposes a link flow into a route.
            od_dist: Distribution the candidate OD pairs are drawn from.
            rng: Generator for the OD draws, the route walks, and the batch
                sampler seeds; defaults to a fresh unseeded generator.
            config: Sampling and acceptance settings.
            active_set_config: Dual-slack tolerance for the active-set test.

        Raises:
            ValueError: If ``beta_true`` length does not match the attribute
                count K.
        """
        self.network = network
        self.beta_true = np.asarray(beta_true, dtype=float).ravel()
        self.forward_solver = forward_solver
        self.route_sampler = route_sampler
        self.od_dist = od_dist
        self.rng = rng or np.random.default_rng()
        self.config = config or DGPConfig()
        self.aset_cfg = active_set_config or ActiveSetConfig()

        K = network.num_attributes
        if self.beta_true.shape[0] != K:
            raise ValueError(f"beta_true has length {self.beta_true.shape[0]}, but Z has K={K}.")

        self.c_true = network.Z @ self.beta_true
        # Perturbation scale of F(x) = 0.5 * sum_j scale_j * x_j^2, taken as the
        # link-length column Z[:, 0] by model convention. determine_active_set
        # rejects a call without it, and passes it straight through to the native
        # layer, which ignores it: the active set is decided by the sign of the
        # dual slack alone.
        self.scale_m = np.ascontiguousarray(network.Z[:, 0], dtype=float)

    def _build_b(self, origin: int, destination: int) -> np.ndarray:
        """Builds the node-imbalance vector of a single OD pair.

        Sign convention: one unit of demand leaves the origin (``-1``) and
        arrives at the destination (``+1``); every other node is balanced.
        """
        V = self.network.num_nodes
        b = np.zeros(V, dtype=float)
        b[origin] = -1.0
        b[destination] = 1.0
        return b

    def _active_flow(self, x: np.ndarray, lam: np.ndarray, dest: int) -> np.ndarray | None:
        r"""Restricts a flow to its dual-slack active set and validates it.

        Applies the KKT sign test the estimator uses: link :math:`j` is active
        when its dual slack

        .. math::

            s_j = c_j + (A^{\top}\lambda)_j

        falls below the cutoff, which is zero and tightens to ``-slack_tol`` when
        a positive tolerance is configured. Generated data and the estimator
        therefore agree on which links carry flow. ``determine_active_set`` also
        runs the acyclicity and component-grounding checks; a flow that fails any
        of them, or that leaves at most one active link, is rejected (``None``).

        The returned flow keeps the active entries clipped at zero and zeroes
        everything else, so the route walk sees exactly the active subgraph.
        """
        aset = determine_active_set(
            x,
            int(dest),
            self.network,
            self.aset_cfg,
            lam_eq=lam,
            c=self.c_true,
            scale_m=self.scale_m,
        )
        if not aset.valid or aset.indices.size <= 1:
            return None
        x_clean = np.zeros_like(x, dtype=float)
        x_clean[aset.indices] = np.maximum(x[aset.indices], 0.0)
        return x_clean

    def _has_multiple_paths(
        self,
        x_clean: np.ndarray,
        origin: int,
        dest: int,
    ) -> bool:
        """Reports whether the active subgraph offers a genuine route choice.

        True when at least two simple origin-destination paths exist. The
        enumeration stops at the second path, so the cost does not grow with the
        (possibly exponential) number of paths.
        """
        active = x_clean > 0.0
        if not np.any(active):
            return False
        tail = self.network.tail[active]
        head = self.network.head[active]
        G = nx.DiGraph()
        G.add_nodes_from(np.unique(np.concatenate([tail, head])))
        G.add_edges_from(zip(tail, head))
        if origin not in G or dest not in G:
            return False
        count = 0
        for _ in nx.all_simple_paths(G, origin, dest):
            count += 1
            if count >= 2:
                return True
        return False

    def _origin_outflow_feasible(self, x_clean: np.ndarray, origin: int) -> bool:
        """Reports whether the origin's active outflow sums to one unit.

        This is the deterministic feasibility test the route walk applies at the
        origin, with the same tolerance. Enforcing it when the flow is accepted
        means every accepted flow admits a valid walk, so the batch sampler has
        no systematic rejections to recover from.
        """
        total = float(np.sum(x_clean[self.network.tail == origin]))
        return bool(np.isclose(total, 1.0, atol=1e-5, rtol=1e-3))

    def sample_dataset(self, n_trips: int) -> dict[str, Any]:
        """Samples a dataset of ``n_trips`` accepted observations.

        ``DGPConfig.batch_sampling`` selects between the multi-threaded native
        batch sampler and the serial per-trip walk. Both draw ODs from
        ``od_dist`` and apply the same acceptance filters, so they agree in
        distribution. A ``RuntimeError`` is raised when the attempt budget
        ``max_tries_per_trip * n_trips`` runs out before ``n_trips`` trips have
        been accepted.

        Args:
            n_trips: Number of accepted trips to generate.

        Returns:
            Dict with keys: y (N,L), Z (L,K), b (N,V), origins (N,), destinations (N,).
        """
        if self.config.batch_sampling:
            return self._sample_dataset_batch(n_trips)
        return self._sample_dataset_sequential(n_trips)

    def _solve_one(self, o: int, d: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Solves one OD forward QP and returns (primal flow, node potentials).

        The dual-slack active set needs the equality duals ``lam_eq``, which the
        multi-threaded ``solve_batch`` entry point does not return, so acceptance
        solves each unique OD on its own. Flows are cached per OD and data
        generation sits outside the estimator's timed path, so the serial cost is
        paid once per distinct OD. Returns ``None`` when the solve fails.
        """
        try:
            res = self.forward_solver.solve(self.c_true, self._build_b(o, d))
        except Exception:  # noqa: BLE001 -- a failed OD solve is a rejected draw
            return None
        return np.asarray(res.x, dtype=float), np.asarray(res.lam_eq, dtype=float)

    def _classify_flow(
        self, x_raw: np.ndarray, lam: np.ndarray, o: int, d: int
    ) -> np.ndarray | None:
        """Applies every acceptance filter to a raw flow; returns it or None.

        Order: dual-slack active set (with the acyclicity and grounding checks),
        the optional multi-path requirement, then the unit origin outflow.
        """
        x_clean = self._active_flow(x_raw, lam, d)
        if x_clean is None:
            return None
        if self.config.require_multiple_paths and not self._has_multiple_paths(x_clean, o, d):
            return None
        if not self._origin_outflow_feasible(x_clean, o):
            return None
        return x_clean

    def _accept_flows(
        self, n_trips: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Draws ODs and accepts their flows, leaving the route walks to callers.

        Candidate ODs are drawn in rounds of the size still needed. Within a
        round, each not-yet-classified unique OD is solved and filtered once; the
        outcome is memoised in ``flow_cache`` or ``failed_od``, so a repeated OD
        costs one dictionary lookup. Trips are then accepted in the original draw
        order, which makes the dataset an i.i.d. draw from ``od_dist``
        conditioned on the OD being acceptable.

        Every accepted flow is acyclic and origin-feasible, which is what lets
        the caller treat a failed walk as a rare stochastic miss.

        Returns ``(B, origins, dests, flow_row, unique_flows, attempts)`` where
        ``unique_flows`` has one row per distinct accepted OD flow and
        ``flow_row[i]`` indexes it for trip ``i``.
        """
        V = self.network.num_nodes
        B = np.zeros((n_trips, V), dtype=float)
        origins = np.zeros(n_trips, dtype=np.int64)
        dests = np.zeros(n_trips, dtype=np.int64)
        flow_row = np.zeros(n_trips, dtype=np.int64)
        unique_flows: list[np.ndarray] = []
        od_to_row: dict[tuple[int, int], int] = {}

        max_attempts = self.config.max_tries_per_trip * n_trips
        valid_count = 0
        attempts = 0
        flow_cache: dict[tuple[int, int], np.ndarray] = {}
        failed_od: set[tuple[int, int]] = set()

        while valid_count < n_trips and attempts < max_attempts:
            # 1. Draw a round of candidate ODs (as many as still needed).
            remaining = n_trips - valid_count
            chunk: list[tuple[int, int]] = []
            while len(chunk) < remaining and attempts < max_attempts:
                attempts += 1
                o_arr, d_arr = self.od_dist.sample(self.rng, 1)
                chunk.append((int(o_arr[0]), int(d_arr[0])))

            # 2. Collect the in-bounds ODs of this round whose verdict is still
            #    unknown, deduplicated against each other and against the cache.
            to_solve: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for o, d in chunk:
                if o == d or not (0 <= o < V) or not (0 <= d < V):
                    continue
                if (o, d) in flow_cache or (o, d) in failed_od or (o, d) in seen:
                    continue
                seen.add((o, d))
                to_solve.append((o, d))

            for o, d in to_solve:
                solved = self._solve_one(o, d)
                if solved is None:
                    failed_od.add((o, d))
                    continue
                x_raw, lam = solved
                x_clean = self._classify_flow(x_raw, lam, o, d)
                if x_clean is None:
                    failed_od.add((o, d))
                    continue
                flow_cache[(o, d)] = x_clean

            # 3. Accept trips in draw order, assigning each distinct OD one row
            #    of unique_flows.
            for o, d in chunk:
                if valid_count >= n_trips:
                    break
                if (o, d) not in flow_cache:
                    continue
                if (o, d) not in od_to_row:
                    od_to_row[(o, d)] = len(unique_flows)
                    unique_flows.append(flow_cache[(o, d)])
                B[valid_count] = self._build_b(o, d)
                origins[valid_count] = o
                dests[valid_count] = d
                flow_row[valid_count] = od_to_row[(o, d)]
                valid_count += 1

        if valid_count < n_trips:
            raise RuntimeError(
                f"Only {valid_count}/{n_trips} valid trips after {attempts} attempts."
            )

        flows = np.ascontiguousarray(np.array(unique_flows), dtype=np.float64)
        return B, origins, dests, flow_row, flows, attempts

    def _sample_dataset_batch(self, n_trips: int) -> dict[str, Any]:
        """Accepts flows, then draws every route walk with the native sampler."""
        B, origins, dests, flow_row, flows, _ = self._accept_flows(n_trips)

        seeds = self.rng.integers(0, 2**63 - 1, size=n_trips, dtype=np.uint64)
        Y, valid = self.route_sampler.sample_routes_batch(
            flows,
            origins,
            dests,
            seeds,
            flow_row=flow_row,
            n_threads=self.config.sampler_threads,
        )

        # Acceptance already guarantees an acyclic, origin-feasible flow, so a
        # failed walk is a stochastic miss: retry it with a fresh seed. Exhausting
        # the retries means the guarantee itself broke, which is an error.
        tries = 0
        remaining = np.flatnonzero(~valid)
        while remaining.size and tries < self.config.max_tries_per_trip:
            tries += 1
            new_seeds = self.rng.integers(0, 2**63 - 1, size=remaining.size, dtype=np.uint64)
            Yr, vr = self.route_sampler.sample_routes_batch(
                flows,
                origins[remaining],
                dests[remaining],
                new_seeds,
                flow_row=flow_row[remaining],
                n_threads=self.config.sampler_threads,
            )
            ok = np.flatnonzero(vr)
            Y[remaining[ok]] = Yr[ok]
            remaining = remaining[~vr]
        if remaining.size:
            raise RuntimeError(f"{remaining.size} accepted flow(s) failed to yield a valid walk.")

        return {
            "y": Y,
            "Z": self.network.Z,
            "b": B,
            "origins": origins,
            "destinations": dests,
        }

    def _sample_dataset_sequential(self, n_trips: int) -> dict[str, Any]:
        """Samples the dataset one trip at a time, solve and walk interleaved.

        This is the draw order that datasets generated before the batch sampler
        used, so a seeded rerun reproduces them bit for bit.
        """
        Z = self.network.Z
        L, K = Z.shape
        V = self.network.num_nodes

        Y = np.zeros((n_trips, L), dtype=float)
        B = np.zeros((n_trips, V), dtype=float)
        origins = np.zeros(n_trips, dtype=int)
        dests = np.zeros(n_trips, dtype=int)

        max_attempts = self.config.max_tries_per_trip * n_trips
        valid_count = 0
        attempts = 0
        flow_cache: dict[tuple[int, int], np.ndarray] = {}
        failed_od: set[tuple[int, int]] = set()

        while valid_count < n_trips and attempts < max_attempts:
            attempts += 1

            o_arr, d_arr = self.od_dist.sample(self.rng, 1)
            o, d = int(o_arr[0]), int(d_arr[0])

            if (o, d) in failed_od or o == d:
                continue
            if not (0 <= o < V) or not (0 <= d < V):
                continue

            b_n = self._build_b(o, d)

            if (o, d) in flow_cache:
                x_clean = flow_cache[(o, d)]
            else:
                solved = self._solve_one(o, d)
                if solved is None:
                    failed_od.add((o, d))
                    continue
                x_raw, lam = solved
                x_clean = self._classify_flow(x_raw, lam, o, d)
                if x_clean is None:
                    failed_od.add((o, d))
                    continue
                flow_cache[(o, d)] = x_clean

            route = self.route_sampler.sample_route(x_clean, o, d, self.rng)
            if not route.valid:
                continue

            Y[valid_count] = route.y
            B[valid_count] = b_n
            origins[valid_count] = o
            dests[valid_count] = d
            valid_count += 1

        if valid_count < n_trips:
            raise RuntimeError(
                f"Only {valid_count}/{n_trips} valid trips after {attempts} attempts."
            )

        return {
            "y": Y,
            "Z": Z,
            "b": B,
            "origins": origins,
            "destinations": dests,
        }

    def _sample_valid_route(self, x_star, origin, dest, rng, tries=20):
        """Samples one route consistent with a flow, retrying stochastic misses.

        Accepted flows are acyclic and origin-feasible (see ``_accept_flows``), so
        a walk almost always succeeds and a miss is retried on a fresh draw.
        Exhausting ``tries`` raises ``RuntimeError``.
        """
        for _ in range(tries):
            route = self.route_sampler.sample_route(x_star, origin, dest, rng)
            if route.valid:
                return route.y
        raise RuntimeError(f"no valid walk for OD ({origin}, {dest}) in {tries} tries.")

    def sample_clustered_dataset(
        self,
        n_travelers: int,
        trips_per_traveler,
        rho,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        r"""Samples a clustered dataset with repeated trips per traveler.

        Tastes are homogeneous: every trip uses :math:`\beta_{\mathrm{true}}` and
        the choice model is held fixed throughout. Each of the :math:`H`
        travelers is a commuter with a single fixed OD drawn i.i.d. from
        ``od_dist``, so several travelers may share an OD and the traveler
        clusters are nested within OD clusters. For traveler :math:`h`:

        1. draw a habitual route :math:`R_h` once from the
           :math:`\beta_{\mathrm{true}}` flow of its OD;
        2. generate :math:`T_h` trips, where trip :math:`i` repeats :math:`R_h`
           with probability :math:`\rho_h` and otherwise takes a fresh
           independent draw from the same flow.

        Write :math:`P` for the route law induced by the
        :math:`\beta_{\mathrm{true}}` flow. Since :math:`R_h \sim P` and each
        fresh draw is also :math:`P`, the marginal law of trip :math:`(h, i)` is

        .. math::

            \rho_h P + (1 - \rho_h) P = P ,

        so the mean estimate stays consistent for
        :math:`\beta_{\mathrm{true}}` at every :math:`\rho`. What :math:`\rho`
        moves is the within-traveler dependence, and through it the standard
        errors. At :math:`\rho = 0` the trips are i.i.d. route draws grouped by
        traveler; at :math:`\rho = 1` all :math:`T_h` trips of a traveler are the
        same route, so the traveler contributes one independent observation.

        Fresh routes are drawn on demand, only for trips whose uniform draw
        satisfies :math:`U_{h,i} \ge \rho_h`. The consumption of ``rng`` is
        therefore a function of the realised draws, and a rerun with the same
        seed and the same :math:`\rho` reproduces the dataset exactly.

        Args:
            n_travelers: Number of travelers (clusters), H.
            trips_per_traveler: Trips per traveler T; int (balanced) or array (H,).
            rho: Habit strength in [0, 1]; float (common) or array (H,).
            rng: Generator for the route and Bernoulli draws (defaults to
                ``self.rng``).

        Returns:
            Dict with keys y (N, L), Z (L, K), b (N, V), origins (N,),
            destinations (N,), traveler_id (N,), od_id (N,), where
            ``N = sum(T_h)``. ``traveler_id`` in [0, H); ``od_id`` indexes distinct
            ODs for OD-level clustering. Trips are ordered by traveler.

        Raises:
            ValueError: If ``trips_per_traveler`` or ``rho`` is neither a scalar
                nor a length-H array, if any ``trips_per_traveler`` < 1, or if any
                ``rho`` lies outside [0, 1].
        """
        rng = rng or self.rng
        H = int(n_travelers)

        T_arr = (
            np.full(H, int(trips_per_traveler), dtype=int)
            if np.isscalar(trips_per_traveler)
            else np.asarray(trips_per_traveler, dtype=int)
        )
        rho_arr = (
            np.full(H, float(rho), dtype=float)
            if np.isscalar(rho)
            else np.asarray(rho, dtype=float)
        )
        if T_arr.shape != (H,) or rho_arr.shape != (H,):
            raise ValueError("trips_per_traveler and rho must be scalars or length-H arrays.")
        if np.any(T_arr < 1):
            raise ValueError("trips_per_traveler must be >= 1.")
        if np.any((rho_arr < 0.0) | (rho_arr > 1.0)):
            raise ValueError("rho must lie in [0, 1].")

        # One accepted OD, and its beta_true flow, per traveler: drawn i.i.d. from
        # od_dist through the same acceptance filters as the i.i.d. sampler.
        B, origins, dests, flow_row, flows, _ = self._accept_flows(H)

        L = self.network.num_links
        V = self.network.num_nodes
        N = int(T_arr.sum())

        Y = np.zeros((N, L), dtype=float)
        b_out = np.zeros((N, V), dtype=float)
        o_out = np.zeros(N, dtype=np.int64)
        d_out = np.zeros(N, dtype=np.int64)
        traveler_id = np.zeros(N, dtype=np.int64)
        od_id = np.zeros(N, dtype=np.int64)

        row = 0
        for h in range(H):
            o_h, d_h = int(origins[h]), int(dests[h])
            x_h = flows[flow_row[h]]
            T_h = int(T_arr[h])
            rho_h = float(rho_arr[h])

            habitual = self._sample_valid_route(x_h, o_h, d_h, rng)
            u = rng.random(T_h)

            for i in range(T_h):
                # Repeat the habitual route with probability rho_h; the fresh draw
                # is evaluated lazily, so rho = 1 consumes no further route draws.
                Y[row] = habitual if u[i] < rho_h else self._sample_valid_route(x_h, o_h, d_h, rng)
                b_out[row] = B[h]
                o_out[row] = o_h
                d_out[row] = d_h
                traveler_id[row] = h
                od_id[row] = int(flow_row[h])  # travelers on one OD share this id
                row += 1

        return {
            "y": Y,
            "Z": self.network.Z,
            "b": b_out,
            "origins": o_out,
            "destinations": d_out,
            "traveler_id": traveler_id,
            "od_id": od_id,
        }

    def add_utility_noise(
        self,
        c: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        r"""Adds i.i.d. Gaussian noise to a cost vector.

        Produces :math:`\tilde c = c + \varepsilon` with
        :math:`\varepsilon_j \sim \mathcal{N}(0, \sigma^2)`. Whenever
        :math:`K < L`, such a draw almost surely leaves the column space of
        :math:`Z`, so no :math:`\beta` reproduces the perturbed costs; this is
        the misspecification lever for robustness experiments.

        Args:
            c: Clean cost vector, shape (L,).
            sigma: Standard deviation of the noise.

        Returns:
            Noisy cost vector, shape (L,).
        """
        return c + self.rng.normal(0, sigma, size=c.shape)
