"""Load archived runs and analyze them: predicted flows, active subnetworks.

The loader is the read side of :class:`~micropurc.reporting.runlog.RunWriter`.
Given a run directory it reconstructs the configuration, the dataset, and —
when the network was snapshotted — a fully functional forward model, so that
estimates can be interrogated without re-running the estimation:

Example:
    >>> from micropurc.reporting import load_run
    >>> run = load_run("runs/ema_ab~03")
    >>> run.beta_hat
    array([0.501, 0.497, 0.503])
    >>> flows = run.predicted_flow(od_pairs=[(3, 17)])       # forward QP at beta_hat
    >>> sub = run.active_subnetwork((3, 17))                  # links carrying flow
    >>> cmp = run.compare_routes(trips=range(10))             # predicted vs observed

The network snapshot convention (see ``snapshot_data``): arrays named
``edges``, ``Z``, ``A`` (sparse), ``scale_m``, and optionally
``attribute_names`` reconstruct the :class:`~micropurc.network.Network`;
``y`` (sparse), ``origins``, ``dests`` reconstruct the observations.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from ..config import ActiveSetConfig, SolverConfig
from ..core.activeset import ActiveSet, determine_active_set
from ..forward.piqp_solver import PIQPFlowSolver
from ..network import Network


def _load_npz(path: Path) -> dict[str, Any]:
    """Loads an npz into a plain dict, reconstructing ``__csr_`` sparse groups.

    Args:
        path: npz file written by the run writer.

    Returns:
        Dict of arrays; sparse matrices are rebuilt as ``csr_matrix``.
    """
    raw = dict(np.load(path, allow_pickle=False))
    out: dict[str, Any] = {}
    sparse_bases = {k.split("__csr_")[0] for k in raw if "__csr_" in k}
    for base in sparse_bases:
        out[base] = sp.csr_matrix(
            (
                raw[f"{base}__csr_data"],
                raw[f"{base}__csr_indices"],
                raw[f"{base}__csr_indptr"],
            ),
            shape=tuple(raw[f"{base}__csr_shape"]),
        )
    for k, v in raw.items():
        if "__csr_" not in k:
            out[k] = v
    return out


@dataclass
class RunArchive:
    """An archived run loaded back into memory.

    Attributes:
        run_dir: Directory the run was loaded from.
        config: Parsed ``config.json``.
        environment: Parsed ``environment.json`` (machine/software info).
        compute: Parsed ``compute.json`` (wall time, peak RSS, ...), if the
            run was finalized.
        result: Arrays from ``result.npz`` (``beta_hat``, ``se_hat``, ...).
        data: Snapshotted inputs from ``data.npz`` (empty when the run used a
            hash reference — see ``data_ref``).
        data_ref: Parsed ``data_ref.json`` when the inputs were too large to
            copy; contains SHA-256 hashes and source paths for verification.
    """

    run_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    compute: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    data_ref: dict[str, Any] = field(default_factory=dict)

    # -- basic accessors ---------------------------------------------------

    @property
    def beta_hat(self) -> np.ndarray:
        """Point estimates from ``result.npz``.

        Returns:
            The ``beta_hat`` array.

        Raises:
            KeyError: If the run has no stored ``beta_hat``.
        """
        if "beta_hat" not in self.result:
            raise KeyError(f"{self.run_dir} has no beta_hat in result.npz")
        return np.asarray(self.result["beta_hat"], dtype=float)

    @property
    def se_hat(self) -> np.ndarray | None:
        """Standard errors from ``result.npz`` (None when absent).

        Returns:
            The ``se_hat`` array or None.
        """
        se = self.result.get("se_hat")
        return None if se is None else np.asarray(se, dtype=float)

    def records(self) -> dict[str, dict[str, Any]]:
        """Loads every per-task record of a batch study.

        Returns:
            Mapping from record name to its arrays.
        """
        rec_dir = self.run_dir / "records"
        if not rec_dir.is_dir():
            return {}
        return {p.stem: _load_npz(p) for p in sorted(rec_dir.glob("*.npz"))}

    # -- model reconstruction ----------------------------------------------

    def network(self) -> Network:
        """Rebuilds the :class:`Network` from the data snapshot.

        Returns:
            The snapshotted network.

        Raises:
            KeyError: If the snapshot lacks the network arrays (``edges``,
                ``Z``, ``A``).
        """
        for key in ("edges", "Z", "A"):
            if key not in self.data:
                raise KeyError(
                    f"{self.run_dir} data snapshot has no '{key}'; snapshot the "
                    "network (edges, Z, A, scale_m) to enable analysis helpers"
                )
        edges = np.asarray(self.data["edges"], dtype=np.int64)
        names = self.data.get("attribute_names")
        return Network(
            nodes=np.arange(int(edges.max()) + 1),
            edges=edges,
            Z=np.asarray(self.data["Z"], dtype=float),
            A=sp.csc_matrix(self.data["A"]),
            attribute_names=[str(s) for s in names] if names is not None else None,
        )

    def solver(self) -> PIQPFlowSolver:
        """Rebuilds the forward solver with the snapshotted perturbation scale.

        Returns:
            A PIQP forward solver matching the archived run.

        Raises:
            KeyError: If the snapshot lacks ``scale_m``.
        """
        if "scale_m" not in self.data:
            raise KeyError(f"{self.run_dir} data snapshot has no 'scale_m'")
        return PIQPFlowSolver(
            self.network(),
            np.asarray(self.data["scale_m"], dtype=float),
            SolverConfig(backend="piqp", eps_abs=1e-12, eps_duality_gap_abs=1e-12),
        )

    # -- analysis helpers ----------------------------------------------------

    def _resolve_od(
        self,
        od_pairs: Iterable[tuple[int, int]] | None,
        trips: Iterable[int] | None,
    ) -> list[tuple[int, int]]:
        """Resolves OD pairs from explicit pairs or trip indices.

        Args:
            od_pairs: Explicit (origin, dest) node pairs.
            trips: Trip indices into the snapshotted ``origins``/``dests``.

        Returns:
            De-duplicated list of (origin, dest) pairs, in input order.

        Raises:
            ValueError: If neither/both arguments are given, or trips are
                requested without snapshotted origins/dests.
        """
        if (od_pairs is None) == (trips is None):
            raise ValueError("pass exactly one of od_pairs= or trips=")
        if trips is not None:
            if "origins" not in self.data or "dests" not in self.data:
                raise ValueError("run has no snapshotted origins/dests")
            o = np.asarray(self.data["origins"], dtype=int)
            d = np.asarray(self.data["dests"], dtype=int)
            od_pairs = [(int(o[t]), int(d[t])) for t in trips]
        seen: dict[tuple[int, int], None] = {}
        for od in od_pairs:  # type: ignore[union-attr]
            seen.setdefault((int(od[0]), int(od[1])), None)
        return list(seen)

    def predicted_flow(
        self,
        od_pairs: Iterable[tuple[int, int]] | None = None,
        *,
        trips: Iterable[int] | None = None,
        beta: np.ndarray | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Solves the forward QP at ``beta`` for selected OD pairs.

        Args:
            od_pairs: (origin, dest) node pairs to solve.
            trips: Alternatively, trip indices whose OD pairs are used.
            beta: Parameters; defaults to the run's ``beta_hat``.

        Returns:
            Mapping ``(origin, dest) -> x`` with the optimal link-flow vector
            (shape ``(L,)``) of each pair.

        Example:
            >>> flows = run.predicted_flow([(3, 17)])
            >>> flows[(3, 17)].sum()
        """
        net = self.network()
        solver = self.solver()
        b_vec = np.zeros(net.num_nodes)
        beta = self.beta_hat if beta is None else np.asarray(beta, dtype=float)
        c = net.Z @ beta
        out: dict[tuple[int, int], np.ndarray] = {}
        for o, d in self._resolve_od(od_pairs, trips):
            b_vec[:] = 0.0
            b_vec[o], b_vec[d] = -1.0, 1.0
            out[(o, d)] = solver.solve(c, b_vec).x
        return out

    def active_subnetwork(
        self,
        od: tuple[int, int],
        *,
        beta: np.ndarray | None = None,
        slack_tol: float = 0.0,
    ) -> dict[str, Any]:
        r"""Extracts the active subnetwork of one OD pair at ``beta``.

        Active links are identified with the dual-slack KKT test the estimator
        uses: link :math:`j` is active when
        :math:`s_j = c_j + (A^{\top}\lambda)_j` falls below the cutoff, which is
        zero and tightens to ``-slack_tol`` when a positive tolerance is passed.

        Args:
            od: (origin, dest) node pair.
            beta: Parameters; defaults to the run's ``beta_hat``.
            slack_tol: Dual-slack dead-band passed to the active-set test.

        Returns:
            Dict with ``links`` (active link ids), ``tail``/``head`` (their
            endpoints), ``flow`` (their optimal flows), ``slack`` (per-link raw
            dual slack), ``margins`` (per-link active-set margin
            :math:`|s_j| / \lVert Z_j \rVert` in beta-units), ``margin`` (the
            per-trip margin, the minimum over links with
            :math:`\lVert Z_j \rVert > 0`, which with the node potentials held
            fixed is the length of the smallest parameter perturbation that
            flips a link), and ``active_set`` (the full :class:`ActiveSet`).

        Example:
            >>> sub = run.active_subnetwork((3, 17))
            >>> list(zip(sub["tail"], sub["head"], sub["flow"]))
        """
        net = self.network()
        solver = self.solver()
        beta = self.beta_hat if beta is None else np.asarray(beta, dtype=float)
        c = net.Z @ beta
        o, d = int(od[0]), int(od[1])
        b_vec = np.zeros(net.num_nodes)
        b_vec[o], b_vec[d] = -1.0, 1.0
        res = solver.solve(c, b_vec)
        z_rownorm = np.linalg.norm(net.Z, axis=1)
        aset: ActiveSet = determine_active_set(
            res.x,
            d,
            net,
            ActiveSetConfig(slack_tol=slack_tol),
            projection_mode="robust_pinv",
            lam_eq=res.lam_eq,
            c=c,
            scale_m=np.asarray(self.data["scale_m"], dtype=float),
            z_rownorm=z_rownorm,
        )
        idx = aset.indices
        finite = np.isfinite(aset.margins)
        return {
            "links": idx,
            "tail": net.tail[idx],
            "head": net.head[idx],
            "flow": res.x[idx],
            "slack": aset.slack,
            "margins": aset.margins,
            "margin": float(np.min(aset.margins[finite])) if finite.any() else float("inf"),
            "active_set": aset,
        }

    def compare_routes(
        self,
        trips: Sequence[int],
        *,
        beta: np.ndarray | None = None,
        flow_tol: float = 1e-8,
    ) -> list[dict[str, Any]]:
        """Compares predicted flows against the observed routes of trips.

        For each trip the forward QP is solved at ``beta`` for the trip's OD
        pair, and the predicted flow pattern is compared with the observed
        route (the nonzero links of the snapshotted ``y`` row).

        Args:
            trips: Trip indices into the snapshotted dataset.
            beta: Parameters; defaults to the run's ``beta_hat``.
            flow_tol: Flow threshold above which a link counts as used.

        Returns:
            One dict per trip with: ``trip``, ``od``, ``n_obs_links``,
            ``n_pred_links``, ``obs_links_carrying_flow`` (count of observed
            links with predicted flow > tol), ``flow_on_route`` (total
            predicted flow mass on observed links), ``flow_share_on_route``
            (that mass over the total predicted flow mass — 1.0 means the
            prediction concentrates entirely on the observed route), and
            ``min_flow_on_route``.

        Raises:
            ValueError: If the run has no snapshotted ``y`` (observed routes).

        Example:
            >>> rows = run.compare_routes(trips=[0, 1, 2])
            >>> rows[0]["flow_share_on_route"]
        """
        if "y" not in self.data:
            raise ValueError("run has no snapshotted y (observed routes)")
        y = self.data["y"]
        y = y.tocsr() if sp.issparse(y) else sp.csr_matrix(np.asarray(y))
        flows = self.predicted_flow(trips=trips, beta=beta)
        o = np.asarray(self.data["origins"], dtype=int)
        d = np.asarray(self.data["dests"], dtype=int)

        rows: list[dict[str, Any]] = []
        for t in trips:
            obs = y.indices[y.indptr[t] : y.indptr[t + 1]]
            x = flows[(int(o[t]), int(d[t]))]
            on_route = x[obs]
            total = float(x.sum())
            rows.append(
                {
                    "trip": int(t),
                    "od": (int(o[t]), int(d[t])),
                    "n_obs_links": int(obs.size),
                    "n_pred_links": int((x > flow_tol).sum()),
                    "obs_links_carrying_flow": int((on_route > flow_tol).sum()),
                    "flow_on_route": float(on_route.sum()),
                    "flow_share_on_route": float(on_route.sum() / total) if total > 0 else np.nan,
                    "min_flow_on_route": float(on_route.min()) if obs.size else np.nan,
                }
            )
        return rows


def load_run(run_dir: str | Path) -> RunArchive:
    """Loads an archived run directory written by :class:`RunWriter`.

    Args:
        run_dir: Path of the run directory (e.g. ``runs/ema_ab~03``).

    Returns:
        RunArchive with config, environment, compute profile,
        result arrays, and the data snapshot (when copied in full).

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    def _json(name: str) -> dict[str, Any]:
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    result = _load_npz(run_dir / "result.npz") if (run_dir / "result.npz").exists() else {}
    data = _load_npz(run_dir / "data.npz") if (run_dir / "data.npz").exists() else {}
    return RunArchive(
        run_dir=run_dir,
        config=_json("config.json"),
        environment=_json("environment.json"),
        compute=_json("compute.json"),
        result=result,
        data=data,
        data_ref=_json("data_ref.json"),
    )
