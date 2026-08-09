"""PIQP-based forward solver for quadratic perturbation."""

from __future__ import annotations

from typing import Any

import numpy as np
import piqp
import scipy.sparse as sp

from ..config import SolverConfig
from ..network import Network
from .base import FlowResult


class PIQPFlowSolver:
    r"""Forward solver using PIQP for the quadratic perturbation.

    .. math::

        \min_x \; c^{\top} x
        + \tfrac12 x^{\top} \mathrm{diag}(\mathrm{scale}_m) x
        \quad \text{s.t.} \quad
        A x = b, \; x \ge 0 .

    The sparsity structure is fixed at construction, and each :meth:`solve`
    updates only :math:`c` and :math:`b`, so one PIQP setup serves every trip.
    """

    def __init__(
        self,
        network: Network,
        scale_m: np.ndarray,
        config: SolverConfig | None = None,
    ) -> None:
        """Set up the PIQP problem for this network.

        Args:
            network: Network with sparse incidence A.
            scale_m: Per-link perturbation weights, shape (L,).
            config: Solver configuration; defaults used if omitted.

        Raises:
            ValueError: If ``scale_m`` length does not match the link count.
        """
        self.network = network
        cfg = config or SolverConfig()
        self.config = cfg
        self._scale_m = np.asarray(scale_m, dtype=np.float64).ravel()

        L = self._scale_m.shape[0]
        if L != network.num_links:
            raise ValueError(f"scale_m has length {L}, but network has {network.num_links} links.")

        self._A_eq = network.A.astype(np.float64).tocsc()
        self._P = sp.diags(self._scale_m, format="csc", dtype=np.float64)

        self._x_l = np.zeros((L, 1), dtype=np.float64)
        self._x_u = np.full((L, 1), np.inf, dtype=np.float64)

        self._solver = piqp.SparseSolver()
        s = self._solver.settings
        s.verbose = cfg.verbose
        s.eps_abs = cfg.eps_abs
        s.eps_duality_gap_abs = cfg.eps_duality_gap_abs
        s.eps_rel = cfg.eps_rel
        s.eps_duality_gap_rel = cfg.eps_duality_gap_rel
        s.max_iter = cfg.max_iter

        c0 = np.zeros((L, 1), dtype=np.float64)
        b0 = np.zeros((self._A_eq.shape[0], 1), dtype=np.float64)

        self._solver.setup(
            self._P,
            c=c0,
            A=self._A_eq,
            b=b0,
            G=None,
            h_l=None,
            h_u=None,
            x_l=self._x_l,
            x_u=self._x_u,
        )

    @property
    def scale_m(self) -> np.ndarray:
        """Per-link perturbation scale, shape (L,)."""
        return self._scale_m

    def solve(
        self,
        c: np.ndarray,
        b: np.ndarray,
        warm: Any | None = None,
    ) -> FlowResult:
        """Solve the quadratic flow problem for one right-hand side.

        Args:
            c: Link cost vector, shape (L,).
            b: Node imbalance vector, shape (V,).
            warm: Unused; PIQP manages its own warm start across calls.

        Returns:
            The optimal flows, the equality duals, and whatever solver
            diagnostics the backend exposes.
        """
        c_vec = np.asarray(c, dtype=np.float64).reshape(-1, 1)
        b_vec = np.asarray(b, dtype=np.float64).reshape(-1, 1)

        self._solver.update(c=c_vec, b=b_vec)
        self._solver.solve()

        result = self._solver.result
        # .copy() is load-bearing: PIQP reuses one internal result buffer, so a
        # view of it would be overwritten by the next solve.
        x_star = np.asarray(result.x, dtype=np.float64).ravel().copy()

        lam_eq = None
        if hasattr(result, "y"):
            lam_eq = np.asarray(result.y, dtype=np.float64).ravel().copy()

        info: dict[str, Any] = {}
        if hasattr(result, "info"):
            info_obj = result.info
            for attr in ("status", "iter", "primal_residual", "dual_residual", "duality_gap"):
                if hasattr(info_obj, attr):
                    info[attr] = getattr(info_obj, attr)

        return FlowResult(x=x_star, lam_eq=lam_eq, info=info)

    def solve_batch(
        self,
        c: np.ndarray,
        origins: np.ndarray,
        dests: np.ndarray,
        n_threads: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve many unit-demand OD flow problems in parallel.

        Every OD shares the cost ``c`` and differs only in the node-imbalance
        right-hand side (-1 at the origin, +1 at the destination), which the
        native multi-threaded batch solver exploits. Each OD's flows agree with
        :meth:`solve`.

        The native path takes ``eps_abs``, ``eps_duality_gap_abs``, and
        ``max_iter`` from ``config`` and leaves the relative termination
        criteria at PIQP's own defaults, so zeroing ``eps_rel`` in ``config``
        leaves this path unaffected.

        Args:
            c: Shared link cost vector, shape (L,).
            origins: Origin node indices, shape (U,).
            dests: Destination node indices, shape (U,).
            n_threads: Worker threads; -1 uses all hardware threads.

        Returns:
            Tuple ``(flows, ok)`` with ``flows`` of shape (U, L) float64 and
            ``ok`` of shape (U,) bool (per-OD solver success).
        """
        from .._native import batch_solve_flows

        A = self._A_eq
        flows, ok = batch_solve_flows(
            A_indptr=A.indptr.astype(np.int64),
            A_indices=A.indices.astype(np.int64),
            A_data=np.ascontiguousarray(A.data, dtype=np.float64),
            scale_m=self._scale_m,
            V=int(A.shape[0]),
            L=int(self._scale_m.shape[0]),
            c=np.ascontiguousarray(c, dtype=np.float64).ravel(),
            origins=np.ascontiguousarray(origins, dtype=np.int64).ravel(),
            dests=np.ascontiguousarray(dests, dtype=np.int64).ravel(),
            qp_eps_abs=float(self.config.eps_abs),
            qp_eps_duality_gap_abs=float(self.config.eps_duality_gap_abs),
            qp_max_iter=int(self.config.max_iter),
            n_threads=int(n_threads),
        )
        return flows, ok.astype(bool)
