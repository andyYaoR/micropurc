"""PIQP-based forward solver for quadratic perturbation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import SolverConfig
from ..network import Network
from .base import FlowResult


class PIQPFlowSolver:
    r"""Forward solver using PIQP for the quadratic perturbation.

    .. math::

        \min_x \; c^{\top} x
        + \tfrac12 x^{\top} \mathrm{diag}(\mathrm{scale}_m) x
        \quad \text{s.t.} \quad
        A x = b, \; x \ge 0 ,

    with no upper bound on :math:`x`.

    The sparsity structure is fixed at construction, and each :meth:`solve`
    updates only :math:`c` and :math:`b`, so one PIQP setup serves every trip.
    PIQP is compiled into the package's native extension, so no separate PIQP
    installation is involved.

    One instance carries one solver state, and :meth:`solve` releases the GIL
    while that state is being written, so an instance serves one thread at a
    time. Give each thread its own solver, or use :meth:`solve_batch`, which
    runs a solver per worker internally.
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

        # Deferred so that a native extension that fails to load (for example
        # one built against an outdated MSVC runtime) cannot take down
        # ``import micropurc``; the failure then surfaces here, where the
        # solver is actually needed.
        from .._native import NativeSparseSolver

        self._solver = NativeSparseSolver(
            A_indptr=self._A_eq.indptr.astype(np.int64),
            A_indices=self._A_eq.indices.astype(np.int64),
            A_data=np.ascontiguousarray(self._A_eq.data, dtype=np.float64),
            scale_m=self._scale_m,
            V=int(self._A_eq.shape[0]),
            L=L,
            verbose=bool(cfg.verbose),
            eps_abs=float(cfg.eps_abs),
            eps_duality_gap_abs=float(cfg.eps_duality_gap_abs),
            eps_rel=float(cfg.eps_rel),
            eps_duality_gap_rel=float(cfg.eps_duality_gap_rel),
            max_iter=int(cfg.max_iter),
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
            warm: Unused; PIQP manages its own state across calls.

        Returns:
            The optimal flows, the equality duals, and the solver diagnostics
            (status, iteration count, residuals, duality gap).
        """
        c_vec = np.ascontiguousarray(c, dtype=np.float64).ravel()
        b_vec = np.ascontiguousarray(b, dtype=np.float64).ravel()

        # Each call returns freshly allocated arrays, so a previous result is
        # never overwritten by the next solve.
        x_star, lam_eq, info = self._solver.solve(c_vec, b_vec)

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
        native multi-threaded batch solver exploits. Both entry points run the
        same solver on the same problem, so each OD's flows agree with
        :meth:`solve` up to the two paths' termination settings.

        This path takes ``eps_abs``, ``eps_duality_gap_abs``, and ``max_iter``
        from ``config`` and leaves the relative termination criteria at PIQP's
        own defaults, so zeroing ``eps_rel`` in ``config`` leaves this path
        unaffected and only :meth:`solve` tightens.

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
