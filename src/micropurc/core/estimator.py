r"""MicroPURC estimator: the fixed-point map :math:`\Phi` and its iteration."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .._native import NativePIQPContext
from .._native import batch_solve_and_accumulate as _native_batch
from ..config import ActiveSetConfig, EstimationConfig, ProjectionConfig, SolverConfig
from ..forward.base import ForwardSolver
from ..network import Network
from .covariance import sandwich_covariance
from .diagnostics import DiagnosticsCollector, IterationRecord
from .lazy_design import LazyDesign


class MicroPURCEstimator:
    r"""Estimator for the microPURC model with quadratic perturbation.

    Each trip contributes a projected least-squares block, and the estimate is
    a fixed point of

    .. math::

        \Phi_N(\beta) = -H_N(\beta)^{-1} g_N(\beta),
        \qquad
        H_N = \sum_i Z_i^{\top} \hat P_i Z_i,
        \qquad
        g_N = \sum_i Z_i^{\top} \hat P_i (\mathrm{scale} \odot y_i),

    where the sums run over the trips whose active set survives grounding,
    :math:`Z_i` is the design restricted to trip :math:`i`'s active links, and
    :math:`\hat P_i` projects onto the null space of that trip's grounded
    active incidence. The dependence of :math:`\Phi_N` on :math:`\beta` runs
    through the active sets, which come from the forward solve, so
    :math:`\Phi_N` is piecewise constant.

    :meth:`fit` drives :math:`\beta = \Phi_N(\beta)` by damped
    Krasnoselskii-Mann iteration: the step is
    :math:`\beta + \lambda (\Phi_N(\beta) - \beta)`, with :math:`\lambda`
    backtracked by a factor ``ls_tau`` until the merit
    :math:`Q(\beta) = \tfrac12 \lVert \Phi_N(\beta) - \beta \rVert^2` drops to
    or below the largest merit over the last ``nm_window`` iterations. Once the
    backtracking budget is spent, the smallest trial step is taken as is.
    """

    def __init__(
        self,
        network: Network,
        forward_solver: ForwardSolver,
        estimation_config: EstimationConfig | None = None,
        active_set_config: ActiveSetConfig | None = None,
        projection_config: ProjectionConfig | None = None,
    ) -> None:
        """Bind the network and forward solver and precompute native arrays.

        Args:
            network: The network the model is estimated on.
            forward_solver: Solver for the inner perturbed-utility flow problem.
            estimation_config: Iteration and tolerance settings; defaults used
                if omitted.
            active_set_config: Active-set detection settings; defaults if omitted.
            projection_config: Range-space projection settings; defaults if
                omitted.
        """
        self.network = network
        self.solver = forward_solver
        self.est_cfg = estimation_config or EstimationConfig()
        self.aset_cfg = active_set_config or ActiveSetConfig()
        self.proj_cfg = projection_config or ProjectionConfig()

        self.scale_m = forward_solver.scale_m
        self.diagnostics = DiagnosticsCollector()

        self._precompute_arrays()

        V = self.network.num_nodes
        # Explicit-Z path: node potentials are round-tripped through Python between
        # iterations, keyed by OD, to warm-start the next forward solve.
        self._lam_warm = np.empty((0, V), dtype=np.float64)
        self._od_keys_prev = np.empty(0, dtype=np.int64)
        # Lazy path keeps its warm-start state (x, y, z_bl) inside the native context.
        self._native_context: NativePIQPContext | None = None

    def _precompute_arrays(self) -> None:
        """Build the contiguous buffers the C++ pipeline reads.

        The native calls take C-contiguous ``float64``/``int64`` arrays, so the
        conversion happens once per estimator and the copies are reused by
        every accumulation pass.
        """
        A_csc = sp.csc_matrix(self.network.A)
        self._A_csc = A_csc
        self._A_indptr = np.ascontiguousarray(A_csc.indptr, dtype=np.int64)
        self._A_indices = np.ascontiguousarray(A_csc.indices, dtype=np.int64)
        self._A_data = np.ascontiguousarray(A_csc.data, dtype=np.float64)
        self._Z_contig = np.ascontiguousarray(self.network.Z, dtype=np.float64)
        self._scale_contig = np.ascontiguousarray(self.scale_m, dtype=np.float64)
        self._tail_contig = np.ascontiguousarray(self.network.tail, dtype=np.int64)
        self._head_contig = np.ascontiguousarray(self.network.head, dtype=np.int64)

    def _get_native_context(self) -> NativePIQPContext:
        """Return the estimator's native PIQP context, built on first use.

        The context holds the network matrices and the lazy path's warm-start
        state, so it lives as long as the estimator.
        """
        if self._native_context is None:
            self._native_context = NativePIQPContext(
                self._A_indptr,
                self._A_indices,
                self._A_data,
                self._scale_contig,
                self._tail_contig,
                self._head_contig,
                self.network.num_nodes,
                self.network.num_links,
            )
        return self._native_context

    def _prepare_sparse_inputs(
        self, y, b, origins=None, dests=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert ``y`` and ``b`` to the compact form the native pipeline reads.

        Route indicators go to CSR. Each row of ``b`` holds -1 at the origin
        and +1 at the destination, so the OD pair is its ``argmin`` and
        ``argmax``; passing ``origins`` and ``dests`` skips that scan. The
        derived arrays are cached under the identities of ``y`` and ``b``, so a
        dataset edited in place under the same objects is served the stale
        cache.
        """
        if (origins is None) != (dests is None):
            raise ValueError("origins and dests must be provided together")

        y_csr = y.tocsr() if sp.issparse(y) else sp.csr_matrix(y)
        N = y_csr.shape[0]
        y_indptr = np.ascontiguousarray(y_csr.indptr, dtype=np.int64)
        y_indices = np.ascontiguousarray(y_csr.indices, dtype=np.int64)
        y_data = np.ascontiguousarray(y_csr.data, dtype=np.float64)

        if origins is not None and dests is not None:
            origins_out = np.asarray(origins, dtype=np.int64).ravel()
            dests_out = np.asarray(dests, dtype=np.int64).ravel()
            if origins_out.shape[0] != N or dests_out.shape[0] != N:
                raise ValueError("origins and dests must have length equal to y.shape[0]")
            return (
                np.ascontiguousarray(origins_out, dtype=np.int64),
                np.ascontiguousarray(dests_out, dtype=np.int64),
                y_indptr,
                y_indices,
                y_data,
            )

        if b is None:
            raise ValueError("b is required when origins/dests are not provided")

        y_id = id(y)
        b_id = id(b)
        if hasattr(self, "_sparse_cache_key") and self._sparse_cache_key == (y_id, b_id):
            return self._sparse_cache_val

        b_arr = np.asarray(b)
        if b_arr.shape[0] != N:
            raise ValueError("b must have the same number of rows as y")
        origins_out = np.argmin(b_arr, axis=1).astype(np.int64)
        dests_out = np.argmax(b_arr, axis=1).astype(np.int64)

        result = (origins_out, dests_out, y_indptr, y_indices, y_data)
        self._sparse_cache_key = (y_id, b_id)
        self._sparse_cache_val = result
        return result

    def _accumulate(
        self,
        beta: np.ndarray,
        y,
        b,
        return_u: bool = False,
        origins: np.ndarray | None = None,
        dests: np.ndarray | None = None,
        z_variants: np.ndarray | None = None,
        trip_variant: np.ndarray | None = None,
        lazy_design: LazyDesign | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
        r"""Solve every trip's forward QP and accumulate ``H``, ``g``, and ``u``.

        One native pass solves the batch of forward QPs at ``beta``, determines
        each trip's active set from the dual slack, grounds it, and sums the
        per-trip Schur blocks. Trips whose active set fails grounding are
        dropped from the sums and counted.

        The design reaches the native layer in one of three ways: the
        network's own ``Z``, a stack of per-variant matrices with
        ``trip_variant``, or a :class:`LazyDesign` whose columns the native
        engine evaluates per link and variant. The last two are mutually
        exclusive.

        Returns ``(H, g, u, dropped)``, with ``u`` the stacked per-trip scores
        for the surviving trips when ``return_u`` is set, otherwise ``None``.
        """
        L = self.network.num_links
        V = self.network.num_nodes

        origins, dests, y_indptr, y_indices, y_data = self._prepare_sparse_inputs(
            y, b, origins, dests
        )
        N = origins.shape[0]

        if lazy_design is not None:
            if z_variants is not None:
                raise ValueError("lazy_design and z_variants are mutually exclusive")
            if lazy_design.num_links != L:
                raise ValueError(f"lazy_design has L={lazy_design.num_links}, expected {L}")
            if lazy_design.trip_variant.shape[0] != N:
                raise ValueError("lazy_design.trip_variant must have length y.shape[0]")
            if getattr(self.solver, "config", SolverConfig()).backend != "piqp":
                raise ValueError("lazy_design native path requires SolverConfig.backend='piqp'")
            K = lazy_design.num_attributes
            beta_arr = np.ascontiguousarray(np.asarray(beta, dtype=np.float64).ravel())
            if beta_arr.shape[0] != K:
                raise ValueError(f"beta has length {beta_arr.shape[0]}, lazy design has K={K}")
            solver_cfg = getattr(self.solver, "config", SolverConfig())
            H, g, u_out, stats, aset_hashes = self._get_native_context().batch_lazy_terms(
                link_attrs=lazy_design.link_attrs,
                variant_attr=lazy_design.variant_attr,
                variant_cov=lazy_design.variant_covariates,
                term_code=lazy_design.term_code,
                term_a=lazy_design.term_a,
                term_b=lazy_design.term_b,
                term_cov=lazy_design.term_cov,
                term_op=lazy_design.term_op,
                term_thr=lazy_design.term_thr,
                beta=beta_arr,
                origins=origins,
                dests=dests,
                y_indptr=y_indptr,
                y_indices=y_indices,
                y_data=y_data,
                trip_variant=lazy_design.trip_variant,
                projection_mode=self.proj_cfg.mode,
                return_u=return_u,
                slack_tol=self.aset_cfg.slack_tol,
                qp_eps_abs=solver_cfg.eps_abs,
                qp_eps_duality_gap_abs=solver_cfg.eps_duality_gap_abs,
                qp_eps_rel=solver_cfg.eps_rel,
                qp_eps_duality_gap_rel=solver_cfg.eps_duality_gap_rel,
                qp_max_iter=solver_cfg.max_iter,
            )
            H_np = np.array(H)
            g_np = np.array(g)
            u_stacked = np.array(u_out) if u_out is not None else None
            self._last_native_stats = dict(stats)
            self.diagnostics.accumulate_native_stats(stats)
            self._last_aset_hashes = np.array(aset_hashes)
            return H_np, g_np, u_stacked, int(stats["dropped"])

        if z_variants is not None:
            z_arr = np.asarray(z_variants, dtype=np.float64)
            if z_arr.ndim != 3:
                raise ValueError("z_variants must have shape (n_variants, L, K)")
            if z_arr.shape[1] != L:
                raise ValueError(f"z_variants has L={z_arr.shape[1]}, expected {L}")
            if z_arr.shape[2] != np.asarray(beta).ravel().shape[0]:
                raise ValueError("z_variants K dimension must match len(beta)")
            if z_arr.shape[0] <= 0:
                raise ValueError("z_variants must contain at least one variant")
            if trip_variant is None:
                raise ValueError("trip_variant required when z_variants is given")
            n_var = z_arr.shape[0]
            K = z_arr.shape[2]
            tv = np.asarray(trip_variant, dtype=np.int64).ravel()
            if tv.shape[0] != N:
                raise ValueError("trip_variant must have length equal to y.shape[0]")
            if np.any((tv < 0) | (tv >= n_var)):
                raise ValueError("trip_variant entries must be in [0, n_variants)")
            Z_flat = np.ascontiguousarray(z_arr.reshape(n_var * L, K), dtype=np.float64)
            c_parts = [z_arr[v] @ beta for v in range(n_var)]
            c = np.ascontiguousarray(np.concatenate(c_parts), dtype=np.float64)
            tv = np.ascontiguousarray(tv, dtype=np.int64)
        else:
            n_var = 0
            Z_flat = self._Z_contig
            K = self.network.num_attributes
            c = np.ascontiguousarray((self.network.Z @ beta).astype(np.float64))
            tv = np.empty(0, dtype=np.int64)

        solver_cfg = getattr(self.solver, "config", SolverConfig())

        H, g, u_out, lam_cache, od_keys_out, stats, aset_hashes = _native_batch(
            Z=Z_flat,
            A_indptr=self._A_indptr,
            A_indices=self._A_indices,
            A_data=self._A_data,
            scale_m=self._scale_contig,
            tail=self._tail_contig,
            head=self._head_contig,
            V=V,
            L=L,
            K=K,
            c=c,
            origins=origins,
            dests=dests,
            y_indptr=y_indptr,
            y_indices=y_indices,
            y_data=y_data,
            lam_warm=self._lam_warm,
            od_keys_prev=self._od_keys_prev,
            trip_variant=tv,
            n_z_variants=n_var,
            projection_mode=self.proj_cfg.mode,
            return_u=return_u,
            slack_tol=self.aset_cfg.slack_tol,
            qp_eps_abs=solver_cfg.eps_abs,
            qp_eps_duality_gap_abs=solver_cfg.eps_duality_gap_abs,
            qp_eps_rel=solver_cfg.eps_rel,
            qp_eps_duality_gap_rel=solver_cfg.eps_duality_gap_rel,
            qp_max_iter=solver_cfg.max_iter,
        )

        self._lam_warm = np.ascontiguousarray(lam_cache, dtype=np.float64)
        self._od_keys_prev = np.ascontiguousarray(od_keys_out, dtype=np.int64)

        H_np = np.array(H)
        g_np = np.array(g)
        u_stacked = np.array(u_out) if u_out is not None else None
        self._last_native_stats = dict(stats)
        self.diagnostics.accumulate_native_stats(stats)
        self._last_aset_hashes = np.array(aset_hashes)
        dropped = stats["dropped"]

        return H_np, g_np, u_stacked, dropped

    def _phi_full(
        self,
        beta: np.ndarray,
        y,
        b,
        origins=None,
        dests=None,
        z_variants=None,
        trip_variant=None,
        lazy_design: LazyDesign | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        r"""Evaluate :math:`\Phi_N(\beta)` and return ``(phi, H, dropped)``."""
        kw: dict[str, Any] = {}
        if origins is not None or dests is not None:
            kw["origins"] = origins
            kw["dests"] = dests
        if z_variants is not None:
            kw["z_variants"] = z_variants
            kw["trip_variant"] = trip_variant
        if lazy_design is not None:
            kw["lazy_design"] = lazy_design
        H, g, _, dropped = self._accumulate(beta, y, b, return_u=False, **kw)

        if not np.any(H):
            raise RuntimeError("H_N is all zeros; cannot compute Φ_N(β).")

        phi_beta = -scipy.linalg.solve(H, g, assume_a="sym")
        return phi_beta, H, dropped

    def _record_aset_hashes(self) -> None:
        h = getattr(self, "_last_aset_hashes", None)
        if h is not None:
            self.diagnostics.record_active_set_hashes(h)

    def phi(
        self,
        beta: np.ndarray,
        y,
        b=None,
        *,
        origins: np.ndarray | None = None,
        dests: np.ndarray | None = None,
        lazy_design: LazyDesign | None = None,
    ) -> np.ndarray:
        r"""Evaluate the fixed-point map :math:`\Phi_N(\beta)`.

        Args:
            beta: Parameter vector to evaluate at, shape (K,).
            y: Route indicators, dense (N, L) or sparse CSR (N, L).
            b: Node imbalances, shape (N, V). Needed when ``origins`` and
                ``dests`` are absent.
            origins: Origin node indices, shape (N,).
            dests: Destination node indices, shape (N,).
            lazy_design: A lazy design evaluated on the fly by the native
                engine.

        Returns:
            :math:`\Phi_N(\beta) = -H_N(\beta)^{-1} g_N(\beta)`, shape (K,).
        """
        phi_beta, _, _ = self._phi_full(
            beta, y, b, origins=origins, dests=dests, lazy_design=lazy_design
        )
        return phi_beta

    def fit(
        self,
        y,
        b=None,
        beta_init: Sequence[float] | None = None,
        *,
        origins: np.ndarray | None = None,
        dests: np.ndarray | None = None,
        z_variants: np.ndarray | None = None,
        trip_variant: np.ndarray | None = None,
        lazy_design: LazyDesign | None = None,
    ) -> dict[str, Any]:
        r"""Solve :math:`\beta = \Phi_N(\beta)` from ``beta_init``.

        The iteration runs until the fixed-point residual meets ``fp_tol``,
        optionally scaled by :math:`\sqrt{K}`, or until ``max_iterations`` is
        spent. Convergence additionally requires the condition number of the
        accumulated Hessian to stay within ``max_cond_H``, which rules out a
        residual that goes small only because :math:`\Phi_N` has flattened at a
        degenerate iterate. The final estimate is accumulated once more to
        obtain the scores and the covariance.

        Args:
            y: Route indicators, dense (N, L) or sparse CSR (N, L).
            b: Node imbalances, shape (N, V). Needed when ``origins`` and
                ``dests`` are absent.
            beta_init: Initial parameter guess, shape (K,); defaults to ones.
            origins: Origin node indices, shape (N,).
            dests: Destination node indices, shape (N,).
            z_variants: Per-variant Z matrices, shape (n_variants, L, K).
            trip_variant: Per-trip variant index, shape (N,). Required with
                z_variants.
            lazy_design: A lazy design whose per-variant regressors are
                evaluated on the fly, keeping the (N, L, K) design out of
                memory. Mutually exclusive with z_variants.

        Returns:
            A dict holding

            - ``beta_hat``: the final iterate, shape (K,);
            - ``se_hat``, ``cov``: the i.i.d. sandwich standard errors and
              covariance, ``None`` when the final accumulation yields no scores
              or a singular Hessian;
            - ``u_scores``: per-trip scores, shape (N, K), aligned with the
              input trips, a dropped trip carrying a zero row;
            - ``kept_mask``: length-N boolean, False at a dropped trip;
            - ``hessian``: the accumulated Hessian, shape (K, K);
            - ``diagnostics``: the per-run summary with the convergence
              verdict, the merit history, and the formatted convergence and
              active-set tables.

            Clustered inference follows from passing ``hessian`` and
            ``u_scores`` to
            :func:`~micropurc.core.covariance.cluster_robust_covariance` with
            length-N cluster labels: the zero rows keep dropped trips out of
            the meat, so the caller needs no drop bookkeeping. ``u_scores`` and
            ``kept_mask`` are ``None`` when the final accumulation returned no
            scores; if the survivor mask cannot be matched to those scores,
            ``u_scores`` keeps only the surviving rows and ``kept_mask`` stays
            ``None``.

        Raises:
            ValueError: If z_variants is given without trip_variant, or if both
                lazy_design and z_variants are given.
        """
        K = lazy_design.num_attributes if lazy_design is not None else self.network.num_attributes
        cfg = self.est_cfg

        fp_tol = cfg.fp_tol
        if cfg.scale_fp_tol_by_sqrt_k:
            fp_tol = cfg.fp_tol * math.sqrt(K)

        if beta_init is None:
            beta = np.ones(K, dtype=float)
        else:
            beta = np.asarray(beta_init, dtype=float).ravel()

        if z_variants is not None and trip_variant is None:
            raise ValueError("trip_variant required when z_variants is given")
        if lazy_design is not None and z_variants is not None:
            raise ValueError("lazy_design and z_variants are mutually exclusive")

        self.diagnostics = DiagnosticsCollector()
        _t_fit0 = time.perf_counter()

        _extra: dict[str, Any] = {}
        if origins is not None or dests is not None:
            _extra["origins"] = origins
            _extra["dests"] = dests
        if z_variants is not None:
            _extra["z_variants"] = z_variants
            _extra["trip_variant"] = trip_variant
        if lazy_design is not None:
            _extra["lazy_design"] = lazy_design
            # Fresh fit: clear any warm-start state the native context persisted from
            # a previous fit so the first KM iteration solves cold.
            self._get_native_context().reset_warm_start()

        phi_beta, H_current, dropped_current = self._phi_full(
            beta,
            y,
            b,
            **_extra,
        )
        self._record_aset_hashes()
        self.diagnostics.flush_active_set_churn()

        Q_history: list[float] = []
        converged = False

        for it in range(cfg.max_iterations):
            t0 = time.perf_counter()

            cond_H, min_eig_H = DiagnosticsCollector.compute_cond_H(H_current)

            r = phi_beta - beta
            Q_k = 0.5 * float(r @ r)
            Q_history.append(Q_k)

            residual_norm = math.sqrt(2.0 * Q_k)

            if residual_norm <= fp_tol:
                # The residual test can also be met at a degenerate iterate, where
                # |beta| has exploded, the forward flows sit at their box bounds and
                # H is ill-conditioned. Convergence therefore also demands cond(H)
                # within EstimationConfig.max_cond_H.
                converged = cond_H <= cfg.max_cond_H
                self.diagnostics.record_iteration(
                    IterationRecord(
                        iteration=it,
                        phi_residual=residual_norm,
                        Q_merit=Q_k,
                        step_size=0.0,
                        accepted=converged,
                        ls_tries=0,
                        cond_H=cond_H,
                        min_eig_H=min_eig_H,
                        dropped_trips=dropped_current,
                        beta=beta.copy(),
                        Q_ref=Q_k,
                        wall_clock_s=time.perf_counter() - t0,
                    )
                )
                break

            Q_ref = max(Q_history[-cfg.nm_window :])

            accepted = False
            ls_tries = 0

            for m in range(cfg.ls_max_tries + 1):
                ls_tries += 1
                lam = cfg.ls_tau**m
                beta_trial = beta + lam * r

                try:
                    phi_trial, H_trial, dropped_trial = self._phi_full(
                        beta_trial,
                        y,
                        b,
                        **_extra,
                    )
                except RuntimeError:
                    continue

                r_trial = phi_trial - beta_trial
                Q_trial = 0.5 * float(r_trial @ r_trial)

                ok = Q_trial <= Q_ref

                if ok:
                    accepted = True
                    beta = beta_trial
                    phi_beta = phi_trial
                    H_current = H_trial
                    dropped_current = dropped_trial
                    break

            if not accepted:
                lam = cfg.ls_tau**cfg.ls_max_tries
                beta = beta + lam * r
                phi_beta, H_current, dropped_current = self._phi_full(
                    beta,
                    y,
                    b,
                    **_extra,
                )

            self._record_aset_hashes()
            link_flips = self.diagnostics.flush_active_set_churn()

            self.diagnostics.record_iteration(
                IterationRecord(
                    iteration=it,
                    phi_residual=residual_norm,
                    Q_merit=Q_k,
                    Q_ref=Q_ref,
                    step_size=lam,
                    accepted=accepted,
                    ls_tries=ls_tries,
                    link_flips=link_flips,
                    cond_H=cond_H,
                    min_eig_H=min_eig_H,
                    dropped_trips=dropped_current,
                    beta=beta.copy(),
                    wall_clock_s=time.perf_counter() - t0,
                )
            )

            self.diagnostics.snapshot_rss()

        self.diagnostics.finalize_churn()
        beta_hat = beta.copy()

        H, _, u_stacked, _ = self._accumulate(
            beta_hat,
            y,
            b,
            return_u=True,
            **_extra,
        )
        self.diagnostics.snapshot_rss()

        cov = None
        se = None
        if u_stacked is not None and u_stacked.shape[0] > 0 and np.any(H):
            try:
                cov, se = sandwich_covariance(H, u_stacked)
            except np.linalg.LinAlgError:
                pass

        # Align per-trip scores to the N input trips so a caller can cluster on any
        # per-trip id. The native pass returns u only for the trips that survived
        # accumulation, stacked in trip order, while ``_last_aset_hashes`` has
        # length N and is nonzero exactly for survivors: a valid active set never
        # hashes to 0 (common.hpp) and a dropped trip keeps the initial 0. Scattering
        # u into a full (N, K) array gives each dropped trip a zero score, which
        # leaves both the i.i.d. and the clustered meat unchanged, so se_hat/cov
        # computed above still hold.
        u_scores = u_stacked
        kept_mask = None
        hashes = getattr(self, "_last_aset_hashes", None)
        if hashes is not None and u_stacked is not None:
            mask = np.asarray(hashes).ravel() != 0
            if u_stacked.shape[0] == int(mask.sum()):
                kept_mask = mask
                u_scores = np.zeros((mask.shape[0], K), dtype=float)
                u_scores[mask] = u_stacked

        self.diagnostics.total_wall_s = time.perf_counter() - _t_fit0
        diag_summary = self.diagnostics.summary()
        diag_summary["converged"] = converged
        diag_summary["Q_history"] = Q_history
        diag_summary["convergence_table"] = self.diagnostics.format_convergence_table()
        diag_summary["activeset_table"] = self.diagnostics.format_activeset_table()

        return {
            "beta_hat": beta_hat,
            "se_hat": se,
            "cov": cov,
            "u_scores": u_scores,
            "kept_mask": kept_mask,
            "hessian": H,
            "diagnostics": diag_summary,
        }
