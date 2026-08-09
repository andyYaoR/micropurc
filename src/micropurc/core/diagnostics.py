"""Convergence and active-set diagnostics for the estimator.

The estimator solves a fixed point whose map is piecewise constant in the
active set of binding non-negativity constraints. This module records the
per-iteration quantities of a run (residuals, merit values, step sizes,
active-set churn, and dual-slack margins), classifies why a run fails to
converge, and probes how sensitive the detected active set is to a
perturbation of the parameter estimate.

The main entry points are:

- :class:`DiagnosticsCollector` -- per-run iteration records, active-set churn,
  margin statistics, and non-convergence classification;
- :class:`ActiveSetReport` -- a per-replication summary of active-set behavior;
- :class:`ConvergenceAnalysis` -- aggregation of diagnostics across
  replications;
- :func:`active_set_stability_probe` -- sensitivity of the active set to a
  perturbation of the estimate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    import resource  # Unix-only; used for peak-RSS measurement (getrusage)
except ImportError:  # Windows has no resource module
    resource = None  # type: ignore[assignment]


def _windows_peak_rss_bytes() -> int:
    """Peak working-set size of this process in bytes via the Win32 API.

    Uses ``GetProcessMemoryInfo`` through ``ctypes`` so no external dependency
    (e.g. ``psutil``) is required. Returns 0 if the query fails.
    """
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # Both prototypes must be declared: ctypes defaults an undeclared return
    # type to C int, which on 64-bit Windows truncates the pseudo-handle
    # returned by GetCurrentProcess (-1 as a 64-bit value) and makes the
    # subsequent query fail.
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE

    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ok = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else 0


class NonConvergenceReason(Enum):
    """Classification of why an estimation run failed to converge.

    Attributes:
        MAX_ITERATIONS: The iteration budget ran out while the residual was
            still moving, or the history is too short to classify.
        STAGNATION: The residual plateaued.
        OSCILLATION: The residual changes alternate in sign.
        LINE_SEARCH_EXHAUSTION: The line search rejected most late steps.
        H_CONDITIONING: The Hessian is ill-conditioned in the late iterations.
        ACTIVE_SET_CYCLING: Active sets keep flipping while the residual holds
            still.
        TRIP_DROP_INSTABILITY: The number of dropped trips fluctuates across
            iterations.
    """

    MAX_ITERATIONS = "max_iterations"
    STAGNATION = "stagnation"
    OSCILLATION = "oscillation"
    LINE_SEARCH_EXHAUSTION = "ls_exhaustion"
    H_CONDITIONING = "h_conditioning"
    ACTIVE_SET_CYCLING = "aset_cycling"
    TRIP_DROP_INSTABILITY = "trip_drop"


@dataclass
class IterationRecord:
    r"""Per-iteration diagnostic record.

    The residual and merit describe the iterate the estimator *entered* the
    iteration with; ``beta`` is the iterate it *left* with.

    Attributes:
        iteration: Zero-based iteration counter.
        phi_residual: :math:`\lVert \Phi_N(\beta) - \beta \rVert`.
        Q_merit: Merit value :math:`Q = \tfrac12 \lVert \Phi_N(\beta) - \beta
            \rVert^2`.
        step_size: Damping factor applied to the fixed-point step.
        accepted: Whether the line search found a step meeting the
            non-monotone acceptance test.
        ls_tries: Number of trial steps evaluated.
        cache_hits: Filled by callers that track their own caches; the native
            accumulation path leaves this at 0.
        cache_misses: Counterpart of ``cache_hits``.
        active_set_changes: Filled by callers that track per-trip active-set
            changes; the native accumulation path leaves this at 0.
        link_flips: Number of trips whose active set differs from the previous
            iteration, backfilled by
            :meth:`DiagnosticsCollector.finalize_churn`.
        cond_H: Condition number of the accumulated Hessian.
        min_eig_H: Smallest eigenvalue of the accumulated Hessian.
        dropped_trips: Trips discarded during accumulation at this iterate.
        wall_clock_s: Seconds spent in this iteration.
        beta: The iterate after this iteration's step.
        Q_ref: Non-monotone reference level the trial step had to beat.
    """

    iteration: int
    phi_residual: float
    Q_merit: float
    step_size: float
    accepted: bool
    ls_tries: int
    cache_hits: int = 0
    cache_misses: int = 0
    active_set_changes: int = 0
    link_flips: int = 0
    cond_H: float = 0.0
    min_eig_H: float = 0.0
    dropped_trips: int = 0
    wall_clock_s: float = 0.0
    beta: np.ndarray | None = None
    Q_ref: float = 0.0


@dataclass
class MarginStats:
    r"""Summary statistics of dual-slack margins for one iteration.

    A link's distance to flipping is :math:`|s_j|`. The active statistics are
    taken over :math:`|s_j|` for links with :math:`s_j < 0`, pooled across the
    trips of one iteration; small values mark links on the verge of leaving the
    active set.

    Attributes:
        min_active: Smallest :math:`|s_j|` over active links, ``inf`` when no
            link is active.
        median_active: Median :math:`|s_j|` over active links.
        q05_active: 5th percentile of :math:`|s_j|` over active links.
        q95_active: 95th percentile of :math:`|s_j|` over active links.
        max_inactive: Largest slack over inactive links.
        n_active_total: Number of active links pooled over trips.
        n_inactive_total: Number of inactive links pooled over trips.
    """

    min_active: float
    median_active: float
    q05_active: float
    q95_active: float
    max_inactive: float
    n_active_total: int
    n_inactive_total: int


@dataclass
class DropReasonCounts:
    """Per-iteration counts of trip-drop reasons.

    Attributes:
        no_active_links: No link passed the dual-slack test.
        trivial_flow: At most one link is active.
        dag_connectivity_failed: An active node fails to reach the destination.
        not_dag: The active subnetwork contains a directed cycle.
        no_kept_rows: Component grounding left no node row to project on.
        other: Reasons outside the buckets above.
    """

    no_active_links: int = 0
    trivial_flow: int = 0
    dag_connectivity_failed: int = 0
    not_dag: int = 0
    no_kept_rows: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        """Total dropped-trip count summed over all reasons."""
        return (
            self.no_active_links
            + self.trivial_flow
            + self.dag_connectivity_failed
            + self.not_dag
            + self.no_kept_rows
            + self.other
        )

    def record(self, reason: str) -> None:
        """Increment the counter for a drop reason, bucketing unknowns as other."""
        if reason == "no_active_links":
            self.no_active_links += 1
        elif reason == "trivial_flow":
            self.trivial_flow += 1
        elif reason == "dag_connectivity_failed":
            self.dag_connectivity_failed += 1
        elif reason == "not_dag":
            self.not_dag += 1
        elif reason in ("no_kept_rows", "no_kept_rows_robust"):
            self.no_kept_rows += 1
        else:
            self.other += 1


@dataclass
class DiagnosticsCollector:
    """Collects per-iteration diagnostics for a single estimation run.

    One collector belongs to one call of
    :meth:`~micropurc.core.estimator.MicroPURCEstimator.fit`, which replaces it
    at the start of every fit.

    Attributes:
        records: One :class:`IterationRecord` per iteration, in order.
        active_set_sizes: Per-iteration lists of per-trip active-set sizes.
        margin_stats: Per-iteration dual-slack margin summaries.
        drop_reasons: Per-iteration trip-drop counts by reason.
        peak_rss_kb: Peak resident set size in the raw platform unit; read it
            through :meth:`peak_rss_mb`.
        total_wall_s: Wall-clock seconds for the whole fit.
        native_time_solve_s: Forward-QP seconds, summed over passes and threads.
        native_time_projection_s: Projection seconds, summed the same way.
        native_time_schur_s: Schur-accumulation seconds, summed the same way.
        native_total_qp_iters: Interior-point iterations over the whole fit.
        n_threads: Worker threads the native pipeline reported.
        aset_hash_history: One per-trip active-set hash array per recorded
            evaluation, consumed by :meth:`finalize_churn`.
        per_trip_cumulative_churn: Trip id to the number of iterations at which
            that trip's active set changed.
    """

    records: list[IterationRecord] = field(default_factory=list)
    active_set_sizes: list[list[int]] = field(default_factory=list)
    margin_stats: list[MarginStats] = field(default_factory=list)
    drop_reasons: list[DropReasonCounts] = field(default_factory=list)
    peak_rss_kb: int = 0
    total_wall_s: float = 0.0
    native_time_solve_s: float = 0.0
    native_time_projection_s: float = 0.0
    native_time_schur_s: float = 0.0
    native_total_qp_iters: int = 0
    n_threads: int = 0
    _prev_trip_active: dict[int, frozenset[int]] = field(
        default_factory=dict,
        repr=False,
    )
    _current_trip_active: dict[int, frozenset[int]] = field(
        default_factory=dict,
        repr=False,
    )
    aset_hash_history: list[np.ndarray] = field(
        default_factory=list,
        repr=False,
    )
    per_trip_cumulative_churn: dict[int, int] = field(
        default_factory=dict,
        repr=False,
    )

    def record_iteration(self, rec: IterationRecord) -> None:
        """Append one iteration's record."""
        self.records.append(rec)

    def record_active_set_sizes(self, sizes: list[int]) -> None:
        """Append the per-trip active-set sizes observed in one iteration."""
        self.active_set_sizes.append(sizes)

    def record_trip_active_set(self, trip_id: int, indices: np.ndarray) -> None:
        """Store one trip's active link indices for the current evaluation."""
        self._current_trip_active[trip_id] = frozenset(indices.tolist())

    def record_active_set_hashes(self, hashes: np.ndarray) -> None:
        r"""Append per-trip active-set hashes from one :math:`\Phi` evaluation.

        The estimator calls this once before the iteration loop and once per
        iteration. The arrays are only stored here; :meth:`finalize_churn`
        compares them in bulk once the loop is over, which keeps churn
        accounting off the hot path.

        Args:
            hashes: Per-trip active-set hash, shape (N,). A hash of 0 marks a
                trip that was dropped.
        """
        self.aset_hash_history.append(hashes.copy())

    def flush_active_set_churn(
        self,
        *,
        detailed: bool = False,
    ) -> int | tuple[int, dict[int, int]]:
        """Compare current against previous active sets; return total link flips.

        This serves callers that register index sets through
        :meth:`record_trip_active_set`. Once any hash array has been recorded,
        churn is accounted for in bulk by :meth:`finalize_churn` and this
        returns zero.

        Args:
            detailed: Also return the per-trip flip counts.

        Returns:
            The total number of link flips, or that total paired with a mapping
            from trip id to flip count when ``detailed`` is set.
        """
        if self.aset_hash_history:
            return (0, {}) if detailed else 0

        total = 0
        per_trip: dict[int, int] = {} if detailed else {}
        all_trips = set(self._prev_trip_active) | set(self._current_trip_active)
        for tid in all_trips:
            prev = self._prev_trip_active.get(tid, frozenset())
            curr = self._current_trip_active.get(tid, frozenset())
            n_flips = len(prev.symmetric_difference(curr))
            total += n_flips
            if detailed and n_flips > 0:
                per_trip[tid] = n_flips
        self._prev_trip_active = self._current_trip_active
        self._current_trip_active = {}
        if detailed:
            return total, per_trip
        return total

    def finalize_churn(self) -> None:
        """Compute per-trip and per-iteration churn from the saved hash arrays.

        Called once after the iteration loop. Populates
        ``per_trip_cumulative_churn`` and backfills ``link_flips`` on each
        :class:`IterationRecord`. A trip counts as churning between two
        evaluations when both of its hashes are nonzero and differ, so a trip
        dropped at either iterate is passed over.
        """
        if len(self.aset_hash_history) < 2:
            return

        flips_per_iter: list[int] = [0]
        for t in range(1, len(self.aset_hash_history)):
            prev = self.aset_hash_history[t - 1]
            curr = self.aset_hash_history[t]
            changed = (prev != curr) & (prev != 0) & (curr != 0)
            trip_ids = np.where(changed)[0]
            flips_per_iter.append(int(trip_ids.shape[0]))
            for tid in trip_ids:
                self.per_trip_cumulative_churn[int(tid)] = (
                    self.per_trip_cumulative_churn.get(int(tid), 0) + 1
                )

        for i, rec in enumerate(self.records):
            if i < len(flips_per_iter):
                rec.link_flips = flips_per_iter[i]

    def record_margins(self, all_slacks: list[np.ndarray]) -> None:
        """Compute and store one iteration's margin statistics.

        Slacks are pooled across trips and split at zero: negative entries are
        the active links and enter as :math:`|s_j|`, the rest are the inactive
        ones.

        Args:
            all_slacks: One dual-slack vector per trip.
        """
        active_vals = []
        inactive_vals = []
        for slack in all_slacks:
            active_vals.extend(slack[slack < 0].tolist())
            inactive_vals.extend(slack[slack >= 0].tolist())

        if active_vals:
            active_arr = np.abs(np.array(active_vals))
            stats = MarginStats(
                min_active=float(np.min(active_arr)),
                median_active=float(np.median(active_arr)),
                q05_active=float(np.percentile(active_arr, 5)),
                q95_active=float(np.percentile(active_arr, 95)),
                max_inactive=float(np.max(inactive_vals)) if inactive_vals else 0.0,
                n_active_total=len(active_vals),
                n_inactive_total=len(inactive_vals),
            )
        else:
            stats = MarginStats(
                min_active=float("inf"),
                median_active=0.0,
                q05_active=0.0,
                q95_active=0.0,
                max_inactive=float(np.max(inactive_vals)) if inactive_vals else 0.0,
                n_active_total=0,
                n_inactive_total=len(inactive_vals),
            )
        self.margin_stats.append(stats)

    def record_drop_reasons(self, counts: DropReasonCounts) -> None:
        """Append one iteration's drop-reason counts."""
        self.drop_reasons.append(counts)

    def snapshot_rss(self) -> None:
        """Raise the recorded peak RSS to the process's current peak.

        Measurement is best effort: a platform that answers neither
        ``getrusage`` nor the Win32 query leaves the previous value standing.
        """
        try:
            if resource is not None:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                self.peak_rss_kb = max(self.peak_rss_kb, ru.ru_maxrss // 1024)
            elif sys.platform == "win32":  # no resource module; query the Win32 API
                peak_bytes = _windows_peak_rss_bytes()
                # Store MiB, the unit peak_rss_mb() expects off macOS.
                self.peak_rss_kb = max(self.peak_rss_kb, peak_bytes // (1024 * 1024))
        except Exception:
            pass

    def peak_rss_mb(self) -> float:
        """Peak resident set size in MiB, normalized across platforms.

        ``ru_maxrss`` is bytes on macOS and KiB on Linux, and ``peak_rss_kb``
        stores ``ru_maxrss // 1024``, so its unit is KiB on macOS and MiB on
        Linux and Windows. This method converts whichever it holds to MiB.

        Returns:
            The peak resident set size in MiB.
        """
        if sys.platform == "darwin":
            return self.peak_rss_kb / 1024.0  # KiB -> MiB
        return float(self.peak_rss_kb)  # already MiB on Linux/Windows

    def accumulate_native_stats(self, stats: dict) -> None:
        r"""Sum one native accumulate pass's phase timings into the fit totals.

        The estimator calls the native batch routine once per
        Krasnoselskii-Mann :math:`\Phi` evaluation and once for the covariance
        pass, and each pass returns its own ``time_solve_s`` /
        ``time_projection_s`` / ``time_schur_s`` / ``total_qp_iters``. Adding
        them here makes :meth:`summary` report totals over the whole fit.

        The phase timings are aggregate CPU-seconds summed over the worker
        threads, so under parallelism they exceed the wall-clock
        ``total_wall_s`` by roughly the achieved thread utilization. They carry
        the relative cost split across solve, projection, and Schur;
        ``total_wall_s`` carries the runtime.

        Args:
            stats: One pass's native statistics dict.
        """
        self.native_time_solve_s += float(stats.get("time_solve_s", 0.0))
        self.native_time_projection_s += float(stats.get("time_projection_s", 0.0))
        self.native_time_schur_s += float(stats.get("time_schur_s", 0.0))
        self.native_total_qp_iters += int(stats.get("total_qp_iters", 0))
        self.n_threads = int(stats.get("n_threads", self.n_threads))

    @staticmethod
    def compute_cond_H(H: np.ndarray) -> tuple[float, float]:
        r"""Compute the condition number and smallest eigenvalue of ``H``.

        ``H`` is treated as symmetric, and the condition number is
        :math:`\lambda_{\max} / \max(|\lambda_{\min}|, 10^{-30})`, the floor
        keeping a singular ``H`` finite.

        Args:
            H: Accumulated Hessian, shape (K, K).

        Returns:
            The condition number and the smallest eigenvalue, or
            ``(inf, 0.0)`` when the eigenvalue solver fails to converge.
        """
        try:
            eigs = np.linalg.eigvalsh(H)
            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            cond = max_eig / max(abs(min_eig), 1e-30)
            return cond, min_eig
        except np.linalg.LinAlgError:
            return float("inf"), 0.0

    def classify_non_convergence(
        self,
        stagnation_window: int = 20,
        stagnation_tol: float = 1e-4,
        oscillation_window: int = 10,
        oscillation_frac: float = 0.6,
        ls_exhaustion_window: int = 10,
        ls_exhaustion_frac: float = 0.7,
        cond_threshold: float = 1e12,
        churn_window: int = 10,
        churn_threshold: int = 50,
        drop_cv_threshold: float = 0.3,
    ) -> tuple[NonConvergenceReason, dict]:
        """Classify why convergence failed from the iteration history.

        The tests run in a fixed order and the first match wins: Hessian
        conditioning, line-search exhaustion, trip-drop instability, active-set
        cycling, oscillation, then stagnation. A history that matches none of
        them, or that holds fewer than three records, is reported as an
        exhausted iteration budget.

        Args:
            stagnation_window: Length of the trailing window shared by the
                conditioning, line-search, drop, and stagnation tests.
            stagnation_tol: Relative residual change below which the residual
                counts as flat, used by the stagnation and cycling tests.
            oscillation_window: Number of trailing residual differences
                inspected for sign changes.
            oscillation_frac: Fraction of those differences that must alternate.
            ls_exhaustion_window: Shortest trailing window for which the
                rejection test applies.
            ls_exhaustion_frac: Fraction of rejected steps that triggers.
            cond_threshold: Condition number above which the Hessian is blamed.
            churn_window: Trailing iterations used by the cycling test.
            churn_threshold: Mean per-iteration link flips that count as churn.
            drop_cv_threshold: Coefficient of variation of the dropped-trip
                counts above which drops count as unstable.

        Returns:
            The reason, paired with an evidence dict holding the metrics that
            triggered it.
        """
        if len(self.records) < 3:
            return NonConvergenceReason.MAX_ITERATIONS, {"n_iters": len(self.records)}

        residuals = [r.phi_residual for r in self.records]
        tail = self.records[-stagnation_window:]

        # Ill-conditioned H in the late iterations.
        late_conds = [r.cond_H for r in tail if r.cond_H > 0]
        if late_conds and max(late_conds) > cond_threshold:
            return NonConvergenceReason.H_CONDITIONING, {
                "max_cond_H": max(late_conds),
                "min_eig_H": min(r.min_eig_H for r in tail),
                "threshold": cond_threshold,
            }

        # Line search exhaustion — majority of late iterations rejected
        late_accepted = [r.accepted for r in tail]
        if len(late_accepted) >= ls_exhaustion_window:
            reject_frac = 1.0 - sum(late_accepted) / len(late_accepted)
            if reject_frac >= ls_exhaustion_frac:
                return NonConvergenceReason.LINE_SEARCH_EXHAUSTION, {
                    "reject_fraction": reject_frac,
                    "window": len(late_accepted),
                    "avg_ls_tries": sum(r.ls_tries for r in tail) / len(tail),
                }

        # Trip drop instability — dropped count fluctuating
        if len(self.records) >= 5:
            drops = [r.dropped_trips for r in self.records]
            if len(set(drops)) > 1:
                drop_arr = np.array(drops[-stagnation_window:], dtype=float)
                mean_drop = np.mean(drop_arr)
                if mean_drop > 0:
                    cv = float(np.std(drop_arr) / mean_drop)
                    if cv > drop_cv_threshold:
                        return NonConvergenceReason.TRIP_DROP_INSTABILITY, {
                            "drop_cv": cv,
                            "drop_range": (int(np.min(drop_arr)), int(np.max(drop_arr))),
                            "mean_drops": float(mean_drop),
                        }

        # Active-set cycling — high churn without progress
        if len(self.records) >= churn_window:
            late_churn = [r.link_flips for r in self.records[-churn_window:]]
            late_residuals = residuals[-churn_window:]
            if sum(late_churn) > churn_threshold * churn_window:
                r_change = abs(late_residuals[-1] - late_residuals[0])
                if r_change < stagnation_tol * abs(late_residuals[0] + 1e-30):
                    return NonConvergenceReason.ACTIVE_SET_CYCLING, {
                        "avg_link_flips": sum(late_churn) / len(late_churn),
                        "residual_change": r_change,
                        "window": churn_window,
                    }

        # Oscillation — the residual bounces, so its differences keep changing sign
        if len(residuals) >= oscillation_window:
            deltas = [residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)]
            late_deltas = deltas[-oscillation_window:]
            sign_changes = sum(
                1 for i in range(len(late_deltas) - 1) if late_deltas[i] * late_deltas[i + 1] < 0
            )
            if sign_changes >= oscillation_frac * (len(late_deltas) - 1):
                return NonConvergenceReason.OSCILLATION, {
                    "sign_changes": sign_changes,
                    "window": len(late_deltas),
                    "residual_range": (
                        min(residuals[-oscillation_window:]),
                        max(residuals[-oscillation_window:]),
                    ),
                }

        # Stagnation — residual plateaued
        if len(residuals) >= stagnation_window:
            r_early = residuals[-stagnation_window]
            r_late = residuals[-1]
            rel_change = abs(r_late - r_early) / (abs(r_early) + 1e-30)
            if rel_change < stagnation_tol:
                return NonConvergenceReason.STAGNATION, {
                    "relative_change": rel_change,
                    "window": stagnation_window,
                    "residual_start": r_early,
                    "residual_end": r_late,
                }

        # Nothing matched: report the budget, with the residual trend as evidence.
        earlier = residuals[-min(5, len(residuals))]
        trend = "decreasing" if residuals[-1] < earlier else "flat"
        return NonConvergenceReason.MAX_ITERATIONS, {
            "n_iters": len(self.records),
            "final_residual": residuals[-1],
            "residual_trend": trend,
        }

    def summary(self) -> dict:
        """Produce a summary dict for diagnostics output.

        The ``converged`` entry is a residual-only verdict: the final
        ``phi_residual`` at or below ``1e-3``. When it comes out False the
        summary also carries a ``non_convergence_reason`` and its evidence.
        :meth:`~micropurc.core.estimator.MicroPURCEstimator.fit` overwrites
        ``converged`` with its own verdict, which uses the configured
        fixed-point tolerance and the Hessian conditioning bound.

        Returns:
            The summary, empty when no iteration was recorded.
        """
        if not self.records:
            return {}

        final = self.records[-1]
        converged = final.phi_residual <= 1e-3

        result = {
            "num_iterations": len(self.records),
            "converged": converged,
            "final_residual": final.phi_residual,
            "final_Q": final.Q_merit,
            "total_cache_hits": sum(r.cache_hits for r in self.records),
            "total_cache_misses": sum(r.cache_misses for r in self.records),
            "total_active_set_changes": sum(r.active_set_changes for r in self.records),
            "total_link_flips": sum(r.link_flips for r in self.records),
            "link_flips_history": [r.link_flips for r in self.records],
            "total_dropped_trips": sum(r.dropped_trips for r in self.records),
            "dropped_trips_history": [r.dropped_trips for r in self.records],
            "Q_history": [r.Q_merit for r in self.records],
            "residual_history": [r.phi_residual for r in self.records],
            "step_sizes": [r.step_size for r in self.records],
            "ls_tries_history": [r.ls_tries for r in self.records],
            "accepted_history": [r.accepted for r in self.records],
            "cond_H_history": [r.cond_H for r in self.records],
            "min_eig_H_history": [r.min_eig_H for r in self.records],
            "beta_history": [r.beta.tolist() for r in self.records if r.beta is not None],
            "peak_rss_kb": self.peak_rss_kb,
            "peak_rss_mb": self.peak_rss_mb(),
            "total_wall_s": self.total_wall_s,
            "time_solve_s": self.native_time_solve_s,
            "time_projection_s": self.native_time_projection_s,
            "time_schur_s": self.native_time_schur_s,
            "total_qp_iters": self.native_total_qp_iters,
            "n_threads": self.n_threads,
            "non_convergence_reason": None,
            "non_convergence_evidence": None,
        }

        if not converged:
            reason, evidence = self.classify_non_convergence()
            result["non_convergence_reason"] = reason.value
            result["non_convergence_evidence"] = evidence

        if self.drop_reasons:
            dr = self.drop_reasons
            result["drop_reason_totals"] = {
                "no_active_links": sum(d.no_active_links for d in dr),
                "trivial_flow": sum(d.trivial_flow for d in dr),
                "dag_connectivity_failed": sum(d.dag_connectivity_failed for d in dr),
                "not_dag": sum(d.not_dag for d in dr),
                "no_kept_rows": sum(d.no_kept_rows for d in dr),
                "other": sum(d.other for d in dr),
            }

        return result

    def format_convergence_table(self) -> str:
        """Render the per-iteration convergence diagnostics as a text table.

        Returns:
            One header line, then one line per iteration carrying the residual,
            the merit and its non-monotone reference, the step size, the
            line-search tries and outcome, the cache and churn counters, the
            dropped trips, the Hessian conditioning, and wall-clock seconds.
        """
        lines = []
        lines.append(
            f"{'iter':>4s}  {'||Φ-β||':>10s}  {'Q':>10s}  {'Q_ref':>10s}  "
            f"{'α':>6s}  {'ls':>3s}  {'ok':>3s}  "
            f"{'hits':>5s}  {'miss':>5s}  {'churn':>5s}  {'Δlinks':>6s}  "
            f"{'drop':>4s}  {'cond(H)':>10s}  {'λ_min(H)':>10s}  "
            f"{'t(s)':>6s}"
        )
        lines.append("-" * len(lines[0]))

        for r in self.records:
            lines.append(
                f"{r.iteration:4d}  {r.phi_residual:10.4e}  {r.Q_merit:10.4e}  "
                f"{r.Q_ref:10.4e}  "
                f"{r.step_size:6.4f}  {r.ls_tries:3d}  "
                f"{'Y' if r.accepted else 'N':>3s}  "
                f"{r.cache_hits:5d}  {r.cache_misses:5d}  "
                f"{r.active_set_changes:5d}  {r.link_flips:6d}  "
                f"{r.dropped_trips:4d}  {r.cond_H:10.2e}  {r.min_eig_H:10.2e}  "
                f"{r.wall_clock_s:6.2f}"
            )
        return "\n".join(lines)

    def format_activeset_table(self) -> str:
        """Render the per-iteration active-set diagnostics as a text table.

        Rows cover the iterations for which active-set sizes were recorded, so
        the table is empty when only hashes were collected.

        Returns:
            One header line, then one line per iteration carrying the trip
            count, the median, smallest and largest active-set size, the
            smallest and median active margin, and the churn counter.
        """
        lines = []
        lines.append(
            f"{'iter':>4s}  {'N_valid':>7s}  {'|A| med':>7s}  "
            f"{'|A| min':>7s}  {'|A| max':>7s}  "
            f"{'gap_min':>10s}  {'gap_med':>10s}  "
            f"{'churn':>5s}"
        )
        lines.append("-" * len(lines[0]))

        n_iters = min(len(self.active_set_sizes), len(self.records))

        for i in range(n_iters):
            sizes = self.active_set_sizes[i] if i < len(self.active_set_sizes) else []
            r = self.records[i]
            ms = self.margin_stats[i] if i < len(self.margin_stats) else None

            if sizes:
                sa = np.array(sizes)
                med_s, min_s, max_s = int(np.median(sa)), int(np.min(sa)), int(np.max(sa))
            else:
                med_s = min_s = max_s = 0

            gap_min = f"{ms.min_active:10.4e}" if ms else f"{'n/a':>10s}"
            gap_med = f"{ms.median_active:10.4e}" if ms else f"{'n/a':>10s}"

            lines.append(
                f"{r.iteration:4d}  {len(sizes):7d}  {med_s:7d}  "
                f"{min_s:7d}  {max_s:7d}  "
                f"{gap_min}  {gap_med}  "
                f"{r.active_set_changes:5d}"
            )
        return "\n".join(lines)


@dataclass
class ActiveSetReport:
    r"""Per-run summary of active-set behavior.

    Records the per-iteration churn, dual-slack margin distributions, and the
    non-convergence classification for one estimation run.

    Attributes:
        converged: Whether the run reached the fixed-point tolerance.
        num_iterations: Total iterations executed.
        final_residual: :math:`\lVert \Phi(\beta) - \beta \rVert` at the last
            iteration.
        beta_hat: The final parameter estimate.
        non_convergence_reason: The :class:`NonConvergenceReason` value for a
            run that failed to converge, ``None`` otherwise.
        link_flips_per_iter: Per-iteration total link flips across all trips.
        residual_per_iter: Per-iteration
            :math:`\lVert \Phi(\beta) - \beta \rVert`.
        margin_stats_per_iter: Per-iteration dual-slack margin distributions.
        per_trip_cumulative_flips: Cumulative link flips per trip, which
            identifies the trips driving most of the churn.
    """

    converged: bool
    num_iterations: int
    final_residual: float
    beta_hat: np.ndarray
    non_convergence_reason: str | None
    link_flips_per_iter: list[int]
    residual_per_iter: list[float]
    margin_stats_per_iter: list[MarginStats]
    per_trip_cumulative_flips: dict[int, int]

    def top_churning_trips(self, n: int = 20) -> list[tuple[int, int]]:
        """Return the n trips with the most cumulative link flips."""
        return sorted(
            self.per_trip_cumulative_flips.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:n]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "converged": self.converged,
            "num_iterations": self.num_iterations,
            "final_residual": self.final_residual,
            "beta_hat": self.beta_hat.tolist(),
            "non_convergence_reason": self.non_convergence_reason,
            "link_flips_per_iter": self.link_flips_per_iter,
            "residual_per_iter": self.residual_per_iter,
            "per_trip_cumulative_flips": {
                str(k): v for k, v in self.per_trip_cumulative_flips.items()
            },
        }


class ConvergenceAnalysis:
    """Aggregate active-set reports across replications and sample sizes.

    Cross-tabulates the non-convergence reasons against the sample size and
    computes summary statistics over replications.
    """

    def __init__(self, reports: dict[int, list[ActiveSetReport]]) -> None:
        """Store the per-sample-size replication reports.

        Args:
            reports: Mapping from sample size ``N`` to the list of
                per-replication reports at that size.
        """
        self.reports = reports

    def convergence_rate_by_n(self) -> dict[int, float]:
        """Fraction of converged runs for each sample size."""
        return {
            n: sum(1 for r in reps if r.converged) / len(reps)
            for n, reps in sorted(self.reports.items())
        }

    def reason_crosstab(self) -> dict[int, dict[str, int]]:
        """Cross-tabulate the non-convergence reason against sample size.

        Returns:
            For each N, counts of each non-convergence reason; converged runs
            are counted under ``"converged"``.
        """
        result: dict[int, dict[str, int]] = {}
        for n, reps in sorted(self.reports.items()):
            counts: dict[str, int] = {}
            for r in reps:
                key = "converged" if r.converged else (r.non_convergence_reason or "unknown")
                counts[key] = counts.get(key, 0) + 1
            result[n] = counts
        return result

    def churn_after_iteration(self, after_iter: int = 3) -> dict[int, float]:
        """Fraction of runs with at least one link flip after a given iteration.

        Args:
            after_iter: Only count flips from this iteration onward.

        Returns:
            For each N, the fraction of runs with a link flip after
            ``after_iter``.
        """
        result: dict[int, float] = {}
        for n, reps in sorted(self.reports.items()):
            n_with_late_churn = 0
            for r in reps:
                late_flips = r.link_flips_per_iter[after_iter:]
                if any(f > 0 for f in late_flips):
                    n_with_late_churn += 1
            result[n] = n_with_late_churn / len(reps) if reps else 0.0
        return result

    def bias_comparison(
        self,
        beta_true: np.ndarray,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Compare estimator bias between converged and non-converged runs.

        Args:
            beta_true: The true parameter vector.

        Returns:
            For each N, a dict with the converged and non-converged bias and
            RMSE arrays.
        """
        result: dict[int, dict[str, np.ndarray]] = {}
        for n, reps in sorted(self.reports.items()):
            conv = [r.beta_hat for r in reps if r.converged]
            nonc = [r.beta_hat for r in reps if not r.converged]
            entry: dict[str, np.ndarray] = {}
            if conv:
                conv_arr = np.array(conv)
                entry["converged_bias"] = np.mean(conv_arr - beta_true, axis=0)
                entry["converged_rmse"] = np.sqrt(np.mean((conv_arr - beta_true) ** 2, axis=0))
                entry["n_converged"] = len(conv)
            if nonc:
                nonc_arr = np.array(nonc)
                entry["nonconv_bias"] = np.mean(nonc_arr - beta_true, axis=0)
                entry["nonconv_rmse"] = np.sqrt(np.mean((nonc_arr - beta_true) ** 2, axis=0))
                entry["n_nonconverged"] = len(nonc)
            result[n] = entry
        return result

    def format_reason_table(self) -> str:
        """Render the non-convergence reason by sample-size cross-tab as a table."""
        crosstab = self.reason_crosstab()
        all_reasons = sorted({reason for counts in crosstab.values() for reason in counts})
        header = f"{'N':>7s}  " + "  ".join(f"{r:>15s}" for r in all_reasons)
        lines = [header, "-" * len(header)]
        for n, counts in sorted(crosstab.items()):
            row = f"{n:7d}  " + "  ".join(f"{counts.get(r, 0):15d}" for r in all_reasons)
            lines.append(row)
        return "\n".join(lines)


@dataclass
class StabilityProbeResult:
    """Result of the active-set stability probe.

    Attributes:
        n_trips: Number of trips probed.
        n_directions: Random directions drawn per radius.
        radii: Perturbation radii, in the units of the parameter norm.
        trips_changed: Radius to a per-trip list holding the largest active-set
            symmetric difference seen over the directions; a direction that
            makes the trip invalid contributes 1.
        links_flipped: Radius to the symmetric-difference sizes pooled over all
            trips and directions that changed at all.
    """

    n_trips: int
    n_directions: int
    radii: list[float]
    trips_changed: dict[float, list[int]]
    links_flipped: dict[float, list[int]]

    def format_table(self) -> str:
        """Format the probe result as a text table over perturbation radii."""
        lines = []
        lines.append(
            f"{'radius':>10s}  {'trips_changed':>14s}  {'frac':>6s}  {'links_flipped (med)':>20s}"
        )
        lines.append("-" * len(lines[0]))

        for r in self.radii:
            tc = self.trips_changed[r]
            lf = self.links_flipped[r]
            n_changed = sum(1 for x in tc if x > 0)
            frac = n_changed / self.n_trips if self.n_trips > 0 else 0.0
            med_flip = float(np.median(lf)) if lf else 0.0
            lines.append(f"{r:10.4e}  {n_changed:14d}  {frac:6.2%}  {med_flip:20.1f}")
        return "\n".join(lines)


def active_set_stability_probe(
    network,
    solver,
    beta_hat: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    active_set_config,
    projection_mode: str = "robust_pinv",
    radii: list[float] | None = None,
    n_directions: int = 5,
    seed: int = 42,
) -> StabilityProbeResult:
    r"""Perturb :math:`\hat\beta` and measure how stable the active set is.

    For every radius and every random unit direction the estimate is perturbed,
    the forward problem is re-solved for each trip, and the resulting active
    set is compared with the one at :math:`\hat\beta`. Trips whose reference
    active set is invalid are skipped, and a perturbed solve that turns invalid
    is scored as a change of size one.

    Args:
        network: Network supplying the incidence and the design ``Z``.
        solver: Forward solver for the inner QP.
        beta_hat: Parameter estimate to perturb, shape (K,).
        y: Route indicators, shape (N, L); only the trip count is read.
        b: Node imbalances, shape (N, V); each destination is its ``argmax``.
        active_set_config: Active-set detection settings.
        projection_mode: Range-space grounding mode.
        radii: Perturbation radii. Defaults to ``[1e-4, 1e-3, 1e-2, 1e-1]``
            scaled by the norm of ``beta_hat``.
        n_directions: Random unit directions drawn per radius.
        seed: Seed for drawing the directions.

    Returns:
        The per-radius active-set changes and flip counts.
    """
    from .activeset import determine_active_set

    if radii is None:
        beta_norm = float(np.linalg.norm(beta_hat))
        radii = [1e-4 * beta_norm, 1e-3 * beta_norm, 1e-2 * beta_norm, 1e-1 * beta_norm]

    K = beta_hat.shape[0]
    N = y.shape[0]
    rng = np.random.default_rng(seed)

    scale_m = solver.scale_m
    c_ref = network.Z @ beta_hat

    ref_active_sets: list[set[int] | None] = []
    for n in range(N):
        dest = int(np.argmax(b[n]))
        result = solver.solve(c_ref, b[n])
        aset = determine_active_set(
            result.x,
            dest,
            network,
            active_set_config,
            projection_mode=projection_mode,
            lam_eq=result.lam_eq,
            c=c_ref,
            scale_m=scale_m,
        )
        ref_active_sets.append(set(aset.indices) if aset.valid else None)

    directions = []
    for _ in range(n_directions):
        d = rng.standard_normal(K)
        d /= np.linalg.norm(d)
        directions.append(d)

    trips_changed: dict[float, list[int]] = {r: [] for r in radii}
    links_flipped: dict[float, list[int]] = {r: [] for r in radii}

    for radius in radii:
        per_trip_max_changes = [0] * N
        per_trip_flip_counts: list[list[int]] = [[] for _ in range(N)]

        for d in directions:
            beta_p = beta_hat + radius * d
            c_p = network.Z @ beta_p

            for n in range(N):
                if ref_active_sets[n] is None:
                    continue

                dest = int(np.argmax(b[n]))
                result_p = solver.solve(c_p, b[n])
                aset_p = determine_active_set(
                    result_p.x,
                    dest,
                    network,
                    active_set_config,
                    projection_mode=projection_mode,
                    lam_eq=result_p.lam_eq,
                    c=c_p,
                    scale_m=scale_m,
                )

                if not aset_p.valid:
                    per_trip_max_changes[n] = max(per_trip_max_changes[n], 1)
                    continue

                perturbed_set = set(aset_p.indices)
                diff = ref_active_sets[n].symmetric_difference(perturbed_set)
                if diff:
                    per_trip_max_changes[n] = max(per_trip_max_changes[n], len(diff))
                    per_trip_flip_counts[n].append(len(diff))

        trips_changed[radius] = per_trip_max_changes
        all_flips = [f for flips in per_trip_flip_counts for f in flips]
        links_flipped[radius] = all_flips

    return StabilityProbeResult(
        n_trips=N,
        n_directions=n_directions,
        radii=radii,
        trips_changed=trips_changed,
        links_flipped=links_flipped,
    )
