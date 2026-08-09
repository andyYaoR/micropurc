r"""Active-set determination for a single trip.

A link carries flow when its dual slack is negative: with link costs
:math:`c = Z\beta` and the node potentials :math:`\lambda` returned by the
forward solve, link :math:`j` is active when

.. math::

    s_j = c_j + (A^{\top}\lambda)_j < 0 .

The C++ extension performs the slack test, the acyclicity and
destination-reachability checks on the active subnetwork, and the
component-aware grounding that makes :math:`\tilde A \tilde A^{\top}`
invertible. This module adapts numpy and :class:`~micropurc.network.Network`
objects to those calls and keeps the Python result types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .._native import (
    active_threshold as _native_active_threshold,
)
from .._native import (
    check_dag_connectivity as _native_check_dag_connectivity,
)
from .._native import (
    determine_active_set as _native_determine_active_set,
)
from .._native import (
    find_kept_rows as _native_find_kept_rows,
)
from ..config import ActiveSetConfig
from ..network import Network


@dataclass
class ActiveSet:
    r"""Result of active-set determination for a single trip.

    Attributes:
        indices: Active (flow-carrying) link ids, those with :math:`s_j < 0`.
        slack: Dual slack :math:`s_j = c_j + (A^{\top}\lambda)_j` for every
            link, shape (L,).
        margins: Per-link margin :math:`|s_j| / \lVert Z_j \rVert`, shape (L,).
            Slack is affine in :math:`\beta` through :math:`c = Z\beta`, so with
            the node potentials held fixed this is the length of the smallest
            parameter perturbation that flips link :math:`j` between active and
            inactive. Links with :math:`\lVert Z_j \rVert = 0` carry ``inf``, so
            the trip's margin is the minimum of the vector. ``None`` unless
            ``z_rownorm`` is supplied to :func:`determine_active_set`.
        kept_node_rows: Node rows of :math:`A` retained for the range-space
            projection, one row per connected component having been dropped.
        num_components: Number of connected components in the active subnetwork.
        valid: Whether the trip yields a usable active set.
        drop_reason: Why the trip was dropped, if ``valid`` is False.
    """

    indices: np.ndarray
    slack: np.ndarray
    margins: np.ndarray | None
    kept_node_rows: np.ndarray
    num_components: int
    valid: bool
    drop_reason: str = ""


def active_threshold(x: np.ndarray, config: ActiveSetConfig) -> float:
    """Compute the flow magnitude below which a link counts as carrying nothing.

    The rule is ``max(threshold_min, threshold_factor * x.max())``, so the
    threshold follows the scale of the flow vector; an empty ``x`` gives
    ``threshold_min``. Route sampling uses this to clean numerically negligible
    flows before walking the support. Estimation reads activity from dual slack.

    Args:
        x: Link flows, shape (L,).
        config: Supplies ``threshold_factor`` and ``threshold_min``.

    Returns:
        The threshold, in flow units.
    """
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    return float(
        _native_active_threshold(
            x_arr,
            float(config.threshold_factor),
            float(config.threshold_min),
        )
    )


def _infer_v(tail_active: np.ndarray, head_active: np.ndarray, dest: int) -> int:
    """Node-array length the native calls need: one past the largest node id used."""
    if tail_active.size == 0 and head_active.size == 0:
        return max(int(dest) + 1, 0)
    max_node = int(max(np.max(tail_active), np.max(head_active), dest))
    return max_node + 1


def _check_dag_connectivity(
    tail_active: np.ndarray,
    head_active: np.ndarray,
    dest: int,
) -> tuple[bool, bool]:
    """Test the active subnetwork for acyclicity and destination reachability.

    Returns ``(is_dag, is_connected)``, where ``is_connected`` holds when every
    node touched by an active edge reaches ``dest`` along active edges.
    """
    tail = np.ascontiguousarray(np.asarray(tail_active, dtype=np.int64).ravel())
    head = np.ascontiguousarray(np.asarray(head_active, dtype=np.int64).ravel())
    if tail.shape[0] != head.shape[0]:
        raise ValueError("tail_active and head_active must have the same length")
    is_dag, is_connected = _native_check_dag_connectivity(
        tail, head, int(dest), _infer_v(tail, head, int(dest))
    )
    return bool(is_dag), bool(is_connected)


def _find_kept_rows_robust(
    tail_active: np.ndarray,
    head_active: np.ndarray,
    dest: int,
) -> tuple[np.ndarray, int] | None:
    """Ground the active incidence by dropping one node row per component.

    Flow conservation is redundant within each connected component of the
    active subnetwork, so one row is removed from each: the destination's row in
    the component that contains ``dest``, and the smallest node id elsewhere.
    Returns the kept rows and the component count, or ``None`` when the native
    routine reports the trip ungroundable.
    """
    tail = np.ascontiguousarray(np.asarray(tail_active, dtype=np.int64).ravel())
    head = np.ascontiguousarray(np.asarray(head_active, dtype=np.int64).ravel())
    kept, num_components, valid = _native_find_kept_rows(
        tail, head, int(dest), _infer_v(tail, head, int(dest)), "robust_pinv"
    )
    if not valid:
        return None
    return np.asarray(kept, dtype=int), int(num_components)


def determine_active_set(
    x: np.ndarray,
    dest: int,
    network: Network,
    config: ActiveSetConfig,
    projection_mode: str = "robust_pinv",
    lam_eq: np.ndarray | None = None,
    c: np.ndarray | None = None,
    scale_m: np.ndarray | None = None,
    z_rownorm: np.ndarray | None = None,
) -> ActiveSet:
    r"""Determine the active set for a single trip.

    A link is active when :math:`s_j = c_j + (A^{\top}\lambda)_j` falls below
    ``-config.slack_tol``. The active subnetwork is then checked for acyclicity
    and for reachability of ``dest``, and grounded one node row per connected
    component; a trip that fails any of these is returned with ``valid=False``
    and a ``drop_reason``.

    Args:
        x: Optimal link flows (unused; kept for call-site symmetry).
        dest: Destination node id.
        network: The network.
        config: Active-set detection settings (``slack_tol``).
        projection_mode: Range-space grounding mode (``robust_pinv``).
        lam_eq: Equality-constraint duals (node potentials) from the solve.
        c: Link cost vector for this trip (``Z_variant @ beta``).
        scale_m: Per-link perturbation scale. Accepted by the native signature
            and unread by the slack test.
        z_rownorm: Per-link design-row norms :math:`\lVert Z_j \rVert` for this
            trip's variant. When given, the returned :class:`ActiveSet` carries
            the margins :math:`|s_j| / \lVert Z_j \rVert`; otherwise
            ``margins`` is ``None``.

    Returns:
        The trip's active set.

    Raises:
        ValueError: If ``lam_eq``, ``c``, or ``scale_m`` is missing, or if
            ``z_rownorm`` has a length other than the number of links.
    """
    del x  # activity is read off the dual slack computed natively below.
    if lam_eq is None or c is None or scale_m is None:
        raise ValueError("determine_active_set requires lam_eq, c, and scale_m")

    A_csc = network.A if sp.isspmatrix_csc(network.A) else network.A.tocsc()
    result = _native_determine_active_set(
        np.ascontiguousarray(A_csc.indptr, dtype=np.int64),
        np.ascontiguousarray(A_csc.indices, dtype=np.int64),
        np.ascontiguousarray(A_csc.data, dtype=np.float64),
        np.ascontiguousarray(network.tail, dtype=np.int64),
        np.ascontiguousarray(network.head, dtype=np.int64),
        int(network.num_nodes),
        int(network.num_links),
        int(dest),
        np.ascontiguousarray(np.asarray(c, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(lam_eq, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(scale_m, dtype=np.float64).ravel()),
        float(config.slack_tol),
        projection_mode,
    )

    slack = np.asarray(result["slack"], dtype=float)
    margins: np.ndarray | None = None
    if z_rownorm is not None:
        zr = np.ascontiguousarray(np.asarray(z_rownorm, dtype=np.float64).ravel())
        if zr.shape[0] != slack.shape[0]:
            raise ValueError("z_rownorm must have length equal to the number of links")
        margins = np.full(slack.shape, np.inf, dtype=float)
        ok = zr > 0.0
        margins[ok] = np.abs(slack[ok]) / zr[ok]

    return ActiveSet(
        indices=np.asarray(result["indices"], dtype=int),
        slack=slack,
        margins=margins,
        kept_node_rows=np.asarray(result["kept_node_rows"], dtype=int),
        num_components=int(result["num_components"]),
        valid=bool(result["valid"]),
        drop_reason=str(result["drop_reason"]),
    )
