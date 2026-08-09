r"""One trip's contribution to the estimating equations, via a Schur complement.

Every contribution is a contraction with the range-space projector

.. math::

    \hat P = I - \tilde A^{\top}
    \big(\tilde A \tilde A^{\top}\big)^{-1} \tilde A ,

where :math:`\tilde A` is the trip's active incidence submatrix after one node
row per connected component has been grounded. The C++ extension applies
:math:`\hat P` through a Cholesky factor of the :math:`m \times m` Gram matrix
:math:`\tilde A \tilde A^{\top}`, so the :math:`e \times e` projector itself is
never formed. This module adapts numpy arrays to that call and names the three
outputs it returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._native import schur_contribution as _native_schur_contribution


@dataclass
class TripContribution:
    r"""Per-trip contribution to the estimator.

    Attributes:
        H_i: Hessian block :math:`Z_{\mathrm{act}}^{\top} \hat P
            Z_{\mathrm{act}}`, shape (K, K).
        g_i: Gradient block :math:`Z_{\mathrm{act}}^{\top} \hat P
            (\mathrm{scale} \odot y)`, shape (K,).
        u_i: Score :math:`Z_{\mathrm{act}}^{\top} \hat P \big(\mathrm{scale}
            \odot (y - \hat x)\big)`, shape (K,); ``None`` when the flows were
            withheld.
    """

    H_i: np.ndarray
    g_i: np.ndarray
    u_i: np.ndarray | None


def trip_contribution_schur(
    Z_act: np.ndarray,
    A_local: np.ndarray,
    y_act: np.ndarray,
    scale_act: np.ndarray,
    x_hat_act: np.ndarray | None = None,
) -> TripContribution:
    r"""Compute one trip's ``(H_i, g_i, u_i)`` via the native Schur solve.

    Args:
        Z_act: Link attributes on active edges, shape (e, K).
        A_local: Active incidence submatrix, shape (m, e), dense, with one node
            row per connected component already dropped so that
            :math:`\tilde A \tilde A^{\top}` is invertible.
        y_act: Route indicator on active edges, shape (e,).
        scale_act: Perturbation scale on active edges, shape (e,).
        x_hat_act: Optimal flows on active edges, shape (e,). Supplying it
            requests the score ``u_i``; ``None`` leaves ``u_i`` unset and skips
            the extra right-hand side in the Schur solve.

    Returns:
        The trip's ``H_i``, ``g_i``, and ``u_i``.
    """
    x_arg = (
        None
        if x_hat_act is None
        else np.ascontiguousarray(np.asarray(x_hat_act, dtype=np.float64).ravel())
    )
    H_i, g_i, u_i = _native_schur_contribution(
        np.ascontiguousarray(np.asarray(Z_act, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(A_local, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(y_act, dtype=np.float64).ravel()),
        np.ascontiguousarray(np.asarray(scale_act, dtype=np.float64).ravel()),
        x_arg,
    )
    return TripContribution(
        H_i=np.asarray(H_i, dtype=float),
        g_i=np.asarray(g_i, dtype=float),
        u_i=None if u_i is None else np.asarray(u_i, dtype=float),
    )
