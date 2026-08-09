"""Forward solver protocol for the inner QP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class FlowResult:
    """Result of a forward flow solve.

    Attributes:
        x: Optimal link-flow vector, shape (L,).
        lam_eq: Equality-constraint duals (node potentials), shape (V,). The
            active-set test reads its dual slack from these.
        info: Solver diagnostics such as status, iteration count, and
            residuals; empty when the backend reports none.
    """

    x: np.ndarray
    lam_eq: np.ndarray | None = None
    # Defaults to None so __post_init__ can give every instance its own dict; a
    # dataclass field cannot carry a mutable default.
    info: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Default ``info`` to an empty dict when not supplied."""
        if self.info is None:
            self.info = {}


@runtime_checkable
class ForwardSolver(Protocol):
    r"""Protocol for forward (inner QP) solvers.

    Given link costs :math:`c` and node imbalances :math:`b`, an implementation
    solves

    .. math::

        \min_x \; c^{\top} x + F(x)
        \quad \text{s.t.} \quad
        A x = b, \; x \ge 0 ,

    where :math:`F` is the perturbation the backend implements. The in-tree
    backend takes :math:`F(x) = \tfrac12 x^{\top}
    \mathrm{diag}(\mathrm{scale}_m) x`, which is why ``scale_m`` is part of the
    protocol.
    """

    def solve(
        self,
        c: np.ndarray,
        b: np.ndarray,
        warm: Any | None = None,
    ) -> FlowResult:
        """Solve for optimal flows.

        Args:
            c: Link cost vector, shape (L,).
            b: Node imbalance vector, shape (V,).
            warm: Optional warm-start data from a previous solve.

        Returns:
            FlowResult with optimal flows and duals.
        """
        ...

    @property
    def scale_m(self) -> np.ndarray:
        """Per-link perturbation scale (e.g., link length), shape (L,)."""
        ...
