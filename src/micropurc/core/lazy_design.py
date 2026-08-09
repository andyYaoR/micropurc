"""Lazy, term-coded design matrix for large problems.

A :class:`LazyDesign` holds per-link attribute columns, per-variant covariates,
and a compact table of *terms* that the native engine evaluates on the fly, so
the ``(N, L, K)`` design is never held in memory. Each term contributes one
parameter's column, evaluated per link and per trip variant:

- ``TERM_COLUMN(a)`` — link attribute column ``a``;
- ``TERM_PRODUCT(a, b)`` — product of columns ``a`` and ``b`` (interaction);
- ``TERM_COVARIATE_INDICATOR(a, cov, op, thr)`` — column ``a`` gated by the
  indicator ``1[covariate[cov] op thr]``.

Link attributes may vary across an *attribute-variant* axis (for instance,
time-of-day); ``variant_attr`` maps each trip variant to its attribute-variant
row, and ``variant_covariates`` holds the covariate values used by indicator
terms. Build one directly, or via :func:`micropurc.spec.compile_design` with
``materialize=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERM_COLUMN = 0
TERM_PRODUCT = 1
TERM_COVARIATE_INDICATOR = 2

COV_LE, COV_GE, COV_EQ, COV_LT, COV_GT = 0, 1, 2, 3, 4
OP_CODE = {"<=": COV_LE, ">=": COV_GE, "==": COV_EQ, "<": COV_LT, ">": COV_GT}


@dataclass
class LazyDesign:
    """A term-coded design evaluated lazily by the native engine.

    Attributes:
        link_attrs: Attribute columns, shape ``(n_attr_variants, L, A)``.
        variant_attr: Attribute-variant row for each variant, shape ``(n_var,)``.
        variant_covariates: Covariate values per variant, shape ``(n_var, C)``.
        trip_variant: Variant index for each trip, shape ``(N,)``.
        term_code: Opcode per term, shape ``(K,)``.
        term_a: Primary attribute column per term, shape ``(K,)``.
        term_b: Second column for product terms, shape ``(K,)``.
        term_cov: Covariate index for indicator terms, shape ``(K,)``.
        term_op: Comparison code for indicator terms, shape ``(K,)``.
        term_thr: Threshold for indicator terms, shape ``(K,)``.
    """

    link_attrs: np.ndarray
    variant_attr: np.ndarray
    variant_covariates: np.ndarray
    trip_variant: np.ndarray
    term_code: np.ndarray
    term_a: np.ndarray
    term_b: np.ndarray
    term_cov: np.ndarray
    term_op: np.ndarray
    term_thr: np.ndarray

    def __post_init__(self) -> None:
        """Coerce arrays to contiguous native dtypes and validate their shapes."""
        self.link_attrs = np.ascontiguousarray(self.link_attrs, dtype=np.float64)
        if self.link_attrs.ndim != 3:
            raise ValueError("link_attrs must have shape (n_attr_variants, L, A)")
        self.variant_attr = np.ascontiguousarray(self.variant_attr, dtype=np.int64)
        self.variant_covariates = np.ascontiguousarray(self.variant_covariates, dtype=np.float64)
        if self.variant_covariates.ndim != 2:
            raise ValueError("variant_covariates must have shape (n_variants, C)")
        self.trip_variant = np.ascontiguousarray(self.trip_variant, dtype=np.int64)
        for name in ("term_code", "term_a", "term_b", "term_cov", "term_op"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.int64))
        self.term_thr = np.ascontiguousarray(self.term_thr, dtype=np.float64)
        k = self.term_code.shape[0]
        for name in ("term_a", "term_b", "term_cov", "term_op", "term_thr"):
            if getattr(self, name).shape[0] != k:
                raise ValueError(f"{name} must have length K={k}")
        if self.variant_attr.shape[0] != self.variant_covariates.shape[0]:
            raise ValueError("variant_attr and variant_covariates disagree on n_variants")

    @property
    def num_attributes(self) -> int:
        """Number of parameters (one per term)."""
        return int(self.term_code.shape[0])

    @property
    def num_links(self) -> int:
        """Number of links."""
        return int(self.link_attrs.shape[1])

    @property
    def num_variants(self) -> int:
        """Number of trip variants."""
        return int(self.variant_attr.shape[0])
