"""Compile a :class:`~micropurc.spec.schema.ModelSpec` into a design matrix.

The compiler binds attribute names to data and evaluates each term into the
design. Attribute values may be static (columns of ``network.Z``) or vary
across an *attribute-variant* axis (supplied via ``variant_attrs``, e.g. a
time-of-day-dependent link time); segmented terms additionally gate on
per-variant covariates.

Without variant structure the design is a single matrix ``Z`` of shape
``(L, K)``. With attribute variance and/or segmentation it is a stack of
per-variant matrices ``z_variants`` of shape ``(n_variants, L, K)`` selected by
``trip_variant``, or a lazy design (``materialize=False``). ``K`` is the number
of distinct parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.lazy_design import OP_CODE, TERM_COLUMN, TERM_COVARIATE_INDICATOR, LazyDesign
from ..network import Network
from .schema import ModelSpec, Segment, Term


@dataclass
class CompiledDesign:
    """A compiled design ready for estimation.

    Exactly one of ``Z`` (no variant structure), ``z_variants`` (variant
    structure, materialized), or ``lazy`` (``materialize=False``) is set.

    Attributes:
        parameters: Ordered parameter names; column ``k`` of the design is the
            regressor for ``parameters[k]``.
        Z: Design matrix, shape ``(L, K)`` when there is no variant structure.
        z_variants: Per-variant designs, shape ``(n_variants, L, K)``.
        trip_variant: Variant index per trip, shape ``(N,)``.
        lazy: Lazy design, set when ``materialize=False``.
    """

    parameters: list[str]
    Z: np.ndarray | None = None
    z_variants: np.ndarray | None = None
    trip_variant: np.ndarray | None = None
    lazy: LazyDesign | None = None


def _attr_columns(
    name: str,
    network: Network,
    variant_attrs: dict[str, np.ndarray] | None,
    n_attr_var: int,
    L: int,
) -> np.ndarray:
    """Per-attribute-variant columns of a named attribute, shape (n_attr_var, L).

    Static network attributes are broadcast across attribute-variants;
    variant-varying attributes are taken from ``variant_attrs``.
    """
    if variant_attrs is not None and name in variant_attrs:
        arr = np.asarray(variant_attrs[name], dtype=float)
        if arr.shape != (n_attr_var, L):
            raise ValueError(
                f"variant attribute '{name}' has shape {arr.shape}, expected {(n_attr_var, L)}"
            )
        return arr
    col = network.Z[:, network.attribute_index(name)]
    return np.broadcast_to(col, (n_attr_var, L)).copy()


def _term_regressor(
    term: Term,
    network: Network,
    variant_attrs: dict[str, np.ndarray] | None,
    n_attr_var: int,
) -> np.ndarray:
    """Per-attribute-variant regressor for a term, shape (n_attr_var, L).

    Products of attributes are taken elementwise; scalar transforms act
    elementwise (``standardize`` uses the mean and std over all entries).
    """
    L = network.num_links
    cols = [_attr_columns(a, network, variant_attrs, n_attr_var, L) for a in term.attributes]
    base = cols[0] if len(cols) == 1 else np.prod(cols, axis=0)
    if term.transform == "log":
        base = np.log(base)
    elif term.transform == "power":
        base = base**term.power
    elif term.transform == "standardize":
        sd = base.std()
        base = (base - base.mean()) / sd if sd > 0 else base - base.mean()
    return term.scale * base


def _segment_indicator(
    segment: Segment,
    variant_covariates: np.ndarray,
    covariate_names: list[str],
) -> np.ndarray:
    """Indicator over variants for a segmentation condition, shape (n_variants,)."""
    try:
        c = covariate_names.index(segment.covariate)
    except ValueError:
        raise KeyError(f"covariate '{segment.covariate}' not in {covariate_names}") from None
    col = variant_covariates[:, c]
    thr = segment.threshold
    cmp = {
        "<=": col <= thr,
        ">=": col >= thr,
        "==": col == thr,
        "<": col < thr,
        ">": col > thr,
    }[segment.op]
    return cmp.astype(float)


def _n_attr_var(variant_attrs: dict[str, np.ndarray] | None) -> int:
    """Number of attribute-variants implied by ``variant_attrs`` (1 if none)."""
    if not variant_attrs:
        return 1
    shapes = {np.asarray(v).shape[0] for v in variant_attrs.values()}
    if len(shapes) != 1:
        raise ValueError("all variant_attrs arrays must share the same n_attr_var")
    return shapes.pop()


def compile_design(
    spec: ModelSpec,
    network: Network,
    *,
    variant_covariates: np.ndarray | None = None,
    covariate_names: list[str] | None = None,
    trip_variant: np.ndarray | None = None,
    variant_attrs: dict[str, np.ndarray] | None = None,
    variant_attr_index: np.ndarray | None = None,
    materialize: bool = True,
) -> CompiledDesign:
    """Compile a model specification against a network.

    Args:
        spec: The model specification.
        network: Network supplying named static link attributes.
        variant_covariates: Per-variant covariate values, shape
            ``(n_variants, C)``. Required if the spec has segmented terms.
        covariate_names: Names of the covariate columns, length ``C``.
        trip_variant: Variant index per trip, shape ``(N,)``. Required whenever
            the design is variant-specific.
        variant_attrs: Optional mapping ``name -> array (n_attr_var, L)`` of
            attributes whose values vary across an attribute-variant axis.
        variant_attr_index: Attribute-variant row per variant, shape
            ``(n_variants,)``. Required if ``variant_attrs`` is given.
        materialize: If True, build explicit design matrices; if False, build a
            :class:`~micropurc.core.lazy_design.LazyDesign`.

    Returns:
        The compiled design.

    Raises:
        ValueError: If the variant structure inputs required by the spec are
            missing or mutually inconsistent.
    """
    params = spec.parameters()
    pidx = {p: k for k, p in enumerate(params)}
    K = len(params)
    L = network.num_links
    segmented = any(t.segment is not None for t in spec.terms)
    n_attr_var = _n_attr_var(variant_attrs)
    variant_structure = segmented or variant_attrs is not None

    if not variant_structure:
        if not materialize:
            return _compile_lazy(spec, network, None, [], None, None, 1)
        Z = np.zeros((L, K), dtype=float)
        for t in spec.terms:
            Z[:, pidx[t.parameter]] += _term_regressor(t, network, None, 1)[0]
        return CompiledDesign(parameters=params, Z=Z)

    # Resolve the variant structure shared by the explicit and lazy paths.
    if variant_attrs is not None:
        if variant_attr_index is None or trip_variant is None:
            raise ValueError("variant_attrs requires variant_attr_index and trip_variant")
        av_index = np.asarray(variant_attr_index, dtype=int)
        n_var = av_index.shape[0]
    else:
        if variant_covariates is None or trip_variant is None:
            raise ValueError("segmented spec requires variant_covariates and trip_variant")
        n_var = variant_covariates.shape[0]
        av_index = np.zeros(n_var, dtype=int)

    if segmented and (variant_covariates is None or covariate_names is None):
        raise ValueError("segmented spec requires variant_covariates and covariate_names")
    if variant_covariates is not None and variant_covariates.shape[0] != n_var:
        raise ValueError("variant_covariates and variant_attr_index disagree on n_variants")

    if not materialize:
        return _compile_lazy(
            spec,
            network,
            variant_covariates,
            covariate_names or [],
            trip_variant,
            variant_attrs,
            n_attr_var,
            av_index,
        )

    Zv = np.zeros((n_var, L, K), dtype=float)
    for t in spec.terms:
        R = _term_regressor(t, network, variant_attrs, n_attr_var)  # (n_attr_var, L)
        R_var = R[av_index, :]  # (n_var, L)
        if t.segment is None:
            ind = np.ones(n_var)
        else:
            ind = _segment_indicator(t.segment, variant_covariates, covariate_names)
        Zv[:, :, pidx[t.parameter]] += R_var * ind[:, None]
    return CompiledDesign(
        parameters=params,
        z_variants=Zv,
        trip_variant=np.asarray(trip_variant, dtype=int),
    )


def _compile_lazy(
    spec: ModelSpec,
    network: Network,
    variant_covariates: np.ndarray | None,
    covariate_names: list[str],
    trip_variant: np.ndarray | None,
    variant_attrs: dict[str, np.ndarray] | None,
    n_attr_var: int,
    av_index: np.ndarray | None = None,
) -> CompiledDesign:
    """Compile a spec into a :class:`~micropurc.core.lazy_design.LazyDesign`.

    Each parameter maps to exactly one term. Every term's attribute product and
    scalar transform is precomputed into one link-attribute column per
    attribute-variant, and a segmented term becomes a covariate indicator, so
    the native engine only has to gate and select columns.
    """
    params = spec.parameters()
    if len(params) != len(spec.terms):
        raise ValueError("lazy compilation requires one term per parameter")
    if trip_variant is None:
        raise ValueError("lazy compilation requires trip_variant")
    trip_variant = np.asarray(trip_variant, dtype=int)

    if av_index is None:
        n_var = int(trip_variant.max()) + 1 if trip_variant.size else 1
        av_index = np.zeros(n_var, dtype=int)
    else:
        av_index = np.asarray(av_index, dtype=int)
        n_var = av_index.shape[0]

    if variant_covariates is None:
        variant_covariates = np.zeros((n_var, 1))
        covariate_names = []

    L = network.num_links
    K = len(spec.terms)
    base = np.zeros((n_attr_var, L, K), dtype=float)
    code = np.zeros(K, dtype=int)
    term_a = np.arange(K, dtype=int)
    term_b = np.zeros(K, dtype=int)
    cov = np.zeros(K, dtype=int)
    op = np.zeros(K, dtype=int)
    thr = np.zeros(K, dtype=float)
    for k, t in enumerate(spec.terms):
        base[:, :, k] = _term_regressor(t, network, variant_attrs, n_attr_var)
        if t.segment is None:
            code[k] = TERM_COLUMN
        else:
            code[k] = TERM_COVARIATE_INDICATOR
            if t.segment.covariate not in covariate_names:
                raise KeyError(f"covariate '{t.segment.covariate}' not in {covariate_names}")
            cov[k] = covariate_names.index(t.segment.covariate)
            op[k] = OP_CODE[t.segment.op]
            thr[k] = t.segment.threshold

    lazy = LazyDesign(
        link_attrs=base,
        variant_attr=av_index,
        variant_covariates=variant_covariates,
        trip_variant=trip_variant,
        term_code=code,
        term_a=term_a,
        term_b=term_b,
        term_cov=cov,
        term_op=op,
        term_thr=thr,
    )
    return CompiledDesign(parameters=params, lazy=lazy)
