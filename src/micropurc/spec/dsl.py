"""Symbolic builders for model specifications.

A small operator-overloaded DSL that lowers to :class:`~micropurc.spec.schema.ModelSpec`.
Example::

    U = (beta("time") * attr("travel_time")
         + beta("fee_low") * attr("fee") * seg("group", "==", 0)
         + beta("logdist") * log(attr("length")))
    spec = U.to_spec()

Each summand is ``beta(name) * <attribute expression> [* seg(...)]``, optionally
scaled by a constant. Attribute expressions may be a product of attributes
(interaction) and may carry a scalar transform (``log``, ``power``, ``standardize``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .schema import ModelSpec, Segment, Term


@dataclass(frozen=True)
class AttrExpr:
    """A named attribute, optionally with an elementwise transform.

    Attributes:
        name: Attribute name, resolved against the network at compile time.
        transform: One of the transforms the schema accepts.
        power: Exponent used when ``transform == "power"``.
    """

    name: str
    transform: str = "identity"
    power: float = 1.0


@dataclass(frozen=True)
class Beta:
    """A parameter placeholder; multiply by an attribute expression to form a term."""

    name: str

    def __mul__(self, other: AttrExpr | ProductExpr) -> TermExpr:
        """Bind this parameter to an attribute expression, starting a term."""
        return TermExpr(parameter=self.name).__mul__(other)


@dataclass(frozen=True)
class ProductExpr:
    """A product (interaction) of attribute expressions."""

    factors: tuple[AttrExpr, ...]

    def __mul__(self, other: AttrExpr | ProductExpr) -> ProductExpr:
        """Extend the interaction with another attribute factor."""
        return ProductExpr(self.factors + _factors(other))


@dataclass(frozen=True)
class TermExpr:
    """A utility term under construction: a parameter, attribute factors, a segment.

    Attributes:
        parameter: Name of the coefficient this term loads on.
        attributes: Attribute factors multiplied together.
        segment: Covariate indicator gating the term, at most one.
        scale: Constant multiplier accumulated from scalar factors.
    """

    parameter: str
    attributes: tuple[AttrExpr, ...] = ()
    segment: Segment | None = None
    scale: float = 1.0

    def __mul__(self, other: object) -> TermExpr:
        """Attach an attribute factor, a segment, or a scalar scale to the term."""
        if isinstance(other, (AttrExpr, ProductExpr)):
            return replace(self, attributes=self.attributes + _factors(other))
        if isinstance(other, Segment):
            if self.segment is not None:
                raise ValueError("a term may carry at most one segment")
            return replace(self, segment=other)
        if isinstance(other, (int, float)):
            return replace(self, scale=self.scale * float(other))
        return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other: TermExpr | Utility) -> Utility:
        """Sum with another term or utility to form a :class:`Utility`."""
        return Utility([self]).__add__(other)

    def to_spec(self) -> ModelSpec:
        """Lower a single-term utility to a :class:`~micropurc.spec.schema.ModelSpec`."""
        return Utility([self]).to_spec()

    def to_term(self) -> Term:
        """Lower to a schema :class:`~micropurc.spec.schema.Term`.

        A term needs at least one attribute. A transform is carried over only
        from a single-attribute term, since an interaction leaves the target of
        the transform ambiguous.
        """
        if not self.attributes:
            raise ValueError(f"parameter '{self.parameter}' has no attribute")
        transforms = {a.transform for a in self.attributes}
        if transforms == {"identity"}:
            transform, power = "identity", 1.0
        elif len(self.attributes) == 1:
            transform, power = self.attributes[0].transform, self.attributes[0].power
        else:
            raise ValueError("transforms are only supported on single-attribute terms")
        return Term(
            parameter=self.parameter,
            attributes=tuple(a.name for a in self.attributes),
            transform=transform,
            power=power,
            segment=self.segment,
            scale=self.scale,
        )


@dataclass(frozen=True)
class Utility:
    """A sum of terms; lower it to a :class:`~micropurc.spec.schema.ModelSpec`.

    Attributes:
        terms: The summands, in declaration order.
    """

    terms: list[TermExpr]

    def __add__(self, other: TermExpr | Utility) -> Utility:
        """Append a term, or concatenate another utility's terms."""
        if isinstance(other, TermExpr):
            return Utility(self.terms + [other])
        if isinstance(other, Utility):
            return Utility(self.terms + other.terms)
        return NotImplemented

    def to_spec(self) -> ModelSpec:
        """Lower to a :class:`~micropurc.spec.schema.ModelSpec`."""
        return ModelSpec(terms=[t.to_term() for t in self.terms])


def _factors(x: AttrExpr | ProductExpr) -> tuple[AttrExpr, ...]:
    return x.factors if isinstance(x, ProductExpr) else (x,)


def beta(name: str) -> Beta:
    """A parameter placeholder named ``name``."""
    return Beta(name)


def attr(name: str) -> AttrExpr:
    """A named link attribute."""
    return AttrExpr(name)


def seg(covariate: str, op: str, threshold: float) -> Segment:
    """A covariate indicator ``1[covariate op threshold]``."""
    return Segment(covariate=covariate, op=op, threshold=threshold)


def log(a: AttrExpr) -> AttrExpr:
    """Natural-log transform of an attribute."""
    return replace(a, transform="log")


def power(a: AttrExpr, p: float) -> AttrExpr:
    """Raise an attribute to the power ``p``."""
    return replace(a, transform="power", power=p)


def standardize(a: AttrExpr) -> AttrExpr:
    """Standardize an attribute to zero mean and unit variance.

    The moments are taken over every entry of the attribute, pooling the
    attribute-variant axis when there is one.
    """
    return replace(a, transform="standardize")
