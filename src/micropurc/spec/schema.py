r"""Declarative, serializable model specification.

A :class:`ModelSpec` describes a linear-in-parameters utility as an ordered
list of :class:`Term` objects. Each term contributes to one parameter's design
column, and the utility is :math:`Z\beta` with one coefficient per distinct
parameter name.

A term's regressor is built from named link attributes (a product of several
gives an interaction), an optional elementwise transform, an optional scale,
and an optional segmentation indicator on a per-variant covariate. All parts
transform the *data*; the parameter enters linearly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPS = ("<=", ">=", "==", "<", ">")
TRANSFORMS = ("identity", "log", "power", "standardize")


@dataclass(frozen=True)
class Segment:
    """Indicator gate ``1[covariate op threshold]`` on a per-variant covariate.

    Attributes:
        covariate: Name of the per-variant covariate to test.
        op: Comparison, one of ``OPS``.
        threshold: Right-hand side of the comparison.
    """

    covariate: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        """Validate the comparison operator."""
        if self.op not in OPS:
            raise ValueError(f"op must be one of {OPS}, got {self.op!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {"covariate": self.covariate, "op": self.op, "threshold": self.threshold}


@dataclass(frozen=True)
class Term:
    r"""One additive utility term mapped to a single parameter.

    Attributes:
        parameter: Name of the coefficient :math:`\beta`. Terms sharing a name
            sum into the same design column.
        attributes: Named attributes multiplied together (interaction).
        transform: Elementwise transform applied to the attribute product.
        power: Exponent used when ``transform == "power"``.
        segment: Optional covariate indicator gating the term.
        scale: Constant multiplier applied to the regressor.
    """

    parameter: str
    attributes: tuple[str, ...]
    transform: str = "identity"
    power: float = 1.0
    segment: Segment | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the attribute list and the transform.

        A term references at least one attribute, its transform is one of
        ``TRANSFORMS``, and ``power`` applies to a single attribute.
        """
        if not self.attributes:
            raise ValueError("a term must reference at least one attribute")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}, got {self.transform!r}")
        if self.transform == "power" and len(self.attributes) != 1:
            raise ValueError("power transform applies to a single attribute")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation, omitting fields left at default."""
        d: dict[str, Any] = {
            "parameter": self.parameter,
            "attributes": list(self.attributes),
        }
        if self.transform != "identity":
            d["transform"] = self.transform
        if self.transform == "power":
            d["power"] = self.power
        if self.segment is not None:
            d["segment"] = self.segment.to_dict()
        if self.scale != 1.0:
            d["scale"] = self.scale
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Term:
        """Build a term from its dict representation."""
        seg = d.get("segment")
        return cls(
            parameter=d["parameter"],
            attributes=tuple(d["attributes"]),
            transform=d.get("transform", "identity"),
            power=d.get("power", 1.0),
            segment=Segment(**seg) if seg is not None else None,
            scale=d.get("scale", 1.0),
        )


@dataclass
class ModelSpec:
    """An ordered collection of utility terms.

    Attributes:
        terms: The terms, in the order they were declared.
    """

    terms: list[Term] = field(default_factory=list)

    def parameters(self) -> list[str]:
        """Distinct parameter names in order of first appearance.

        This order fixes the design's column order, so it also fixes the order
        of the estimated coefficients.
        """
        seen: dict[str, None] = {}
        for t in self.terms:
            seen.setdefault(t.parameter, None)
        return list(seen)

    def covariates(self) -> list[str]:
        """Distinct segmentation covariate names, in order of first appearance."""
        seen: dict[str, None] = {}
        for t in self.terms:
            if t.segment is not None:
                seen.setdefault(t.segment.covariate, None)
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {"terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        """Build a spec from its dict representation."""
        return cls(terms=[Term.from_dict(t) for t in d["terms"]])

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, source: str | Path) -> ModelSpec:
        """Load a spec from a JSON string or a file path.

        A :class:`~pathlib.Path` is always read from disk. A ``str`` is read
        from disk when it names an existing file and parsed as JSON text
        otherwise, so a document containing a newline is always treated as
        text.
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            s = str(source)
            try:
                is_file = "\n" not in s and len(s) < 4096 and Path(s).is_file()
            except OSError:
                is_file = False
            text = Path(s).read_text(encoding="utf-8") if is_file else s
        return cls.from_dict(json.loads(text))
