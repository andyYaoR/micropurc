"""Model specification: declare linear-in-parameters utilities and compile them.

Two equivalent front-ends produce a :class:`~micropurc.spec.schema.ModelSpec`:
a serializable declarative schema, and a Python DSL
(:func:`beta`/:func:`attr`/:func:`seg`/:func:`log`/:func:`power`/:func:`standardize`).
:func:`compile_design` binds a spec to a network's named attributes and
produces the estimation design.
"""

from .compile import CompiledDesign, compile_design
from .dsl import Utility, attr, beta, log, power, seg, standardize
from .schema import ModelSpec, Segment, Term

__all__ = [
    "ModelSpec",
    "Term",
    "Segment",
    "Utility",
    "beta",
    "attr",
    "seg",
    "log",
    "power",
    "standardize",
    "compile_design",
    "CompiledDesign",
]
