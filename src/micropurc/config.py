"""Configuration for micropurc estimation runs (JSON and TOML).

JSON is the canonical format: human-readable and loadable straight into a
:class:`RunConfig`. TOML loading is kept for backward compatibility with
existing configuration files.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class ActiveSetConfig:
    r"""Active-set determination parameters.

    A link is active when its dual slack (the KKT sign test) is negative:

    .. math::

        s_j = c_j + (A^{\top}\lambda)_j < 0 .

    Attributes:
        threshold_factor: Multiplier in the flow-magnitude threshold
            ``max(threshold_min, threshold_factor * x.max())`` that route
            sampling uses to clean numerically negligible flows.
        threshold_min: Floor of that same threshold.
        slack_tol: A positive value tightens the cutoff above to
            :math:`-\texttt{slack\_tol}`. At the default 0.0 a link is active
            when its slack is strictly negative.
    """

    threshold_factor: float = 1e-8
    threshold_min: float = 1e-10
    slack_tol: float = 0.0


@dataclass
class SolverConfig:
    """Inner quadratic-program solver parameters.

    ``backend`` selects the forward solver; ``"piqp"`` is the only backend
    shipped with the package. Additional backends can be added by
    implementing the :class:`~micropurc.forward.base.ForwardSolver` protocol.
    """

    backend: str = "piqp"
    eps_abs: float = 1e-12
    eps_duality_gap_abs: float = 1e-12
    # Relative termination criteria, spelled out at PIQP's own defaults. PIQP
    # terminates on the absolute OR the relative criterion, so these set the
    # effective tolerance at ~|c|*1e-9; accepted deliberately for speed (user
    # decision 2026-07-11). Set to 0.0 for a purely absolute 1e-12.
    eps_rel: float = 1e-9
    eps_duality_gap_rel: float = 1e-9
    max_iter: int = 1000
    verbose: bool = False


@dataclass
class ProjectionConfig:
    """Projection computation parameters.

    ``robust_pinv`` is the only mode: one node row is grounded per connected
    component of the active subnetwork, which is the grounding under which the
    projection agrees with the pseudoinverse of the active incidence.
    """

    mode: str = "robust_pinv"

    def __post_init__(self) -> None:
        """Validate the projection mode.

        Raises:
            ValueError: If ``mode`` is not ``"robust_pinv"``.
        """
        if self.mode != "robust_pinv":
            raise ValueError(
                f"projection mode {self.mode!r} is not supported; "
                "'robust_pinv' is the only mode (legacy_drop was removed)"
            )


@dataclass
class EstimationConfig:
    r"""Krasnoselskii-Mann fixed-point iteration parameters.

    Attributes:
        max_iterations: Cap on the number of fixed-point iterations.
        fp_tol: Tolerance on the residual
            :math:`\lVert \Phi_N(\beta) - \beta \rVert`.
        scale_fp_tol_by_sqrt_k: Scale ``fp_tol`` by :math:`\sqrt{K}`, which
            holds the per-coordinate tolerance fixed as :math:`K` grows.
        max_cond_H: Largest condition number of the accumulated Hessian that
            still counts as convergence; ``inf`` waives the check. A run can
            meet ``fp_tol`` at a degenerate fixed point: from an extreme start
            :math:`\beta` explodes, the forward-QP flows saturate at their box
            bounds, and :math:`H` turns ill-conditioned while the residual test
            passes because :math:`\Phi_N` has flattened. Healthy fits sit at
            :math:`\mathrm{cond}(H) \approx 10^1` to :math:`10^2` and
            degenerate ones near :math:`10^6`, so the default separates the two
            by orders of magnitude.
        ls_max_tries: Backtracking steps allowed per iteration.
        ls_tau: Backtracking factor; trial :math:`m` takes the step
            :math:`\tau^m`.
        nm_window: Number of recent merit values; an accepted trial step lands
            at or below their maximum.
    """

    max_iterations: int = 500
    fp_tol: float = 1e-4
    scale_fp_tol_by_sqrt_k: bool = False
    max_cond_H: float = 1e4
    ls_max_tries: int = 5
    ls_tau: float = 0.5
    nm_window: int = 3


@dataclass
class SpecConfig:
    """Model specification: the link attributes entering the utility."""

    attributes: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """Top-level configuration for a micropurc run.

    Attributes:
        name: Run label, used in output paths and reports.
        data_path: Path to the trip data.
        network_path: Path to the network definition.
        output_dir: Directory that receives the run's output.
        num_od_pairs_to_use: ``"all"``, or the number of OD pairs to keep.
        init_beta: Starting parameter vector.
        seed: Seed for the run's random draws.
        active_set: Active-set detection settings.
        solver: Forward-QP settings.
        projection: Range-space projection settings.
        estimation: Fixed-point iteration settings.
        spec: Link attributes entering the utility.
    """

    name: str = "unnamed"
    data_path: str = ""
    network_path: str = ""
    output_dir: str = "."
    num_od_pairs_to_use: str | int = "all"
    init_beta: list[float] = field(default_factory=list)
    seed: int = 42

    active_set: ActiveSetConfig = field(default_factory=ActiveSetConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string, unwrapping numpy scalars, arrays, and paths."""

        def _default(o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, Path):
                return str(o)
            raise TypeError(f"Not serializable: {type(o)}")

        return json.dumps(self.to_dict(), indent=indent, default=_default)

    def save_json(self, path: str | Path) -> Path:
        """Write the config to ``path`` as JSON, creating parent directories.

        Returns the path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        """Build a RunConfig from a flat or nested dict.

        Top-level scalars are read by name and each section key updates its
        dataclass in place; keys with no matching field are ignored, so a
        config carrying extra entries still loads.
        """
        cfg = cls()
        for k in ("name", "data_path", "network_path", "output_dir", "num_od_pairs_to_use", "seed"):
            if k in raw:
                setattr(cfg, k, raw[k])
        if "init_beta" in raw:
            cfg.init_beta = list(raw["init_beta"])

        section_map: dict[str, Any] = {
            "active_set": cfg.active_set,
            "solver": cfg.solver,
            "projection": cfg.projection,
            "estimation": cfg.estimation,
            "spec": cfg.spec,
        }
        for section_name, dc in section_map.items():
            if section_name in raw:
                _merge_dataclass(dc, raw[section_name])
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        """Load RunConfig from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)


def _merge_dataclass(dc: Any, raw: dict[str, Any]) -> None:
    """Update dataclass fields from a dict, ignoring unknown keys."""
    for k, v in raw.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from a TOML or JSON file.

    Format is detected by extension: .json → JSON, anything else → TOML.
    """
    path = Path(path)
    if path.suffix == ".json":
        return RunConfig.from_json(path)

    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return RunConfig.from_dict(raw)


def load_sim_config(path: str | Path) -> dict[str, Any]:
    """Load a simulation experiment config from TOML or JSON.

    Returns the raw dict; simulation experiments have heterogeneous
    structure that doesn't warrant a fixed dataclass yet.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rb") as f:
        return tomllib.load(f)
