"""micropurc: estimation of the perturbed utility route choice (microPURC) model.

The package estimates the microPURC model from trip-level data. It is organized
into subpackages:

- :mod:`micropurc.core` -- the estimator and its numerical machinery;
- :mod:`micropurc.forward` -- forward solvers behind the
  :class:`~micropurc.forward.base.ForwardSolver` protocol (PIQP by default);
- :mod:`micropurc.simulation` -- a data-generating process and route sampler;
- :mod:`micropurc.spec` -- a declarative schema and Python DSL for specifying
  linear-in-parameters utilities;
- :mod:`micropurc.reporting` -- run archiving (config/data/compute snapshots,
  auto-numbered run directories), HTML reports, and post-estimation analysis.

The most common entry points are re-exported here for convenience.
"""

# The re-exports below must follow _add_windows_dll_dirs() so the MinGW runtime
# directory is on the DLL search path before _native loads (see the function).
# ruff: noqa: E402

import os as _os


def _add_windows_dll_dirs() -> None:
    """Register the MinGW runtime directory on Windows before importing ``_native``.

    From Python 3.8 on, the extension-module loader ignores ``PATH`` when it
    resolves a ``.pyd``'s own dependencies, so a MinGW-built ``_native`` that
    links libgcc/libstdc++/winpthread dynamically fails to import
    (``DLL load failed``) with the toolchain on ``PATH``. The directory is
    therefore added to the DLL search path explicitly: an
    ``os.pathsep``-separated ``MICROPURC_DLL_DIR`` is honoured when set, and
    ``PATH`` is otherwise scanned for the MinGW runtime DLLs and whichever
    directories hold them are added. Off Windows this returns at once, and a
    statically-linked build needs nothing from it.
    """
    if _os.name != "nt":
        return
    added: set[str] = set()

    def _add(directory: str) -> None:
        directory = directory.strip()
        if directory and directory not in added and _os.path.isdir(directory):
            try:
                _os.add_dll_directory(directory)
                added.add(directory)
            except OSError:
                pass

    for directory in _os.environ.get("MICROPURC_DLL_DIR", "").split(_os.pathsep):
        _add(directory)  # explicit override: trust the user's directory as-is

    runtime_dlls = ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")
    for directory in _os.environ.get("PATH", "").split(_os.pathsep):
        cand = directory.strip()
        if (
            cand
            and cand not in added
            and _os.path.isdir(cand)
            and any(_os.path.exists(_os.path.join(cand, dll)) for dll in runtime_dlls)
        ):
            _add(cand)


_add_windows_dll_dirs()  # must run before the first import that pulls in _native

from .config import (
    ActiveSetConfig,
    EstimationConfig,
    ProjectionConfig,
    RunConfig,
    SolverConfig,
    SpecConfig,
    load_config,
)
from .core.covariance import cluster_robust_covariance, sandwich_covariance
from .core.estimator import MicroPURCEstimator
from .core.lazy_design import LazyDesign
from .demands import ODDistribution, od_from_csv, od_uniform_all_pairs
from .forward.base import FlowResult, ForwardSolver
from .forward.piqp_solver import PIQPFlowSolver
from .network import Network, NodeIndex
from .reporting import RunArchive, RunWriter, load_run
from .simulation.dgp import DGP, DGPConfig
from .simulation.route_sampler import MarkovSampler
from .spec import (
    CompiledDesign,
    ModelSpec,
    Segment,
    Term,
    Utility,
    attr,
    beta,
    compile_design,
    log,
    power,
    seg,
    standardize,
)

# Single source of truth for the version is pyproject.toml; installed
# distribution metadata carries it from there.
try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("micropurc")
except _PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+uninstalled"

__all__ = [
    # network & demands
    "Network",
    "NodeIndex",
    "ODDistribution",
    "od_from_csv",
    "od_uniform_all_pairs",
    # estimation
    "MicroPURCEstimator",
    "sandwich_covariance",
    "cluster_robust_covariance",
    "LazyDesign",
    # forward solver
    "ForwardSolver",
    "FlowResult",
    "PIQPFlowSolver",
    # simulation
    "DGP",
    "DGPConfig",
    "MarkovSampler",
    # reporting
    "RunWriter",
    "RunArchive",
    "load_run",
    # model specification
    "ModelSpec",
    "Term",
    "Segment",
    "Utility",
    "CompiledDesign",
    "beta",
    "attr",
    "seg",
    "log",
    "power",
    "standardize",
    "compile_design",
    # configuration
    "ActiveSetConfig",
    "SolverConfig",
    "ProjectionConfig",
    "EstimationConfig",
    "SpecConfig",
    "RunConfig",
    "load_config",
]
