"""Run archiving: auto-numbered run directories with full reproducibility snapshots.

A *run* is one estimation (or one batch study). Each run gets its own
directory that is never overwritten — repeated runs with the same name get an
auto-incremented ``~NN`` suffix (biogeme convention: ``model``, ``model~00``,
``model~01``, ...). The directory contains everything needed to reproduce the
run:

- ``config.json`` — every estimator/solver config, seeds, spec description,
  package versions, and the git commit of the code that produced the run.
- ``data.npz`` / ``data_ref.json`` — the estimation inputs, either snapshotted
  in full (compressed) or, above a size threshold, referenced by content hash
  and source path so staleness is detectable.
- ``environment.json`` — machine info: hostname, platform, CPU count, native
  thread count, total RAM, BLAS-relevant env vars.
- ``records/*.npz`` — per-task records for batch studies (one file per
  replication).
- ``result.npz`` + ``compute.json`` — final estimates and the compute profile
  (wall time, peak RSS, CPU time).
- ``report.html`` — human-readable report (see :mod:`micropurc.reporting.html`).

Example:
    >>> writer = RunWriter("runs", "sim_grid", config={
    ...     "estimation": est_cfg, "projection": proj_cfg, "seed": 0})
    >>> writer.snapshot_data(y=y_csr, origins=origins, dests=dests)
    >>> result = est.fit(...)
    >>> writer.finalize(result={"beta_hat": result["beta_hat"]})
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

#: Size ceiling (bytes, summed over one snapshot) for copying inputs in full.
#: A snapshot above it is recorded as a content-hash reference. Override per
#: call via ``max_full_bytes``.
DEFAULT_MAX_FULL_BYTES = 200 * 1024 * 1024


def next_run_dir(base: str | Path, name: str) -> Path:
    """Returns the first non-existing run directory for ``name`` under ``base``.

    Follows the biogeme naming convention: the first run is ``base/name``;
    subsequent runs are ``base/name~00``, ``base/name~01``, ... so no run is
    ever overwritten. The next index is one past the highest suffix present, so
    deleting an intermediate run leaves its number retired.

    Args:
        base: Parent directory that holds all runs (created if missing).
        name: Logical run name, e.g. ``"ema_ab"``.

    Returns:
        Path of a fresh, not-yet-created run directory.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / name
    if not candidate.exists():
        return candidate
    pat = re.compile(re.escape(name) + r"~(\d+)$")
    taken = [int(m.group(1)) for p in base.iterdir() if (m := pat.match(p.name))]
    nxt = (max(taken) + 1) if taken else 0
    return base / f"{name}~{nxt:02d}"


def _git_commit() -> str:
    """Returns the short git commit of the working tree, or ``"unknown"``.

    The commit identifies the code that produced a run, so it is provenance a
    reader needs even when git is unavailable; every failure path therefore
    resolves to ``"unknown"``.

    Returns:
        Short commit hash, with ``"+dirty"`` appended when the tree has
        uncommitted changes; ``"unknown"`` outside a git checkout.
    """
    try:
        root = Path(__file__).resolve()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=root.parent,
            stderr=subprocess.DEVNULL,
        ).returncode
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _total_ram_gb() -> float | None:
    """Returns total system RAM in GiB, or None if it cannot be determined.

    Cross-platform with no external dependency: ``/proc/meminfo`` on Linux,
    ``GlobalMemoryStatusEx`` on Windows, and ``sysctl hw.memsize`` on macOS.
    Linux reports kibibytes and the other two report bytes, hence the differing
    divisors.
    """
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024**2, 2)
        elif sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return round(stat.ullTotalPhys / 1024**3, 2)
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            )
            return round(int(out.stdout.strip()) / 1024**3, 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def peak_rss_mb() -> float:
    """Returns this process's peak resident set size in MiB.

    Uses ``getrusage`` on macOS/Linux and the Win32 API on Windows, so no
    external dependency is needed. The value is a high-water mark over the whole
    process lifetime, so it covers memory already released by the time it is
    read.

    Returns:
        Peak RSS in MiB (0.0 if it cannot be determined).
    """
    try:
        import resource  # Unix-only

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in bytes on macOS and in KiB on Linux.
        return ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru_maxrss / 1024.0
    except ImportError:  # Windows
        from ..core.diagnostics import _windows_peak_rss_bytes

        return _windows_peak_rss_bytes() / (1024.0 * 1024.0)


def environment_info() -> dict[str, Any]:
    """Collects machine and software environment info for a run archive.

    Every probe falls back to a placeholder on failure, so archiving a run
    survives an unavailable extension or missing package metadata.

    Returns:
        JSON-serializable dict with hostname, platform, Python/NumPy/SciPy
        versions, micropurc version + git commit, CPU count, native thread
        count, total RAM, and thread-related environment variables.
    """
    import numpy
    import scipy

    try:
        from micropurc._native import get_num_threads

        native_threads = int(get_num_threads())
    except Exception:
        native_threads = None
    try:
        from importlib.metadata import version

        pkg_version = version("micropurc")
    except Exception:
        pkg_version = "unknown"

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "micropurc": pkg_version,
        "git_commit": _git_commit(),
        "cpu_count": os.cpu_count(),
        "native_threads": native_threads,
        "total_ram_gb": _total_ram_gb(),
        "env_threads": {
            k: os.environ[k]
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if k in os.environ
        },
    }


def _jsonable(obj: Any) -> Any:
    """Recursively converts configs and arrays into JSON-serializable values.

    Handles dataclasses (via ``asdict``), numpy scalars and arrays, paths, and
    nested mappings and sequences. A dataclass keeps its type name under the
    ``__dataclass__`` key, so a reader of ``config.json`` can tell which config
    class produced a block of settings. Anything else is passed through, and
    :func:`json.dumps` is left to reject it.

    Args:
        obj: Any value to convert.

    Returns:
        A JSON-serializable equivalent of ``obj``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__name__, **_jsonable(dataclasses.asdict(obj))}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _hash_array(arr: np.ndarray) -> str:
    """Returns the SHA-256 of an array's raw bytes (C-contiguous view).

    The digest covers the buffer alone, so two arrays agree only when their
    dtype, layout, and values all agree; shape and dtype are archived beside it.

    Args:
        arr: Array to hash.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _flatten_sparse(name: str, mat: sp.spmatrix) -> dict[str, np.ndarray]:
    """Decomposes a sparse matrix into npz-storable CSR components.

    The ``__csr_`` infix is the marker the loader regroups on, so it must stay in
    step with ``_load_npz`` in :mod:`micropurc.reporting.loader`.

    Args:
        name: Base name for the component keys.
        mat: Sparse matrix (converted to CSR).

    Returns:
        Dict of ``{name}__csr_{data,indices,indptr,shape}`` arrays.
    """
    csr = mat.tocsr()
    return {
        f"{name}__csr_data": csr.data,
        f"{name}__csr_indices": csr.indices,
        f"{name}__csr_indptr": csr.indptr,
        f"{name}__csr_shape": np.asarray(csr.shape, dtype=np.int64),
    }


class RunWriter:
    """Archives one estimation run into a fresh auto-numbered directory.

    Attributes:
        run_dir: The directory created for this run.

    Example:
        >>> writer = RunWriter("runs", "ema_ab", config={"seed": 0})
        >>> writer.log("starting probe")
        >>> writer.add_record("ab_N500_rep0", beta_hat=beta_hat)
        >>> writer.finalize(result={"beta_hat": beta_hat})
    """

    def __init__(
        self,
        base: str | Path,
        name: str,
        *,
        config: Mapping[str, Any] | None = None,
        resume: str | Path | None = None,
    ) -> None:
        """Creates or resumes a run directory and snapshots config and environment.

        The wall-clock and CPU-time origins are taken here, so a resumed run
        reports the time spent since the resume.

        Args:
            base: Parent directory holding all runs.
            name: Logical run name; the directory auto-increments to avoid
                overwriting earlier runs with the same name.
            config: Everything needed to reproduce the run: estimator/solver
                config dataclasses, seeds, protocol constants, data
                descriptions. Serialized to ``config.json``.
            resume: Existing run directory to continue (skips creating a new
                one and does not rewrite ``config.json``).

        Raises:
            FileNotFoundError: If ``resume`` points to a missing directory.
        """
        if resume is not None:
            self.run_dir = Path(resume)
            if not self.run_dir.is_dir():
                raise FileNotFoundError(f"cannot resume: {self.run_dir} does not exist")
        else:
            self.run_dir = next_run_dir(base, name)
            self.run_dir.mkdir(parents=True)
            (self.run_dir / "config.json").write_text(
                json.dumps(_jsonable(dict(config or {})), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (self.run_dir / "environment.json").write_text(
                json.dumps(environment_info(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()
        self.log(f"run {'resumed' if resume else 'created'} at {self.run_dir}")

    def log(self, message: str) -> None:
        """Appends a timestamped line to ``run.log``.

        Args:
            message: Text to log.
        """
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.run_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")

    def snapshot_data(
        self,
        *,
        max_full_bytes: int = DEFAULT_MAX_FULL_BYTES,
        source_paths: Mapping[str, str | Path] | None = None,
        name: str = "data",
        **arrays: Any,
    ) -> Path:
        """Snapshots estimation inputs for exact future replication.

        Inputs totalling at most ``max_full_bytes`` are copied in full into
        ``{name}.npz`` (compressed). Above that, ``{name}_ref.json`` records
        SHA-256 content hashes and optional source paths, so a future load can
        verify it is reading identical data. Sparse inputs are flattened to
        their CSR components first, and each component is hashed on its own.

        Args:
            max_full_bytes: Full-copy size threshold in bytes.
            source_paths: Optional map from array name to the file it came
                from (stored alongside hashes for reference snapshots).
            name: Base filename for the snapshot.
            **arrays: Arrays or scipy sparse matrices to snapshot.

        Returns:
            Path of the file written (``.npz`` or ``_ref.json``).
        """
        flat: dict[str, np.ndarray] = {}
        for key, val in arrays.items():
            if sp.issparse(val):
                flat.update(_flatten_sparse(key, val))
            else:
                flat[key] = np.asarray(val)
        total = sum(a.nbytes for a in flat.values())

        if total <= max_full_bytes:
            out = self.run_dir / f"{name}.npz"
            np.savez_compressed(out, **flat)
            self.log(f"data snapshot: {out.name} ({total / 1e6:.1f} MB raw)")
            return out

        ref = {
            "mode": "hash_reference",
            "total_bytes": total,
            "arrays": {
                k: {
                    "sha256": _hash_array(a),
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "source": str((source_paths or {}).get(k.split("__csr_")[0], "")),
                }
                for k, a in flat.items()
            },
        }
        out = self.run_dir / f"{name}_ref.json"
        out.write_text(json.dumps(ref, indent=2, sort_keys=True), encoding="utf-8")
        self.log(f"data snapshot: hash reference ({total / 1e6:.1f} MB raw)")
        return out

    def add_record(self, record_name: str, **arrays: Any) -> Path:
        """Writes one per-task record under ``records/``.

        Each replication of a batch study gets its own file, so a crashed study
        resumes from the records already on disk (see :meth:`record_exists`).

        Args:
            record_name: Filename stem, e.g. ``"ab_N500_rep0"``.
            **arrays: Scalars/arrays to store (strings allowed).

        Returns:
            Path of the record written.
        """
        out = self.run_dir / "records" / f"{record_name}.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez(out, **arrays)
        return out

    def record_exists(self, record_name: str) -> bool:
        """Reports whether a per-task record already exists, for resume.

        Args:
            record_name: Filename stem used in :meth:`add_record`.

        Returns:
            True when the record file is present.
        """
        return (self.run_dir / "records" / f"{record_name}.npz").exists()

    def finalize(
        self,
        result: Mapping[str, Any] | None = None,
        status: str = "completed",
    ) -> Path:
        """Writes the final result, the compute profile, and the run status.

        ``compute.json`` is the marker of a finished run, so a directory without
        it was interrupted.

        Args:
            result: Final estimates and diagnostics (arrays/scalars) stored in
                ``result.npz``.
            status: Run status recorded in ``compute.json`` (``"completed"``,
                ``"failed"``, ...).

        Returns:
            The run directory.
        """
        if result:
            np.savez(self.run_dir / "result.npz", **{k: v for k, v in result.items()})
        compute = {
            "status": status,
            "wall_time_s": round(time.perf_counter() - self._t0, 3),
            "cpu_time_s": round(time.process_time() - self._cpu0, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "cpu_count": os.cpu_count(),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "compute.json").write_text(json.dumps(compute, indent=2), encoding="utf-8")
        self.log(f"finalized: {compute}")
        return self.run_dir
