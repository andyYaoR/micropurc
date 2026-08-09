"""Shared helpers for the section scripts under ``scripts/paper/``.

Each section script is a standalone process that imports its group's ``setup``
module, computes one section, and writes its tables/figures to
``results/<nb>/<section>/`` immediately. ``run_all.py`` runs them one by one as
isolated subprocesses, so a crash in one section never loses another's output.

Scale: full by default; pass ``--demo`` to any script for a quick run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/paper/common.py -> repo root
assert (ROOT / "pyproject.toml").exists(), f"repo root not found from {__file__}"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA = Path(__file__).resolve().parent / "data"  # bundled EMA / benchmark network CSVs
RESULTS = ROOT / "results"  # output tree


def full_scale() -> bool:
    """True at full scale, i.e. unless ``--demo`` is on the command line."""
    return "--demo" not in sys.argv


def out_dir(nb: str, section: str) -> Path:
    d = RESULTS / nb / section
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(df, outdir: Path, name: str) -> None:
    """Persist a DataFrame as CSV."""
    import pandas as pd

    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    df.to_csv(outdir / f"{name}.csv")
    print(f"  table  -> {outdir / (name + '.csv')}")


def save_fig(fig, outdir: Path, name: str) -> None:
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    print(f"  figure -> {outdir / (name + '.pdf')}")


def save_arrays(outdir: Path, name: str, **arrays) -> None:
    """Persist the raw arrays behind a figure/table so it can be redrawn."""
    import numpy as np

    np.savez(outdir / f"{name}.npz", **arrays)
    print(f"  data   -> {outdir / (name + '.npz')}")


def agg_backend() -> None:
    """Select the non-interactive Agg backend, so sections run headless.

    Call this before importing ``matplotlib.pyplot``; the E402 pragmas on those
    imports mark the resulting import order.
    """
    import matplotlib

    matplotlib.use("Agg")
