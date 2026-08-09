"""Run archiving, reporting, and post-estimation analysis.

This subpackage provides a biogeme-inspired results workflow:

- :class:`~micropurc.reporting.runlog.RunWriter` archives every estimation run
  into its own auto-numbered directory (``name``, ``name~00``, ``name~01``,
  ...) containing a full config snapshot, a dataset snapshot (or content-hash
  reference for large inputs), per-task records, compute/environment info, and
  an HTML report — enough to reproduce the run exactly later.
- :func:`~micropurc.reporting.loader.load_run` loads an archived run back and
  exposes analysis helpers: recompute predicted flows for selected OD pairs,
  extract the active subnetwork, and compare predictions against the observed
  routes.

Example:
    >>> from micropurc.reporting import RunWriter, load_run
    >>> writer = RunWriter("runs", "ema_ab", config={"fp_tol": 1e-3})
    >>> writer.snapshot_data(y=y, origins=origins, dests=dests)
    >>> # ... run the estimation, then:
    >>> writer.finalize(result={"beta_hat": beta_hat, "se_hat": se_hat})
    >>> run = load_run(writer.run_dir)
    >>> flows = run.predicted_flow([(3, 17)])
"""

from .html import render_batch_report, render_fit_report
from .loader import RunArchive, load_run
from .runlog import RunWriter, environment_info, next_run_dir

__all__ = [
    "RunArchive",
    "RunWriter",
    "environment_info",
    "load_run",
    "next_run_dir",
    "render_batch_report",
    "render_fit_report",
]
