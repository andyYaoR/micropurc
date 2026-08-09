"""Self-contained HTML reports for archived runs, in the biogeme tradition.

Two renderers:

- :func:`render_fit_report` — one estimation: parameter table with SE/t/p,
  convergence summary with a Q-history sparkline, dataset provenance, full
  config (collapsible), and the compute/environment profile.
- :func:`render_batch_report` — a batch study (e.g. a Monte Carlo sweep):
  arbitrary summary tables, gate results, and the compute profile.

Reports are single files with inline CSS — no external assets — so they can
be archived inside the run directory and opened anywhere.

Example:
    >>> from micropurc.reporting import load_run, render_fit_report
    >>> run = load_run("runs/sim_grid")
    >>> render_fit_report(run, param_names=["b_length", "b_time"])  # -> report.html
"""

from __future__ import annotations

import html as _html
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .loader import RunArchive

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 60rem; color: #1a1a2e; line-height: 1.45; padding: 0 1rem; }
h1 { font-size: 1.5rem; border-bottom: 2px solid #4361ee; padding-bottom: .4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; color: #30345e; }
table { border-collapse: collapse; margin: .8rem 0; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #d7d9e3; padding: .35rem .7rem; text-align: right; }
th { background: #eef0f7; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
.meta { color: #555; font-size: .9rem; }
.ok { color: #0a7d33; font-weight: 600; } .bad { color: #c0392b; font-weight: 600; }
details { margin: .6rem 0; } summary { cursor: pointer; color: #30345e; font-weight: 600; }
pre { background: #f6f7fb; border: 1px solid #d7d9e3; padding: .8rem; overflow-x: auto;
      font-size: .85rem; }
svg.spark { vertical-align: middle; }
.kv td { border: none; padding: .15rem .9rem .15rem 0; text-align: left; }
"""


def _esc(x: Any) -> str:
    """Escapes a value for embedding in HTML text.

    Args:
        x: Value to escape.

    Returns:
        Escaped string.
    """
    return _html.escape(str(x))


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Renders an HTML table.

    Args:
        headers: Column headers.
        rows: Row cell values (pre-formatted).

    Returns:
        HTML string.
    """
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _kv(pairs: Iterable[tuple[str, Any]]) -> str:
    """Renders a borderless key/value table.

    Args:
        pairs: (label, value) tuples.

    Returns:
        HTML string.
    """
    body = "".join(f"<tr><td><b>{_esc(k)}</b></td><td>{_esc(v)}</td></tr>" for k, v in pairs)
    return f'<table class="kv">{body}</table>'


def _sparkline(values: np.ndarray, width: int = 220, height: int = 36) -> str:
    r"""Renders a log-scale sparkline of a positive series as inline SVG.

    Convergence residuals fall by orders of magnitude, so the vertical axis is
    :math:`\log_{10}` and is rescaled to the series' own range: the picture
    shows the shape of the decay, and the numbers beside it carry the level.

    Args:
        values: Series to plot. Non-finite entries are dropped and the rest are
            clipped up to 1e-300, which keeps the logarithm defined for an
            exactly converged residual.
        width: SVG width in px.
        height: SVG height in px.

    Returns:
        Inline SVG string ('' when fewer than 2 finite points remain).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return ""
    logv = np.log10(np.clip(v, 1e-300, None))
    lo, hi = float(logv.min()), float(logv.max())
    span = (hi - lo) or 1.0
    xs = np.linspace(2, width - 2, v.size)
    ys = height - 3 - (logv - lo) / span * (height - 6)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f'<svg class="spark" width="{width}" height="{height}">'
        f'<polyline fill="none" stroke="#4361ee" stroke-width="1.5" points="{pts}"/></svg>'
    )


def _page(title: str, sections: list[str]) -> str:
    """Assembles a full HTML page.

    Args:
        title: Page title.
        sections: HTML fragments in order.

    Returns:
        Complete HTML document.
    """
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{_esc(title)}</h1><p class='meta'>generated {stamp} by micropurc.reporting</p>"
        + "".join(sections)
        + "</body></html>"
    )


def _compute_section(run: RunArchive) -> str:
    """Renders the compute and environment section of a run.

    Missing keys render as ``?`` or NaN, so a report of an interrupted run still
    builds.

    Args:
        run: Loaded run archive.

    Returns:
        HTML fragment.
    """
    env, comp = run.environment, run.compute
    return "<h2>Compute</h2>" + _kv(
        [
            ("status", comp.get("status", "?")),
            ("wall time", f"{comp.get('wall_time_s', float('nan')):.1f} s"),
            ("CPU time", f"{comp.get('cpu_time_s', float('nan')):.1f} s"),
            ("peak RSS", f"{comp.get('peak_rss_mb', float('nan')):.0f} MB"),
            ("CPU cores", env.get("cpu_count", "?")),
            ("native threads", env.get("native_threads", "?")),
            ("total RAM", f"{env.get('total_ram_gb', '?')} GB"),
            ("host", env.get("hostname", "?")),
            ("platform", env.get("platform", "?")),
            (
                "versions",
                f"micropurc {env.get('micropurc', '?')} @ {env.get('git_commit', '?')}, "
                f"numpy {env.get('numpy', '?')}, scipy {env.get('scipy', '?')}, "
                f"python {env.get('python', '?')}",
            ),
        ]
    )


def _provenance_section(run: RunArchive) -> str:
    """Renders dataset provenance: snapshot mode, sizes, hashes.

    Args:
        run: Loaded run archive.

    Returns:
        HTML fragment.
    """
    if run.data:
        rows = []
        for k, v in run.data.items():
            shape = "sparse " + str(v.shape) if hasattr(v, "tocsr") else str(np.shape(v))
            rows.append((k, shape))
        body = (
            "<p>Inputs snapshotted in full — load with "
            f"<code>load_run('{_esc(run.run_dir.name)}')</code>; replication needs no "
            "external files.</p>" + _table(["array", "shape"], rows)
        )
    elif run.data_ref:
        arrays = run.data_ref.get("arrays", {})
        body = (
            "<p>Inputs referenced by content hash (too large to copy). Verify sources "
            "against these SHA-256 digests before replicating.</p>"
            + _table(
                ["array", "shape", "sha256", "source"],
                [
                    (k, str(v.get("shape")), v.get("sha256", "")[:16] + "…", v.get("source", ""))
                    for k, v in arrays.items()
                ],
            )
        )
    else:
        body = "<p class='meta'>No dataset snapshot stored.</p>"
    return "<h2>Data provenance</h2>" + body


def _config_section(run: RunArchive) -> str:
    """Renders the collapsible full-config section.

    Args:
        run: Loaded run archive.

    Returns:
        HTML fragment.
    """
    return (
        "<h2>Configuration snapshot</h2><details><summary>config.json "
        "(everything needed to reproduce this run)</summary><pre>"
        + _esc(json.dumps(run.config, indent=2, sort_keys=True))
        + "</pre></details>"
    )


def render_fit_report(
    run: RunArchive,
    *,
    param_names: Sequence[str] | None = None,
    out_name: str = "report.html",
) -> Path:
    r"""Writes the single-estimation HTML report into the run directory.

    The reported statistic is :math:`t_k = \hat\beta_k / \mathrm{se}_k` and its
    p-value is the two-sided standard-normal tail
    :math:`2\big(1 - \Phi(|t_k|)\big)`. Parameters whose standard error is
    missing, non-finite, or non-positive show a dash in the three test columns.

    Args:
        run: Loaded run archive (must contain ``beta_hat`` in its result).
        param_names: Parameter names for the estimates table; defaults to
            ``beta_0, beta_1, ...``.
        out_name: Output filename inside the run directory.

    Returns:
        Path of the written report.

    Example:
        >>> render_fit_report(load_run("runs/sim_grid"),
        ...                   param_names=["b_length", "b_time", "b_toll"])
    """
    beta = run.beta_hat
    se = run.se_hat
    k = beta.shape[0]
    names = list(param_names) if param_names else [f"beta_{i}" for i in range(k)]

    rows = []
    for i in range(k):
        if se is not None and np.isfinite(se[i]) and se[i] > 0:
            t = beta[i] / se[i]
            from math import erf, sqrt

            p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
            rows.append((names[i], f"{beta[i]:.6f}", f"{se[i]:.6f}", f"{t:.2f}", f"{p:.2g}"))
        else:
            rows.append((names[i], f"{beta[i]:.6f}", "—", "—", "—"))
    params = _table(["parameter", "estimate", "std err", "t-stat", "p-value"], rows)

    conv_bits: list[tuple[str, Any]] = []
    res = run.result
    if "converged" in res:
        ok = bool(np.asarray(res["converged"]))
        conv_bits.append(("converged", "yes" if ok else "NO"))
    if "iterations" in res:
        conv_bits.append(("iterations", int(np.asarray(res["iterations"]))))
    q_hist = np.asarray(res.get("Q_history", []), dtype=float)
    spark = ""
    if q_hist.size:
        conv_bits.append(("final residual", f"{np.sqrt(2.0 * q_hist[-1]):.3e}"))
        spark = "<p>Q-history (log scale): " + _sparkline(q_hist) + "</p>"
    convergence = "<h2>Convergence</h2>" + _kv(conv_bits) + spark

    page = _page(
        f"micropurc run — {run.run_dir.name}",
        [
            "<h2>Parameter estimates</h2>",
            params,
            convergence,
            _provenance_section(run),
            _config_section(run),
            _compute_section(run),
        ],
    )
    out = run.run_dir / out_name
    out.write_text(page, encoding="utf-8")
    return out


def render_batch_report(
    run: RunArchive,
    *,
    title: str,
    tables: Sequence[tuple[str, Sequence[str], Iterable[Sequence[Any]]]],
    gates: Sequence[tuple[str, bool]] = (),
    out_name: str = "report.html",
) -> Path:
    """Writes a batch-study HTML report (Monte Carlo sweeps, ablations, ...).

    Args:
        run: Loaded run archive of the batch run.
        title: Report title.
        tables: Sequence of ``(section_title, headers, rows)`` summary tables.
        gates: ``(gate_name, passed)`` pairs rendered as a pass/fail list.
        out_name: Output filename inside the run directory.

    Returns:
        Path of the written report.

    Example:
        >>> render_batch_report(run, title="EMA AB study",
        ...     tables=[("Goal A", ["N", "bias"], [[500, 0.01]])],
        ...     gates=[("RMSE decreasing", True)])
    """
    sections: list[str] = []
    for sec_title, headers, rows in tables:
        sections.append(f"<h2>{_esc(sec_title)}</h2>" + _table(headers, rows))
    if gates:
        items = "".join(
            f"<tr><td>{_esc(name)}</td>"
            f"<td class='{'ok' if ok else 'bad'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
            for name, ok in gates
        )
        n_fail = sum(1 for _, ok in gates if not ok)
        verdict = (
            "<p class='ok'>ALL GATES PASS</p>"
            if n_fail == 0
            else f"<p class='bad'>{n_fail} GATE(S) FAILED</p>"
        )
        sections.append("<h2>Gates</h2>" + verdict + f"<table><tbody>{items}</tbody></table>")
    sections += [_provenance_section(run), _config_section(run), _compute_section(run)]

    out = run.run_dir / out_name
    out.write_text(_page(title, sections), encoding="utf-8")
    return out
