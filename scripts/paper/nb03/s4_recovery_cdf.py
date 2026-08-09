"""nb03 / recovery CDF: threshold-free view of initialization robustness.

Post-processes the saved per-estimator estimates in s2_init_robust's init_robust.npz
(no re-run) into the empirical CDF of the distance-to-truth, F(x) = fraction of starts
with ||beta_hat - beta0|| < x, per estimator and starting radius. This replaces the
single hard 0.25 recovery cutoff with the whole distribution. Non-finite
(degenerate/diverged) estimates count as infinite distance, so a CDF that plateaus
below 1 is exactly the divergence fraction.

Outputs (results/nb03/s4_recovery_cdf/): one standalone panel per estimator
(fig_recovery_cdf_{micropurc,directrv,mle}.pdf, stitched into a single figure in
the manuscript) + recovery_cdf table (fraction within a few reference thresholds,
per estimator x radius).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb03 import setup as S  # noqa: E402

# (display name, npz key, filename slug). One standalone PDF per estimator; the
# manuscript stitches the three panels into a single figure (subfigures).
ESTS = [("microPURC", "mp_beta", "micropurc"),
        ("direct-RV", "rv_beta", "directrv"),
        ("MLE", "mle_beta", "mle")]
SHOW_RADII = [0.5, 2.0, 5.0, 8.0]  # representative starting radii
THRESHOLDS = [0.05, 0.10, 0.25, 0.50, 1.0]  # reference cutoffs for the summary table


def _dist(beta_r):
    """Distance to beta0 for one (n_starts, 3) block; non-finite -> +inf (diverged)."""
    fin = np.isfinite(beta_r).all(1)
    dist = np.linalg.norm(beta_r - S.BETA0, axis=1)
    return np.where(fin, dist, np.inf)


def main():
    out = common.out_dir(S.NB, "s4_recovery_cdf")
    src = common.RESULTS / S.NB / "s2_init_robust" / "init_robust.npz"
    if not src.exists():
        print(f"SKIP s4_recovery_cdf: {src} not found (run s2_init_robust first)", flush=True)
        return

    d = np.load(src)
    radii = [float(r) for r in d["radii"]]
    xgrid = np.logspace(-3, 3, 400)

    # --- figures: one standalone panel per estimator, CDF curves per radius ------
    # Each estimator is its own PDF so the manuscript can place the three panels as
    # subfigures of a single figure. To reproduce the shared-y look when the panels
    # sit side by side, only the leftmost (microPURC) carries the y-axis label,
    # tick labels, and legend; the others suppress them (mimicking sharey=True).
    for name, key, slug in ESTS:
        beta = d[key]  # (nR, nStarts, 3)
        leftmost = key == "mp_beta"
        with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.2}):
            fig, ax = plt.subplots(figsize=(4.05 if leftmost else 3.6, 3.7))
            for r in SHOW_RADII:
                ri = radii.index(r)
                dist = _dist(beta[ri])
                cdf = (dist[None, :] < xgrid[:, None]).mean(axis=1)
                ax.plot(xgrid, cdf, label=f"$r={r:g}$")
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1e3)
            ax.set_ylim(0, 1.02)
            ax.set_title(name)
            ax.set_xlabel(r"distance $\|\hat\beta-\beta_0\|$")
            if leftmost:
                ax.set_ylabel("fraction of starts within distance")
                ax.legend(frameon=False, loc="lower right", fontsize=8, title="start radius")
            else:
                ax.tick_params(labelleft=False)
            fig.tight_layout(pad=0.5)
            common.save_fig(fig, out, f"fig_recovery_cdf_{slug}")
            # raster preview alongside each PDF (common.save_fig writes only the PDF)
            fig.savefig(out / f"fig_recovery_cdf_{slug}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --- summary table: F(threshold) per estimator x radius ----------------------
    rows = []
    for name, key, _slug in ESTS:
        beta = d[key]
        for ri, r in enumerate(radii):
            dist = _dist(beta[ri])
            row = {"Estimator": name, "Radius r": r}
            for t in THRESHOLDS:
                row[f"within {t:g}"] = 100.0 * float(np.mean(dist < t))
            rows.append(row)
    tab = pd.DataFrame(rows).set_index(["Estimator", "Radius r"])
    common.save_table(tab.round(1), out, "recovery_cdf")
    print("PASS s4_recovery_cdf", flush=True)


if __name__ == "__main__":
    main()
