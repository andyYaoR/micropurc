"""nb02 / paper panels: standalone runtime-vs-N and runtime-vs-|L| figures.

s5_figure combines all four scaling axes into one 2x2 panel. The manuscript
places the N and |L| axes side by side as two separate figures, so this section
redraws those two on their own from the same saved arrays. Outputs
(results/nb02/s7_paper_panels/): fig_scaling_N.pdf and fig_scaling_L.pdf.
"""

from __future__ import annotations

import numpy as np

from scripts.paper import common
from scripts.paper.nb02 import setup as S

LINE = "#1f4e79"
BAND = "0.82"


def _panel(x, med, lo, hi, ref, xlabel, path_name, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.fill_between(x, lo, hi, color=BAND, linewidth=0)
    ax.plot(x, med, "-o", color=LINE, markersize=2.6, linewidth=1.3)
    if ref is not None:
        ax.plot(x, ref, "--", color="0.25", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Runtime [s]")
    ax.tick_params(labelsize=8)
    common.save_fig(fig, out, path_name)
    plt.close(fig)


def main():
    out = common.out_dir(S.NB, "s7_paper_panels")

    pn = common.RESULTS / S.NB / "s1_sweep_N" / "sweep_N.npz"
    pl = common.RESULTS / S.NB / "s2_sweep_L" / "sweep_L.npz"
    if not pn.exists() or not pl.exists():
        print(f"  missing {pn} or {pl} — nothing to draw", flush=True)
        print("PASS s7_paper_panels", flush=True)
        return

    d = dict(np.load(pn, allow_pickle=True))
    _panel(
        d["N"], d["t_med"], d["t_lo"], d["t_hi"], d["linear_ref"],
        r"Number of trips $N$", "fig_scaling_N", out,
    )

    g = dict(np.load(pl, allow_pickle=True))
    L, t = g["grid_L"], g["grid_t_med"]
    # Linear reference anchored at the smallest grid so the slope is readable
    # against the measured curve.
    ref = t[0] * (L / L[0])
    _panel(
        L, t, g["grid_t_lo"], g["grid_t_hi"], ref,
        r"Number of links $|\mathcal{L}|$", "fig_scaling_L", out,
    )

    print("PASS s7_paper_panels", flush=True)


if __name__ == "__main__":
    main()
