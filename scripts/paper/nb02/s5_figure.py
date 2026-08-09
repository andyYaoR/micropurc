"""nb02 / Figure: the four scaling axes.

Loads the four sweeps' saved arrays (results/nb02/sX_*/*.npz) and rebuilds the
2x2 scaling figure. A sweep whose .npz is absent (its section failed, or has
yet to run) leaves its panel empty and the figure is still written. Output
(results/nb02/s5_figure/): fig_scaling.pdf.
"""

from __future__ import annotations

import numpy as np

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb02 import setup as S  # noqa: E402


def _load(section, name):
    """Load results/nb02/<section>/<name>.npz, or None if absent."""
    path = common.RESULTS / S.NB / section / f"{name}.npz"
    if not path.exists():
        print(f"  missing {path} — panel skipped", flush=True)
        return None
    return dict(np.load(path, allow_pickle=True))


def _slope_r2(x, y):
    """Log-log slope of y against x, together with the fit's R^2.

    C4: every slope drawn on a panel is annotated with this R^2, so a claimed
    scaling exponent always carries a fit-quality number beside it.
    """
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), r2


def main():
    out = common.out_dir(S.NB, "s5_figure")
    d_N = _load("s1_sweep_N", "sweep_N")
    d_L = _load("s2_sweep_L", "sweep_L")
    d_K = _load("s3_sweep_K", "sweep_K")
    d_C = _load("s4_cores", "sweep_cores")

    with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.2}):
        fig, ax = plt.subplots(2, 2, figsize=(7.6, 6.0))

        # (1) runtime vs N
        a = ax[0, 0]
        if d_N is not None:
            a.fill_between(d_N["N"], d_N["t_lo"], d_N["t_hi"], alpha=0.2)
            a.loglog(d_N["N"], d_N["t_med"], "o-", ms=4, label="microPURC")
            a.loglog(d_N["N"], d_N["linear_ref"], "k--", lw=0.8, label="linear (unit-slope fit)")
            _b, _r2 = _slope_r2(d_N["N"], d_N["t_med"])
            a.set_title(f"(1) sample size  (slope {_b:.2f}, $R^2$={_r2:.3f})", fontsize=9)
            a.legend(frameon=False, fontsize=8)
        else:
            a.set_title("(1) sample size  (no data)", fontsize=9)
        a.set_xlabel(r"observations $N$")
        a.set_ylabel("runtime (s)")

        # (2) runtime and active-set size vs |L|
        a = ax[0, 1]
        if d_L is not None:
            a.fill_between(d_L["grid_L"], d_L["grid_t_lo"], d_L["grid_t_hi"], alpha=0.2)
            a.loglog(d_L["grid_L"], d_L["grid_t_med"], "o-", ms=4, label="runtime (grid)")
            if "anchor_L" in d_L and len(d_L["anchor_L"]):
                a.loglog(
                    d_L["anchor_L"], d_L["anchor_runtime_s"], "k*", ms=9, label="benchmark nets"
                )
            a2 = a.twinx()
            a2.loglog(
                d_L["grid_L"],
                d_L["grid_active"],
                "s--",
                ms=3,
                color="C1",
                label="active links/trip",
            )
            a2.set_ylabel("mean active links/trip", color="C1")
            _bL, _r2L = _slope_r2(d_L["grid_L"], d_L["grid_t_med"])
            a.set_title(
                f"(2) network size  (runtime slope {_bL:.2f}, $R^2$={_r2L:.3f})",
                fontsize=9,
            )
            a.legend(frameon=False, fontsize=8, loc="upper left")
        else:
            a.set_title("(2) network size  (no data)", fontsize=9)
        a.set_xlabel(r"links $|\mathcal{L}|$")
        a.set_ylabel("runtime (s)")

        # (3) runtime vs K
        a = ax[1, 0]
        if d_K is not None:
            a.fill_between(d_K["K"], d_K["t_lo"], d_K["t_hi"], alpha=0.2)
            a.plot(d_K["K"], d_K["t_med"], "D-", ms=4)
            a.set_ylim(bottom=0)
            a.set_title("(3) parameter count", fontsize=9)
        else:
            a.set_title("(3) parameter count  (no data)", fontsize=9)
        a.set_xlabel(r"parameters $K$")
        a.set_ylabel("runtime (s)")

        # (4) parallel speedup + efficiency
        a = ax[1, 1]
        if d_C is not None:
            a.plot(d_C["cores"], d_C["speedup"], "o-", ms=4, label="speedup")
            a.plot(d_C["cores"], d_C["cores"], "k--", lw=0.8, label="ideal")
            a.legend(frameon=False, fontsize=8)
        a.set_xlabel("cores")
        a.set_ylabel(r"speedup $T(1)/T(c)$")
        a.set_title("(4) parallel scaling", fontsize=9)

        fig.tight_layout(pad=0.6)
        common.save_fig(fig, out, "fig_scaling")

    print("PASS s5_figure", flush=True)


if __name__ == "__main__":
    main()
