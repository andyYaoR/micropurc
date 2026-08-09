"""nb04 / Figure: how flat the estimate is across tuning.

Two-panel figure: max ||beta_hat - beta_def|| against the fixed-point tolerance
(from s1's fp_tol sweep) and against the inner QP tolerance (from s2's eps
sweep). Reloads the raw arrays that s1_config_sweeps and s2_qp_tol persisted;
if either .npz is missing, the section prints what to run and exits without
writing a figure. Outputs (results/nb04/s4_figure/): fig_algo_robustness.pdf.
"""

from __future__ import annotations

import numpy as np

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb04 import setup as S  # noqa: E402


def main():
    out = common.out_dir(S.NB, "s4_figure")
    fp_npz = common.RESULTS / S.NB / "s1_config_sweeps" / "config_sweeps.npz"
    eps_npz = common.RESULTS / S.NB / "s2_qp_tol" / "qp_tol.npz"

    missing = [p for p in (fp_npz, eps_npz) if not p.exists()]
    if missing:
        print(
            "SKIP s4_figure: missing sweep data -> " + ", ".join(str(p) for p in missing),
            flush=True,
        )
        print("  run s1_config_sweeps and s2_qp_tol first.", flush=True)
        return

    fp = np.load(fp_npz)
    eps = np.load(eps_npz)
    tol_x = np.asarray(fp["fp_tol__values"], float)
    tol_y = np.asarray(fp["fp_tol__max_dbeta"], float)
    eps_x = np.asarray(eps["eps_abs"], float)
    eps_y = np.asarray(eps["max_dbeta"], float)

    with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.2}):
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
        ax[0].loglog(tol_x, np.clip(tol_y, 1e-12, None), "o-", ms=4)
        ax[0].loglog(tol_x, tol_x, "k--", lw=0.8, label=r"$=\tau$")
        ax[0].set_xlabel(r"fixed-point tolerance $\tau$")
        ax[0].set_ylabel(r"max $\|\hat\beta-\hat\beta_{\rm def}\|$")
        ax[0].set_title("estimate vs tolerance", fontsize=9)
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].loglog(eps_x, np.clip(eps_y, 1e-16, None), "s-", ms=4, color="C1")
        ax[1].set_xlabel(r"inner QP tolerance $\varepsilon_{\rm abs}$")
        ax[1].set_ylabel(r"max $\|\hat\beta-\hat\beta_{\rm def}\|$")
        ax[1].set_title("estimate vs solver tolerance", fontsize=9)
        fig.tight_layout(pad=0.6)
        common.save_fig(fig, out, "fig_algo_robustness")

    print("PASS s4_figure", flush=True)


if __name__ == "__main__":
    main()
