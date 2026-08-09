"""nb01 / estimates by convergence status.

Outputs (results/nb01/s5_nonconv/): fig_nonconv_estimates.pdf + nonconv.npz.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb01 import setup as S  # noqa: E402


def main():
    out = common.out_dir(S.NB, "s5_nonconv")
    conv_est, nonc_est, n_exc = [], [], 0
    for r in tqdm(range(S.NONCONV_R), desc="nonconv"):
        d = S.make_dataset(S.dataset_seed(S.NONCONV_N, r), S.NONCONV_N)
        o = S.fit_one(d["y"], d["b"], np.ones(3))
        bh = o["beta_hat"]
        if not np.isfinite(bh).all():
            n_exc += 1
            continue
        (conv_est if o["converged"] else nonc_est).append(bh)
    conv_est = np.array(conv_est)
    nonc_est = np.array(nonc_est)
    print(
        f"converged n={len(conv_est)}  non-converged n={len(nonc_est)}  exceptions={n_exc}",
        flush=True,
    )

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.6,
        }
    ):
        c_conv, c_nc = "#3b6ea5", "#c1121f"
        labels = [r"$\beta_1$", r"$\beta_2$", r"$\beta_3$"]
        jrng = np.random.default_rng(0)
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True)
        for k, ax in enumerate(axes):
            ax.axhline(S.BETA0[k], color="0.4", ls="--", lw=0.8, zorder=1)
            for x0, dat, col in [(1, conv_est, c_conv), (2, nonc_est, c_nc)]:
                if len(dat) == 0:
                    continue
                v = dat[:, k]
                jit = x0 + (jrng.random(len(v)) - 0.5) * 0.28
                ax.scatter(jit, v, s=6, color=col, alpha=0.3, linewidths=0, zorder=2)
                ax.errorbar(
                    x0,
                    v.mean(),
                    yerr=v.std(ddof=1) if len(v) > 1 else 0.0,
                    fmt="o",
                    color=col,
                    ms=5,
                    capsize=3,
                    lw=1.4,
                    mec="white",
                    mew=0.6,
                    zorder=3,
                )
            ax.set_title(labels[k])
            ax.set_xlim(0.5, 2.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Converged", "Non-\nconverged"])
            if k == 0:
                ax.set_ylabel(r"Estimate $\hat\beta_k$")
        axes[0].set_ylim(-0.5, 1.5)
        fig.tight_layout(pad=0.5)
        common.save_fig(fig, out, "fig_nonconv_estimates")

    common.save_arrays(
        out,
        "nonconv",
        conv=conv_est if len(conv_est) else np.zeros((0, 3)),
        nonc=nonc_est if len(nonc_est) else np.zeros((0, 3)),
    )
    print("PASS s5_nonconv", flush=True)


if __name__ == "__main__":
    main()
