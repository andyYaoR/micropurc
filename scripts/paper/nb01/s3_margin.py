"""nb01 / active-set margin CDF.

Outputs (results/nb01/s3_margin/): fig_margin_cdf.pdf + margin_cdf.npz.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb01 import setup as S  # noqa: E402

GUIDE, GUIDE_LIGHT = "gray", "lightgray"


def main():
    out = common.out_dir(S.NB, "s3_margin")
    znorm = np.linalg.norm(S.net.Z, axis=1)
    ok = znorm > 0

    def margin_per_trip(b, beta_hat, n_trips):
        """Per trip, the smallest |dual slack| over links, normalised by ||z_l||.

        A step delta in beta moves link l's slack by at most ||z_l|| ||delta||, so
        dividing by the attribute-vector norm turns the slack into a distance in
        beta-space: the plotted margin lower-bounds how far beta_hat must move
        before that link's sign test can flip.
        """
        c = S.net.Z @ beta_hat
        res = np.empty(n_trips)
        for i in range(n_trips):
            fr = S.solver.solve(c, np.asarray(b[i]).ravel())
            slack = c + S.net.A.T @ fr.lam_eq
            res[i] = float(np.min(np.abs(slack[ok]) / znorm[ok]))
        return res

    cdf_per_rep, all_margins = [], []
    for r in tqdm(range(S.MARGIN_R), desc="margin"):
        d = S.make_dataset(S.dataset_seed(S.MARGIN_N, r), S.MARGIN_N)
        o = S.fit_one(d["y"], d["b"], np.ones(3))
        m = margin_per_trip(d["b"], o["beta_hat"], S.MARGIN_N)
        m = m[np.isfinite(m)]
        all_margins.append(m)
        cdf_per_rep.append(np.mean(m[None, :] <= S.MARGIN_GRID[:, None], axis=1))

    margin_cdf = np.array(cdf_per_rep)
    cdf_mean = margin_cdf.mean(axis=0)
    cdf_std = margin_cdf.std(axis=0, ddof=1)
    pooled_min = float(np.concatenate(all_margins).min())
    print(f"pooled margin minimum = {pooled_min:.2e}", flush=True)

    cdf = 100 * cdf_mean
    band = 100 * cdf_std
    xlo = 1e-12
    with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.1}):
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        ax.axvspan(xlo, pooled_min, color=GUIDE_LIGHT, alpha=0.35, linewidth=0)
        ax.axvline(pooled_min, color=GUIDE, ls="--", lw=0.8)
        ax.text(
            np.sqrt(xlo * pooled_min),
            50,
            "Constant\nactive-set\nregion",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )
        ax.fill_between(
            S.MARGIN_GRID, cdf - band, cdf + band, color=GUIDE_LIGHT, alpha=0.6, linewidth=0
        )
        ax.semilogx(S.MARGIN_GRID, cdf)
        ax.set_xlabel(r"Distance $m$ to nearest active-set change")
        ax.set_ylabel(r"Trips with margin $m_n \leq m$ (%)")
        ax.set_xlim(xlo, S.MARGIN_GRID[-1])
        ax.set_ylim(0, 100)
        fig.tight_layout(pad=0.4)
        common.save_fig(fig, out, "fig_margin_cdf")

    common.save_arrays(
        out,
        "margin_cdf",
        grid=S.MARGIN_GRID,
        cdf_mean=cdf_mean,
        cdf_std=cdf_std,
        pooled_min=np.array(pooled_min),
    )
    print("PASS s3_margin", flush=True)


if __name__ == "__main__":
    main()
