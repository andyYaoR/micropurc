"""nb03 / robustness to initialization: init-ball over RADII.

Builds INIT_DATASETS independent datasets of INIT_N trips each and, at every
radius, runs INIT_M random starts per dataset — INIT_DATASETS * INIT_M starts
per radius in total. Each start is fit by microPURC, direct-RV, and MLE, and
the reported rates pool over all of them, so each row averages across data
draws as well as starts.

Outputs (results/nb03/s2_init_robust/): the init_robust table as CSV, and
init_robust.npz (per-(radius, start) starting distance, recovery flags, and the
raw per-estimator estimates).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb03 import setup as S


def _dist_to_truth(betas):
    """Mean/median ||beta_hat - beta0|| among recovered runs (finite and < 0.25).

    Diverged and degenerate (nan) estimates land arbitrarily far from beta0, so
    averaging over them would report a number driven by the worst outlier. The
    recovery rate in the same table carries how often that happened.
    """
    B = np.asarray(betas, float)
    fin = np.isfinite(B).all(1)
    dist = np.linalg.norm(B - S.BETA0, axis=1)
    rec = fin & (dist < 0.25)
    if not rec.any():
        return float("nan"), float("nan")
    return float(dist[rec].mean()), float(np.median(dist[rec]))


def main():
    out = common.out_dir(S.NB, "s2_init_robust")

    logb0 = np.log(S.BETA0)
    init_rng = np.random.default_rng(S.SEED_INIT)
    # C8: average over INIT_DATASETS independent datasets, so the reported recovery
    # rates carry sampling variation in the data alongside the variation in starts.
    datasets = [
        S.make_dataset(S.dataset_seed(S.INIT_N, ds), S.INIT_N) for ds in range(S.INIT_DATASETS)
    ]

    nR, nDM = len(S.RADII), S.INIT_DATASETS * S.INIT_M
    starts_a = np.full((nR, nDM), np.nan)
    mp_conv_a = np.full((nR, nDM), np.nan)
    mp_rec_a = np.full((nR, nDM), np.nan)
    rv_rec_a = np.full((nR, nDM), np.nan)
    mle_rec_a = np.full((nR, nDM), np.nan)
    # Retain the raw per-estimator estimates alongside the recovery flags, so a
    # downstream analysis can ask questions the flags collapse away (spread by
    # radius, where the non-recovered runs land). Mirrors s1_finite_sample's
    # beta_hat array.
    mp_beta_a = np.full((nR, nDM, 3), np.nan)
    rv_beta_a = np.full((nR, nDM, 3), np.nan)
    mle_beta_a = np.full((nR, nDM, 3), np.nan)

    rows_rob = []
    for ri, r in enumerate(S.RADII):
        mp_conv, mp_rec, rv_rec, mle_rec, starts = [], [], [], [], []
        mp_beta, rv_beta, mle_beta = [], [], []
        for d0 in datasets:
            y0, b0, o0, d0d = d0["y"], d0["b"], d0["origins"], d0["destinations"]
            for _ in tqdm(range(S.INIT_M), desc=f"r={r:>4}", leave=False):
                binit = np.exp(S.sample_in_ball(init_rng, logb0, r))
                starts.append(np.linalg.norm(binit - S.BETA0))
                mp = S.fit_micropurc(y0, b0, binit)
                rv = S.fit_rv(y0, o0, d0d, binit)
                ml = S.fit_mle(y0, o0, d0d, binit)
                mp_beta.append(np.asarray(mp["beta_hat"], float))
                rv_beta.append(np.asarray(rv["beta_hat"], float))
                mle_beta.append(np.asarray(ml["beta_hat"], float))
                mp_conv.append(mp["converged"])
                mp_rec.append(np.linalg.norm(mp["beta_hat"] - S.BETA0) < 0.25)
                rv_rec.append(np.linalg.norm(rv["beta_hat"] - S.BETA0) < 0.25)
                mle_rec.append(np.linalg.norm(ml["beta_hat"] - S.BETA0) < 0.25)
        starts_a[ri] = starts
        mp_conv_a[ri] = mp_conv
        mp_rec_a[ri] = mp_rec
        rv_rec_a[ri] = rv_rec
        mle_rec_a[ri] = mle_rec
        mp_beta_a[ri] = mp_beta
        rv_beta_a[ri] = rv_beta
        mle_beta_a[ri] = mle_beta
        mp_d, mp_dmed = _dist_to_truth(mp_beta)
        rv_d, rv_dmed = _dist_to_truth(rv_beta)
        mle_d, mle_dmed = _dist_to_truth(mle_beta)
        rows_rob.append(
            {
                "Radius r": r,
                "Starting distance": np.mean(starts),
                "microPURC conv %": 100 * np.mean(mp_conv),
                "microPURC rec %": 100 * np.mean(mp_rec),
                "direct-RV rec %": 100 * np.mean(rv_rec),
                "MLE rec %": 100 * np.mean(mle_rec),
                "microPURC dist": mp_d,
                "direct-RV dist": rv_d,
                "MLE dist": mle_d,
            }
        )
        print(
            f"r={r}: rec%(dist) microPURC {rows_rob[-1]['microPURC rec %']:.0f}%({mp_d:.4f})  "
            f"RV {rows_rob[-1]['direct-RV rec %']:.0f}%({rv_d:.4f})  "
            f"MLE {rows_rob[-1]['MLE rec %']:.0f}%({mle_d:.4f})",
            flush=True,
        )

    tab_rob = pd.DataFrame(rows_rob).set_index("Radius r")
    common.save_table(tab_rob.round(2), out, "init_robust")
    common.save_arrays(
        out,
        "init_robust",
        radii=np.array(S.RADII),
        starts=starts_a,
        mp_conv=mp_conv_a,
        mp_rec=mp_rec_a,
        rv_rec=rv_rec_a,
        mle_rec=mle_rec_a,
        mp_beta=mp_beta_a,
        rv_beta=rv_beta_a,
        mle_beta=mle_beta_a,
    )
    print("PASS s2_init_robust", flush=True)


if __name__ == "__main__":
    main()
