"""nb02 / Sweep 1: sample size N.

Fixed grid (k=SWEEP_K_FIXED), fixed K, all cores. Message: runtime is linear
in N. Outputs (results/nb02/s1_sweep_N/): sweep_N table + sweep_N.npz.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb02 import setup as S


def main():
    out = common.out_dir(S.NB, "s1_sweep_N")
    S.set_threads(os.cpu_count())
    n_core = S.build_grid_core(
        S.SWEEP_K_FIXED, S.GRID_KDIM, seed=S.SEED_SIM, n_pairs=max(4 * max(S.N_LIST), 4000)
    )
    sweep_N = []
    for N in tqdm(S.N_LIST, desc="N sweep"):
        reps = [S.time_fit(n_core, N, S.SEED_SIM + r) for r in range(S.R)]
        med, lo, hi = S.band(reps, "time_s")
        sweep_N.append(
            dict(
                N=N,
                t_med=med,
                t_lo=lo,
                t_hi=hi,
                n_iter=np.median([r["n_iter"] for r in reps]),
                # C2: unique ODs solved. If this saturates as N grows, the remaining
                # growth is the cheap per-trip accumulation, and the linear trend would
                # then be linear for the wrong reason -- so the count is reported.
                n_unique_od=int(np.median([r["n_unique_od"] for r in reps])),
            )
        )
        print(
            f"N={N:6d}: runtime={med:7.2f}s  [{lo:.2f}, {hi:.2f}]  "
            f"iters={sweep_N[-1]['n_iter']:.0f}  unique_OD={sweep_N[-1]['n_unique_od']}",
            flush=True,
        )
    df_N = pd.DataFrame(sweep_N)
    # C4: the linear reference fixes the slope at 1 and fits only the intercept, by
    # least squares on log t - log N over every sweep point. Anchoring it to the whole
    # sweep keeps the drawn line independent of any single measurement.
    logN, logt = np.log(df_N["N"].to_numpy(float)), np.log(df_N["t_med"].to_numpy(float))
    intercept = np.mean(logt - logN)  # least-squares intercept at unit slope
    df_N["linear_ref"] = np.exp(intercept) * df_N["N"]

    common.save_table(df_N.round(3), out, "sweep_N")
    common.save_arrays(
        out,
        "sweep_N",
        N=df_N["N"].to_numpy(),
        t_med=df_N["t_med"].to_numpy(),
        t_lo=df_N["t_lo"].to_numpy(),
        t_hi=df_N["t_hi"].to_numpy(),
        n_iter=df_N["n_iter"].to_numpy(),
        n_unique_od=df_N["n_unique_od"].to_numpy(),
        linear_ref=df_N["linear_ref"].to_numpy(),
    )
    print("PASS s1_sweep_N", flush=True)


if __name__ == "__main__":
    main()
