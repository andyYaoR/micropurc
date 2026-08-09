"""nb02 / Sweep 3: parameter count K.

Fixed grid, fixed N, number of link attributes K swept. Message: runtime is
flat in K. Outputs (results/nb02/s3_sweep_K/): sweep_K table + sweep_K.npz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb02 import setup as S


def main():
    out = common.out_dir(S.NB, "s3_sweep_K")
    sweep_K = []
    for Kdim in tqdm(S.K_SWEEP, desc="K sweep"):
        core = S.build_grid_core(S.SWEEP_K_FIXED, Kdim, seed=S.SEED_SIM)
        reps = [S.time_fit(core, S.GRID_N, S.SEED_SIM + r) for r in range(S.R)]
        med, lo, hi = S.band(reps, "time_s")
        times = np.array([r["time_s"] for r in reps])
        sweep_K.append(
            dict(
                K=Kdim, t_med=med, t_lo=lo, t_hi=hi,
                t_mean=float(times.mean()),
                t_std=float(times.std(ddof=1)) if times.size > 1 else 0.0,
                n_iter=np.median([r["n_iter"] for r in reps]),
            )
        )
        print(
            f"K={Kdim}: runtime mean={times.mean():7.2f}s  std={times.std(ddof=1):.3f}"
            f"  (median {med:.2f})",
            flush=True,
        )
    df_K = pd.DataFrame(sweep_K)

    common.save_table(df_K.round(3), out, "sweep_K")
    common.save_arrays(
        out,
        "sweep_K",
        K=df_K["K"].to_numpy(),
        t_med=df_K["t_med"].to_numpy(),
        t_lo=df_K["t_lo"].to_numpy(),
        t_hi=df_K["t_hi"].to_numpy(),
        t_mean=df_K["t_mean"].to_numpy(),
        t_std=df_K["t_std"].to_numpy(),
        n_iter=df_K["n_iter"].to_numpy(),
    )
    print("PASS s3_sweep_K", flush=True)


if __name__ == "__main__":
    main()
