"""nb02 / Sweep 2: network size |L|.

Grid side k swept (with benchmark networks as anchors), N and K fixed; also
records the mean active-set size per trip. Message: runtime tracks the
active-set size, which grows far more slowly than |L|. Outputs
(results/nb02/s2_sweep_L/): sweep_L and sweep_L_anchor tables + sweep_L.npz
(grid_* and anchor_* arrays).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb02 import setup as S


def main():
    out = common.out_dir(S.NB, "s2_sweep_L")

    # --- grid sweep ----------------------------------------------------------
    sweep_L = []
    for k in tqdm(S.GRID_K_LIST, desc="grid size"):
        core = S.build_grid_core(k, S.GRID_KDIM, seed=S.SEED_SIM)
        reps = [
            S.time_fit(core, S.GRID_N, S.SEED_SIM + r, want_active=(r == 0)) for r in range(S.R)
        ]
        med, lo, hi = S.band(reps, "time_s")
        sweep_L.append(
            dict(
                kind="grid",
                label=f"grid k={k}",
                V=core["net"].num_nodes,
                L=core["net"].num_links,
                t_med=med,
                t_lo=lo,
                t_hi=hi,
                active=reps[0]["active"],
                active_p90=reps[0]["active_p90"],
                active_p99=reps[0]["active_p99"],
                mem_mb=np.median([r["mem_mb"] for r in reps]),
            )
        )
        print(
            f"grid k={k:2d}: |L|={core['net'].num_links:5d}  runtime={med:7.2f}s  "
            f"active={reps[0]['active']:.1f}",
            flush=True,
        )
    df_L = pd.DataFrame(sweep_L)

    # --- benchmark-network anchors (real topologies, synthetic demand) -------
    anchor = []
    for name in S.REAL_NETWORKS:
        if not ((S.DATA / f"{name}_net.csv").exists() and (S.DATA / f"{name}_trips.csv").exists()):
            print(f"{name:>13}: data not bundled — skipping", flush=True)
            continue
        core = S.build_benchmark_core(name, seed=S.SEED_SIM)
        reps = [
            S.time_fit(core, S.N_REAL, S.SEED_SIM + r, want_active=(r == 0)) for r in range(S.R)
        ]
        med, lo, hi = S.band(reps, "time_s")
        anchor.append(
            dict(
                Network=name,
                V=core["net"].num_nodes,
                L=core["net"].num_links,
                N=S.N_REAL,
                conv=100 * np.mean([r["converged"] for r in reps]),
                runtime_s=med,
                mem_mb=np.median([r["mem_mb"] for r in reps]),
                active=reps[0]["active"],
                active_p99=reps[0]["active_p99"],
                iters=np.median([r["n_iter"] for r in reps]),
            )
        )
        print(
            f"{name:>13}: |L|={core['net'].num_links:6d}  runtime={med:7.2f}s  "
            f"mem+={anchor[-1]['mem_mb']:6.0f}MB  active={reps[0]['active']:.1f}",
            flush=True,
        )
    tab_real = pd.DataFrame(anchor)

    common.save_table(df_L.round(2), out, "sweep_L")
    if len(tab_real):
        common.save_table(tab_real.set_index("Network").round(2), out, "sweep_L_anchor")

    arrays = dict(
        grid_L=df_L["L"].to_numpy(),
        grid_V=df_L["V"].to_numpy(),
        grid_t_med=df_L["t_med"].to_numpy(),
        grid_t_lo=df_L["t_lo"].to_numpy(),
        grid_t_hi=df_L["t_hi"].to_numpy(),
        grid_active=df_L["active"].to_numpy(),
        grid_active_p90=df_L["active_p90"].to_numpy(),
        grid_active_p99=df_L["active_p99"].to_numpy(),
        grid_mem_mb=df_L["mem_mb"].to_numpy(),
    )
    if len(tab_real):
        arrays.update(
            anchor_Network=tab_real["Network"].to_numpy().astype(str),
            anchor_V=tab_real["V"].to_numpy(),
            anchor_L=tab_real["L"].to_numpy(),
            anchor_N=tab_real["N"].to_numpy(),
            anchor_conv=tab_real["conv"].to_numpy(),
            anchor_runtime_s=tab_real["runtime_s"].to_numpy(),
            anchor_mem_mb=tab_real["mem_mb"].to_numpy(),
            anchor_active=tab_real["active"].to_numpy(),
            anchor_active_p99=tab_real["active_p99"].to_numpy(),
            anchor_iters=tab_real["iters"].to_numpy(),
        )
    common.save_arrays(out, "sweep_L", **arrays)
    print("PASS s2_sweep_L", flush=True)


if __name__ == "__main__":
    main()
