"""nb02 / Sweep 4: cores (parallel scaling).

Fixed problem (grid k=CORE_K, N=CORE_N), core count swept via OMP_NUM_THREADS.
Reports speedup S(c)=T(1)/T(c) and efficiency S(c)/c. Message: the trips are
solved independently, so the estimator scales near-linearly with cores.
Outputs (results/nb02/s4_cores/): sweep_cores table + sweep_cores.npz.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb02 import setup as S


def main():
    out = common.out_dir(S.NB, "s4_cores")
    core_core = S.build_grid_core(
        S.CORE_K, S.GRID_KDIM, seed=S.SEED_SIM, n_pairs=max(4 * S.CORE_N, 4000)
    )
    max_c = os.cpu_count() or 1
    cores = [c for c in S.CORES if c <= max_c] or [1]
    # C5: physical-core cap. Any requested core count above the machine's logical
    # CPU count is dropped and printed, so the reader can see the measured range.
    # Counts above the physical core count run on hyperthreads, and the speedup
    # claim holds only up to that point.
    dropped = [c for c in S.CORES if c > max_c]
    if dropped:
        print(f"NOTE: os.cpu_count()={max_c}; dropped requested core counts {dropped}", flush=True)

    sweep_C = []
    for c in tqdm(cores, desc="cores"):
        S.set_threads(c)
        reps = [S.time_fit(core_core, S.CORE_N, S.SEED_SIM + r) for r in range(S.R)]
        med, lo, hi = S.band(reps, "time_s")
        sweep_C.append(dict(cores=c, t_med=med, t_lo=lo, t_hi=hi, n_threads=reps[0]["n_threads"]))
        print(
            f"cores={c:3d} (native n_threads={reps[0]['n_threads']:3d}): runtime={med:7.2f}s",
            flush=True,
        )
    S.set_threads(max_c)
    df_C = pd.DataFrame(sweep_C)
    t1 = df_C["t_med"].iloc[0]
    df_C["speedup"] = t1 / df_C["t_med"]
    df_C["efficiency"] = df_C["speedup"] / df_C["cores"]

    # C5: Amdahl serial fraction f from the largest-core point, T(c)=T(1)(f+(1-f)/c).
    # Printing f pins the asymptotic speedup ceiling at 1/f, a number the reader can
    # check against the measured curve.
    cmax = df_C["cores"].to_numpy(float)[-1]
    smax = df_C["speedup"].to_numpy(float)[-1]
    serial_frac = float((cmax / smax - 1.0) / (cmax - 1.0)) if cmax > 1 else float("nan")
    ceiling = float(1.0 / serial_frac) if serial_frac and serial_frac > 0 else float("inf")
    print(
        f"Amdahl serial fraction f={serial_frac:.4f} -> speedup ceiling ~{ceiling:.1f}x "
        f"(measured up to {int(cmax)} cores / {max_c} logical CPUs)",
        flush=True,
    )

    common.save_table(df_C.round(3), out, "sweep_cores")
    common.save_arrays(
        out,
        "sweep_cores",
        cores=df_C["cores"].to_numpy(),
        t_med=df_C["t_med"].to_numpy(),
        t_lo=df_C["t_lo"].to_numpy(),
        t_hi=df_C["t_hi"].to_numpy(),
        n_threads=df_C["n_threads"].to_numpy(),
        speedup=df_C["speedup"].to_numpy(),
        efficiency=df_C["efficiency"].to_numpy(),
        serial_frac=np.array(serial_frac),
        logical_cpus=np.array(max_c),
    )
    print("PASS s4_cores", flush=True)


if __name__ == "__main__":
    main()
