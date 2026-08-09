"""nb04 / estimation-config knobs, swept one at a time.

For each knob in S.SWEEPS, refit every dataset over that knob's grid with the
other knobs held at their defaults, and measure how far the estimate moves from
the default-tuning reference. Outputs (results/nb04/s1_config_sweeps/): one CSV
table per knob plus the raw sweep arrays config_sweeps.npz (consumed by
s4_figure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb04 import setup as S


def summarize_configs(param, values, datasets, base_beta):
    rows = []
    for v in values:
        params = {**S.DEF, param: v}
        fits = [S.fit_cfg(data, params, S.base_solver) for data in datasets]
        d_beta = [np.linalg.norm(f["beta"] - bb) for f, bb in zip(fits, base_beta)]
        rows.append(
            {
                param: v,
                "max |dbeta|": float(np.max(d_beta)),
                "mean |dbeta|": float(np.mean(d_beta)),
                "mean ||b-b0||": float(
                    np.mean([np.linalg.norm(f["beta"] - S.BETA0) for f in fits])
                ),
                "conv %": 100 * float(np.mean([f["converged"] for f in fits])),
                "mean iters": float(np.mean([f["n_iter"] for f in fits])),
                "med time (s)": float(np.median([f["time_s"] for f in fits])),
            }
        )
    return pd.DataFrame(rows).set_index(param)


def main():
    out = common.out_dir(S.NB, "s1_config_sweeps")
    print(f"Building {S.D} datasets of N={S.SWEEP_N}", flush=True)
    datasets = S.build_datasets()
    _, base_beta = S.build_baseline(datasets)
    print(
        "baseline mean ||beta_hat - beta0|| =",
        float(np.mean([np.linalg.norm(b - S.BETA0) for b in base_beta])),
        flush=True,
    )

    sweep_tables, arrays = {}, {}
    for param, values in tqdm(S.SWEEPS.items(), desc="sweeps"):
        print(f"\n=== sweep: {param} (default {S.DEF[param]}) ===", flush=True)
        df = summarize_configs(param, values, datasets, base_beta)
        sweep_tables[param] = df
        print(df.round(5).to_string(), flush=True)
        common.save_table(df.round(6), out, f"sweep_{param}")
        arrays[f"{param}__values"] = np.asarray(df.index.values, float)
        arrays[f"{param}__max_dbeta"] = df["max |dbeta|"].to_numpy(float)

    common.save_arrays(out, "config_sweeps", **arrays)
    print("PASS s1_config_sweeps", flush=True)


if __name__ == "__main__":
    main()
