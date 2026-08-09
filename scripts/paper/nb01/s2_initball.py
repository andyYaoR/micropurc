"""nb01 / robustness to initialization: init-ball over RADII.

For each radius, draw INIT_M starts per dataset over INIT_DATASETS independent
datasets and record how often the fit converges and how close it lands.

Outputs (results/nb01/s2_initball/): the initball table, as CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb01 import setup as S


def main():
    out = common.out_dir(S.NB, "s2_initball")
    logb0 = np.log(S.BETA0)
    init_rng = np.random.default_rng(S.SEED_INIT)
    datasets = [
        (lambda ds: (ds["y"], ds["b"]))(S.make_dataset(S.dataset_seed(S.INIT_N, d), S.INIT_N))
        for d in range(S.INIT_DATASETS)
    ]
    n_starts = S.INIT_DATASETS * S.INIT_M
    rows = []
    for r in S.RADII:
        start, acc, conv, iters = [], [], [], []
        bar = tqdm(total=n_starts, desc=f"r={r:>4}")
        for y, b in datasets:
            for _ in range(S.INIT_M):
                binit = np.exp(S.sample_in_ball(init_rng, logb0, r))
                o = S.fit_one(y, b, binit)
                start.append(np.linalg.norm(binit - S.BETA0))
                conv.append(o["converged"])
                if o["converged"] and np.isfinite(o["beta_hat"]).all():
                    acc.append(np.linalg.norm(o["beta_hat"] - S.BETA0))
                    iters.append(o["n_iter"])
                bar.update(1)
                bar.set_postfix(conv=f"{100 * np.mean(conv):.0f}%")
        bar.close()
        rows.append(
            {
                "Radius r": r,
                "Starting distance": np.mean(start),
                "Accuracy": np.mean(acc) if acc else np.nan,
                "Conv. rate %": 100 * np.mean(conv),
                "Avg. iters": np.mean(iters) if iters else np.nan,
            }
        )
        print(
            f"r={r}: conv {rows[-1]['Conv. rate %']:.0f}%  acc {rows[-1]['Accuracy']}", flush=True
        )

    tab_cball = pd.DataFrame(rows).set_index("Radius r")
    common.save_table(tab_cball.round(4), out, "initball")
    print("PASS s2_initball", flush=True)


if __name__ == "__main__":
    main()
