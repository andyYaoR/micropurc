"""nb04 / Hard-start check: does the line-search window change the fixed point?

HARD_M far-away starts (drawn in a log-space ball of radius HARD_R around
beta0) are fit on one dataset for each nonmonotone-window size nm_window in
{1, 3, 5}. The reported "reached estimate %" is the share of those starts whose
estimate lands within 1e-2 of the common-start (beta = 1) estimate on the same
dataset, so a window that moved the fixed point would show up as a drop.

Outputs (results/nb04/s3_hardstart/): the hardstart table as CSV, plus
hardstart.npz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb04 import setup as S


def main():
    out = common.out_dir(S.NB, "s3_hardstart")
    hard_data = S.make_dataset(S.SEED_SIM + 999, S.HARD_N)
    ref_beta = S.fit_cfg(hard_data, S.DEF, S.base_solver)["beta"]  # common-start estimate
    logb0 = np.log(S.BETA0)
    rng = np.random.default_rng(S.SEED_INIT)
    starts = [np.exp(S.sample_in_ball(rng, logb0, S.HARD_R)) for _ in range(S.HARD_M)]

    rows = []
    for M_window in [1, 3, 5]:
        params = {**S.DEF, "nm_window": M_window}
        reached, conv, iters = [], [], []
        for b0 in tqdm(starts, desc=f"M={M_window}", leave=False):
            f = S.fit_cfg(hard_data, params, S.base_solver, beta_init=b0)
            reached.append(np.linalg.norm(f["beta"] - ref_beta) < 1e-2)
            conv.append(f["converged"])
            iters.append(f["n_iter"])
        rows.append(
            {
                "nm_window M": M_window,
                "conv %": 100 * np.mean(conv),
                "reached estimate %": 100 * np.mean(reached),
                "mean iters": np.mean(iters),
            }
        )
    tab_hard = pd.DataFrame(rows).set_index("nm_window M")
    print(tab_hard.round(1).to_string(), flush=True)

    common.save_table(tab_hard.round(4), out, "hardstart")
    common.save_arrays(
        out,
        "hardstart",
        nm_window=tab_hard.index.to_numpy(float),
        conv=tab_hard["conv %"].to_numpy(float),
        reached=tab_hard["reached estimate %"].to_numpy(float),
        mean_iters=tab_hard["mean iters"].to_numpy(float),
        ref_beta=np.asarray(ref_beta, float),
    )
    print("PASS s3_hardstart", flush=True)


if __name__ == "__main__":
    main()
