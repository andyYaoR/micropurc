"""nb04 / inner QP solver tolerance.

Refits every dataset (default tuning) across a grid of PIQP absolute
tolerances and measures how far the estimate moves from the default-tolerance
reference. Outputs (results/nb04/s2_qp_tol/): the qp_tol table as CSV, plus the
raw sweep arrays qp_tol.npz (consumed by s4_figure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb04 import setup as S


def main():
    out = common.out_dir(S.NB, "s2_qp_tol")
    print(f"Building {S.D} datasets of N={S.SWEEP_N}", flush=True)
    datasets = S.build_datasets()
    _, base_beta = S.build_baseline(datasets)

    rows = []
    for eps in tqdm(S.EPS_SWEEP, desc="eps_abs"):
        solver = S.make_solver(eps_abs=eps)
        fits = [S.fit_cfg(data, S.DEF, solver) for data in datasets]
        d_beta = [np.linalg.norm(f["beta"] - bb) for f, bb in zip(fits, base_beta)]
        rows.append(
            {
                "eps_abs": eps,
                "max |dbeta|": float(np.max(d_beta)),
                "mean ||b-b0||": float(
                    np.mean([np.linalg.norm(f["beta"] - S.BETA0) for f in fits])
                ),
                "conv %": 100 * float(np.mean([f["converged"] for f in fits])),
                "mean iters": float(np.mean([f["n_iter"] for f in fits])),
            }
        )
    tab_eps = pd.DataFrame(rows).set_index("eps_abs")
    print(tab_eps.round(6).to_string(), flush=True)

    common.save_table(tab_eps.round(6), out, "qp_tol")
    common.save_arrays(
        out,
        "qp_tol",
        eps_abs=np.asarray(tab_eps.index.values, float),
        max_dbeta=tab_eps["max |dbeta|"].to_numpy(float),
    )
    print("PASS s2_qp_tol", flush=True)


if __name__ == "__main__":
    main()
