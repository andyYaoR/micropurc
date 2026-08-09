"""nb03 / finite-sample comparison of the four estimators.

For each sample size we draw R datasets and run all four estimators from the
common start beta_(0) = 1, then report per-estimator RMSE, runtime, recovery
rate, and the MLE degenerate-start share.

Outputs (results/nb03/s1_finite_sample/): the estimator_comparison table as CSV,
and finite_sample.npz (raw per-(estimator, N) time_s and beta_hat arrays, which
the efficiency figure in s3 loads to rebuild the runtime curves).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb03 import setup as S


def summarize(recs):
    # B4: every denominator is all R replications, so a failure (e.g. the MLE from a
    # degenerate start -> NaN) is carried into the reported rate and charged to the
    # estimator that produced it. Recovery counts a non-finite estimate as a
    # non-recovery; RMSE is computed over the finite estimates and reported beside
    # the finite share, which supplies the conditioning the RMSE alone would leave
    # implicit. The definition is then identical across the four estimators and
    # matches the init-robustness table (s2).
    R = len(recs)
    B = np.array([np.asarray(x["beta_hat"], float) for x in recs])
    fin = np.isfinite(B).all(1)
    Bf = B[fin]
    rmse = np.sqrt(np.mean(np.sum((Bf - S.BETA0) ** 2, axis=1))) if len(Bf) else np.nan
    rt = np.median([x["time_s"] for x in recs])
    recovered = [bool(f) and np.linalg.norm(b - S.BETA0) < 0.25 for b, f in zip(B, fin)]
    out = {
        "RMSE (finite)": rmse,
        "Finite %": 100 * np.mean(fin),
        "Runtime (s)": rt,
        "Recovered %": 100 * np.mean(recovered),  # denominator = all R
    }
    if "n_zero_init" in recs[0]:
        out["degenerate start %"] = 100 * np.mean([x.get("n_zero_init", 0) > 0 for x in recs])
    return out


def main():
    out = common.out_dir(S.NB, "s1_finite_sample")

    records = {e: {N: [] for N in S.N_LIST} for e in S.ESTIMATORS}
    for N in S.N_LIST:
        for rep in tqdm(range(S.R), desc=f"N={N:>5d}", leave=True):
            d = S.make_dataset(S.dataset_seed(N, rep), N)  # A3: independent per-(N, rep) draw
            y, b, origins, dests = d["y"], d["b"], d["origins"], d["destinations"]
            beta_init = np.ones(3)
            records["microPURC"][N].append(S.fit_micropurc(y, b, beta_init))
            records["PURC-OLS"][N].append(S.fit_purc_ols(y, origins, dests))
            records["MLE"][N].append(S.fit_mle(y, origins, dests, beta_init))
            records["direct-RV"][N].append(S.fit_rv(y, origins, dests, beta_init))
        print(f"N={N}: done", flush=True)

    rows = []
    for e in S.ESTIMATORS:
        for N in S.N_LIST:
            rows.append({"Estimator": e, "N": N, **summarize(records[e][N])})
    tab_est = pd.DataFrame(rows).set_index(["Estimator", "N"])
    common.save_table(tab_est.round(3), out, "estimator_comparison")

    # Raw per-(estimator, N) arrays behind the efficiency figure (rectangular:
    # R reps is constant across every N).
    E, nN = len(S.ESTIMATORS), len(S.N_LIST)
    time_s = np.full((E, nN, S.R), np.nan)
    beta_hat = np.full((E, nN, S.R, 3), np.nan)
    for ei, e in enumerate(S.ESTIMATORS):
        for ni, N in enumerate(S.N_LIST):
            recs = records[e][N]
            for ri, x in enumerate(recs):
                time_s[ei, ni, ri] = x["time_s"]
                beta_hat[ei, ni, ri] = np.asarray(x["beta_hat"], float)
    common.save_arrays(
        out,
        "finite_sample",
        estimators=np.array(S.ESTIMATORS),
        N_list=np.array(S.N_LIST),
        time_s=time_s,
        beta_hat=beta_hat,
    )
    print("PASS s1_finite_sample", flush=True)


if __name__ == "__main__":
    main()
