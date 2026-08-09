"""nb01 / finite-sample performance: recovery over R reps x N_LIST.

Outputs (results/nb01/s1_recovery/): the bias_rmse, sd_se, and runtime tables,
as CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common
from scripts.paper.nb01 import setup as S


def _converged(rows):
    """Reps that converged and returned a finite estimate."""
    return [x for x in rows if x["converged"] and np.isfinite(x["beta_hat"]).all()]


def _finite(rows):
    """Reps that returned a finite estimate, whether or not they converged."""
    return [x for x in rows if np.isfinite(np.asarray(x["beta_hat"], float)).all()]


def _se_matrix(subset, K):
    """Stack the per-rep standard errors; a rep with no usable SE stays NaN."""
    M = np.full((len(subset), K), np.nan)
    for i, x in enumerate(subset):
        s = np.asarray(x["se_hat"], float).ravel()
        if s.shape == (K,):
            M[i] = s
    return M


def main():
    out = common.out_dir(S.NB, "s1_recovery")
    records = {}
    print(f"Running {S.R} reps for each N in {S.N_LIST}", flush=True)
    for N in S.N_LIST:
        rows = []
        for r in tqdm(range(S.R), desc=f"N={N:>6d}"):
            d = S.make_dataset(S.dataset_seed(N, r), N)  # A3: independent per-(N, rep) draw
            rows.append(S.fit_one(d["y"], d["b"], np.ones(3)))
        records[N] = rows
        n_crash = sum(1 for x in rows if not np.isfinite(np.asarray(x["beta_hat"], float)).all())
        print(
            f"N={N:6d}: convergence rate = {100 * np.mean([x['converged'] for x in rows]):.1f}%"
            f"  (crashed/non-finite estimates: {n_crash}/{S.R})",
            flush=True,
        )

    rows_br, rows_ss, rows_rt = [], [], []
    K = len(S.ATTR)
    for N, rows in records.items():
        # B3: the "finite" subset is every attempted rep whose estimate is finite,
        # and the "converged" subset is the reps that also converged. Both carry
        # their own count n, and the crashed remainder is reported separately, so
        # every denominator behind a number stays visible.
        subsets = {"converged": _converged(rows), "finite": _finite(rows)}
        for label, subset in subsets.items():
            B = np.array([x["beta_hat"] for x in subset], dtype=float)
            SE = _se_matrix(subset, K)
            err = B - S.BETA0
            br = {"N": N, "set": label, "n": len(subset)}
            ss = {"N": N, "set": label, "n": len(subset)}
            for k, a in enumerate(S.ATTR):
                rmse = np.sqrt((err[:, k] ** 2).mean())
                br[(a, "Bias")] = err[:, k].mean()
                br[(a, "RMSE")] = rmse
                br[(a, "sqrtN-RMSE")] = np.sqrt(N) * rmse
                ss[(a, "SD")] = B[:, k].std(ddof=1)
                ss[(a, "SE")] = np.nanmean(SE[:, k])
                # B2: the coverage denominator is every rep in the subset, and a rep
                # whose SE is non-finite counts as uncovered. Dropping those reps from
                # both numerator and denominator would inflate the reported coverage.
                fin = np.isfinite(SE[:, k])
                covered = fin & (np.abs(err[:, k]) <= 1.96 * np.where(fin, SE[:, k], 0.0))
                ss[(a, "Cov%")] = 100 * np.mean(covered) if len(subset) else np.nan
                ss[(a, "SE_fail")] = int((~fin).sum())
            rows_br.append(br)
            rows_ss.append(ss)
        conv = _converged(rows)
        n_crash = sum(1 for x in rows if not np.isfinite(np.asarray(x["beta_hat"], float)).all())
        rows_rt.append(
            {
                "N": N,
                "Conv. rate %": 100 * np.mean([x["converged"] for x in rows]),
                "Avg. iters": np.mean([x["n_iter"] for x in conv]) if conv else np.nan,
                "Avg. runtime [s]": np.mean([x["time_s"] for x in rows]),
                "Crashed": n_crash,
            }
        )

    tab_bias_rmse = pd.DataFrame(rows_br).set_index(["N", "set", "n"])
    tab_bias_rmse.columns = pd.MultiIndex.from_tuples(tab_bias_rmse.columns)
    tab_sd_se = pd.DataFrame(rows_ss).set_index(["N", "set", "n"])
    tab_sd_se.columns = pd.MultiIndex.from_tuples(tab_sd_se.columns)
    tab_runtime = pd.DataFrame(rows_rt).set_index("N").T

    common.save_table(tab_bias_rmse.round(4), out, "bias_rmse")
    common.save_table(tab_sd_se.round(4), out, "sd_se")
    common.save_table(tab_runtime.round(3), out, "runtime")
    print("PASS s1_recovery", flush=True)


if __name__ == "__main__":
    main()
