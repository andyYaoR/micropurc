"""nb06 / robustness to repeated trips: bias & coverage vs trips-per-traveler.

Fix N trips at rho = 1 (identical repeated trips) and sweep T = trips per traveler
(H = N/T travelers). For each (T, rep) fit homogeneous microPURC and record the
estimate together with the naive (i.i.d.) SE and the cluster-robust SE clustered
by traveler (user). Aggregate over R reps into:

  bias_sd   : bias (+/- MC SE) and Monte-Carlo SD per coefficient  -> unbiasedness
  coverage  : 95% Wald coverage under naive / user-clustered SEs   -> inference
  se_summary: mean naive SE, cluster SE, MC-SD, empirical DEFF     -> mechanism
  fig_repeated_coverage.pdf : coverage vs T (naive dives, cluster holds)

Message: the bias stays flat and near zero as T grows, so repeated trips leave
the point estimates alone. What they change is the naive SE, which deflates and
drags coverage below nominal; clustering by traveler restores it. The empirical
design effect in se_summary tracks T, matching the sqrt(T) closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb06 import setup as S  # noqa: E402

Z196 = 1.96  # naive Wald uses the normal quantile (as in nb01 s1_recovery)


def run_cell(T):
    """R reps at one T; return per-rep (beta, se_naive, se_trav) + G, tcrit."""
    beta = np.full((S.R, S.K), np.nan)
    se_naive = np.full((S.R, S.K), np.nan)
    se_trav = np.full((S.R, S.K), np.nan)
    t_trav = np.full(S.R, np.nan)
    G_trav = np.full(S.R, np.nan)
    for r in tqdm(range(S.R), desc=f"T={T:>3d}", leave=False):
        d = S.make_clustered(S.seed_for(T, r), S.N, T, S.RHO)
        res = S.fit_clustered(d)
        beta[r] = res["beta_hat"]
        se_naive[r] = res["se_naive"]
        s_t, g_t, tc_t = S.cluster_se(res["H"], res["u"], d["traveler_id"])
        se_trav[r], t_trav[r], G_trav[r] = s_t, tc_t, g_t
    return dict(beta=beta, se_naive=se_naive, se_trav=se_trav,
               t_trav=t_trav, G_trav=G_trav)


def _cov(err, se, crit):
    """Coverage flags: |err| <= crit*se; a non-finite SE counts as NOT covered."""
    fin = np.isfinite(se) & np.isfinite(crit)
    return fin & (np.abs(err) <= crit * np.where(fin, se, 0.0))


def summarize(T, c):
    """Aggregate one cell into bias/SD, coverage, and SE-summary row dicts."""
    beta, err = c["beta"], c["beta"] - S.BETA0
    finite = np.isfinite(beta).all(1)
    nf = int(finite.sum())
    br = {"T": T, "H": S.N // T, "n": nf}
    cov = {"T": T, "H": S.N // T, "n": nf}
    se = {"T": T, "H": S.N // T, "G_trav": int(np.nanmean(c["G_trav"]))}
    crit_t = c["t_trav"][:, None]
    naive_ok = _cov(err, c["se_naive"], Z196)
    trav_ok = _cov(err, c["se_trav"], crit_t)
    for k, a in enumerate(S.ATTR):
        e = err[finite, k]
        b = beta[finite, k]
        mc_sd = b.std(ddof=1) if nf > 1 else np.nan
        mean_naive = np.nanmean(c["se_naive"][finite, k])
        br[(a, "Bias")] = e.mean() if nf else np.nan
        br[(a, "MCSE")] = (e.std(ddof=1) / np.sqrt(nf)) if nf > 1 else np.nan
        br[(a, "MC-SD")] = mc_sd
        for label, ok in [("Naive", naive_ok), ("Cluster", trav_ok)]:
            p = 100 * np.mean(ok[finite, k]) if nf else np.nan
            cov[(a, label)] = p
            cov[(a, label + " se")] = 100 * np.sqrt((p / 100) * (1 - p / 100) / nf) if nf else np.nan
        se[(a, "Naive SE")] = mean_naive
        se[(a, "Cluster SE")] = np.nanmean(c["se_trav"][finite, k])
        se[(a, "MC-SD")] = mc_sd
        se[(a, "DEFF")] = (mc_sd / mean_naive) ** 2 if mean_naive else np.nan  # ~ T at rho=1
    return br, cov, se


def main():
    out = common.out_dir(S.NB, "s1_repeated_trips")
    print(f"Repeated-trips study: N={S.N}, rho={S.RHO}, R={S.R}, T in {S.T_LIST}", flush=True)

    cells = {}
    rows_br, rows_cov, rows_se = [], [], []
    for T in S.T_LIST:
        c = run_cell(T)
        cells[T] = c
        nf = int(np.isfinite(c["beta"]).all(1).sum())
        br, cov, se = summarize(T, c)
        rows_br.append(br)
        rows_cov.append(cov)
        rows_se.append(se)
        print(
            f"T={T:>3d} (H={S.N // T:>5d}): finite {nf}/{S.R}  "
            f"naive/cluster cov "
            f"{np.mean([cov[(a, 'Naive')] for a in S.ATTR]):.0f}/"
            f"{np.mean([cov[(a, 'Cluster')] for a in S.ATTR]):.0f}%",
            flush=True,
        )

    def _mk(rows, idx):
        df = pd.DataFrame(rows).set_index(idx)
        df.columns = pd.MultiIndex.from_tuples(
            [c if isinstance(c, tuple) else (c, "") for c in df.columns]
        )
        return df

    tab_br = _mk(rows_br, ["T", "H", "n"])
    tab_cov = _mk(rows_cov, ["T", "H", "n"])
    tab_se = _mk(rows_se, ["T", "H", "G_trav"])
    common.save_table(tab_br.round(4), out, "bias_sd")
    common.save_table(tab_cov.round(2), out, "coverage")
    common.save_table(tab_se.round(4), out, "se_summary")

    # raw per-rep arrays, for redrawing / re-analysis without a re-run
    common.save_arrays(
        out, "repeated_trips",
        T_list=np.array(S.T_LIST), N=S.N, R=S.R, rho=S.RHO, beta0=S.BETA0,
        **{f"beta_T{T}": cells[T]["beta"] for T in S.T_LIST},
        **{f"se_naive_T{T}": cells[T]["se_naive"] for T in S.T_LIST},
        **{f"se_trav_T{T}": cells[T]["se_trav"] for T in S.T_LIST},
    )

    # coverage as plain (nT, K) arrays per SE type, for the figure
    cov_arr = {
        label: np.array([[row[(a, label)] for a in S.ATTR] for row in rows_cov])
        for label in ("Naive", "Cluster")
    }

    # --- figure: coverage vs T, one panel per coefficient ------------------------
    with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.2}):
        fig, axes = plt.subplots(1, S.K, figsize=(11.0, 3.6), sharey=True)
        for k, (ax, a) in enumerate(zip(axes, S.ATTR)):
            for label, mk in [("Naive", "o"), ("Cluster", "s")]:
                ax.plot(S.T_LIST, cov_arr[label][:, k], marker=mk, ms=4, label=label)
            ax.axhline(95, color="0.5", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(S.T_LIST)
            ax.set_xticklabels([str(t) for t in S.T_LIST])
            ax.set_ylim(0, 100)
            ax.set_title(a)
            ax.set_xlabel("trips per traveler $T$")
        axes[0].set_ylabel("95% Wald coverage (%)")
        axes[0].legend(frameon=False, loc="lower left", fontsize=8)
        fig.tight_layout(pad=0.5)
        common.save_fig(fig, out, "fig_repeated_coverage")
        fig.savefig(out / "fig_repeated_coverage.png", dpi=150, bbox_inches="tight")

    print("PASS s1_repeated_trips", flush=True)


if __name__ == "__main__":
    main()
