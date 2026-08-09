"""nb01 / active-set switching per iteration.

Outputs (results/nb01/s4_switching/): fig_switch_iter.pdf + switching.npz.
The fit is guarded (a singular H_N at small N skips the rep) so a bad seed
never aborts the section.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from micropurc.config import ProjectionConfig  # noqa: E402
from micropurc.core.estimator import MicroPURCEstimator  # noqa: E402
from scripts.paper.nb01 import setup as S  # noqa: E402


def switch_series(rep, N):
    """Per-iteration share of trips whose active set changed, padded to SWITCH_MAXIT.

    Returns (padded_percentages, converged), or None when the fit raised.
    """
    d = S.make_dataset(S.dataset_seed(N, rep), N)  # A3: independent per-(N, rep) draw
    est = MicroPURCEstimator(
        network=S.net,
        forward_solver=S.solver,
        estimation_config=S.switch_cfg,
        active_set_config=S.active_filter,
        projection_config=ProjectionConfig(),
    )
    try:
        res = est.fit(y=d["y"], b=d["b"], beta_init=np.ones(3))
    except Exception:  # noqa: BLE001  -- singular H_N at small N; skip this rep
        return None
    flips = np.array([int(r.link_flips or 0) for r in est.diagnostics.records], dtype=float)
    pct = 100.0 * flips / N
    padded = np.zeros(S.SWITCH_MAXIT)
    k = min(len(pct), S.SWITCH_MAXIT)
    padded[:k] = pct[:k]
    return padded, bool(res["diagnostics"]["converged"])


def main():
    out = common.out_dir(S.NB, "s4_switching")
    switches = {}
    for N in S.SWITCH_N_LIST:
        rows, n_conv, n_exc = [], 0, 0
        for seed in tqdm(range(S.SWITCH_R), desc=f"switching N={N}"):
            res = switch_series(seed, N)
            if res is None:
                n_exc += 1
                continue
            pct, conv = res
            rows.append(pct)
            n_conv += int(conv)
        M = np.array(rows)
        switches[N] = {
            "mean_pct": M.mean(axis=0),
            "std_pct": M.std(axis=0, ddof=1),
            "conv_rate": n_conv / len(rows) if rows else float("nan"),
            "n_exc": n_exc,
        }
        print(
            f"N={N}: conv_rate={switches[N]['conv_rate']:.2f} ({len(rows)} valid, {n_exc} skipped)",
            flush=True,
        )

    xmax = 5.4
    markers = {200: "o", 1000: "s", 10000: "^"}
    with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.1}):
        fig, ax = plt.subplots(figsize=(5.8, 3.9))
        for N in S.SWITCH_N_LIST:
            m = np.array(switches[N]["mean_pct"])[1:]
            sd = np.array(switches[N]["std_pct"])[1:]
            it = np.arange(1, len(m) + 1)
            zeros = np.where(m <= 0)[0]
            end = (zeros[0] + 1) if len(zeros) else len(m)
            it, m, sd = it[:end], m[:end], sd[:end]
            (line,) = ax.plot(
                it,
                m,
                marker=markers.get(N, "o"),
                ms=3.5,
                lw=1.0,
                label=f"$N={N:,}$".replace(",", r"{,}"),
            )
            ax.fill_between(
                it, np.maximum(m - sd, 0), m + sd, color=line.get_color(), alpha=0.18, linewidth=0
            )
        ax.set_ylim(0, 80)
        ax.set_xlim(1.0, xmax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Trips changing active set (%)")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout(pad=0.4)
        common.save_fig(fig, out, "fig_switch_iter")

    common.save_arrays(
        out,
        "switching",
        **{f"mean_N{N}": switches[N]["mean_pct"] for N in S.SWITCH_N_LIST},
        **{f"std_N{N}": switches[N]["std_pct"] for N in S.SWITCH_N_LIST},
    )
    print("PASS s4_switching", flush=True)


if __name__ == "__main__":
    main()
