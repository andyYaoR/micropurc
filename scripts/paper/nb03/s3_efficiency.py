"""nb03 / efficiency: runtime-vs-N curves for the four estimators.

Each section runs as its own process, so this script loads the raw runtime
arrays saved by s1_finite_sample (finite_sample.npz) and rebuilds the figure.
If that file is missing (s1 not run yet), it skips.

Outputs (results/nb03/s3_efficiency/): fig_estimator_efficiency.pdf.
"""

from __future__ import annotations

import numpy as np

from scripts.paper import common

common.agg_backend()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.paper.nb03 import setup as S  # noqa: E402

MARKERS = {"microPURC": "o", "PURC-OLS": "s", "MLE": "^", "direct-RV": "D"}


def main():
    out = common.out_dir(S.NB, "s3_efficiency")
    src = common.RESULTS / S.NB / "s1_finite_sample" / "finite_sample.npz"
    if not src.exists():
        print(f"SKIP s3_efficiency: {src} not found (run s1_finite_sample first)", flush=True)
        return

    data = np.load(src, allow_pickle=True)
    estimators = [str(e) for e in data["estimators"]]
    N_list = [int(n) for n in data["N_list"]]
    time_s = data["time_s"]  # (E, nN, R)
    rt_med = np.nanmedian(time_s, axis=2)  # (E, nN) median runtime per (estimator, N)

    with plt.rc_context({"axes.linewidth": 0.6, "lines.linewidth": 1.1}):
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        for ei, e in enumerate(estimators):
            ax.loglog(N_list, rt_med[ei], marker=MARKERS.get(e, "o"), ms=4, label=e)
        ax.set_xlabel(r"observations $N$")
        ax.set_ylabel("runtime (s)")
        ax.legend(frameon=False)
        fig.tight_layout(pad=0.4)
        common.save_fig(fig, out, "fig_estimator_efficiency")

    print("PASS s3_efficiency", flush=True)


if __name__ == "__main__":
    main()
