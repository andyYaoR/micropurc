"""End-to-end microPURC pipeline on a synthetic dataset.

Runnable companion to ``docs/examples/pipeline.rst``. Builds a synthetic
route-choice dataset on the Sioux Falls topology, then walks the whole
pipeline: data selection, model specification, estimation, reporting, and
reloading the archived run. Run it from anywhere::

    python pipeline.py

It writes a run archive under ``runs/`` in the current directory and prints
the recovered coefficients (which match the known ``beta_true``).
"""

import pathlib

import numpy as np
import scipy.sparse as sp

from micropurc import (
    DGP,
    DGPConfig,
    MarkovSampler,
    MicroPURCEstimator,
    Network,
    PIQPFlowSolver,
    attr,
    beta,
    compile_design,
    log,
    od_from_csv,
)
from micropurc.config import EstimationConfig, SolverConfig
from micropurc.reporting import RunWriter, load_run, render_fit_report

HERE = pathlib.Path(__file__).resolve().parent
NET_CSV = HERE / "data" / "siouxfalls_net.csv"
TRIPS_CSV = HERE / "data" / "siouxfalls_trips.csv"


def build_synthetic_network(seed: int = 7) -> tuple[Network, np.ndarray]:
    """Load the benchmark topology and attach generated link attributes.

    Returns the network (its ``Z`` holds ``tt``, ``toll``, ``scenic``) and a
    positive per-link perturbation scale.
    """
    rng = np.random.default_rng(seed)
    net = Network.from_csv(str(NET_CSV), attribute_cols=["length", "tt"])
    n_links = net.num_links
    travel_time = net.Z[:, net.attribute_index("tt")].copy()

    # Generated attributes: tolls on ~40% of links, and a scenic score.
    toll = np.where(rng.random(n_links) < 0.4, rng.uniform(1.0, 3.0, n_links), 0.0)
    scenic = rng.normal(0.0, 1.0, n_links)

    net.Z = np.column_stack([travel_time, toll, scenic])
    net.attribute_names = ["tt", "toll", "scenic"]
    scale_m = np.ones(n_links)  # uniform quadratic-perturbation scale
    return net, scale_m


def main() -> None:
    # 1. Synthetic network with generated attributes.
    net, scale_m = build_synthetic_network()

    # 2. Model specification with the DSL: log travel time, linear toll, scenic.
    utility = (
        beta("b_time") * log(attr("tt"))
        + beta("b_toll") * attr("toll")
        + beta("b_scenic") * attr("scenic")
    )
    design = compile_design(utility.to_spec(), net)
    net.Z = design.Z  # install the compiled design as the estimation attributes
    net.attribute_names = ["log_tt", "toll", "scenic"]
    param_names = ["b_time", "b_toll", "b_scenic"]

    # 3. Simulate observations at a known beta (the "ground truth").
    solver = PIQPFlowSolver(
        net, scale_m, SolverConfig(backend="piqp", eps_abs=1e-10, eps_duality_gap_abs=1e-10)
    )
    od = od_from_csv(
        net, str(TRIPS_CSV), origin_col="origin", dest_col="destination",
        weight_col="sampling_weight",
    )
    beta_true = np.array([0.5, 0.4, 0.3])
    dgp = DGP(
        network=net, beta_true=beta_true, forward_solver=solver,
        route_sampler=MarkovSampler(net), od_dist=od,
        rng=np.random.default_rng(1), config=DGPConfig(max_tries_per_trip=30),
    )
    data = dgp.sample_dataset(n_trips=3000)
    y, b, origins, dests = data["y"], data["b"], data["origins"], data["destinations"]

    # 4. Data selection: keep trips whose observed route uses more than one link.
    route_len = np.asarray((y > 0).sum(axis=1)).ravel()
    keep = route_len >= 2
    y, b, origins, dests = y[keep], b[keep], origins[keep], dests[keep]
    print(f"selection: kept {int(keep.sum())}/{keep.size} multi-link trips")

    # 5. Estimation.
    est_cfg = EstimationConfig(max_iterations=200, fp_tol=1e-3, nm_window=3, nm_c=1e-4)
    est = MicroPURCEstimator(network=net, forward_solver=solver, estimation_config=est_cfg)
    result = est.fit(y=y, b=b, beta_init=np.zeros(3))
    print(f"beta_true = {beta_true}")
    print(f"beta_hat  = {np.round(result['beta_hat'], 3)}  "
          f"(converged={result['diagnostics']['converged']})")

    # 6. Reporting: archive config, data snapshot, estimates, and compute profile.
    writer = RunWriter(
        "runs", "sioux_synthetic",
        config={"estimation": est_cfg, "beta_true": beta_true.tolist(), "seed": 1},
    )
    writer.snapshot_data(
        edges=net.edges, Z=net.Z, A=sp.csc_matrix(net.A), scale_m=scale_m,
        y=sp.csr_matrix(y), origins=origins, dests=dests,
    )
    run_dir = writer.finalize(result={
        "beta_hat": result["beta_hat"],
        "se_hat": result.get("se_hat"),
        "converged": result["diagnostics"]["converged"],
        "Q_history": result["diagnostics"].get("Q_history"),
    })
    print(f"archived run at {run_dir}")

    # 7. Reload the archived run and analyze predictions.
    run = load_run(run_dir)
    assert np.allclose(run.beta_hat, result["beta_hat"])
    od_pair = (int(origins[0]), int(dests[0]))
    flows = run.predicted_flow(od_pairs=[od_pair])
    active = run.active_subnetwork(od_pair)
    print(f"reloaded beta_hat = {np.round(run.beta_hat, 3)}; "
          f"OD {od_pair}: {int((flows[od_pair] > 1e-9).sum())} links carry flow, "
          f"{len(active['tail'])} in the active subnetwork")
    report = render_fit_report(run, param_names=param_names)
    print(f"HTML report at {report}")


if __name__ == "__main__":
    main()
