An end-to-end pipeline
======================

This walkthrough runs the whole microPURC workflow on a **synthetic dataset**
built from the Sioux Falls topology: generate link attributes, simulate
observations at a known parameter vector, then select observations, specify a
utility, estimate, archive the run, and reload it for analysis. Because the
data are simulated at a known ``beta_true``, the recovered coefficients are a
built-in correctness check.

The runnable script for the main pipeline is :download:`pipeline.py
<pipeline.py>`; run it from this ``docs/examples`` directory so the ``data/``
paths below resolve. The snippets that follow mirror that script, plus one
standalone aside on segmented models.

A synthetic network
-------------------

We load the benchmark topology and attach two *generated* link attributes -- a
``toll`` on a subset of links and a ``scenic`` score -- alongside the real
travel time ``tt``. A positive per-link ``scale_m`` sets the
quadratic-perturbation scale; a uniform scale keeps it independent of the
utility.

.. code-block:: python

    import numpy as np
    from micropurc import Network

    rng = np.random.default_rng(7)
    net = Network.from_csv("data/siouxfalls_net.csv", attribute_cols=["length", "tt"])
    n_links = net.num_links
    travel_time = net.Z[:, net.attribute_index("tt")].copy()

    toll = np.where(rng.random(n_links) < 0.4, rng.uniform(1.0, 3.0, n_links), 0.0)
    scenic = rng.normal(0.0, 1.0, n_links)

    net.Z = np.column_stack([travel_time, toll, scenic])
    net.attribute_names = ["tt", "toll", "scenic"]
    scale_m = np.ones(n_links)

Specifying the utility
----------------------

The utility is declared with the model-spec DSL and lowered to a design with
:func:`~micropurc.spec.compile.compile_design`. Here travel time enters through
a ``log`` transform; toll and scenic enter linearly. Installing ``design.Z`` as
the network's attributes makes the compiled design the estimation design.

.. code-block:: python

    from micropurc import beta, attr, log, compile_design

    utility = (
        beta("b_time") * log(attr("tt"))
        + beta("b_toll") * attr("toll")
        + beta("b_scenic") * attr("scenic")
    )
    design = compile_design(utility.to_spec(), net)
    net.Z = design.Z
    net.attribute_names = ["log_tt", "toll", "scenic"]

The DSL also expresses coefficients that vary across covariate segments. To let
the toll sensitivity differ by an ``income`` covariate, compile a segmented
design -- it lowers to per-variant designs (or a lazy design) that
:meth:`~micropurc.core.estimator.MicroPURCEstimator.fit` accepts via
``z_variants``/``trip_variant`` or ``lazy_design`` (see
:doc:`model_spec`):

.. code-block:: python

    from micropurc import seg

    segmented = (
        beta("b_toll_low") * attr("toll") * seg("income", "<=", 3.0)
        + beta("b_toll_high") * attr("toll") * seg("income", ">", 3.0)
    ).to_spec()
    # A small standalone illustration: two income variants, one id per trip
    # (in a real fit, trip_variant is one id per simulated trip).
    cd = compile_design(
        segmented, net,
        variant_covariates=np.array([[1.0], [5.0]]),   # low- and high-income
        covariate_names=["income"],
        trip_variant=np.array([0, 1, 0, 1]),
    )

Simulating observations
------------------------

With the design installed, the :class:`~micropurc.simulation.dgp.DGP` draws
trips at a known ``beta_true``. The Sioux Falls OD demand supplies the
origin-destination distribution.

.. code-block:: python

    from micropurc import PIQPFlowSolver, DGP, DGPConfig, MarkovSampler, od_from_csv
    from micropurc.config import SolverConfig

    solver = PIQPFlowSolver(
        net, scale_m,
        SolverConfig(backend="piqp", eps_abs=1e-10, eps_duality_gap_abs=1e-10),
    )
    od = od_from_csv(net, "data/siouxfalls_trips.csv", origin_col="origin",
                     dest_col="destination", weight_col="sampling_weight")

    beta_true = np.array([0.5, 0.4, 0.3])
    dgp = DGP(network=net, beta_true=beta_true, forward_solver=solver,
              route_sampler=MarkovSampler(net), od_dist=od,
              rng=np.random.default_rng(1), config=DGPConfig(max_tries_per_trip=30))
    data = dgp.sample_dataset(n_trips=3000)
    y, b, origins, dests = data["y"], data["b"], data["origins"], data["destinations"]

Selecting observations
----------------------

Observation filters are plain array operations on ``data``. Here we keep trips
whose observed route uses more than one link, since single-link trips carry no
routing information.

.. code-block:: python

    route_len = np.asarray((y > 0).sum(axis=1)).ravel()
    keep = route_len >= 2
    y, b, origins, dests = y[keep], b[keep], origins[keep], dests[keep]

Estimation
----------

:meth:`~micropurc.core.estimator.MicroPURCEstimator.fit` iterates the
fixed-point map to self-consistency and returns the estimates with convergence
diagnostics.

.. code-block:: python

    from micropurc import MicroPURCEstimator
    from micropurc.config import EstimationConfig

    est_cfg = EstimationConfig(max_iterations=200, fp_tol=1e-3, nm_window=3, nm_c=1e-4)
    est = MicroPURCEstimator(network=net, forward_solver=solver, estimation_config=est_cfg)
    result = est.fit(y=y, b=b, beta_init=np.zeros(3))
    # beta_true = [0.5, 0.4, 0.3]  ->  beta_hat = [0.477, 0.418, 0.303]

The recovered coefficients sit within a few percent of ``beta_true`` at
``n_trips=3000`` -- the simulate-then-estimate round trip closes.

Archiving the run
-----------------

:class:`~micropurc.reporting.RunWriter` snapshots the config, the data, and the
estimates into an auto-numbered directory, with an HTML report and the compute
profile (see :doc:`reporting` for the full archive contents).

.. code-block:: python

    import scipy.sparse as sp
    from micropurc.reporting import RunWriter

    writer = RunWriter("runs", "sioux_synthetic",
                       config={"estimation": est_cfg, "beta_true": beta_true.tolist(), "seed": 1})
    writer.snapshot_data(edges=net.edges, Z=net.Z, A=sp.csc_matrix(net.A), scale_m=scale_m,
                         y=sp.csr_matrix(y), origins=origins, dests=dests)
    run_dir = writer.finalize(result={
        "beta_hat": result["beta_hat"], "se_hat": result.get("se_hat"),
        "converged": result["diagnostics"]["converged"],
        "Q_history": result["diagnostics"].get("Q_history"),
    })

Reloading and analyzing
-----------------------

:func:`~micropurc.reporting.load_run` reopens the archive. The reloaded run
forward-solves at ``beta_hat`` for any OD pair and exposes the active
subnetwork under the same dual-slack rule the estimator uses.

.. code-block:: python

    from micropurc.reporting import load_run, render_fit_report

    run = load_run(run_dir)
    run.beta_hat                                  # matches the fitted estimate
    od_pair = (int(origins[0]), int(dests[0]))
    flows = run.predicted_flow(od_pairs=[od_pair])
    active = run.active_subnetwork(od_pair)       # links carrying flow for this OD
    render_fit_report(run, param_names=["b_time", "b_toll", "b_scenic"])

Running the script end to end prints the recovered coefficients and writes the
report to ``runs/sioux_synthetic/report.html``.
