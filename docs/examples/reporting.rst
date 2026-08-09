Archiving and analyzing runs
============================

The :mod:`micropurc.reporting` package archives every estimation run into its
own auto-numbered directory — ``runs/mymodel``, ``runs/mymodel~00``,
``runs/mymodel~01``, ... — so results are never overwritten (the naming
convention follows biogeme). Each archive holds the full configuration, a
dataset snapshot, machine/compute info, and an HTML report: everything needed
to know *which data and config produced these numbers* and to reproduce them.

Archiving a run
---------------

The snippets on this page continue from :doc:`pipeline`: ``net``, ``scale_m``,
``y``, ``origins``, ``dests``, ``est_cfg``, and the fitted ``result`` are the
objects built there.

.. code-block:: python

    import scipy.sparse as sp
    from micropurc.config import ActiveSetConfig, ProjectionConfig
    from micropurc.reporting import RunWriter

    writer = RunWriter(
        "runs", "sim_grid",
        config={
            "estimation": est_cfg,             # dataclasses serialize automatically
            "active_set": ActiveSetConfig(),
            "projection": ProjectionConfig(),
            "seed": 1,
        },
    )

    # Snapshot the inputs. Small inputs are copied in full (compressed);
    # above ``max_full_bytes`` they are stored as SHA-256 references so a
    # future load can verify it reads identical data.
    writer.snapshot_data(
        edges=net.edges, Z=net.Z, A=sp.csc_matrix(net.A), scale_m=scale_m,
        y=sp.csr_matrix(y), origins=origins, dests=dests,
    )

    writer.finalize(result={
        "beta_hat": result["beta_hat"],
        "se_hat": result.get("se_hat"),
        "converged": result["diagnostics"]["converged"],
        "Q_history": result["diagnostics"].get("Q_history"),
    })

``finalize`` also records the compute profile: wall time, CPU time, peak
resident memory, and core count. The archive's ``environment.json`` captures
the host, platform, package versions, and the git commit of the code.

Loading a run and analyzing the estimates
-----------------------------------------

.. code-block:: python

    from micropurc.reporting import load_run, render_fit_report

    run = load_run("runs/sim_grid")
    run.beta_hat                      # point estimates
    run.config["estimation"]          # exact config that produced them

    # Forward-solve at beta_hat for selected OD pairs (or trips):
    od_pair = (int(origins[0]), int(dests[0]))
    flows = run.predicted_flow(od_pairs=[od_pair])
    flows = run.predicted_flow(trips=range(10))

    # The links that carry flow for one OD pair, identified with the same
    # dual-slack KKT rule the estimator uses:
    sub = run.active_subnetwork(od_pair)
    list(zip(sub["tail"], sub["head"], sub["flow"]))

    # Predicted flow pattern vs the observed routes:
    rows = run.compare_routes(trips=range(100))
    rows[0]["flow_share_on_route"]    # 1.0 = all predicted mass on the route

    # Human-readable report (parameters, convergence, provenance, compute):
    render_fit_report(run, param_names=["b_time", "b_toll", "b_scenic"])

Batch studies
-------------

For Monte Carlo sweeps, write one record per replication and aggregate at the
end; :func:`~micropurc.reporting.html.render_batch_report` renders the summary
tables and pass/fail gates. Replace the body of the loop with one full
simulate-then-estimate replication; ``record_exists`` makes an interrupted
sweep resumable:

.. code-block:: python

    batch = RunWriter("runs", "mc_study", config={"n_reps": 3})
    for rep in range(3):                   # hundreds, in a real study
        if batch.record_exists(f"rep{rep}"):
            continue                       # resumable by construction
        batch.add_record(f"rep{rep}", beta_hat=result["beta_hat"], seed=rep)
    batch.finalize()
