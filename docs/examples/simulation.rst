Simulating data
===============

The :mod:`micropurc.simulation` package draws synthetic trips from the
microPURC model at a known parameter vector, which is useful for testing
estimator recovery. The
:class:`~micropurc.simulation.dgp.DGP` combines a network, a forward solver, a
route sampler, and an origin-destination demand.

.. code-block:: python

    import numpy as np
    from micropurc import (
        Network, PIQPFlowSolver, DGP, DGPConfig,
        MarkovSampler, od_uniform_all_pairs,
    )

    net = Network.grid(rows=4, cols=4, K=3)
    net.attribute_names = ["length", "time", "toll"]

    rng = np.random.default_rng(0)
    beta_true = np.array([1.0, 0.5, 0.8])
    # The quadratic-perturbation scale must be strictly positive on every link;
    # any positive per-link vector works.
    scale_m = rng.uniform(0.5, 1.5, net.num_links)
    solver = PIQPFlowSolver(net, scale_m)

    dgp = DGP(
        network=net,
        beta_true=beta_true,
        forward_solver=solver,
        route_sampler=MarkovSampler(net),
        od_dist=od_uniform_all_pairs(net),
        rng=rng,
        config=DGPConfig(),
    )
    data = dgp.sample_dataset(n_trips=500)

``data`` holds the sampled route indicators, the design ``Z``, node imbalances,
and the origin/destination arrays -- exactly the inputs
:meth:`~micropurc.core.estimator.MicroPURCEstimator.fit` expects, so a recovery
experiment is a simulate-then-estimate round trip.
