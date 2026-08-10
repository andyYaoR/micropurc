Getting started
===============

Installation
------------

::

    pip install micropurc

Pre-built wheels cover Linux (x86_64, aarch64), macOS (x86_64, arm64), and
Windows (AMD64) for Python 3.10 and newer; new stable Python releases are
picked up automatically.

Building from source
~~~~~~~~~~~~~~~~~~~~

``micropurc`` builds a small C++ extension via scikit-build-core, so a source
build needs a C++ compiler and CMake; Eigen is fetched automatically when it is
not found on the system. From a checkout::

    pip install -e .

For a development environment with the test, lint, and docs tools::

    ./scripts/dev_setup.sh

Requirements
------------

The runtime dependencies are NumPy, SciPy, pandas, networkx, and PIQP (the
default forward solver). Documentation and linting tools live in the ``docs``
and ``dev`` optional-dependency groups.

A first estimation
------------------

The workflow has three parts: build a :class:`~micropurc.network.Network`,
declare a utility, and run the :class:`~micropurc.core.estimator.MicroPURCEstimator`.

.. code-block:: python

    import numpy as np
    from micropurc import (
        Network, PIQPFlowSolver, DGP, DGPConfig, MarkovSampler,
        od_uniform_all_pairs, MicroPURCEstimator, beta, attr, compile_design,
    )

    # 1. A network with three synthetic link attributes.
    net = Network.grid(rows=5, cols=5, K=3)
    net.attribute_names = ["length", "time", "toll"]

    # 2. A utility over those attributes, declared with the model-spec DSL.
    spec = (beta("b_length") * attr("length")
            + beta("b_time") * attr("time")
            + beta("b_toll") * attr("toll")).to_spec()
    net.Z = compile_design(spec, net).Z          # install the compiled design

    # 3. The forward solver and the estimator.
    scale_m = np.ones(net.num_links)             # quadratic-perturbation scale
    solver = PIQPFlowSolver(net, scale_m)
    est = MicroPURCEstimator(net, solver)

    # Simulate data at a known beta and recover it.
    dgp = DGP(network=net, beta_true=np.array([1.0, 0.5, 0.8]),
              forward_solver=solver, route_sampler=MarkovSampler(net),
              od_dist=od_uniform_all_pairs(net), rng=np.random.default_rng(1),
              config=DGPConfig())
    data = dgp.sample_dataset(n_trips=20000)
    result = est.fit(y=data["y"], b=data["b"], beta_init=np.zeros(3))
    print(result["beta_hat"], result["diagnostics"]["converged"])
    # -> [0.993 0.498 0.798] True   (recovers beta_true; N=20000 -> ~0.01 sampling error)

See :doc:`examples/index` for complete, runnable examples of specifying a
model, simulating data, and estimating parameters.

Troubleshooting
---------------

**Silent crash on import (Windows, conda/Anaconda).** If ``import micropurc``
exits the interpreter with no traceback (PowerShell shows exit code
``-1073741819``), the environment has loaded an ``msvcp140.dll`` older than
14.40 -- conda environments ship their own copy of the MSVC runtime, which
shadows the system one, and extensions built by recent MSVC crash against it.
Update the runtime in the affected environment::

    conda install -c conda-forge "vc14_runtime>=14.40"

From 0.1.1 on, micropurc's own extension links the MSVC runtime statically
and loads no ``msvcp140.dll`` at all, so it is immune to the runtime
version. The interpreter can still crash the same way inside a
*dependency's* extension module -- observed with ``piqp``, whose wheels
link the runtime dynamically -- and the ``conda install`` above cures those
too. Bisect with ``python -c "import piqp"`` if unsure where the crash
lives.
