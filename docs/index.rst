micropurc
=========

``micropurc`` estimates the perturbed utility route choice (microPURC) model
from trip-level data. It is data-agnostic: a network, an origin-destination
demand, and a linear-in-parameters utility declared with the model-spec DSL are
all that is needed to simulate data or estimate parameters. It accompanies
Fosgerau, Nielsen, Rasmussen, and Yao, *Estimating the perturbed utility route
choice model with trip-level data*.

The package is organized into subpackages:

- :mod:`micropurc.core` -- the estimator and its numerical machinery;
- :mod:`micropurc.forward` -- forward solvers behind the
  :class:`~micropurc.forward.base.ForwardSolver` protocol (PIQP by default);
- :mod:`micropurc.simulation` -- a data-generating process and route sampler;
- :mod:`micropurc.spec` -- a schema and Python DSL for specifying utilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   examples/index
   api/index

Acknowledgments
---------------

The forward route-choice quadratic programs are solved with `PIQP
<https://github.com/PREDICT-EPFL/piqp>`_ (Schwan et al., 2023), whose C++
sources are vendored and compiled into micropurc's native extension under its
BSD-2-Clause license.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
