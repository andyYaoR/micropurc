# microPURC

A Python package for estimating the perturbed utility route choice (microPURC)
model from trip-level data. It accompanies the paper:

> M. Fosgerau, N. Nielsen, T. Rasmussen, and R. Yao.
> *Estimating the perturbed utility route choice model with trip-level data.*

📖 **Documentation:** https://andyyaor.github.io/micropurc_estimation/

## Installation

```bash
pip install micropurc
```

Pre-built wheels cover Linux (x86_64, aarch64), macOS (x86_64, arm64), and
Windows (AMD64) for Python 3.10 and newer; new stable Python releases are
picked up automatically.

### Building from source

A source build compiles a small C++ extension and needs a C++ compiler and
CMake (Eigen is fetched automatically when it is not found on the system):

- **macOS** — `xcode-select --install`, then `brew install cmake`
- **Linux** — `sudo apt install build-essential cmake` (or the equivalent)
- **Windows** — Visual Studio Build Tools (the "Desktop development with C++"
  workload) and CMake

```bash
pip install -e .
```

The development tools (tests, linting, docs) install via `./scripts/dev_setup.sh`.

## Quick start

```python
import numpy as np
from micropurc import (
    Network, PIQPFlowSolver, DGP, DGPConfig, MarkovSampler,
    od_uniform_all_pairs, MicroPURCEstimator,
)

# A 5x5 grid; its links carry three standardized synthetic attributes.
net = Network.grid(rows=5, cols=5, K=3)
net.attribute_names = ["length", "time", "toll"]
scale_m = np.ones(net.num_links)          # quadratic-perturbation scale
solver = PIQPFlowSolver(net, scale_m)

# Simulate trips at a known beta, then recover it.
dgp = DGP(network=net, beta_true=np.array([1.0, 0.5, 0.8]),
          forward_solver=solver, route_sampler=MarkovSampler(net),
          od_dist=od_uniform_all_pairs(net), rng=np.random.default_rng(1),
          config=DGPConfig())
data = dgp.sample_dataset(n_trips=20000)

est = MicroPURCEstimator(net, solver)
result = est.fit(y=data["y"], b=data["b"], beta_init=np.zeros(3))
print(result["beta_hat"], result["diagnostics"]["converged"])
# -> [0.993 0.498 0.798] True   (recovers beta_true; N=20000 -> ~0.01 sampling error)
```

## Documentation and examples

Full documentation is hosted at
**https://andyyaor.github.io/micropurc_estimation/**:

- [Getting started](https://andyyaor.github.io/micropurc_estimation/getting_started.html)
- [Examples](https://andyyaor.github.io/micropurc_estimation/examples/index.html) —
  end-to-end pipeline, model specification, simulation, and run reporting
- [API reference](https://andyyaor.github.io/micropurc_estimation/api/index.html)

Also in this repo: `scripts/paper/` holds the paper's simulation experiments
(with the benchmark network data bundled), `pytest` runs the test suite, and
`make -C docs html` builds the docs locally.

## Citing

If you use microPURC, please cite the paper above; citation metadata is in
[`CITATION.cff`](CITATION.cff). The forward route-choice quadratic programs are
solved with [PIQP](https://github.com/PREDICT-EPFL/piqp) (Schwan et al., 2023),
which the pre-built wheels bundle under its BSD-2-Clause license.
