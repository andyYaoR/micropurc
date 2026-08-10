# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-08-10

### Changed

- `import micropurc` no longer imports `piqp`; the dependency loads when a
  `PIQPFlowSolver` is constructed, so a piqp extension that cannot load (for
  example against an outdated MSVC runtime) no longer prevents importing the
  package.
- `PIQPFlowSolver.solve` now runs through the PIQP C++ library vendored in
  `extern/piqp` and compiled into the native extension, which every other
  solve path already used. The problem, the settings taken from
  `SolverConfig`, and the `FlowResult` contract are unchanged; the single and
  batch entry points now produce identical flows at the default configuration.
- The `piqp` Python package is no longer a runtime dependency. It moved to the
  `dev` optional-dependency group, where the test suite uses it as an
  independent equivalence oracle for the compiled-in solver
  (`tests/test_forward_solver_native.py`, marker `oracle`).

### Added

- `micropurc._native.NativeSparseSolver`: a persistent single-right-hand-side
  forward QP solver, set up once per network and updated per solve, returning
  flows, equality duals, and PIQP's status, iteration count, residuals, and
  duality gap.

## [0.1.1] - 2026-08-10

### Fixed

- Windows wheels no longer crash with a silent access violation on import in
  environments whose `msvcp140.dll` is older than 14.40 (conda ships one):
  the extension links the MSVC runtime statically and loads no
  `msvcp140.dll` at all; source builds that keep the dynamic runtime compile
  with the pre-14.40 mutex-layout compatibility define.
- The end-to-end documentation example is self-contained after
  `pip install micropurc` (grid network, no data files) and every
  documentation code block is executed by the test suite.

## [0.1.0] - 2026-08-09

### Added

- First public release.
- Trip-level PURC estimator: a nested fixed-point solver with a non-monotone
  line search.
- Forward route-choice solver built on [PIQP](https://github.com/PREDICT-EPFL/piqp),
  behind a `ForwardSolver` protocol for additional backends.
- Simulation tooling: a data-generating process and route samplers for
  producing synthetic trip datasets at known parameters.
- Lazy design evaluation in C++ for memory-efficient handling of large
  networks and specifications.
- Sandwich and cluster-robust covariance estimators.
- Structured run reporting with frozen configuration, diagnostics, and
  reproducibility metadata.
