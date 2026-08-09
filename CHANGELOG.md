# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Windows wheels no longer crash with a silent access violation on import in
  environments whose `msvcp140.dll` is older than 14.40 (conda ships one):
  the extension is compiled with the pre-14.40 mutex-layout compatibility
  define and the wheel vendors the MSVC C++ runtime.
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
