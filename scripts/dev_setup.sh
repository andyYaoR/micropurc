#!/usr/bin/env bash
# Editable-install micropurc into the ACTIVE conda env and register pre-commit hooks.
#
# Prerequisite: create and activate the build env first (installs Eigen + the
# native build toolchain that this step relies on):
#     conda env create -f environment.yml    # one-time
#     conda activate micropurc
#
# Two non-obvious requirements, both discovered rebuilding from scratch (2026-07-12):
#   1. --no-build-isolation: the editable build must compile the native
#      extension with THIS env's build tools (environment.yml installs them),
#      keeping the build toolchain identical to the runtime env. pyproject sets
#      editable.rebuild=false, so after C++/CMake changes rerun
#      `pip install -e .` manually.
#   2. torch is installed CPU-only: it is used solely as an optional numerical
#      oracle in the test suite (torch.linalg.pinv cross-check), which is a CPU
#      operation, so the CUDA build would be pure bloat.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Editable-installing micropurc with the dev extra (no build isolation)..."
pip install -e ".[dev]" --no-build-isolation

echo "Installing CPU torch (optional pinv oracle for the test suite; no CUDA needed)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing pre-commit hooks..."
pre-commit install

echo "Done. Sanity check:"
python -c "import micropurc, micropurc._native; print('micropurc', micropurc.__version__, 'OK')"
