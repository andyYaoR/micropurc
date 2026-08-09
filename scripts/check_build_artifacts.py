#!/usr/bin/env python3
"""Pre-commit hook: block build artifacts and misplaced native sources.

Enforces two rules:

1. No compiled artifacts or CMake build state are committed anywhere under the
   package (``*.so`` / ``*.dylib`` / ``*.pyd``, ``CMakeCache.txt``,
   ``CMakeFiles/``, ``build/``, ...).
2. The native source directory ``src/micropurc/_native`` holds only C/C++
   sources (``*.cpp`` / ``*.hpp`` / ``*.h`` / ``*.cu`` / ``*.cuh``).

Exit code 0 means the tree is clean; 1 means violations were found.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
violations: list[str] = []

ARTIFACT_SUFFIXES = {".so", ".dylib", ".pyd", ".o", ".a"}
CMAKE_ARTIFACT_NAMES = {
    "CMakeCache.txt",
    "cmake_install.cmake",
    "build.ninja",
    ".ninja_deps",
    ".ninja_log",
    ".skbuild-info.json",
    "libnanobind-static.a",
}
NATIVE_SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cu", ".cuh"}

# 1. No compiled or CMake build artifacts committed under src/.
src = ROOT / "src"
if src.exists():
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(ROOT)
            if f.suffix in ARTIFACT_SUFFIXES:
                violations.append(f"{rel}: build artifact must not be committed")
            elif f.name in CMAKE_ARTIFACT_NAMES:
                violations.append(f"{rel}: CMake artifact must not be committed")
    for d in src.rglob("*"):
        if d.is_dir() and d.name in ("CMakeFiles", ".cmake", "build"):
            violations.append(f"{d.relative_to(ROOT)}/: CMake build dir must not be committed")

# 2. The native source directory holds only C/C++ sources.
native_src = ROOT / "src" / "micropurc" / "_native"
if native_src.exists():
    for f in native_src.iterdir():
        if f.is_file() and f.suffix not in NATIVE_SOURCE_SUFFIXES:
            violations.append(f"{f.relative_to(ROOT)}: only C/C++ sources belong in _native/")

if violations:
    print("Build-artifact / native-source violations found:")
    for v in violations:
        print(f"  - {v}")
    sys.exit(1)

print("Build-artifact guard: clean.")
sys.exit(0)
