"""Gate wheel builds on upstream support for each CPython minor.

A new stable CPython should get micropurc wheels only once every runtime
dependency ships wheels for it; a micropurc wheel whose dependencies must be
compiled from source is a worse user experience than no wheel. This script
asks PyPI which CPython minors every runtime dependency covers and emits a
CIBW_SKIP value excluding the rest. The CIBW_SKIP environment variable
replaces the [tool.cibuildwheel] skip option, so the base skips are included
here as well.

On any query failure the script fails open (no extra skips): cibuildwheel's
own test step then still catches an unsupported minor, loudly, at build time.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

import tomllib

MINORS = range(10, 26)
BASE_SKIP = ["*-musllinux*", "pp*", "cp3*t-*"]  # keep in sync with pyproject skip


def runtime_dependencies() -> list[str]:
    """Names of the runtime dependencies declared in pyproject.toml."""
    with open("pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    return [re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0] for d in deps]


def supported_minors(package: str) -> set[int]:
    """CPython minors for which any release of ``package`` ships a wheel."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as r:
            releases = json.load(r)["releases"]
    except Exception as exc:  # noqa: BLE001 -- fail open, per module docstring
        print(f"::warning::{package}: {exc}; assuming full support", file=sys.stderr)
        return set(MINORS)
    # A minor counts as supported when ANY release ships a wheel for it: users
    # on an older minor resolve an older dependency release, so judging by the
    # latest release alone would wrongly exclude minors the ecosystem still
    # serves. New minors stay excluded until some release actually ships them.
    supported: set[int] = set()
    for f in (f for files in releases.values() for f in files):
        name = f["filename"]
        if not name.endswith(".whl"):
            continue
        tags = name[:-4].split("-")[-3:-1]
        if len(tags) != 2:
            continue  # not a well-formed wheel name; fail open on this file
        py_tag, abi_tag = tags
        if py_tag.startswith("py3"):
            return set(MINORS)  # pure-Python wheel covers every minor
        if py_tag.startswith("cp3") and py_tag[3:].isdigit():
            v = int(py_tag[3:])
            if abi_tag == "abi3":
                supported.update(m for m in MINORS if m >= v)
            elif v in MINORS:
                supported.add(v)
    return supported


def main() -> None:
    """Emit CIBW_SKIP covering minors not supported by every dependency."""
    common = set(MINORS)
    for dep in runtime_dependencies():
        minors = supported_minors(dep)
        common &= minors
        print(f"{dep}: cp3{{{','.join(str(m) for m in sorted(minors))}}}")
    skips = BASE_SKIP + [f"cp3{m}-*" for m in sorted(set(MINORS) - common)]
    value = " ".join(skips)
    print(f"CIBW_SKIP={value}")
    if env := os.environ.get("GITHUB_ENV"):
        with open(env, "a") as f:
            f.write(f"CIBW_SKIP={value}\n")


if __name__ == "__main__":
    main()
