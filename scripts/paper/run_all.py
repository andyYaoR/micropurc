"""Run the section scripts one by one as isolated subprocesses.

Each section runs in its own ``python -m scripts.paper.nbXX.sYY`` process, so a
crash in one section can never abort another -- the runner records the failure
and moves on. Tables/figures land under ``results/`` as each section finishes.

    python -m scripts.paper.run_all                    # every section, full scale
    python -m scripts.paper.run_all --demo             # quick demo scale
    python -m scripts.paper.run_all nb01               # only nb01's sections
    python -m scripts.paper.run_all nb01.s4_switching  # a single section
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # scripts/paper
ROOT = HERE.parents[1]


def discover(selectors):
    """Collect (key, module) pairs for the sections matching the given selectors."""
    out = []
    for nb in sorted(HERE.glob("nb*")):
        if not nb.is_dir():
            continue
        for s in sorted(nb.glob("s[0-9]*.py")):  # the sNN_* section scripts; skips setup.py
            key = f"{nb.name}.{s.stem}"
            if selectors and not any(key.startswith(sel) or nb.name == sel for sel in selectors):
                continue
            out.append((key, f"scripts.paper.{nb.name}.{s.stem}"))
    return out


def main():
    selectors = [a for a in sys.argv[1:] if not a.startswith("-")]
    demo = "--demo" in sys.argv
    sections = discover(selectors)
    if not sections:
        print("no sections matched:", selectors or "(all)")
        return
    if "--list" in sys.argv:  # enumerate the matched sections and stop
        print(f"{len(sections)} sections:")
        for k, _ in sections:
            print("   -", k)
        return
    print(f"running {len(sections)} sections ({'DEMO' if demo else 'FULL'} scale):")
    for k, _ in sections:
        print("   -", k)

    results = []
    for key, mod in sections:
        cmd = [sys.executable, "-u", "-m", mod] + (["--demo"] if demo else [])
        print(f"\n{'=' * 72}\n[run] {key}\n{'=' * 72}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        results.append((key, status, dt))
        print(f"[done] {key}: {status} in {dt / 60:.1f} min", flush=True)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for key, status, dt in results:
        print(f"  {status:12s} {key:30s} {dt / 60:7.1f} min")
    n_fail = sum(1 for _, s, _ in results if s != "OK")
    print(f"\n{len(results) - n_fail}/{len(results)} sections OK, {n_fail} failed")


if __name__ == "__main__":
    main()
