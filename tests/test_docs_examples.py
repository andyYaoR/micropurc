"""Execute every Python code block in the documentation.

Each documentation page must run against the installed package exactly as a
reader would type it. Pages are executed block by block in file order within
a shared namespace; ``reporting.rst`` states in prose that it continues from
``pipeline.rst``, so the two share one namespace. RuntimeWarnings are
promoted to errors so a snippet that silently produces ``inf`` or ``nan``
fails loudly.

These tests need the ``docs/`` tree and are skipped when it is absent (for
example when the test suite runs against an installed wheel without a source
checkout).
"""

import re
import textwrap
import warnings
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Files listed in one group execute sequentially in a shared namespace.
GROUPS = {
    "pipeline+reporting": ["examples/pipeline.rst", "examples/reporting.rst"],
    "simulation": ["examples/simulation.rst"],
    "model_spec": ["examples/model_spec.rst"],
    "getting_started": ["getting_started.rst"],
}

CODE_BLOCK = re.compile(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not DOCS.is_dir(), reason="docs/ not present"),
]


def test_pipeline_script_runs(tmp_path, monkeypatch):
    """The downloadable pipeline.py runs standalone from an empty directory."""
    import runpy

    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(DOCS / "examples" / "pipeline.py"), run_name="__main__")


@pytest.mark.parametrize("group", GROUPS, ids=list(GROUPS))
def test_doc_code_blocks_execute(group, tmp_path, monkeypatch):
    """Every code block in the group runs without errors or RuntimeWarnings."""
    # Run in a scratch cwd; the CSV-loading aside expects the Sioux Falls
    # files next to the script, as its prose instructs.
    for csv in (DOCS / "examples" / "data").glob("siouxfalls_*.csv"):
        (tmp_path / csv.name).write_bytes(csv.read_bytes())
    monkeypatch.chdir(tmp_path)

    namespace = {}
    for rel in GROUPS[group]:
        source = (DOCS / rel).read_text()
        blocks = [textwrap.dedent(m.group(1)) for m in CODE_BLOCK.finditer(source)]
        assert blocks, f"{rel} contains no python code blocks"
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for i, code in enumerate(blocks, 1):
                exec(compile(code, f"{rel}[block {i}]", "exec"), namespace)
