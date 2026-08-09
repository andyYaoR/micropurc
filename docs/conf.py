"""Sphinx configuration for the micropurc documentation."""

import sys
import tomllib  # requires Python >= 3.11; the docs build must run on 3.11+
from pathlib import Path

# Import the package from source; no build or install is required because the
# compiled extension is mocked below.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# -- Project information -----------------------------------------------------

project = "micropurc"
author = "Rui Yao"
copyright = "2026, Rui Yao"
# Single source of truth for the version is pyproject.toml. It is read
# directly (instead of importlib.metadata) because the docs build imports the
# package from source without installing it.
with (project_root / "pyproject.toml").open("rb") as _f:
    release = tomllib.load(_f)["project"]["version"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon (Google-style docstrings).
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
# Render dataclass "Attributes:" sections as inline info fields so they do not
# collide with autodoc's own attribute entries.
napoleon_use_ivar = True

# Autodoc.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# The compiled extension (and the PIQP backend) are not importable in the docs
# build environment; mock them so autodoc can import the pure-Python modules.
autodoc_mock_imports = ["micropurc._native", "piqp"]

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "micropurc"
