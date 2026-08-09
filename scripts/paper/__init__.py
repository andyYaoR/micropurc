"""Section scripts for the simulation study.

Importing this package puts ``src/`` on sys.path, so the setup and section
modules can import ``micropurc`` regardless of import order.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
