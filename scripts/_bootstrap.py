"""Make ``football_predictor`` importable when running scripts without install.

Each script imports this first. It prepends ``../src`` to ``sys.path`` so the
package resolves whether or not the project was ``pip install``-ed.
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
