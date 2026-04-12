"""Shared pytest configuration and fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on sys.path so `from shared.contracts import ...` works
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
