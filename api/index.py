"""Vercel entry point.

vercel.json routes every path here, so this one ASGI app serves both the GUI at
/ and the four required endpoints under /api/.
"""

import os
import sys

# Vercel executes this file from the api/ directory; the project root has to be
# importable for `atex` to resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atex.app import app  # noqa: E402

__all__ = ["app"]
