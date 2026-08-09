"""Vercel entrypoint. The Python runtime looks for a top-level ASGI `app`
object in files under /api — this just re-exports the real FastAPI app so
there is exactly one place (app/api.py) that defines routes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import app  # noqa: E402,F401
