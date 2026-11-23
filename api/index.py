"""Vercel's entry point: the whole service is one WSGI app (src/decisionintake/web.py).

vercel.json rewrites /v1/*, /mcp, /healthz and /openapi.json here; the screener in public/ is served by
the CDN. The engine loads once per instance (about 1 s cold) and keeps nothing between requests.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from decisionintake.web import create_app  # noqa: E402

app = create_app(static_dir=None)
