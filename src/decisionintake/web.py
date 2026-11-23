"""DecisionIntake as a hosted service: a JSON API, a remote MCP endpoint, and the public screener.

    GET  /healthz          liveness, version, which engine is loaded
    GET  /v1/questions     the programmes, every question, and the starting state
    POST /v1/step          {"state"?, "facts"?} -> {"state", "readings", "next"}
    POST /v1/assessment    {"state"} -> every programme's probability, tier and value range
    GET  /openapi.json     this API, described
    POST /mcp              the Model Context Protocol over Streamable HTTP (2025-06-18, 2025-03-26)
    GET  /...              the screener's static files (local runs; on Vercel the CDN serves them)

Nothing is kept between calls. The facts travel in `state` (state.py) and each call rebuilds the belief
from it in about 0.1 s, so any replica can answer any call and the service scales to zero. Logs record
the route, status and timing, never an answer or a state. Standard library and numpy only: one WSGI
callable that runs on Vercel's Python runtime, under gunicorn, or with `python -m decisionintake.web`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs

from . import bundle as bundle_mod
from .mcp_server import ABSTAIN_BELOW, DEFAULT_BUNDLE, PROTOCOLS, VERSION, StatelessServer
from .guards import DEFAULT_RULES, attach
from .service import FactError, Intake, Settings
from .state import StateError

MAX_BODY = 64 * 1024
MAX_FACTS = 40
ROOT = Path(__file__).resolve().parents[2]
STATUS = {200: "200 OK", 202: "202 Accepted", 204: "204 No Content", 400: "400 Bad Request", 403: "403 Forbidden",
          404: "404 Not Found", 405: "405 Method Not Allowed", 413: "413 Payload Too Large",
          415: "415 Unsupported Media Type", 500: "500 Internal Server Error"}
CORS = [("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Accept, Mcp-Session-Id, MCP-Protocol-Version"),
        ("Access-Control-Max-Age", "86400")]
SECURITY = [("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer")]


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str, headers=()):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, list(headers)


def _json(status: int, obj, headers=()) -> tuple[int, list, bytes]:
    return status, [("Content-Type", "application/json; charset=utf-8"), *headers], json.dumps(obj).encode()


class App:
    """The WSGI application. One instance serves every route; it holds the engine and nothing else."""

    def __init__(self, bundle_path=None, settings: Settings | None = None, static_dir=None,
                 allowed_origins: list[str] | None = None, log=sys.stderr, rules=DEFAULT_RULES):
        t0 = time.perf_counter()
        self.b = bundle_mod.load(bundle_path or DEFAULT_BUNDLE)
        if rules:
            attach(self.b, rules)
        self.settings = settings or Settings(abstain_below=ABSTAIN_BELOW)
        Intake(self.b)  # warm the model's cache of initial scores before the first call
        self.mcp = StatelessServer(self.b, self.settings)
        self.static = Path(static_dir).resolve() if static_dir else None
        # Browsers send Origin; the MCP spec asks servers to check it against DNS rebinding. This service
        # holds no data and touches nothing local, so by default any origin may call it; set
        # DI_ALLOWED_ORIGINS (comma-separated) to restrict browser callers anyway.
        self.origins = allowed_origins
        self.log_to = log
        self.boot_s = time.perf_counter() - t0
        self.started = time.time()

    # -------------------------------------------------------------------------------- plumbing
    def __call__(self, environ, start_response):
        t0 = time.perf_counter()
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO") or "/"
        if path.startswith("/api/"):  # Vercel rewrites land on the function; the route rides along
            path = "/" + parse_qs(environ.get("QUERY_STRING", "")).get("route", [""])[0].lstrip("/")
        log = {"method": method, "path": path}
        try:
            status, headers, body = self.route(method, path, environ, log)
        except HTTPError as e:
            status, headers, body = _json(e.status, {"error": {"code": e.code, "message": e.message}}, e.headers)
            log["error"] = e.code
        except Exception as e:  # never leak a traceback to the caller; the log keeps the type
            status, headers, body = _json(500, {"error": {"code": "internal", "message": "internal error"}})
            log["error"] = type(e).__name__
        if path.startswith(("/v1/", "/mcp", "/healthz", "/openapi.json")):
            headers += CORS + [("Cache-Control", "no-store")]
        headers += SECURITY + [("Content-Length", str(len(body)))]
        log.update(status=status, ms=round((time.perf_counter() - t0) * 1000, 1))
        if self.log_to is not None:
            print(json.dumps(log), file=self.log_to, flush=True)
        start_response(STATUS[status], headers)
        return [body] if method != "HEAD" else [b""]

    def _body(self, environ) -> object:
        ctype = (environ.get("CONTENT_TYPE") or "").split(";")[0].strip().lower()
        if ctype and ctype != "application/json":
            raise HTTPError(415, "unsupported_media_type", "send JSON (Content-Type: application/json)")
        try:
            n = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            n = 0
        if n > MAX_BODY:
            raise HTTPError(413, "too_large", f"request body is over {MAX_BODY // 1024} KB")
        raw = environ["wsgi.input"].read(n) if n else b""
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPError(400, "bad_json", "the request body is not valid JSON") from None

    def _intake(self, state) -> Intake:
        if state in (None, ""):
            return Intake(self.b, self.settings)
        try:
            return Intake.from_state(self.b, state, self.settings)
        except (StateError, FactError) as e:
            raise HTTPError(400, "bad_state", str(e)) from None

    # -------------------------------------------------------------------------------- routes
    def route(self, method, path, environ, log):
        if method == "OPTIONS":
            return 204, [], b""
        if path == "/mcp":
            if method != "POST":
                raise HTTPError(405, "method_not_allowed", "this server answers MCP over POST only (no SSE stream)",
                                [("Allow", "POST, OPTIONS")])
            return self.mcp_post(environ, log)
        if path.startswith("/v1/") or path in ("/healthz", "/openapi.json"):
            handler = {("GET", "/healthz"): self.health, ("HEAD", "/healthz"): self.health,
                       ("GET", "/openapi.json"): self.openapi, ("GET", "/v1/questions"): self.questions,
                       ("POST", "/v1/step"): self.step, ("POST", "/v1/assessment"): self.assess}.get((method, path))
            if handler is None:
                known = {"/healthz", "/openapi.json", "/v1/questions", "/v1/step", "/v1/assessment"}
                if path in known:
                    raise HTTPError(405, "method_not_allowed", f"{method} is not supported on {path}")
                raise HTTPError(404, "not_found", f"no route {path}; see /openapi.json")
            return handler(environ, log)
        if method in ("GET", "HEAD") and self.static is not None:
            return self.static_file(path)
        raise HTTPError(404, "not_found", f"no route {path}")

    def health(self, environ, log):
        return _json(200, {"ok": True, "version": VERSION, "engine": self.b.info.get("name"),
                           "rules": self.b.info.get("rules"),
                           "households": len(self.b.pop), "questions": len(self.b.pop.questions),
                           "programmes": len(self.b.pop.decisions), "boot_seconds": round(self.boot_s, 2),
                           "uptime_seconds": round(time.time() - self.started)})

    def questions(self, environ, log):
        it = Intake(self.b, self.settings)
        about = {k: self.b.info.get(k) for k in ("name", "data", "model", "rules", "loss", "caveat")}
        return _json(200, {"version": VERSION, "about": about, "state": it.state(),
                           "programmes": [{"id": k, "label": v} for k, v in self.b.labels.items()],
                           "questions": it.catalogue()})

    def step(self, environ, log):
        body = self._body(environ) or {}
        if not isinstance(body, dict):
            raise HTTPError(400, "bad_request", 'send an object: {"state": "...", "facts": [...]}')
        facts = body.get("facts", [])
        if not isinstance(facts, list):
            raise HTTPError(400, "bad_request", "facts must be a list")
        if len(facts) > MAX_FACTS:
            raise HTTPError(400, "bad_request", f"at most {MAX_FACTS} facts in one call")
        it = self._intake(body.get("state"))
        readings = it.record(facts) if facts else []
        nxt = it.next_step()
        log.update(facts=len(it.facts), action=nxt["action"])
        return _json(200, {"state": it.state(), "readings": readings, "next": nxt})

    def assess(self, environ, log):
        body = self._body(environ) or {}
        if not isinstance(body, dict):
            raise HTTPError(400, "bad_request", 'send an object: {"state": "..."}')
        it = self._intake(body.get("state"))
        log.update(facts=len(it.facts))
        return _json(200, {"state": it.state(), **it.assessment()})

    def mcp_post(self, environ, log):
        origin = environ.get("HTTP_ORIGIN")
        if origin and self.origins is not None and origin not in self.origins:
            raise HTTPError(403, "forbidden_origin", f"origin {origin} may not call this server")
        version = environ.get("HTTP_MCP_PROTOCOL_VERSION")
        log["protocol"] = version
        if version and version not in PROTOCOLS and not self._is_initialize(environ):
            raise HTTPError(400, "unsupported_protocol_version",
                            f"MCP-Protocol-Version {version} is not supported; use one of {', '.join(PROTOCOLS)}")
        try:
            msg = environ.get("decisionintake.body") or self._body(environ)
        except HTTPError as e:
            if e.code != "bad_json":
                raise
            return _json(200, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
        msgs = msg if isinstance(msg, list) else [msg]
        replies = []
        for m in msgs:
            if not isinstance(m, dict) or m.get("jsonrpc") != "2.0":
                replies.append({"jsonrpc": "2.0", "id": m.get("id") if isinstance(m, dict) else None,
                                "error": {"code": -32600, "message": "invalid request"}})
                continue
            if m.get("method") == "tools/call":
                log["tool"] = (m.get("params") or {}).get("name")
            elif m.get("method"):
                log["rpc"] = m.get("method")
            r = self.mcp.handle(m)
            if r is not None:
                replies.append(r)
        if not replies:  # only notifications or responses: accepted, nothing to say
            return 202, [], b""
        return _json(200, replies if isinstance(msg, list) else replies[0])

    def _is_initialize(self, environ) -> bool:
        """Version negotiation happens inside initialize, so its header can't be held against it."""
        try:
            msg = self._body(environ)
        except HTTPError:
            return False
        environ["decisionintake.body"] = msg
        return isinstance(msg, dict) and msg.get("method") == "initialize"

    def static_file(self, path):
        rel = path.lstrip("/") or "index.html"
        f = (self.static / rel).resolve()
        if f.is_dir():
            f = f / "index.html"
        if not f.is_relative_to(self.static) or not f.is_file():
            raise HTTPError(404, "not_found", f"no file {path}")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        return 200, [("Content-Type", ctype), ("Cache-Control", "no-cache")], f.read_bytes()

    # -------------------------------------------------------------------------------- the API, described
    def openapi(self, environ, log):
        return _json(200, openapi_doc())


def openapi_doc() -> dict:
    fact = {"type": "object", "required": ["question"], "properties": {
        "question": {"type": "string", "description": "a question id from /v1/questions"},
        "choice": {"type": ["string", "number", "boolean"], "description": "the option's text, e.g. \"yes\", \"3\", \"rent\", \"CA\""},
        "amount": {"type": "object", "properties": {
            "value": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
            "period": {"type": "string", "enum": ["year", "month", "week", "none"]},
            "approximate": {"type": "boolean"}}},
        "dont_know": {"type": "boolean"}}}
    state = {"type": "string", "description": "the facts so far, from the previous response (state.py); omit to start",
             "example": "di1;state=TX;household_size=3;#1a2b"}
    err = {"description": "error", "content": {"application/json": {"schema": {"type": "object", "properties": {
        "error": {"type": "object", "properties": {"code": {"type": "string"}, "message": {"type": "string"}}}}}}}}
    ok = {"description": "ok", "content": {"application/json": {"schema": {"type": "object"}}}}
    return {
        "openapi": "3.1.0",
        "info": {"title": "DecisionIntake", "version": VERSION,
                 "description": "Decides what to ask next in a US benefits screening, how far to trust each answer, "
                                "and when it knows enough to recommend or should hand off to a person. Stateless: "
                                "every response carries `state`; send it back with the next call."},
        "paths": {
            "/v1/questions": {"get": {"summary": "The programmes, every question, and the starting state",
                                      "responses": {"200": ok}}},
            "/v1/step": {"post": {
                "summary": "Record facts (any questions, asked or not) and get the next step: ask, finish or hand_off",
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
                    "state": state, "facts": {"type": "array", "maxItems": MAX_FACTS, "items": fact}}}}}},
                "responses": {"200": ok, "400": err, "413": err}}},
            "/v1/assessment": {"post": {
                "summary": "Every programme's probability, tier and yearly value range; the facts so far",
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {
                    "state": state}}}}},
                "responses": {"200": ok, "400": err}}},
            "/mcp": {"post": {"summary": "Model Context Protocol (Streamable HTTP, JSON responses, stateless): "
                                         "tools start_intake, record_facts, next_step, get_assessment",
                              "responses": {"200": ok, "202": {"description": "notification accepted"}}}},
            "/healthz": {"get": {"summary": "Liveness and the engine loaded", "responses": {"200": ok}}},
        },
    }


def create_app(**kw) -> App:
    origins = os.environ.get("DI_ALLOWED_ORIGINS")
    kw.setdefault("allowed_origins", [o.strip() for o in origins.split(",")] if origins else None)
    return App(**kw)


def main(argv=None):
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

    class Server(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):  # the app logs one JSON line per request itself
            pass

    ap = argparse.ArgumentParser(description="Run the DecisionIntake service locally")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--static", default=str(ROOT / "public"), help="the screener's files ('' for API only)")
    a = ap.parse_args(argv)
    app = create_app(bundle_path=a.bundle, static_dir=a.static or None)
    print(f"decisionintake {VERSION}: {len(app.b.pop):,} households, ready in {app.boot_s:.1f}s on "
          f"http://{a.host}:{a.port}", file=sys.stderr, flush=True)
    with make_server(a.host, a.port, app, server_class=Server, handler_class=Quiet) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
