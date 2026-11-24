"""DecisionIntake as a hosted service: a JSON API, a remote MCP endpoint, and the public screener.

    GET  /healthz          liveness, version, which engine is loaded
    GET  /v1/questions     the programmes, every question, and the starting state
    POST /v1/step          {"state"?, "facts"?} -> {"state", "readings", "next"}
    POST /v1/assessment    {"state"} -> every programme's probability, tier and value range
    POST /v1/explain       {"state"} -> which single answer, one step different, would change each result
    POST /v1/cliff         {"state", "question"?} -> every option of an amount question, the rest kept: a cliff view
    GET  /openapi.json     this API, described
    POST /mcp              the Model Context Protocol over Streamable HTTP (2026-07-28, 2025-11-25, 2025-06-18,
                           2025-03-26)
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
from .mcp_server import ABSTAIN_BELOW, DEFAULT_BUNDLE, K_VERSION, MODERN, VERSION, StatelessServer
from .guards import DEFAULT_RULES, attach
from .service import FactError, Intake, Settings
from .state import StateError

MAX_BODY = 64 * 1024
MAX_FACTS = 40
ROOT = Path(__file__).resolve().parents[2]
STATUS = {200: "200 OK", 202: "202 Accepted", 204: "204 No Content", 400: "400 Bad Request", 403: "403 Forbidden",
          404: "404 Not Found", 405: "405 Method Not Allowed", 413: "413 Payload Too Large",
          415: "415 Unsupported Media Type", 500: "500 Internal Server Error"}
CORS = [("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Accept, Mcp-Session-Id, MCP-Protocol-Version, Mcp-Method, Mcp-Name"),
        ("Access-Control-Max-Age", "86400")]
SECURITY = [("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer")]


class HTTPError(Exception):
    def __init__(self, status: int, code: str, message: str, headers=()):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, list(headers)


def _header(environ, key):
    """A request header's value without the optional whitespace around it (RFC 9110 5.5)."""
    v = environ.get(key)
    return v.strip(" \t") if isinstance(v, str) else v


def _decode_header(value):
    """An Mcp-Name value, decoded from the =?base64?...?= form the 2026-07-28 transport allows."""
    if isinstance(value, str) and value.startswith("=?base64?") and value.endswith("?="):
        import base64
        try:
            return base64.b64decode(value[9:-2]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
    return value


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _hostname(value, url: bool = False) -> str | None:
    """The lower-case host name of a Host header ("localhost:8080") or, with url=True, of an Origin."""
    from urllib.parse import urlsplit
    if not value:
        return None
    try:
        return urlsplit(value if url else "//" + value).hostname
    except ValueError:
        return None


def _sha256(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rules_sha256(path) -> str:
    """The rules' identity, not their bytes: canonical JSON, so a Windows checkout (CRLF) and the Linux build
    (LF) report the same hash."""
    import hashlib
    canon = json.dumps(json.loads(Path(path).read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _base_url(environ) -> str | None:
    """This deployment's own origin, from the request (Vercel passes the public host and scheme along)."""
    host = environ.get("HTTP_X_FORWARDED_HOST") or environ.get("HTTP_HOST")
    if not host:
        return None
    scheme = environ.get("HTTP_X_FORWARDED_PROTO") or environ.get("wsgi.url_scheme") or "https"
    return f"{scheme.split(',')[0].strip()}://{host.split(',')[0].strip()}"


def _json(status: int, obj, headers=()) -> tuple[int, list, bytes]:
    return status, [("Content-Type", "application/json; charset=utf-8"), *headers], json.dumps(obj).encode()


class App:
    """The WSGI application. One instance serves every route; it holds the engine and nothing else."""

    def __init__(self, bundle_path=None, settings: Settings | None = None, static_dir=None,
                 allowed_origins: list[str] | None = None, log=sys.stderr, rules=DEFAULT_RULES,
                 allowed_hosts: list[str] | None = None):
        t0 = time.perf_counter()
        path = Path(bundle_path or DEFAULT_BUNDLE).resolve()
        self.b = bundle_mod.load(path)
        if rules:
            # the production bundle must get its rules: if the file doesn't fit it, refuse to start
            attach(self.b, rules, strict=True if path == Path(DEFAULT_BUNDLE).resolve() else None)
        self.provenance = {"commit": (os.environ.get("VERCEL_GIT_COMMIT_SHA") or "")[:7] or None,
                           "bundle_sha256": _sha256(path)[:12],
                           "rules_sha256": _rules_sha256(rules)[:12] if rules and self.b.info.get("rules") else None}
        self.settings = settings or Settings(abstain_below=ABSTAIN_BELOW)
        Intake(self.b)  # warm the model's cache of initial scores before the first call
        self.mcp = StatelessServer(self.b, self.settings)
        self.static = Path(static_dir).resolve() if static_dir else None
        # /mcp checks Origin, as the MCP transport requires (DNS rebinding). Server-side clients send none and
        # pass; browsers may call from this deployment's own address (from Vercel's environment, never from the
        # request, whose Host an attacker controls), from localhost (MCP Inspector and other dev tools) and from
        # DI_ALLOWED_ORIGINS (comma-separated; "*" accepts any). The JSON API under /v1 is open to any origin.
        self.origins = set(allowed_origins or ())
        for var in ("VERCEL_PROJECT_PRODUCTION_URL", "VERCEL_URL", "VERCEL_BRANCH_URL"):
            if os.environ.get(var):
                self.origins.add("https://" + os.environ[var])
        # A server bound to localhost also refuses any other Host, so a page on an attacker's domain that
        # resolves to 127.0.0.1 can't reach it at all. None (the hosted service) checks no Host.
        self.hosts = {h.strip().lower() for h in allowed_hosts} if allowed_hosts else None
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
        except Exception as e:  # never leak a traceback to the caller; the server log keeps it
            status, headers, body = _json(500, {"error": {"code": "internal", "message": "internal error"}})
            log["error"] = type(e).__name__
            import traceback
            traceback.print_exception(e, file=sys.stderr)
        origin = environ.get("HTTP_ORIGIN")
        if path == "/mcp":
            headers += [("Access-Control-Allow-Methods", "POST, OPTIONS"), *CORS[1:],
                        ("Cache-Control", "no-store"), ("Vary", "Origin")]
            if origin and self.origin_ok(origin, environ):
                headers.append(("Access-Control-Allow-Origin", origin))
        elif path.startswith(("/v1/", "/healthz", "/openapi.json")):
            headers += CORS + [("Cache-Control", "no-store"), ("Access-Control-Allow-Origin", "*")]
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

    def _intake(self, state, new=()) -> tuple[Intake, list]:
        """The intake for a state plus any new facts, rebuilt and re-scored in one batch."""
        try:
            return Intake.resume(self.b, state if state not in (None, "") else "", list(new), self.settings)
        except (StateError, FactError) as e:
            raise HTTPError(400, "bad_state", str(e)) from None

    # -------------------------------------------------------------------------------- routes
    def route(self, method, path, environ, log):
        if self.hosts is not None and _hostname(environ.get("HTTP_HOST")) not in self.hosts:
            raise HTTPError(403, "forbidden_host", "this server answers only to " + ", ".join(sorted(self.hosts)))
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
                       ("POST", "/v1/step"): self.step, ("POST", "/v1/assessment"): self.assess,
                       ("POST", "/v1/explain"): self.explain, ("POST", "/v1/cliff"): self.cliff}.get((method, path))
            if handler is None:
                known = {"/healthz", "/openapi.json", "/v1/questions", "/v1/step", "/v1/assessment", "/v1/explain",
                         "/v1/cliff"}
                if path in known:
                    raise HTTPError(405, "method_not_allowed", f"{method} is not supported on {path}")
                raise HTTPError(404, "not_found", f"no route {path}; see /openapi.json")
            return handler(environ, log)
        if method in ("GET", "HEAD") and self.static is not None:
            return self.static_file(path)
        raise HTTPError(404, "not_found", f"no route {path}")

    def health(self, environ, log):
        return _json(200, {"ok": True, "version": VERSION, "engine": self.b.info.get("name"),
                           "rules": self.b.info.get("rules"), **self.provenance,
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
        it, readings = self._intake(body.get("state"), facts)
        nxt = it.next_step()
        log.update(facts=len(it.facts), action=nxt["action"])
        return _json(200, {"state": it.state(), "readings": readings, "next": nxt})

    def assess(self, environ, log):
        body = self._body(environ) or {}
        if not isinstance(body, dict):
            raise HTTPError(400, "bad_request", 'send an object: {"state": "..."}')
        it, _ = self._intake(body.get("state"))
        log.update(facts=len(it.facts))
        return _json(200, {"state": it.state(), **it.assessment()})

    def explain(self, environ, log):
        body = self._body(environ) or {}
        if not isinstance(body, dict):
            raise HTTPError(400, "bad_request", 'send an object: {"state": "..."}')
        it, _ = self._intake(body.get("state"))
        out = it.explain()
        log.update(facts=len(it.facts), tried=out["answers_tried"])
        return _json(200, {"state": it.state(), **out})

    def cliff(self, environ, log):
        body = self._body(environ) or {}
        if not isinstance(body, dict):
            raise HTTPError(400, "bad_request", 'send an object: {"state": "...", "question"?: "earned_income"}')
        question = body.get("question", "earned_income")
        if not isinstance(question, str):
            raise HTTPError(400, "bad_request", "question must be a question id")
        it, _ = self._intake(body.get("state"))
        try:
            out = it.cliff(question)
        except FactError as e:
            raise HTTPError(400, "bad_request", str(e)) from None
        log.update(facts=len(it.facts), question=question)
        return _json(200, {"state": it.state(), **out})

    def origin_ok(self, origin: str, environ) -> bool:
        if "*" in self.origins or origin in self.origins:
            return True
        return _hostname(origin, url=True) in LOCAL_HOSTS

    def mcp_post(self, environ, log):
        origin = environ.get("HTTP_ORIGIN")
        if origin and not self.origin_ok(origin, environ):
            log["error"] = "forbidden_origin"
            return _json(403, {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32600, "message": f"Forbidden: browser origin {origin} may not call this server"}})
        version = _header(environ, "HTTP_MCP_PROTOCOL_VERSION")
        log["protocol"] = version
        try:
            msg = self._body(environ)
        except HTTPError as e:
            if e.code != "bad_json":
                raise
            return _json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
        params = msg.get("params") if isinstance(msg, dict) else None
        meta = params.get("_meta") if isinstance(params, dict) else None
        if version in MODERN or (isinstance(meta, dict) and K_VERSION in meta):
            return self.mcp_modern(environ, msg, version, meta, log)
        # Legacy (2025-11-25 and earlier): an initialize handshake, then requests that name their version in the
        # header. A version we don't speak gets the modern error body, which lists what we do speak.
        is_init = isinstance(msg, dict) and msg.get("method") == "initialize"
        if version and version not in self.mcp.PROTOCOLS and not is_init:
            mid = msg.get("id") if isinstance(msg, dict) else None
            return _json(400, {"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32022, "message": "Unsupported protocol version",
                "data": {"supported": self.mcp.supported, "requested": version}}})
        if isinstance(msg, list) and not msg:  # JSON-RPC: an empty batch is an invalid request
            return _json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "empty batch"}})
        msgs = msg if isinstance(msg, list) else [msg]
        replies = []
        for m in msgs:
            if not isinstance(m, dict) or m.get("jsonrpc") != "2.0":
                replies.append({"jsonrpc": "2.0", "id": m.get("id") if isinstance(m, dict) else None,
                                "error": {"code": -32600, "message": "invalid request"}})
                continue
            if m.get("method") == "tools/call" and isinstance(m.get("params"), dict):
                log["tool"] = m["params"].get("name")
            elif m.get("method"):
                log["rpc"] = m.get("method")
            r = self.mcp.handle(m)
            if r is not None:
                replies.append(r)
        if not replies:  # only notifications or responses: accepted, nothing to say
            return 202, [], b""
        return _json(200, replies if isinstance(msg, list) else replies[0])

    def mcp_modern(self, environ, msg, version, meta, log):
        """MCP 2026-07-28 over Streamable HTTP: one request per POST, its metadata mirrored in headers that must
        match the body (Mcp-Method always, Mcp-Name for tools/call), errors mapped to HTTP statuses."""
        mid = msg.get("id") if isinstance(msg, dict) else None
        log["era"] = "modern"

        def fail(status, code, message, data=None):
            e = {"code": code, "message": message}
            if data is not None:
                e["data"] = data
            return _json(status, {"jsonrpc": "2.0", "id": mid, "error": e})

        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return fail(400, -32600, "invalid request: one JSON-RPC request per POST")
        method = msg["method"]
        log["rpc"] = method
        body_version = meta.get(K_VERSION) if isinstance(meta, dict) else None
        if version is None:
            return fail(400, -32020, "Header mismatch: MCP-Protocol-Version header is missing")
        if body_version is not None and body_version != version:
            return fail(400, -32020, f"Header mismatch: MCP-Protocol-Version header value '{version}' does not "
                                     f"match body value '{body_version}'")
        header_method = _header(environ, "HTTP_MCP_METHOD")
        if header_method != method:
            return fail(400, -32020, f"Header mismatch: Mcp-Method header value '{header_method}' does not match "
                                     f"body value '{method}'")
        if method == "tools/call":
            p = msg.get("params")
            name = p.get("name") if isinstance(p, dict) else None
            log["tool"] = name
            header_name = _decode_header(_header(environ, "HTTP_MCP_NAME"))
            if header_name != name:
                return fail(400, -32020, f"Header mismatch: Mcp-Name header value '{header_name}' does not match "
                                         f"body value '{name}'")
        r = self.mcp.handle(msg, modern=True)
        if r is None:
            return 202, [], b""
        code = (r.get("error") or {}).get("code")
        status = 200 if code is None else 404 if code == -32601 else 500 if code == -32603 else 400
        return _json(status, r)

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
        return _json(200, openapi_doc(_base_url(environ)))


def openapi_doc(base: str | None = None) -> dict:
    """The JSON API and the MCP endpoint, described with full response schemas. Examples are real: the state
    example is produced by the same encoder the service uses, and tests/test_web.py checks it decodes."""
    from .state import encode
    example_state = encode([{"question": "state", "choice": "TX"}, {"question": "household_size", "choice": "3"},
                            {"question": "earned_income", "amount": {"low": 2400, "high": 2400, "period": "month",
                                                                     "approximate": True}}])

    def ref(name):
        return {"$ref": f"#/components/schemas/{name}"}

    def body(schema, example=None):
        c = {"schema": schema}
        if example is not None:
            c["example"] = example
        return {"required": True, "content": {"application/json": c}}

    def resp(desc, schema):
        return {"description": desc, "content": {"application/json": {"schema": schema}}}

    err = {"$ref": "#/components/responses/Error"}
    limited = {"$ref": "#/components/responses/RateLimited"}
    tiers = ["likely", "worth checking", "can't tell", "not recommended", "unlikely"]
    schemas = {
        "State": {"type": "string", "maxLength": 2000, "example": example_state,
                  "description": "Every fact recorded so far, readable, ending in a 16-bit checksum. Send the state "
                                 "from the most recent response with the next call; omit it (or send \"\") to start. "
                                 "The server keeps nothing between calls."},
        "Amount": {"type": "object", "additionalProperties": False, "properties": {
            "value": {"type": "number", "minimum": 0, "description": "the number as the person said it"},
            "low": {"type": "number", "minimum": 0}, "high": {"type": "number", "minimum": 0},
            "period": {"type": "string", "enum": ["year", "month", "week", "none"],
                       "description": "what the number is per; the engine converts to the question's unit"},
            "approximate": {"type": "boolean", "description": "true for \"about\", \"around\", ranges"}},
            "description": "Give value, or low and high. Do no arithmetic: send the number and its period."},
        "Fact": {"type": "object", "required": ["question"],
                 "description": "One fact the person stated: exactly one of choice, amount or dont_know. A fact for "
                                "an already answered question replaces it.",
                 "properties": {
                     "question": {"type": "string", "description": "a question id from /v1/questions"},
                     "choice": {"type": ["string", "number", "boolean"],
                                "description": "the option's text (\"yes\", \"3\", \"rent\", \"CA\"); on amount "
                                               "questions a range's text (\"$20,000 to $30,000\") also works"},
                     "amount": ref("Amount"),
                     "dont_know": {"type": "boolean", "const": True}}},
        "Reading": {"type": "object", "required": ["question"], "properties": {
            "question": {"type": "string"},
            "reading": {"type": "string", "description": "how the fact was understood, for the person to check"},
            "replaces": {"type": "string", "description": "the earlier reading this fact replaced"},
            "warning": {"type": "string", "description": "no household on record fits this with the earlier answers; "
                                                         "check it with the person"},
            "error": {"type": "string", "description": "the fact was not recorded, and how to fix it; the other "
                                                       "facts in the call were"}}},
        "Question": {"type": "object", "required": ["id", "text", "kind"], "properties": {
            "id": {"type": "string"}, "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["choice", "amount"]},
            "options": {"type": "array", "items": {"type": "string"}, "description": "choice questions"},
            "unit": {"type": "string", "description": "amount questions: \"dollars a year\", \"dollars a month\" "
                                                      "or \"dollars in total\""},
            "ranges": {"type": "array", "items": {"type": "string"}, "description": "amount questions: the brackets "
                                                                                    "the engine reasons in"}}},
        "Support": {"type": "object", "properties": {
            "households_like_this": {"type": "number", "description": "effective number of training households "
                                                                      "that fit the answers (a weighted count)"},
            "level": {"type": "string", "enum": ["supported", "thin"],
                      "description": "thin: fewer than 10; the model is extrapolating"},
            "conflicting_answers": {"type": "integer"}}},
        "Progress": {"type": "object", "properties": {
            "facts_recorded": {"type": "integer"},
            "expected_mistake_cost": {"type": "integer", "description": "dollars a year the remaining doubt is "
                                                                         "expected to cost in wrong recommendations"},
            "similar_households": {"type": "integer", "description": "effective number of households the belief "
                                                                     "rests on, after the support floor"},
            "support": ref("Support")}},
        "Programme": {"type": "object", "required": ["id", "programme", "tier", "probability"], "properties": {
            "id": {"type": "string", "example": "snap"}, "programme": {"type": "string", "example": "SNAP (food)"},
            "tier": {"type": "string", "enum": tiers,
                     "description": "likely / worth checking: recommended. can't tell: too few households like this "
                                    "to call a negative; never present it as unlikely"},
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
            "value_if_eligible_per_year": {"type": "array", "items": {"type": "integer"}, "minItems": 2,
                                           "maxItems": 2, "description": "[low, high] dollars a year if eligible "
                                                                         "(health coverage: its value, not cash)"}}},
        "Assessment": {"type": "object", "required": ["recommend", "unsure", "note"], "properties": {
            "recommend": {"type": "array", "items": ref("Programme"), "description": "worth applying for, "
                                                                                     "most likely first"},
            "unsure": {"type": "array", "items": ref("Programme"), "description": "always present; empty when the "
                                                                                    "data covers the person"},
            "note": {"type": "string", "description": "the caveat; show it with any result"}}},
        "Ask": {"type": "object", "required": ["action", "question", "why", "progress"], "properties": {
            "action": {"const": "ask"}, "question": ref("Question"), "progress": ref("Progress"),
            "why": {"type": "object", "properties": {
                "worth_per_year": {"type": "integer", "description": "dollars of expected mistakes it prevents"},
                "question_cost": {"type": "number", "description": "what a question costs the person, in dollars"},
                "mostly_settles": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "programme": {"type": "string"}, "share": {"type": "number"}}}},
                "note": {"type": "string"}}}}},
        "Finish": {"allOf": [ref("Assessment"), {"type": "object", "required": ["action", "progress"], "properties": {
            "action": {"const": "finish"}, "progress": ref("Progress")}}]},
        "HandOff": {"allOf": [ref("Assessment"), {"type": "object", "required": ["action", "progress",
                                                                                 "hand_off_reason"], "properties": {
            "action": {"const": "hand_off"}, "progress": ref("Progress"),
            "hand_off_reason": {"type": "string", "description": "why a person should review; advice to show, "
                                                                 "not a referral"}}}]},
        "Step": {"type": "object", "required": ["state", "readings", "next"], "properties": {
            "state": ref("State"), "readings": {"type": "array", "items": ref("Reading")},
            "next": {"oneOf": [ref("Ask"), ref("Finish"), ref("HandOff")],
                     "discriminator": {"propertyName": "action", "mapping": {
                         "ask": "#/components/schemas/Ask", "finish": "#/components/schemas/Finish",
                         "hand_off": "#/components/schemas/HandOff"}}}}},
        "FullAssessment": {"allOf": [ref("Assessment"), {"type": "object", "properties": {
            "state": ref("State"), "all_programmes": {"type": "array", "items": ref("Programme")},
            "facts": {"type": "array", "items": {"type": "object", "properties": {
                "question": {"type": "string"}, "reading": {"type": "string"}}}},
            "expected_mistake_cost": {"type": "integer"}, "support": ref("Support")}}]},
        "Explanation": {"type": "object", "required": ["state", "near_the_line", "answers_tried", "note"], "properties": {
            "state": ref("State"),
            "near_the_line": {"type": "array", "description": "programmes whose result one answer, one step "
                                                              "different, would change; the rest would not change",
                              "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "programme": {"type": "string"},
                "tier": {"type": "string", "enum": tiers}, "probability": {"type": "number"},
                "would_change": {"type": "array", "items": {"type": "object", "properties": {
                    "question": {"type": "string"}, "kind": {"type": "string", "enum": ["amount", "household"]},
                    "from": {"type": "string"}, "to": {"type": "string", "description": "the neighbouring option"},
                    "tier": {"type": "string", "enum": tiers}, "probability": {"type": "number"},
                    "probability_change": {"type": "number"}}}}}}},
            "answers_tried": {"type": "integer"},
            "note": {"type": "string", "description": "how to read it: a guide to re-checking, never advice on "
                                                      "what to report"}}},
        "Catalogue": {"type": "object", "properties": {
            "version": {"type": "string"}, "state": ref("State"),
            "about": {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
            "programmes": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "label": {"type": "string"}}}},
            "questions": {"type": "array", "items": ref("Question")}}},
        "Health": {"type": "object", "properties": {
            "ok": {"type": "boolean"}, "version": {"type": "string"}, "engine": {"type": "string"},
            "rules": {"type": ["string", "null"]}, "commit": {"type": ["string", "null"]},
            "bundle_sha256": {"type": "string", "description": "first 12 hex digits of the model file's sha256"},
            "rules_sha256": {"type": ["string", "null"], "description": "first 12 hex digits of the sha256 of the "
                                                                        "rules file as canonical JSON"},
            "households": {"type": "integer"}, "questions": {"type": "integer"}, "programmes": {"type": "integer"},
            "boot_seconds": {"type": "number"}, "uptime_seconds": {"type": "integer"}}},
        "Error": {"type": "object", "required": ["error"], "properties": {"error": {"type": "object", "properties": {
            "code": {"type": "string", "enum": ["bad_json", "bad_request", "bad_state", "unsupported_media_type",
                                                "too_large", "not_found", "method_not_allowed", "forbidden_host",
                                                "internal"]},
            "message": {"type": "string"}}}}},
    }
    doc = {
        "openapi": "3.1.0",
        "info": {"title": "DecisionIntake", "version": VERSION,
                 "license": {"name": "MIT", "identifier": "MIT"},
                 "description": "Decides what to ask next in a US benefits screening (ten federal programmes, 2026 "
                                "rules from PolicyEngine-US), how far to trust each answer, and when it knows enough "
                                "to recommend or should suggest a person review. A screening estimate, not an "
                                "eligibility decision. Stateless: every response carries `state`; send it back with "
                                "the next call. No authentication; requests are not logged with their content."},
        "paths": {
            "/v1/questions": {"get": {"operationId": "getQuestions",
                                      "summary": "The programmes, every question, and the starting state",
                                      "responses": {"200": resp("the catalogue", ref("Catalogue")), "429": limited}}},
            "/v1/step": {"post": {
                "operationId": "step",
                "summary": "Record facts (any questions, asked or not) and get the next step: ask, finish or hand_off",
                "requestBody": body({"type": "object", "properties": {
                    "state": ref("State"), "facts": {"type": "array", "maxItems": MAX_FACTS, "items": ref("Fact")}}},
                    {"state": example_state, "facts": [{"question": "children", "choice": "2"}]}),
                "responses": {"200": resp("the new state, how each fact was read, and the next step", ref("Step")),
                              "400": err, "413": err, "415": err, "429": limited}}},
            "/v1/assessment": {"post": {
                "operationId": "assessment",
                "summary": "Every programme's probability, tier and yearly value range; the facts so far",
                "requestBody": body({"type": "object", "required": ["state"],
                                     "properties": {"state": ref("State")}}, {"state": example_state}),
                "responses": {"200": resp("the assessment", ref("FullAssessment")), "400": err, "413": err,
                              "415": err, "429": limited}}},
            "/v1/explain": {"post": {
                "operationId": "explain",
                "summary": "What would change each result: every single answer, moved one step (an income bracket, "
                           "yes/no), that changes a programme's tier, with the others kept",
                "requestBody": body({"type": "object", "required": ["state"],
                                     "properties": {"state": ref("State")}}, {"state": example_state}),
                "responses": {"200": resp("the programmes near the line", ref("Explanation")), "400": err,
                              "413": err, "415": err, "429": limited}}},
            "/v1/cliff": {"post": {
                "operationId": "cliff",
                "summary": "A benefits-cliff view: every option of one amount question (earned_income unless named), "
                           "the other answers kept, with each programme's tier, probability and expected yearly value",
                "requestBody": body({"type": "object", "required": ["state"], "properties": {
                    "state": ref("State"), "question": {"type": "string", "default": "earned_income",
                                                        "description": "an amount question id"}}},
                    {"state": example_state}),
                "responses": {"200": resp("one entry per option", {"type": "object", "properties": {
                    "state": ref("State"), "question": {"type": "string"}, "unit": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "object", "properties": {
                        "option": {"type": "string"}, "current": {"type": "boolean"},
                        "expected_value": {"type": "integer", "description": "dollars a year, all programmes"},
                        "by_group": {"type": "object", "additionalProperties": {"type": "integer"}},
                        "programmes": {"type": "array", "items": {"type": "object", "properties": {
                            "id": {"type": "string"}, "tier": {"type": "string", "enum": tiers},
                            "probability": {"type": "number"}, "typical_value": {"type": "integer"},
                            "expected_value": {"type": "integer"}}}}}}},
                    "groups": {"type": "object"}, "note": {"type": "string"}}}),
                              "400": err, "413": err, "415": err, "429": limited}}},
            "/mcp": {"post": {
                "operationId": "mcp",
                "summary": "Model Context Protocol over Streamable HTTP (JSON responses, no SSE, stateless). "
                           "Revisions 2026-07-28 (no handshake; _meta plus MCP-Protocol-Version, Mcp-Method and "
                           "Mcp-Name headers) and 2025-11-25, 2025-06-18, 2025-03-26 (initialize handshake). Tools: "
                           "start_intake, record_facts, get_assessment, what_would_change.",
                "requestBody": body({"type": "object", "description": "a JSON-RPC 2.0 request"}),
                "responses": {"200": resp("a JSON-RPC response", {"type": "object"}),
                              "202": {"description": "notification accepted"},
                              "400": resp("a JSON-RPC error (bad version, header mismatch, missing _meta)",
                                          {"type": "object"}),
                              "403": resp("a browser Origin this server does not accept", {"type": "object"}),
                              "404": resp("2026-07-28: an unknown method (-32601)", {"type": "object"}),
                              "429": limited}}},
            "/healthz": {"get": {"operationId": "health", "summary": "Liveness and exactly what is deployed",
                                 "responses": {"200": resp("ok", ref("Health"))}}},
        },
        "components": {"schemas": schemas, "responses": {
            "Error": resp("error", ref("Error")),
            "RateLimited": {"description": "over 120 requests a minute from one IP address (enforced at the edge); "
                                           "wait a minute"}}},
    }
    if base:
        doc["servers"] = [{"url": base}]
    return doc


def create_app(**kw) -> App:
    origins = os.environ.get("DI_ALLOWED_ORIGINS")
    kw.setdefault("allowed_origins", [o.strip() for o in origins.split(",") if o.strip()] if origins else None)
    hosts = os.environ.get("DI_ALLOWED_HOSTS")
    if hosts:
        kw.setdefault("allowed_hosts", [h for h in hosts.split(",") if h.strip()])
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
    local = a.host in LOCAL_HOSTS or a.host == "[::1]"
    app = create_app(bundle_path=a.bundle, static_dir=a.static or None,
                     **({"allowed_hosts": sorted(LOCAL_HOSTS)} if local else {}))  # DNS-rebinding guard
    print(f"decisionintake {VERSION}: {len(app.b.pop):,} households, ready in {app.boot_s:.1f}s on "
          f"http://{a.host}:{a.port}", file=sys.stderr, flush=True)
    with make_server(a.host, a.port, app, server_class=Server, handler_class=Quiet) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
