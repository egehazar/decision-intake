"""DecisionIntake as an MCP server: any assistant that speaks the Model Context Protocol can run an intake.

The assistant keeps the conversation; the engine keeps the decision. Four tools (three in --stateless mode,
which has no next_step: record_facts returns the next step):

    start_intake    open an intake; returns every question (so volunteered facts can be recorded) and the first step
    record_facts    typed facts from what the person said, asked for or not; returns how each was read, and the next step
    next_step       ask (which question, what it's worth, which programmes it settles), finish, or hand off
    get_assessment  every programme's probability and tier, the facts so far, the expected cost of mistakes

Transport: JSON-RPC 2.0 over stdio, one message per line (the MCP stdio transport). No dependencies
beyond numpy; nothing but protocol messages is written to stdout.

Two ways to keep an intake between calls. The default keeps sessions in memory (a local server for one
assistant). --stateless keeps nothing: every result carries `state`, a short string holding the facts so far
(state.py), and every call passes it back. That is how the hosted service runs (web.py).

usage: python -m decisionintake.mcp_server [--bundle bundles/benefits_v2.npz] [--stateless]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from . import bundle as bundle_mod
from .guards import DEFAULT_RULES, attach
from .service import FactError, Intake, Settings
from .state import StateError

# Legacy revisions (an initialize handshake). 2024-11-05 is stdio only: over HTTP that revision used the old
# HTTP+SSE transport, which the hosted service does not serve, so it is not offered there.
PROTOCOLS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
HTTP_PROTOCOLS = PROTOCOLS[:3]
MODERN = ["2026-07-28"]  # stateless: version and capabilities in every request's _meta, server/discover
SUPPORTED = MODERN + PROTOCOLS
K_VERSION, K_CAPS = "io.modelcontextprotocol/protocolVersion", "io.modelcontextprotocol/clientCapabilities"
K_SERVER = "io.modelcontextprotocol/serverInfo"
LIST_TTL_MS = 3_600_000  # tools/list and server/discover change only with a deploy
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "benefits_v3.npz"  # v2 + five programmes (D43)
ABSTAIN_BELOW = 10.0  # D33/D35: under 10 households like the person, uncertain negatives are "can't tell"
VERSION = "1.1.0"
MAX_FACTS = 40  # facts in one record_facts call (the HTTP API uses the same limit)

INSTRUCTIONS = """\
DecisionIntake decides what to ask in a benefits screening, how far to trust each answer, and when it knows
enough. You handle the conversation; it handles the decisions.

1. Call start_intake. Greet the person; invite them to describe their situation in their own words.
2. Whenever the person says something, call record_facts with EVERY fact it states or clearly implies,
   for any question in the catalogue, not only the one you asked. Amounts: give the number exactly as said
   with its period ("2,400 a month" -> {"value": 2400, "period": "month"}) and approximate=true for
   "about"/"around"/ranges. Do no arithmetic yourself. If they don't know, record dont_know.
   If a reading comes back with an error, fix the fact; with a warning, check it with the person.
3. Follow the step it returns. "ask": ask that one question in plain, friendly words (you may say why,
   using why.mostly_settles). "finish": present the recommendations, "likely" ones first, "worth checking"
   ones as worth a look, with the yearly value ranges, and always the note. Programmes in "unsure" are ones
   the engine can't call from these answers: say so, and that the person can still apply or call 211; never
   call them unlikely. "hand_off": say a caseworker or benefits navigator should review, using
   hand_off_reason, and still share the recommendations so far. get_assessment's plan gives the order to
   apply in (apply now, then, at tax time) and the federal rules by which one approval helps another (SNAP
   approval letter proves WIC income, for one): use it when there are several programmes. After a result,
   what_would_change says
   which answers sit near a line (one income bracket away, say): mention those as "check again if this
   changes", never as what the person should report.
4. Never decide eligibility yourself and never ask questions the engine didn't ask for, unless the person
   needs something clarified. This is a screening estimate from 2026 US rules (PolicyEngine-US), not
   an eligibility decision; only the programme decides.
5. Speak the person's language; the engine's ids and options stay in English.
"""

FACT = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "question id from the catalogue"},
        "choice": {"type": ["string", "number", "boolean"], "description": "the option's text, e.g. \"yes\", \"2\", \"rent\", \"CA\""},
        "amount": {
            "type": "object",
            "description": "amount questions only: the number as the person said it",
            "properties": {
                "value": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
                "period": {"type": "string", "enum": ["year", "month", "week", "none"]},
                "approximate": {"type": "boolean"},
            },
        },
        "dont_know": {"type": "boolean"},
    },
    "required": ["question"],
}
SID = {"session_id": {"type": "string"}}
TOOLS = [
    {"name": "start_intake", "title": "Start a benefits screening",
     "description": "Open a new US benefits-screening intake (fifteen programmes: federal benefits and tax credits, "
                    "state tax credits, Head Start, housing; 2026 rules). Use it when a person wants to know which "
                    "benefits to apply for. Returns a session_id, the programmes, "
                    "every question the engine can use (ids, options, amount units), and the first step.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "record_facts", "title": "Record what the person said",
     "description": "Record what the person said as typed facts (any questions, asked or volunteered; a fact for an "
                    "answered question replaces it). Returns how each fact was read and the next step.",
     "inputSchema": {"type": "object", "properties": {**SID, "facts": {"type": "array", "items": FACT}},
                     "required": ["session_id", "facts"]}},
    {"name": "next_step", "title": "What to do next",
     "description": "What to do now: ask a question (with what it is worth and which programmes it settles), "
                    "finish with recommendations, or hand off to a person.",
     "inputSchema": {"type": "object", "properties": SID, "required": ["session_id"]}},
    {"name": "get_assessment", "title": "Current assessment",
     "description": "The current assessment: every programme's probability, tier and yearly value range, the facts "
                    "recorded so far, the expected cost of mistakes, and a plan: the recommended programmes in the order to "
                    "apply, with the federal rules linking them, each cited. A screening estimate, not an "
                    "eligibility decision.",
     "inputSchema": {"type": "object", "properties": SID, "required": ["session_id"]}},
    {"name": "what_would_change", "title": "What would change the result",
     "description": "For each programme near the line: which single answer, one step different (one bracket of "
                    "income, the other yes/no), would change its result, and to what. Use it to tell the person "
                    "what to re-check if their situation changes; never to suggest what they should report.",
     "inputSchema": {"type": "object", "properties": SID, "required": ["session_id"]}},
]


READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
WRITES_SESSION = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}
for _t in TOOLS:  # in-memory sessions: opening one and recording into it change what the server holds
    _t["annotations"] = WRITES_SESSION if _t["name"] in ("start_intake", "record_facts") else READ_ONLY

INSTRUCTIONS_STATELESS = INSTRUCTIONS.replace(
    "1. Call start_intake.",
    "0. This server keeps nothing between calls. Every result carries `state`, a short string holding the facts "
    "so far; pass the state from the MOST RECENT result to every call, exactly as given, and never send two calls "
    "at once with the same state. record_facts returns the next step (with no facts, it only returns the next "
    "step).\n1. Call start_intake.")
STATE = {"state": {"type": "string", "description": "the state string from the most recent result, exactly as given"}}
TOOLS_STATELESS = [
    {**TOOLS[0], "description": "Open a new benefits-screening intake. Returns the first `state` (pass it to every "
                                "later call), the fifteen programmes, every question the engine can use (ids, options, "
                                "amount units), and the first step."},
    {**TOOLS[1], "description": TOOLS[1]["description"] + " Returns the new `state`; use it from now on. With an "
                                                          "empty facts list it only returns the next step.",
     "inputSchema": {"type": "object", "properties": {**STATE, "facts": {"type": "array", "items": FACT}},
                     "required": ["state", "facts"]}},
    {**TOOLS[3], "inputSchema": {"type": "object", "properties": STATE, "required": ["state"]}},
    {**TOOLS[4], "inputSchema": {"type": "object", "properties": STATE, "required": ["state"]}},
]  # no next_step: record_facts returns it, and a next_step sent alongside answers for an older state (D31)
for _t in TOOLS_STATELESS:  # nothing is kept: every call is a pure function of its arguments
    _t["annotations"] = READ_ONLY


class Server:
    TOOLS = TOOLS
    INSTRUCTIONS = INSTRUCTIONS
    PROTOCOLS = PROTOCOLS  # what initialize may agree to

    @property
    def supported(self) -> list[str]:
        return MODERN + self.PROTOCOLS

    def __init__(self, b, settings: Settings, max_sessions: int = 200):
        self.b, self.settings, self.max_sessions = b, settings, max_sessions
        self.sessions: OrderedDict[str, Intake] = OrderedDict()

    def _intake(self, args) -> Intake:
        sid = args.get("session_id")
        if sid not in self.sessions:
            raise ToolError(f"unknown session_id {sid!r}; call start_intake first")
        self.sessions.move_to_end(sid)
        return self.sessions[sid]

    def call(self, name: str, args: dict) -> dict:
        if name == "start_intake":
            it = Intake(self.b, self.settings)
            sid = uuid.uuid4().hex[:12]
            self.sessions[sid] = it
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)
            return {"session_id": sid, "programmes": list(self.b.labels.values()), "questions": it.catalogue(),
                    "next": it.next_step()}
        if name == "record_facts":
            it = self._intake(args)
            facts = args.get("facts")
            if not isinstance(facts, list):
                raise ToolError("facts must be a list")
            readings = it.record(facts)
            return {"readings": readings, "next": it.next_step()}
        if name == "next_step":
            return self._intake(args).next_step()
        if name == "get_assessment":
            it = self._intake(args)
            return {**it.assessment(), "plan": it.plan()["steps"]}
        if name == "what_would_change":
            return self._intake(args).explain()
        raise ToolError(f"unknown tool {name!r}")

    # JSON-RPC -----------------------------------------------------------------------------------
    def _tool_result(self, params) -> dict:
        name, args = params.get("name"), params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(name, str) or name not in {t["name"] for t in self.TOOLS}:
            raise InvalidParams(f"unknown tool: {name!r}")
        if not isinstance(args, dict):
            raise InvalidParams("arguments must be an object")
        try:
            out = self.call(name, args)
            return {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out, "isError": False}
        except ToolError as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}

    def _internal(self, e: Exception) -> dict:
        """An unexpected failure: the caller gets a fixed message; the traceback goes to stderr (the server log)."""
        import traceback
        traceback.print_exception(e, file=sys.stderr)
        return {"code": -32603, "message": "internal error"}

    def handle(self, msg: dict, modern: bool = False) -> dict | None:
        """One JSON-RPC message. modern=True when the transport already knows it is 2026-07-28 (the HTTP header),
        so a request without _meta is refused instead of being answered the old way."""
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params")
        if method is None:  # a response from the client; we never send requests
            return None
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None if mid is None else {"jsonrpc": "2.0", "id": mid,
                                             "error": {"code": -32602, "message": "params must be an object"}}
        meta = params.get("_meta")
        if modern or (isinstance(meta, dict) and K_VERSION in meta) or method == "server/discover":
            return self.handle_modern(mid, method, params, meta if isinstance(meta, dict) else {})
        if mid is None:  # notification (initialized, cancelled): nothing to answer
            return None
        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                result = {"protocolVersion": want if want in self.PROTOCOLS else self.PROTOCOLS[0],
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "decisionintake", "version": VERSION},
                          "instructions": self.INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.TOOLS}
            elif method == "tools/call":
                result = self._tool_result(params)
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        except InvalidParams as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": str(e)}}
        except Exception as e:  # never let one bad call take the server down
            return {"jsonrpc": "2.0", "id": mid, "error": self._internal(e)}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def handle_modern(self, mid, method, params, meta) -> dict | None:
        """MCP 2026-07-28: no handshake; every request names its version and capabilities in _meta."""
        def error(code, message, data=None):
            e = {"code": code, "message": message}
            if data is not None:
                e["data"] = data
            return {"jsonrpc": "2.0", "id": mid, "error": e}

        if mid is None:  # this revision defines no client notifications over HTTP; nothing to answer
            return None
        version = meta.get(K_VERSION)
        if version is None:
            return error(-32602, f"request is missing _meta[{K_VERSION!r}]")
        if version not in MODERN:
            return error(-32022, "Unsupported protocol version", {"supported": self.supported, "requested": version})
        if not isinstance(meta.get(K_CAPS), dict):
            return error(-32602, f"request is missing _meta[{K_CAPS!r}]")
        server = {K_SERVER: {"name": "decisionintake", "version": VERSION}}
        try:
            if method == "server/discover":
                result = {"supportedVersions": self.supported, "capabilities": {"tools": {"listChanged": False}},
                          "instructions": self.INSTRUCTIONS, "ttlMs": LIST_TTL_MS, "cacheScope": "public"}
            elif method == "tools/list":
                result = {"tools": self.TOOLS, "ttlMs": LIST_TTL_MS, "cacheScope": "public"}
            elif method == "tools/call":
                result = self._tool_result(params)
            else:  # initialize, ping and logging/setLevel are gone in this revision
                return error(-32601, f"method not found: {method}")
        except InvalidParams as e:
            return error(-32602, str(e))
        except Exception as e:  # never let one bad call take the server down
            return error(**self._internal(e))
        return {"jsonrpc": "2.0", "id": mid, "result": {"resultType": "complete", **result, "_meta": server}}


class ToolError(Exception):
    """A tool ran but the input was wrong (bad state, bad fact): reported in the result with isError."""


class InvalidParams(Exception):
    """The request itself is wrong (unknown tool, arguments not an object): a JSON-RPC -32602 error."""


class StatelessServer(Server):
    """Keeps nothing: the facts travel in `state` (state.py) and each call rebuilds the belief (~0.1 s)."""
    TOOLS = TOOLS_STATELESS
    INSTRUCTIONS = INSTRUCTIONS_STATELESS
    PROTOCOLS = HTTP_PROTOCOLS  # the hosted server: Streamable HTTP only

    def __init__(self, b, settings: Settings):
        super().__init__(b, settings, max_sessions=0)

    def _restore(self, args, new=()) -> tuple[Intake, list]:
        try:
            token = args.get("state")
            if not isinstance(token, str) or not token.strip():
                raise StateError("state is missing; call start_intake and pass the state from the most recent result")
            return Intake.resume(self.b, token, list(new), self.settings)
        except (StateError, FactError) as e:
            raise ToolError(str(e)) from None

    def call(self, name: str, args: dict) -> dict:
        if name == "start_intake":
            it = Intake(self.b, self.settings)
            return {"state": it.state(), "programmes": list(self.b.labels.values()), "questions": it.catalogue(),
                    "next": it.next_step()}
        if name == "record_facts":
            facts = args.get("facts")
            if not isinstance(facts, list):
                raise ToolError("facts must be a list")
            if len(facts) > MAX_FACTS:
                raise ToolError(f"at most {MAX_FACTS} facts in one call")
            it, readings = self._restore(args, facts)
            return {"state": it.state(), "readings": readings, "next": it.next_step()}
        if name == "get_assessment":
            it, _ = self._restore(args)
            return {"state": it.state(), **it.assessment(), "plan": it.plan()["steps"]}
        if name == "what_would_change":
            it, _ = self._restore(args)
            return {"state": it.state(), **it.explain()}
        raise ToolError(f"unknown tool {name!r}")


def serve(server: Server, inp=None, out=None) -> None:
    inp = inp or sys.stdin.buffer
    out = out or sys.stdout.buffer
    for raw in inp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            if isinstance(msg, list):  # a batch (2025-03-26)
                reply = [r for r in (server.handle(m) for m in msg) if r is not None] or None
            else:
                reply = server.handle(msg)
        if reply is not None:
            out.write(json.dumps(reply).encode() + b"\n")
            out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE),
                    help="engine bundle (.npz); default: the v2 benefits bundle shipped in the repo")
    ap.add_argument("--question-cost", type=float, default=Settings.question_cost)
    ap.add_argument("--handoff-cost", type=float, default=Settings.handoff_cost)
    ap.add_argument("--stateless", action="store_true", help="keep nothing; the facts travel in `state`")
    ap.add_argument("--no-abstain", action="store_true", help="always call negatives (no can't-tell tier, D33)")
    ap.add_argument("--rules", default=str(DEFAULT_RULES), help="statutory limits on the model ('' for none)")
    a = ap.parse_args(argv)
    t0 = time.time()
    b = bundle_mod.load(a.bundle)
    if a.rules:
        attach(b, a.rules)
    Intake(b)  # warm the model's cache of initial scores before the first client call
    print(f"decisionintake: {len(b.pop):,} households, {len(b.pop.questions)} questions, "
          f"{len(b.pop.decisions)} programmes, ready in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    settings = Settings(question_cost=a.question_cost, handoff_cost=a.handoff_cost,
                        abstain_below=None if a.no_abstain else ABSTAIN_BELOW)
    serve(StatelessServer(b, settings) if a.stateless else Server(b, settings))


if __name__ == "__main__":
    main()
