"""DecisionIntake as an MCP server: any assistant that speaks the Model Context Protocol can run an intake.

The assistant keeps the conversation; the engine keeps the decision. Four tools:

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

PROTOCOLS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "benefits_v2.npz"  # v2.1 did not pass D35
ABSTAIN_BELOW = 10.0  # D33/D35: under 10 households like the person, uncertain negatives are "can't tell"
VERSION = "1.0.0"

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
   ones as worth a look, with the yearly value ranges, and the note. Programmes in "unsure" are ones the
   engine can't call from these answers: say so, and that the person can apply or ask 211; never call them
   unlikely. "hand_off": say a caseworker should
   review, and why.
4. Never decide eligibility yourself and never ask questions the engine didn't ask for, unless the person
   needs something clarified.
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
    {"name": "start_intake",
     "description": "Open a new benefits-screening intake. Returns a session_id, the ten programmes, every question "
                    "the engine can use (ids, options, amount units), and the first step.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "record_facts",
     "description": "Record what the person said as typed facts (any questions, asked or volunteered; a fact for an "
                    "answered question replaces it). Returns how each fact was read and the next step.",
     "inputSchema": {"type": "object", "properties": {**SID, "facts": {"type": "array", "items": FACT}},
                     "required": ["session_id", "facts"]}},
    {"name": "next_step",
     "description": "What to do now: ask a question (with what it is worth and which programmes it settles), "
                    "finish with recommendations, or hand off to a person.",
     "inputSchema": {"type": "object", "properties": SID, "required": ["session_id"]}},
    {"name": "get_assessment",
     "description": "The current assessment: every programme's probability, tier and yearly value range, the facts "
                    "recorded so far, and the expected cost of mistakes.",
     "inputSchema": {"type": "object", "properties": SID, "required": ["session_id"]}},
]


READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
for _t in TOOLS:
    _t["annotations"] = READ_ONLY

INSTRUCTIONS_STATELESS = INSTRUCTIONS.replace(
    "1. Call start_intake.",
    "0. This server keeps nothing between calls. Every result carries `state`, a short string holding the facts "
    "so far; pass the state from the MOST RECENT result to every call, exactly as given, and never send two calls "
    "at once with the same state. record_facts returns the next step (with no facts, it only returns the next "
    "step).\n1. Call start_intake.")
STATE = {"state": {"type": "string", "description": "the state string from the most recent result, exactly as given"}}
TOOLS_STATELESS = [
    {**TOOLS[0], "description": TOOLS[0]["description"] + " Also returns the first `state`."},
    {**TOOLS[1], "description": TOOLS[1]["description"] + " Returns the new `state`; use it from now on. With an "
                                                          "empty facts list it only returns the next step.",
     "inputSchema": {"type": "object", "properties": {**STATE, "facts": {"type": "array", "items": FACT}},
                     "required": ["state", "facts"]}},
    {**TOOLS[3], "inputSchema": {"type": "object", "properties": STATE, "required": ["state"]}},
]  # no next_step: record_facts returns it, and a next_step sent alongside answers for an older state (D31)


class Server:
    TOOLS = TOOLS
    INSTRUCTIONS = INSTRUCTIONS

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
            return self._intake(args).assessment()
        raise ToolError(f"unknown tool {name!r}")

    # JSON-RPC -----------------------------------------------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method is None:  # a response from the client; we never send requests
            return None
        if mid is None:  # notification (initialized, cancelled): nothing to answer
            return None
        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                result = {"protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "decisionintake", "version": VERSION},
                          "instructions": self.INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.TOOLS}
            elif method == "tools/call":
                try:
                    out = self.call(params.get("name"), params.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out,
                              "isError": False}
                except ToolError as e:
                    result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as e:  # never let one bad call take the server down
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}


class ToolError(Exception):
    pass


class StatelessServer(Server):
    """Keeps nothing: the facts travel in `state` (state.py) and each call rebuilds the belief (~0.1 s)."""
    TOOLS = TOOLS_STATELESS
    INSTRUCTIONS = INSTRUCTIONS_STATELESS

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
            if len(facts) > 40:
                raise ToolError("at most 40 facts in one call")
            it, readings = self._restore(args, facts)
            return {"state": it.state(), "readings": readings, "next": it.next_step()}
        if name == "get_assessment":
            it, _ = self._restore(args)
            return {"state": it.state(), **it.assessment()}
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
