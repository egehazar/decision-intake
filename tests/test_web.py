"""The hosted service: a state string rebuilds an intake exactly, the JSON API and the remote MCP endpoint
behave as documented, and the whole thing answers over a real socket. Runs on the shipped bundle."""

import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from decisionintake import bundle
from decisionintake.mcp_server import Server, StatelessServer
from decisionintake.service import Intake, Settings
from decisionintake.state import StateError, decode, encode

SHIPPED = Path(__file__).resolve().parents[1] / "bundles" / "benefits_v2.npz"
pytestmark = pytest.mark.skipif(not SHIPPED.exists(), reason="shipped bundle not present")

FACTS = [
    {"question": "state", "choice": "TX"},
    {"question": "household_size", "choice": 3},
    {"question": "children", "choice": "1"},
    {"question": "earned_income", "amount": {"value": 2400, "period": "month", "approximate": True}},
    {"question": "tenure", "choice": "rent"},
    {"question": "housing_cost", "amount": {"low": 1100, "high": 1300, "period": "month"}},
    {"question": "assets", "dont_know": True},
    {"question": "senior", "choice": False},
]


@pytest.fixture(scope="module")
def b():
    return bundle.load(str(SHIPPED))


@pytest.fixture(scope="module")
def app():
    from decisionintake.web import App
    return App(bundle_path=str(SHIPPED), static_dir=str(Path(__file__).resolve().parents[1] / "public"), log=None)


def call(app, method, path, body=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "QUERY_STRING": "", "CONTENT_LENGTH": str(len(data)),
           "CONTENT_TYPE": "application/json" if data else "", "wsgi.input": io.BytesIO(data)}
    if "?" in path:
        env["PATH_INFO"], env["QUERY_STRING"] = path.split("?", 1)
    for k, v in (headers or {}).items():
        env["HTTP_" + k.upper().replace("-", "_")] = v
    out = {}

    def start(status, hdrs):
        out["status"], out["headers"] = int(status.split()[0]), dict(hdrs)

    payload = b"".join(app(env, start))
    out["json"] = json.loads(payload) if payload and out["headers"].get("Content-Type", "").startswith("application/json") else None
    out["body"] = payload
    return out


# --------------------------------------------------------------------------------- the state string
def test_state_round_trips_and_reads_plainly(b):
    it = Intake(b)
    it.record(FACTS)
    s = it.state()
    assert s.startswith("di1;state=TX;household_size=3;children=1;earned_income=~2400/month;tenure=rent;"
                        "housing_cost=1100..1300/month;assets=?;senior=no;#")
    assert encode(decode(b.pop.questions, s)) == s


def test_a_rebuilt_intake_is_the_same_intake(b):
    """Stateless (rebuild from the string every call) and in-memory give identical answers, corrections included."""
    rng = np.random.default_rng(7)
    for trial in range(6):
        order = rng.permutation(len(FACTS))
        live, token = Intake(b), Intake(b).state()
        for k in order:
            f = FACTS[k]
            live.record([f])
            it = Intake.from_state(b, token)
            it.record([f])
            token = it.state()
        live.record([{"question": "children", "choice": "2"}])  # a correction replays everything
        it = Intake.from_state(b, token)
        it.record([{"question": "children", "choice": "2"}])
        a, c = live.next_step(), Intake.from_state(b, it.state()).next_step()
        assert a == c
        assert live.assessment() == Intake.from_state(b, it.state()).assessment()


def test_a_damaged_or_foreign_state_is_refused(b):
    it = Intake(b)
    it.record(FACTS[:3])
    s = it.state()
    for bad in [s.replace("TX", "CA"), s[:-1] + ("0" if s[-1] != "0" else "1"), "di2;#0000", "", "x" * 3000,
                s.replace("state=TX", "planet=Mars")]:
        with pytest.raises(StateError):
            decode(b.pop.questions, bad)


# --------------------------------------------------------------------------------- the JSON API
def test_health_questions_and_openapi(app):
    h = call(app, "GET", "/healthz")
    assert h["status"] == 200 and h["json"]["ok"] and h["json"]["households"] == 48657
    assert h["headers"]["Cache-Control"] == "no-store" and h["headers"]["Access-Control-Allow-Origin"] == "*"
    q = call(app, "GET", "/v1/questions")["json"]
    assert len(q["questions"]) == 18 and len(q["programmes"]) == 10 and q["state"].startswith("di1;#")
    assert "/v1/step" in call(app, "GET", "/openapi.json")["json"]["paths"]


def test_a_whole_screening_through_the_api(app):
    r = call(app, "POST", "/v1/step", {"facts": FACTS[:3]})
    assert r["status"] == 200 and r["json"]["next"]["action"] == "ask"
    state, asked = r["json"]["state"], 0
    answers = {"child_under5": "no", "infant_or_pregnant": "no", "disabled": "no", "married": "yes",
               "self_employed": "no", "employer_insurance": "no", "college": "no", "senior": "no", "tenure": "rent"}
    amounts = {"earned_income": 28000, "social_security": 0, "other_income": 0, "housing_cost": 1200,
               "childcare": 0, "assets": 1500}
    nxt = r["json"]["next"]
    while nxt["action"] == "ask":
        qid = nxt["question"]["id"]
        fact = {"question": qid, "amount": {"value": amounts[qid]}} if qid in amounts else {"question": qid, "choice": answers[qid]}
        r = call(app, "POST", "/v1/step", {"state": state, "facts": [fact]})["json"]
        assert r["readings"][0].get("error") is None
        state, nxt, asked = r["state"], r["next"], asked + 1
        assert asked <= 18
    assert nxt["action"] in ("finish", "hand_off") and nxt["recommend"]
    a = call(app, "POST", "/v1/assessment", {"state": state})["json"]
    assert len(a["all_programmes"]) == 10 and a["state"] == state


def test_the_api_refuses_bad_input_cleanly(app):
    assert call(app, "POST", "/v1/step", raw=b"{not json")["json"]["error"]["code"] == "bad_json"
    assert call(app, "POST", "/v1/step", {"state": "di1;state=TX;#0000"})["json"]["error"]["code"] == "bad_state"
    assert call(app, "POST", "/v1/step", {"facts": [{}] * 41})["status"] == 400
    assert call(app, "POST", "/v1/step", raw=b"x" * (65 * 1024))["status"] == 413
    assert call(app, "GET", "/v1/step")["status"] == 405
    assert call(app, "GET", "/v1/nothing")["status"] == 404
    r = call(app, "POST", "/v1/step", {"facts": [{"question": "children", "choice": "eleventy"}, {"question": "nope"}]})
    assert r["status"] == 200 and all("error" in x for x in r["json"]["readings"])
    assert call(app, "GET", "/../pyproject.toml")["status"] == 404  # no escaping the static folder


def test_vercel_rewrites_route_by_query(app):
    assert call(app, "GET", "/api/index?route=healthz")["json"]["ok"]


# --------------------------------------------------------------------------------- MCP over HTTP
def rpc(app, method, params=None, id=1, headers=None):
    msg = {"jsonrpc": "2.0", "method": method, **({"id": id} if id is not None else {}), **({"params": params} if params else {})}
    return call(app, "POST", "/mcp", msg, headers=headers)


def test_mcp_over_http_runs_an_intake_statelessly(app):
    init = rpc(app, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
    assert init["json"]["result"]["protocolVersion"] == "2025-06-18"
    assert "Mcp-Session-Id" not in init["headers"]
    assert rpc(app, "notifications/initialized", id=None)["status"] == 202
    tools = rpc(app, "tools/list")["json"]["result"]["tools"]
    assert {t["name"] for t in tools} == {"start_intake", "record_facts", "next_step", "get_assessment"}
    assert all(t["annotations"]["readOnlyHint"] for t in tools)
    assert "state" in tools[1]["inputSchema"]["required"]
    start = rpc(app, "tools/call", {"name": "start_intake", "arguments": {}})["json"]["result"]["structuredContent"]
    rec = rpc(app, "tools/call", {"name": "record_facts", "arguments": {"state": start["state"], "facts": FACTS}})
    out = rec["json"]["result"]["structuredContent"]
    assert out["state"].count(";") == len(FACTS) + 1 and out["next"]["action"] in ("ask", "finish", "hand_off")
    nxt = rpc(app, "tools/call", {"name": "next_step", "arguments": {"state": out["state"]}})["json"]["result"]
    assert nxt["structuredContent"]["state"] == out["state"]
    bad = rpc(app, "tools/call", {"name": "next_step", "arguments": {"state": "di1;#0000"}})["json"]["result"]
    assert bad["isError"] and "damaged" in bad["content"][0]["text"]


def test_mcp_transport_rules(app):
    assert call(app, "GET", "/mcp")["status"] == 405
    assert rpc(app, "ping", headers={"MCP-Protocol-Version": "1999-01-01"})["status"] == 400
    # version negotiation happens inside initialize: a newer client's header can't be refused there
    init = rpc(app, "initialize", {"protocolVersion": "2099-01-01"}, headers={"MCP-Protocol-Version": "2099-01-01"})
    assert init["status"] == 200 and init["json"]["result"]["protocolVersion"] == "2025-11-25"
    assert rpc(app, "initialize", {"protocolVersion": "2025-11-25"})["json"]["result"]["protocolVersion"] == "2025-11-25"
    assert rpc(app, "nope")["json"]["error"]["code"] == -32601
    batch = call(app, "POST", "/mcp", [{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                        {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert batch["json"] == [{"jsonrpc": "2.0", "id": 1, "result": {}}]
    assert call(app, "POST", "/mcp", raw=b"{oops")["json"]["error"]["code"] == -32700


def test_stateless_and_session_servers_give_the_same_steps(b):
    s, sl = Server(b, Settings()), StatelessServer(b, Settings())
    sid = s.call("start_intake", {})["session_id"]
    st = sl.call("start_intake", {})["state"]
    for f in FACTS:
        a = s.call("record_facts", {"session_id": sid, "facts": [f]})
        c = sl.call("record_facts", {"state": st, "facts": [f]})
        st = c.pop("state")
        assert a == c


# --------------------------------------------------------------------------------- a real socket
def test_the_service_answers_over_a_real_socket(app):
    from wsgiref.simple_server import WSGIRequestHandler, make_server

    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):
            pass

    srv = make_server("127.0.0.1", 0, app, handler_class=Quiet)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        with urllib.request.urlopen(base + "/healthz") as r:
            assert json.loads(r.read())["ok"]
        req = urllib.request.Request(base + "/v1/step", data=json.dumps({"facts": FACTS[:2]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["next"]["action"] == "ask"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/v1/nothing")
        assert e.value.code == 404
    finally:
        srv.shutdown()
