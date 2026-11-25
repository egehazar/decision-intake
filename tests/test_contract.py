"""The public contract, as the round-7 launch audit read it: protocol errors are JSON-RPC errors that never leak
internals, browsers are held to an Origin policy on /mcp, the OpenAPI document is complete and its examples are
real, /healthz says exactly what is deployed, and responses have one shape per field."""

import json
from pathlib import Path

import pytest

from decisionintake import bundle
from decisionintake.mcp_server import Server, StatelessServer
from decisionintake.service import Settings
from decisionintake.state import decode

from test_web import SHIPPED, call

pytestmark = pytest.mark.skipif(not SHIPPED.exists(), reason="shipped bundle not present")
MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
               "io.modelcontextprotocol/clientCapabilities": {}}


@pytest.fixture(scope="module")
def app():
    from decisionintake.web import App
    return App(bundle_path=str(SHIPPED), log=None)


def rpc(app, msg, headers=None, raw=None):
    return call(app, "POST", "/mcp", msg, headers=headers, raw=raw)


# ------------------------------------------------------------------------------------------ protocol errors
@pytest.mark.parametrize("msg, code", [
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]}, -32602),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "record_facts", "arguments": "x"}}, -32602),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}, -32602),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": 7}}, -32602),
    ({"jsonrpc": "2.0", "id": 1, "method": "no/such"}, -32601),
])
def test_bad_requests_are_jsonrpc_errors(app, msg, code):
    r = rpc(app, msg)
    assert r["status"] == 200 and r["json"]["error"]["code"] == code and r["json"]["id"] == 1


def test_empty_batch_is_invalid(app):
    r = rpc(app, None, raw=b"[]")
    assert r["status"] == 400 and r["json"]["error"]["code"] == -32600


def test_internal_errors_say_nothing_about_the_code(app, monkeypatch, capsys):
    def boom(name, args):
        raise RuntimeError("secret detail /home/x/y.py")
    monkeypatch.setattr(app.mcp, "call", boom)
    for msg, headers in [({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "start_intake"}}, None),
                         ({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "start_intake", "_meta": MODERN_META}},
                          {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "start_intake"})]:
        r = rpc(app, msg, headers)
        assert r["json"]["error"] == {"code": -32603, "message": "internal error"}
        assert b"secret" not in r["body"] and b"RuntimeError" not in r["body"]
    assert "secret detail" in capsys.readouterr().err  # the server log keeps it


def test_modern_header_without_meta_is_refused(app):
    r = rpc(app, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/list"})
    assert r["status"] == 400 and r["json"]["error"]["code"] == -32602


def test_hosted_server_does_not_offer_the_sse_era_revision(app, b_):
    r = rpc(app, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
    assert r["json"]["result"]["protocolVersion"] == "2025-11-25"
    d = rpc(app, {"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {"_meta": MODERN_META}},
            {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "server/discover"})
    assert "2024-11-05" not in d["json"]["result"]["supportedVersions"]
    stdio = Server(b_, Settings())  # stdio never changed transport, so the local server keeps it
    init = stdio.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert init["result"]["protocolVersion"] == "2024-11-05"


@pytest.fixture(scope="module")
def b_():
    return bundle.load(str(SHIPPED))


def test_tool_annotations_tell_the_truth(b_):
    hosted = {t["name"]: t for t in StatelessServer(b_, Settings()).TOOLS}
    assert set(hosted) == {"start_intake", "record_facts", "get_assessment", "what_would_change"}
    assert all(t["annotations"]["readOnlyHint"] and t["title"] for t in hosted.values())
    local = {t["name"]: t["annotations"]["readOnlyHint"] for t in Server(b_, Settings()).TOOLS}
    assert local == {"start_intake": False, "record_facts": False, "next_step": True, "get_assessment": True,
                     "what_would_change": True}


def test_header_values_ignore_surrounding_whitespace_but_not_case(app):
    """SEP-2243 as the conformance suite checks it: optional whitespace around a header value is not part of it
    (RFC 9110 5.5); method values are case-sensitive."""
    body = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "start_intake", "arguments": {}, "_meta": MODERN_META}}
    ok = rpc(app, body, {"MCP-Protocol-Version": " 2026-07-28 ", "Mcp-Method": " tools/call\t", "Mcp-Name": "  start_intake "})
    assert ok["status"] == 200 and ok["json"]["result"]["resultType"] == "complete"
    lst = {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {"_meta": MODERN_META}}
    upper = rpc(app, lst, {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "TOOLS/LIST"})
    assert upper["status"] == 400 and upper["json"]["error"]["code"] == -32020


# ------------------------------------------------------------------------------------------ Origin and CORS
LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


@pytest.mark.parametrize("origin, ok", [
    (None, True), ("http://localhost:6274", True), ("http://127.0.0.1:5173", True),
    ("https://decision-intake.vercel.app", False), ("https://evil.example", False), ("null", False),
    ("http://evil.localhost", False), ("http://[::1]:8080", True),
    ("https://localhost.attacker.example", False), ("http://127.0.0.1.attacker.example", False),
    ("https://decision-intake.vercel.app.evil.example", False), ("https://decision-intake.vercel.app:8443", False)])
def test_mcp_origin_policy(app, origin, ok):
    h = {"Host": "decision-intake.vercel.app", "X-Forwarded-Proto": "https"}
    if origin:
        h["Origin"] = origin
    r = rpc(app, LIST, h)
    if ok:
        assert r["status"] == 200 and "tools" in r["json"]["result"]
        assert r["headers"].get("Access-Control-Allow-Origin") == origin  # never "*" on /mcp
    else:
        assert r["status"] == 403 and r["json"]["error"]["code"] == -32600 and r["json"]["id"] is None
        assert "Access-Control-Allow-Origin" not in r["headers"]


def test_same_origin_comes_from_the_deployment_not_the_request(monkeypatch):
    """The conformance suite's DNS-rebinding attack sets Host and Origin to the same attacker domain, so a
    'same origin as the Host header' rule would let it through. The deployment's own address comes from Vercel."""
    from decisionintake.web import App
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "decision-intake.vercel.app")
    a = App(bundle_path=str(SHIPPED), log=None)
    assert rpc(a, LIST, {"Origin": "https://decision-intake.vercel.app", "Host": "decision-intake.vercel.app"})["status"] == 200
    attack = {"Host": "evil.example.com", "Origin": "http://evil.example.com"}
    assert rpc(a, LIST, attack)["status"] == 403


def test_local_server_refuses_a_rebound_host():
    """As the MCP conformance suite's dns-rebinding-protection scenario checks: bound to localhost, a request
    whose Host/Origin name another domain gets a 4xx; one naming localhost is served."""
    from decisionintake.web import App, LOCAL_HOSTS
    a = App(bundle_path=str(SHIPPED), log=None, allowed_hosts=sorted(LOCAL_HOSTS))
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
    bad = rpc(a, init, {"Host": "evil.example.com", "Origin": "http://evil.example.com"})
    assert 400 <= bad["status"] < 500
    assert call(a, "GET", "/v1/questions", headers={"Host": "evil.example.com"})["status"] == 403
    for host in ("localhost:8080", "127.0.0.1:8080", "[::1]:8080"):
        ok = rpc(a, init, {"Host": host, "Origin": "http://" + host})
        assert ok["status"] == 200 and ok["json"]["result"]["protocolVersion"] == "2025-11-25", host


def test_json_api_is_open_to_any_origin(app):
    r = call(app, "GET", "/v1/questions", headers={"Origin": "https://evil.example"})
    assert r["status"] == 200 and r["headers"]["Access-Control-Allow-Origin"] == "*"


def test_allowed_origins_setting(monkeypatch):
    from decisionintake.web import create_app
    monkeypatch.setenv("DI_ALLOWED_ORIGINS", "https://partner.example, ")
    a = create_app(bundle_path=str(SHIPPED), log=None)
    assert rpc(a, LIST, {"Origin": "https://partner.example"})["status"] == 200
    assert rpc(a, LIST, {"Origin": "https://other.example"})["status"] == 403


# ------------------------------------------------------------------------------------------ OpenAPI and health
def refs(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref":
                yield v
            else:
                yield from refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from refs(v)


def test_openapi_is_complete_and_its_examples_are_real(app, b_):
    doc = call(app, "GET", "/openapi.json", headers={"Host": "di.test", "X-Forwarded-Proto": "https"})["json"]
    assert doc["servers"] == [{"url": "https://di.test"}]
    for r in refs(doc):  # every reference resolves
        node = doc
        for part in r.lstrip("#/").split("/"):
            node = node[part]
    ex = doc["components"]["schemas"]["State"]["example"]
    assert [f["question"] for f in decode(b_.pop.questions, ex)] == ["state", "household_size", "earned_income"]
    mcp = doc["paths"]["/mcp"]["post"]["summary"]
    assert "next_step" not in mcp and "2026-07-28" in mcp
    for path in ("/v1/step", "/v1/assessment"):
        req = doc["paths"][path]["post"]["requestBody"]["content"]["application/json"]
        assert call(app, "POST", path, req["example"])["status"] == 200  # the documented example works
    assert doc["components"]["schemas"]["Step"]["properties"]["next"]["discriminator"]["propertyName"] == "action"


def test_health_says_what_is_deployed(app):
    h = call(app, "GET", "/healthz")["json"]
    assert h["rules"] and len(h["bundle_sha256"]) == 12 and len(h["rules_sha256"]) == 12
    assert h["commit"] is None  # VERCEL_GIT_COMMIT_SHA is set on Vercel only


def test_rules_hash_ignores_line_endings(tmp_path):
    from decisionintake.guards import DEFAULT_RULES
    from decisionintake.web import _rules_sha256
    text = Path(DEFAULT_RULES).read_bytes().decode("utf-8").replace("\r\n", "\n")
    (tmp_path / "lf.json").write_bytes(text.encode())
    (tmp_path / "crlf.json").write_bytes(text.replace("\n", "\r\n").encode())
    assert _rules_sha256(tmp_path / "lf.json") == _rules_sha256(tmp_path / "crlf.json") == _rules_sha256(DEFAULT_RULES)


def test_production_bundle_refuses_to_start_without_its_rules(tmp_path):
    from decisionintake.web import App
    bad = tmp_path / "rules.json"
    bad.write_text(json.dumps({"rules": [{"id": "x", "decisions": ["no_such_programme"], "cap": 0.02,
                                          "when": {"children": ["none"]}}]}))
    with pytest.raises((ValueError, KeyError)):
        App(log=None, rules=str(bad))


# ------------------------------------------------------------------------------------------ response shapes
def test_a_fact_is_exactly_one_kind(app):
    r = call(app, "POST", "/v1/step", {"facts": [{"question": "children", "choice": "2", "dont_know": True},
                                                 {"question": "state", "choice": "TX", "dont_know": False}]})["json"]
    assert "exactly one" in r["readings"][0]["error"] and r["readings"][1]["reading"] == "TX"


def test_unsure_is_always_present(app):
    first = call(app, "POST", "/v1/step", {"facts": []})["json"]
    st = call(app, "POST", "/v1/assessment", {"state": first["state"]})["json"]
    assert st["unsure"] == [] or isinstance(st["unsure"], list)
    r = call(app, "POST", "/v1/step", {"facts": [{"question": "state", "choice": "NY"},
                                                 {"question": "senior", "choice": "yes"}]})["json"]
    while r["next"]["action"] == "ask":
        r = call(app, "POST", "/v1/step", {"state": r["state"],
                                           "facts": [{"question": r["next"]["question"]["id"], "dont_know": True}]})["json"]
    assert "unsure" in r["next"] and isinstance(r["next"]["unsure"], list)


def test_mcp_cors_offers_only_post():
    from decisionintake.web import App
    a = App(bundle_path=str(SHIPPED), log=None)
    h = call(a, "OPTIONS", "/mcp", headers={"Origin": "http://localhost:6274"})["headers"]
    assert h["Access-Control-Allow-Methods"] == "POST, OPTIONS" and h["Access-Control-Allow-Origin"] == "http://localhost:6274"


def test_one_shape_per_field(app):
    first = call(app, "POST", "/v1/step", {"facts": []})["json"]
    assert all("id" in s for s in first["next"]["why"]["mostly_settles"])
    a = call(app, "POST", "/v1/assessment", {"state": first["state"]})["json"]
    for row in a["all_programmes"]:
        v = row["value_if_eligible_per_year"]
        assert isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)


# ------------------------------------------------------------------------------------------ what would change it
NEAR = [{"question": "state", "choice": "OH"}, {"question": "household_size", "choice": "1"},
        {"question": "children", "choice": "none"},
        {"question": "earned_income", "amount": {"value": 1900, "period": "month"}},
        {"question": "tenure", "choice": "rent"}, {"question": "disabled", "choice": "no"},
        {"question": "senior", "choice": "no"}]


def test_what_if_equals_a_full_replay(b_):
    """Session.what_if swaps one answer without replaying the rest; it must agree with replaying them all."""
    import numpy as np
    from decisionintake.guards import DEFAULT_RULES, attach
    from decisionintake.service import Intake
    b = bundle.load(str(SHIPPED))
    attach(b, DEFAULT_RULES)
    it = Intake(b, Settings(abstain_below=10.0))
    it.record(NEAR)
    j = it.qi["earned_income"]
    k = int(np.argmax(it.facts["earned_income"]["evidence"]))
    before = it.s.w.copy()
    lik = np.zeros(len(b.pop.questions[j].options))
    lik[k - 1] = 1.0
    fast = it.s.what_if(j, lik)
    ref = Intake(b, Settings(abstain_below=10.0))
    ref.record([f if f["question"] != "earned_income" else
                {"question": "earned_income", "choice": b.pop.questions[j].options[k - 1]} for f in NEAR])
    pa = np.array([d.p_eligible for d in fast.decisions()])
    pb = np.array([d.p_eligible for d in ref.s.decisions()])
    assert np.abs(pa - pb).max() < 1e-6 and np.array_equal(before, it.s.w)  # and the original is untouched


def test_explain_names_the_answer_that_changes_a_result(app):
    step = call(app, "POST", "/v1/step", {"facts": NEAR})["json"]
    r = call(app, "POST", "/v1/explain", {"state": step["state"]})
    assert r["status"] == 200 and r["json"]["state"] == step["state"] and r["json"]["note"]
    rows = {x["id"]: x for x in r["json"]["near_the_line"]}
    assert r["json"]["answers_tried"] >= 6
    # a single adult on $22,800 a year: one income bracket decides SNAP and Medicaid
    for pid in ("snap", "medicaid"):
        assert any(c["question"] == "earned_income" and c["kind"] == "amount" for c in rows[pid]["would_change"]), pid
    for row in rows.values():
        for c in row["would_change"]:
            assert c["tier"] != row["tier"] and c["question"] != "state"
            assert abs(c["probability"] - (row["probability"] + c["probability_change"])) <= 0.011


def test_explain_over_mcp(app):
    start = rpc(app, {"jsonrpc": "2.0", "id": 0, "method": "tools/call", "params": {"name": "start_intake"}})
    first = start["json"]["result"]["structuredContent"]["state"]
    st = rpc(app, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "record_facts", "arguments": {"state": first, "facts": NEAR}}})
    state = st["json"]["result"]["structuredContent"]["state"]
    r = rpc(app, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "what_would_change", "arguments": {"state": state}}})
    out = r["json"]["result"]["structuredContent"]
    assert not r["json"]["result"]["isError"] and out["near_the_line"] and "not advice" in out["note"]


# ------------------------------------------------------------------------------------------ the cliff view
def test_cliff_covers_every_bracket_and_marks_the_current_one(app):
    step = call(app, "POST", "/v1/step", {"facts": NEAR})["json"]
    r = call(app, "POST", "/v1/cliff", {"state": step["state"]})
    assert r["status"] == 200 and r["json"]["question"] == "earned_income" and r["json"]["unit"] == "dollars a year"
    opts = r["json"]["options"]
    assert len(opts) == 11 and sum(o["current"] for o in opts) == 1
    assert [o["option"] for o in opts if o["current"]] == ["$20,000 to $30,000"]  # $1,900 a month
    for o in opts:
        assert o["expected_value"] == pytest.approx(sum(o["by_group"].values()), abs=30)
        assert len(o["programmes"]) == 15
    # help falls away as earnings rise: the lowest brackets carry more than the highest
    assert opts[1]["expected_value"] > opts[-1]["expected_value"]


def test_cliff_refuses_a_question_that_is_not_an_amount(app):
    step = call(app, "POST", "/v1/step", {"facts": NEAR})["json"]
    assert call(app, "POST", "/v1/cliff", {"state": step["state"], "question": "married"})["status"] == 400
    assert call(app, "POST", "/v1/cliff", {"state": step["state"], "question": "nope"})["status"] == 400
    assert call(app, "POST", "/v1/cliff", {"state": step["state"], "question": "housing_cost"})["status"] == 200


# ------------------------------------------------------------------------------------------ caseload triage
def test_batch_ranks_a_caseload(app):
    recs = [{"id": "rich", "facts": [{"question": "state", "choice": "CA"}, {"question": "household_size", "choice": "1"},
                                     {"question": "earned_income", "amount": {"value": 95000, "period": "year"}}]},
            {"id": "family", "facts": [{"question": "state", "choice": "TX"}, {"question": "household_size", "choice": "4"},
                                       {"question": "children", "choice": "2"},
                                       {"question": "earned_income", "amount": {"value": 24000, "period": "year"}}]},
            {"id": "typo", "facts": [{"question": "state", "choice": "Texas!"}]}]
    r = call(app, "POST", "/v1/batch", {"records": recs})
    assert r["status"] == 200
    out = {x["id"]: x for x in r["json"]["results"]}
    assert r["json"]["call_order"][0] == "family" and out["family"]["expected_value"] > out["rich"]["expected_value"]
    assert out["family"]["recommend"] and out["family"]["state"].startswith("di1;")
    assert out["typo"]["errors"][0]["question"] == "state"  # a bad value is reported, the row still comes back
    assert "not an eligibility decision" in r["json"]["note"]


def test_batch_limits(app):
    assert call(app, "POST", "/v1/batch", {"records": []})["status"] == 400
    assert call(app, "POST", "/v1/batch", {"records": [{"id": k, "facts": []} for k in range(51)]})["status"] == 400
    one = call(app, "POST", "/v1/batch", {"records": [{"id": "x", "facts": "nope"}]})["json"]["results"][0]
    assert "error" in one


# ------------------------------------------------------------------------------------------ application plan
def test_plan_orders_and_links_programmes(app):
    facts = [{"question": "state", "choice": "NY"}, {"question": "household_size", "choice": "4"},
             {"question": "children", "choice": "2"}, {"question": "child_under5", "choice": "yes"},
             {"question": "earned_income", "amount": {"value": 21000, "period": "year"}},
             {"question": "tenure", "choice": "rent"}]
    st = call(app, "POST", "/v1/step", {"facts": facts})["json"]["state"]
    recs = {r["id"] for r in call(app, "POST", "/v1/assessment", {"state": st})["json"]["recommend"]}
    r = call(app, "POST", "/v1/plan", {"state": st})
    assert r["status"] == 200
    steps = r["json"]["steps"]
    assert [s["number"] for s in steps] == list(range(1, len(steps) + 1))
    order = {"now": 0, "next": 1, "tax": 2}
    assert [order[s["step"]] for s in steps] == sorted(order[s["step"]] for s in steps)
    planned = [p["id"] for s in steps for p in s["programmes"]]
    assert set(planned) == recs and len(planned) == len(set(planned))  # every recommendation once, nothing else
    for s in steps:
        for p in s["programmes"]:
            for link in p["links"]:
                assert set(link["from"]) <= recs and p["id"] not in link["from"]
                assert "CFR" in link["source"] or "U.S.C." in link["source"]
    wic = [p for s in steps for p in s["programmes"] if p["id"] == "wic"]
    if wic and "snap" in recs:
        assert any(l["kind"] == "income_met" and "snap" in l["from"] for l in wic[0]["links"])


def test_plan_is_empty_when_nothing_is_recommended(app):
    st = call(app, "POST", "/v1/step", {"facts": [{"question": "state", "choice": "CA"},
                                                   {"question": "household_size", "choice": "1"},
                                                   {"question": "earned_income", "amount": {"value": 250000, "period": "year"}}]})["json"]["state"]
    assert call(app, "POST", "/v1/plan", {"state": st})["json"]["steps"] == []
    assert call(app, "POST", "/v1/plan", {"state": "di1;#bad"})["status"] == 400


def test_mcp_assessment_carries_the_plan(app):
    from decisionintake.mcp_server import StatelessServer
    srv = StatelessServer(app.b, app.settings)
    st = srv.call("start_intake", {})["state"]
    st = srv.call("record_facts", {"state": st, "facts": [{"question": "state", "choice": "TX"}, {"question": "household_size", "choice": "4"},
                                             {"question": "children", "choice": "2"},
                                             {"question": "earned_income", "amount": {"value": 24000, "period": "year"}}]})["state"]
    out = srv.call("get_assessment", {"state": st})
    planned = {p["id"] for s in out["plan"] for p in s["programmes"]}
    assert planned == {r["id"] for r in out["recommend"]}


def test_plan_with_programmes_already_received(app):
    kids = [{"question": "state", "choice": "OH"}, {"question": "children", "choice": "2"},
            {"question": "child_under5", "choice": "no"},
            {"question": "earned_income", "amount": {"value": 60000, "period": "year"}}]
    st = call(app, "POST", "/v1/step", {"facts": kids})["json"]["state"]
    base = call(app, "POST", "/v1/plan", {"state": st})["json"]
    have = call(app, "POST", "/v1/plan", {"state": st, "already": ["snap"]})["json"]
    assert have["already"] == [{"id": "snap", "programme": have["already"][0]["programme"]}]
    planned = {p["id"]: p for s in have["steps"] for p in s["programmes"]}
    assert "snap" not in planned
    meals = planned.get("school_meals")  # SNAP makes school-age children eligible for free meals (7 CFR 245.2)
    assert meals and any("snap" in l["already"] for l in meals["links"])
    if "school_meals" not in {p["id"] for s in base["steps"] for p in s["programmes"]}:
        assert meals["added_by_rule"] and meals["tier"] == "worth checking"
    for pid in ("wic", "head_start", "early_head_start"):  # no child under 5: the rule never adds these
        assert not planned.get(pid, {}).get("added_by_rule")
    st0 = call(app, "POST", "/v1/step", {"facts": [{"question": "state", "choice": "OH"},
                                                    {"question": "children", "choice": "none"}]})["json"]["state"]
    alone = call(app, "POST", "/v1/plan", {"state": st0, "already": ["snap", "medicaid"]})["json"]
    assert not [p for s in alone["steps"] for p in s["programmes"] if p.get("added_by_rule")]
    assert call(app, "POST", "/v1/plan", {"state": st, "already": ["food stamps"]})["status"] == 400
    assert call(app, "POST", "/v1/plan", {"state": st, "already": "snap"})["status"] == 400


# ------------------------------------------------------------------------------------------ embedding (D47)
def test_embed_serves_the_screener_and_the_loader():
    from decisionintake.web import App
    app = App(bundle_path=str(SHIPPED), log=None, static_dir=str(Path(__file__).resolve().parents[1] / "public"))
    page = call(app, "GET", "/embed")
    assert page["status"] == 200 and b"const EMBED = " in page["body"]
    js = call(app, "GET", "/embed.js")
    assert js["status"] == 200 and b"decisionintake:height" in js["body"] and b"sandbox" in js["body"]
    assert b"allow-forms" in js["body"]  # without it the question form's submit event never fires (review, D50)


def test_only_embed_may_be_framed():
    """vercel.json: frame-ancestors 'none' on every path but /embed, which https pages may frame."""
    import re
    cfg = json.loads((Path(__file__).resolve().parents[1] / "vercel.json").read_text(encoding="utf-8"))
    def csp_for(path):
        hits = []
        for rule in cfg["headers"]:
            pattern = "^" + re.sub(r"\(\?!embed\$\)", "(?!embed$)", rule["source"]) + "$"
            if re.match(pattern, path):
                hits += [h["value"] for h in rule["headers"] if h["key"] == "Content-Security-Policy"]
        return hits
    for path in ("/", "/docs.html", "/caseload.html", "/walkthrough/", "/embed.js"):
        assert csp_for(path) and all("frame-ancestors 'none'" in v for v in csp_for(path)), path
    assert len(csp_for("/embed")) == 1 and "frame-ancestors https:" in csp_for("/embed")[0]
    assert {"source": "/embed", "destination": "/index.html"} in cfg["rewrites"]


def test_batch_with_programmes_already_received(app):
    fam = [{"question": "state", "choice": "MI"}, {"question": "household_size", "choice": "3"},
           {"question": "children", "choice": "1"}, {"question": "child_under5", "choice": "yes"},
           {"question": "earned_income", "amount": {"value": 54000, "period": "year"}}]
    r = call(app, "POST", "/v1/batch", {"records": [{"id": "plain", "facts": fam},
                                                    {"id": "snap", "facts": fam, "already": ["snap"]},
                                                    {"id": "wic", "facts": fam, "already": ["wic"]},
                                                    {"id": "typo", "facts": fam, "already": ["food stamps"]}]})["json"]
    out = {x["id"]: x for x in r["results"]}
    assert "opened_by_rule" not in out["plain"]
    assert out["snap"]["already"] == ["snap"]
    for o in out["snap"]["opened_by_rule"]:  # each opened programme names what opened it and the rule
        assert o["because"] == ["snap"] and ("CFR" in o["rule"] or "U.S.C." in o["rule"])
        assert o["id"] not in {p["id"] for p in out["snap"]["recommend"]}
    if any(p["id"] == "wic" for p in out["plain"]["recommend"]):  # what they already get leaves the value
        assert out["wic"]["expected_value"] < out["plain"]["expected_value"]
        assert "wic" not in {p["id"] for p in out["wic"]["recommend"]}
    assert "error" in out["typo"] and "typo" not in r["call_order"]


# ------------------------------------------------------------------------------- review fixes (D50)
def _plan_ids(app, facts, already):
    st = call(app, "POST", "/v1/step", {"facts": facts})["json"]["state"]
    p = call(app, "POST", "/v1/plan", {"state": st, "already": already})["json"]
    return {x["id"]: x for s in p["steps"] for x in s["programmes"]}


def test_plan_categories_follow_the_answers(app):
    only_baby = [{"question": "state", "choice": "OH"}, {"question": "children", "choice": "1"},
                 {"question": "child_under5", "choice": "yes"}]
    assert not _plan_ids(app, only_baby, ["snap"]).get("school_meals", {}).get("added_by_rule")  # not school age
    older = [{"question": "state", "choice": "OH"}, {"question": "children", "choice": "2"},
             {"question": "child_under5", "choice": "yes"}]
    pregnant = [{"question": "state", "choice": "OH"}, {"question": "children", "choice": "none"},
                {"question": "infant_or_pregnant", "choice": "yes"}]
    plan = _plan_ids(app, pregnant, ["snap"])
    assert "wic" in plan and "school_meals" not in plan and "head_start" not in plan  # WIC serves pregnancy
    not_pregnant = pregnant[:2] + [{"question": "infant_or_pregnant", "choice": "no"}]
    assert not any(x.get("added_by_rule") for x in _plan_ids(app, not_pregnant, ["snap"]).values())
    assert "school_meals" in _plan_ids(app, older + [{"question": "earned_income", "amount": {"value": 90000, "period": "year"}}], ["snap"])


def test_triage_question_ignores_what_they_already_get(app):
    fam = [{"question": "state", "choice": "TX"}, {"question": "household_size", "choice": "3"},
           {"question": "children", "choice": "2"}]
    r = call(app, "POST", "/v1/batch", {"records": [{"id": "a", "facts": fam},
                                                    {"id": "b", "facts": fam, "already": ["snap", "medicaid", "chip"]}]})
    a, b = r["json"]["results"]
    assert b["expected_mistake_cost"] < a["expected_mistake_cost"]


def test_numbers_json_cannot_read_are_a_400(app):
    for raw in (b'{"state": ' + b"9" * 5000 + b"}", b'{"records": [{"id": NaN, "facts": []}]}',
                b'{"facts": [{"question": "children", "choice": Infinity}]}'):
        for path in ("/v1/plan", "/v1/batch", "/v1/step"):
            assert call(app, "POST", path, raw=raw)["status"] == 400, (path, raw[:30])
    assert call(app, "POST", "/v1/batch", {"records": [{"id": k, "facts": []} for k in range(21)]})["status"] == 400


# ------------------------------------------------------------------------------- receipt as evidence (D51)
def test_receipt_round_trips_and_is_off_by_default(b_):
    from decisionintake.service import Intake, RECEIPT_EPS
    from decisionintake.state import StateError
    import numpy as np
    facts = [{"question": "state", "choice": "OH"}, {"question": "household_size", "choice": "3"},
             {"question": "children", "choice": "2"}]
    plain, got = Intake(b_), Intake(b_)
    plain.record(facts)
    got.receive(["snap"])
    got.record(facts)
    assert ";gets=snap;" in got.state()
    p = lambda it: np.array([dv.p_eligible for dv in it.s.decisions()])
    assert np.array_equal(p(plain), p(got))  # recorded, but no evidence unless the settings say so
    on = Settings(receipt_evidence={"snap": RECEIPT_EPS["snap"]})
    a, c = Intake(b_, on), Intake(b_, on)
    a.receive(["snap"]); a.record(facts)
    c.record(facts); c.receive(["snap"])
    assert np.allclose(p(a), p(c))  # the order doesn't matter
    snap = b_.pop.decisions.index("snap")
    assert p(a)[snap] > p(plain)[snap]
    back = Intake.from_state(b_, a.state(), on)
    assert back.receives == ["snap"] and np.allclose(p(back), p(a))
    with pytest.raises(Exception):
        Intake(b_).receive(["food stamps"])
    bad = a.state().replace("gets=snap", "gets=lunch")
    body = bad.rpartition(";#")[0]
    from decisionintake.state import _check
    with pytest.raises(StateError):
        Intake.from_state(b_, f"{body};#{_check(body)}", on)  # a well-formed state naming an unknown programme


def test_served_engine_uses_snap_receipt(app):
    facts = [{"question": "state", "choice": "OH"}, {"question": "household_size", "choice": "3"},
             {"question": "children", "choice": "2"}, {"question": "child_under5", "choice": "yes"}]
    plain = call(app, "POST", "/v1/step", {"facts": facts})["json"]
    got = call(app, "POST", "/v1/step", {"facts": facts, "already": ["snap"]})["json"]
    assert ";gets=snap;" in got["state"] and "gets=" not in plain["state"]
    pa = {r["id"]: r["probability"] for r in call(app, "POST", "/v1/assessment", {"state": plain["state"]})["json"]["all_programmes"]}
    pb = {r["id"]: r["probability"] for r in call(app, "POST", "/v1/assessment", {"state": got["state"]})["json"]["all_programmes"]}
    assert pb["snap"] > pa["snap"] and pb["wic"] >= pa["wic"]
    # already given to /v1/assessment replaces the state's list, and an empty list clears it
    cleared = call(app, "POST", "/v1/assessment", {"state": got["state"], "already": []})["json"]
    assert "gets=" not in cleared["state"]
    assert {r["id"]: r["probability"] for r in cleared["all_programmes"]} == pa
    assert call(app, "POST", "/v1/step", {"facts": facts, "already": ["lunch"]})["status"] == 400
    assert call(app, "POST", "/v1/assessment", {"state": got["state"], "already": "snap"})["status"] == 400


def test_connector_takes_already(app):
    from decisionintake.mcp_server import StatelessServer, ToolError
    srv = StatelessServer(app.b, app.settings)
    st = srv.call("start_intake", {})["state"]
    out = srv.call("record_facts", {"state": st, "facts": [{"question": "state", "choice": "OH"}], "already": ["snap"]})
    assert ";gets=snap;" in out["state"]
    later = srv.call("record_facts", {"state": out["state"], "facts": [{"question": "children", "choice": "2"}]})
    assert ";gets=snap;" in later["state"]  # kept when not given again
    with pytest.raises(ToolError):
        srv.call("record_facts", {"state": st, "facts": [], "already": ["lunch"]})


# ------------------------------------------------------------------------------------- where to apply (D53)
def test_apply_links_file_is_well_formed():
    from decisionintake import service
    data = json.loads(Path(service._APPLY_FILE).read_text(encoding="utf-8"))
    keys = set(service.APPLY_KEY.values())
    import re
    assert set(data["sources"]) == keys
    for st, progs in data["links"].items():
        assert re.fullmatch(r"[A-Z]{2}", st)
        for key, link in progs.items():
            assert key in keys
            assert re.match(r"https?://[^\s\"'<>]+$", link["url"])
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", link["checked"])


def test_where_to_apply_follows_the_state(app, monkeypatch):
    from decisionintake import service
    fake = {"NC": {"snap": {"url": "https://nc.example.gov/snap", "title": "SNAP", "checked": "2026-11-25"},
                   "medicaid": {"url": "https://nc.example.gov/medicaid", "title": "Medicaid", "checked": "2026-11-25"}}}
    monkeypatch.setattr(service, "_apply_links", fake)
    base = [{"question": "household_size", "choice": "3"}, {"question": "children", "choice": "2"},
            {"question": "child_under5", "choice": "yes"},
            {"question": "earned_income", "amount": {"value": 18000, "period": "year"}}]
    # no state answered: no state links anywhere
    st = call(app, "POST", "/v1/step", {"facts": base})["json"]["state"]
    a = call(app, "POST", "/v1/assessment", {"state": st})["json"]
    assert not any("apply" in r for r in a["all_programmes"])
    # North Carolina: SNAP and Medicaid carry NC's pages, CHIP shares Medicaid's, the rest have none
    st = call(app, "POST", "/v1/step", {"facts": base + [{"question": "state", "choice": "NC"}]})["json"]["state"]
    a = call(app, "POST", "/v1/assessment", {"state": st})["json"]
    rows = {r["id"]: r for r in a["all_programmes"]}
    for r in a["recommend"] + a["unsure"]:
        want = fake["NC"].get(service.APPLY_KEY.get(r["id"]))
        assert r.get("apply") == ({"url": want["url"], "state": "NC", "checked": "2026-11-25"} if want else None)
    for pid, r in rows.items():  # never on a programme the engine advises against
        if r["tier"] in ("unlikely", "not recommended"):
            assert "apply" not in r
    assert "snap" in {r["id"] for r in a["recommend"]}  # the case exercises a state link
    plan = call(app, "POST", "/v1/plan", {"state": st})["json"]
    for s in plan["steps"]:
        for p in s["programmes"]:
            key = service.APPLY_KEY.get(p["id"])
            assert (p.get("apply") or {}).get("url") == (fake["NC"].get(key) or {}).get("url")
    # a state without links: none
    st = call(app, "POST", "/v1/step", {"facts": base + [{"question": "state", "choice": "VT"}]})["json"]["state"]
    assert not any("apply" in r for r in call(app, "POST", "/v1/assessment", {"state": st})["json"]["all_programmes"])


def test_screener_uses_state_links_safely():
    page = (Path(__file__).parent.parent / "public" / "index.html").read_text(encoding="utf-8")
    assert "function applyFor(r)" in page and r"/^https?:\/\//.test(w.url)" in page
    assert '<a href="${esc(href)}"' in page and 'href="${esc(a.href)}"' in page  # server-sent URLs are escaped
    for lang_key in ("how_in", "state_site"):
        assert page.count(f"    {lang_key}:") == 2  # English and Spanish
