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
    assert set(hosted) == {"start_intake", "record_facts", "get_assessment"}
    assert all(t["annotations"]["readOnlyHint"] and t["title"] for t in hosted.values())
    local = {t["name"]: t["annotations"]["readOnlyHint"] for t in Server(b_, Settings()).TOOLS}
    assert local == {"start_intake": False, "record_facts": False, "next_step": True, "get_assessment": True}


# ------------------------------------------------------------------------------------------ Origin and CORS
LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


@pytest.mark.parametrize("origin, ok", [
    (None, True), ("http://localhost:6274", True), ("http://127.0.0.1:5173", True),
    ("https://decision-intake.vercel.app", True), ("https://evil.example", False), ("null", False),
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
def test_one_shape_per_field(app):
    first = call(app, "POST", "/v1/step", {"facts": []})["json"]
    assert all("id" in s for s in first["next"]["why"]["mostly_settles"])
    a = call(app, "POST", "/v1/assessment", {"state": first["state"]})["json"]
    for row in a["all_programmes"]:
        v = row["value_if_eligible_per_year"]
        assert isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)
