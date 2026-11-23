"""The engine as a product: the bundle round-trips exactly, the typed intake makes the same decisions as the
simulated interviews the results come from, and the MCP server speaks the protocol over a real pipe."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from decisionintake import LossModel, Session, VOIPolicy, bundle
from decisionintake.neural import NeuralOutcome
from decisionintake.service import Intake, Settings
from decisionintake.simulate import UserModel, run_one

from test_toy import toy


def toy_terms(pop):
    """The one-hot wide features as data: one term per question, plus the intercept."""
    sizes = [len(q.options) for q in pop.questions]
    off = np.concatenate([[0], np.cumsum(sizes)]).tolist()
    return [{"qs": [j], "strides": [1], "off": off[j]} for j in range(len(sizes))] + [{"qs": [], "strides": [], "off": off[-1]}]


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    pop, loss = toy()
    q = pop.questions[1]  # income becomes an amount question: monthly ranges with these upper edges
    pop.questions[1] = replace(q, ordered=True, edges=(1000, 2000, 3000, 4000), per="month")
    terms = toy_terms(pop)
    tf = bundle.TermFeatures(terms, terms[-1]["off"] + 1)
    model = NeuralOutcome(tf, tf.n_features, hidden=(32, 16), epochs=60, batch=256, holdout=0).fit(pop)
    path = str(tmp_path_factory.mktemp("b") / "toy.npz")
    bundle.save(path, pop, loss, model, terms, {"child": "Child benefit"}, {"caveat": "toy"})
    return pop, loss, model, path


def test_bundle_round_trips_exactly(made):
    pop, loss, model, path = made
    b = bundle.load(path)
    assert [q.id for q in b.pop.questions] == [q.id for q in pop.questions]
    assert b.pop.questions[1].edges == (1000, 2000, 3000, 4000) and b.pop.questions[1].per == "month"
    p0, v0 = model.predict(pop, pop.answers[:200])
    p1, v1 = b.model.predict(b.pop, b.pop.answers[:200])
    assert np.allclose(p0, p1, atol=1e-6) and np.allclose(v0, v1, atol=1e-3)
    assert b.labels == {"child": "Child benefit", "senior": "senior", "credit": "credit"}


def test_typed_intake_makes_the_same_decisions_as_the_simulated_interview(made):
    _, _, _, path = made
    b = bundle.load(path)
    for i in [0, 7, 19, 42, 101]:
        row = b.pop.answers[i]
        sim = run_one(VOIPolicy(question_cost=15.0, budget=18), b.pop, b.loss, row, UserModel(), np.random.default_rng(0),
                      min_ess=10.0, outcome=b.model)
        it, asked = Intake(b), []
        while (step := it.next_step())["action"] == "ask":
            qid = step["question"]["id"]
            asked.append(qid)
            j = it.qi[qid]
            it.record([{"question": qid, "choice": b.pop.questions[j].options[row[j]]}])
        assert asked == sim.asked
        assert np.array_equal(it.s.recommend_mask(), sim.recommend)


def test_voi_splits_exactly_over_decisions(made):
    _, _, _, path = made
    b = bundle.load(path)
    s = Session(b.pop, b.loss, min_ess=10.0, outcome=b.model)
    s.answer(0, 1)
    for j in range(1, 4):
        assert s.voi_by_decision(j).sum() == pytest.approx(s.voi(j), abs=1e-6)


def test_facts_are_read_in_code_and_corrections_replace(made):
    _, _, _, path = made
    b = bundle.load(path)
    it = Intake(b)
    r = it.record([{"question": "kids", "choice": 2}, {"question": "senior", "choice": True},
                   {"question": "income", "amount": {"value": 450, "period": "week", "approximate": True}},
                   {"question": "pets", "choice": "a parrot"}, {"question": "nope", "choice": "x"}])
    assert r[0]["reading"] == "2+" and r[1]["reading"] == "yes"
    assert "$450 a week" in r[2]["reading"] and "$1,657-$2,242 a month" in r[2]["reading"]  # 450*52/12 = 1,950, +-15%
    assert "error" in r[3] and "no, yes" in r[3]["error"] and "error" in r[4]
    assert set(it.facts) == {"kids", "senior", "income"}
    # a correction replays every fact on a fresh belief: same as having said the right thing first
    it.record([{"question": "kids", "choice": "0"}])
    fresh = Intake(b)
    fresh.record([{"question": "senior", "choice": "yes"},
                  {"question": "income", "amount": {"value": 450, "period": "week", "approximate": True}},
                  {"question": "kids", "choice": "0"}])
    assert np.allclose(it.s.w, fresh.s.w) and it.s.risk() == pytest.approx(fresh.s.risk())
    assert it.history[-1]["facts"] == ["kids"]


def test_finish_or_hand_off_carries_tiered_recommendations(made):
    _, _, _, path = made
    b = bundle.load(path)
    it = Intake(b, Settings(handoff_cost=0.0))  # any doubt left at all goes to a person
    it.record([{"question": "kids", "choice": "1"}, {"question": "income", "amount": {"value": 1500, "period": "month"}},
               {"question": "senior", "choice": "no"}])
    step = it.next_step()
    while step["action"] == "ask":
        it.record([{"question": step["question"]["id"], "dont_know": True}])
        step = it.next_step()
    assert step["action"] == "hand_off" and "caseworker" in step["hand_off_reason"]
    assert {r["tier"] for r in step["recommend"]} <= {"likely", "worth checking"}
    child = next(r for r in step["recommend"] if r["programme"] == "Child benefit")
    assert child["tier"] == "likely" and child["probability"] > 0.9


def rpc(proc, msg):
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()
    if "id" not in msg:
        return None
    return json.loads(proc.stdout.readline())


def test_mcp_server_over_stdio(made):
    _, _, _, path = made
    proc = subprocess.Popen([sys.executable, "-m", "decisionintake.mcp_server", "--bundle", path],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        init = rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        assert init["result"]["protocolVersion"] == "2025-06-18" and "tools" in init["result"]["capabilities"]
        assert "record_facts" in init["result"]["instructions"]
        rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        assert [t["name"] for t in tools] == ["start_intake", "record_facts", "next_step", "get_assessment"]

        def call(i, name, args):
            r = rpc(proc, {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}})["result"]
            if not r["isError"]:
                assert json.loads(r["content"][0]["text"]) == r["structuredContent"]
            return r

        start = call(3, "start_intake", {})["structuredContent"]
        sid = start["session_id"]
        assert len(start["questions"]) == 4 and start["next"]["action"] == "ask"
        step, i = start["next"], 4
        while step["action"] == "ask":
            q = step["question"]
            fact = {"question": q["id"], "amount": {"value": 2500, "period": "month"}} if q["kind"] == "amount" \
                else {"question": q["id"], "choice": q["options"][-1]}
            step = call(i, "record_facts", {"session_id": sid, "facts": [fact]})["structuredContent"]["next"]
            i += 1
        assert step["action"] in ("finish", "hand_off") and "recommend" in step
        full = call(i, "get_assessment", {"session_id": sid})["structuredContent"]
        assert len(full["all_programmes"]) == 3 and full["note"] == "toy"
        bad = call(i + 1, "next_step", {"session_id": "nope"})
        assert bad["isError"] and "start_intake" in bad["content"][0]["text"]
        assert rpc(proc, {"jsonrpc": "2.0", "id": 99, "method": "resources/list"})["error"]["code"] == -32601
        assert rpc(proc, {"jsonrpc": "2.0", "id": 100, "method": "ping"})["result"] == {}
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    assert proc.returncode == 0
