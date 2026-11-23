"""End to end with a real LLM agent: Claude interviews simulated households over MCP, with and without the engine.

Each household is a held-out test household from D28's fresh sample. It speaks through templated answers
generated from its own record (the language experiments' generator and seeds: exact figures, round
numbers, monthly figures for yearly questions, ranges, vague words, implied facts, "I'm not sure"), and
opens with a few facts unprompted. Two arms, same model, same households, same words:

  engine  the agent has DecisionIntake's tools (start_intake, record_facts, next_step, get_assessment)
          and follows them; it reads the person's words into typed facts, the engine decides
  alone   the agent decides itself what to ask, when to stop and what to recommend, from its own
          knowledge of the programmes; it gets the same question catalogue and the same costs

Both talk to the person only through ask_person (message + which catalogue questions it asks about,
so the simulated person can answer) and end with submit_recommendations. Scored against PolicyEngine's
result for the household: a missed programme costs its yearly value (at least $120), a wrong
recommendation $1,000, every question $15.

usage:
  python experiments/agent_e2e.py prep dev|test N                 personas -> results/agent/SET/personas
  python experiments/agent_e2e.py run SET ARM MODEL N PARALLEL     claude -p on the first N personas of SET
  python experiments/agent_e2e.py analyse SET MODEL
  python experiments/agent_e2e.py serve ARM PERSONA TRACE      (the MCP server claude -p launches)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(ROOT, "results", "agent")
BUNDLE = os.path.join(ROOT, "data", "bundle", "benefits_v2.npz")  # the runs used this copy; identical to bundles/
if not os.path.exists(BUNDLE):
    BUNDLE = os.path.join(ROOT, "bundles", "benefits_v2.npz")
FR, QCOST = 1000.0, 15.0
PROGRAMMES = ["snap", "wic", "tanf", "ssi", "medicaid", "chip", "eitc", "ctc", "aca_ptc", "school_meals"]
OPENER_QS = ["state", "household_size", "children", "senior", "married", "earned_income", "tenure", "disabled"]


# ------------------------------------------------------------------------------------------- personas
DEV_SIZE = 100  # households reserved for developing the agent integration; the test set never overlaps them


def fresh_positions(N: int, run_v2) -> np.ndarray:
    """Test-split positions no earlier experiment touched: not D21-D27's (seeds 2026-2028), not D28's 2,000."""
    used = set(np.random.default_rng(2026).choice(N, size=min(1200, N), replace=False).tolist())
    for seed, size in [(2027, 600), (2028, 300)]:
        fresh = np.array([i for i in range(N) if i not in used])
        used |= set(np.random.default_rng(seed).choice(fresh, size=size, replace=False).tolist())
    used |= set(run_v2.sample(2000).tolist())
    return np.array([i for i in range(N) if i not in used])


def prep(which: str, n: int):
    """which = "dev" (the 100 households for building the integration) or "test" (never seen while building)."""
    from benefits.population import LABELS, load_table
    from benefits.utterances import answer, styles_for, yesno_answer
    from common import splits
    import run_v2

    t = load_table()
    t["_other_income"] = np.maximum(t.market_income - t.earned_income - t.social_security, 0)
    pop, loss, meta, tr, va, test = splits()
    run_v2.G["test"] = pop.subset(test)
    fresh = fresh_positions(len(test), run_v2)
    dev = np.random.default_rng(2032).permutation(fresh)[:DEV_SIZE]
    rest = np.setdiff1d(fresh, dev)
    pos = (dev if which == "dev" else np.random.default_rng(2033).permutation(rest))[:n]
    idx = test[pos]  # positions in the test split -> rows of the full table
    out = os.path.join(OUT, which, "personas")
    print(f"{len(fresh)} fresh test households; {which}: {len(pos)}")
    os.makedirs(out, exist_ok=True)
    for k, i in enumerate(idx):
        i = int(i)
        utt = {}
        for j, q in enumerate(pop.questions):
            r = np.random.default_rng(i * 131 + j)  # the language experiments' seeds
            style = str(r.choice(styles_for(q.id)))
            utt[q.id] = {"style": style, "text": answer(q.id, t.iloc[i], int(pop.answers[i, j]), style, r)}
        r = np.random.default_rng(i * 7 + 3)
        said = [OPENER_QS[x] for x in sorted(r.choice(len(OPENER_QS), size=3, replace=False))]
        # the opener says each fact in words that stand on their own ("None." would be ambiguous there)
        parts = []
        for qid in said:
            j = pop.qindex(qid)
            opt, rr = int(pop.answers[i, j]), np.random.default_rng(i)
            if qid in ("senior", "married", "disabled"):
                text = yesno_answer(qid, bool(opt), "indirect", rr).removeprefix("Yes, ").removeprefix("No, ")
                text = text[0].upper() + text[1:]
            elif qid == "children":
                text = "We don't have kids." if opt == 0 else answer(qid, t.iloc[i], opt, "exact", rr)
            elif qid == "earned_income":
                style = utt[qid]["style"] if utt[qid]["style"] in ("exact", "approx", "monthly", "range") else "exact"
                said_ = answer(qid, t.iloc[i], opt, style, rr)
                text = "Nobody's working right now." if opt == 0 else "From work we make " + said_[0].lower() + said_[1:]
            elif qid == "state":
                text = answer(qid, t.iloc[i], opt, "indirect", rr)  # "We're in Ohio."
            elif utt[qid]["style"] == "dont_know":
                text = answer(qid, t.iloc[i], opt, "exact", rr)
            else:
                text = utt[qid]["text"]
            parts.append(text)
        eligible = pop.eligible[i]
        persona = {
            "k": k, "i": i, "test_pos": int(pos[k]), "weight": float(pop.weight[i]), "volunteered": said,
            "opener": "Hi, I'd like to find out what help we might qualify for. " + " ".join(parts),
            "utterances": utt,
            "truth": {p: (float(pop.value[i, d]) if eligible[d] else 0.0) for d, p in enumerate(pop.decisions)},
            "labels": LABELS,
        }
        with open(os.path.join(out, f"{k:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=1)
    print(f"wrote {len(pos)} personas to {out}")


# ------------------------------------------------------------------------------------------- the MCP server
def serve(arm: str, persona_path: str, trace_path: str):
    from decisionintake import bundle
    from decisionintake.mcp_server import (TOOLS, TOOLS_STATELESS, Server, Settings, StatelessServer, ToolError,
                                           serve as serve_stdio)

    persona = json.load(open(persona_path, encoding="utf-8"))
    trace = open(trace_path, "a", encoding="utf-8")
    asked: list[str] = []

    def log(kind, **kw):
        trace.write(json.dumps({"t": time.time(), "kind": kind, **kw}) + "\n")
        trace.flush()

    labels = persona["labels"]
    by_name = {p: p for p in PROGRAMMES} | {v.lower(): k for k, v in labels.items()}
    person_tools = [
        {"name": "ask_person",
         "description": "Send a message to the person and get their reply. List in `about` the ids of the catalogue "
                        "questions your message asks (the person answers those).",
         "inputSchema": {"type": "object", "properties": {
             "message": {"type": "string"},
             "about": {"type": "array", "items": {"type": "string"}}}, "required": ["message", "about"]}},
        {"name": "submit_recommendations",
         "description": "End the screening: the programmes the household should apply for (ids or names), and whether "
                        "the case should go to a caseworker. Call once, at the end.",
         "inputSchema": {"type": "object", "properties": {
             "programmes": {"type": "array", "items": {"type": "string"}},
             "hand_off": {"type": "boolean"}, "summary": {"type": "string"}}, "required": ["programmes", "hand_off"]}},
    ]

    last_state = [None]  # stateless arm: the state the last result handed back, to spot stale or mangled copies
    chain = [None]  # the latest state from start_intake or record_facts: a record_facts from anything else drops facts

    class Sim(StatelessServer if arm == "stateless" else Server):
        def call(self, name, args):
            if name == "ask_person":
                ids = [q for q in args.get("about") or [] if q in persona["utterances"]]
                for q in ids:
                    if q not in asked:
                        asked.append(q)
                reply = " ".join(persona["utterances"][q]["text"] for q in ids) or \
                    "Sorry, I'm not sure what you're asking. What do you need to know?"
                log("ask", message=args.get("message"), about=args.get("about"), reply=reply)
                return {"reply": reply}
            if name == "submit_recommendations":
                recs, unknown = [], []
                for p in args.get("programmes") or []:
                    key = by_name.get(str(p).strip().lower())
                    (recs if key else unknown).append(key or p)
                log("submit", programmes=recs, unknown=unknown, hand_off=bool(args.get("hand_off")),
                    summary=args.get("summary"), asked=asked)
                return {"ok": True, "recorded": recs, "unrecognised": unknown}
            if arm in ("alone", "remote") or (arm == "predictor" and name == "next_step"):
                raise ToolError(f"unknown tool {name!r}")
            stale = arm == "stateless" and name != "start_intake" and args.get("state") != last_state[0]
            dropped = arm == "stateless" and name == "record_facts" and args.get("state") != chain[0]
            try:
                out = super().call(name, args)
            except ToolError as e:
                log("engine", tool=name, args=args, error=str(e), stale=stale)
                raise
            if arm == "stateless":
                last_state[0] = out.get("state")
                if name in ("start_intake", "record_facts"):
                    chain[0] = out.get("state")
                out_log = {"stale": stale, "dropped": dropped}
            else:
                out_log = {}
            if arm == "predictor":  # same model and decision rule, but no advice on what to ask or when to stop
                it = self.sessions[out["session_id"] if name == "start_intake" else args["session_id"]]
                if name == "start_intake":
                    out = {k: v for k, v in out.items() if k != "next"} | {"assessment": it.assessment()}
                elif name == "record_facts":
                    out = {"readings": out["readings"], "assessment": it.assessment()}
            log("engine", tool=name, args=args, result=out, **out_log)
            return out

        def handle(self, msg):
            if msg.get("method") == "tools/list" and msg.get("id") is not None:
                tools = person_tools + {"alone": [], "remote": [], "engine": TOOLS, "stateless": TOOLS_STATELESS}.get(arm, PREDICTOR_TOOLS)
                return {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": tools}}
            out = super().handle(msg)
            if msg.get("method") == "initialize" and out and "result" in out:
                out["result"]["serverInfo"]["name"] = "screening"
                if arm not in ("engine", "stateless"):
                    out["result"].pop("instructions", None)
            return out

    PREDICTOR_TOOLS = [
        {"name": "start_intake", "description": "Open a new intake with the eligibility model. Returns a session_id, the "
         "programmes, every question, and the model's assessment before any answer.",
         "inputSchema": TOOLS[0]["inputSchema"]},
        {"name": "record_facts", "description": "Record what the person said as typed facts (any questions; a fact for an "
         "answered question replaces it). Returns how each fact was read and the model's updated assessment: every "
         "programme's probability, tier and value range, the recommend list, and the expected cost of mistakes left.",
         "inputSchema": TOOLS[1]["inputSchema"]},
        {"name": "get_assessment", "description": TOOLS[3]["description"], "inputSchema": TOOLS[3]["inputSchema"]},
    ]
    b = bundle.load(BUNDLE) if arm in ("engine", "predictor", "stateless") else None
    serve_stdio(Sim(b, Settings(question_cost=QCOST)))


# ------------------------------------------------------------------------------------------- running claude -p
COMMON = """You are a benefits-screening assistant. You talk with a person ONLY through the ask_person tool: each \
message you send goes to them and their reply comes back. Find out which of these ten US programmes the household \
should apply for: {programmes}.

What matters to the person: recommending a programme they can't get costs them about $1,000 (time, paperwork, a \
refusal); missing one they qualify for costs them its full yearly value; every question costs them about $15 of time. \
So ask what you need and no more. Questions you can ask about (use these ids in ask_person's `about`):
{catalogue}

When you're done, call submit_recommendations exactly once with the ids of the programmes the household should \
apply for, and hand_off true if a caseworker should also look at the case. Even when you hand off, list what you \
would recommend now: the person acts on that list. Then reply with one short line. The person has already said: \
"{opener}"
"""
ENGINE = """
You have the DecisionIntake engine. Use it for every decision:
1. Call start_intake, then record_facts with every fact in what the person has already said.
2. Loop: call ask_person with the engine's question (plain friendly words; `about` = its id), then record_facts with \
EVERY fact in the reply (amounts exactly as said, with period and approximate; dont_know when they don't know). \
record_facts returns the next step.
3. When the step is "finish" or "hand_off", call submit_recommendations with the `id` of every item in its \
`recommend` list, the "worth checking" ones included, and hand_off true only if the action is hand_off.
Do no eligibility reasoning of your own and ask nothing the engine didn't ask for.
"""
PREDICTOR = """
You have DecisionIntake's eligibility model, trained on PolicyEngine's rules. It judges eligibility; you decide \
what to ask and when to stop.
1. Call start_intake, then record_facts with every fact in what the person has already said.
2. Choose the next question yourself, ask it with ask_person (`about` = its id), then call record_facts with EVERY \
fact in the reply (amounts exactly as said, with period and approximate; dont_know when they don't know). \
record_facts returns the model's updated assessment: each programme's probability, tier and yearly value range, \
the `recommend` list, and the expected cost of mistakes still left.
3. Stop when you judge more questions aren't worth their cost. Then call submit_recommendations with the `id` of \
every item in the assessment's `recommend` list, the "worth checking" ones included, and hand_off true if the \
expected cost of mistakes left is above $2,500.
Do no eligibility reasoning of your own: the assessment is the judgement.
"""
ALONE = """
Decide yourself what to ask, when to stop and what to recommend, using your own knowledge of the 2026 rules for \
these programmes and the person's state.
"""


def catalogue_text(persona):
    from decisionintake import bundle

    b = bundle.load(BUNDLE)
    lines = []
    for q in b.pop.questions:
        lines.append(f"- {q.id}: {q.text} ({'; '.join(q.options) if q.id != 'state' else 'US state'})")
    return "\n".join(lines)


def run_one(arm: str, model: str, persona_path: str, out_dir: str) -> dict:
    persona = json.load(open(persona_path, encoding="utf-8"))
    k = persona["k"]
    trace = os.path.join(out_dir, f"{k:03d}.trace.jsonl")
    if os.path.exists(trace):
        os.remove(trace)
    local = {"command": sys.executable, "args": [os.path.abspath(__file__), "serve", arm, os.path.abspath(persona_path), trace]}
    if arm == "remote":  # the engine over HTTP (a deployed service); this process only plays the person
        remote = {"type": "http", "url": os.environ.get("DI_REMOTE_URL", "http://127.0.0.1:8766/mcp")}
        cfg = {"mcpServers": {"screening": local, "di": remote}}
    else:
        cfg = {"mcpServers": {"di": local}}
    cfg_path = os.path.join(out_dir, f"{k:03d}.mcp.json")
    json.dump(cfg, open(cfg_path, "w"))
    progs = ", ".join(f"{p} ({persona['labels'][p]})" for p in PROGRAMMES)
    system = COMMON.format(programmes=progs, catalogue=catalogue_text(persona), opener=persona["opener"])
    system += {"engine": ENGINE, "stateless": ENGINE, "remote": ENGINE, "alone": ALONE, "predictor": PREDICTOR}[arm]
    cmd = ["claude", "-p", "Please start the screening.", "--model", model, "--system-prompt", system,
           "--strict-mcp-config", "--mcp-config", cfg_path, "--tools", "", "--allowedTools",
           "mcp__di,mcp__screening" if arm == "remote" else "mcp__di",
           "--setting-sources", "", "--no-session-persistence", "--output-format", "stream-json", "--verbose",
           "--max-turns", "80"]
    t0 = time.time()
    with open(os.path.join(out_dir, f"{k:03d}.stream.jsonl"), "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.PIPE, cwd=out_dir, timeout=1800,
                           env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    return {"k": k, "seconds": round(time.time() - t0, 1), "returncode": p.returncode,
            "stderr": p.stderr.decode(errors="replace")[-500:]}


def run(which: str, arm: str, model: str, n: int, par: int):
    out_dir = os.path.join(OUT, which, f"{model}_{arm}")
    os.makedirs(out_dir, exist_ok=True)
    pdir = os.path.join(OUT, which, "personas")
    personas = sorted(os.listdir(pdir))[:n]
    todo = []
    for name in personas:
        k = int(name[:3])
        tr = os.path.join(out_dir, f"{k:03d}.trace.jsonl")
        if os.path.exists(tr) and any(json.loads(l)["kind"] == "submit" for l in open(tr, encoding="utf-8")):
            continue  # finished earlier
        todo.append(os.path.join(pdir, name))
    print(f"{arm} / {model}: {len(todo)} of {len(personas)} to run", flush=True)
    with ThreadPoolExecutor(par) as ex:
        for r in ex.map(lambda p: run_one(arm, model, p, out_dir), todo):
            print(json.dumps(r), flush=True)


# ------------------------------------------------------------------------------------------- scoring
def score_one(persona, trace_path):
    events = [json.loads(l) for l in open(trace_path, encoding="utf-8")] if os.path.exists(trace_path) else []
    sub = next((e for e in events if e["kind"] == "submit"), None)
    if sub is None:  # ended without a submission (pre-registered: rerun once, then scored as recommending nothing)
        if not events:
            return None
        asked = []
        for e in events:
            for q in e.get("about") or [] if e["kind"] == "ask" else []:
                if q in persona["utterances"] and q not in asked:
                    asked.append(q)
        sub = {"programmes": [], "hand_off": False, "asked": asked, "failed": True}
    recs = set(sub["programmes"])
    final = None  # the engine's own last verdict, to see whether the agent passed it on faithfully
    for e in events:
        if e["kind"] == "engine" and "result" in e:
            step = e["result"].get("next", e["result"])
            if "assessment" in e["result"]:  # predictor arm: the latest assessment is the model's verdict
                final = e["result"]["assessment"]
            elif step.get("action") in ("finish", "hand_off"):
                final = step
            elif step.get("action") == "ask":
                final = None
    inv = {v: k for k, v in persona["labels"].items()}
    engine_list = None if final is None else sorted(r.get("id") or inv[r["programme"]] for r in final["recommend"])
    miss = sum(max(v, 120.0) for p, v in persona["truth"].items() if v > 0 and p not in recs)
    false = sum(FR for p, v in persona["truth"].items() if v <= 0 and p in recs)
    asked = [q for q in sub["asked"] if q not in persona["volunteered"]]
    return {"k": persona["k"], "loss": miss + false, "missed": miss, "false_recs": false, "questions": len(asked),
            "messages": sum(e["kind"] == "ask" for e in events), "hand_off": sub["hand_off"],
            "recs": sorted(recs), "engine_calls": sum(e["kind"] == "engine" for e in events), "failed": bool(sub.get("failed")),
            "engine_list": engine_list, "deviates": engine_list is not None and engine_list != sorted(recs),
            "engine_errors": sum(e["kind"] == "engine" and "error" in e for e in events),
            "stale_state": sum(e["kind"] == "engine" and bool(e.get("stale")) for e in events)}


def analyse(which: str, model: str, a: str = "engine", b: str = "alone"):
    rows, weight, truth = {}, {}, {}
    pdir = os.path.join(OUT, which, "personas")
    for arm in [a, b]:
        d = os.path.join(OUT, which, f"{model}_{arm}")
        for name in sorted(os.listdir(pdir)):
            persona = json.load(open(os.path.join(pdir, name), encoding="utf-8"))
            weight[persona["k"]] = persona["weight"]
            truth[persona["k"]] = {q for q, v in persona["truth"].items() if v > 0}
            s = score_one(persona, os.path.join(d, f"{persona['k']:03d}.trace.jsonl"))
            if s:
                rows.setdefault(persona["k"], {})[arm] = s
    both = sorted(k for k, v in rows.items() if len(v) == 2)
    if not both:
        print("no household finished in both arms yet")
        return
    w = np.array([weight[k] for k in both])
    w = w / w.sum()
    total = {arm: np.array([rows[k][arm]["loss"] + QCOST * rows[k][arm]["questions"] for k in both]) for arm in [a, b]}
    d = total[a] - total[b]
    boots = np.random.default_rng(5).choice(len(d), size=(4000, len(d)))
    print(f"{which} / {model}: {len(both)} households finished in both arms (mean per household; survey-weighted in brackets)")
    for arm in [a, b]:
        r = [rows[k][arm] for k in both]
        m = lambda key: (np.mean([x[key] for x in r]), w @ np.array([x[key] for x in r], float))
        q, l, mi, fr, ho = m("questions"), m("loss"), m("missed"), m("false_recs"), m("hand_off")
        print(f"  {arm:<7} questions {q[0]:4.1f} ({q[1]:4.1f})  mistakes ${l[0]:6,.0f} (${l[1]:6,.0f})"
              f"  missed ${mi[0]:5,.0f}  wrong recs ${fr[0]:5,.0f}  hand-offs {ho[0]:.0%}")
        hit = sum(len(set(rows[k][arm]["recs"]) & truth[k]) for k in both)
        nrec, nel = sum(len(rows[k][arm]["recs"]) for k in both), sum(len(truth[k]) for k in both)
        print(f"  {'':<7} recommendations {nrec}, precision {hit / max(nrec, 1):.0%}, recall {hit / max(nel, 1):.0%} of {nel} eligible programmes")
    out = {}
    for name, ww in [("unweighted", np.full(len(d), 1 / len(d))), ("weighted", w)]:
        est = float(ww @ d)
        bs = np.array([np.average(d[b], weights=ww[b]) for b in boots])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = [est, float(lo), float(hi)]
        print(f"  {a} minus {b}, total cost, {name}: ${est:,.0f} [{lo:,.0f}, {hi:,.0f}]")
    dev_ = [k for k in both if rows[k][a]["deviates"]]
    fails = {arm: sum(rows[k][arm]["failed"] for k in both) for arm in [a, b]}
    print(f"  {a}-arm submissions that differ from the engine's own last list: {len(dev_)}; runs without a submission: {fails}")
    calm = [x for x, k in enumerate(both) if not rows[k][a]["hand_off"] and not rows[k][b]["hand_off"]]
    if calm:
        print(f"  households neither arm handed off ({len(calm)}): {a} minus {b} ${d[calm].mean():,.0f}")
    print(f"  per programme, households where it was missed / wrongly recommended ({a} | {b}):")
    for p in PROGRAMMES:
        cnt = {arm: [0, 0] for arm in [a, b]}
        for k in both:
            truth = json.load(open(os.path.join(pdir, f"{k:03d}.json"), encoding="utf-8"))["truth"][p]
            for arm in cnt:
                rec = p in rows[k][arm]["recs"]
                cnt[arm][0] += truth > 0 and not rec
                cnt[arm][1] += truth <= 0 and rec
        print(f"    {p:<13} missed {cnt[a][0]:3d} | {cnt[b][0]:3d}   wrong {cnt[a][1]:3d} | {cnt[b][1]:3d}")
    name = f"{model}_summary.json" if (a, b) == ("engine", "alone") else f"{model}_{a}_vs_{b}_summary.json"
    json.dump({"set": which, "model": model, "arms": [a, b], "households": both, "rows": {str(k): rows[k] for k in both},
               "diff": out}, open(os.path.join(OUT, which, name), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], int(sys.argv[3]))
    elif cmd == "serve":
        serve(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    elif cmd == "analyse":
        analyse(sys.argv[2], sys.argv[3], *sys.argv[4:6])
