"""D43, the registered test: the expanded engine (fifteen programmes) against v2 plus a rule for the five.

    A  v3: v2's network and the add-on side by side (bundles/benefits_v3.npz), the D32 rules, fifteen programmes,
       its own questions
    B  v2 as served (bundles/benefits_v2.npz) with the D32 rules, its own questions, and for the five added
       programmes a rule on B's final belief (see rule_b)

Both answer from the same households with the same simulated answers (paired draws). Loss: a missed
programme costs its yearly value (at least $120), a wrong recommendation $1,000.

usage: python experiments/run_d43.py dev WORKERS            # 300 validation households, clean: sizes the P2 margin
       python experiments/run_d43.py test WORKERS           # 2,000 held-out households, clean and realistic
       python experiments/run_d43.py {dev,test} 1 analyse
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from decisionintake import VOIPolicy, bundle  # noqa: E402
from decisionintake.core import LossModel, Session  # noqa: E402
from decisionintake.guards import GuardedOutcome, Rules  # noqa: E402
from decisionintake.simulate import UserModel, reported_option  # noqa: E402

QCOST, MIN_ESS, FR, FLOOR = 15.0, 10.0, 1000.0, 120.0
RULES = os.path.join(ROOT, "bundles", "guards_2026.json")
ARMS = {"A": "benefits_v3.npz", "B": "benefits_v2.npz"}
NEW = ["state_eitc", "state_ctc", "head_start", "early_head_start", "housing"]
G = {}


def out_path(mode):
    return os.path.join(ROOT, "results", f"d43_{mode}.jsonl")


def init(mode="test"):
    from benefits.population import PROGRAMMES, build, load_table_d43
    from common import splits

    _, _, meta, tr, va, test = splits()
    t = load_table_d43()
    pop15, _, _ = build(t, programmes=PROGRAMMES + NEW)
    G["meta"] = meta
    if mode == "dev":
        G["rows"] = np.sort(np.random.default_rng(2044).choice(va, size=300, replace=False))
    else:
        G["rows"] = np.sort(np.random.default_rng(2043).choice(test, size=2000, replace=False))
    G["pop"] = pop15  # true answers and true outcomes (fifteen programmes) for every household
    G["arms"] = {}
    for arm, f in ARMS.items():
        b = bundle.load(os.path.join(ROOT, "bundles", f))
        rules = Rules.load(RULES, b.pop.questions, b.pop.decisions)
        G["arms"][arm] = (b, GuardedOutcome(b.model, rules))
    # "the state has one": a non-zero amount for any training household in that state
    rest = np.sort(np.concatenate([tr, va]))
    st = pop15.answers[rest, pop15.qindex("state")]
    G["has"] = {p: set(np.unique(st[pop15.value[rest, pop15.decisions.index(p)] > 0]).tolist())
                for p in ("state_eitc", "state_ctc")}


def users():
    v = G["meta"]["volunteerable"]
    return {"clean": UserModel(name="clean"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.10, name="realistic")}


def interview(b, outcome, answers_row, user, rng):
    """simulate.run_one, keeping the final session for rule_b."""
    pol = VOIPolicy(question_cost=QCOST)
    s = Session(b.pop, b.loss, min_ess=MIN_ESS, outcome=outcome)
    pol.start(s)
    if user.opener and user.volunteerable:
        for j in rng.choice(user.volunteerable, size=min(user.opener, len(user.volunteerable)), replace=False):
            j = int(j)
            s.answer(j, reported_option(rng, b.pop.questions[j], int(answers_row[j]), user.misreport), source="opener")
    n = 0
    while True:
        j = pol.next_question(s, n)
        if j is None:
            break
        n += 1
        if user.dont_know and rng.random() < user.dont_know:
            s.observe(j, np.ones(len(b.pop.questions[j].options)), source="dont_know")
        else:
            s.answer(j, reported_option(rng, b.pop.questions[j], int(answers_row[j]), user.misreport))
    return n, np.asarray(pol.final_recommend(s), bool), s


def rule_b(s, rec10, decisions) -> list[int]:
    """B's recommendations for the five added programmes: a rule a screener could use without a model."""
    pop, w = s.pop, s.w
    q = {qq.id: j for j, qq in enumerate(pop.questions)}
    share = lambda qid, opt: float(w[pop.answers[:, q[qid]] == pop.questions[q[qid]].options.index(opt)].sum())
    state = int(np.argmax(np.bincount(pop.answers[:, q["state"]], weights=w, minlength=len(pop.questions[q["state"]].options))))
    r = dict(zip(decisions, rec10))
    need = r["snap"] or r["tanf"] or r["ssi"]
    return [int(r["eitc"] and state in G["has"]["state_eitc"]),
            int(r["ctc"] and state in G["has"]["state_ctc"]),
            int(need and share("child_under5", "yes") >= 0.5),
            int(need and share("infant_or_pregnant", "yes") >= 0.5),
            int(r["snap"] and share("tenure", "rent") >= 0.5)]


def work(job):
    mode, arm, uname, rows = job
    b, outcome = G["arms"][arm]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 43 + int(i))  # the same draws for both arms: paired
        n, rec, s = interview(b, outcome, G["pop"].answers[i], users()[uname], rng)
        rec = rec.astype(int).tolist()
        if arm == "B":
            rec = rec + rule_b(s, rec, list(b.pop.decisions))
        out.append({"arm": arm, "user": uname, "i": int(i), "q": n, "rec": rec,
                    "raw_ess": round(s.support()["raw_ess"], 3)})
    return out


def run(mode, workers):
    init(mode)
    path = out_path(mode)
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["arm"], r["user"], r["i"]))
    jobs = []
    for uname in (["clean"] if mode == "dev" else ["clean", "realistic"]):
        for arm in ARMS:
            todo = [int(i) for i in G["rows"] if (arm, uname, int(i)) not in done]
            jobs += [(mode, arm, uname, todo[k:k + 25]) for k in range(0, len(todo), 25)]
    print(f"{mode}: {len(G['rows'])} households, {sum(len(j[3]) for j in jobs)} interviews to run", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=init, initargs=(mode,)) as pool, open(path, "a") as f:
        for k, res in enumerate(pool.imap_unordered(work, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"  {k + 1}/{len(jobs)} batches, {time.time() - t0:.0f}s", flush=True)


def analyse(mode):
    init(mode)
    rows, pop = G["rows"], G["pop"]
    loss = LossModel(np.full(len(pop.decisions), FR), np.full(len(pop.decisions), FLOOR))
    miss, fr = loss.realised(pop)
    runs = {}
    for line in open(out_path(mode)):
        r = json.loads(line)
        runs[(r["arm"], r["user"], r["i"])] = r
    w = pop.weight[rows] / pop.weight[rows].sum()
    boots = np.random.default_rng(43).choice(len(rows), size=(4000, len(rows)))
    lines, summary = [f"D43 {mode}: {len(rows)} households, A (v3, fifteen programmes) minus B (v2 + rule)"], {}

    def ci(d):
        bs = np.array([np.average(d[b], weights=w[b]) for b in boots])
        return float(w @ d), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for uname in (["clean"] if mode == "dev" else ["clean", "realistic"]):
        rec = {arm: np.array([runs[(arm, uname, int(i))]["rec"] for i in rows], bool) for arm in ARMS}
        q = {arm: np.array([runs[(arm, uname, int(i))]["q"] for i in rows], float) for arm in ARMS}
        cost = {arm: np.where(rec[arm], fr[rows], miss[rows]) for arm in ARMS}  # (households, 15)
        new, old = slice(10, 15), slice(0, 10)
        p1 = ci(cost["A"][:, new].sum(1) - cost["B"][:, new].sum(1))
        p2 = ci(cost["A"][:, old].sum(1) - cost["B"][:, old].sum(1))
        tot = ci(cost["A"].sum(1) + QCOST * q["A"] - cost["B"].sum(1) - QCOST * q["B"])
        el = pop.eligible[rows]
        wrong_neg = {arm: float((~rec[arm][:, new] & el[:, new]).sum() / max((~rec[arm][:, new]).sum(), 1)) for arm in ARMS}
        found = {arm: float((rec[arm][:, new] & el[:, new]).sum() / max(el[:, new].sum(), 1)) for arm in ARMS}
        per = []
        for d, name in enumerate(NEW):
            col = 10 + d
            a, b_ = cost["A"][:, col], cost["B"][:, col]
            per.append(f"{name} A ${w @ a:,.0f} B ${w @ b_:,.0f}")
        lines += [f"\n{uname} answers",
                  f"  P1 five added programmes, decision loss: {p1[0]:+,.0f} [{p1[1]:+,.0f}, {p1[2]:+,.0f}]",
                  f"  P2 original ten, decision loss: {p2[0]:+,.0f} [{p2[1]:+,.0f}, {p2[2]:+,.0f}]  "
                  f"(95% half-width {(p2[2] - p2[1]) / 2:,.0f})",
                  f"  total over fifteen incl. $15 a question: {tot[0]:+,.0f} [{tot[1]:+,.0f}, {tot[2]:+,.0f}]",
                  f"  questions A {np.average(q['A'], weights=w):.1f}, B {np.average(q['B'], weights=w):.1f}",
                  f"  the five: eligible programmes found A {found['A']:.1%} B {found['B']:.1%}; "
                  f"'not recommended' that is wrong A {wrong_neg['A']:.2%} B {wrong_neg['B']:.2%}",
                  "  per programme: " + "; ".join(per)]
        summary[uname] = {"P1": p1, "P2": p2, "total": tot, "questions": [float(np.average(q[a], weights=w)) for a in ARMS],
                          "found": found, "wrong_negative": wrong_neg}
    text = "\n".join(lines)
    print(text)
    open(out_path(mode).replace(".jsonl", "_analysis.txt"), "w").write(text + "\n")
    json.dump(summary, open(out_path(mode).replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    mode, workers = sys.argv[1], int(sys.argv[2])
    (analyse if len(sys.argv) > 3 and sys.argv[3] == "analyse" else lambda m: run(m, workers))(mode)
