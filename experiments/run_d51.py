"""D51, registered: "gets SNAP" as evidence. See DECISIONS D51 for the criteria.

    python experiments/run_d51.py dev     600 validation households (the receipt_dev selection), full answers:
                                          sizes P2's margin before the test
    python experiments/run_d51.py test    every SNAP-eligible test household, getting SNAP with p 0.82 (seed 2051)

Arms: B = served engine; A = the same with Settings.receipt_evidence = {"snap": RECEIPT_EPS["snap"]}. Both
record the receipt (it.receive(["snap"])); only A uses it as evidence. Loss on the 14 programmes other than
SNAP against PolicyEngine's labels: $1,000 a wrong recommendation, the value (at least $120) a miss.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, HERE)
from benefits.population import ADDED_D43, MIN_VALUE, PROGRAMMES, build, load_table_d43  # noqa: E402
from common import splits  # noqa: E402
from decisionintake import bundle  # noqa: E402
from decisionintake.guards import DEFAULT_RULES, attach  # noqa: E402
from decisionintake.service import RECEIPT_EPS, Intake, Settings  # noqa: E402

SPARSE = ["state", "household_size", "children", "child_under5", "tenure"]
FR = 1000.0


def main():
    mode = sys.argv[1]
    _, _, _, tr, va, test = splits()
    pop, _, _ = build(load_table_d43(), programmes=PROGRAMMES + ADDED_D43)
    d_snap = pop.decisions.index("snap")
    if mode == "dev":  # the receipt_dev selection, so P2's margin comes from the same development households
        rng = np.random.default_rng(2047)
        elig = va[pop.eligible[va, d_snap].astype(bool)]
        gets = elig[rng.random(len(elig)) < 0.82]
        rows = np.sort(rng.choice(gets, size=min(600, len(gets)), replace=False))
        settings_list = [("full", None)]
    else:
        rng = np.random.default_rng(2051)
        elig = test[pop.eligible[test, d_snap].astype(bool)]
        rows = np.sort(elig[rng.random(len(elig)) < 0.82])
        settings_list = [("sparse", SPARSE), ("full", None)]
    b = bundle.load(os.path.join(ROOT, "bundles", "benefits_v3.npz"))
    attach(b, DEFAULT_RULES)
    arms = {"B": Settings(abstain_below=10.0),
            "A": Settings(abstain_below=10.0, receipt_evidence={"snap": RECEIPT_EPS["snap"]})}
    qi = {q.id: j for j, q in enumerate(pop.questions)}
    out_path = os.path.join(ROOT, "results", f"d51_{mode}.jsonl")
    t0 = time.time()
    with open(out_path, "w") as fh:
        for k, i in enumerate(rows):
            for fname, fields in settings_list:
                ids = fields or [q.id for q in pop.questions]
                facts = [{"question": f, "choice": pop.questions[qi[f]].options[int(pop.answers[i, qi[f]])]} for f in ids]
                for arm, cfg in arms.items():
                    it = Intake(b, cfg)
                    it.receive(["snap"])
                    it.record(facts)
                    rec = [bool(dv.recommend) for dv in it.s.decisions()]
                    fh.write(json.dumps({"i": int(i), "facts": fname, "arm": arm, "rec": rec}) + "\n")
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(rows)} households, {time.time() - t0:.0f}s", flush=True)
    analyse(mode, pop, rows, out_path, [f for f, _ in settings_list])


def analyse(mode, pop, rows, path, fact_sets):
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs[(r["facts"], r["arm"], r["i"])] = np.array(r["rec"], bool)
    d_snap = pop.decisions.index("snap")
    other = [k for k in range(len(pop.decisions)) if k != d_snap]
    miss = np.where(pop.eligible, np.maximum(pop.value, MIN_VALUE), 0.0)
    fr = np.where(pop.eligible, 0.0, FR)
    w = pop.weight[rows]
    boots = np.random.default_rng(51).choice(len(rows), size=(4000, len(rows)))
    lines = [f"D51 {mode}: {len(rows)} households that get SNAP; A (receipt as evidence) minus B (served), decision "
             f"loss on the 14 programmes other than SNAP, $/household/year, survey-weighted"]
    summary = {"households": int(len(rows))}
    for fs in fact_sets:
        cost = {}
        for arm in ("A", "B"):
            rec = np.array([runs[(fs, arm, int(i))] for i in rows])
            cost[arm] = np.where(rec, fr[rows], miss[rows])[:, other]
        d = cost["A"].sum(1) - cost["B"].sum(1)
        bs = np.array([np.average(d[k], weights=w[k]) for k in boots])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mean = float(np.average(d, weights=w))
        p_ge0 = float((np.sum(bs >= 0) + 1) / (len(bs) + 1))
        p_ge25 = float((np.sum(bs >= 25) + 1) / (len(bs) + 1))
        differ = int((np.array([runs[(fs, "A", int(i))][other] != runs[(fs, "B", int(i))][other] for i in rows])).any(1).sum())
        summary[fs] = {"diff": mean, "ci": [float(lo), float(hi)], "half_width": float((hi - lo) / 2),
                       "p_diff_ge_0": p_ge0, "p_diff_ge_25": p_ge25, "households_deciding_differently": differ,
                       "A": float(np.average(cost["A"].sum(1), weights=w)), "B": float(np.average(cost["B"].sum(1), weights=w))}
        lines.append(f"  {fs} answers: A {summary[fs]['A']:,.0f}  B {summary[fs]['B']:,.0f}  A-B {mean:+,.0f} "
                     f"[{lo:+,.0f}, {hi:+,.0f}] (half-width {summary[fs]['half_width']:.0f}); "
                     f"p(A-B>=0) {p_ge0:.4f}, p(A-B>=25) {p_ge25:.4f}; decisions differ in {differ} households")
        per = {pop.decisions[k]: float(np.average(cost["A"][:, j] - cost["B"][:, j], weights=w)) for j, k in enumerate(other)}
        summary[fs]["per_programme"] = per
        lines.append("    per programme A-B: " + ", ".join(f"{n} {v:+,.0f}" for n, v in per.items() if abs(v) >= 1))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(ROOT, "results", f"d51_{mode}_analysis.txt"), "w").write(text + "\n")
    json.dump(summary, open(os.path.join(ROOT, "results", f"d51_{mode}_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
