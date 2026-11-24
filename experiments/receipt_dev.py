"""Feasibility, development only (not registered): is "the household gets SNAP" worth using as evidence?

Validation households only. Receipt is simulated: a household PolicyEngine labels SNAP-eligible gets SNAP with
probability 0.82 (USDA's recent participation rate among eligible people). Among households that get it, each is
known by five fields (state, household size, children, a child under 5, rent or own), as on a caseload list.
Arm B: the served engine on those fields. Arm A: the same plus receipt as evidence, a likelihood of 1 for
belief households PolicyEngine labels SNAP-eligible and EPS for the rest (reports and the rules model don't
always agree). Decision loss on the other fourteen programmes against the labels, $1,000 a wrong
recommendation, the benefit's value (at least $120) a miss; survey-weighted mean per household.

usage: python experiments/receipt_dev.py [N]
"""
from __future__ import annotations

import json
import os
import sys

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
from decisionintake.service import Intake, Settings  # noqa: E402

FIELDS = ["state", "household_size", "children", "child_under5", "tenure"]
EPS = [0.02, 0.05, 0.1, 0.2, 0.4]
FR = 1000.0


def decisions_after(it):
    return np.array([dv.recommend for dv in it.s.decisions()], bool)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    _, _, _, tr, va, test = splits()
    pop, _, _ = build(load_table_d43(), programmes=PROGRAMMES + ADDED_D43)
    b = bundle.load(os.path.join(ROOT, "bundles", "benefits_v3.npz"))
    attach(b, DEFAULT_RULES)
    d_snap = pop.decisions.index("snap")
    rng = np.random.default_rng(2047)
    elig_rows = va[pop.eligible[va, d_snap].astype(bool)]
    gets = elig_rows[rng.random(len(elig_rows)) < 0.82]
    rows = np.sort(rng.choice(gets, size=min(n, len(gets)), replace=False))
    qi = {q.id: j for j, q in enumerate(pop.questions)}
    other = [k for k in range(len(pop.decisions)) if k != d_snap]
    miss = np.where(pop.eligible, np.maximum(pop.value, MIN_VALUE), 0.0)
    fr = np.where(pop.eligible, 0.0, FR)
    belief_snap = b.pop.eligible[:, b.pop.decisions.index("snap")]
    loss = {"B": []} | {f"A{e}": [] for e in EPS}
    for i in rows:
        facts = [{"question": f, "choice": pop.questions[qi[f]].options[int(pop.answers[i, qi[f]])]} for f in FIELDS]
        it = Intake(b, Settings(abstain_below=10.0))
        it.record(facts)
        rec = decisions_after(it)
        loss["B"].append(np.where(rec, fr[i], miss[i])[other].sum())
        for e in EPS:
            it2 = Intake(b, Settings(abstain_below=10.0))
            it2.record(facts)
            s = it2.s
            s.loglik = s.loglik + np.log(np.where(belief_snap, 1.0, e))
            s._reweight()
            s._rescore()
            rec = decisions_after(it2)
            loss[f"A{e}"].append(np.where(rec, fr[i], miss[i])[other].sum())
    w = pop.weight[rows]
    boots = np.random.default_rng(0).choice(len(rows), size=(2000, len(rows)))
    base = np.array(loss["B"])
    lines = [f"Receipt as evidence, development: {len(rows)} validation households that get SNAP (simulated), "
             f"five fields each; decision loss on the other 14 programmes, $/household/year",
             f"  B (no receipt evidence): {np.average(base, weights=w):,.0f}"]
    out = {"households": int(len(rows)), "B": float(np.average(base, weights=w))}
    for e in EPS:
        a = np.array(loss[f"A{e}"])
        d = a - base
        bs = np.array([np.average(d[k], weights=w[k]) for k in boots])
        out[f"A{e}"] = {"loss": float(np.average(a, weights=w)), "diff": float(np.average(d, weights=w)),
                        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        lines.append(f"  A eps={e}: {out[f'A{e}']['loss']:,.0f}  A-B {out[f'A{e}']['diff']:+,.0f} "
                     f"[{out[f'A{e}']['ci'][0]:+,.0f}, {out[f'A{e}']['ci'][1]:+,.0f}]")
    text = "\n".join(lines)
    print(text)
    json.dump(out, open(os.path.join(ROOT, "results", "receipt_dev.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "receipt_dev.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
