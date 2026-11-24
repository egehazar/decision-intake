"""Caseload triage, development check (not registered): does ranking by the engine's expected value put the
money first?

2,000 validation households (never test households), each cut down to the five fields an organisation's records
typically hold (state, household size, children, earnings bracket, rent or own). Each is ranked by Intake.triage's
expected value; the score is the share of the true benefit dollars (PolicyEngine's, all ten programmes, and
without health coverage) that falls in the top 20% of the list. Baselines: random order, lowest earnings first,
and largest household first (earnings as the tie-break).

usage: python experiments/triage_dev.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from common import splits  # noqa: E402
from decisionintake import bundle  # noqa: E402
from decisionintake.guards import DEFAULT_RULES, attach  # noqa: E402
from decisionintake.service import Intake, Settings  # noqa: E402

FIELDS = ["state", "household_size", "children", "earned_income", "tenure"]
HEALTH = {"medicaid", "chip", "aca_ptc"}


def capture(order, value, top=0.2):
    k = int(round(len(order) * top))
    return float(value[order[:k]].sum() / value.sum())


def main():
    pop, _, _, tr, va, test = splits()
    rows = np.sort(np.random.default_rng(2045).choice(va, size=2000, replace=False))
    b = bundle.load(os.path.join(ROOT, "bundles", "benefits_v2.npz"))
    attach(b, DEFAULT_RULES)
    qi = {q.id: j for j, q in enumerate(pop.questions)}
    ev = np.zeros(len(rows))
    for n, i in enumerate(rows):
        it = Intake(b, Settings(abstain_below=10.0))
        it.record([{"question": f, "choice": pop.questions[qi[f]].options[int(pop.answers[i, qi[f]])]} for f in FIELDS])
        ev[n] = it.triage()["expected_value"]
    value_all = (pop.eligible[rows] * pop.value[rows]).sum(1)
    cash = [d for d, name in enumerate(pop.decisions) if name not in HEALTH]
    value_cash = (pop.eligible[rows][:, cash] * pop.value[rows][:, cash]).sum(1)
    earn = pop.answers[rows, qi["earned_income"]]
    size = pop.answers[rows, qi["household_size"]]
    orders = {
        "engine (expected value)": np.argsort(-ev, kind="stable"),
        "lowest earnings first": np.lexsort((-size, earn)),
        "largest household first": np.lexsort((earn, -size)),
    }
    rng = np.random.default_rng(0)
    rand = [capture(rng.permutation(len(rows)), value_all) for _ in range(200)]
    rand_cash = [capture(rng.permutation(len(rows)), value_cash) for _ in range(200)]
    out = {"households": len(rows), "fields": FIELDS, "top_share": 0.2,
           "random": {"all": float(np.mean(rand)), "without_health": float(np.mean(rand_cash))}}
    lines = [f"Caseload triage, development: {len(rows)} validation households, five fields each; share of true "
             f"benefit dollars in the top 20% of the list",
             f"  random order: {np.mean(rand):.1%} (without health coverage {np.mean(rand_cash):.1%})"]
    for name, o in orders.items():
        a, c = capture(o, value_all), capture(o, value_cash)
        out[name] = {"all": a, "without_health": c}
        lines.append(f"  {name}: {a:.1%} (without health coverage {c:.1%})")
    text = "\n".join(lines)
    print(text)
    json.dump(out, open(os.path.join(ROOT, "results", "triage_dev.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "triage_dev.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
