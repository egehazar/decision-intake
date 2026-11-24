"""Do the plan's federal links (D45) agree with PolicyEngine's own labels?

For each link "getting SOURCE settles part of TARGET's eligibility", among training and validation households
eligible for SOURCE whose answers fit TARGET's category, the share PolicyEngine also labels eligible for TARGET.
A link that holds in the rules model should be near 100% (the rest is category detail the answers can't see,
such as a child's exact age). Labels, not the engine's predictions; no test household is used.

usage: python experiments/plan_rules_check.py
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
from benefits.population import ADDED_D43, PROGRAMMES, build, load_table_d43  # noqa: E402
from common import splits  # noqa: E402
from decisionintake.service import PLAN_CATEGORY, PLAN_LINKS  # noqa: E402


def main():
    _, _, _, tr, va, test = splits()
    rows = np.sort(np.concatenate([tr, va]))
    pop, _, _ = build(load_table_d43(), programmes=PROGRAMMES + ADDED_D43)
    qi = {q.id: j for j, q in enumerate(pop.questions)}
    dix = {d: k for k, d in enumerate(pop.decisions)}
    w = pop.weight[rows]
    elig = pop.eligible[rows].astype(bool)

    def fits(target):
        ok = np.ones(len(rows), bool)
        for rule in PLAN_CATEGORY.get(target, []):  # every household answers every question here
            out_ = np.ones(len(rows), bool)
            for q, ruled_out in rule.items():
                opts = pop.questions[qi[q]].options
                out_ &= np.isin(pop.answers[rows, qi[q]], [opts.index(o) for o in opts if o.lower() in ruled_out])
            ok &= ~out_
        return ok

    out, lines = {}, ["Plan links against PolicyEngine labels (training + validation households, survey-weighted)"]
    for target, sources, kind, text, cite in PLAN_LINKS:
        for s in sources:
            base = elig[:, dix[s]] & fits(target)
            n = int(base.sum())
            if n == 0:
                continue
            share = float(np.average(elig[base, dix[target]], weights=w[base]))
            out[f"{s}->{target}"] = {"households": n, "target_eligible": share, "rule": cite}
            lines.append(f"  {s:>8} -> {target:<16} {share:6.1%} of {n:>6,} households  ({cite})")
    text = "\n".join(lines)
    print(text)
    json.dump(out, open(os.path.join(ROOT, "results", "plan_rules_check.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "plan_rules_check.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
