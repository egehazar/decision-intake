"""End-to-end interviews with spoken answers: each reader against typed answers on the same households.

With 40 households, a few heavily weighted households dominate survey-weighted means, so this reports
UNWEIGHTED means and paired differences (bootstrap 95% intervals), plus how often each reader's top
reading differed from the truth when it committed.

usage: python experiments/analyze_language.py N READER [READER...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from common import splits  # noqa: E402

from decisionintake import LossModel  # noqa: E402


def main():
    n, names = int(sys.argv[1]), sys.argv[2:]
    pop, loss0, *_ = splits()
    L = LossModel(np.full(len(pop.decisions), 1000.0), loss0.miss_floor)
    res = {}
    for name in ["typed"] + names:
        path = f"results/language_{name}_{n}_15_10.json"
        if os.path.exists(path):
            res[name] = {r["i"]: r for r in json.load(open(path))}
    idx = np.array(sorted(res["typed"]))
    miss, fr = L.realised(pop.subset(idx))
    el = pop.eligible[idx]

    def per(name):
        rec = np.array([res[name][i]["recommend"] for i in idx], bool)
        li = (np.where(rec, 0, miss) + np.where(rec, fr, 0)).sum(1)
        q = np.array([res[name][i]["n_questions"] for i in idx], float)
        steps = [t for i in idx for t in res[name][i]["trace"]]
        committed = [t for t in steps if not t["flat"]]
        misread = np.mean([t["top"] != t["truth"] for t in committed]) if committed else float("nan")
        found = (el & rec).sum() / el.sum()
        wrong = (~el & rec).sum() / max(rec.sum(), 1)
        return li, q, misread, len(steps), found, wrong

    base = per("typed")[0]
    boots = np.random.default_rng(3).choice(len(idx), size=(4000, len(idx)))
    print(f"{len(idx)} test households, unweighted, $1,000 per wrong recommendation, question cost $15, floor 10")
    print(f"{'reader':<14}{'questions':>10}{'loss':>8}{'found':>7}{'wrong':>7}{'misread':>9}   loss vs typed [95% CI]")
    for name in res:
        li, q, mis, steps, found, wrong = per(name)
        d = li - base
        lo, hi = np.percentile([d[b].mean() for b in boots], [2.5, 97.5])
        cmp = "" if name == "typed" else f"   {d.mean():+7.0f} [{lo:+.0f}, {hi:+.0f}]"
        print(f"{name:<14}{q.mean():10.2f}{li.mean():8.0f}{found:7.3f}{wrong:7.3f}{mis:9.3f}{cmp}")


if __name__ == "__main__":
    main()
