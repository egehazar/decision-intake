"""Development check (not registered): with "no child under 5" answered, how often does the served engine still
recommend a programme for young children (Early Head Start, Head Start, WIC), and how often is it right?

Validation households only (never test). Each answers every question cleanly (the full set of facts), the way a
careful navigator would record them; the served settings are used (v3 with the D32 rules, abstention at 10).

usage: python experiments/young_child_dev.py [N]
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
from decisionintake import bundle  # noqa: E402
from decisionintake.guards import DEFAULT_RULES, attach  # noqa: E402
from decisionintake.service import Intake, Settings  # noqa: E402

YOUNG = ["early_head_start", "head_start", "wic"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    _, _, _, tr, va, test = splits()
    pop, _, _ = build(load_table_d43(), programmes=PROGRAMMES + ADDED_D43)
    qi = {q.id: j for j, q in enumerate(pop.questions)}
    j5 = qi["child_under5"]
    no = va[pop.answers[va, j5] == pop.questions[j5].options.index("no")]
    rows = np.sort(np.random.default_rng(2046).choice(no, size=min(n, len(no)), replace=False))
    b = bundle.load(os.path.join(ROOT, "bundles", "benefits_v3.npz"))
    attach(b, DEFAULT_RULES)
    dix = {d: pop.decisions.index(d) for d in YOUNG}
    rec = {d: [] for d in YOUNG}
    prob = {d: [] for d in YOUNG}
    thin = []
    for i in rows:
        it = Intake(b, Settings(abstain_below=10.0))
        it.record([{"question": q.id, "choice": q.options[int(pop.answers[i, j])]} for j, q in enumerate(pop.questions)])
        a = {r["id"]: r for r in it.assessment()["all_programmes"]}
        thin.append(it.support()["level"] == "thin")
        for d in YOUNG:
            rec[d].append(a[d]["tier"] in ("likely", "worth checking"))
            prob[d].append(a[d]["probability"])
    thin = np.array(thin)
    w = pop.weight[rows] / pop.weight[rows].sum()
    out = {"households": int(len(rows)), "thin_share": float(np.average(thin, weights=w))}
    lines = [f"No child under 5 answered: {len(rows)} validation households, every answer given; "
             f"thin support for {out['thin_share']:.1%} (weighted)"]
    for d in YOUNG:
        r, p = np.array(rec[d]), np.array(prob[d])
        y = pop.eligible[rows, dix[d]].astype(bool)
        wrong = r & ~y
        out[d] = {"recommended": float(np.average(r, weights=w)), "eligible": float(np.average(y, weights=w)),
                  "wrong_recommendation": float(np.average(wrong, weights=w)),
                  "recommended_thin": float(r[thin].mean()) if thin.any() else 0.0,
                  "recommended_supported": float(r[~thin].mean()) if (~thin).any() else 0.0,
                  "mean_probability": float(np.average(p, weights=w))}
        lines.append(f"  {d}: recommended {out[d]['recommended']:.1%}, actually eligible {out[d]['eligible']:.1%}, "
                     f"wrong recommendation {out[d]['wrong_recommendation']:.1%} "
                     f"(thin {out[d]['recommended_thin']:.1%} vs supported {out[d]['recommended_supported']:.1%} "
                     f"recommended); mean probability {out[d]['mean_probability']:.3f}")
    text = "\n".join(lines)
    print(text)
    json.dump(out, open(os.path.join(ROOT, "results", "young_child_dev.json"), "w"), indent=1)
    open(os.path.join(ROOT, "results", "young_child_dev.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
