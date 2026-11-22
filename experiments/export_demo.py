"""Export a compact population for the browser demo.

The training population is resampled (systematic resampling by survey weight) to N equally weighted
households, so the page can run the same maths as the engine without weights. Test households are
never included. Also exports a few held-out households as "try this person" examples, with their
true programme results for comparison after the interview.

usage: python experiments/export_demo.py N OUT.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))

from benefits.population import LABELS, build, load_table  # noqa: E402

from decisionintake.simulate import split_by_group  # noqa: E402


def main():
    n, out = int(sys.argv[1]), sys.argv[2]
    t = load_table()
    pop, loss, meta = build(t)
    rest, test = split_by_group(pop, 0.15, seed=0)
    train = pop.subset(rest)
    rng = np.random.default_rng(0)
    c = np.cumsum(train.weight / train.weight.sum())
    c[-1] = 1.0
    idx = np.searchsorted(c, (rng.random() + np.arange(n)) / n)
    examples = rng.choice(test, size=12, replace=False)
    data = {
        "source": "PolicyEngine-US 2.152.15 on Populace 2024 (populace-us-2024-spm-20260915), 2026 rules, "
                  "take-up assumed; a calibrated synthetic population, not real people",
        "questions": [
            {"id": q.id, "text": q.text, "options": list(q.options),
             "noise": np.round(q.report_matrix(), 4).tolist()}
            for q in pop.questions
        ],
        "programmes": [{"id": p, "label": LABELS[p]} for p in pop.decisions],
        "false_rec_cost": loss.false_rec_cost.tolist(),
        "miss_floor": loss.miss_floor.tolist(),
        "answers": train.answers[idx].astype(int).tolist(),
        "value": np.round(train.value[idx]).astype(int).tolist(),
        "examples": [
            {"answers": pop.answers[i].astype(int).tolist(), "value": np.round(pop.value[i]).astype(int).tolist()}
            for i in examples
        ],
    }
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {n} households, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
