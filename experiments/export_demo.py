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
sys.path.insert(0, os.path.dirname(__file__))

from benefits.population import INTERACT, LABELS, _layout  # noqa: E402
from common import smoothed, splits  # noqa: E402


def terms(pop):
    off, sizes, F = _layout(pop)
    qi = {q.id: j for j, q in enumerate(pop.questions)}
    out = [{"qs": [qi[q.id]], "strides": [1], "off": off[q.id]} for q in pop.questions]
    for k in INTERACT:
        out.append({"qs": [qi["earned_income"], qi[k]], "strides": [sizes[k], 1], "off": off[("x", k)]})
    out.append({"qs": [qi["children"], qi["state"]], "strides": [sizes["state"], 1], "off": off[("x", "children_state")]})
    out.append({"qs": [], "strides": [], "off": F - 1})
    return out


def main():
    n, out = int(sys.argv[1]), sys.argv[2]
    fr = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
    pop, loss, meta, tr, va, test = splits()
    train, model = smoothed(pop, np.sort(np.concatenate([tr, va])), "rest")
    loss.false_rec_cost[:] = fr
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
        # outcome model: each term is an indicator column off + sum(answer * stride); logit and dollars
        # are sums of B and R over a household's active columns (the last column is the intercept)
        "terms": terms(pop),
        "B": np.round(model.B, 5).tolist(),
        "R": np.round(model.R, 2).tolist(),
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
