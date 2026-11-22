"""Export the browser demo's data for v2: a resampled population and the wide-and-deep outcome model.

The training population (train + validation) is resampled by survey weight to N equally weighted
households, so the page runs the engine's maths without weights. Test households appear only as the
"answer as" examples, with PolicyEngine's true results shown after the interview. The model's weights
ship as base64 float32 arrays; the page runs the same forward pass as decisionintake.neural.

usage: python experiments/export_demo_v2.py N OUT.json
"""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))

from benefits.population import LABELS  # noqa: E402
from common import smoothed_neural, splits  # noqa: E402
from export_demo import terms  # noqa: E402
from run_v2 import FR, NEURAL  # noqa: E402


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def main():
    n, out = int(sys.argv[1]), sys.argv[2]
    pop, loss, meta, tr, va, test = splits()
    train, model = smoothed_neural(pop, np.sort(np.concatenate([tr, va])), "rest", **NEURAL)
    rng = np.random.default_rng(0)
    c = np.cumsum(train.weight / train.weight.sum())
    c[-1] = 1.0
    idx = np.searchsorted(c, (rng.random() + np.arange(n)) / n)
    examples = rng.choice(test, size=16, replace=False)
    H1, H2 = model.W1.shape[1], model.W2.shape[1]
    data = {
        "source": "PolicyEngine-US 2.152.15 on Populace 2024 (populace-us-2024-spm-20260915), 2026 rules, "
                  "take-up assumed; a calibrated synthetic population, not real people",
        "questions": [
            {"id": q.id, "text": q.text, "options": list(q.options), "noise": np.round(q.report_matrix(), 4).tolist(),
             "edges": list(q.edges) if q.edges is not None else None, "per": q.per}
            for q in pop.questions
        ],
        "programmes": [{"id": p, "label": LABELS[p]} for p in pop.decisions],
        "false_rec_cost": [FR] * len(pop.decisions),
        "miss_floor": loss.miss_floor.tolist(),
        "answers": train.answers[idx].astype(int).tolist(),
        "model": {"terms": terms(pop), "F": int(model.Ww.shape[0]), "H1": int(H1), "H2": int(H2),
                  "Ww": b64(model.Ww), "W1": b64(model.W1), "b1": b64(model.b1), "W2": b64(model.W2),
                  "b2": b64(model.b2), "W3": b64(model.W3), "scale": [float(s) for s in model.scale]},
        "examples": [{"answers": pop.answers[i].astype(int).tolist(), "value": np.round(pop.value[i]).astype(int).tolist()}
                     for i in examples],
    }
    # self-check: the page must reproduce Python's probabilities; ship a few reference rows
    p, v = model.predict(train, train.answers[idx[:5]])
    data["check"] = {"rows": idx[:5].tolist(), "p": np.round(p, 5).tolist(), "v": np.round(v, 1).tolist()}
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out}: {n} households, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
