"""Export the v2 engine as one self-contained bundle (decisionintake.bundle), for the MCP server.

The bundle holds the training population (train + validation, 48,657 households), the loss, and the
wide-and-deep outcome model frozen in D28. It checks itself before writing: the stored terms must give
the adapter's exact features, and the loaded model must give the same probabilities.

usage: python experiments/export_bundle.py OUT.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))

from benefits.population import LABELS, features  # noqa: E402
from common import smoothed_neural, splits  # noqa: E402
from decisionintake import LossModel, bundle  # noqa: E402
from export_demo import terms  # noqa: E402
from run_v2 import FR, NEURAL  # noqa: E402


def main():
    out = sys.argv[1]
    pop, loss, meta, tr, va, test = splits()
    train, model = smoothed_neural(pop, np.sort(np.concatenate([tr, va])), "rest", **NEURAL)
    loss = LossModel(np.full(len(pop.decisions), FR), loss.miss_floor)
    t = terms(pop)
    tf = bundle.TermFeatures(t, model.Ww.shape[0])
    rows = np.random.default_rng(0).choice(len(train), 5000, replace=False)
    assert np.array_equal(tf(train, train.answers[rows]), features(train, train.answers[rows])), "terms != features"
    info = {
        "name": "DecisionIntake v2, US benefits screening",
        "data": "PolicyEngine-US 2.152.15 rules (2026) on Populace 2024 (populace-us-2024-spm-20260915), a "
                "calibrated synthetic population; 48,657 training households",
        "model": "wide-and-deep outcome network, settings frozen in DECISIONS D28",
        "loss": f"a missed programme costs its yearly value (at least $120); a wrong recommendation costs ${FR:,.0f}",
        "caveat": "A screening estimate from PolicyEngine's rules, not an eligibility decision. Apply through the "
                  "official programme to get one. Immigration status is not asked and can change eligibility.",
    }
    bundle.save(out, train, loss, model, t, LABELS, info)
    b = bundle.load(out)
    p0, v0 = model.predict(train, train.answers[rows[:500]])
    p1, v1 = b.model.predict(b.pop, b.pop.answers[rows[:500]])
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB; max |dp| {np.abs(p0 - p1).max():.2e}, max |dv| {np.abs(v0 - v1).max():.2e}")


if __name__ == "__main__":
    main()
