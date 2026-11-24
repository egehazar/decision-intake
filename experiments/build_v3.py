"""D43: train the add-on network for the five added programmes and ship it beside the frozen v2 network.

v2 (bundles/benefits_v2.npz) is loaded, not retrained: its ten programmes' probabilities must come out of the
new bundle bit for bit. The add-on has v2's architecture and settings (D28) and is trained on the same
48,657 training households for the five D43 programmes only.

usage: python experiments/build_v3.py OUT.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from benefits.population import ADDED_D43, LABELS, MIN_VALUE, PROGRAMMES, build, load_table_d43  # noqa: E402
from common import smoothed_neural, splits  # noqa: E402
from decisionintake import LossModel, bundle  # noqa: E402
from export_demo import terms  # noqa: E402
from run_v2 import FR, NEURAL  # noqa: E402

V2 = os.path.join(HERE, "..", "bundles", "benefits_v2.npz")


def main():
    out = sys.argv[1]
    table = load_table_d43()
    pop10, _, _, tr, va, test = splits()
    rest = np.sort(np.concatenate([tr, va]))
    pop5, _, _ = build(table, programmes=ADDED_D43)
    assert np.array_equal(pop5.answers, pop10.answers), "the D43 table must line up with the original"
    _, addon = smoothed_neural(pop5, rest, "d43_rest", **NEURAL)

    v2 = bundle.load(V2)
    pop15, _, _ = build(table, programmes=PROGRAMMES + ADDED_D43)
    cur = pop15.subset(rest)
    # v2's belief population carries the answers as coded before D34's earnings fix; keep it, so the ten
    # programmes behave exactly as served. The same households in the same order: weights and outcomes match,
    # and only the three answers D34 changed differ. The five added outcomes come from PolicyEngine (untouched
    # by the coding bug); the add-on itself was trained on the corrected answers.
    assert np.allclose(cur.weight, v2.pop.weight) and np.array_equal(cur.eligible[:, :10], v2.pop.eligible)
    differ = {cur.questions[j].id for j in range(len(cur.questions))
              if not np.array_equal(cur.answers[:, j], v2.pop.answers[:, j])}
    assert differ <= {"earned_income", "self_employed", "other_income"}, differ
    from decisionintake.core import Population
    train15 = Population(v2.pop.questions, list(pop15.decisions), v2.pop.answers, v2.pop.weight,
                         np.hstack([v2.pop.eligible, cur.eligible[:, 10:]]), np.hstack([v2.pop.value, cur.value[:, 10:]]))
    model = bundle.StackedOutcome([v2.model, addon])
    loss = LossModel(np.full(len(pop15.decisions), FR), np.full(len(pop15.decisions), MIN_VALUE))
    info = {**v2.info,
            "name": "DecisionIntake v3, US benefits screening (v2 plus five programmes, DECISIONS D43)",
            "model": v2.info["model"] + "; five programmes added by a second network with the same settings "
                     "(D43), v2's own network unchanged"}
    bundle.save(out, train15, loss, model, terms(train15), LABELS, info)

    b = bundle.load(out)
    rows = np.random.default_rng(0).choice(len(train15), 2000, replace=False)
    p_v2, v_v2 = v2.model.predict(v2.pop, v2.pop.answers[rows])
    p_v3, v_v3 = b.model.predict(b.pop, b.pop.answers[rows])
    assert np.array_equal(p_v3[:, :10], p_v2) and np.array_equal(v_v3[:, :10], v_v2), "v2's programmes changed"
    pa, _ = addon.predict(pop5, train15.answers[rows])
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB; v2's ten unchanged; add-on max |dp| "
          f"{np.abs(p_v3[:, 10:] - pa).max():.2e}")
    for d, name in enumerate(ADDED_D43):  # calibration on the training households, a sanity check
        w = train15.weight
        print(f"  {name}: mean predicted {np.average(p_v3[:, 10 + d], weights=w[rows]):.3f}, "
              f"rate {np.average(train15.eligible[rows, 10 + d], weights=w[rows]):.3f}")


if __name__ == "__main__":
    main()
