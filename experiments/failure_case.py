"""A reproducible failure: a held-out household the engine wrongly recommends the ACA credit (or CHIP)
to, traced turn by turn, so the mechanism behind the low ACA/CHIP precision is visible.

Picks the lowest-index test household in results/final_600.jsonl where the engine with its own
stopping ($15 a question, clean answers) recommended the programme but PolicyEngine says the household
can't get it, then re-runs that interview and prints, after every answer, the belief's probability
of eligibility, the expected yearly value if eligible, and the probability at which recommending
starts to pay under the loss ($1,000 per wrong recommendation).

usage: python experiments/failure_case.py [aca_ptc|chip|medicaid]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
import run_final as RF  # noqa: E402
from decisionintake import VOIPolicy  # noqa: E402
from decisionintake.core import Session  # noqa: E402

G = RF.G


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "aca_ptc"
    RF.init()
    pop, train, loss = G["test"], G["train"], G["loss"]
    names = list(pop.decisions)
    d = next(k for k, nm in enumerate(names) if want in nm.lower())
    wrong = []
    for line in open("results/final_600.jsonl"):
        r = json.loads(line)
        if r["policy"] == "engine" and r["k"] == 15 and r["user"] == "clean" and r["rec"][d] and not pop.eligible[r["i"], d]:
            wrong.append(r["i"])
    total = sum(1 for line in open("results/final_600.jsonl")
                if (r := json.loads(line))["policy"] == "engine" and r["k"] == 15 and r["user"] == "clean" and r["rec"][d])
    print(f"{names[d]}: {len(wrong)} of {total} recommendations wrong (engine, $15 a question, clean answers, 600 households)")
    i = min(wrong)
    row = pop.answers[i]
    s = Session(train, loss, min_ess=RF.MIN_ESS, outcome=G["outcome"])
    pol = VOIPolicy(question_cost=15.0)
    pol.start(s)

    def state(label):
        v = s.decisions()[d]
        pe = (s.p_now if s.p_now is not None else train.p_eligible)[:, d]
        vv = np.maximum((s.v_now if s.v_now is not None else train.value_if_eligible)[:, d], loss.miss_floor[d])
        val = float((s.w * pe) @ vv / max(s.w @ pe, 1e-12))
        p_at = loss.false_rec_cost[d] / (loss.false_rec_cost[d] + val)
        print(f"  {label:<40} p={v.p_eligible:5.2f}  worth ${val:7,.0f} if eligible  pays above p={p_at:4.2f}"
              f"  -> {'RECOMMEND' if v.recommend else 'skip'}")

    print(f"\ntest household {i}; truth: eligible={bool(pop.eligible[i, d])}, value ${pop.value[i, d]:,.0f}")
    state("before any question")
    n = 0
    while True:
        j = pol.next_question(s, n)
        if j is None:
            break
        n += 1
        q = train.questions[j]
        s.answer(j, int(row[j]))
        state(f"{q.id} = {q.options[int(row[j])]}")
    print("\nunasked:", ", ".join(f"{q.id} = {q.options[int(row[k])]}" for k, q in enumerate(train.questions)
                                  if k not in s.asked))


if __name__ == "__main__":
    main()
