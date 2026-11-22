"""Per programme, with every question answered: does the belief rank eligible households above
ineligible ones (AUC), and is its probability calibrated? Validation split only."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
from benefits.population import build, load_table
from decisionintake import LossModel, Session
from decisionintake.simulate import split_by_group


def auc(score, y, w):
    o = np.argsort(score); s, y, w = score[o], y[o], w[o]
    pos = (w * y).sum(); neg = (w * ~y).sum()
    cum_neg = np.cumsum(w * ~y)
    return float((w * y * (cum_neg - 0.5 * w * ~y)).sum() / (pos * neg)) if pos > 0 and neg > 0 else float("nan")


t = load_table()
pop, loss, meta = build(t)
rest, _ = split_by_group(pop, 0.15, seed=0)
sub = pop.subset(rest)
tr, va = split_by_group(sub, 0.12, seed=1)
train, val = sub.subset(tr), sub.subset(va)
L = LossModel(np.full(10, 1000.0), loss.miss_floor)
rng = np.random.default_rng(5)
sample = rng.choice(len(val), size=400, replace=False)
for m in [1, 10, 100]:
    P = []
    for i in sample:
        s = Session(train, L, min_ess=m)
        for j in range(len(train.questions)):
            s.answer(j, int(val.answers[i, j]))
        P.append(s.w @ train.eligible)
    P = np.array(P)
    w = val.weight[sample]
    el = val.eligible[sample]
    print(f"min_ess={m}")
    for d, name in enumerate(train.decisions):
        y = el[:, d]
        base = (w * y).sum() / w.sum()
        hi = P[:, d] > 0.5
        print(f"   {name:<13} share eligible {base:.3f}  AUC {auc(P[:, d], y, w):.3f}  "
              f"mean P when eligible {np.average(P[y, d], weights=w[y]) if y.any() else float('nan'):.2f}  "
              f"when not {np.average(P[~y, d], weights=w[~y]):.2f}")
