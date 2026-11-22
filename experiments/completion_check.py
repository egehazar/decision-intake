"""Ask-all on validation, with and without completion (re-scoring on the person's own answers)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from common import outcome_model, smoothed, splits
from decisionintake import AskAll, LossModel
from decisionintake.simulate import UserModel, run_one

pop, loss, meta, tr, va, te = splits()
train = smoothed(pop, tr, "train_tune")
model = outcome_model(pop, tr)
val = pop.subset(va)
L = LossModel(np.full(10, 1000.0), loss.miss_floor)
sample = np.random.default_rng(5).choice(len(val), size=300, replace=False)
w = val.weight[sample] / val.weight[sample].sum()
el = val.eligible[sample]
miss, frm = L.realised(val.subset(sample))
for use, m in [(False, 1), (True, 1), (True, 10), (True, 30)]:
    t0 = time.time()
    rec = np.array([run_one(AskAll(), train, L, val.answers[i], UserModel(), np.random.default_rng(int(i)),
                            min_ess=m, outcome=model if use else None).recommend for i in sample], bool)
    real = (np.where(rec, 0, miss) + np.where(rec, frm, 0)).sum(1)
    print(f"completion={use!s:<5} min_ess={m:>3}: loss ${w @ real:6.0f}  found {(w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum():.3f}"
          f"  wrong {(w[:, None] * (~el & rec)).sum() / (w[:, None] * rec).sum():.3f}  ({time.time() - t0:.0f}s)", flush=True)
