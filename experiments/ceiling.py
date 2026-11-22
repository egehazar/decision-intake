"""How good can decisions get if every question is answered? With and without income measured
against the poverty line. Validation split only."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
from benefits.population import build, load_table
from decisionintake import AskAll, LossModel
from decisionintake.simulate import UserModel, run_one, split_by_group

t = load_table()
for pq in [False, True]:
    pop, loss, meta = build(t, poverty_question=pq)
    rest, _ = split_by_group(pop, 0.15, seed=0)
    sub = pop.subset(rest)
    tr, va = split_by_group(sub, 0.12, seed=1)
    train, val = sub.subset(tr), sub.subset(va)
    rng = np.random.default_rng(5)
    sample = rng.choice(len(val), size=300, replace=False)
    for fr in [1000]:
        L = LossModel(np.full(10, float(fr)), loss.miss_floor)
        for m in [10, 30]:
            recs = []
            for i in sample:
                o = run_one(AskAll(), train, L, val.answers[i], UserModel(), np.random.default_rng(int(i)), min_ess=m)
                recs.append(o.recommend)
            rec = np.array(recs, bool)
            w = val.weight[sample] / val.weight[sample].sum()
            el = val.eligible[sample]
            miss, frm = L.realised(val.subset(sample))
            real = (np.where(rec, 0, miss) + np.where(rec, frm, 0)).sum(1)
            print(f"poverty question={pq!s:<5} fr={fr} min_ess={m:>3}: loss ${w @ real:6.0f}  "
                  f"programmes found {(w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum():.3f}  "
                  f"wrong recs {(w[:, None] * (~el & rec)).sum() / (w[:, None] * rec).sum():.3f}", flush=True)
