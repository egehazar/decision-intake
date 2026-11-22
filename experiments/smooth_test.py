"""Does a learned outcome model rank households better than the particles' raw outcomes?
Numpy logistic regression on one-hot answers plus income interactions. Validation split only."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from benefits.population import build, load_table
from decisionintake.simulate import split_by_group
from diagnose_ceiling import auc  # noqa


def features(pop, answers):
    oh_parts, blocks = [], {}
    for j, q in enumerate(pop.questions):
        b = len(q.options)
        m = np.zeros((len(answers), b)); m[np.arange(len(answers)), answers[:, j]] = 1
        blocks[q.id] = m; oh_parts.append(m)
    inc = blocks["earned_income"]
    inter = [np.einsum("ni,nj->nij", inc, blocks[k]).reshape(len(answers), -1)
             for k in ["household_size", "children", "senior", "disabled", "married", "social_security", "assets"]]
    inter.append(np.einsum("ni,nj->nij", blocks["children"], blocks["state"]).reshape(len(answers), -1))
    X = np.concatenate(oh_parts + inter + [np.ones((len(answers), 1))], axis=1)
    return X


def fit_logistic(X, y, w, l2=1.0, iters=400, lr=0.5):
    beta = np.zeros(X.shape[1]); v = np.zeros_like(beta)
    w = w / w.mean()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (w * (p - y)) / len(y) + l2 * beta / len(y)
        v = 0.9 * v + g
        beta -= lr * v
    return beta


t = load_table()
pop, loss, meta = build(t)
rest, _ = split_by_group(pop, 0.15, seed=0)
sub = pop.subset(rest)
tr, va = split_by_group(sub, 0.12, seed=1)
train, val = sub.subset(tr), sub.subset(va)
Xtr, Xva = features(train, train.answers), features(val, val.answers)
print("features", Xtr.shape)
for d, name in enumerate(train.decisions):
    t0 = time.time()
    beta = fit_logistic(Xtr, train.eligible[:, d].astype(float), train.weight, l2=5.0)
    p = 1 / (1 + np.exp(-(Xva @ beta)))
    print(f"   {name:<13} AUC {auc(p, val.eligible[:, d], val.weight):.3f}   ({time.time() - t0:.0f}s)", flush=True)
