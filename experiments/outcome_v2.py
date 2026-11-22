"""Outcome model v2 vs the linear model, on the VALIDATION split only (test is not touched).

Both are trained on the train split with out-of-fold Platt scaling. For every validation household,
with all 18 answers known: per-programme log loss and AUC, and the decision loss of deciding straight
from the model ($1,000 per wrong recommendation, misses at programme value). That decision loss is the
floor any question-asking policy works against.

usage: python experiments/outcome_v2.py [epochs] [hidden1] [hidden2]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from benefits.population import features, n_features  # noqa: E402
from common import splits  # noqa: E402

from decisionintake.neural import smooth_neural  # noqa: E402
from decisionintake.smoothing import smooth  # noqa: E402

FR = 1000.0


def auc(s, y, w):
    o = np.argsort(s)
    s, y, w = s[o], y[o], w[o]
    pos, neg = (w * y).sum(), (w * (1 - y)).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    cneg = np.cumsum(w * (1 - y))
    return float((w * y * (cneg - 0.5 * w * (1 - y))).sum() / (pos * neg))


def report(name, model, val, loss_floor):
    p, v = model.predict(val, val.answers)
    w = val.weight / val.weight.sum()
    y = val.eligible
    miss = np.where(y, np.maximum(val.value, loss_floor), 0.0)
    fr = np.where(y, 0.0, FR)
    rec = p * np.maximum(v, loss_floor) > (1 - p) * FR
    loss = (np.where(rec, fr, miss)).sum(1)
    wrong = (rec & ~y).sum() / max(rec.sum(), 1)
    found = (rec & y).sum() / max(y.sum(), 1)
    ll = -(w[:, None] * (y * np.log(np.clip(p, 1e-9, 1)) + (~y) * np.log(np.clip(1 - p, 1e-9, 1)))).sum(0)
    print(f"\n{name}: decision loss ${w @ loss:,.0f}/household, {100 * wrong:.0f}% of recommendations wrong, "
          f"{100 * found:.0f}% of eligible programmes found, log loss {ll.sum():.3f}")
    print("  " + "  ".join(f"{d}:{auc(p[:, k], y[:, k].astype(float), w):.3f}" for k, d in enumerate(val.decisions)))
    return loss, ll


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    h = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (128, 64)
    pop, loss, meta, tr, va, test = splits()
    train, val = pop.subset(tr), pop.subset(va)
    floor = loss.miss_floor[None, :]
    print(f"train {len(train)}, validation {len(val)}")
    t0 = time.time()
    _, lin = smooth(train, features, n_features)
    print(f"linear fitted in {time.time() - t0:.0f}s")
    t0 = time.time()
    _, neu = smooth_neural(train, features, n_features, epochs=epochs, hidden=h)
    print(f"neural fitted in {time.time() - t0:.0f}s (epochs {epochs}, hidden {h})")
    l0, _ = report("linear", lin, val, floor)
    l1, _ = report("neural", neu, val, floor)
    w = val.weight / val.weight.sum()
    d = l1 - l0
    rng = np.random.default_rng(5)
    bs = [np.average(d[b], weights=w[b]) for b in rng.choice(len(d), size=(2000, len(d)))]
    print(f"\nneural minus linear decision loss: {w @ d:,.0f} [{np.percentile(bs, 2.5):,.0f}, {np.percentile(bs, 97.5):,.0f}]")


if __name__ == "__main__":
    main()
