"""Decisions straight from the outcome model on validation households (all answers known), per
programme: is the problem the probabilities, the dollar estimates, or the decision rule?"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from benefits.population import features
from common import splits
from decisionintake.smoothing import OutcomeModel

pop, loss, meta, tr, va, te = splits()
train, val = pop.subset(tr), pop.subset(va)
m = OutcomeModel(features).fit(train)
p, v = m.predict(val, val.answers)
w = val.weight / val.weight.sum()
el = val.eligible
FR = 1000.0
print(f"{'programme':<13}{'base':>6}{'mean p|el':>10}{'mean p|not':>11}{'v true|el':>10}{'v pred|el':>10}{'thr':>6}{'recall':>8}{'false':>7}")
for d, name in enumerate(pop.decisions):
    y = el[:, d]
    vt = np.average(val.value[y, d], weights=w[y]) if y.any() else 0
    vp = np.average(v[y, d], weights=w[y]) if y.any() else 0
    thr = FR / (FR + max(vt, 120))
    rec = p[:, d] * np.maximum(v[:, d], 120) > (1 - p[:, d]) * FR
    rc = (w * (y & rec)).sum() / max((w * y).sum(), 1e-12)
    fr = (w * (~y & rec)).sum() / max((w * rec).sum(), 1e-12)
    print(f"{name:<13}{(w*y).sum():6.3f}{np.average(p[y, d], weights=w[y]):10.2f}{np.average(p[~y, d], weights=w[~y]):11.2f}"
          f"{vt:10.0f}{vp:10.0f}{thr:6.2f}{rc:8.3f}{fr:7.3f}")
rec = p * np.maximum(v, 120) > (1 - p) * FR
miss, frm = loss.realised(val)
frm = np.where(el, 0, FR)
real = (np.where(rec, 0, miss) + np.where(rec, frm, 0)).sum(1)
print('model decisions, all answers: loss', round(float(w @ real)), ' recall', round(float((w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum()), 3),
      ' false', round(float((w[:, None] * (~el & rec)).sum() / (w[:, None] * rec).sum()), 3))
for thr in [0.3, 0.5]:
    rec = p > thr
    real = (np.where(rec, 0, miss) + np.where(rec, frm, 0)).sum(1)
    print(f'plain threshold p>{thr}: loss', round(float(w @ real)), ' recall', round(float((w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum()), 3),
          ' false', round(float((w[:, None] * (~el & rec)).sum() / (w[:, None] * rec).sum()), 3))
