"""Replace each record's raw outcome with a learned estimate given its answers.

A record's own yes/no outcome depends on details no question captures (exact income, the ages of each
person, ...). When the belief concentrates on a few records, averaging their raw outcomes is a noisy
nearest-neighbour vote. Replacing each record's outcome with E[outcome | its answers] keeps the
belief over answers intact and removes that noise (Rao-Blackwellization).

The model is a regularised logistic regression for eligibility and a ridge regression for the
amount when eligible, on features the domain adapter supplies. Training records get out-of-fold
predictions, so no record's smoothed outcome has seen its own label.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import Population


def fit_logistic(X, y, w, l2=5.0, iters=400, lr=0.5):
    beta = np.zeros(X.shape[1])
    v = np.zeros_like(beta)
    w = w / w.mean()
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        g = X.T @ (w * (p - y)) / len(y) + l2 * beta / len(y)
        v = 0.9 * v + g
        beta -= lr * v
    return beta


def fit_ridge(X, y, w, l2=10.0):
    xw = X * w[:, None]
    return np.linalg.solve(X.T @ xw + l2 * np.eye(X.shape[1]), xw.T @ y)


def predict_logistic(beta, X):
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


class OutcomeModel:
    def __init__(self, features):
        self.features = features  # (pop, answers) -> design matrix
        self.cls, self.reg, self.mean_value = [], [], []

    def fit(self, pop: Population):
        X = self.features(pop, pop.answers)
        w = pop.weight / pop.weight.mean()
        self.cls, self.reg, self.mean_value = [], [], []
        for d in range(len(pop.decisions)):
            y = pop.eligible[:, d].astype(float)
            self.cls.append(fit_logistic(X, y, w))
            el = pop.eligible[:, d]
            if el.sum() >= 20:
                self.reg.append(fit_ridge(X[el], np.log1p(pop.value[el, d]), w[el]))
                self.mean_value.append(float(np.average(pop.value[el, d], weights=w[el])))
            else:
                self.reg.append(None)
                self.mean_value.append(float(pop.value[el, d].mean()) if el.any() else 0.0)
        return self

    def predict(self, pop: Population, answers) -> tuple[np.ndarray, np.ndarray]:
        X = self.features(pop, answers)
        p = np.stack([predict_logistic(b, X) for b in self.cls], 1)
        v = np.stack([np.expm1(X @ r) if r is not None else np.full(len(X), m)
                      for r, m in zip(self.reg, self.mean_value)], 1)
        return p, np.maximum(v, 0.0)


def smooth(pop: Population, features, folds: int = 5, seed: int = 0) -> tuple[Population, OutcomeModel]:
    """Population whose outcomes are out-of-fold model estimates; also the model fitted on all of it."""
    rng = np.random.default_rng(seed)
    groups = pop.group if pop.group is not None else np.arange(len(pop))
    uniq = np.unique(groups)
    fold_of_group = dict(zip(uniq.tolist(), rng.integers(0, folds, len(uniq)).tolist()))
    fold = np.array([fold_of_group[g] for g in groups.tolist()])
    p = np.zeros(pop.eligible.shape)
    v = np.zeros(pop.value.shape)
    for k in range(folds):
        tr, te = fold != k, fold == k
        m = OutcomeModel(features).fit(pop.subset(np.flatnonzero(tr)))
        p[te], v[te] = m.predict(pop, pop.answers[te])
    full = OutcomeModel(features).fit(pop)
    out = replace(pop)
    out.p_eligible, out.value_if_eligible = p, v
    out._onehot = None
    return out, full
