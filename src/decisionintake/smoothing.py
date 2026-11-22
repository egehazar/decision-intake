"""A learned outcome model, used two ways.

1. Smoothing. A record's own yes/no outcome depends on details no question captures. Averaging the
   raw outcomes of the few records a belief concentrates on is a noisy nearest-neighbour vote;
   replacing each record's outcome with E[outcome | answers] removes that noise (Rao-Blackwellization).

2. Completion. During an interview, the outcome for a candidate record is evaluated on a hybrid:
   the person's own answers where known, the record's answers only for questions still open. Records
   then only supply plausible values for unknowns, so a neighbour from another income bracket can't
   bring that bracket's eligibility along.

The model is additive in binary indicator features (one-hot answers and products of them), given by
the domain adapter as column indices: logit = sum of beta over a row's active columns. That makes
re-scoring every record after each answer a table lookup. Eligibility: regularised logistic
regression. Dollars if eligible: ridge regression on dollars.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import Population


def dense(idx: np.ndarray, n_features: int) -> np.ndarray:
    X = np.zeros((len(idx), n_features))
    rows = np.repeat(np.arange(len(idx)), idx.shape[1])
    X[rows, idx.ravel()] = 1.0
    return X


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


def fit_platt(z, y, w, iters=50):
    """p = sigmoid(a*z + b) by Newton's method (two parameters, weighted)."""
    X = np.stack([z, np.ones_like(z)], 1)
    theta = np.array([1.0, 0.0])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ theta)))
        g = X.T @ (w * (p - y))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-8:
            break
    return float(theta[0]), float(theta[1])


def fit_ridge(X, y, w, l2=10.0):
    xw = X * w[:, None]
    return np.linalg.solve(X.T @ xw + l2 * np.eye(X.shape[1]), xw.T @ y)


class OutcomeModel:
    """features(pop, answers) -> (N, K) int column indices of active indicators; n_features(pop)."""

    def __init__(self, features, n_features):
        self.features, self.n_features = features, n_features
        self.B = None  # (F, D) logistic coefficients
        self.R = None  # (F, D) ridge coefficients on log1p(dollars)

    def fit(self, pop: Population):
        F = self.n_features(pop)
        X = dense(self.features(pop, pop.answers), F)
        w = pop.weight / pop.weight.mean()
        D = len(pop.decisions)
        self.B, self.R = np.zeros((F, D)), np.zeros((F, D))
        for d in range(D):
            y = pop.eligible[:, d]
            self.B[:, d] = fit_logistic(X, y.astype(float), w)
            if y.sum() >= 20:
                self.R[:, d] = fit_ridge(X[y], pop.value[y, d], w[y])
            else:
                self.R[-1, d] = pop.value[y, d].mean() if y.any() else 0.0  # last column: intercept
        return self

    def predict(self, pop: Population, answers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self.features(pop, answers)
        logit = self.B[idx].sum(1)  # (N, D)
        p = 1.0 / (1.0 + np.exp(-logit))
        v = np.maximum(self.R[idx].sum(1), 0.0)
        return p, v


def smooth(pop: Population, features, n_features, folds: int = 5, seed: int = 0) -> tuple[Population, OutcomeModel]:
    """Population whose outcomes are out-of-fold model estimates, and the model fitted on all of it.

    Platt scaling of each programme's logit is fitted on the out-of-fold predictions and folded into
    the full model, so its probabilities can be taken at face value (the full model is re-used on the
    households it was fitted on, which makes raw probabilities too sharp). Dollars are fitted on the
    dollar scale, so their average is unbiased without a correction (a log-scale fit, tried first,
    estimates medians; its smearing correction then overshot by up to 40%).
    The design's last column must be the intercept."""
    rng = np.random.default_rng(seed)
    groups = pop.group if pop.group is not None else np.arange(len(pop))
    uniq = np.unique(groups)
    fold_of_group = dict(zip(uniq.tolist(), rng.integers(0, folds, len(uniq)).tolist()))
    fold = np.array([fold_of_group[g] for g in groups.tolist()])
    D = len(pop.decisions)
    logit = np.zeros(pop.eligible.shape)
    logv = np.zeros(pop.value.shape)  # out-of-fold dollars
    for k in range(folds):
        tr, te = fold != k, fold == k
        m = OutcomeModel(features, n_features).fit(pop.subset(np.flatnonzero(tr)))
        idx = m.features(pop, pop.answers[te])
        logit[te] = m.B[idx].sum(1)
        logv[te] = m.R[idx].sum(1)
    full = OutcomeModel(features, n_features).fit(pop)
    w = pop.weight / pop.weight.mean()
    p = np.zeros(pop.eligible.shape)
    v = np.zeros(pop.value.shape)
    full.platt = []
    for d in range(D):
        y = pop.eligible[:, d]
        a, b = fit_platt(logit[:, d], y.astype(float), w)
        full.B[:, d] *= a
        full.B[-1, d] += b
        p[:, d] = 1.0 / (1.0 + np.exp(-(a * logit[:, d] + b)))
        v[:, d] = np.maximum(logv[:, d], 0.0)
        full.platt.append((float(a), float(b)))
    out = replace(pop)
    out.p_eligible, out.value_if_eligible = p, v
    out._onehot = None
    return out, full
