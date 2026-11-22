"""A wide-and-deep outcome model in plain numpy, a drop-in for smoothing.OutcomeModel.

The linear model (smoothing.py) only knows the interactions the adapter lists by hand: income against
household size, children, age, and so on. Eligibility rules nest further than that (income against
household size against state, employer coverage against income for the ACA credit), so a small
multi-task network learns the rest from the one-hot answers, next to the same hand-listed linear terms:

    logit_d = wide_d(listed indicators) + deep_d(one-hot answers)
    dollars_d = the same two parts on a second head, fitted on households eligible for d

All ten programmes share the hidden layers. Probabilities are Platt-scaled on out-of-fold predictions
and the scaling is folded into the last layer, so `predict` returns calibrated probabilities like the
linear model does. Inference is a gather-sum for the wide part and two small matrix products for the
deep part.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import Population
from .smoothing import fit_platt


def _onehot_base(pop: Population, answers: np.ndarray) -> np.ndarray:
    sizes = [len(q.options) for q in pop.questions]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    X = np.zeros((len(answers), int(sum(sizes))), dtype=np.float32)
    rows = np.arange(len(answers))
    for j in range(len(sizes)):
        X[rows, offsets[j] + answers[:, j]] = 1.0
    return X


def _dense(idx: np.ndarray, F: int) -> np.ndarray:
    X = np.zeros((len(idx), F), dtype=np.float32)
    rows = np.repeat(np.arange(len(idx)), idx.shape[1])
    np.add.at(X, (rows, idx.ravel()), 1.0)
    return X


class NeuralOutcome:
    def __init__(self, features, n_features, hidden=(128, 64), epochs=40, batch=1024, lr=2e-3,
                 l2=1e-5, value_weight=0.5, seed=0, wd=0.0, dropout=0.0, holdout=0.1):
        self.features, self.n_features = features, n_features
        self.hidden, self.epochs, self.batch, self.lr = hidden, epochs, batch, lr
        self.l2, self.value_weight, self.seed = l2, value_weight, seed
        self.wd, self.dropout, self.holdout = wd, dropout, holdout

    # ------------------------------------------------------------------------------------------
    def _forward(self, Xw, Xb):
        h1 = np.maximum(Xb @ self.W1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0)
        out = h2 @ self.W3 + Xw @ self.Ww
        return h1, h2, out

    def fit(self, pop: Population):
        rng = np.random.default_rng(self.seed)
        F = self.n_features(pop)
        idx = self.features(pop, pop.answers)
        Xb_all = _onehot_base(pop, pop.answers)
        D = len(pop.decisions)
        y = pop.eligible.astype(np.float32)
        val = pop.value.astype(np.float32)
        self.scale = np.array([val[y[:, d] > 0, d].std() if y[:, d].sum() > 1 else 1.0 for d in range(D)],
                              dtype=np.float32) + 1.0
        t = val / self.scale
        w = (pop.weight / pop.weight.mean()).astype(np.float32)
        H1, H2 = self.hidden
        Q1 = Xb_all.shape[1]
        self.W1 = (rng.standard_normal((Q1, H1)) * np.sqrt(2 / 18)).astype(np.float32)
        self.b1 = np.zeros(H1, np.float32)
        self.W2 = (rng.standard_normal((H1, H2)) * np.sqrt(2 / H1)).astype(np.float32)
        self.b2 = np.zeros(H2, np.float32)
        self.W3 = (rng.standard_normal((H2, 2 * D)) * 0.01).astype(np.float32)
        self.Ww = np.zeros((F, 2 * D), np.float32)
        base = y.mean(0).clip(1e-4, 1 - 1e-4)
        self.Ww[-1, :D] = np.log(base / (1 - base))  # last wide column is the intercept
        self.Ww[-1, D:] = (t * y).sum(0) / np.maximum(y.sum(0), 1)
        params = ["W1", "b1", "W2", "b2", "W3", "Ww"]
        m = {k: np.zeros_like(getattr(self, k)) for k in params}
        v = {k: np.zeros_like(getattr(self, k)) for k in params}
        step = 0
        # early stopping on a hold-out of whole source households (groups) inside the training data
        groups = pop.group if pop.group is not None else np.arange(len(pop))
        uniq = np.unique(groups)
        held = set(rng.choice(uniq, size=int(self.holdout * len(uniq)), replace=False).tolist()) if self.holdout else set()
        is_held = np.array([g in held for g in groups.tolist()])
        fit_rows, held_rows = np.flatnonzero(~is_held), np.flatnonzero(is_held)
        best, best_score = None, np.inf
        n = len(fit_rows)
        keep = 1.0 - self.dropout
        for epoch in range(self.epochs):
            lr = self.lr * 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
            perm = fit_rows[rng.permutation(n)]
            for s in range(0, n, self.batch):
                bi = perm[s : s + self.batch]
                Xw, Xb = _dense(idx[bi], F), Xb_all[bi]
                h1 = np.maximum(Xb @ self.W1 + self.b1, 0)
                if self.dropout:
                    d1 = (rng.random(h1.shape) < keep).astype(np.float32) / keep
                    h1 = h1 * d1
                h2 = np.maximum(h1 @ self.W2 + self.b2, 0)
                if self.dropout:
                    d2 = (rng.random(h2.shape) < keep).astype(np.float32) / keep
                    h2 = h2 * d2
                out = h2 @ self.W3 + Xw @ self.Ww
                z, u = out[:, :D], out[:, D:]
                p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
                wb = w[bi][:, None] / len(bi)
                gz = wb * (p - y[bi])
                gu = self.value_weight * 2 * wb * y[bi] * (u - t[bi])
                g = np.concatenate([gz, gu], 1).astype(np.float32)
                grads = {"Ww": Xw.T @ g, "W3": h2.T @ g}
                g2 = (g @ self.W3.T) * (h2 > 0)
                if self.dropout:
                    g2 = g2 * d2
                grads["W2"], grads["b2"] = h1.T @ g2, g2.sum(0)
                g1 = (g2 @ self.W2.T) * (h1 > 0)
                if self.dropout:
                    g1 = g1 * d1
                grads["W1"], grads["b1"] = Xb.T @ g1, g1.sum(0)
                step += 1
                for k in params:
                    P = getattr(self, k)
                    gk = grads[k] + (self.l2 * P if k[0] == "W" else 0)
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v[k] = 0.999 * v[k] + 0.001 * gk * gk
                    mh, vh = m[k] / (1 - 0.9**step), v[k] / (1 - 0.999**step)
                    if self.wd and k[0] == "W":
                        P *= 1 - lr * self.wd
                    P -= lr * mh / (np.sqrt(vh) + 1e-8)
            if len(held_rows):
                score = self._score(pop, held_rows, idx, Xb_all, F, y, t, w, D)
                if score < best_score:
                    best_score, best = score, {k: getattr(self, k).copy() for k in params}
                    self.best_epoch = epoch + 1
        if best is not None:
            for k in params:
                setattr(self, k, best[k])
        return self

    def _score(self, pop, rows, idx, Xb_all, F, y, t, w, D):
        """Weighted log loss plus the value term, on held-out rows (the training objective)."""
        out = []
        for s in range(0, len(rows), 8192):
            r = rows[s : s + 8192]
            _, _, o = self._forward(_dense(idx[r], F), Xb_all[r])
            out.append(o)
        o = np.concatenate(out)
        z, u = o[:, :D], o[:, D:]
        p = np.clip(1 / (1 + np.exp(-np.clip(z, -30, 30))), 1e-7, 1 - 1e-7)
        ww = w[rows][:, None]
        yy = y[rows]
        bce = -(ww * (yy * np.log(p) + (1 - yy) * np.log(1 - p))).sum() / ww.sum()
        mse = (ww * yy * (u - t[rows]) ** 2).sum() / ww.sum()
        return float(bce + self.value_weight * mse)

    def raw(self, pop: Population, answers: np.ndarray):
        """(logits, dollars) before Platt scaling of any kind is applied (it lives in the weights)."""
        idx = self.features(pop, answers)
        D = len(pop.decisions)
        wide = np.zeros((len(answers), 2 * D), np.float32)
        for k in range(idx.shape[1]):
            wide += self.Ww[idx[:, k]]
        Xb = _onehot_base(pop, answers)
        h1 = np.maximum(Xb @ self.W1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0)
        out = h2 @ self.W3 + wide
        return out[:, :D].astype(np.float64), (out[:, D:] * self.scale).astype(np.float64)

    def predict(self, pop: Population, answers: np.ndarray):
        z, v = self.raw(pop, answers)
        return 1 / (1 + np.exp(-np.clip(z, -30, 30))), np.maximum(v, 0.0)

    def calibrate(self, a: np.ndarray, b: np.ndarray):
        """Fold p = sigmoid(a*z + b) into the last layer (per programme)."""
        D = len(a)
        self.W3[:, :D] *= a.astype(np.float32)
        self.Ww[:, :D] *= a.astype(np.float32)
        self.Ww[-1, :D] += b.astype(np.float32)
        return self


def smooth_neural(pop: Population, features, n_features, folds: int = 5, seed: int = 0, **kw):
    """As smoothing.smooth: out-of-fold calibrated outcomes for pop, and the calibrated full model."""
    rng = np.random.default_rng(seed)
    groups = pop.group if pop.group is not None else np.arange(len(pop))
    uniq = np.unique(groups)
    fold_of_group = dict(zip(uniq.tolist(), rng.integers(0, folds, len(uniq)).tolist()))
    fold = np.array([fold_of_group[g] for g in groups.tolist()])
    D = len(pop.decisions)
    z = np.zeros(pop.eligible.shape)
    v = np.zeros(pop.value.shape)
    for k in range(folds):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        m = NeuralOutcome(features, n_features, seed=seed + k, **kw).fit(pop.subset(tr))
        z[te], v[te] = m.raw(pop, pop.answers[te])
    full = NeuralOutcome(features, n_features, seed=seed + folds, **kw).fit(pop)
    w = pop.weight / pop.weight.mean()
    a, b = np.ones(D), np.zeros(D)
    p = np.zeros(pop.eligible.shape)
    for d in range(D):
        a[d], b[d] = fit_platt(z[:, d], pop.eligible[:, d].astype(float), w)
        p[:, d] = 1 / (1 + np.exp(-(a[d] * z[:, d] + b[d])))
    full.calibrate(a, b)
    full.platt = list(zip(a.tolist(), b.tolist()))
    out = replace(pop)
    out.p_eligible, out.value_if_eligible = p, np.maximum(v, 0.0)
    out._onehot = None
    return out, full
