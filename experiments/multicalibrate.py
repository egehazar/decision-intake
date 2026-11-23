"""v3 outcome model: v2 plus group-calibration terms, fitted as a second stage (DECISIONS D32).

v2's network is well calibrated overall and by income band, but wrong in thin groups whose answers the
rules treat sharply. A lone parent with one child and no earnings gets the EITC with probability 0.81
from v2; in PolicyEngine's data the rate for that group is 0. Its wide part crosses income with one
other answer at a time, so "one adult, one child, this income" is out of its reach, and the deep part's
regularisation smooths the cliff away.

The fix is multicalibration (Hebert-Johnson et al. 2018) over groups the programme rules are built on:
add one logit offset per group cell and programme, on top of v2's logits, fitted by penalised logistic
regression so a cell with few households stays close to v2 and a well-populated cell follows its data.
The offsets are wide-part terms, so the bundle format, the service and the browser engine are unchanged.

Fitting is stacked: the offsets are fitted on v2's out-of-fold logits (honest residuals), the penalty is
chosen on the validation split, then refitted on train + validation and added to the full v2 model.

usage: python experiments/multicalibrate.py select     # choose the penalty on validation; audit
       python experiments/multicalibrate.py build      # refit on train + validation, write bundles/benefits_v3.npz
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")
V2 = os.path.join(ROOT, "bundles", "benefits_v2.npz")
V3 = os.path.join(ROOT, "bundles", "benefits_v3.npz")
OOF = os.path.join(ROOT, "data", "cache", "neural_rest_dropout0.4_hidden(512x256)_wd0.3.npz")

# Groups the rules are built on (fixed before looking at any test household):
FAMILIES = [
    ("tax_unit_income", ["household_size", "children", "married", "earned_income"]),  # EITC, CTC, SNAP, meals
    ("state_income", ["state", "earned_income"]),  # Medicaid expansion, SNAP's state limits, TANF
    ("ssi", ["senior", "disabled", "social_security", "other_income", "assets"]),  # SSI's income and asset tests
    ("wic", ["infant_or_pregnant", "child_under5", "household_size", "earned_income"]),  # WIC's categories
    ("aca", ["employer_insurance", "household_size", "earned_income"]),  # the premium credit's firewall
    ("self_employed", ["self_employed", "children", "earned_income"]),  # earnings the earned-income question misses
]
GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def family_terms(questions, offset: int) -> tuple[list[dict], int]:
    """TermFeatures terms for the families, starting at column `offset`; returns (terms, columns used)."""
    qi = {q.id: j for j, q in enumerate(questions)}
    terms, pos = [], offset
    for name, ids in FAMILIES:
        sizes = [len(questions[qi[k]].options) for k in ids]
        strides = [int(np.prod(sizes[i + 1:])) for i in range(len(sizes))]
        terms.append({"qs": [qi[k] for k in ids], "strides": strides, "off": pos, "family": name})
        pos += int(np.prod(sizes))
    return terms, pos - offset


def cells(questions, answers) -> list[np.ndarray]:
    terms, _ = family_terms(questions, 0)
    return [sum(answers[:, q].astype(np.int64) * s for q, s in zip(t["qs"], t["strides"])) for t in terms]


def sizes(questions) -> list[int]:
    terms, total = family_terms(questions, 0)
    offs = [t["off"] for t in terms] + [total]
    return [offs[i + 1] - offs[i] for i in range(len(terms))]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(z0, y, w, C, nC, lam, sweeps=40):
    """Penalised logistic regression of y on per-cell offsets, with z0 as a fixed offset; every programme at
    once. Block-coordinate Newton over the families: each cell's offset is a one-dimensional problem."""
    D = y.shape[1]
    U = [np.zeros((n, D)) for n in nC]
    z = z0.copy()
    for _ in range(sweeps):
        for f, (c, n) in enumerate(zip(C, nC)):
            p = 1 / (1 + np.exp(-z))
            g = np.stack([np.bincount(c, weights=w * (p[:, d] - y[:, d]), minlength=n) for d in range(D)], 1) + lam * U[f]
            h = np.stack([np.bincount(c, weights=w * p[:, d] * (1 - p[:, d]), minlength=n) for d in range(D)], 1) + lam
            step = g / h
            U[f] -= step
            z -= step[c]
    return U


def apply(z0, C, U):
    z = z0.copy()
    for c, u in zip(C, U):
        z += u[c]
    return z


def logloss(z, y, w):
    p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    return -(w[:, None] * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum(0) / w.sum()


def load():
    from common import splits
    from decisionintake import bundle

    pop, loss, meta, tr, va, test = splits()
    rest = np.sort(np.concatenate([tr, va]))
    b = bundle.load(V2)
    assert np.array_equal(b.pop.answers, pop.answers[rest]), "the v2 bundle's population is not train + validation"
    z = np.load(OOF)
    assert len(z["p"]) == len(rest)
    y = pop.eligible[rest].astype(float)
    w = pop.weight[rest] / pop.weight[rest].mean()
    return pop, rest, tr, va, b, logit(z["p"]), y, w


def select():
    pop, rest, tr, va, b, z0, y, w = load()
    q = b.pop.questions
    C, nC = cells(q, b.pop.answers), sizes(q)
    itr, iva = np.searchsorted(rest, tr), np.searchsorted(rest, va)
    sub = lambda idx: [c[idx] for c in C]  # noqa: E731
    base = logloss(z0[iva], y[iva], w[iva])
    print(f"validation log loss, v2 out-of-fold: {base.mean():.5f}")
    best = None
    for lam in GRID:
        U = fit(z0[itr], y[itr], w[itr], sub(itr), nC, lam)
        ll = logloss(apply(z0[iva], sub(iva), U), y[iva], w[iva])
        print(f"  penalty {lam:6.1f}: {ll.mean():.5f}  ({(ll.mean() / base.mean() - 1) * 100:+.2f}%)  worst programme "
              f"{(ll / base - 1).max() * 100:+.2f}%")
        if best is None or ll.mean() < best[1].mean():
            best = (lam, ll, U)
    lam, ll, U = best
    print(f"chosen penalty {lam}; per programme change: "
          + ", ".join(f"{d} {(a / c - 1) * 100:+.1f}%" for d, a, c in zip(b.pop.decisions, ll, base)))
    audit(b, z0, y, w, C, itr, iva, U)
    json.dump({"penalty": lam, "validation_logloss_v2": base.tolist(), "validation_logloss_v3": ll.tolist()},
              open(os.path.join(ROOT, "results", "multicalibrate_select.json"), "w"), indent=1)


def audit(b, z0, y, w, C, itr, iva, U):
    """Rule-clear groups: tax-unit x income cells where at least 30 training households all (<=2% or >=98%)
    agree. On validation households in those cells, how far is the prediction from what the data says?"""
    c = C[0]
    zv = z0[iva]
    p2 = 1 / (1 + np.exp(-zv))
    p3 = 1 / (1 + np.exp(-apply(zv, [cc[iva] for cc in C], U)))
    print("rule-clear groups (training says 0-2% or 98-100%), validation households in them:")
    for d, name in enumerate(b.pop.decisions):
        n = np.bincount(c[itr], minlength=c.max() + 1)
        r = np.bincount(c[itr], weights=w[itr] * y[itr, d], minlength=c.max() + 1) / np.maximum(
            np.bincount(c[itr], weights=w[itr], minlength=c.max() + 1), 1e-9)
        clear = (n >= 30) & ((r <= 0.02) | (r >= 0.98))
        m = clear[c[iva]]
        if m.sum() < 20:
            continue
        e2 = np.average(np.abs(p2[m, d] - y[iva][m, d]), weights=w[iva][m])
        e3 = np.average(np.abs(p3[m, d] - y[iva][m, d]), weights=w[iva][m])
        bad2 = np.average(np.abs(p2[m, d] - y[iva][m, d]) > 0.5, weights=w[iva][m])
        bad3 = np.average(np.abs(p3[m, d] - y[iva][m, d]) > 0.5, weights=w[iva][m])
        print(f"  {name:13s} {m.sum():5d} households: mean error {e2:.3f} -> {e3:.3f}; wrong side of 50% {bad2:.1%} -> {bad3:.1%}")


def build():
    from decisionintake import bundle
    from decisionintake.neural import NeuralOutcome

    sel = json.load(open(os.path.join(ROOT, "results", "multicalibrate_select.json")))
    pop, rest, tr, va, b, z0, y, w = load()
    q = b.pop.questions
    C, nC = cells(q, b.pop.answers), sizes(q)
    U = fit(z0, y, w, C, nC, sel["penalty"])
    meta = json.loads(bytes(np.load(V2)["meta"]).decode())
    old_terms = meta["terms"]
    assert old_terms[-1]["qs"] == [] and old_terms[-1]["off"] == meta["n_features"] - 1, "intercept must come last"
    F2 = meta["n_features"] - 1  # columns before the intercept
    new_terms, added = family_terms(q, F2)
    terms = old_terms[:-1] + [{k: v for k, v in t.items() if k != "family"} for t in new_terms] + [
        {"qs": [], "strides": [], "off": F2 + added}]
    m2 = b.model
    D = len(b.pop.decisions)
    Ww = np.zeros((F2 + added + 1, 2 * D), np.float32)
    Ww[:F2] = m2.Ww[:F2]
    Ww[F2:F2 + added, :D] = np.concatenate(U).astype(np.float32)
    Ww[-1] = m2.Ww[-1]
    tf = bundle.TermFeatures(terms, F2 + added + 1)
    m3 = NeuralOutcome(tf, tf.n_features)
    for k in ["W1", "b1", "W2", "b2", "W3", "scale"]:
        setattr(m3, k, getattr(m2, k))
    m3.Ww = Ww
    info = dict(b.info)
    info["name"] = "DecisionIntake v3, US benefits screening"
    info["model"] = (b.info.get("model", "") + "; plus group calibration over the groups the rules use "
                     f"(DECISIONS D32, penalty {sel['penalty']:g})")
    bundle.save(V3, b.pop, b.loss, m3, terms, b.labels, info)
    b3 = bundle.load(V3)
    p2, v2 = m2.predict(b.pop, b.pop.answers[:500])
    p3, v3 = b3.model.predict(b3.pop, b3.pop.answers[:500])
    assert np.allclose(v2, v3, atol=1e-3), "the value head must be unchanged"
    print(f"wrote {V3} ({os.path.getsize(V3) / 1e6:.2f} MB): {added} group columns, penalty {sel['penalty']:g}; "
          f"mean |p change| on 500 records {np.abs(p3 - p2).mean():.4f}")


if __name__ == "__main__":
    {"select": select, "build": build}[sys.argv[1]]()
