"""The wide-and-deep outcome model: it learns the toy rules, its calibration is folded in exactly, and the
session's incremental re-scoring gives the same scores as scoring every record again."""

import numpy as np
import pytest

from decisionintake import Session
from decisionintake.neural import NeuralOutcome, smooth_neural

from test_toy import toy


def onehot_features(pop, answers):
    """Wide part: one indicator per (question, option), plus an intercept in the last column."""
    sizes = [len(q.options) for q in pop.questions]
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return np.concatenate([answers + off, np.full((len(answers), 1), sum(sizes))], 1).astype(np.int64)


def n_onehot(pop):
    return sum(len(q.options) for q in pop.questions) + 1


@pytest.fixture(scope="module")
def fitted():
    pop, loss = toy()
    sm, model = smooth_neural(pop, onehot_features, n_onehot, folds=3, hidden=(32, 16), epochs=60, batch=256, dropout=0.1)
    return pop, loss, sm, model


def test_learns_the_rules_the_linear_part_cannot(fitted):
    pop, _, sm, model = fitted
    p, v = model.predict(pop, pop.answers)
    assert p.shape == pop.eligible.shape and np.all((p >= 0) & (p <= 1)) and np.all(v >= 0)
    # the toy rules are conjunctions (kids AND low income); one-hot terms alone can't express them
    acc = ((p > 0.5) == pop.eligible).mean(0)
    assert acc.min() > 0.98
    oof = ((sm.p_eligible > 0.5) == pop.eligible).mean(0)  # out of fold, so not memorised
    assert oof.min() > 0.98


def test_calibration_is_folded_into_the_weights_exactly(fitted):
    pop, *_ = fitted
    m = NeuralOutcome(onehot_features, n_onehot, hidden=(8, 4), epochs=2, holdout=0).fit(pop)
    z0, _ = m.raw(pop, pop.answers[:50])
    a, b = np.array([0.5, 2.0, 1.3]), np.array([0.2, -1.0, 0.0])
    m.calibrate(a, b)
    z1, _ = m.raw(pop, pop.answers[:50])
    assert np.allclose(z1, a * z0 + b, atol=1e-4)


def test_incremental_rescoring_matches_scoring_everything_again(fitted):
    pop, loss, _, model = fitted
    fast = Session(pop, loss, min_ess=10, outcome=model)
    for q, b in [(1, 2), (0, 1), (2, 0)]:
        fast.answer(q, b)
        slow = Session(pop, loss, min_ess=10, outcome=model)
        for qq, ev in fast.asked.items():
            slow.asked[qq] = ev
        slow.w, slow.p_now, slow.v_now = fast.w, fast.p_now.copy(), fast.v_now.copy()
        slow.h_now = np.full_like(fast.h_now, 255)  # forces every live record to be scored again
        slow._rescore()
        live = fast.w > 1e-9 * fast.w.max()
        assert np.allclose(fast.p_now[live], slow.p_now[live], atol=1e-6)
        assert fast.risk() == pytest.approx(slow.risk(), rel=1e-6)
