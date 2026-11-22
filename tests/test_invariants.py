"""Invariants the engine must keep whatever it is told: the properties a refactor must not break."""

import numpy as np
import pytest

from decisionintake import GreedyTree, Session, VOIPolicy
from decisionintake.simulate import UserModel, run_one

from test_toy import toy


def _session(**kw):
    pop, loss = toy()
    return pop, loss, Session(pop, loss, **kw)


@pytest.mark.parametrize("min_ess", [1.0, 10.0])
def test_weights_stay_a_distribution_after_every_kind_of_evidence(min_ess):
    pop, loss, s = _session(min_ess=min_ess)
    s.answer(0, 1)
    s.observe(1, np.array([0.1, 0.6, 0.3, 0.0, 0.0]))
    s.observe(2, np.ones(2), source="dont_know")
    s.observe(3, np.array([0.0, 0.0]))  # impossible: logged, not applied
    assert np.all(s.w >= 0) and s.w.sum() == pytest.approx(1.0)
    assert np.isfinite(s.risk()) and s.risk() >= 0


def test_dont_know_changes_nothing_but_counts_as_asked():
    pop, loss, s = _session()
    w = s.w.copy()
    s.observe(1, np.ones(5), source="dont_know")
    assert np.allclose(s.w, w)
    assert 1 in s.asked  # not asked again


def test_impossible_answer_marks_the_question_asked_so_it_is_not_reasked():
    pop, loss, s = _session()
    s.observe(2, np.array([0.0, 0.0]))
    assert 2 in s.asked
    pol = VOIPolicy(question_cost=0.0)
    seen = set()
    for n in range(10):
        j = pol.next_question(s, n)
        if j is None:
            break
        assert j not in seen and j != 2
        seen.add(j)
        s.answer(j, 0)


def test_rescaling_the_likelihood_changes_nothing():
    pop, loss = toy()
    a, b = Session(pop, loss), Session(pop, loss)
    lik = np.array([0.1, 0.6, 0.3, 0.05, 0.0])
    a.observe(1, lik)
    b.observe(1, 7.0 * lik)
    assert np.allclose(a.w, b.w)


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_engine_never_exceeds_its_budget(budget):
    pop, loss = toy()
    o = run_one(VOIPolicy(question_cost=0.0, budget=budget), pop, loss, pop.answers[0], UserModel(),
                np.random.default_rng(0))
    assert o.n_questions <= budget


def test_engine_asks_nothing_when_questions_cost_more_than_any_mistake():
    pop, loss = toy()
    o = run_one(VOIPolicy(question_cost=1e9), pop, loss, pop.answers[0], UserModel(), np.random.default_rng(0))
    assert o.n_questions == 0


def test_raising_a_questions_cost_never_makes_the_engine_ask_it_sooner():
    pop, loss = toy()
    first_cheap = VOIPolicy(question_cost=50.0).next_question(Session(pop, loss), 0)
    pop.questions[first_cheap].cost = 1e6
    first_dear = VOIPolicy(question_cost=50.0).next_question(Session(pop, loss), 0)
    assert first_dear != first_cheap


def test_tree_with_unknown_backoff_keeps_asking_after_a_refusal():
    pop, loss = toy()
    tree = GreedyTree(train=pop, loss=loss, min_leaf=5, question_cost=1.0, decide_with_session=True,
                      backoff=True, unknown_backoff=True)
    s = Session(pop, loss)
    tree.start(s)
    j = tree.next_question(s, 0)
    s.observe(j, np.ones(len(pop.questions[j].options)), source="refused")
    tree.after_answer(s, j)
    assert s.policy_state["path"][-1] == (j, None)  # the refusal didn't narrow the branch
    nxt = tree.next_question(s, 1)
    assert nxt is not None and nxt != j
