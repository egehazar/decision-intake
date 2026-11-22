"""A made-up domain where the right answer is known by construction."""

import numpy as np
import pytest

from decisionintake import AskAll, FixedOrder, GreedyTree, LossModel, Population, Question, Session, VOIPolicy
from decisionintake.simulate import UserModel, evaluate, run_one, split_by_group


def toy(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    qs = [
        Question("kids", "How many children live with you?", ("0", "1", "2+")),
        Question("income", "Monthly income?", ("<1k", "1-2k", "2-3k", "3-4k", "4k+")),
        Question("senior", "Anyone 65 or older?", ("no", "yes")),
        Question("pets", "Any pets?", ("no", "yes")),  # irrelevant to every decision
    ]
    kids = rng.choice(3, n, p=[0.5, 0.25, 0.25])
    inc = rng.choice(5, n)
    sen = rng.choice(2, n, p=[0.8, 0.2])
    pets = rng.choice(2, n)
    ans = np.stack([kids, inc, sen, pets], 1)
    el = np.stack([(kids >= 1) & (inc <= 2), (sen == 1) & (inc <= 1), (kids >= 1) & (inc >= 1) & (inc <= 3)], 1)
    val = el * np.array([3000.0, 5000.0, 1000.0])
    pop = Population(qs, ["child", "senior", "credit"], ans, np.ones(n), el, val, group=np.arange(n) // 2)
    loss = LossModel(false_rec_cost=np.array([200.0, 200.0, 200.0]), miss_floor=np.zeros(3))
    return pop, loss


def test_voi_is_nonnegative_and_zero_for_irrelevant_question():
    pop, loss = toy()
    s = Session(pop, loss)
    vals = dict(s.voi_table())
    assert all(v >= -1e-9 for v in vals.values())
    assert abs(vals[3]) < 1e-9  # pets
    assert max(vals, key=vals.get) in (0, 1)  # kids or income matter most


def test_full_answers_give_exact_decisions():
    pop, loss = toy()
    train_i, test_i = split_by_group(pop, 0.25)
    train, test = pop.subset(train_i), pop.subset(test_i)
    r = evaluate(lambda: AskAll(), train, test, loss, UserModel(), limit=300)
    assert r.programme_recall == pytest.approx(1.0) and r.false_rec_rate == pytest.approx(0.0)


def test_engine_matches_full_questionnaire_with_fewer_questions():
    pop, loss = toy()
    train_i, test_i = split_by_group(pop, 0.25)
    train, test = pop.subset(train_i), pop.subset(test_i)
    full = evaluate(lambda: AskAll(), train, test, loss, UserModel(), limit=300)
    eng = evaluate(lambda: VOIPolicy(question_cost=1.0), train, test, loss, UserModel(), limit=300)
    assert eng.mean_loss <= full.mean_loss + 1e-6
    assert eng.mean_questions < full.mean_questions  # never needs "pets", often skips "senior"


def test_soft_evidence_leaves_more_uncertainty_than_a_crisp_answer():
    pop, loss = toy()
    crisp, soft = Session(pop, loss), Session(pop, loss)
    crisp.answer(1, 1)
    soft.observe(1, np.array([0.1, 0.6, 0.3, 0.0, 0.0]))
    assert soft.risk() > crisp.risk()
    assert soft.ess() > crisp.ess()


def test_incompatible_evidence_is_logged_not_applied():
    pop, loss = toy()
    s = Session(pop, loss)
    s.answer(0, 0)  # no kids
    w = s.w.copy()
    s.observe(0, np.array([0.0, 0.0, 0.0]))  # evidence nothing can produce
    assert s.log[-1].get("conflict") and np.allclose(s.w, w)


def test_tree_matches_engine_on_clean_answers_in_a_deterministic_domain():
    pop, loss = toy()
    train_i, test_i = split_by_group(pop, 0.25)
    train, test = pop.subset(train_i), pop.subset(test_i)
    tree = evaluate(lambda: GreedyTree(train=train, loss=loss, min_leaf=5, question_cost=1.0), train, test, loss, UserModel(), limit=300)
    assert tree.programme_recall == pytest.approx(1.0) and tree.false_rec_rate == pytest.approx(0.0)


def test_fixed_order_puts_the_irrelevant_question_last():
    pop, loss = toy()
    f = FixedOrder.fit(pop, loss)
    assert f.order[-1] == 3


def test_split_keeps_groups_whole():
    pop, _ = toy()
    a, b = split_by_group(pop, 0.3)
    assert not set(pop.group[a]) & set(pop.group[b])


def test_volunteered_fact_is_not_asked_again():
    pop, loss = toy()
    rng = np.random.default_rng(0)
    user = UserModel(opener=1, volunteerable=[0], name="opener")
    o = run_one(VOIPolicy(question_cost=1.0), pop, loss, pop.answers[0], user, rng)
    assert "kids" not in o.asked
