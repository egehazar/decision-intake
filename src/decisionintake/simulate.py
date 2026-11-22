"""Simulated intakes on held-out households, and the metrics GPT and Fable asked for.

A held-out household answers from its own record. Its truth (eligibility and dollars per
decision) comes from the rules engine run on its full record, never from the particles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import LossModel, Population, Session
from .policies import Policy


@dataclass
class UserModel:
    """How the simulated person talks.

    opener: how many facts they volunteer before any question (drawn from `volunteerable`).
    misreport: answers are drawn from each question's noise model instead of being exact.
    dont_know: chance of "I don't know" to an asked question (no evidence, still a question asked).
    """

    opener: int = 0
    volunteerable: list[int] = field(default_factory=list)
    misreport: bool = False
    dont_know: float = 0.0
    name: str = "clean"


def reported_option(rng, question, true_option: int, misreport: bool) -> int:
    if not misreport or question.noise is None:
        return true_option
    p = question.noise[true_option]
    return int(rng.choice(len(p), p=p / p.sum()))


@dataclass
class Outcome:
    n_questions: int
    recommend: np.ndarray  # (D,) bool
    risk_at_stop: float
    ess_at_stop: float
    asked: list[str]


def run_one(policy: Policy, train: Population, loss: LossModel, answers_row: np.ndarray, user: UserModel, rng,
            min_ess: float = 1.0, outcome=None) -> Outcome:
    s = Session(train, loss, min_ess=min_ess, outcome=outcome)
    policy.start(s)
    if user.opener and user.volunteerable:
        k = min(user.opener, len(user.volunteerable))
        for j in rng.choice(user.volunteerable, size=k, replace=False):
            j = int(j)
            s.answer(j, reported_option(rng, train.questions[j], int(answers_row[j]), user.misreport), source="opener")
    n = 0
    asked = []
    while True:
        j = policy.next_question(s, n)
        if j is None:
            break
        n += 1
        asked.append(train.questions[j].id)
        if user.dont_know and rng.random() < user.dont_know:
            s.observe(j, np.ones(len(train.questions[j].options)), source="dont_know")
        else:
            s.answer(j, reported_option(rng, train.questions[j], int(answers_row[j]), user.misreport))
        if hasattr(policy, "after_answer"):
            policy.after_answer(s, j)
    return Outcome(n, np.asarray(policy.final_recommend(s), bool), s.risk(), s.ess(), asked)


@dataclass
class EvalResult:
    policy: str
    user: str
    n: int
    mean_questions: float
    value_recall: float  # dollars of eligible programmes recommended / dollars of eligible programmes
    programme_recall: float  # eligible programmes recommended / eligible programmes
    false_rec_rate: float  # recommendations the household can't get / recommendations
    mean_loss: float  # realised loss per household under the loss model, dollars a year
    households_missing_any: float  # share of eligible households with at least one eligible programme missed
    per_decision_recall: dict
    per_decision_false_rec: dict

    def row(self) -> str:
        return (
            f"{self.policy:<12} {self.user:<10} q={self.mean_questions:5.2f}  $recall={self.value_recall:6.3f}  "
            f"prog_recall={self.programme_recall:6.3f}  false_rec={self.false_rec_rate:6.3f}  loss=${self.mean_loss:8.1f}  "
            f"miss_any={self.households_missing_any:5.3f}"
        )


def evaluate(policy_factory, train: Population, test: Population, loss: LossModel, user: UserModel, seed: int = 0, limit: int | None = None) -> EvalResult:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(test)) if limit is None else rng.choice(len(test), size=min(limit, len(test)), replace=False)
    D = len(test.decisions)
    recs = np.zeros((len(idx), D), bool)
    nq = np.zeros(len(idx))
    name = None
    for k, i in enumerate(idx):
        pol = policy_factory()
        name = pol.name
        o = run_one(pol, train, loss, test.answers[i], user, rng)
        recs[k] = o.recommend
        nq[k] = o.n_questions
    w = test.weight[idx] / test.weight[idx].sum()
    el = test.eligible[idx]
    val = test.value[idx]
    miss, fr = loss.realised(test.subset(idx))
    loss_per = (np.where(recs, 0.0, miss) + np.where(recs, fr, 0.0)).sum(1)
    val_el = np.where(el, val, 0.0)
    value_recall = float((w[:, None] * val_el * recs).sum() / max((w[:, None] * val_el).sum(), 1e-12))
    prog_recall = float((w[:, None] * (el & recs)).sum() / max((w[:, None] * el).sum(), 1e-12))
    false_rec = float((w[:, None] * (~el & recs)).sum() / max((w[:, None] * recs).sum(), 1e-12))
    any_el = el.any(1)
    missing_any = float((w * (any_el & (el & ~recs).any(1))).sum() / max((w * any_el).sum(), 1e-12))
    per_recall = {}
    per_fr = {}
    for d, nm in enumerate(test.decisions):
        e = el[:, d]
        per_recall[nm] = float((w * (e & recs[:, d])).sum() / max((w * e).sum(), 1e-12)) if e.any() else float("nan")
        r = recs[:, d]
        per_fr[nm] = float((w * (~e & r)).sum() / max((w * r).sum(), 1e-12)) if r.any() else float("nan")
    return EvalResult(name, user.name, len(idx), float((w * nq).sum()), value_recall, prog_recall, false_rec,
                      float((w * loss_per).sum()), missing_any, per_recall, per_fr)


def split_by_group(pop: Population, test_share: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices with every lineage group wholly on one side. Apply twice for train/val/test."""
    rng = np.random.default_rng(seed)
    groups = pop.group if pop.group is not None else np.arange(len(pop))
    uniq = np.unique(groups)
    test_groups = set(rng.choice(uniq, size=int(round(len(uniq) * test_share)), replace=False).tolist())
    is_test = np.array([g in test_groups for g in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)
