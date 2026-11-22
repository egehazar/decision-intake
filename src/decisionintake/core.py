"""Belief over a weighted population, decision loss, and value of information.

The engine knows nothing about any domain. A domain adapter supplies a Population:
records with known answers to every question and known outcomes for every decision.
A session keeps weights over those records (particles), updates them with evidence,
and picks the next question by how much it is expected to reduce decision loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class Question:
    id: str
    text: str
    options: tuple[str, ...]
    cost: float = 1.0
    # noise[a, b] = P(person reports option b | true option a). None means exact answers.
    noise: np.ndarray | None = None

    def report_matrix(self) -> np.ndarray:
        if self.noise is None:
            return np.eye(len(self.options))
        return self.noise


@dataclass(eq=False)
class Population:
    """Particles: one row per record. answers[i, q] is record i's true option for question q."""

    questions: list[Question]
    decisions: list[str]
    answers: np.ndarray  # (N, Q) int
    weight: np.ndarray  # (N,) survey weight
    eligible: np.ndarray  # (N, D) bool
    value: np.ndarray  # (N, D) float, dollars per year when eligible
    group: np.ndarray | None = None  # (N,) lineage id, records sharing one must not straddle a split

    def __post_init__(self):
        n, q = self.answers.shape
        assert q == len(self.questions)
        assert self.eligible.shape == self.value.shape == (n, len(self.decisions))
        for j, qu in enumerate(self.questions):
            col = self.answers[:, j]
            assert col.min() >= 0 and col.max() < len(qu.options), f"answers out of range for {qu.id}"

    def __len__(self):
        return self.answers.shape[0]

    def subset(self, idx: np.ndarray) -> "Population":
        return Population(
            self.questions,
            self.decisions,
            self.answers[idx],
            self.weight[idx],
            self.eligible[idx],
            self.value[idx],
            None if self.group is None else self.group[idx],
        )

    def qindex(self, qid: str) -> int:
        for j, q in enumerate(self.questions):
            if q.id == qid:
                return j
        raise KeyError(qid)


@dataclass
class LossModel:
    """Per-decision costs. Recommending a programme the household can't get costs
    false_rec_cost[d] (time spent applying, a refusal). Not recommending one it can get
    costs the benefit's value, floored at miss_floor[d] so small but real eligibility counts."""

    false_rec_cost: np.ndarray  # (D,)
    miss_floor: np.ndarray  # (D,)

    def matrices(self, pop: Population) -> tuple[np.ndarray, np.ndarray]:
        miss = np.where(pop.eligible, np.maximum(pop.value, self.miss_floor[None, :]), 0.0)
        false_rec = np.where(pop.eligible, 0.0, self.false_rec_cost[None, :])
        return miss, false_rec  # loss if we skip, loss if we recommend


def bayes_risk(w: np.ndarray, l_skip: np.ndarray, l_rec: np.ndarray) -> float:
    """Expected loss of the best action per decision under weights w (normalised inside)."""
    return float(np.minimum(w @ l_skip, w @ l_rec).sum() / w.sum())


def expected_posterior_risk(
    w: np.ndarray, answers_q: np.ndarray, report: np.ndarray, l_skip: np.ndarray, l_rec: np.ndarray
) -> float:
    """E over the reported option of the Bayes risk after hearing it.

    report[a, b] = P(report b | true a). For particle i, P(report b) = report[answers_q[i], b].
    Sum_b min_d-actions of the (unnormalised) posterior-weighted losses, over the prior mass.
    """
    lik = report[answers_q]  # (N, B)
    wb = lik * w[:, None]  # (N, B) joint mass of particle and report
    r_skip = wb.T @ l_skip  # (B, D)
    r_rec = wb.T @ l_rec
    return float(np.minimum(r_skip, r_rec).sum() / w.sum())


@dataclass
class DecisionView:
    decision: str
    p_eligible: float
    expected_value: float  # dollars per year, over the posterior including ineligible mass
    value_low: float  # 10th percentile of value among eligible posterior mass
    value_high: float  # 90th percentile
    recommend: bool
    risk_skip: float
    risk_rec: float


def weighted_quantile(x: np.ndarray, w: np.ndarray, qs) -> list[float]:
    if w.sum() <= 0 or len(x) == 0:
        return [float("nan")] * len(qs)
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w) / w.sum()
    return [float(x[min(np.searchsorted(c, q), len(x) - 1)]) for q in qs]


@dataclass
class Session:
    """One person's intake. Evidence is virtual evidence: a likelihood over a question's options."""

    pop: Population
    loss: LossModel
    w: np.ndarray = field(init=False)
    asked: dict[int, np.ndarray] = field(init=False, default_factory=dict)
    log: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.l_skip, self.l_rec = self.loss.matrices(self.pop)
        self.w = self.pop.weight.astype(float).copy()
        self.w /= self.w.sum()

    # evidence -------------------------------------------------------------------------------
    def observe(self, q: int, likelihood: np.ndarray, source: str = "answer") -> None:
        """likelihood[b] = P(this evidence | person would report option b).

        A crisp answer b is a one-hot vector. A hedged reading of free text is a spread vector.
        The question's report-noise model maps it to a likelihood over each particle's TRUE option.
        """
        likelihood = np.asarray(likelihood, dtype=float)
        qu = self.pop.questions[q]
        assert likelihood.shape == (len(qu.options),)
        per_true = qu.report_matrix() @ likelihood  # (B,) P(evidence | true option a)
        lik = per_true[self.pop.answers[:, q]]
        new = self.w * lik
        mass = new.sum()
        self.log.append(
            {"q": qu.id, "source": source, "likelihood": likelihood.round(3).tolist(), "evidence_mass": float(mass)}
        )
        if mass <= 0:
            # No record is compatible with this evidence: keep the old belief, mark the conflict.
            self.log[-1]["conflict"] = True
            return
        self.w = new / mass
        prev = self.asked.get(q)
        self.asked[q] = likelihood if prev is None else prev * likelihood

    def answer(self, q: int, option: int, source: str = "answer") -> None:
        v = np.zeros(len(self.pop.questions[q].options))
        v[option] = 1.0
        self.observe(q, v, source)

    # diagnostics ----------------------------------------------------------------------------
    def ess(self) -> float:
        return float(1.0 / np.square(self.w).sum())

    def risk(self) -> float:
        return bayes_risk(self.w, self.l_skip, self.l_rec)

    def decisions(self) -> list[DecisionView]:
        out = []
        r_skip = self.w @ self.l_skip
        r_rec = self.w @ self.l_rec
        for d, name in enumerate(self.pop.decisions):
            el = self.pop.eligible[:, d]
            p = float(self.w[el].sum())
            lo, hi = weighted_quantile(self.pop.value[el, d], self.w[el], [0.1, 0.9])
            out.append(
                DecisionView(
                    name,
                    p,
                    float(self.w @ self.pop.value[:, d]),
                    lo,
                    hi,
                    bool(r_rec[d] < r_skip[d]),
                    float(r_skip[d]),
                    float(r_rec[d]),
                )
            )
        return out

    def recommend_mask(self) -> np.ndarray:
        return (self.w @ self.l_rec) < (self.w @ self.l_skip)

    # value of information ---------------------------------------------------------------------
    def voi(self, q: int) -> float:
        qu = self.pop.questions[q]
        return self.risk() - expected_posterior_risk(
            self.w, self.pop.answers[:, q], qu.report_matrix(), self.l_skip, self.l_rec
        )

    def voi_table(self, candidates=None) -> list[tuple[int, float]]:
        if candidates is None:
            candidates = [j for j in range(len(self.pop.questions)) if j not in self.asked]
        now = self.risk()
        rows = []
        for j in candidates:
            qu = self.pop.questions[j]
            post = expected_posterior_risk(self.w, self.pop.answers[:, j], qu.report_matrix(), self.l_skip, self.l_rec)
            rows.append((j, now - post))
        rows.sort(key=lambda r: -r[1])
        return rows
