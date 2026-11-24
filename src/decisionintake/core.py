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
    ordered: bool = False  # options are an ordered scale (amounts, counts)
    # For amounts: the options are ranges with these upper edges (the last range is open), stated per
    # "year", "month", or None for a stock such as savings. Lets a reader do the arithmetic in code.
    edges: tuple | None = None
    per: str | None = None

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
    # Optional smoothed outcomes (see smoothing.py): P(eligible | answers) and dollars if eligible.
    # When set, decisions and losses use these instead of each record's raw outcome.
    p_eligible: np.ndarray | None = None  # (N, D)
    value_if_eligible: np.ndarray | None = None  # (N, D)

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
            None if self.p_eligible is None else self.p_eligible[idx],
            None if self.value_if_eligible is None else self.value_if_eligible[idx],
        )

    def onehot(self) -> tuple[np.ndarray, np.ndarray]:
        """(N, sum of option counts) one-hot of every record's answers, and each question's
        column offset. Lets one matrix product aggregate any per-record quantity by (question, option)."""
        if getattr(self, "_onehot", None) is None:
            sizes = np.array([len(q.options) for q in self.questions])
            offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
            oh = np.zeros((len(self), int(sizes.sum())), dtype=np.float64)
            rows = np.arange(len(self))
            for j in range(len(self.questions)):
                oh[rows, offsets[j] + self.answers[:, j]] = 1.0
            self._onehot = (oh, offsets)
        return self._onehot

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
        if pop.p_eligible is not None:  # expected losses under the smoothed outcome
            p = pop.p_eligible
            miss = p * np.maximum(pop.value_if_eligible, self.miss_floor[None, :])
            false_rec = (1.0 - p) * self.false_rec_cost[None, :]
            return miss, false_rec
        miss = np.where(pop.eligible, np.maximum(pop.value, self.miss_floor[None, :]), 0.0)
        false_rec = np.where(pop.eligible, 0.0, self.false_rec_cost[None, :])
        return miss, false_rec  # loss if we skip, loss if we recommend

    def realised(self, pop: Population) -> tuple[np.ndarray, np.ndarray]:
        """Losses against each record's TRUE outcome (for scoring), ignoring any smoothing."""
        miss = np.where(pop.eligible, np.maximum(pop.value, self.miss_floor[None, :]), 0.0)
        false_rec = np.where(pop.eligible, 0.0, self.false_rec_cost[None, :])
        return miss, false_rec


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
    min_ess: float = 1.0  # support floor: soften the evidence until the belief rests on this many records
    # Optional outcome model (smoothing.OutcomeModel). When set, each record's outcome is re-scored
    # after every answer on a hybrid: the person's answers where known, the record's own elsewhere.
    outcome: object = None
    w: np.ndarray = field(init=False)
    asked: dict[int, np.ndarray] = field(init=False, default_factory=dict)
    log: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.l_skip, self.l_rec = self.loss.matrices(self.pop)
        self.prior = self.pop.weight.astype(float) / self.pop.weight.sum()
        self.loglik = np.zeros(len(self.pop))
        self.ll_terms: dict[int, np.ndarray] = {}  # each question's share of loglik, for what_if
        self.tau = 1.0
        self.w = self.prior.copy()
        self.p_now = self.v_now = None
        if self.outcome is not None:
            self._rescore()

    def _rescore(self) -> None:
        """Re-score records on the hybrid answers (see `outcome`). Before any answer every session
        scores the same, so that is cached on the model. Afterwards only records that still carry
        weight are re-scored; the rest keep stale scores, which can't affect any sum they're weighted in."""
        if not self.asked:
            cache = getattr(self.outcome, "_initial", None)
            if cache is None or cache[0] is not self.pop:
                cache = (self.pop, *self.outcome.predict(self.pop, self.pop.answers))
                self.outcome._initial = cache
            p, v = cache[1].copy(), cache[2].copy()
            self.h_now = self.pop.answers.astype(np.uint8)  # the inputs each record is scored on
        else:
            p, v = self.p_now, self.v_now
            live = np.flatnonzero(self.w > 1e-9 * self.w.max())
            h = self.pop.answers[live].copy()
            for j, ev in self.asked.items():
                if np.ptp(ev) == 0:
                    continue  # "don't know": the record keeps its own answer
                own_ok = ev[h[:, j]] >= 0.5 * ev.max()  # the record's own answer fits the evidence
                h[:, j] = np.where(own_ok, h[:, j], int(np.argmax(ev)))
            # Only records whose inputs changed need scoring again; the rest keep identical scores.
            changed = (h != self.h_now[live]).any(1)
            rows = live[changed]
            if len(rows):
                p[rows], v[rows] = self.outcome.predict(self.pop, h[changed])
                self.h_now[rows] = h[changed]
        self.p_now, self.v_now = p, v
        self.l_skip = p * np.maximum(v, self.loss.miss_floor[None, :])
        self.l_rec = (1.0 - p) * self.loss.false_rec_cost[None, :]

    def _reweight(self) -> None:
        """w = prior * likelihood^tau, with tau the largest value in (0, 1] whose ESS meets the floor.

        Records that the evidence rules out entirely (likelihood 0, e.g. another state) stay out.
        Without a floor tau is 1 and this is plain Bayes."""
        finite = np.isfinite(self.loglik)
        ll = np.where(finite, self.loglik - self.loglik[finite].max(), 0.0)

        def weights(tau):
            w = np.where(finite, self.prior * np.exp(tau * ll), 0.0)
            return w / w.sum()

        def ess(w):
            return 1.0 / np.square(w).sum()

        w = weights(1.0)
        tau = 1.0
        if self.min_ess > 1 and ess(w) < self.min_ess:
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                if ess(weights(mid)) >= self.min_ess:
                    lo = mid
                else:
                    hi = mid
            tau = lo
            w = weights(tau)
        self.w, self.tau = w, tau

    # evidence -------------------------------------------------------------------------------
    def observe(self, q: int, likelihood: np.ndarray, source: str = "answer", rescore: bool = True) -> None:
        """likelihood[b] = P(this evidence | person would report option b).

        A crisp answer b is a one-hot vector. A hedged reading of free text is a spread vector.
        The question's report-noise model maps it to a likelihood over each particle's TRUE option.
        """
        likelihood = np.asarray(likelihood, dtype=float)
        qu = self.pop.questions[q]
        assert likelihood.shape == (len(qu.options),)
        per_true = qu.report_matrix() @ likelihood  # (B,) P(evidence | true option a)
        lik = per_true[self.pop.answers[:, q]]
        mass = float((self.w * lik).sum())
        self.log.append(
            {"q": qu.id, "source": source, "likelihood": likelihood.round(3).tolist(), "evidence_mass": mass}
        )
        prev = self.asked.get(q)
        self.asked[q] = likelihood if prev is None else prev * likelihood
        if mass <= 0:
            # No record is compatible with this evidence: keep the old belief, mark the conflict.
            # The question still counts as asked, so it isn't asked again.
            self.log[-1]["conflict"] = True
            return
        with np.errstate(divide="ignore"):
            term = np.log(lik)
        self.loglik = self.loglik + term
        self.ll_terms[q] = self.ll_terms[q] + term if q in self.ll_terms else term
        self._reweight()
        if self.outcome is not None and rescore:  # rescore=False: the caller rescores once after a batch
            self._rescore()
        self.log[-1]["tau"] = self.tau

    def rescore(self) -> None:
        """Re-score after a batch of observe(..., rescore=False): only the final belief's records matter."""
        if self.outcome is not None and self.asked:
            self._rescore()

    def what_if(self, q: int, likelihood: np.ndarray) -> "Session":
        """A copy of this session in which question q's evidence is `likelihood` instead of what was said;
        this session is not changed. It swaps one question's log-likelihood term, re-weights, and re-scores
        only the records whose inputs change, so it costs a small fraction of replaying every answer."""
        import copy
        likelihood = np.asarray(likelihood, dtype=float)
        qu = self.pop.questions[q]
        assert likelihood.shape == (len(qu.options),)
        with np.errstate(divide="ignore"):
            term = np.log((qu.report_matrix() @ likelihood)[self.pop.answers[:, q]])
        rest = [t for j, t in self.ll_terms.items() if j != q]
        base = np.sum(rest, axis=0) if rest else np.zeros(len(self.pop))
        other = copy.copy(self)
        other.asked = {**self.asked, q: likelihood}
        other.ll_terms = {j: t for j, t in self.ll_terms.items() if j != q}
        other.log = list(self.log)
        if np.isfinite(base + term).any():  # as observe(): evidence no record fits is not applied
            other.ll_terms[q] = term
            other.loglik = base + term
        else:
            other.loglik = base
        if self.p_now is not None:  # _rescore writes into these; keep this session's untouched
            other.p_now, other.v_now, other.h_now = self.p_now.copy(), self.v_now.copy(), self.h_now.copy()
        other._reweight()
        if other.outcome is not None:
            other._rescore()
        return other

    def answer(self, q: int, option: int, source: str = "answer") -> None:
        v = np.zeros(len(self.pop.questions[q].options))
        v[option] = 1.0
        self.observe(q, v, source)

    # diagnostics ----------------------------------------------------------------------------
    def ess(self) -> float:
        return float(1.0 / np.square(self.w).sum())

    def risk(self) -> float:
        return bayes_risk(self.w, self.l_skip, self.l_rec)

    def support(self) -> dict:
        """How well the data covers this person: households behind the belief before the support floor
        softened the evidence (raw_ess), the softening applied (tau, 1 = none), and answers no household fits."""
        finite = np.isfinite(self.loglik)
        if finite.any():
            ll = np.where(finite, self.loglik - self.loglik[finite].max(), -np.inf)
            w = self.prior * np.exp(ll)
            raw = float(1.0 / np.square(w / w.sum()).sum())
        else:
            raw = 0.0
        return {"raw_ess": raw, "tau": float(self.tau), "ess": self.ess(),
                "conflicts": sum(bool(e.get("conflict")) for e in self.log)}

    def decisions(self) -> list[DecisionView]:
        out = []
        r_skip = self.w @ self.l_skip
        r_rec = self.w @ self.l_rec
        for d, name in enumerate(self.pop.decisions):
            if self.p_now is not None or self.pop.p_eligible is not None:
                pe = (self.p_now if self.p_now is not None else self.pop.p_eligible)[:, d]
                vv = (self.v_now if self.v_now is not None else self.pop.value_if_eligible)[:, d]
                p = float(self.w @ pe)
                lo, hi = weighted_quantile(vv, self.w * pe, [0.1, 0.9])
            else:
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
        """Value of information of every candidate question, sorted best first.

        One product aggregates the weighted losses by (question, true option); each question's
        report-noise matrix then maps true options to reported ones. Exact, same as voi(q)."""
        if candidates is None:
            candidates = [j for j in range(len(self.pop.questions)) if j not in self.asked]
        if not candidates:
            return []
        oh, offsets = self.pop.onehot()
        d = self.l_skip.shape[1]
        wl = np.concatenate([self.w[:, None] * self.l_skip, self.w[:, None] * self.l_rec], axis=1)
        agg = oh.T @ wl
        now = self.risk()
        total = self.w.sum()
        rows = []
        for j in candidates:
            qu = self.pop.questions[j]
            b = len(qu.options)
            block = agg[offsets[j] : offsets[j] + b]  # (B_true, 2D)
            rep = qu.report_matrix()  # (B_true, B_reported)
            post = rep.T @ block  # (B_reported, 2D)
            exp_post = np.minimum(post[:, :d], post[:, d:]).sum() / total
            rows.append((j, now - exp_post))
        rows.sort(key=lambda r: -r[1])
        return rows

    def voi_by_decision(self, q: int) -> np.ndarray:
        """voi(q) split by decision: how much of the expected drop in loss falls on each one. Sums to voi(q).
        Explains a question to the person ("this mostly settles SNAP and school meals")."""
        qu = self.pop.questions[q]
        b, total = len(qu.options), self.w.sum()
        col = self.pop.answers[:, q]
        block_s = np.stack([np.bincount(col, weights=self.w * self.l_skip[:, d], minlength=b)
                            for d in range(self.l_skip.shape[1])], 1)
        block_r = np.stack([np.bincount(col, weights=self.w * self.l_rec[:, d], minlength=b)
                            for d in range(self.l_rec.shape[1])], 1)
        rep = qu.report_matrix()
        post_s, post_r = rep.T @ block_s, rep.T @ block_r  # (reported option, decision)
        now = np.minimum(self.w @ self.l_skip, self.w @ self.l_rec) / total
        return now - np.minimum(post_s, post_r).sum(0) / total

    def pair_table(self, candidates=None, n_plan: int = 4000, seed: int = 0) -> list[tuple[int, int, float]]:
        """Value of asking two questions, for every pair of candidates, best first.

        For planning, the belief is resampled to n_plan equally weighted records (systematic
        resampling), losses are summed by the pair's joint true options, and both questions'
        report-noise models map them to joint reported options."""
        if candidates is None:
            candidates = [j for j in range(len(self.pop.questions)) if j not in self.asked]
        if len(candidates) < 2:
            return []
        rng = np.random.default_rng(seed)
        c = np.cumsum(self.w)
        c[-1] = 1.0
        idx = np.searchsorted(c, (rng.random() + np.arange(n_plan)) / n_plan)
        ls, lr = self.l_skip[idx], self.l_rec[idx]
        d = ls.shape[1]
        losses = np.concatenate([ls, lr], axis=1)  # (M, 2D)
        now = float(np.minimum(ls.sum(0), lr.sum(0)).sum() / n_plan)
        ans = self.pop.answers[idx]
        rows = []
        for x, q1 in enumerate(candidates):
            b1 = len(self.pop.questions[q1].options)
            m1 = self.pop.questions[q1].report_matrix()
            for q2 in candidates[x + 1 :]:
                b2 = len(self.pop.questions[q2].options)
                m2 = self.pop.questions[q2].report_matrix()
                code = ans[:, q1] * b2 + ans[:, q2]
                agg = np.stack([np.bincount(code, weights=losses[:, k], minlength=b1 * b2) for k in range(2 * d)], 1)
                agg = agg.reshape(b1, b2, 2 * d)
                post = np.einsum("ab,cd,ack->bdk", m1, m2, agg)  # joint reported options
                exp_post = np.minimum(post[..., :d], post[..., d:]).sum() / n_plan
                rows.append((q1, q2, now - exp_post))
        rows.sort(key=lambda r: -r[2])
        return rows
