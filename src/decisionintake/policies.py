"""Question-asking policies. Each returns the next question index, or None to stop.

VOIPolicy is the engine. The others are the baselines it has to beat:
  FixedOrder   - the best fixed form: one order for everybody, chosen greedily on the training population
  EntropyGreedy - adaptive, but asks what most reduces uncertainty about eligibility, ignoring what's at stake
  GreedyTree   - an adaptive decision tree trained on the same decision loss, with hard answers and a
                 minimum leaf size; it can skip a question already answered, but it can't re-plan
AskAll is the full questionnaire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import Population, Session, bayes_risk


class Policy:
    name = "policy"

    def start(self, s: Session) -> None:
        pass

    def next_question(self, s: Session, n_asked: int) -> int | None:
        raise NotImplementedError

    def final_recommend(self, s: Session) -> np.ndarray:
        return s.recommend_mask()


def _unasked(s: Session) -> list[int]:
    return [j for j in range(len(s.pop.questions)) if j not in s.asked]


@dataclass
class VOIPolicy(Policy):
    """Ask the question with the largest expected drop in decision loss, net of its cost.
    question_cost is in the loss's units (dollars a year): ask only while a question is
    expected to be worth more than that."""

    question_cost: float = 50.0
    budget: int = 99
    lookahead: bool = True  # when no single question pays, price pairs before stopping
    n_plan: int = 4000
    name: str = "engine"

    def next_question(self, s, n_asked):
        if n_asked >= self.budget:
            return None
        table = s.voi_table(_unasked(s))
        best, best_net = None, 0.0
        for j, v in table:
            net = v - self.question_cost * s.pop.questions[j].cost
            if net > best_net:
                best, best_net = j, net
        if best is not None or not self.lookahead or n_asked + 2 > self.budget:
            return best
        single = dict(table)
        for q1, q2, v in s.pair_table(_unasked(s), n_plan=self.n_plan, seed=n_asked):
            cost = self.question_cost * (s.pop.questions[q1].cost + s.pop.questions[q2].cost)
            if v - cost > 0:
                return q1 if single.get(q1, 0) >= single.get(q2, 0) else q2
            break  # sorted: if the best pair doesn't pay, none will
        return None


@dataclass
class AskAll(Policy):
    name: str = "ask_all"

    def next_question(self, s, n_asked):
        rest = _unasked(s)
        return rest[0] if rest else None


def _binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@dataclass
class EntropyGreedy(Policy):
    """Information gain about eligibility (summed over decisions), blind to the stakes."""

    budget: int = 8
    min_gain: float = 1e-3
    name: str = "entropy"

    def next_question(self, s, n_asked):
        if n_asked >= self.budget:
            return None
        el = s.pop.eligible.astype(float)
        now = _binary_entropy(s.w @ el).sum()
        oh, offsets = s.pop.onehot()
        agg_w = oh.T @ s.w
        agg_e = oh.T @ (s.w[:, None] * el)
        best, best_gain = None, self.min_gain
        for j in _unasked(s):
            b = len(s.pop.questions[j].options)
            rep = s.pop.questions[j].report_matrix()
            mass = rep.T @ agg_w[offsets[j] : offsets[j] + b]  # (B_reported,)
            pe = rep.T @ agg_e[offsets[j] : offsets[j] + b]  # (B_reported, D)
            keep = mass > 1e-15
            p = pe[keep] / mass[keep, None]
            post = (mass[keep] * _binary_entropy(p).sum(1)).sum() / mass.sum()
            gain = now - post
            if gain > best_gain:
                best, best_gain = j, gain
        return best


def _grouped_risk(pop: Population, w, l_skip, l_rec, cols: list[int]) -> float:
    """Bayes risk when every record's answers to `cols` are known exactly (hard partition)."""
    if not cols:
        return bayes_risk(w, l_skip, l_rec)
    keys = np.zeros(len(pop), dtype=np.int64)
    for j in cols:
        keys = keys * len(pop.questions[j].options) + pop.answers[:, j]
    _, inv = np.unique(keys, return_inverse=True)
    g = inv.max() + 1
    r_skip = np.zeros((g, l_skip.shape[1]))
    r_rec = np.zeros((g, l_skip.shape[1]))
    np.add.at(r_skip, inv, w[:, None] * l_skip)
    np.add.at(r_rec, inv, w[:, None] * l_rec)
    return float(np.minimum(r_skip, r_rec).sum() / w.sum())


@dataclass
class FixedOrder(Policy):
    """One question order for everyone. order=None builds the best greedy order on `train`."""

    order: list[int] | None = None
    budget: int = 8
    name: str = "fixed_form"

    @classmethod
    def fit(cls, train: Population, loss, budget: int = 8, max_len: int | None = None):
        l_skip, l_rec = loss.matrices(train)
        w = train.weight / train.weight.sum()
        chosen: list[int] = []
        rest = list(range(len(train.questions)))
        for _ in range(max_len or len(rest)):
            scores = [(_grouped_risk(train, w, l_skip, l_rec, chosen + [j]), j) for j in rest]
            r, j = min(scores)
            chosen.append(j)
            rest.remove(j)
        return cls(order=chosen, budget=budget)

    def next_question(self, s, n_asked):
        if n_asked >= self.budget:
            return None
        for j in self.order:
            if j not in s.asked:
                return j
        return None


@dataclass
class GreedyTree(Policy):
    """Decision tree grown greedily on the training population with the same decision loss.

    Nodes are built lazily (a node is fully determined by its path), so this is exactly the
    tree you would precompute. Answers are taken as hard: the reported option with the most
    likelihood. A volunteered answer to the node's question is used without asking.
    Decisions come from the leaf's training records, not from the session's belief.
    """

    train: Population = None
    loss: object = None
    budget: int = 8
    min_leaf: int = 30
    question_cost: float = 50.0
    name: str = "tree"
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.l_skip, self.l_rec = self.loss.matrices(self.train)
        self.w = self.train.weight / self.train.weight.sum()

    def _node(self, path: tuple):
        if path in self._cache:
            return self._cache[path]
        mask = np.ones(len(self.train), dtype=bool)
        for j, b in path:
            mask &= self.train.answers[:, j] == b
        idx = np.flatnonzero(mask)
        w = self.w[idx]
        ls, lr = self.l_skip[idx], self.l_rec[idx]
        rec = (w @ lr) < (w @ ls) if len(idx) else np.zeros(self.l_skip.shape[1], bool)
        split = None
        if len(idx) >= 2 * self.min_leaf and len(path) < self.budget:
            used = {j for j, _ in path}
            now = bayes_risk(w, ls, lr)
            best_net = 0.0
            oh, offsets = self.train.onehot()
            sub = oh[idx]
            counts = sub.sum(0)
            d = ls.shape[1]
            agg = sub.T @ np.concatenate([w[:, None] * ls, w[:, None] * lr], axis=1)
            for j in range(len(self.train.questions)):
                if j in used:
                    continue
                b = len(self.train.questions[j].options)
                if counts[offsets[j] : offsets[j] + b].max() > len(idx) - self.min_leaf:
                    continue  # one branch would hold almost everything: no real split
                block = agg[offsets[j] : offsets[j] + b]
                post = np.minimum(block[:, :d], block[:, d:]).sum() / w.sum()
                net = (now - post) - self.question_cost * self.train.questions[j].cost
                if net > best_net:
                    split, best_net = j, net
        node = {"n": len(idx), "rec": rec, "split": split}
        self._cache[path] = node
        return node

    def start(self, s):
        s.policy_state = {"path": ()}

    def _branch(self, s, j, path):
        """The tree's branch for question j: the most likely reported option, or, when the
        evidence doesn't single one out (a "don't know"), the branch most training records take."""
        ev = s.asked[j]
        if np.ptp(ev) > 0:
            return int(np.argmax(ev))
        mask = np.ones(len(self.train), dtype=bool)
        for jj, b in path:
            mask &= self.train.answers[:, jj] == b
        col = self.train.answers[mask, j]
        return int(np.bincount(col, weights=self.w[mask]).argmax())

    def next_question(self, s, n_asked):
        st = s.policy_state
        if st.get("dead_end"):
            return None
        while True:
            node = self._node(st["path"])
            j = node["split"]
            if j is None or n_asked >= self.budget:
                st["leaf"] = node
                return None
            if j in s.asked:  # volunteered or already answered: follow it for free
                b = self._branch(s, j, st["path"])
                child = self._node(st["path"] + ((j, b),))
                if child["n"] == 0:
                    st["leaf"] = node
                    return None
                st["path"] = st["path"] + ((j, b),)
                continue
            st["pending"] = j
            return j

    def after_answer(self, s, j):
        st = s.policy_state
        if st.get("pending") != j:
            return
        b = self._branch(s, j, st["path"])
        child = self._node(st["path"] + ((j, b),))
        if child["n"] == 0:
            st["dead_end"] = True
            return
        st["path"] = st["path"] + ((j, b),)

    def final_recommend(self, s):
        st = s.policy_state
        return self._node(st["path"])["rec"]
