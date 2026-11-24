"""A self-contained engine bundle: one .npz file holds everything a live intake needs.

The bundle carries the questions (text, options, misreport models, amount ranges), the decisions, the
population the belief is kept over, the loss, and the outcome network's weights. The network's wide
features are stored as data ("terms"), so loading needs numpy and this package and nothing else: no
PolicyEngine, no pandas, no domain adapter.

A term is one indicator group: column = off + sum(answer[q] * stride). The adapter's `features`
function is exactly a list of such terms (single answers, answer pairs, an intercept).
"""

from __future__ import annotations

import json

import numpy as np

from .core import LossModel, Population, Question
from .neural import NeuralOutcome

FORMAT = "decisionintake-bundle/1"
FORMAT2 = "decisionintake-bundle/2"  # several outcome networks side by side (StackedOutcome)
WEIGHTS = ["W1", "b1", "W2", "b2", "W3", "Ww", "scale"]


class StackedOutcome:
    """Outcome networks side by side, each scoring its own block of programmes, in order. Used to add
    programmes to a frozen network without retraining it (DECISIONS D43)."""

    def __init__(self, parts: list[NeuralOutcome]):
        self.parts = list(parts)

    def predict(self, pop: Population, answers: np.ndarray):
        ps, vs = zip(*(m.predict(pop, answers) for m in self.parts))
        return np.concatenate(ps, axis=1), np.concatenate(vs, axis=1)

    @property
    def sizes(self) -> list[int]:
        return [m.W3.shape[1] // 2 for m in self.parts]


class TermFeatures:
    """The outcome model's wide indicators, computed from stored terms instead of adapter code."""

    def __init__(self, terms: list[dict], n: int):
        self.terms, self.n = terms, int(n)

    def __call__(self, pop: Population, answers: np.ndarray) -> np.ndarray:
        cols = []
        for t in self.terms:
            c = np.full(len(answers), int(t["off"]), dtype=np.int64)
            for q, s in zip(t["qs"], t["strides"]):
                c = c + answers[:, q].astype(np.int64) * int(s)
            cols.append(c)
        return np.stack(cols, axis=1)

    def n_features(self, pop: Population) -> int:
        return self.n


def save(path: str, pop: Population, loss: LossModel, model: NeuralOutcome, terms: list[dict],
         labels: dict[str, str], info: dict) -> None:
    """pop: the population the belief runs over (answers, survey weights, true outcomes for display).
    model: a fitted, calibrated NeuralOutcome. terms: its wide features as data (see TermFeatures).
    info: free-form provenance (data source, rules version, settings) shipped to every client."""
    meta = {
        "format": FORMAT,
        "questions": [
            {"id": q.id, "text": q.text, "options": list(q.options), "cost": q.cost, "ordered": q.ordered,
             "noise": None if q.noise is None else np.asarray(q.noise).tolist(),
             "edges": None if q.edges is None else list(q.edges), "per": q.per}
            for q in pop.questions
        ],
        "decisions": list(pop.decisions),
        "labels": {d: labels.get(d, d) for d in pop.decisions},
        "false_rec_cost": np.asarray(loss.false_rec_cost, float).tolist(),
        "miss_floor": np.asarray(loss.miss_floor, float).tolist(),
        "terms": terms,
        "info": info,
    }
    parts = model.parts if isinstance(model, StackedOutcome) else [model]
    assert sum(m.W3.shape[1] // 2 for m in parts) == len(pop.decisions), "the networks must cover every decision"
    if isinstance(model, StackedOutcome):
        meta["format"] = FORMAT2
        meta["networks"] = [{"decisions": m.W3.shape[1] // 2, "n_features": int(m.Ww.shape[0])} for m in parts]
        weights = {f"n{i}_{k}": getattr(m, k).astype(np.float32) for i, m in enumerate(parts) for k in WEIGHTS}
    else:
        meta["n_features"] = int(model.Ww.shape[0])
        weights = {k: getattr(model, k).astype(np.float32) for k in WEIGHTS}
    np.savez_compressed(
        path,
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        answers=pop.answers.astype(np.uint8),
        weight=pop.weight.astype(np.float64),
        eligible=pop.eligible.astype(bool),
        value=pop.value.astype(np.float32),
        **weights,
    )


class Bundle:
    def __init__(self, pop: Population, loss: LossModel, model: NeuralOutcome, labels: dict, info: dict):
        self.pop, self.loss, self.model, self.labels, self.info = pop, loss, model, labels, info


def load(path: str) -> Bundle:
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes().decode())
    if meta.get("format") not in (FORMAT, FORMAT2):
        raise ValueError(f"{path}: not a {FORMAT} or {FORMAT2} file")
    questions = [
        Question(q["id"], q["text"], tuple(q["options"]), q["cost"],
                 None if q["noise"] is None else np.array(q["noise"]), q["ordered"],
                 None if q["edges"] is None else tuple(q["edges"]), q["per"])
        for q in meta["questions"]
    ]
    pop = Population(questions, meta["decisions"], z["answers"].astype(np.int64), z["weight"],
                     z["eligible"], z["value"].astype(np.float64))
    loss = LossModel(np.array(meta["false_rec_cost"]), np.array(meta["miss_floor"]))
    if meta["format"] == FORMAT:
        feats = TermFeatures(meta["terms"], meta["n_features"])
        model = NeuralOutcome(feats, feats.n_features)
        for k in WEIGHTS:
            setattr(model, k, z[k])
    else:
        parts = []
        for i, net in enumerate(meta["networks"]):
            feats = TermFeatures(meta["terms"], net["n_features"])
            m = NeuralOutcome(feats, feats.n_features)
            for k in WEIGHTS:
                setattr(m, k, z[f"n{i}_{k}"])
            parts.append(m)
        model = StackedOutcome(parts)
    return Bundle(pop, loss, model, meta["labels"], meta["info"])
