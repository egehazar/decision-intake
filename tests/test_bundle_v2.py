"""Bundle format 2: outcome networks side by side (D43). A stacked bundle must reload to the same predictions,
each network scoring its own block of programmes, and format 1 files must still load."""

from pathlib import Path

import numpy as np
import pytest

from decisionintake import bundle
from decisionintake.core import LossModel, Population

SHIPPED = Path(__file__).resolve().parents[1] / "bundles" / "benefits_v2.npz"
pytestmark = pytest.mark.skipif(not SHIPPED.exists(), reason="shipped bundle not present")


def test_stacked_bundle_round_trip(tmp_path):
    v2 = bundle.load(str(SHIPPED))
    assert isinstance(v2.model, bundle.NeuralOutcome)  # format 1 still loads as one network
    p = v2.pop
    rows = np.arange(0, len(p), 97)[:400]
    sub = p.subset(rows)
    twice = Population(sub.questions, list(sub.decisions) + [d + "_b" for d in sub.decisions], sub.answers,
                       sub.weight, np.hstack([sub.eligible, sub.eligible]), np.hstack([sub.value, sub.value]))
    stacked = bundle.StackedOutcome([v2.model, v2.model])
    assert stacked.sizes == [10, 10]
    loss = LossModel(np.full(20, 1000.0), np.full(20, 120.0))
    out = tmp_path / "stacked.npz"
    bundle.save(str(out), twice, loss, stacked, v2.model.features.terms, {}, {"name": "test"})
    b = bundle.load(str(out))
    assert isinstance(b.model, bundle.StackedOutcome) and b.model.sizes == [10, 10]
    p1, v1 = v2.model.predict(p, sub.answers)
    p2, v2b = b.model.predict(b.pop, b.pop.answers)
    assert p2.shape == (len(rows), 20)
    assert np.array_equal(p2[:, :10], p1) and np.array_equal(p2[:, 10:], p1) and np.array_equal(v2b[:, :10], v1)


def test_networks_must_cover_every_programme(tmp_path):
    v2 = bundle.load(str(SHIPPED))
    sub = v2.pop.subset(np.arange(50))
    wider = Population(sub.questions, list(sub.decisions) + ["extra"], sub.answers, sub.weight,
                       np.hstack([sub.eligible, sub.eligible[:, :1]]), np.hstack([sub.value, sub.value[:, :1]]))
    with pytest.raises(AssertionError):
        bundle.save(str(tmp_path / "x.npz"), wider, LossModel(np.full(11, 1000.0), np.full(11, 120.0)),
                    v2.model, v2.model.features.terms, {}, {})
