"""Shared setup: splits, and smoothed training populations cached on disk."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))

from benefits.population import build, features, load_table  # noqa: E402

from decisionintake.simulate import split_by_group  # noqa: E402
from decisionintake.smoothing import smooth  # noqa: E402


def splits(poverty_question=False):
    """train / validation / test, every source household wholly on one side. Test is 15%, validation
    12% of the rest. Returns the full population and the three index arrays into it."""
    pop, loss, meta = build(load_table(), poverty_question=poverty_question)
    rest, test = split_by_group(pop, 0.15, seed=0)
    tr, va = split_by_group(pop.subset(rest), 0.12, seed=1)
    return pop, loss, meta, rest[tr], rest[va], test


def smoothed(pop, idx, key):
    """pop.subset(idx) with out-of-fold smoothed outcomes, cached in data/cache/smooth_<key>.npz."""
    path = f"data/cache/smooth_{key}.npz"
    sub = pop.subset(idx)
    if os.path.exists(path):
        z = np.load(path)
        if len(z["p"]) == len(sub):
            sub.p_eligible, sub.value_if_eligible = z["p"], z["v"]
            return sub
    sm, _ = smooth(sub, features)
    np.savez_compressed(path, p=sm.p_eligible, v=sm.value_if_eligible)
    return sm
