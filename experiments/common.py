"""Shared setup: splits, and smoothed training populations cached on disk."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))

from benefits.population import build, features, load_table, n_features  # noqa: E402

from decisionintake.simulate import split_by_group  # noqa: E402
from decisionintake.smoothing import OutcomeModel, smooth  # noqa: E402


def splits(poverty_question=False):
    """train / validation / test, every source household wholly on one side. Test is 15%, validation
    12% of the rest. Returns the full population and the three index arrays into it."""
    pop, loss, meta = build(load_table(), poverty_question=poverty_question)
    rest, test = split_by_group(pop, 0.15, seed=0)
    tr, va = split_by_group(pop.subset(rest), 0.12, seed=1)
    return pop, loss, meta, rest[tr], rest[va], test


def smoothed(pop, idx, key):
    """(pop.subset(idx) with out-of-fold calibrated outcomes, the calibrated full model), cached in
    data/cache/smooth2_<key>.npz / .pkl."""
    import pickle

    path = f"data/cache/smooth2_{key}"
    sub = pop.subset(idx)
    if os.path.exists(path + ".npz") and os.path.exists(path + ".pkl"):
        z = np.load(path + ".npz")
        if len(z["p"]) == len(sub):
            sub.p_eligible, sub.value_if_eligible = z["p"], z["v"]
            with open(path + ".pkl", "rb") as f:
                B, R = pickle.load(f)
            model = OutcomeModel(features, n_features)
            model.B, model.R = B, R
            return sub, model
    sm, model = smooth(sub, features, n_features)
    np.savez_compressed(path + ".npz", p=sm.p_eligible, v=sm.value_if_eligible)
    with open(path + ".pkl", "wb") as f:
        pickle.dump((model.B, model.R), f)
    return sm, model


def smoothed_neural(pop, idx, key, **cfg):
    """As smoothed(), with the wide-and-deep outcome model (decisionintake.neural); cached in
    data/cache/neural_<key>.npz. cfg are NeuralOutcome settings and are part of the cache key."""
    from decisionintake.neural import NeuralOutcome, smooth_neural

    tag = "_".join(f"{k}{v}" for k, v in sorted(cfg.items())).replace(" ", "").replace(",", "x")
    path = f"data/cache/neural_{key}_{tag}.npz"
    sub = pop.subset(idx)
    names = ["W1", "b1", "W2", "b2", "W3", "Ww", "scale"]
    if os.path.exists(path):
        z = np.load(path)
        if len(z["p"]) == len(sub):
            sub.p_eligible, sub.value_if_eligible = z["p"], z["v"]
            model = NeuralOutcome(features, n_features, **cfg)
            for k in names:
                setattr(model, k, z[k])
            return sub, model
    sm, model = smooth_neural(sub, features, n_features, **cfg)
    np.savez_compressed(path, p=sm.p_eligible, v=sm.value_if_eligible, **{k: getattr(model, k) for k in names})
    return sm, model
