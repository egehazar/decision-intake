"""Tune on the VALIDATION split only (the test split stays untouched until the final run).

1. Does a support floor (min_ess) fix the collapse where asking everything beats asking less?
2. Does pair look-ahead stop the engine from quitting too early?

usage: python experiments/tune.py [n_val] [workers]
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))

from benefits.population import build, load_table  # noqa: E402

from decisionintake import AskAll, FixedOrder, GreedyTree, LossModel, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one, split_by_group  # noqa: E402

G = {}
SMOOTH = os.environ.get("SMOOTH", "1") == "1"
sys.path.insert(0, os.path.dirname(__file__))


def init():
    from common import smoothed, splits

    pop, loss, meta, tr, va, _test = splits()  # the test split is never touched here
    G.update(pop=pop, loss=loss, meta=meta, val=pop.subset(va))
    if SMOOTH:
        G["train"], G["outcome"] = smoothed(pop, tr, "train_tune")
    else:
        G["train"], G["outcome"] = pop.subset(tr), None
    G["fixed_order"] = FixedOrder.fit(G["train"], loss).order


def loss_for(fr):
    base = G["loss"]
    return LossModel(false_rec_cost=np.full_like(base.false_rec_cost, float(fr)), miss_floor=base.miss_floor)


def policy(kind, knob, look, fr):
    if kind == "engine":
        return VOIPolicy(question_cost=knob, lookahead=look)
    if kind == "fixed":
        return FixedOrder(order=G["fixed_order"], budget=int(knob))
    if kind == "tree":
        lam, leaf = knob
        key = (fr, lam, leaf)
        trees = G.setdefault("trees", {})
        if key not in trees:
            trees[key] = GreedyTree(train=G["train"], loss=loss_for(fr), min_leaf=leaf, question_cost=lam, lookahead=look)
        return trees[key]
    return AskAll()


def work(job):
    kind, knob, look, min_ess, uname, fr, rows = job
    v = G["meta"]["volunteerable"]
    user = {"clean": UserModel(name="clean"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.1, name="realistic")}[uname]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 11 + int(i))
        o = run_one(policy(kind, knob, look, fr), G["train"], loss_for(fr), G["val"].answers[i], user, rng,
                    min_ess=min_ess, outcome=G["outcome"])
        out.append((int(i), o.n_questions, o.recommend, o.risk_at_stop))
    return job[:6], out


def score(val, loss, out):
    idx = np.array([r[0] for r in out])
    nq = np.array([r[1] for r in out], float)
    rec = np.array([r[2] for r in out], bool)
    claimed = np.array([r[3] for r in out], float)
    w = val.weight[idx] / val.weight[idx].sum()
    miss, fr = loss.realised(val.subset(idx))
    realised = (np.where(rec, 0.0, miss) + np.where(rec, fr, 0.0)).sum(1)
    el = val.eligible[idx]
    return float(w @ nq), float(w @ realised), float(w @ claimed), \
        float((w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum()), \
        float((w[:, None] * (~el & rec)).sum() / max((w[:, None] * rec).sum(), 1e-12))


def main():
    n_val = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    init()
    rng = np.random.default_rng(5)
    sample = rng.choice(len(G["val"]), size=min(n_val, len(G["val"])), replace=False)
    print(f"train {len(G['train'])}, validation {len(G['val'])} (scoring {len(sample)})", flush=True)
    grid = []
    for fr in [1000, 2000]:
        for m in [1, 10]:
            grid += [("ask_all", 0, False, m, "clean", fr), ("fixed", 6, False, m, "clean", fr),
                     ("fixed", 9, False, m, "clean", fr), ("engine", 100, False, m, "clean", fr),
                     ("engine", 50, False, m, "clean", fr), ("engine", 25, False, m, "clean", fr)]
        for leaf in [10, 30]:
            grid.append(("tree", (50, leaf), True, 1, "clean", fr))
    chunks = np.array_split(sample, workers)
    jobs = [(*g, ch) for g in grid for ch in chunks]
    t0 = time.time()
    res = {}
    with Pool(workers, initializer=init) as pool:
        for key, out in pool.imap_unordered(work, jobs):
            res.setdefault(key, []).extend(out)
    print(f"{'fr cost':>7} {'policy':<8}{'knob':>10}{'min_ess':>8}  {'questions':>9}{'real loss':>10}{'claimed':>9}"
          f"{'prog recall':>12}{'false rec':>10}")
    for key in grid:
        kind, knob, look, m, _, frc = key
        q, real, claimed, recall, fr = score(G["val"], loss_for(frc), res[key])
        print(f"{frc:>7} {kind:<8}{str(knob):>10}{m:>8}  {q:9.2f}{real:10.0f}{claimed:9.0f}{recall:12.3f}{fr:10.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
