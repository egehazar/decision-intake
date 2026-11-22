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

from decisionintake import AskAll, FixedOrder, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one, split_by_group  # noqa: E402

G = {}


def init():
    pop, loss, meta = build(load_table())
    rest, _test = split_by_group(pop, 0.15, seed=0)  # same test split as run_core: never touched here
    sub = pop.subset(rest)
    tr, va = split_by_group(sub, 0.12, seed=1)
    G.update(pop=pop, loss=loss, meta=meta, train=sub.subset(tr), val=sub.subset(va))
    G["fixed_order"] = FixedOrder.fit(G["train"], loss).order


def policy(kind, knob, look):
    if kind == "engine":
        return VOIPolicy(question_cost=knob, lookahead=look)
    if kind == "fixed":
        return FixedOrder(order=G["fixed_order"], budget=int(knob))
    return AskAll()


def work(job):
    kind, knob, look, min_ess, uname, rows = job
    v = G["meta"]["volunteerable"]
    user = {"clean": UserModel(name="clean"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.1, name="realistic")}[uname]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 11 + int(i))
        o = run_one(policy(kind, knob, look), G["train"], G["loss"], G["val"].answers[i], user, rng, min_ess=min_ess)
        out.append((int(i), o.n_questions, o.recommend, o.risk_at_stop))
    return job[:5], out


def score(val, loss, out):
    idx = np.array([r[0] for r in out])
    nq = np.array([r[1] for r in out], float)
    rec = np.array([r[2] for r in out], bool)
    claimed = np.array([r[3] for r in out], float)
    w = val.weight[idx] / val.weight[idx].sum()
    miss, fr = loss.matrices(val.subset(idx))
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
    for m in [1, 10, 30, 100, 300]:
        grid += [("ask_all", 0, False, m, "clean"), ("fixed", 8, False, m, "clean"), ("fixed", 12, False, m, "clean")]
    for m in [1, 30, 100]:
        for look in [False, True]:
            for lam in [100, 25]:
                grid.append(("engine", lam, look, m, "clean"))
    chunks = np.array_split(sample, workers)
    jobs = [(*g, ch) for g in grid for ch in chunks]
    t0 = time.time()
    res = {}
    with Pool(workers, initializer=init) as pool:
        for key, out in pool.imap_unordered(work, jobs):
            res.setdefault(key, []).extend(out)
    print(f"{'policy':<8}{'knob':>5}{'look':>6}{'min_ess':>8}  {'questions':>9}{'real loss':>10}{'claimed':>9}"
          f"{'prog recall':>12}{'false rec':>10}")
    for key in grid:
        q, real, claimed, recall, fr = score(G["val"], G["loss"], res[key])
        kind, knob, look, m, _ = key
        print(f"{kind:<8}{knob:>5}{str(look):>6}{m:>8}  {q:9.2f}{real:10.0f}{claimed:9.0f}{recall:12.3f}{fr:10.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
