"""Question choice alone: engine and fixed form each ask exactly k questions. Validation split.
Also the engine with its own stopping rule at a few question costs.

usage: python experiments/k_compare.py N_VAL WORKERS FR_COST
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from decisionintake import AskAll, FixedOrder, LossModel, VOIPolicy
from decisionintake.simulate import UserModel, run_one

G = {}


def init(fr):
    from common import smoothed, splits
    pop, loss, meta, tr, va, _ = splits()
    G["loss"] = LossModel(np.full(10, float(fr)), loss.miss_floor)
    G["train"], G["outcome"] = smoothed(pop, tr, "train_tune")
    G["val"] = pop.subset(va)
    G["order"] = FixedOrder.fit(G["train"], G["loss"]).order
    G["meta"] = meta


def make(kind, k):
    if kind == "engine_k":
        return VOIPolicy(question_cost=0.0, budget=k)
    if kind == "engine":
        return VOIPolicy(question_cost=float(k))
    if kind == "fixed":
        return FixedOrder(order=G["order"], budget=k)
    return AskAll()


def work(job):
    kind, k, m, user, rows = job
    v = G["meta"]["volunteerable"]
    um = {"clean": UserModel(name="clean"),
          "opener": UserModel(opener=3, volunteerable=v, name="opener"),
          "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.1, name="realistic")}[user]
    out = []
    for i in rows:
        o = run_one(make(kind, k), G["train"], G["loss"], G["val"].answers[i], um, np.random.default_rng(77 + int(i)),
                    min_ess=m, outcome=G["outcome"])
        out.append((int(i), o.n_questions, o.recommend, o.risk_at_stop))
    return (kind, k, m, user), out


def main():
    n, workers, fr = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    init(fr)
    sample = np.random.default_rng(5).choice(len(G["val"]), size=n, replace=False)
    grid = []
    for user in ["clean", "realistic"]:
        for m in [1, 10]:
            grid += [(kind, k, m, user) for kind in ["engine_k", "fixed"] for k in [3, 5, 7, 9, 12]]
            grid += [("engine", lam, m, user) for lam in [50, 15, 5]]
        grid.append(("ask_all", 0, 1, user))
    jobs = [(*g, ch) for g in grid for ch in np.array_split(sample, workers)]
    t0 = time.time()
    res = {}
    with Pool(workers, initializer=init, initargs=(fr,)) as pool:
        for key, out in pool.imap_unordered(work, jobs):
            res.setdefault(key, []).extend(out)
    val, L = G["val"], G["loss"]
    print(f"wrong-recommendation cost ${fr:.0f}, {n} validation households")
    print(f"{'user':<10}{'policy':<9}{'k/cost':>7}{'floor':>6}{'questions':>10}{'loss':>8}{'claimed':>9}{'found':>7}{'wrong':>7}")
    for key in grid:
        out = res[key]
        idx = np.array([r[0] for r in out]); nq = np.array([r[1] for r in out], float)
        rec = np.array([r[2] for r in out], bool); cl = np.array([r[3] for r in out], float)
        w = val.weight[idx] / val.weight[idx].sum(); el = val.eligible[idx]
        miss, frm = L.realised(val.subset(idx))
        real = (np.where(rec, 0, miss) + np.where(rec, frm, 0)).sum(1)
        kind, k, m, user = key
        print(f"{user:<10}{kind:<9}{k:>7}{m:>6}{w @ nq:10.2f}{w @ real:8.0f}{w @ cl:9.0f}"
              f"{(w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum():7.3f}"
              f"{(w[:, None] * (~el & rec)).sum() / max((w[:, None] * rec).sum(), 1e-12):7.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
