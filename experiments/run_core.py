"""The decisive comparison: questions asked vs decision quality, engine vs baselines.

Three kinds of simulated user:
  clean     - answers every question exactly, volunteers nothing
  opener    - first says 3 facts unprompted ("single mum, two kids, in Ohio"), then answers exactly
  realistic - volunteers 2 facts, picks a neighbouring range per the noise model, says "don't know" 10% of the time

Each policy is swept over its knob (question cost for the engine and the tree, length for the
fixed form and entropy) so we compare whole curves, not one operating point.

usage: python experiments/run_core.py [n_test] [workers]
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))

from benefits.population import build, load_table  # noqa: E402

from decisionintake import AskAll, EntropyGreedy, FixedOrder, GreedyTree, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one, split_by_group  # noqa: E402

G = {}


def init():
    pop, loss, meta = build(load_table())
    tr, te = split_by_group(pop, 0.15, seed=0)
    G.update(pop=pop, loss=loss, meta=meta, train=pop.subset(tr), test=pop.subset(te), test_idx=te)
    G["fixed_order"] = FixedOrder.fit(G["train"], loss).order
    G["trees"] = {}


def users(meta):
    v = meta["volunteerable"]
    return {
        "clean": UserModel(name="clean"),
        "opener": UserModel(opener=3, volunteerable=v, name="opener"),
        "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.10, name="realistic"),
    }


def make_policy(kind, knob):
    if kind == "engine":
        return VOIPolicy(question_cost=knob, budget=99)
    if kind == "tree":
        key = knob
        if key not in G["trees"]:
            G["trees"][key] = GreedyTree(train=G["train"], loss=G["loss"], budget=99, min_leaf=30, question_cost=knob)
        t = G["trees"][key]
        return t
    if kind == "fixed":
        return FixedOrder(order=G["fixed_order"], budget=int(knob))
    if kind == "entropy":
        return EntropyGreedy(budget=int(knob))
    if kind == "ask_all":
        return AskAll()
    raise ValueError(kind)


def work(job):
    kind, knob, uname, rows, seed = job
    user = users(G["meta"])[uname]
    test = G["test"]
    out = []
    for i in rows:
        rng = np.random.default_rng(seed * 1_000_003 + int(i))
        pol = make_policy(kind, knob)
        o = run_one(pol, G["train"], G["loss"], test.answers[i], user, rng)
        out.append((int(i), o.n_questions, o.recommend.astype(int).tolist(), o.ess_at_stop))
    return kind, knob, uname, out


def metrics(test, loss, rows):
    idx = np.array([r[0] for r in rows])
    nq = np.array([r[1] for r in rows], float)
    rec = np.array([r[2] for r in rows], bool)
    w = test.weight[idx] / test.weight[idx].sum()
    el, val = test.eligible[idx], test.value[idx]
    miss, fr = loss.matrices(test.subset(idx))
    realised = (np.where(rec, 0.0, miss) + np.where(rec, fr, 0.0)).sum(1)
    val_el = np.where(el, val, 0.0)
    any_el = el.any(1)
    return {
        "questions": float(w @ nq),
        "loss": float(w @ realised),
        "dollar_recall": float((w[:, None] * val_el * rec).sum() / (w[:, None] * val_el).sum()),
        "programme_recall": float((w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum()),
        "false_rec_share": float((w[:, None] * (~el & rec)).sum() / max((w[:, None] * rec).sum(), 1e-12)),
        "miss_any": float((w * (any_el & (el & ~rec).any(1))).sum() / (w * any_el).sum()),
        "n": int(len(idx)),
    }


def main():
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    init()
    test = G["test"]
    rng = np.random.default_rng(42)
    sample = rng.choice(len(test), size=min(n_test, len(test)), replace=False)
    print(f"train {len(G['train'])} households, test {len(test)} (scoring {len(sample)}), "
          f"fixed order: {[G['train'].questions[j].id for j in G['fixed_order'][:8]]}", flush=True)
    grid = (
        [("engine", c) for c in [400, 200, 100, 50, 25, 10]]
        + [("tree", c) for c in [400, 200, 100, 50, 25, 10]]
        + [("fixed", k) for k in [2, 4, 6, 8, 10, 12]]
        + [("entropy", k) for k in [2, 4, 6, 8, 10, 12]]
        + [("ask_all", 0)]
    )
    chunks = np.array_split(sample, workers)
    jobs = [(k, knob, u, ch, 7) for k, knob in grid for u in ["clean", "opener", "realistic"] for ch in chunks]
    t0 = time.time()
    results = {}
    with Pool(workers, initializer=init) as pool:
        for kind, knob, uname, out in pool.imap_unordered(work, jobs):
            results.setdefault((kind, knob, uname), []).extend(out)
    rows = []
    for (kind, knob, uname), out in sorted(results.items(), key=lambda kv: (kv[0][2], kv[0][0], -kv[0][1])):
        m = metrics(test, G["loss"], out)
        rows.append({"policy": kind, "knob": knob, "user": uname, **m})
        print(f"{uname:<10} {kind:<8} {knob:>5}  q={m['questions']:5.2f}  loss=${m['loss']:7.0f}  "
              f"$recall={m['dollar_recall']:.3f}  prog_recall={m['programme_recall']:.3f}  "
              f"false_rec={m['false_rec_share']:.3f}  miss_any={m['miss_any']:.3f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/core_{len(sample)}.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
