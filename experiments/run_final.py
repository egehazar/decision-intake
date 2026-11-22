"""The final comparison on the held-out TEST split. Run once, after all tuning (on validation) is done.

Settings fixed in advance from validation (DECISIONS D16): wrong-recommendation cost $1,000, support
floor 10 for every policy that decides from the belief, tree minimum leaf 10. Model trained on
train + validation. Policies, each at equal question counts k = 3, 5, 7, 9, 12:
  engine_k - the engine, asking exactly k questions
  fixed_k  - the best fixed order (greedy on the training population), first k questions
  tree_k   - a tree grown greedily on the same loss, depth k
plus the engine with its own stopping rule (question cost $50, $15, $5) and the full questionnaire.
Users: clean, opener (volunteers 3 facts first), realistic (2 facts, misreports, 10% "don't know").
Every household's outcome is saved to results/final_<n>.jsonl for paired bootstrap intervals.

usage: python experiments/run_final.py N_TEST WORKERS
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))

from decisionintake import AskAll, FixedOrder, GreedyTree, LossModel, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one  # noqa: E402

FR_COST, MIN_ESS, TREE_LEAF = 1000.0, 10.0, 10
KS = [3, 5, 7, 9, 12]
G = {}


def init():
    from common import smoothed, splits

    pop, loss, meta, tr, va, test = splits()
    G["loss"] = LossModel(np.full(len(pop.decisions), FR_COST), loss.miss_floor)
    G["train"], G["outcome"] = smoothed(pop, np.sort(np.concatenate([tr, va])), "rest")
    G["test"], G["test_idx"], G["meta"] = pop.subset(test), test, meta
    G["order"] = FixedOrder.fit(G["train"], G["loss"]).order
    G["tree"] = {}


def users():
    v = G["meta"]["volunteerable"]
    return {"clean": UserModel(name="clean"),
            "opener": UserModel(opener=3, volunteerable=v, name="opener"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.10, name="realistic")}


def make(kind, k):
    if kind == "engine_k":
        return VOIPolicy(question_cost=0.0, budget=k)
    if kind == "engine":
        return VOIPolicy(question_cost=float(k))
    if kind == "fixed_k":
        return FixedOrder(order=G["order"], budget=k)
    if kind == "tree_k":
        if k not in G["tree"]:
            G["tree"][k] = GreedyTree(train=G["train"], loss=G["loss"], budget=k, min_leaf=TREE_LEAF,
                                      question_cost=0.0, lookahead=False)
        return G["tree"][k]
    return AskAll()


def work(job):
    kind, k, uname, rows = job
    user = users()[uname]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 13 + int(i))  # same draws for every policy: paired
        o = run_one(make(kind, k), G["train"], G["loss"], G["test"].answers[i], user, rng,
                    min_ess=MIN_ESS, outcome=G["outcome"])
        out.append({"i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist(),
                    "claimed": round(o.risk_at_stop, 2), "ess": round(o.ess_at_stop, 1)})
    return (kind, k, uname), out


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    init()
    sample = np.random.default_rng(2026).choice(len(G["test"]), size=min(n, len(G["test"])), replace=False)
    print(f"train {len(G['train'])}, test {len(G['test'])} (scoring {len(sample)}); "
          f"fixed order {[G['train'].questions[j].id for j in G['order'][:12]]}", flush=True)
    grid = []
    for u in ["clean", "opener", "realistic"]:
        grid += [(kind, k, u) for kind in ["engine_k", "fixed_k", "tree_k"] for k in KS]
        grid += [("engine", lam, u) for lam in [50, 15, 5]]
        grid.append(("ask_all", 0, u))
    jobs = [(*g, ch) for g in grid for ch in np.array_split(sample, workers * 2)]
    os.makedirs("results", exist_ok=True)
    path = f"results/final_{len(sample)}.jsonl"
    t0 = time.time()
    done = 0
    with Pool(workers, initializer=init) as pool, open(path, "w") as f:
        for key, out in pool.imap_unordered(work, jobs):
            for r in out:
                f.write(json.dumps({"policy": key[0], "k": key[1], "user": key[2], **r}) + "\n")
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
