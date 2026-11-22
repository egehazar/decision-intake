"""Run-time adaptation test, pre-registered in DECISIONS D25.

The engine and a cost-sensitive tree both stop on their own at a $15 base question cost. Conditions
then change in ways the tree wasn't grown for: facts volunteered up front, refused questions,
unequal question costs, and all three at once. The tree is never told; a regrown tree is the
check on whether rebuilding closes any gap.

usage: python experiments/run_adapt.py N WORKERS
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
import run_final as RF  # noqa: E402
from decisionintake import GreedyTree, VOIPolicy  # noqa: E402
from decisionintake.core import Session  # noqa: E402

G = RF.G
BASE = 15.0
SCENARIOS = ["S0", "S1", "S2", "S3", "S4"]
COSTLY = ["earned_income", "self_employed", "other_income", "social_security", "assets", "disabled",
          "infant_or_pregnant"]
CHEAP = ["state", "household_size", "children", "senior", "married", "tenure"]


def multipliers(scenario):
    m = np.ones(len(G["train"].questions))
    if scenario in ("S3", "S4"):
        for q in COSTLY:
            m[G["train"].qindex(q)] = 3.0
        for q in CHEAP:
            m[G["train"].qindex(q)] = 0.5
    return m


def with_costs(pop, m):
    qs = [dataclasses.replace(q, cost=float(c)) for q, c in zip(pop.questions, m)]
    return dataclasses.replace(pop, questions=qs)


def init():
    RF.init()
    G["base_questions"] = list(G["train"].questions)
    G["trees"] = {}


def tree(kind):
    if kind not in G["trees"]:
        m = multipliers("S3") if kind == "tree_regrown" else np.ones(len(G["base_questions"]))
        G["trees"][kind] = GreedyTree(train=with_costs(G["train"], m), loss=G["loss"], budget=18, min_leaf=30,
                                      question_cost=BASE, lookahead=True, decide_with_session=True, backoff=True,
                                      unknown_backoff=True, name=kind)
    return G["trees"][kind]


def conditions(i):
    """The household's scenario draws: same for every policy and every scenario that uses them."""
    rng = np.random.default_rng(2027 * 1_000_003 + int(i))
    nq = len(G["base_questions"])
    n_vol = int(rng.integers(0, 4))
    volunteered = rng.choice(nq, size=n_vol, replace=False).tolist()
    refused = rng.choice(nq, size=2, replace=False).tolist()
    return volunteered, refused


def run(policy, i, scenario, m):
    train = G["train"]
    row = G["test"].answers[i]
    volunteered, refused = conditions(i)
    s = Session(train, G["loss"], min_ess=RF.MIN_ESS, outcome=G["outcome"])
    policy.start(s)
    if scenario in ("S1", "S4"):
        for j in volunteered:
            s.answer(int(j), int(row[j]), source="opener")
    refuse = set(refused) if scenario in ("S2", "S4") else set()
    n, paid, t_choose = 0, 0.0, 0.0
    while n < len(train.questions):
        t0 = time.perf_counter()
        j = policy.next_question(s, n)
        t_choose += time.perf_counter() - t0
        if j is None:
            break
        n += 1
        paid += BASE * m[j]
        if j in refuse:
            s.observe(j, np.ones(len(train.questions[j].options)), source="refused")
        else:
            s.answer(j, int(row[j]))
        if hasattr(policy, "after_answer"):
            policy.after_answer(s, j)
    rec = np.asarray(policy.final_recommend(s), bool)
    return {"q": n, "paid": round(paid, 2), "rec": rec.astype(int).tolist(), "ms": round(1000 * t_choose / max(n, 1), 1)}


def work(job):
    kind, scenario, rows = job
    m = multipliers(scenario)
    G["train"].questions = [dataclasses.replace(q, cost=float(c)) for q, c in zip(G["base_questions"], m)]
    pol = VOIPolicy(question_cost=BASE) if kind == "engine" else tree(kind)
    out = []
    for i in rows:
        out.append({"i": int(i), **run(pol, i, scenario, m)})
    return (kind, scenario), out


def grid():
    g = [(kind, sc) for sc in SCENARIOS for kind in ["engine", "tree"]]
    return g + [("tree_regrown", "S3"), ("tree_regrown", "S4")]


def sample(n):
    used = set(np.random.default_rng(2026).choice(len(G["test"]), size=min(1200, len(G["test"])), replace=False).tolist())
    fresh = np.array([i for i in range(len(G["test"])) if i not in used])
    return np.sort(np.random.default_rng(2027).choice(fresh, size=n, replace=False))


def analyse(path, idx):
    pop = G["test"]
    miss, fr = G["loss"].realised(pop)
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs.setdefault((r["policy"], r["scenario"]), {})[r["i"]] = r
    w = pop.weight[idx] / pop.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(2000, len(idx)))

    def cols(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        paid = np.array([runs[key][i]["paid"] for i in idx])
        q = np.array([runs[key][i]["q"] for i in idx], float)
        ms = np.array([runs[key][i]["ms"] for i in idx])
        return loss, paid, q, ms

    print(f"{len(idx)} fresh test households, survey-weighted, $/household/year (loss + question costs paid)")
    print(f"{'scenario':<9}{'policy':<14}{'questions':>10}{'loss':>8}{'paid':>7}{'total':>8}{'ms/turn':>9}")
    for sc in SCENARIOS:
        for kind in ["engine", "tree", "tree_regrown"]:
            if (kind, sc) not in runs:
                continue
            loss, paid, q, ms = cols((kind, sc))
            print(f"{sc:<9}{kind:<14}{w @ q:10.2f}{w @ loss:8.0f}{w @ paid:7.0f}{w @ (loss + paid):8.0f}{ms.mean():9.1f}")
    print("\nengine minus tree, total cost [95% CI]   (negative = engine cheaper)")
    for sc in SCENARIOS:
        for other in ["tree", "tree_regrown"]:
            if (other, sc) not in runs:
                continue
            le, pe, *_ = cols(("engine", sc))
            lt, pt, *_ = cols((other, sc))
            d = (le + pe) - (lt + pt)
            bs = [np.average(d[b], weights=w[b]) for b in boots]
            print(f"  {sc} vs {other:<13} {w @ d:7.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    init()
    idx = sample(n)
    path = f"results/adapt_{n}.jsonl"
    if len(sys.argv) > 3 and sys.argv[3] == "analyse":
        analyse(path, idx)
        return
    jobs = [(kind, sc, ch) for kind, sc in grid() for ch in np.array_split(idx, max(1, n // 25))]
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(path, "w") as f:
        for d, (key, out) in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in out:
                f.write(json.dumps({"policy": key[0], "scenario": key[1], **r}) + "\n")
            f.flush()
            if d % 24 == 0:
                print(f"{d}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    analyse(path, idx)


if __name__ == "__main__":
    main()
