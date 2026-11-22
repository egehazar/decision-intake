"""POST HOC diagnostic for D25 (not pre-registered): why does the cost-sensitive tree lose to the engine
by ~$700 when both stop on their own, when a tree asking a fixed number of questions tied (D22)?

Plain intakes only (S0), on 300 further test households never used before (seed 2028), so the
D25 sample stays as it was. Variants:
  engine            - $15 a question, own stopping (as D25)
  tree              - as D25: cost $15, pair look-ahead, min leaf 30
  tree_nolook       - cost $15, no pair look-ahead
  tree_leaf10       - cost $15, min leaf 10
  engine_k8         - engine asking exactly 8 questions
  fair_tree_k8      - the D22 fair tree at 8 questions (cost 0, no look-ahead, min leaf 30)
For each tree session it records why the tree stopped: its node had too few training households
to split ("small"), or no question was worth its cost ("no_gain").

usage: python experiments/diag_adapt.py N WORKERS
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
import run_adapt as A  # noqa: E402
from decisionintake import GreedyTree, VOIPolicy  # noqa: E402

G = A.G
KINDS = ["engine", "tree", "tree_nolook", "tree_leaf10", "engine_k8", "fair_tree_k8"]


def make(kind):
    if kind == "engine":
        return VOIPolicy(question_cost=A.BASE)
    if kind == "engine_k8":
        return VOIPolicy(question_cost=0.0, budget=8)
    if kind not in G["trees"]:
        kw = dict(train=G["train"], loss=G["loss"], budget=18, min_leaf=30, question_cost=A.BASE, lookahead=True,
                  decide_with_session=True, backoff=True, unknown_backoff=True, name=kind)
        if kind == "tree_nolook":
            kw["lookahead"] = False
        if kind == "tree_leaf10":
            kw["min_leaf"] = 10
        if kind == "fair_tree_k8":
            kw.update(budget=8, question_cost=0.0, lookahead=False)
        G["trees"][kind] = GreedyTree(**kw)
    return G["trees"][kind]


def work(job):
    kind, rows = job
    pol = make(kind)
    m = np.ones(len(G["train"].questions))
    out = []
    for i in rows:
        out.append({"i": int(i), **A.run(pol, i, "S0", m)})
    return kind, out


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    A.init()
    used = set(np.random.default_rng(2026).choice(len(G["test"]), size=1200, replace=False).tolist())
    used |= set(A.sample(600).tolist())
    fresh = np.array([i for i in range(len(G["test"])) if i not in used])
    idx = np.sort(np.random.default_rng(2028).choice(fresh, size=n, replace=False))
    path = f"results/diag_adapt_{n}.jsonl"
    jobs = [(k, ch) for k in KINDS for ch in np.array_split(idx, max(1, n // 25))]
    t0 = time.time()
    with Pool(workers, initializer=A.init) as pool, open(path, "w") as f:
        for kind, out in pool.imap_unordered(work, jobs):
            for r in out:
                f.write(json.dumps({"policy": kind, **r}) + "\n")
            f.flush()
    print(f"done in {time.time() - t0:.0f}s")
    pop = G["test"]
    miss, fr = G["loss"].realised(pop)
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs.setdefault(r["policy"], {})[r["i"]] = r
    w = pop.weight[idx] / pop.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(2000, len(idx)))

    def total(k):
        rec = np.array([runs[k][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        return loss, np.array([runs[k][i]["paid"] for i in idx]), np.array([runs[k][i]["q"] for i in idx], float)

    print(f"POST HOC, {n} further fresh test households, plain intakes, survey-weighted $/household/year")
    for k in KINDS:
        loss, paid, q = total(k)
        stops = [runs[k][i].get("stop") for i in idx if runs[k][i].get("stop")]
        st = ""
        if stops:
            c = {s: stops.count(s) / len(stops) for s in set(stops)}
            st = "  stopped: " + ", ".join(f"{s} {v:.0%}" for s, v in sorted(c.items()))
        print(f"  {k:<14} questions {w @ q:5.2f}  loss {w @ loss:6.0f}  loss+questions {w @ (loss + paid):6.0f}{st}")
    for a, b, use_paid in [("engine", "tree", True), ("engine", "tree_nolook", True), ("engine", "tree_leaf10", True),
                           ("engine_k8", "fair_tree_k8", False)]:
        la, pa, _ = total(a)
        lb, pb, _ = total(b)
        d = (la + pa * use_paid) - (lb + pb * use_paid)
        bs = [np.average(d[x], weights=w[x]) for x in boots]
        print(f"  {a} minus {b:<14} {w @ d:6.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")


if __name__ == "__main__":
    main()
