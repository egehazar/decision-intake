"""POST HOC, follows diag_adapt.py: a stronger tree that never runs out of data (deep_backoff: an answer
that would leave too few training households doesn't narrow its household set). Same 300 households
and plain intakes as diag_adapt.py; the engine's rows are reused from results/diag_adapt_300.jsonl.

usage: python experiments/diag_adapt2.py 300 WORKERS
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
from decisionintake import GreedyTree  # noqa: E402

G = A.G
KINDS = ["deep_tree", "deep_tree_leaf100", "deep_tree_k8"]


def make(kind):
    if kind not in G["trees"]:
        kw = dict(train=G["train"], loss=G["loss"], budget=18, min_leaf=30, question_cost=A.BASE, lookahead=True,
                  decide_with_session=True, backoff=True, unknown_backoff=True, deep_backoff=True, name=kind)
        if kind == "deep_tree_leaf100":
            kw["min_leaf"] = 100
        if kind == "deep_tree_k8":
            kw.update(budget=8, question_cost=0.0, lookahead=False)
        G["trees"][kind] = GreedyTree(**kw)
    return G["trees"][kind]


def work(job):
    kind, rows = job
    pol = make(kind)
    m = np.ones(len(G["train"].questions))
    return kind, [{"i": int(i), **A.run(pol, i, "S0", m)} for i in rows]


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    A.init()
    runs = {}
    for line in open(f"results/diag_adapt_{n}.jsonl"):
        r = json.loads(line)
        runs.setdefault(r["policy"], {})[r["i"]] = r
    idx = np.array(sorted(runs["engine"]))
    path = f"results/diag_adapt2_{n}.jsonl"
    jobs = [(k, ch) for k in KINDS for ch in np.array_split(idx, max(1, n // 25))]
    t0 = time.time()
    with Pool(workers, initializer=A.init) as pool, open(path, "w") as f:
        for kind, out in pool.imap_unordered(work, jobs):
            for r in out:
                f.write(json.dumps({"policy": kind, **r}) + "\n")
            f.flush()
    print(f"done in {time.time() - t0:.0f}s")
    for line in open(path):
        r = json.loads(line)
        runs.setdefault(r["policy"], {})[r["i"]] = r
    pop = G["test"]
    miss, fr = G["loss"].realised(pop)
    w = pop.weight[idx] / pop.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(2000, len(idx)))

    def total(k):
        rec = np.array([runs[k][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        return loss, np.array([runs[k][i]["paid"] for i in idx]), np.array([runs[k][i]["q"] for i in idx], float)

    print(f"POST HOC, the same {len(idx)} fresh test households as diag_adapt, plain intakes")
    for k in ["engine", "tree"] + KINDS + ["engine_k8", "fair_tree_k8"]:
        loss, paid, q = total(k)
        stops = [runs[k][i].get("stop") for i in idx if runs[k][i].get("stop")]
        st = ("  stopped: " + ", ".join(f"{s} {stops.count(s) / len(stops):.0%}" for s in sorted(set(stops)))) if stops else ""
        print(f"  {k:<18} questions {w @ q:5.2f}  loss {w @ loss:6.0f}  loss+questions {w @ (loss + paid):6.0f}{st}")
    for a, b, use_paid in [("engine", "deep_tree", True), ("engine", "deep_tree_leaf100", True),
                           ("engine_k8", "deep_tree_k8", False)]:
        la, pa, _ = total(a)
        lb, pb, _ = total(b)
        d = (la + pa * use_paid) - (lb + pb * use_paid)
        bs = [np.average(d[x], weights=w[x]) for x in boots]
        print(f"  {a} minus {b:<18} {w @ d:6.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")


if __name__ == "__main__":
    main()
