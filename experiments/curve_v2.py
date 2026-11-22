"""Loss against questions, every policy on the v2 model (secondary, after D28; GPT round 4).

Separates better prediction from better question choice: the v2 engine asking exactly k questions
against the best fixed set of k questions (beam search on the training population), k = 3, 5, 7, 9,
12, both deciding with the same v2 model, on 600 of D28's 2,000 fresh test households (seed 2030),
clean answers. The full questionnaire and the engine's own stopping come from results/v2_2000.jsonl.

usage: python experiments/curve_v2.py N WORKERS [analyse]
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import run_v2 as R  # noqa: E402
from decisionintake import FixedOrder, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one  # noqa: E402

KS = [3, 5, 7, 9, 12]
G = R.G


def init():
    R.init()
    G["fixed"] = {}


def work(job):
    kind, k, rows = job
    train, outcome = G["neural"]
    if kind == "engine_k":
        pol = VOIPolicy(question_cost=0.0, budget=k)
    else:
        if k not in G["fixed"]:
            G["fixed"][k] = FixedOrder.fit_subset(train, G["loss"], k)
        pol = G["fixed"][k]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 29 + int(i))
        o = run_one(pol, train, G["loss"], G["test"].answers[i], UserModel(name="clean"), rng, min_ess=R.MIN_ESS,
                    outcome=outcome)
        out.append({"policy": kind, "k": k, "i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist()})
    return out


def analyse(path, idx):
    test = G["test"]
    miss, fr = G["loss"].realised(test)
    w = test.weight[idx] / test.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(4000, len(idx)))
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs.setdefault((r["policy"], r["k"]), {})[r["i"]] = r
    main = R.load("results/v2_2000.jsonl")
    for key in [("neural", "engine", "clean"), ("neural", "ask_all", "clean")]:
        runs[(key[1], 0)] = main[key]

    def loss(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        return (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1), np.array([runs[key][i]["q"] for i in idx], float)

    print(f"{len(idx)} fresh test households (a seed-2030 subset of D28's sample), clean answers, v2 model for every policy")
    print(f"{'k':>4}{'engine_k loss':>15}{'fixed set loss':>16}{'engine minus fixed [95% CI]':>34}")
    rows = []
    for k in KS:
        le, _ = loss(("engine_k", k))
        lf, _ = loss(("fixed_k", k))
        d = le - lf
        bs = np.array([np.average(d[b], weights=w[b]) for b in boots])
        print(f"{k:>4}{w @ le:15.0f}{w @ lf:16.0f}{w @ d:12.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")
        rows.append({"k": k, "engine": float(w @ le), "fixed": float(w @ lf), "diff": [float(w @ d), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]})
    la, qa = loss(("ask_all", 0))
    lo, qo = loss(("engine", 0))
    print(f"  full questionnaire: 18 questions, loss {w @ la:.0f}; engine with its own stopping: {w @ qo:.1f} questions, loss {w @ lo:.0f}")
    with open(path.replace(".jsonl", "_curve.json"), "w") as f:
        json.dump({"rows": rows, "full": float(w @ la), "own": [float(w @ qo), float(w @ lo)]}, f, indent=1)


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    init()
    idx = np.sort(np.random.default_rng(2030).choice(R.sample(2000), size=n, replace=False))
    path = f"results/curve_v2_{n}.jsonl"
    if len(sys.argv) > 3:
        analyse(path, idx)
        return
    jobs = [(kind, k, ch) for k in KS for kind in ["engine_k", "fixed_k"] for ch in np.array_split(idx, max(1, n // 25))]
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(path, "w") as f:
        for d, out in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in out:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if d % 20 == 0:
                print(f"{d}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    analyse(path, idx)


if __name__ == "__main__":
    main()
