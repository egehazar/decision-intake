"""Engine with the linear vs the neural outcome model, on the VALIDATION split only.

Each policy runs on the same validation households with the same draws (paired). Question cost $15
(the D25 base). Policies: the engine with its own stopping rule, and the full questionnaire.
Users: clean answers, and realistic ones (2 volunteered facts, misreports, 10% don't know).

usage: python experiments/val_v2.py N WORKERS [analyse]
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")  # several workers share the CPU
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from decisionintake import AskAll, LossModel, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one  # noqa: E402

FR, MIN_ESS, QCOST = 1000.0, 10.0, 15.0
NEURAL = dict(hidden=(512, 256), dropout=0.4, wd=0.3)
G = {}


def init():
    from common import smoothed, smoothed_neural, splits

    pop, loss, meta, tr, va, _ = splits()
    G["loss"] = LossModel(np.full(len(pop.decisions), FR), loss.miss_floor)
    G["val"], G["meta"] = pop.subset(va), meta
    G["linear"] = smoothed(pop, tr, "train_tune")
    G["neural"] = smoothed_neural(pop, tr, "train_tune", **NEURAL)


def users():
    v = G["meta"]["volunteerable"]
    return {"clean": UserModel(name="clean"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.10, name="realistic")}


def work(job):
    model, kind, uname, rows = job
    train, outcome = G[model]
    pol = VOIPolicy(question_cost=QCOST) if kind == "engine" else AskAll()
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 17 + int(i))
        o = run_one(pol, train, G["loss"], G["val"].answers[i], users()[uname], rng, min_ess=MIN_ESS, outcome=outcome)
        out.append({"model": model, "policy": kind, "user": uname, "i": int(i), "q": o.n_questions,
                    "rec": o.recommend.astype(int).tolist(), "claimed": round(o.risk_at_stop, 2)})
    return out


def sample(n):
    return np.sort(np.random.default_rng(77).choice(len(G["val"]), size=n, replace=False))


def analyse(path, idx):
    val = G["val"]
    miss, fr = G["loss"].realised(val)
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs.setdefault((r["model"], r["policy"], r["user"]), {})[r["i"]] = r
    w = val.weight[idx] / val.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(2000, len(idx)))

    def tot(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        q = np.array([runs[key][i]["q"] for i in idx], float)
        claimed = np.array([runs[key][i]["claimed"] for i in idx])
        wrong = (rec & ~val.eligible[idx]).sum() / max(rec.sum(), 1)
        found = (rec & val.eligible[idx]).sum() / max(val.eligible[idx].sum(), 1)
        return loss, q, claimed, wrong, found

    print(f"{len(idx)} validation households, $/household/year; total = loss + ${QCOST:.0f} per question")
    for u in ["clean", "realistic"]:
        for p in ["engine", "ask_all"]:
            for m in ["linear", "neural"]:
                if (m, p, u) not in runs:
                    continue
                loss, q, c, wr, fo = tot((m, p, u))
                print(f"  {u:<10}{p:<9}{m:<8} q {w @ q:5.2f}  loss {w @ loss:6.0f}  total {w @ (loss + QCOST * q):6.0f}"
                      f"  claimed {w @ c:6.0f}  wrong recs {100 * wr:3.0f}%  found {100 * fo:3.0f}%")
        for p in ["engine", "ask_all"]:
            if ("linear", p, u) in runs and ("neural", p, u) in runs:
                l0, q0, *_ = tot(("linear", p, u))
                l1, q1, *_ = tot(("neural", p, u))
                d = (l1 + QCOST * q1) - (l0 + QCOST * q0)
                bs = [np.average(d[b], weights=w[b]) for b in boots]
                print(f"  {u} {p}: neural minus linear total {w @ d:6.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    init()
    idx = sample(n)
    path = f"results/val_v2_{n}.jsonl"
    if len(sys.argv) > 3:
        analyse(path, idx)
        return
    jobs = [(m, k, u, ch) for m in ["linear", "neural"] for k in ["engine", "ask_all"] for u in ["clean", "realistic"]
            for ch in np.array_split(idx, max(1, n // 25))]
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(path, "w") as f:
        for d, out in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in out:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if d % 16 == 0:
                print(f"{d}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    analyse(path, idx)


if __name__ == "__main__":
    main()
