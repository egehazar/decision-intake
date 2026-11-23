"""D33 development: does the engine's own support measure predict where it is wrong?

At the end of each interview the belief rests on some number of training households that fit the
answers before the support floor softened the evidence (Session.support()["raw_ess"]). Where that is
thin, the outcome model extrapolates. This runs the production engine (v2 + rules) on D32's 2,000
households (used only for D32's primary so far) and reports mistakes by support, to design an
abstention rule. The rule is then tested on fresh households, registered first.

usage: python experiments/support_dev.py N WORKERS [analyse]
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import run_guard as RG  # noqa: E402
import run_v2 as R  # noqa: E402
from decisionintake import VOIPolicy  # noqa: E402
from decisionintake.simulate import run_one  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "support_dev.jsonl")


def work(job):
    uname, rows = job
    train = R.G["neural"][0]
    outcome = R.G["arms"]["v2_rules"]
    pol = VOIPolicy(question_cost=R.QCOST)
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 34 + int(i))  # D32's draws
        o = run_one(pol, train, R.G["loss"], R.G["test"].answers[i], R.users()[uname], rng, min_ess=R.MIN_ESS,
                    outcome=outcome)
        out.append({"user": uname, "i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist(),
                    "p": np.round(o.p_final, 4).tolist(), "support": {k: round(v, 3) for k, v in o.support.items()}})
    return out


def run(n, workers):
    RG.init()
    idx = RG.sample(n)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["user"], r["i"]))
    jobs = []
    for uname in ["clean", "realistic"]:
        todo = [int(i) for i in idx if (uname, int(i)) not in done]
        jobs += [(uname, todo[k:k + 25]) for k in range(0, len(todo), 25)]
    t0 = time.time()
    with Pool(workers, initializer=RG.init) as pool, open(OUT, "a") as f:
        for k, res in enumerate(pool.imap_unordered(work, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"  {k + 1}/{len(jobs)} batches, {time.time() - t0:.0f}s", flush=True)


def analyse(n):
    RG.init()
    idx = RG.sample(n)
    test = R.G["test"]
    miss, fr = R.G["loss"].realised(test)
    el = test.eligible
    rows = {}
    for line in open(OUT):
        r = json.loads(line)
        rows[(r["user"], r["i"])] = r
    w_all = test.weight
    for uname in ["clean", "realistic"]:
        rs = [rows[(uname, int(i))] for i in idx]
        raw = np.array([r["support"]["raw_ess"] for r in rs])
        rec = np.array([r["rec"] for r in rs], bool)
        p = np.array([r["p"] for r in rs])
        e = el[idx]
        w = w_all[idx] / w_all[idx].sum()
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        fn = (~rec & e)  # missed programmes
        print(f"\n{uname}: {len(idx)} households; final raw support: median {np.median(raw):.1f}, "
              f"share under 10 {np.mean(raw < 10):.1%}, under 3 {np.mean(raw < 3):.1%}")
        for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 100), (100, np.inf)]:
            m = (raw >= lo) & (raw < hi)
            if not m.any():
                continue
            print(f"  raw support {lo:>4}-{hi:<4}: {m.sum():5d} households ({w[m].sum():5.1%} weighted); mistakes "
                  f"${np.average(loss[m], weights=w[m]):6,.0f}/household; missed {fn[m].sum() / max(e[m].sum(), 1):5.1%} of "
                  f"eligible programmes; 'unlikely' (p<0.1) calls that were eligible "
                  f"{(fn[m] & (p[m] < 0.1)).sum()} of {(~rec[m] & (p[m] < 0.1)).sum()}")


def grid(n):
    """Candidate abstention rules: in profiles backed by fewer than T households, programmes not recommended
    but with probability >= q become "can't tell". For each: how often a negative is wrong where the engine
    is thin vs supported, how many misses get flagged, and how many flags a household sees."""
    RG.init()
    idx = RG.sample(n)
    test = R.G["test"]
    miss, _ = R.G["loss"].realised(test)
    rows = {}
    for line in open(OUT):
        r = json.loads(line)
        rows[(r["user"], r["i"])] = r
    for uname in ["clean", "realistic"]:
        rs = [rows[(uname, int(i))] for i in idx]
        raw = np.array([r["support"]["raw_ess"] for r in rs])
        rec = np.array([r["rec"] for r in rs], bool)
        p = np.array([r["p"] for r in rs])
        e = test.eligible[idx]
        mv = np.where(e, np.maximum(miss[idx], 0), 0)
        neg = ~rec
        print(f"\n{uname}: negatives {neg.sum()}, of which eligible {(neg & e).sum()} (${mv[neg].sum():,.0f} a year missed)")
        for T in [3, 5, 10, 20]:
            thin = raw < T
            fn_thin = (neg & e)[thin].sum() / max(neg[thin].sum(), 1)
            fn_sup = (neg & e)[~thin].sum() / max(neg[~thin].sum(), 1)
            for q in [0.01, 0.02, 0.05]:
                flag = neg & thin[:, None] & (p >= q)
                conf = neg & ~flag
                print(f"  T={T:>2} q={q:.2f}: thin {thin.mean():5.1%} of households; a negative is wrong "
                      f"{fn_thin:5.1%} thin vs {fn_sup:5.1%} supported; flags/household {flag.sum(1).mean():.2f} "
                      f"(thin: {flag.sum(1)[thin].mean() if thin.any() else 0:.2f}); misses flagged "
                      f"{(flag & e).sum()}/{(neg & e).sum()} (${mv[flag].sum():,.0f} of ${mv[neg].sum():,.0f}); "
                      f"confident negatives wrong {(conf & e).sum() / max(conf.sum(), 1):.2%} (was {(neg & e).sum() / neg.sum():.2%})")


if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == "analyse":
        analyse(n)
    elif len(sys.argv) > 3 and sys.argv[3] == "grid":
        grid(n)
    else:
        run(n, workers)
