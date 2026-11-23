"""D32: the engine with the 2026 rules file against the engine without it, on fresh test households.

Same machinery as run_v2 (D28): same model, same belief population, same simulated answers per household
(paired draws). Fresh households: none used by D21-D28 or by D29-D31's 300 (seed 2034).

usage: python experiments/run_guard.py N WORKERS          # run (resumes)
       python experiments/run_guard.py N WORKERS analyse
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
import run_v2 as R  # noqa: E402
from decisionintake import VOIPolicy  # noqa: E402
from decisionintake.guards import DEFAULT_RULES, GuardedOutcome, Rules  # noqa: E402
from decisionintake.simulate import run_one  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "guard_2000.jsonl")
ARMS = ["v2", "v2_rules"]


def init():
    R.G.clear()
    R.init()
    train, model = R.G["neural"]
    rules = Rules.load(DEFAULT_RULES, train.questions, train.decisions)
    R.G["arms"] = {"v2": model, "v2_rules": GuardedOutcome(model, rules)}
    R.G["rules"] = rules


def sample(n):
    import agent_e2e as A

    N = len(R.G["test"])
    fresh = A.fresh_positions(N, R)
    dev = np.random.default_rng(2032).permutation(fresh)[:A.DEV_SIZE]
    rest = np.setdiff1d(fresh, dev)
    d29 = np.random.default_rng(2033).permutation(rest)[:200]
    left = np.setdiff1d(rest, d29)
    return np.sort(np.random.default_rng(2034).choice(left, size=n, replace=False))


def work(job):
    arm, uname, rows = job
    train = R.G["neural"][0]
    outcome = R.G["arms"][arm]
    pol = VOIPolicy(question_cost=R.QCOST)
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 34 + int(i))  # same draws for both arms: paired
        t0 = time.perf_counter()
        o = run_one(pol, train, R.G["loss"], R.G["test"].answers[i], R.users()[uname], rng, min_ess=R.MIN_ESS,
                    outcome=outcome)
        out.append({"arm": arm, "user": uname, "i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist(),
                    "p": np.round(o.p_final, 4).tolist(), "s": round(time.perf_counter() - t0, 3)})
    return out


def run(n, workers):
    init()
    idx = sample(n)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["arm"], r["user"], r["i"]))
    jobs = []
    for uname in ["clean", "realistic"]:
        for arm in ARMS:
            todo = [int(i) for i in idx if (arm, uname, int(i)) not in done]
            jobs += [(arm, uname, todo[k:k + 25]) for k in range(0, len(todo), 25)]
    print(f"{len(idx)} households, {sum(len(j[2]) for j in jobs)} interviews to run", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(OUT, "a") as f:
        for k, res in enumerate(pool.imap_unordered(work, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"  {k + 1}/{len(jobs)} batches, {time.time() - t0:.0f}s", flush=True)


def analyse(n):
    init()
    idx = sample(n)
    test = R.G["test"]
    miss, fr = R.G["loss"].realised(test)
    runs = {}
    for line in open(OUT):
        r = json.loads(line)
        runs.setdefault((r["arm"], r["user"]), {})[r["i"]] = r
    w = test.weight[idx] / test.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(4000, len(idx)))
    touched = np.zeros(len(idx), bool)
    for m in R.G["rules"].matches(test.answers[idx]):
        touched |= m
    lines = [f"D32: engine with the rules file minus engine without, {len(idx)} fresh households (seed 2034)",
             f"households whose own answers match a rule: {touched.sum()} ({touched.mean():.1%})"]
    out = {}
    for uname in ["clean", "realistic"]:
        cost = {}
        for arm in ARMS:
            rr = runs[(arm, uname)]
            rec = np.array([rr[i]["rec"] for i in idx], bool)
            q = np.array([rr[i]["q"] for i in idx], float)
            loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
            cost[arm] = (loss + R.QCOST * q, loss, q, rec)
        d = cost["v2_rules"][0] - cost["v2"][0]
        bs = np.array([np.average(d[b], weights=w[b]) for b in boots])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        du = np.array([d[b].mean() for b in boots])
        changed = (cost["v2_rules"][3] != cost["v2"][3]).any(1) | (cost["v2_rules"][2] != cost["v2"][2])
        lines.append(f"{uname}: weighted total cost v2 ${w @ cost['v2'][0]:,.0f}, with rules ${w @ cost['v2_rules'][0]:,.0f}; "
                     f"difference ${w @ d:+,.1f} [{lo:+,.1f}, {hi:+,.1f}] (non-inferior if the upper bound < +$25: "
                     f"{'PASS' if hi < 25 else 'FAIL'}); unweighted ${d.mean():+,.1f} [{np.percentile(du, 2.5):+,.1f}, "
                     f"{np.percentile(du, 97.5):+,.1f}]; households where anything changed: {changed.sum()}")
        out[uname] = {"diff": float(w @ d), "ci": [float(lo), float(hi)], "changed": int(changed.sum())}
    text = "\n".join(lines)
    print(text)
    open(OUT.replace(".jsonl", "_analysis.txt"), "w").write(text + "\n")
    json.dump(out, open(OUT.replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == "analyse":
        analyse(n)
    else:
        run(n, workers)
