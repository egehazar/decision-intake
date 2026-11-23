"""D35, the release test: the production candidate against the release, on the last unused test households.

    candidate  v2 (bundles/benefits_v2.npz) + the D32 rules (bundles/guards_2026_d32.json)
    release    v2.1, retrained on D34's corrected earnings answers (bundles/benefits_v2_1.npz) + the rules
               re-checked on that data (bundles/guards_2026.json) + D33's abstention: when fewer than 10
               training households fit the answers, a programme not recommended but at 5% or more is
               "can't tell", not "doesn't look likely"

Both answer from the same test households in the corrected coding (what a person would report), with the
same simulated answers (paired draws). Households: every test household no earlier experiment used.

usage: python experiments/run_release.py WORKERS            # run (resumes)
       python experiments/run_release.py WORKERS analyse
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
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from decisionintake import VOIPolicy, bundle  # noqa: E402
from decisionintake.guards import GuardedOutcome, Rules  # noqa: E402
from decisionintake.simulate import UserModel, run_one  # noqa: E402

OUT = os.path.join(ROOT, "results", "release_d35.jsonl")
ARMS = {"candidate": ("benefits_v2.npz", "guards_2026_d32.json"), "release": ("benefits_v2_1.npz", "guards_2026_v2_1.json")}  # renamed after D35
QCOST, MIN_ESS, ABSTAIN_BELOW, UNSURE_FLOOR = 15.0, 10.0, 10.0, 0.05
G = {}


def init():
    import agent_e2e as A
    import run_v2 as R
    from common import splits

    pop, loss, meta, tr, va, test = splits()
    G["test"], G["meta"] = pop.subset(test), meta
    R.G["test"] = G["test"]
    fresh = A.fresh_positions(len(test), R)
    dev = np.random.default_rng(2032).permutation(fresh)[:A.DEV_SIZE]
    rest = np.setdiff1d(fresh, dev)
    d29 = np.random.default_rng(2033).permutation(rest)[:200]
    left = np.setdiff1d(rest, d29)
    d32 = np.random.default_rng(2034).choice(left, size=2000, replace=False)
    G["idx"] = np.sort(np.setdiff1d(left, d32))
    G["arms"] = {}
    for arm, (bf, rf) in ARMS.items():
        b = bundle.load(os.path.join(ROOT, "bundles", bf))
        rules = Rules.load(os.path.join(ROOT, "bundles", rf), b.pop.questions, b.pop.decisions)
        G["arms"][arm] = (b, GuardedOutcome(b.model, rules))
        assert [q.options for q in b.pop.questions] == [q.options for q in G["test"].questions]


def users():
    v = G["meta"]["volunteerable"]
    return {"clean": UserModel(name="clean"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.10, name="realistic")}


def work(job):
    arm, uname, rows = job
    b, outcome = G["arms"][arm]
    pol = VOIPolicy(question_cost=QCOST)
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 35 + int(i))  # same draws for both arms: paired
        o = run_one(pol, b.pop, b.loss, G["test"].answers[i], users()[uname], rng, min_ess=MIN_ESS, outcome=outcome)
        out.append({"arm": arm, "user": uname, "i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist(),
                    "p": np.round(o.p_final, 4).tolist(), "raw_ess": round(o.support["raw_ess"], 3)})
    return out


def run(workers):
    init()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["arm"], r["user"], r["i"]))
    jobs = []
    for uname in ["clean", "realistic"]:
        for arm in ARMS:
            todo = [int(i) for i in G["idx"] if (arm, uname, int(i)) not in done]
            jobs += [(arm, uname, todo[k:k + 25]) for k in range(0, len(todo), 25)]
    print(f"{len(G['idx'])} households, {sum(len(j[2]) for j in jobs)} interviews to run", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(OUT, "a") as f:
        for k, res in enumerate(pool.imap_unordered(work, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"  {k + 1}/{len(jobs)} batches, {time.time() - t0:.0f}s", flush=True)


def analyse(_):
    init()
    idx, test = G["idx"], G["test"]
    loss_model = G["arms"]["candidate"][0].loss
    miss, fr = loss_model.realised(test)
    runs = {}
    for line in open(OUT):
        r = json.loads(line)
        runs[(r["arm"], r["user"], r["i"])] = r
    w = test.weight[idx] / test.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(4000, len(idx)))
    el = test.eligible[idx]
    names = list(test.decisions)
    lines = [f"D35 release test: {len(idx)} unused test households; release minus candidate"]
    summary = {}
    for uname in ["clean", "realistic"]:
        A, B = ({k: runs[(arm, uname, int(i))] for i, k in zip(idx, range(len(idx)))} for arm in ARMS)
        rec = {arm: np.array([R_[k]["rec"] for k in range(len(idx))], bool) for arm, R_ in zip(ARMS, (A, B))}
        q = {arm: np.array([R_[k]["q"] for k in range(len(idx))], float) for arm, R_ in zip(ARMS, (A, B))}
        cost = {arm: (np.where(rec[arm], 0, miss[idx]) + np.where(rec[arm], fr[idx], 0)).sum(1) + QCOST * q[arm] for arm in ARMS}
        d = cost["release"] - cost["candidate"]
        bs = np.array([np.average(d[b], weights=w[b]) for b in boots])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # safety: how often a negative is wrong
        pB = np.array([B[k]["p"] for k in range(len(idx))])
        rawB = np.array([B[k]["raw_ess"] for k in range(len(idx))])
        negA = ~rec["candidate"]
        negB = ~rec["release"]
        flag = negB & (rawB < ABSTAIN_BELOW)[:, None] & (pB >= UNSURE_FLOOR)
        confB = negB & ~flag
        errA = (negA & el).sum(1), negA.sum(1)
        errB = (confB & el).sum(1), confB.sum(1)
        rateA, rateB = errA[0].sum() / errA[1].sum(), errB[0].sum() / errB[1].sum()
        bd = np.array([errB[0][b].sum() / errB[1][b].sum() - errA[0][b].sum() / errA[1][b].sum() for b in boots])
        dlo, dhi = np.percentile(bd, [2.5, 97.5])
        flags = flag.sum(1).mean()
        thin = rawB < ABSTAIN_BELOW
        wt = (negB & el)[thin].sum() / max(negB[thin].sum(), 1)
        ws = (negB & el)[~thin].sum() / max(negB[~thin].sum(), 1)
        caught = (flag & el).sum(), (negB & el).sum()
        lines += [f"\n{uname} answers",
                  f"  total cost a household (survey-weighted): candidate ${w @ cost['candidate']:,.0f}, release "
                  f"${w @ cost['release']:,.0f}; difference ${w @ d:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"
                  + (f"  -> non-inferior (bound +$25): {'PASS' if hi < 25 else 'FAIL'}" if uname == "clean" else ""),
                  f"  questions {np.average(q['candidate'], weights=w):.1f} -> {np.average(q['release'], weights=w):.1f}",
                  f"  a 'doesn't look likely' is wrong: candidate {rateA:.2%}, release {rateB:.2%}; difference "
                  f"{rateB - rateA:+.2%} [{dlo:+.2%}, {dhi:+.2%}]"
                  + (f"  -> at most half and CI below 0: {'PASS' if rateB <= rateA / 2 and dhi < 0 else 'FAIL'}" if uname == "clean" else ""),
                  f"  'can't tell' labels a household: {flags:.2f}" + (f"  -> at most 1.25: {'PASS' if flags <= 1.25 else 'FAIL'}" if uname == "clean" else ""),
                  f"  misses the release labels 'can't tell': {caught[0]} of {caught[1]}; thin support {thin.mean():.1%} of "
                  f"households; a negative is wrong {wt:.2%} where thin, {ws:.2%} where supported"]
        per = []
        for dd, n in enumerate(names):
            a = (negA[:, dd] & el[:, dd]).sum() / max(negA[:, dd].sum(), 1)
            b_ = (confB[:, dd] & el[:, dd]).sum() / max(confB[:, dd].sum(), 1)
            per.append(f"{n} {a:.1%}->{b_:.1%}")
        lines.append("  per programme, negative wrong: " + ", ".join(per))
        summary[uname] = {"cost_diff": float(w @ d), "cost_ci": [float(lo), float(hi)], "neg_wrong": [float(rateA), float(rateB)],
                          "neg_diff_ci": [float(dlo), float(dhi)], "flags_per_household": float(flags)}
    text = "\n".join(lines)
    print(text)
    open(OUT.replace(".jsonl", "_analysis.txt"), "w").write(text + "\n")
    json.dump(summary, open(OUT.replace(".jsonl", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    workers = int(sys.argv[1])
    (analyse if len(sys.argv) > 2 and sys.argv[2] == "analyse" else run)(workers)
