"""DecisionIntake v2 confirmation run, pre-registered in DECISIONS D28.

v2 = the same engine with the wide-and-deep outcome model (settings frozen on validation). It runs on
fresh test households that no earlier run touched (seed 2029, excluding the 2,100 households used by
D21-D27). Every policy sees the same households and the same draws (paired). Total cost per household =
decision loss + $15 per question asked.

usage: python experiments/run_v2.py N WORKERS [analyse]
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
from decisionintake import AskAll, FixedOrder, GreedyTree, LossModel, VOIPolicy  # noqa: E402
from decisionintake.simulate import UserModel, run_one  # noqa: E402

FR, MIN_ESS, QCOST = 1000.0, 10.0, 15.0
NEURAL = dict(hidden=(512, 256), dropout=0.4, wd=0.3)  # frozen in D28
FIXED_K = 8
POLICIES = [("neural", "engine"), ("neural", "ask_all"), ("neural", "deep_tree"), ("neural", "tree"),
            ("neural", "fixed8"), ("linear", "engine"), ("linear", "ask_all")]
USERS = ["clean", "realistic"]
G = {}


def init():
    from common import smoothed, smoothed_neural, splits

    pop, loss, meta, tr, va, test = splits()
    rest = np.sort(np.concatenate([tr, va]))
    G["loss"] = LossModel(np.full(len(pop.decisions), FR), loss.miss_floor)
    G["test"], G["meta"] = pop.subset(test), meta
    G["linear"] = smoothed(pop, rest, "rest")
    G["neural"] = smoothed_neural(pop, rest, "rest", **NEURAL)
    G["pol"] = {}


def policy(model, kind):
    key = (model, kind)
    if key in G["pol"]:
        return G["pol"][key]
    train = G[model][0]
    if kind == "engine":
        p = VOIPolicy(question_cost=QCOST)
    elif kind == "ask_all":
        p = AskAll()
    elif kind == "fixed8":
        p = FixedOrder.fit_subset(train, G["loss"], FIXED_K)
    else:  # the D25 tree and the strongest tree of D27, both deciding with the same model as the engine
        deep = kind == "deep_tree"
        p = GreedyTree(train=train, loss=G["loss"], budget=18, min_leaf=100 if deep else 30, question_cost=QCOST,
                       lookahead=True, decide_with_session=True, backoff=True, unknown_backoff=True,
                       deep_backoff=deep, name=kind)
    G["pol"][key] = p
    return p


def users():
    v = G["meta"]["volunteerable"]
    return {"clean": UserModel(name="clean"),
            "realistic": UserModel(opener=2, volunteerable=v, misreport=True, dont_know=0.10, name="realistic")}


def work(job):
    model, kind, uname, rows = job
    train, outcome = G[model]
    pol = policy(model, kind)
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 29 + int(i))  # same draws for every policy: paired
        t0 = time.perf_counter()
        o = run_one(pol, train, G["loss"], G["test"].answers[i], users()[uname], rng, min_ess=MIN_ESS, outcome=outcome)
        out.append({"model": model, "policy": kind, "user": uname, "i": int(i), "q": o.n_questions,
                    "rec": o.recommend.astype(int).tolist(), "claimed": round(o.risk_at_stop, 2),
                    "p": np.round(o.p_final, 4).tolist(), "s": round(time.perf_counter() - t0, 3)})
    return out


def sample(n):
    """Fresh test households: none used by D21-D27 (first 1,200 of seed 2026, 600 of seed 2027, 300 of seed 2028)."""
    N = len(G["test"])
    used = set(np.random.default_rng(2026).choice(N, size=min(1200, N), replace=False).tolist())
    fresh = np.array([i for i in range(N) if i not in used])
    d25 = set(np.random.default_rng(2027).choice(fresh, size=600, replace=False).tolist())
    used |= d25
    fresh = np.array([i for i in range(N) if i not in used])
    d27 = set(np.random.default_rng(2028).choice(fresh, size=300, replace=False).tolist())
    used |= d27
    fresh = np.array([i for i in range(N) if i not in used])
    return np.sort(np.random.default_rng(2029).choice(fresh, size=n, replace=False))


def load(path):
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs.setdefault((r["model"], r["policy"], r["user"]), {})[r["i"]] = r
    return runs


def analyse(path, idx):
    test = G["test"]
    miss, fr = G["loss"].realised(test)
    runs = load(path)
    want = set(idx.tolist())
    runs = {k: v for k, v in runs.items() if want <= set(v)}  # only policies finished on every household
    w = test.weight[idx] / test.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(4000, len(idx)))
    el = test.eligible[idx]

    def cols(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        q = np.array([runs[key][i]["q"] for i in idx], float)
        claimed = np.array([runs[key][i]["claimed"] for i in idx])
        return rec, loss, q, claimed

    def ci(d):
        bs = np.array([np.average(d[b], weights=w[b]) for b in boots])
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        return w @ d, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p

    # robustness (GPT round 4): resample whole source households, since 82 of them give two test records
    _, cl = np.unique(test.group[idx], return_inverse=True)
    cboots = np.random.default_rng(6).integers(0, cl.max() + 1, size=(4000, cl.max() + 1))

    def ci_cluster(d):
        wd, ww = np.bincount(cl, weights=w * d), np.bincount(cl, weights=w)
        bs = wd[cboots].sum(1) / ww[cboots].sum(1)
        return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"{len(idx)} fresh test households (seed 2029), survey-weighted $/household/year; "
          f"total = decision loss + ${QCOST:.0f} per question")
    print(f"{'user':<10}{'model':<8}{'policy':<10}{'questions':>10}{'loss':>8}{'total':>8}{'claimed':>9}"
          f"{'found':>7}{'wrong recs':>11}{'s/interview':>12}")
    for u in USERS:
        for m, k in POLICIES:
            if (m, k, u) not in runs:
                continue
            rec, loss, q, c = cols((m, k, u))
            found = (rec & el).sum() / max(el.sum(), 1)
            wrong = (rec & ~el).sum() / max(rec.sum(), 1)
            sec = np.mean([runs[(m, k, u)][i]["s"] for i in idx])
            print(f"{u:<10}{m:<8}{k:<10}{w @ q:10.2f}{w @ loss:8.0f}{w @ (loss + QCOST * q):8.0f}{w @ c:9.0f}"
                  f"{100 * found:6.0f}%{100 * wrong:10.0f}%{sec:12.2f}")
    primaries = [(("neural", "engine"), ("linear", "engine"), "P1 v2 engine vs v1 engine"),
                 (("neural", "engine"), ("linear", "ask_all"), "P2 v2 engine vs v1 full questionnaire"),
                 (("neural", "engine"), ("neural", "deep_tree"), "P3 v2 engine vs strongest tree"),
                 (("neural", "engine"), ("neural", "fixed8"), "P4 v2 engine vs best fixed 8-question form")]
    secondary = [(("neural", "engine"), ("neural", "ask_all"), "S1 v2 engine vs v2 full questionnaire"),
                 (("neural", "engine"), ("neural", "tree"), "S2 v2 engine vs D25 tree"),
                 (("neural", "ask_all"), ("linear", "ask_all"), "S3 v2 vs v1 model, all 18 answers")]
    print("\nDescriptive (v2 engine): verdict tiers, self-assessment, hand-off")
    for u in USERS:
        for m in ["neural", "linear"]:
            key = (m, "engine", u)
            if key not in runs:
                continue
            rec, loss, q, c = cols(key)
            p = np.array([runs[key][i]["p"] for i in idx])
            likely = rec & (p >= 0.5)
            check = rec & (p < 0.5)
            prec = lambda mask: (mask & el).sum() / max(mask.sum(), 1)
            ratio = (w @ c) / (w @ loss)
            order = np.argsort(-c)
            share = lambda f: loss[order[: int(round(f * len(idx)))]] @ w[order[: int(round(f * len(idx)))]] / (w @ loss)
            print(f"  {u:<10}{m:<8} 'likely' tier: {likely.sum()} recs, {100 * prec(likely):.0f}% eligible, "
                  f"finds {100 * (likely & el).sum() / max(el.sum(), 1):.0f}% of eligible programmes | 'worth checking': "
                  f"{check.sum()} recs, {100 * prec(check):.0f}% eligible | claimed/actual loss {ratio:.2f} | "
                  f"top 10% least sure carry {100 * share(0.1):.0f}% of mistake cost, top 20% {100 * share(0.2):.0f}%")
    for u in USERS:
        print(f"\n{u}: difference in total cost [95% CI] (negative = first is cheaper), Holm over P1-P4")
        res = []
        for a, b, name in primaries + secondary:
            if (*a, u) not in runs or (*b, u) not in runs:
                continue
            _, la, qa, _ = cols((*a, u))
            _, lb, qb, _ = cols((*b, u))
            est, lo, hi, p = ci((la + QCOST * qa) - (lb + QCOST * qb))
            dl, llo, lhi, _ = ci(la - lb)
            clo, chi = ci_cluster((la + QCOST * qa) - (lb + QCOST * qb))
            res.append((name, est, lo, hi, p, dl, llo, lhi, clo, chi))
        prim = sorted([r for r in res if r[0].startswith("P")], key=lambda r: r[4])
        holm = {}
        for rank, r in enumerate(prim):
            holm[r[0]] = min(1.0, max([holm.get(x[0], 0) for x in prim[:rank]] + [r[4] * (len(prim) - rank)]))
        for name, est, lo, hi, p, dl, llo, lhi, clo, chi in res:
            tag = f"Holm p {holm[name]:.3f}" if name in holm else f"p {p:.3f}"
            print(f"  {name:<42} total {est:6.0f} [{lo:6.0f}, {hi:6.0f}]  {tag:<14} loss only {dl:6.0f} [{llo:6.0f}, {lhi:6.0f}]"
                  f"  by source household [{clo:6.0f}, {chi:6.0f}]")


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    init()
    idx = sample(n)
    path = f"results/v2_{n}.jsonl"
    if len(sys.argv) > 3:
        analyse(path, idx)
        return
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["model"], r["policy"], r["user"], r["i"]))
    # primary comparisons (clean answers) first, then the secondary ones, so the primaries finish early
    primary = [("neural", "engine"), ("linear", "engine"), ("linear", "ask_all"), ("neural", "deep_tree"), ("neural", "fixed8")]
    order = [(m, k, "clean") for m, k in primary] + [(m, k, "clean") for m, k in POLICIES if (m, k) not in primary]
    order += [(m, k, "realistic") for m, k in POLICIES]
    jobs = []
    for m, k, u in order:
        rows = [i for i in idx.tolist() if (m, k, u, i) not in done]
        jobs += [(m, k, u, np.array(ch)) for ch in np.array_split(rows, max(1, len(rows) // 20)) if len(ch)]
    print(f"{len(done)} interviews already done, {sum(len(j[3]) for j in jobs)} to run", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(path, "a") as f:
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
