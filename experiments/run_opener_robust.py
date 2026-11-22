"""SECONDARY: is the engine's edge when facts are volunteered first a property of the engine or of how
openers are simulated? Engine vs fair tree under different opener assumptions.

  open1         - one volunteered fact (from the 8 volunteerable ones)
  open2_natural - two facts, only from the most natural openers (state, household size, children,
                  65+, married, rent/own)
  open3_noisy   - three facts that may be misreported (per-question misreport model)

usage: python experiments/run_opener_robust.py N WORKERS
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
import run_final as RF  # noqa: E402
import run_secondary as RS  # noqa: E402
from decisionintake.simulate import UserModel, run_one  # noqa: E402

G = RF.G


def users():
    v = G["meta"]["volunteerable"]
    natural = [G["train"].qindex(q) for q in ["state", "household_size", "children", "senior", "married", "tenure"]]
    return {"open1": UserModel(opener=1, volunteerable=v, name="open1"),
            "open2_natural": UserModel(opener=2, volunteerable=natural, name="open2_natural"),
            "open3_noisy": UserModel(opener=3, volunteerable=v, misreport=True, name="open3_noisy")}


def work(job):
    kind, k, uname, rows = job
    user = users()[uname]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 17 + int(i))
        o = run_one(RS.make(kind, k), G["train"], G["loss"], G["test"].answers[i], user, rng, min_ess=RF.MIN_ESS,
                    outcome=G["outcome"])
        out.append({"i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist()})
    return (kind, k, uname), out


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    RS.init()
    order = np.random.default_rng(2026).choice(len(G["test"]), size=min(1200, len(G["test"])), replace=False)
    sample = order[:n]
    grid = [(kind, k, u) for u in ["open1", "open2_natural", "open3_noisy"] for kind in ["engine_k", "fair_tree_k"]
            for k in [3, 5, 9]]
    jobs = [(*g, ch) for g in grid for ch in np.array_split(sample, max(1, n // 30))]
    path = f"results/opener_robust_{n}.jsonl"
    t0 = time.time()
    with Pool(workers, initializer=RS.init) as pool, open(path, "w") as f:
        for d, (key, out) in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in out:
                f.write(json.dumps({"policy": key[0], "k": key[1], "user": key[2], **r}) + "\n")
            f.flush()
            if d % 20 == 0:
                print(f"{d}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    # paired comparison
    pop = G["test"]
    miss, fr = G["loss"].realised(pop)
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs.setdefault((r["policy"], r["k"], r["user"]), {})[r["i"]] = r
    idx = np.array(sorted(sample.tolist()))
    w = pop.weight[idx] / pop.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(2000, len(idx)))

    def li(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        return (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)

    print(f"engine minus fair tree, $/household/year, {len(idx)} test households [95% CI]:")
    for u in ["open1", "open2_natural", "open3_noisy"]:
        row = []
        for k in [3, 5, 9]:
            d = li(("engine_k", k, u)) - li(("fair_tree_k", k, u))
            bs = [np.average(d[b], weights=w[b]) for b in boots]
            row.append(f"k={k}: {w @ d:6.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")
        print(f"  {u:<14} " + "   ".join(row))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
