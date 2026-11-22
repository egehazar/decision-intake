"""SECONDARY experiments on the same test households, added after GPT's second review (D17).
Not part of the frozen comparison; reported separately and labelled as such.

  fixed_set_k  - the best fixed set of k questions, chosen per k by beam search on training data
  fair_tree_k  - greedy tree on the same loss, deciding with the session's calibrated belief and
                 completion (only question choice differs from the engine), rare-branch backoff
  engine_k_nocomp - engine without completion (ablation: does completion help under misreports?)
Each run also saves the final probability of eligibility per programme, for precision-recall curves.

usage: python experiments/run_secondary.py N WORKERS
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
import run_final as RF  # noqa: E402
from decisionintake import FixedOrder, GreedyTree, VOIPolicy  # noqa: E402
from decisionintake.simulate import run_one  # noqa: E402

G = RF.G
KS = RF.KS


def init():
    RF.init()
    G["sets"] = {}
    G["ftree"] = {}


def make(kind, k):
    if kind == "fixed_set_k":
        if k not in G["sets"]:
            G["sets"][k] = FixedOrder.fit_subset(G["train"], G["loss"], k)
        return G["sets"][k]
    if kind == "fair_tree_k":
        if k not in G["ftree"]:
            G["ftree"][k] = GreedyTree(train=G["train"], loss=G["loss"], budget=k, min_leaf=30, question_cost=0.0,
                                       lookahead=False, decide_with_session=True, backoff=True)
        return G["ftree"][k]
    return VOIPolicy(question_cost=0.0, budget=k)  # engine_k / engine_k_nocomp / engine_k_p


def work(job):
    kind, k, uname, rows = job
    user = RF.users()[uname]
    out = []
    for i in rows:
        rng = np.random.default_rng(1_000_003 * 13 + int(i))  # same draws as the primary run
        o = run_one(make(kind, k), G["train"], G["loss"], G["test"].answers[i], user, rng, min_ess=RF.MIN_ESS,
                    outcome=None if kind == "engine_k_nocomp" else G["outcome"])
        out.append({"i": int(i), "q": o.n_questions, "rec": o.recommend.astype(int).tolist(),
                    "claimed": round(o.risk_at_stop, 2), "p": np.round(o.p_final, 4).tolist()})
    return (kind, k, uname), out


def main():
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    init()
    order = np.random.default_rng(2026).choice(len(G["test"]), size=min(1200, len(G["test"])), replace=False)
    sample = order[:n]
    grid = []
    for u in ["clean", "realistic"]:
        grid += [("fixed_set_k", k, u) for k in KS] + [("fair_tree_k", k, u) for k in KS]
        grid += [("engine_k_nocomp", k, u) for k in [5, 9]] + [("engine_k_p", k, u) for k in [5, 9]]
    path = f"results/secondary_{n}.jsonl"
    jobs = [(*g, ch) for g in grid for ch in np.array_split(sample, max(1, n // 30))]
    t0 = time.time()
    with Pool(workers, initializer=init) as pool, open(path, "w") as f:
        for d, (key, out) in enumerate(pool.imap_unordered(work, jobs), 1):
            for r in out:
                f.write(json.dumps({"policy": key[0], "k": key[1], "user": key[2], **r}) + "\n")
            f.flush()
            if d % 20 == 0:
                print(f"{d}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
