"""Secondary results (labelled as such): fairer baselines, the completion ablation, per-programme
precision and recall, and precision against how many recommendations are shown.

usage: python experiments/analyze_secondary.py results/final_600.jsonl results/secondary_600.jsonl
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
from common import splits  # noqa: E402
from run_final import FR_COST  # noqa: E402

from decisionintake import LossModel  # noqa: E402


def load(path, runs):
    for line in open(path):
        r = json.loads(line)
        runs[(r["policy"], r["k"], r["user"])][r["i"]] = r


def main():
    primary, secondary = sys.argv[1], sys.argv[2]
    pop, loss, meta, tr, va, test = splits()
    T = pop.subset(test)
    L = LossModel(np.full(len(pop.decisions), FR_COST), loss.miss_floor)
    miss, fr = L.realised(T)
    runs = defaultdict(dict)
    load(primary, runs)
    load(secondary, runs)
    idx = np.array(sorted(runs[("engine_k", 5, "clean")].keys()))
    w = T.weight[idx] / T.weight[idx].sum()
    el = T.eligible[idx]
    boots = np.random.default_rng(1).choice(len(idx), size=(2000, len(idx)))

    def hh(key):
        rs = [runs[key][i] for i in idx]
        rec = np.array([r["rec"] for r in rs], bool)
        li = (np.where(rec, 0.0, miss[idx]) + np.where(rec, fr[idx], 0.0)).sum(1)
        p = np.array([r["p"] for r in rs]) if "p" in rs[0] else None
        return np.array([r["q"] for r in rs], float), rec, li, p

    def ci(d):
        bs = [np.average(d[b], weights=w[b]) for b in boots]
        return float(w @ d), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print("SECONDARY (added after the freeze). Paired loss difference, engine minus baseline, $/household/year:")
    for user in ["clean", "opener", "realistic"]:
        for k in [3, 5, 7, 9, 12]:
            e = hh(("engine_k", k, user))[2]
            row = []
            for base in ["fixed_set_k", "fair_tree_k", "fixed_k"]:
                if (base, k, user) in runs and len(runs[(base, k, user)]) >= len(idx):
                    m, lo, hi = ci(e - hh((base, k, user))[2])
                    row.append(f"vs {base:<11} {m:6.0f} [{lo:5.0f},{hi:5.0f}]")
            print(f"  {user:<10} k={k:<2} " + "  ".join(row))
    print("\nCompletion ablation (engine asking exactly k): loss with completion minus without:")
    for user in ["clean", "realistic"]:
        for k in [5, 9]:
            if ("engine_k_nocomp", k, user) in runs:
                m, lo, hi = ci(hh(("engine_k", k, user))[2] - hh(("engine_k_nocomp", k, user))[2])
                print(f"  {user:<10} k={k}: {m:6.0f} [{lo:5.0f}, {hi:5.0f}]  (negative = completion helps)")
    print("\nPer programme at the engine's operating point (engine asking 9, clean answers):")
    key = ("engine_k_p", 9, "clean")
    if key in runs:
        _, rec, _, p = hh(key)
        print(f"  {'programme':<13}{'eligible':>9}{'recall':>8}{'precision':>10}{'share of wrong recs':>21}")
        wrong_total = (w[:, None] * (~el & rec)).sum()
        for d, name in enumerate(pop.decisions):
            y, r = el[:, d], rec[:, d]
            rc = (w * (y & r)).sum() / max((w * y).sum(), 1e-12)
            pr = (w * (y & r)).sum() / max((w * r).sum(), 1e-12)
            print(f"  {name:<13}{(w * y).sum():9.3f}{rc:8.3f}{pr:10.3f}{(w * (~y & r)).sum() / wrong_total:21.3f}")
        print("\n  Show only recommendations the engine is at least t sure of (precision / programmes found):")
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            show = p >= t
            pr = (w[:, None] * (el & show)).sum() / max((w[:, None] * show).sum(), 1e-12)
            rc = (w[:, None] * (el & show)).sum() / (w[:, None] * el).sum()
            print(f"    t={t:.1f}: precision {pr:.3f}, programmes found {rc:.3f}, "
                  f"recommendations per household {(w[:, None] * show).sum() / w.sum():.2f}")


if __name__ == "__main__":
    main()
