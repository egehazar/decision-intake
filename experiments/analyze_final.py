"""Summarise results/final_<n>.jsonl: per policy and user, and paired bootstrap intervals.

usage: python experiments/analyze_final.py results/final_<n>.jsonl
"""

from __future__ import annotations

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


def main():
    path = sys.argv[1]
    pop, loss, meta, tr, va, test = splits()
    T = pop.subset(test)
    L = LossModel(np.full(len(pop.decisions), FR_COST), loss.miss_floor)
    miss, fr = L.realised(T)
    runs = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        runs[(r["policy"], r["k"], r["user"])][r["i"]] = r
    idx = np.array(sorted(next(iter(runs.values())).keys()))
    w = T.weight[idx] / T.weight[idx].sum()
    el = T.eligible[idx]
    noncit = meta["noncitizen"][test][idx]

    def per_household(key):
        rs = [runs[key][i] for i in idx]
        rec = np.array([r["rec"] for r in rs], bool)
        loss_i = (np.where(rec, 0.0, miss[idx]) + np.where(rec, fr[idx], 0.0)).sum(1)
        return np.array([r["q"] for r in rs], float), rec, loss_i, np.array([r["claimed"] for r in rs])

    rng = np.random.default_rng(0)
    boots = rng.choice(len(idx), size=(2000, len(idx)), replace=True)

    def ci(diff):
        """weighted mean of a per-household difference, with a paired bootstrap 95% interval"""
        m = float(w @ diff)
        bs = np.array([np.average(diff[b], weights=w[b]) for b in boots])
        return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print(f"test households scored: {len(idx)}; wrong-recommendation cost ${FR_COST:.0f}\n")
    print(f"{'user':<10}{'policy':<9}{'k':>4}{'questions':>10}{'loss':>8}{'claimed':>9}{'found':>7}{'$found':>8}"
          f"{'wrong':>7}{'loss, non-citizen hh':>22}")
    for key in sorted(runs, key=lambda k: (k[2], k[0], k[1])):
        q, rec, li, cl = per_household(key)
        val_el = np.where(el, T.value[idx], 0)
        print(f"{key[2]:<10}{key[0]:<9}{key[1]:>4}{w @ q:10.2f}{w @ li:8.0f}{w @ cl:9.0f}"
              f"{(w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum():7.3f}"
              f"{(w[:, None] * val_el * rec).sum() / (w[:, None] * val_el).sum():8.3f}"
              f"{(w[:, None] * (~el & rec)).sum() / max((w[:, None] * rec).sum(), 1e-12):7.3f}"
              f"{np.average(li[noncit], weights=w[noncit]) if noncit.any() else float('nan'):22.0f}")
    print("\nPaired loss differences, engine minus baseline, dollars per household per year (negative = engine better):")
    for user in ["clean", "opener", "realistic"]:
        for k in [3, 5, 7, 9, 12]:
            e = per_household(("engine_k", k, user))[2]
            parts = []
            for base in ["fixed_k", "tree_k"]:
                if ("" + base, k, user) in runs:
                    m, lo, hi = ci(e - per_household((base, k, user))[2])
                    parts.append(f"vs {base[:-2]:<5} {m:7.0f} [{lo:6.0f}, {hi:6.0f}]")
            print(f"  {user:<10} k={k:<3} " + "   ".join(parts))
        full = per_household(("ask_all", 0, user))[2]
        for lam in [50, 15, 5]:
            q, _, li, _ = per_household(("engine", lam, user))
            m, lo, hi = ci(li - full)
            print(f"  {user:<10} engine at cost ${lam:<3} asks {w @ q:4.1f}: vs asking everything {m:7.0f} [{lo:6.0f}, {hi:6.0f}]")

    print("\nHand-off: hand the households the engine is least sure about (highest expected loss at stop) to a person.")
    print("Share of all realised mistake cost those households account for (random would equal the share handed off):")
    for user in ["clean", "opener", "realistic"]:
        for lam in [15]:
            _, _, li, cl = per_household(("engine", lam, user))
            order = np.argsort(-cl)
            cw = np.cumsum(w[order]) / w.sum()
            cl_share = np.cumsum((w * li)[order]) / (w @ li)
            row = []
            for frac in [0.05, 0.10, 0.20]:
                k = int(np.searchsorted(cw, frac))
                row.append(f"top {int(frac * 100)}% -> {cl_share[k]:.2f}")
            corr = np.corrcoef(cl, li)[0, 1]
            print(f"  {user:<10} engine at cost ${lam}: " + "   ".join(row) + f"   (correlation claimed vs real {corr:.2f})")


if __name__ == "__main__":
    main()
