"""Numbers for the demo page's evidence section, computed from results/v2_<n>.jsonl (D28).

usage: python experiments/evidence_v2.py N OUT.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import run_v2 as R  # noqa: E402


def main():
    n, out = int(sys.argv[1]), sys.argv[2]
    R.init()
    idx = R.sample(n)
    runs = R.load(f"results/v2_{n}.jsonl")
    test = R.G["test"]
    miss, fr = R.G["loss"].realised(test)
    w = test.weight[idx] / test.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(4000, len(idx)))
    el = test.eligible[idx]

    def cols(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        q = np.array([runs[key][i]["q"] for i in idx], float)
        c = np.array([runs[key][i]["claimed"] for i in idx])
        p = np.array([runs[key][i]["p"] for i in idx])
        return rec, loss, q, c, p

    def ci(d):
        bs = np.array([np.average(d[b], weights=w[b]) for b in boots])
        return float(w @ d), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    res = {"households": int(n), "users": {}}
    for u in R.USERS:
        e_rec, e_loss, e_q, e_c, e_p = cols(("neural", "engine", u))
        tot_e = e_loss + R.QCOST * e_q
        rows = {}
        for m, k in R.POLICIES:
            rec, loss, q, c, p = cols((m, k, u))
            est, lo, hi = ci(tot_e - (loss + R.QCOST * q))
            lest, llo, lhi = ci(e_loss - loss)
            rows[f"{m}/{k}"] = {"questions": float(w @ q), "loss": float(w @ loss), "total": float(w @ (loss + R.QCOST * q)),
                                "found": float((rec & el).sum() / el.sum()), "wrong": float((rec & ~el).sum() / max(rec.sum(), 1)),
                                "diff_total": [est, lo, hi], "diff_loss": [lest, llo, lhi]}
        likely = e_rec & (e_p >= 0.5)
        order = np.argsort(-e_c)
        top = lambda f: float(e_loss[order[: int(round(f * len(idx)))]] @ w[order[: int(round(f * len(idx)))]] / (w @ e_loss))
        res["users"][u] = {"policies": rows,
                           "likely_precision": float((likely & el).sum() / max(likely.sum(), 1)),
                           "likely_recall": float((likely & el).sum() / el.sum()),
                           "claimed_over_actual": float((w @ e_c) / (w @ e_loss)),
                           "handoff_top10": top(0.1), "handoff_top20": top(0.2)}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({u: {k: v for k, v in r.items() if k != "policies"} for u, r in res["users"].items()}, indent=1))


if __name__ == "__main__":
    main()
