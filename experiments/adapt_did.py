"""POST HOC (not pre-registered) reading of D25: how much does the engine's edge over the tree CHANGE
when conditions change, relative to the plain intake (S0)? Same households, so the difference of
differences is paired. Negative = the engine gains more from the change than the tree does.

usage: python experiments/adapt_did.py 600
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))
import run_adapt as A  # noqa: E402

G = A.G


def main():
    n = int(sys.argv[1])
    A.init()
    idx = A.sample(n)
    pop = G["test"]
    miss, fr = G["loss"].realised(pop)
    runs = {}
    for line in open(f"results/adapt_{n}.jsonl"):
        r = json.loads(line)
        runs.setdefault((r["policy"], r["scenario"]), {})[r["i"]] = r

    def total(key):
        rec = np.array([runs[key][i]["rec"] for i in idx], bool)
        loss = (np.where(rec, 0, miss[idx]) + np.where(rec, fr[idx], 0)).sum(1)
        return loss + np.array([runs[key][i]["paid"] for i in idx])

    w = pop.weight[idx] / pop.weight[idx].sum()
    boots = np.random.default_rng(5).choice(len(idx), size=(2000, len(idx)))
    gap0 = total(("engine", "S0")) - total(("tree", "S0"))
    print("change in (engine minus tree) relative to the plain intake, $/household/year [95% CI]")
    for sc, other in [("S1", "tree"), ("S2", "tree"), ("S3", "tree"), ("S3", "tree_regrown"), ("S4", "tree"),
                      ("S4", "tree_regrown")]:
        d = (total(("engine", sc)) - total((other, sc))) - gap0
        bs = [np.average(d[b], weights=w[b]) for b in boots]
        print(f"  {sc} vs {other:<13} {w @ d:6.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")
    print("cost of the change itself (scenario minus plain), $/household/year [95% CI]")
    for kind in ["engine", "tree"]:
        for sc in ["S1", "S2", "S3", "S4"]:
            d = total((kind, sc)) - total((kind, "S0"))
            bs = [np.average(d[b], weights=w[b]) for b in boots]
            print(f"  {kind:<7}{sc}  {w @ d:6.0f} [{np.percentile(bs, 2.5):6.0f}, {np.percentile(bs, 97.5):6.0f}]")


if __name__ == "__main__":
    main()
