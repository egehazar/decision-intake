"""Before D43: which further programmes can PolicyEngine-US compute from Populace, and how common are they?

Runs one chunk of the population through PolicyEngine (as adapters/benefits/build_table.py does, every take-up
flag set) and, for each candidate variable, reports whether it computes and the share of TRAINING households
with a non-zero amount, overall and by state. Held-out households are dropped before anything is summarised,
so no held-out label is looked at before the test is registered.

usage: python experiments/probe_new_programmes.py [CHUNK] [NCHUNKS]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapters", "benefits"))
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, HERE)

# (column, PolicyEngine variable, what it is)
CANDIDATES = [
    ("state_eitc", "state_eitc", "state earned income tax credit"),
    ("state_ctc", "state_ctc", "state child tax credit"),
    ("head_start", "head_start", "Head Start"),
    ("early_head_start", "early_head_start", "Early Head Start"),
    ("lifeline", "lifeline", "Lifeline phone or internet discount"),
    ("childcare_subsidy", "child_care_subsidies", "state childcare subsidy (CCDF)"),
    ("pell", "pell_grant", "Pell grant"),
    ("housing", "housing_assistance", "housing assistance (vouchers, public housing)"),
    ("liheap", "liheap", "home energy assistance (LIHEAP)"),
    ("csfp", "commodity_supplemental_food_program", "food boxes for older adults (CSFP)"),
]


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    os.chdir(ROOT)
    import build_table as BT
    from policyengine_us import Microsimulation

    t0 = time.time()
    ds, hh = BT.load_chunk(chunk, nchunks)
    sim = Microsimulation(dataset=ds, dataset_end_year=BT.YEAR)
    hid = np.asarray(sim.calculate("household_id", BT.YEAR).values)
    state = np.asarray(sim.calculate("state_code_str", BT.YEAR, map_to="household").values).astype(str)
    weight = np.asarray(sim.calculate("household_weight", BT.YEAR).values, dtype=float)

    # keep training households only (the split is by source household, as everywhere else)
    from common import splits
    pop, _, _, tr, va, test = splits()
    import pandas as pd
    table = pd.concat([pd.read_csv(f) for f in sorted(__import__("glob").glob("data/cache/chunks/c*.csv"))],
                      ignore_index=True)
    held = set(table["household_id"].to_numpy()[test].tolist())
    keep = np.array([h not in held for h in hid])

    out = {"chunk": chunk, "households": int(len(hid)), "training_households": int(keep.sum()), "programmes": {}}
    for col, var, what in CANDIDATES:
        try:
            v = np.asarray(sim.calculate(var, BT.YEAR, map_to="household").values, dtype=float)
        except Exception as e:  # noqa: BLE001 - report, don't stop
            out["programmes"][col] = {"variable": var, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            continue
        k, w = v[keep], weight[keep]
        nz = k > 0
        by_state = {}
        for s in np.unique(state[keep]):
            m = state[keep] == s
            by_state[s] = round(float(w[m & nz].sum() / max(w[m].sum(), 1e-9)), 3)
        out["programmes"][col] = {
            "variable": var, "what": what,
            "share_nonzero_weighted": round(float(w[nz].sum() / w.sum()), 4),
            "households_nonzero": int(nz.sum()),
            "median_if_nonzero": round(float(np.median(k[nz])), 0) if nz.any() else 0.0,
            "states_with_any": int(sum(1 for x in by_state.values() if x > 0)),
            "states_seen": len(by_state),
        }
    out["seconds"] = round(time.time() - t0)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "probe_new_programmes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
