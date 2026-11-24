"""D43: the five added programmes' annual entitlement for one chunk, in a table of their own.

Same chunks, households and take-up flags as build_table.py (it reuses load_chunk), so rows join on
household_id; the original table is not touched.

usage: python build_table_d43.py CHUNK NCHUNKS OUT.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_table as BT  # noqa: E402

NEW = {  # column -> PolicyEngine-US variable (D43)
    "v_state_eitc": "state_eitc",
    "v_state_ctc": "state_ctc",
    "v_head_start": "head_start",
    "v_early_head_start": "early_head_start",
    "v_housing": "housing_assistance",
}


def main():
    chunk, nchunks, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    ds, _ = BT.load_chunk(chunk, nchunks)
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=ds, dataset_end_year=BT.YEAR)
    res = pd.DataFrame({"household_id": np.asarray(sim.calculate("household_id", BT.YEAR).values)})
    for col, var in NEW.items():
        res[col] = np.asarray(sim.calculate(var, BT.YEAR, map_to="household").values, dtype=float)
    res.to_csv(out_path, index=False)
    print(f"chunk {chunk}/{nchunks}: {len(res)} households, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
