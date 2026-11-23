"""Compute one chunk of the benefits outcome table with PolicyEngine-US.

For each household in the chunk: the facts the intake can ask about, and per programme the
annual dollars it is entitled to under 2026 rules with every take-up flag set to "takes it up"
(screening is about entitlement, not about who claimed). Chunks keep all copies of a source
household together. Run each chunk in its own process: PolicyEngine holds a lot of memory.

usage: python build_table.py CHUNK NCHUNKS OUT.csv
"""

import sys
import time

import numpy as np
import pandas as pd

RAW = "data/raw/populace_us_2024.h5"
YEAR = 2026
ENTITIES = ["person", "household", "tax_unit", "spm_unit", "family", "marital_unit"]
SELF_EMPLOYMENT = ["self_employment_income", "sstb_self_employment_income", "partnership_self_employment_net_earnings",
                   "farm_operations_income"]  # every self-employment source the EITC counts (D34)
TAKEUP_TRUE = {
    "spm_unit": ["takes_up_snap_if_eligible", "takes_up_tanf_if_eligible", "takes_up_housing_assistance_if_eligible"],
    "person": ["takes_up_ssi_if_eligible", "takes_up_medicaid_if_eligible", "would_claim_wic", "takes_up_head_start_if_eligible"],
    "tax_unit": ["takes_up_eitc", "takes_up_aca_if_eligible", "would_file_taxes_voluntarily"],
}


def load_chunk(chunk: int, nchunks: int):
    from policyengine_us.data.dataset_schema import USSingleYearDataset

    frames = {e: pd.read_hdf(RAW, e) for e in ENTITIES}
    hh = frames["household"]
    src = hh["household_source_id"].astype("int64").to_numpy()
    keep_hh = hh[(src * 2654435761 % 2**32) % nchunks == chunk]
    ids = set(keep_hh["household_id"].tolist())
    person = frames["person"]
    person = person[person["person_household_id"].isin(ids)]
    out = {"household": keep_hh, "person": person.copy()}
    for ent in ["tax_unit", "spm_unit", "family", "marital_unit"]:
        key = f"person_{ent}_id"
        df = frames[ent]
        out[ent] = df[df[f"{ent}_id"].isin(set(person[key].tolist()))].copy()
    for ent, cols in TAKEUP_TRUE.items():
        for c in cols:
            if c in out[ent].columns:
                out[ent][c] = True
    return USSingleYearDataset(**out, time_period=2024), keep_hh


def main():
    chunk, nchunks, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    ds, hh = load_chunk(chunk, nchunks)
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=ds, dataset_end_year=YEAR)

    def hh_sum(var):
        return np.asarray(sim.calculate(var, YEAR, map_to="household").values, dtype=float)

    def hh_any(values_person):
        # sum a person-level array to households
        return np.asarray(sim.map_result(values_person, "person", "household"), dtype=float)

    def person(var):
        return np.asarray(sim.calculate(var, YEAR).values)

    age = person("age")
    self_emp = sum(person(v).astype(float) for v in SELF_EMPLOYMENT)
    res = pd.DataFrame(
        {
            "household_id": np.asarray(sim.calculate("household_id", YEAR).values),
            "household_weight": hh_sum("household_weight"),
            "state_fips": np.asarray(sim.calculate("state_fips", YEAR).values),
            "household_size": hh_sum("household_size"),
            "n_children": hh_any((age < 18).astype(float)),
            "n_under5": hh_any((age < 5).astype(float)),
            "n_infant_or_pregnant": hh_any(((age < 1) | person("is_pregnant").astype(bool)).astype(float)),
            "n_65plus": hh_any((age >= 65).astype(float)),
            "n_disabled": hh_any(person("is_disabled").astype(float)),
            "n_married": hh_any(np.asarray(sim.calculate("is_married", YEAR, map_to="person").values).astype(float)),
            # Earnings from work as a person would report them: wages plus every kind of self-employment
            # PolicyEngine keeps separately (service businesses, partnerships, farms). Before D34 only
            # self_employment_income counted, so some self-employed households looked like non-earners.
            "earned_income": hh_sum("employment_income") + hh_any(self_emp),
            "self_employed": hh_any((self_emp != 0).astype(float)),
            "social_security": hh_sum("social_security"),
            "market_income": hh_sum("household_market_income"),
            "housing_cost": hh_sum("housing_cost"),
            "childcare_expenses": hh_sum("childcare_expenses"),
            "n_esi": hh_any(person("has_esi").astype(float)),
            "assets": hh_any(person("bank_account_assets").astype(float) + person("stock_assets").astype(float)
                             + person("bond_assets").astype(float)),
            "n_college": hh_any(person("is_full_time_college_student").astype(float)),
            "n_noncitizen": hh_any((person("immigration_status_str") != "CITIZEN").astype(float))
            if "immigration_status_str" in ds.person.columns
            else 0.0,
        }
    )
    hh_src = hh.set_index("household_id")
    res["source_id"] = hh_src.loc[res["household_id"], "household_source_id"].to_numpy()
    res["tenure_type"] = hh_src.loc[res["household_id"], "tenure_type"].astype(str).to_numpy()

    # programme outcomes: annual dollars of entitlement
    res["v_snap"] = hh_sum("snap_if_takes_up")
    res["v_wic"] = hh_sum("wic_if_takes_up")
    res["v_tanf"] = hh_sum("tanf_if_takes_up")
    res["v_ssi"] = hh_sum("ssi_if_takes_up")
    # Medicaid: PolicyEngine's medicaid_cost_if_enrolled divides state spending by a total over the
    # simulated sample, so it is meaningless for a subsample. Use its per-capita cost by eligibility
    # group (MACPAC figures) instead, which doesn't depend on who else is in the sample.
    med_el = person("is_medicaid_eligible").astype(bool)
    per_capita = sim.tax_benefit_system.parameters(f"{YEAR}-01-01").calibration.gov.hhs.medicaid.totals.per_capita
    groups = np.asarray(sim.calculate("medicaid_group", YEAR).values).astype(str)
    cost = np.array([float(getattr(per_capita, g, 0.0)) if g != "NONE" else 0.0 for g in groups])
    res["v_medicaid"] = hh_any(med_el * cost)
    res["n_medicaid_eligible"] = hh_any(med_el.astype(float))
    chip_el = person("is_chip_eligible").astype(bool)
    res["v_chip"] = hh_any(chip_el * person("per_capita_chip").astype(float))
    res["n_chip_eligible"] = hh_any(chip_el.astype(float))
    res["v_eitc"] = hh_sum("eitc")
    res["v_ctc"] = hh_sum("refundable_ctc")
    res["v_aca_ptc"] = hh_sum("aca_ptc")
    res["v_school_meals"] = hh_sum("school_meal_net_subsidy")
    res.to_csv(out_path, index=False)
    print(f"chunk {chunk}/{nchunks}: {len(res)} households, {len(ds.person)} people, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
