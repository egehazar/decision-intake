"""Why PolicyEngine contradicts a rule: every training household a candidate rule matches but where the
credit is paid, traced tax unit by tax unit through PolicyEngine-US (DECISIONS D32, GPT round 6).

The rules reason about the household from its answers: one adult (or a married couple) with the stated
children, household earnings in a band. PolicyEngine computes the credits per tax unit, from the filer's
own earned income under 26 U.S.C. 32(c)(2), which is not quite what the earnings question asks. Each
exception gets the first cause that explains it:

    dependants' wages    the household's earnings include a dependant's wages, which are not the filer's
    other earned income  the filer has earned income the question doesn't count (farm, partnership,
                         taxable disability pension, taxable dependent-care benefits, strike benefits)
    pre-tax deductions   the filer's taxable wages (W-2 box 1, what the credit uses) are below gross wages,
                         e.g. 401(k) contributions, which can bring earnings under the limit
    another tax unit     the credit goes to a second filer in the household
    unexplained          none of these

usage: PE_SHORT_PATH=... python experiments/contradiction_ledger.py      # writes results/contradiction_ledger.{json,md}
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "populace_us_2024.h5")
YEAR = 2026
OTHER_EARNED = ["taxable_dependent_care_benefits", "farm_operations_income", "partnership_self_employment_net_earnings",
                "earned_disability_benefits", "taxable_strike_benefits", "sstb_self_employment_income"]


def exceptions():
    """(rule, household_id, programme) for every matched training household where the programme is paid."""
    from common import splits
    from decisionintake import bundle
    from decisionintake.guards import Rules, build_rules
    from benefits.population import load_table

    table = load_table()
    pop, loss, meta, tr, va, test = splits()
    rest = np.sort(np.concatenate([tr, va]))
    b = bundle.load(os.path.join(ROOT, "bundles", "benefits_v2.npz"))
    assert np.array_equal(b.pop.answers, pop.answers[rest])
    rules = build_rules(b.pop.questions)
    R = Rules(rules, b.pop.questions, b.pop.decisions)
    shipped = {r["id"] for r in json.load(open(os.path.join(ROOT, "bundles", "guards_2026.json")))["rules"]}
    di = {d: j for j, d in enumerate(b.pop.decisions)}
    out, sizes = [], {}
    for r, m in zip(rules, R.matches(b.pop.answers)):
        sizes[r["id"]] = int(m.sum())
        for d in r["decisions"]:
            for k in np.flatnonzero(m & (b.pop.value[:, di[d]] > 0)):
                out.append((r["id"], int(table["household_id"].iloc[rest[k]]), d, r["id"] in shipped))
    return out, sizes, shipped


def trace(hh_ids):
    from policyengine_us import Microsimulation
    from policyengine_us.data.dataset_schema import USSingleYearDataset
    from benefits.build_table import ENTITIES, TAKEUP_TRUE

    frames = {e: pd.read_hdf(RAW, e) for e in ENTITIES}
    keep = frames["household"][frames["household"]["household_id"].isin(hh_ids)]
    person = frames["person"][frames["person"]["person_household_id"].isin(hh_ids)].copy()
    data = {"household": keep, "person": person}
    for ent in ["tax_unit", "spm_unit", "family", "marital_unit"]:
        df = frames[ent]
        data[ent] = df[df[f"{ent}_id"].isin(set(person[f"person_{ent}_id"].tolist()))].copy()
    for ent, cols in TAKEUP_TRUE.items():
        for c in cols:
            if c in data[ent].columns:
                data[ent][c] = True
    sim = Microsimulation(dataset=USSingleYearDataset(**data, time_period=2024), dataset_end_year=YEAR)
    calc = lambda v: np.asarray(sim.calculate(v, YEAR).values)  # noqa: E731
    p = pd.DataFrame({"household": calc("person_household_id"), "tax_unit": calc("person_tax_unit_id"),
                      "age": calc("age"), "dependent": calc("is_tax_unit_dependent").astype(bool),
                      "wages": calc("employment_income"), "self_emp": calc("self_employment_income"),
                      "taxable_wages": calc("irs_employment_income")})
    for v in OTHER_EARNED:
        try:
            p[v] = calc(v)
        except Exception:  # not every source exists in every version
            p[v] = 0.0
    t = pd.DataFrame({"tax_unit": calc("tax_unit_id"), "eitc": calc("eitc"), "ctc": calc("refundable_ctc"),
                      "eitc_earned": calc("eitc_earned_income"), "children": calc("eitc_child_count"),
                      "filing": calc("filing_status").astype(str)})
    return p, t


def classify(hid, programme, p, t):
    ph = p[p.household == hid]
    units = t[t.tax_unit.isin(ph.tax_unit.unique())]
    paid = units[units[programme] > 0]
    filers = ph[~ph.dependent]
    if (ph.dependent & ((ph.wages > 0) | (ph.self_emp > 0))).any():
        return "dependants' wages"
    if (filers[OTHER_EARNED].sum(axis=1) > 0).any():
        return "other earned income", filers[OTHER_EARNED].sum().idxmax()
    if ((filers.wages - filers.taxable_wages) > 1).any():
        return "pre-tax deductions"
    if units.tax_unit.nunique() > 1 and len(paid):
        return "another tax unit"
    return "unexplained"


def main():
    rows, sizes, shipped = exceptions()
    hh = sorted({h for _, h, _, _ in rows})
    print(f"{len(rows)} exceptions in {len(hh)} households; tracing through PolicyEngine-US {YEAR}", flush=True)
    p, t = trace(hh)
    ledger = defaultdict(Counter)
    sources = Counter()
    detail = []
    for rid, hid, d, ok in rows:
        cause = classify(hid, d, p, t)
        if isinstance(cause, tuple):
            cause, src = cause
            sources[src] += 1
        ledger[rid][cause] += 1
        detail.append({"rule": rid, "household_id": hid, "programme": d, "cause": cause})
    lines = ["# Contradiction ledger (D32)", "",
             "Every training household a candidate rule matches but where PolicyEngine-US pays the credit, traced by "
             "tax unit (experiments/contradiction_ledger.py).", "",
             "| rule | shipped | households matched | exceptions | dependants' wages | other earned income | "
             "pre-tax deductions | another tax unit | unexplained |", "|---|---|---|---|---|---|---|---|---|"]
    for rid in sorted(ledger, key=lambda r: (r not in shipped, r)):
        c = ledger[rid]
        cells = [c[k] for k in ("dependants' wages", "other earned income", "pre-tax deductions", "another tax unit",
                                "unexplained")]
        lines.append(f"| {rid} | {'yes' if rid in shipped else 'no'} | {sizes[rid]} | {sum(c.values())} | "
                     + " | ".join(str(x) for x in cells) + " |")
    lines += ["", "Other earned income, by largest source: " + ", ".join(f"{k} {v}" for k, v in sources.most_common())]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(ROOT, "results", "contradiction_ledger.md"), "w", encoding="utf-8").write(text + "\n")
    json.dump(detail, open(os.path.join(ROOT, "results", "contradiction_ledger.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
