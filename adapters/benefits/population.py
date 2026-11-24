"""Turn the PolicyEngine outcome table into a DecisionIntake Population.

Questions are things a person can answer in a sentence. Numbers are asked in ranges. Each
question has a report-noise model: what the engine assumes about how often people pick a
neighbouring range (income is the hardest to state exactly). Immigration status is never asked,
by design; it stays an unasked source of uncertainty.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from decisionintake import LossModel, Population, Question

PROGRAMMES = ["snap", "wic", "tanf", "ssi", "medicaid", "chip", "eitc", "ctc", "aca_ptc", "school_meals"]
LABELS = {
    "snap": "SNAP (food)", "wic": "WIC", "tanf": "TANF (cash aid)", "ssi": "SSI", "medicaid": "Medicaid",
    "chip": "CHIP", "eitc": "Earned income tax credit", "ctc": "Child tax credit (refundable part)",
    "aca_ptc": "ACA premium tax credit", "school_meals": "Free or reduced-price school meals",
    "state_eitc": "State earned income tax credit", "state_ctc": "State child tax credit",
    "head_start": "Head Start", "early_head_start": "Early Head Start", "housing": "Housing assistance",
}
ADDED_D43 = ["state_eitc", "state_ctc", "head_start", "early_head_start", "housing"]  # DECISIONS D43
MIN_VALUE = 120.0  # dollars a year: below $10 a month we don't count it as eligible

STATES = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]
STATE_ABBR = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
              "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
              "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


WILD = 0.01  # chance of an answer that's simply wrong, spread over all options


def adjacent_noise(b: int, p: float) -> np.ndarray:
    """Report one range off with probability p each side (renormalised at the ends), plus a small
    chance of any option at all, so no single answer can rule a record out entirely."""
    m = np.eye(b) * (1.0 - 2 * p)
    for a in range(b):
        if a > 0:
            m[a, a - 1] = p
        if a < b - 1:
            m[a, a + 1] = p
    m = m / m.sum(1, keepdims=True)
    return (1 - WILD) * m + WILD / b


def flip_noise(p: float) -> np.ndarray:
    return np.array([[1 - p, p], [p, 1 - p]])


def bucket(x, edges) -> np.ndarray:
    """edges are upper bounds (exclusive) of all but the last range."""
    return np.searchsorted(np.asarray(edges, float), np.asarray(x, float), side="right")


def ranges(edges, fmt=lambda v: f"${v:,.0f}") -> tuple[str, ...]:
    labs = ["none" if edges[0] <= 1 else f"under {fmt(edges[0])}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labs.append(f"{fmt(lo)} to {fmt(hi)}")
    labs.append(f"{fmt(edges[-1])} or more")
    return tuple(labs)


def load_table(pattern="data/cache/chunks/c*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    assert files, "build the table first: adapters/benefits/build_all.sh"
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_table_d43(base="data/cache/chunks/c*.csv", extra="data/cache/chunks_d43/c*.csv") -> pd.DataFrame:
    """The original table with D43's five programmes joined on household_id; row order is the original's,
    so every split and index stays the same."""
    t = load_table(base)
    x = load_table(extra)
    assert len(x) == len(t) and x["household_id"].is_unique, "the D43 table must cover every household once"
    out = t.merge(x, on="household_id", how="left", validate="one_to_one")
    assert (out["household_id"].to_numpy() == t["household_id"].to_numpy()).all()
    return out


def poverty_ratio(t: pd.DataFrame) -> np.ndarray:
    """Household income as a percentage of the federal poverty guideline for its size (48 states,
    2025 guideline uprated 2.5% to 2026: $16,040 plus $5,640 per extra person). Bracket edges sit at
    real programme cutoffs (SNAP 130%, Medicaid 138%, WIC and school meals 185%, ...)."""
    income = np.maximum(t["market_income"], t["earned_income"] + t["social_security"]).clip(lower=0)
    size = t["household_size"].clip(lower=1)
    return 100 * income / (16_040 + 5_640 * (size - 1))


FPL_EDGES = [1, 50, 100, 130, 138, 150, 185, 200, 250, 300, 400]


def build(table: pd.DataFrame, poverty_question: bool = False,
          programmes: list[str] = PROGRAMMES) -> tuple[Population, LossModel, dict]:
    t = table
    income_edges = [1, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 80_000, 100_000, 150_000]
    ss_edges = [1, 10_000, 20_000, 30_000]
    other = np.maximum(t["market_income"] - t["earned_income"] - t["social_security"], 0)
    other_edges = [1, 5_000, 15_000]
    housing_edges = [1, 500, 1_000, 1_500, 2_000, 3_000]  # monthly
    care_edges = [1, 3_000, 8_000]
    asset_edges = [2_000, 5_000, 20_000, 100_000]

    spec = [
        ("state", "Which state do you live in?", tuple(STATE_ABBR),
         np.searchsorted(STATES, t["state_fips"].astype(int)), None),
        ("household_size", "How many people live in your household, including you?", ("1", "2", "3", "4", "5", "6", "7 or more"),
         np.clip(t["household_size"].round().astype(int), 1, 7) - 1, adjacent_noise(7, 0.03)),
        ("children", "How many children under 18 live with you?", ("none", "1", "2", "3", "4 or more"),
         np.clip(t["n_children"].round().astype(int), 0, 4), adjacent_noise(5, 0.02)),
        ("child_under5", "Is any child under 5?", ("no", "yes"), (t["n_under5"] > 0).astype(int), flip_noise(0.02)),
        ("infant_or_pregnant", "Is anyone pregnant, or is there a baby under 1?", ("no", "yes"),
         (t["n_infant_or_pregnant"] > 0).astype(int), flip_noise(0.03)),
        ("senior", "Is anyone 65 or older?", ("no", "yes"), (t["n_65plus"] > 0).astype(int), flip_noise(0.01)),
        ("disabled", "Does anyone have a disability?", ("no", "yes"), (t["n_disabled"] > 0).astype(int), flip_noise(0.08)),
        ("married", "Is there a married couple in the household?", ("no", "yes"), (t["n_married"] > 0).astype(int), flip_noise(0.02)),
        ("earned_income", "Roughly how much does the household earn from work in a year, before tax?", ranges(income_edges),
         bucket(t["earned_income"].clip(lower=0), income_edges), adjacent_noise(11, 0.10)),
        ("self_employed", "Is anyone self-employed?", ("no", "yes"), (t["self_employed"] > 0).astype(int), flip_noise(0.03)),
        ("social_security", "How much Social Security does the household get in a year?", ranges(ss_edges),
         bucket(t["social_security"], ss_edges), adjacent_noise(5, 0.05)),
        ("other_income", "Any other income a year (pensions, unemployment, interest, child support)?", ranges(other_edges),
         bucket(other, other_edges), adjacent_noise(4, 0.10)),
        ("housing_cost", "What do you pay for rent or mortgage each month?", ranges(housing_edges),
         bucket(t["housing_cost"] / 12, housing_edges), adjacent_noise(7, 0.08)),
        ("tenure", "Do you rent, own, or neither?", ("rent", "own", "neither"),
         np.where(t["tenure_type"] == "RENTED", 0, np.where(t["tenure_type"].str.startswith("OWNED"), 1, 2)), None),
        ("childcare", "How much do you spend on childcare in a year?", ranges(care_edges),
         bucket(t["childcare_expenses"], care_edges), adjacent_noise(4, 0.08)),
        ("employer_insurance", "Does anyone get health insurance through a job?", ("no", "yes"),
         (t["n_esi"] > 0).astype(int), flip_noise(0.03)),
        ("assets", "Roughly how much is in bank accounts and investments?", ranges(asset_edges),
         bucket(t["assets"].clip(lower=0), asset_edges), adjacent_noise(5, 0.12)),
        ("college", "Is anyone a full-time college student?", ("no", "yes"), (t["n_college"] > 0).astype(int), flip_noise(0.02)),
    ]
    if poverty_question:
        spec.append(("income_vs_poverty_line", "Household income compared with the poverty line for its size",
                     tuple(["none"] + [f"{a}% to {b}%" for a, b in zip(FPL_EDGES[:-1], FPL_EDGES[1:])] + ["400% or more"]),
                     bucket(poverty_ratio(t), FPL_EDGES), adjacent_noise(len(FPL_EDGES) + 1, 0.10)))
    ordered = {"income_vs_poverty_line", "household_size", "children", "earned_income", "social_security", "other_income", "housing_cost",
               "childcare", "assets"}
    amounts = {"earned_income": (income_edges, "year"), "social_security": (ss_edges, "year"),
               "other_income": (other_edges, "year"), "housing_cost": (housing_edges, "month"),
               "childcare": (care_edges, "year"), "assets": (asset_edges, None)}
    questions = [Question(qid, text, opts, 1.0, noise, qid in ordered,
                          tuple(amounts[qid][0]) if qid in amounts else None, amounts[qid][1] if qid in amounts else None)
                 for qid, text, opts, _, noise in spec]
    answers = np.stack([np.asarray(col, dtype=int) for *_, col, _ in spec], axis=1)
    value = np.stack([t[f"v_{p}"].to_numpy(float) for p in programmes], axis=1)
    value = np.where(value >= MIN_VALUE, value, 0.0)
    eligible = value > 0
    pop = Population(questions, list(programmes), answers, t["household_weight"].to_numpy(float), eligible, value,
                     group=t["source_id"].to_numpy())
    loss = LossModel(false_rec_cost=np.full(len(programmes), 300.0), miss_floor=np.full(len(programmes), MIN_VALUE))
    meta = {
        "volunteerable": [pop.qindex(q) for q in ["state", "household_size", "children", "senior", "married",
                                                   "earned_income", "tenure", "disabled"]],
        "noncitizen": (t["n_noncitizen"] > 0).to_numpy(),
    }
    return pop, loss, meta


INTERACT = ["household_size", "children", "senior", "disabled", "married", "social_security", "assets"]


def _layout(pop: Population):
    sizes = {q.id: len(q.options) for q in pop.questions}
    off, pos = {}, 0
    for q in pop.questions:
        off[q.id] = pos
        pos += sizes[q.id]
    for k in INTERACT:
        off[("x", k)] = pos
        pos += sizes["earned_income"] * sizes[k]
    off[("x", "children_state")] = pos
    pos += sizes["children"] * sizes["state"]
    return off, sizes, pos + 1  # last column: intercept


def n_features(pop: Population) -> int:
    return _layout(pop)[2]


def features(pop: Population, answers: np.ndarray) -> np.ndarray:
    """Active indicator columns for the outcome model, one row per record: every answer one-hot,
    plus the interactions the rules actually use (income against household size, children, age,
    disability, marriage, Social Security, savings; children by state), plus an intercept."""
    off, sizes, F = _layout(pop)
    col = {q.id: answers[:, j] for j, q in enumerate(pop.questions)}
    parts = [off[q.id] + col[q.id] for q in pop.questions]
    for k in INTERACT:
        parts.append(off[("x", k)] + col["earned_income"] * sizes[k] + col[k])
    parts.append(off[("x", "children_state")] + col["children"] * sizes["state"] + col["state"])
    parts.append(np.full(len(answers), F - 1))
    return np.stack(parts, axis=1).astype(np.int64)
