"""Spoken-style answers generated from a household's exact record, in controlled styles.

The record fixes the truth, so a parser's reading can be scored without hand labels. Styles:
  exact     - the figure, lightly rounded
  approx    - a round number within about 15% of the truth
  monthly   - an annual figure said per month (the parser has to convert)
  range     - a span around the truth that ignores our brackets
  vague     - words, not numbers ("not much", "pretty comfortable")
  indirect  - the fact is implied, not stated ("my dad lives with us, he's 72")
  dont_know - no information
This is a proxy for real people: templates are cleaner and less varied than speech, and the
write-up says so.
"""

from __future__ import annotations

import numpy as np

from .population import STATE_ABBR, STATES

STATE_NAMES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
               "the District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
               "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
               "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
               "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
               "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
               "Washington", "West Virginia", "Wisconsin", "Wyoming"]
NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
DONT_KNOW = ["I'm not sure.", "I'd have to check.", "No idea, honestly.", "I don't really know.", "Hard to say."]

# column in the table, whether the question is asked per year (else per month), vague words by level
MONEY = {
    "earned_income": ("earned_income", True),
    "social_security": ("social_security", True),
    "other_income": ("_other_income", True),
    "housing_cost": ("housing_cost", False),
    "childcare": ("childcare_expenses", True),
    "assets": ("assets", None),  # a stock, not a flow
}
VAGUE = {
    "earned_income": [(0, ["Nobody's working right now.", "Nothing from work at the moment."]),
                      (15_000, ["Not much, I only get a few shifts.", "Very little, it's part time."]),
                      (45_000, ["Enough to get by, barely.", "A normal paycheck, nothing special."]),
                      (100_000, ["We do okay.", "Decent money, both of us work."]),
                      (1e18, ["We're pretty comfortable.", "Good money, honestly."])],
    "social_security": [(0, ["None.", "We don't get any."]), (15_000, ["A small check."]),
                        (1e18, ["A decent amount, it's most of what we live on."])],
    "other_income": [(0, ["No, nothing else.", "Nope."]), (5_000, ["A little bit here and there."]),
                     (1e18, ["Yes, a fair amount from a pension."])],
    "housing_cost": [(0, ["We don't pay rent.", "Nothing, it's paid off."]), (800, ["It's pretty cheap."]),
                     (1_800, ["About what you'd expect around here."]), (1e18, ["A lot, it's most of our money."])],
    "childcare": [(0, ["We don't pay for childcare.", "None, family watches the kids."]),
                  (5_000, ["A little, just after-school care."]), (1e18, ["Daycare costs us a fortune."])],
    "assets": [(2_000, ["Almost nothing saved.", "Barely anything in the bank."]),
               (20_000, ["A little cushion, not much."]), (1e18, ["We have some savings and investments."])],
}


def _money(v: float) -> str:
    v = float(v)
    if v >= 10_000:
        return f"{v / 1000:,.0f} thousand" if v % 1000 < 50 else f"${v:,.0f}"
    return f"${v:,.0f}"


def _round_nice(x: float) -> float:
    if x <= 0:
        return 0.0
    mag = 10 ** max(int(np.floor(np.log10(x))) - 1, 0)
    step = mag * (5 if x / mag > 20 else 1)
    return float(round(x / step) * step)


def money_answer(qid: str, true_value: float, style: str, rng) -> str:
    _, per_year = MONEY[qid]
    v = max(float(true_value), 0.0)
    unit = "a year" if per_year else ("a month" if per_year is False else "")
    if style == "vague":
        for top, phrases in VAGUE[qid]:
            if v <= top:
                return str(rng.choice(phrases))
    if v == 0:
        return str(rng.choice(["None.", "Zero.", "Nothing.", "We don't have any of that."]))
    if style == "exact":
        return f"{_money(round(v, -2))} {unit}".strip() + "."
    if style == "approx":
        return f"About {_money(_round_nice(v * rng.uniform(0.85, 1.15)))} {unit}".strip() + "."
    if style == "monthly" and per_year:
        return f"Around {_money(_round_nice(v / 12 * rng.uniform(0.9, 1.1)))} a month."
    if style == "range":
        lo, hi = v * rng.uniform(0.7, 0.95), v * rng.uniform(1.05, 1.35)
        return f"Somewhere between {_money(_round_nice(lo))} and {_money(_round_nice(hi))} {unit}".strip() + "."
    return f"{_money(round(v, -2))} {unit}".strip() + "."


def count_answer(n: int, kind: str, style: str, rng) -> str:
    n = int(n)
    if kind == "children":
        if n == 0:
            return str(rng.choice(["No kids.", "None.", "No, no children."]))
        if style == "indirect":
            parts = rng.choice(["son", "daughter"], size=n)
            names = [f"my {p}" for p in parts]
            return ("I have " + ", ".join(names[:-1]) + (" and " if n > 1 else "") + names[-1] + ".").replace("have , ", "have ")
        return f"{NUMBER_WORDS[min(n, 10)].capitalize()} kid{'s' if n > 1 else ''}."
    # household size
    if style == "indirect" and n <= 6:
        others = {1: "Just me.", 2: "Me and my partner.", 3: "Me, my wife and our son.",
                  4: "Me, my husband and our two kids.", 5: "Five of us, two adults and three kids.",
                  6: "Me, my partner, our three kids and my mom."}
        return others[n]
    return f"{NUMBER_WORDS[min(n, 10)].capitalize()} of us." if n > 1 else "Just me."


def yesno_answer(qid: str, yes: bool, style: str, rng) -> str:
    if style == "indirect":
        bank = {
            "senior": ("My dad lives with us, he's 72.", "No, we're all in our thirties."),
            "disabled": ("My wife can't work because of her back.", "No, everyone's healthy."),
            "married": ("Yes, my husband and I.", "No, I'm single."),
            "child_under5": ("Our youngest just turned three.", "No, the kids are older."),
            "infant_or_pregnant": ("I'm expecting in March.", "No."),
            "self_employed": ("I drive for a delivery app.", "No, we both have regular jobs."),
            "employer_insurance": ("Yes, I'm on my company's plan.", "No, we buy our own or go without."),
            "college": ("My son's at the state university full time.", "No."),
        }
        if qid in bank:
            return bank[qid][0 if yes else 1]
    return str(rng.choice(["Yes.", "Yeah.", "Yes, that's right."] if yes else ["No.", "Nope.", "No, not us."]))


def answer(qid: str, row, opt: int, style: str, rng) -> str:
    """A spoken answer to question qid for a household (table row `row`, true option `opt`)."""
    if style == "dont_know":
        return str(rng.choice(DONT_KNOW))
    if qid == "state":
        i = STATES.index(int(row["state_fips"]))
        # (an abbreviation alone is ambiguous: "OK." reads as "okay")
        return f"We're in {STATE_NAMES[i]}." if style != "exact" else f"{STATE_NAMES[i]}."
    if qid == "tenure":
        return ["We rent an apartment.", "We own our house.", "I live with my sister, I don't pay rent."][opt]
    if qid in ("children", "household_size"):
        n = row["n_children"] if qid == "children" else row["household_size"]
        return count_answer(int(round(n)), qid, style, rng)
    if qid in MONEY:
        col, per_year = MONEY[qid]
        v = row[col] if col in row else 0.0
        if per_year is False:  # the table holds annual figures; this question asks per month
            v = v / 12
        return money_answer(qid, v, style, rng)
    return yesno_answer(qid, bool(opt), style, rng)


def styles_for(qid: str) -> list[str]:
    if qid in MONEY:
        s = ["exact", "approx", "range", "vague", "dont_know"]
        return s + ["monthly"] if MONEY[qid][1] else s
    if qid in ("children", "household_size"):
        return ["exact", "indirect"]
    if qid in ("state", "tenure"):
        return ["exact", "indirect"]
    return ["exact", "indirect", "dont_know"]
