"""The rules file: every rule reads against the shipped questions, the cases that prompted it are fixed, and
nothing a rule doesn't cover moves."""

from pathlib import Path

import numpy as np
import pytest

from decisionintake import bundle
from decisionintake.guards import DEFAULT_RULES, GuardedOutcome, Rules, attach, build_rules, eitc_end
from decisionintake.service import Intake

SHIPPED = Path(__file__).resolve().parents[1] / "bundles" / "benefits_v2.npz"
pytestmark = pytest.mark.skipif(not (SHIPPED.exists() and DEFAULT_RULES.exists()), reason="shipped files not present")

LONE_PARENT = [
    {"question": "state", "choice": "TX"}, {"question": "household_size", "choice": "2"},
    {"question": "children", "choice": "1"}, {"question": "child_under5", "choice": "yes"},
    {"question": "married", "choice": "no"}, {"question": "college", "choice": "no"},
    {"question": "disabled", "choice": "no"}, {"question": "self_employed", "choice": "no"},
    {"question": "senior", "choice": "no"}, {"question": "social_security", "choice": "none"},
    {"question": "other_income", "choice": "none"}, {"question": "tenure", "choice": "rent"},
    {"question": "housing_cost", "amount": {"value": 800, "period": "month"}},
]


@pytest.fixture(scope="module")
def b():
    return bundle.load(str(SHIPPED))


def eitc(b, guarded, earnings):
    bb = bundle.load(str(SHIPPED))
    if guarded:
        attach(bb)
    it = Intake(bb)
    it.record(LONE_PARENT + [{"question": "earned_income", "amount": {"value": earnings, "period": "year"}}])
    return next(r for r in it.assessment()["all_programmes"] if r["id"] == "eitc")


def test_eitc_limits_match_the_2026_parameters():
    assert round(eitc_end(1, False)) == 51_593 and round(eitc_end(0, False)) == 19_540
    assert round(eitc_end(3, True)) == 70_244 and eitc_end(5, False) == eitc_end(3, False)


def test_every_shipped_rule_reads_and_was_checked(b):
    import json
    spec = json.loads(DEFAULT_RULES.read_text(encoding="utf-8"))
    rules = Rules(spec["rules"], b.pop.questions, b.pop.decisions)
    assert len(rules.rules) == 14
    for r in spec["rules"]:
        assert r["source"] and r["why"] and all(v["weighted_share"] <= r["cap"] for v in r["checked"].values())
    assert {r["id"] for r in spec["rules"]} <= {r["id"] for r in build_rules(b.pop.questions)}


def test_the_cases_that_prompted_the_rules(b):
    for earnings in (0, 78_000):
        before, after = eitc(b, False, earnings), eitc(b, True, earnings)
        assert before["tier"] in ("likely", "worth checking"), before  # what v2 alone said
        assert after["tier"] in ("unlikely", "not recommended") and after["probability"] <= 0.05, after
    assert eitc(b, True, 18_000)["tier"] == "likely"  # well inside the credit: untouched


def test_rules_only_lower_what_they_cover(b):
    rules = Rules.load(DEFAULT_RULES, b.pop.questions, b.pop.decisions)
    A = b.pop.answers[:5000]
    p0, v0 = b.model.predict(b.pop, A)
    p1, v1 = GuardedOutcome(b.model, rules).predict(b.pop, A)
    covered = np.zeros_like(p0, bool)
    for (conds, ds, cap), m in zip(rules.compiled, rules.matches(A)):
        for d in ds:
            covered[m, d] = True
    assert np.array_equal(p0[~covered], p1[~covered]) and np.array_equal(v0, v1)
    assert (p1[covered] <= 0.02 + 1e-12).all() and covered.any()
