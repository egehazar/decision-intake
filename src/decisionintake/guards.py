"""Rules the model may not break: statutory limits applied on top of the outcome model (DECISIONS D32).

The outcome network learned eligibility from PolicyEngine's results on real-looking households. Where
answers combine in ways few households show (a lone parent with no income at all, one earning $78,000),
it extrapolates, and v2 called the EITC likely for both. A rule is cheaper than more data there: when the
answers pin the household down to one tax unit and the statute rules a programme out for every household
that fits them, the probability is capped. Each rule names its source and is checked against
PolicyEngine's own results on the training households before it ships (experiments/guards_build.py).

A rules file is JSON, one rule per entry:

    {"id": "...", "decisions": ["eitc"], "cap": 0.02,
     "when": {"household_size": ["2"], "children": ["1"], "earned_income": ["$60,000 to $80,000", ...]},
     "why": "...", "source": "..."}

A record matches when, for every question in `when`, its answer is one of the listed options. The guard
applies record by record to the answers the model is asked about, so it acts inside the engine's belief
(which question is worth asking takes it into account), not only on the final list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "bundles" / "guards_2026.json"
CAP = 0.02  # the rules hold in PolicyEngine's data for all but a few households (see guards_build.py)

# 26 U.S.C. 32 as PolicyEngine-US 2.152.15 parameterises it for 2026 (gov.irs.credits.eitc): the credit
# phases out completely at start + max / rate, plus the joint-filer bonus for married couples.
EITC_2026 = {"max": [664, 4_427, 7_316, 8_231], "start": [10_860, 23_890, 23_890, 23_890],
             "rate": [0.0765, 0.1598, 0.2106, 0.2106], "joint_bonus": [7_280, 7_270, 7_270, 7_270]}
SOURCE = "26 U.S.C. 32 and 24(d); PolicyEngine-US 2.152.15 parameters gov.irs.credits (tax year 2026)"


def eitc_end(children: int, joint: bool) -> float:
    k = min(children, 3)
    end = EITC_2026["start"][k] + EITC_2026["max"][k] / EITC_2026["rate"][k]
    return end + (EITC_2026["joint_bonus"][k] if joint else 0)


def _band_low(option: str) -> float:
    """'$60,000 to $80,000' -> 60000; '$150,000 or more' -> 150000; 'none' -> 0."""
    if option == "none":
        return 0.0
    return float(option.split(" ")[0].replace("$", "").replace(",", ""))


def build_rules(questions) -> list[dict]:
    """The 2026 rule set for the benefits bundle's questions."""
    q = {x.id: x for x in questions}
    income = list(q["earned_income"].options)
    # one tax unit for certain: no college student (who could be a qualifying child or a separate filer), no
    # disability (an adult child with one can be a qualifying child), no self-employment (losses move earnings)
    sure = {"college": ["no"], "disabled": ["no"], "self_employed": ["no"]}
    units = []  # (filing, adults, children option, children count for the credit, household size option)
    for joint, adults in [(False, 1), (True, 2)]:
        for k, opt in enumerate(["none", "1", "2", "3"]):
            units.append((joint, adults, opt, k, str(adults + k)))
        units.append((joint, adults, "4 or more", 4, str(adults + 4)))  # household size pins the count at 4
    rules = []
    for joint, adults, kids_opt, k, hh in units:
        if hh not in q["household_size"].options:
            continue
        unit = {"household_size": [hh], "children": [kids_opt], "married": ["yes" if joint else "no"], **sure}
        who = f"{'a married couple' if joint else 'one adult'} with {'no' if k == 0 else k} child{'' if k == 1 else 'ren'}"
        end = eitc_end(k, joint)
        over = [o for o in income if _band_low(o) >= end]
        if over:
            rules.append({"id": f"eitc_income_{'joint' if joint else 'single'}_{k}",
                          "decisions": ["eitc"], "cap": CAP, "when": {**unit, "earned_income": over},
                          "why": f"For {who}, the EITC is zero once earnings reach ${end:,.0f} (2026).",
                          "source": SOURCE})
        rules.append({"id": f"no_earnings_{'joint' if joint else 'single'}_{k}",
                      "decisions": ["eitc", "ctc"] if k else ["eitc"], "cap": CAP,
                      "when": {**unit, "earned_income": ["none"]},
                      "why": f"For {who} and no earnings, there is no EITC (it needs earned income) and no "
                             f"refundable child tax credit (it needs earnings above $2,500).",
                      "source": SOURCE})
    return rules


class Rules:
    def __init__(self, rules: list[dict], questions, decisions):
        self.rules = rules
        qi = {x.id: j for j, x in enumerate(questions)}
        di = {d: j for j, d in enumerate(decisions)}
        self.compiled = []
        for r in rules:
            conds = []
            unknown = [k for k in r["when"] if k not in qi] + [d for d in r["decisions"] if d not in di]
            if unknown:
                raise ValueError(f"rule {r['id']}: this bundle has no {unknown[0]!r}")
            for qid, opts in r["when"].items():
                options = list(questions[qi[qid]].options)
                missing = [o for o in opts if o not in options]
                if missing:
                    raise ValueError(f"rule {r['id']}: {qid} has no option {missing[0]!r}")
                conds.append((qi[qid], np.array([options.index(o) for o in opts])))
            self.compiled.append((conds, [di[d] for d in r["decisions"]], float(r["cap"])))

    def matches(self, answers: np.ndarray) -> list[np.ndarray]:
        out = []
        for conds, _, _ in self.compiled:
            m = np.ones(len(answers), bool)
            for j, allowed in conds:
                m &= np.isin(answers[:, j], allowed)
            out.append(m)
        return out

    def apply(self, answers: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = p.copy()
        for (conds, ds, cap), m in zip(self.compiled, self.matches(answers)):
            if m.any():
                for d in ds:
                    p[m, d] = np.minimum(p[m, d], cap)
        return p

    @classmethod
    def load(cls, path, questions, decisions) -> "Rules":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["rules"], questions, decisions)


class GuardedOutcome:
    """An outcome model whose probabilities obey the rules. Same interface as the model it wraps."""

    _initial = None  # Session caches initial scores here; never fall through to the wrapped model's cache

    def __init__(self, model, rules: Rules):
        self.model, self.rules = model, rules

    def predict(self, pop, answers):
        p, v = self.model.predict(pop, answers)
        return self.rules.apply(np.asarray(answers), p), v

    def __getattr__(self, name):  # features, n_features, ... come from the wrapped model
        return getattr(self.model, name)


def attach(b, path=DEFAULT_RULES, strict: bool | None = None):
    """Wrap a bundle's model in its rules file, in place; returns the bundle. Records the rules in its info.
    The default rules file only applies to bundles it fits; a file named explicitly must fit (ValueError)."""
    strict = Path(path).resolve() != DEFAULT_RULES if strict is None else strict
    try:
        rules = Rules.load(path, b.pop.questions, b.pop.decisions)
    except ValueError:
        if strict:
            raise
        return b
    b.model = GuardedOutcome(b.model, rules)
    b.info = {**b.info, "rules": f"{len(rules.rules)} statutory limits ({Path(path).name}, DECISIONS D32)"}
    return b
