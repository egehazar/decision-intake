"""One person's intake behind a typed interface, for an AI assistant to drive.

The assistant does the language: it asks the engine's question in its own words and turns whatever the
person says, including facts nobody asked for, into typed facts. The engine does the decisions: which
question is worth asking, how far each answer moves the belief, when to stop, and whether the doubt
left over is costly enough to hand to a person. Amount arithmetic stays in code (a monthly figure
becomes a yearly one, "about" widens the range), never in the language model.

A fact is {"question": id} plus one of:
    "choice":    the option's text, e.g. "yes", "3", "rent", "CA" (numbers map onto "4 or more")
    "amount":    {"value": n} or {"low": n, "high": n}, with "period" ("year", "month", "week", "none")
                 and "approximate" (true for "about", "around", a range); amount questions only
    "dont_know": true
A fact for a question that already has one replaces it (a correction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .bundle import Bundle
from .core import Session
from .language import PERIOD_PER_YEAR, StructuredReader
from .policies import VOIPolicy
from .state import decode, encode

YES = {"yes", "y", "true", "yeah", "yep"}
NO = {"no", "n", "false", "nope", "none"}


@dataclass
class Settings:
    question_cost: float = 15.0  # dollars a year a question must be expected to save before it's asked
    min_ess: float = 10.0  # support floor (see Session)
    handoff_cost: float = 2500.0  # expected mistakes left at the end above this go to a person
    max_questions: int = 18
    # Abstention (D33): below this many training households like the person, programmes that aren't
    # recommended but keep at least `unsure_floor` probability are "can't tell", not "unlikely". None = off.
    abstain_below: float | None = None
    unsure_floor: float = 0.05  # chosen on development data (DECISIONS D33)


class FactError(ValueError):
    pass


EXPLAIN_NOTE = ("Each line changes one answer and keeps the others. It shows how close each result is to the line, "
                "so the person knows to check again if that part of their situation changes. It is not advice on "
                "what to report: answer as things are.")
_GROUP = {"likely": "likely", "worth checking": "worth checking", "can't tell": "can't tell",
          "not recommended": "not recommended", "unlikely": "not recommended"}
CLIFF_NOTE = ("Each bar keeps every other answer and changes only this one. Expected value is the chance of "
              "qualifying times the typical yearly amount if eligible (health coverage counted at its value, not "
              "cash). A screening estimate for planning, not a promise of any amount.")
CLIFF_GROUPS = {"food and cash": ["snap", "wic", "tanf", "ssi", "school_meals"],
                "tax credits": ["eitc", "ctc", "state_eitc", "state_ctc"],
                "health coverage": ["medicaid", "chip", "aca_ptc"],
                "care and housing": ["head_start", "early_head_start", "housing"]}  # the last three: D43
_COUNT = re.compile(r"none|\d+|\d+ or more|under \d+")

# ------------------------------------------------------------------------------------ the application plan (D45)
# The order to apply in. Step 1 is what pays every month or covers health (and housing, whose waiting lists
# are long); step 2 is what an approval from step 1 makes easier; step 3 waits for the tax return.
PLAN_STEPS = [
    ("now", "Apply now", ["snap", "tanf", "ssi", "medicaid", "chip", "aca_ptc", "housing"]),
    ("next", "Then", ["wic", "school_meals", "head_start", "early_head_start"]),
    ("tax", "At tax time: the 2026 tax return, filed in early 2027", ["eitc", "ctc", "state_eitc", "state_ctc"]),
]
# Statutory links: getting a source programme settles part of the target's eligibility. Shown only when a
# source is also recommended. (target, sources, kind, English text with {src}, citation)
PLAN_LINKS = [
    ("wic", ["snap", "medicaid", "tanf"], "income_met",
     "Getting {src} counts as meeting WIC's income limit, so bring the approval letter.", "7 CFR 246.7(d)(2)(vi)"),
    ("school_meals", ["snap", "tanf"], "free_meals",
     "Children in a household getting {src} qualify for free school meals, and the school is often told "
     "directly, with no form to fill in.", "7 CFR 245.2 and 245.6a"),
    ("head_start", ["tanf", "ssi"], "income_met",
     "A family getting {src} meets Head Start's income rule.", "45 CFR 1302.12(c)(1)(ii) and 1305.2"),
    ("early_head_start", ["tanf", "ssi"], "income_met",
     "A family getting {src} meets Early Head Start's income rule.", "45 CFR 1302.12(c)(1)(ii) and 1305.2"),
    ("medicaid", ["ssi"], "comes_with",
     "In most states, getting SSI comes with Medicaid.", "42 U.S.C. 1396a(a)(10)(A)(i)(II)"),
]
PLAN_NOTES = {  # (kind, English text); shown when one of the programmes is in the step
    "one_health_application": (["medicaid", "chip", "aca_ptc"],
                               "One application covers Medicaid, CHIP and the Marketplace tax credit: through your "
                               "state or HealthCare.gov."),
    "waiting_list": (["housing"], "Waiting lists are often long, so put your name down now, with more than one "
                                  "housing agency if you can."),
    "one_return": (["eitc", "ctc", "state_eitc", "state_ctc"],
                   "The federal credits and the state credits are claimed on your federal and state returns. "
                   "Free filing help: IRS VITA sites."),
}
PLAN_NOTE = ("The order to apply in, for the programmes worth applying for now. The links between programmes are "
             "federal rules; a state or programme office decides each application.")
# A programme the household already gets can settle another's income test even where the answers alone don't
# point to it; the target is then added as "worth checking" if its category can fit (who it is for).
# programme -> [(question, the canonical choices that rule the category out)]
_YOUNG = [("children", {"none"}), ("child_under5", {"no"})]
PLAN_CATEGORY = {"wic": _YOUNG, "head_start": _YOUNG, "early_head_start": _YOUNG,
                 "school_meals": [("children", {"none"})]}
ALREADY_SOURCES = ["snap", "medicaid", "tanf", "ssi"]
_SHORT = {"snap": "SNAP", "tanf": "TANF", "ssi": "SSI", "medicaid": "Medicaid"}


def _either(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _dollars(q, x) -> float:
    """A stated amount as a number of dollars: finite, not negative, not absurd."""
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise FactError(f"amount for {q.id} must be a number of dollars, like 2400")
    try:
        v = float(str(x).replace(",", "").replace("$", "").strip()) if isinstance(x, str) else float(x)
    except ValueError:
        raise FactError(f"amount for {q.id} must be a number of dollars, like 2400 (got {x!r})") from None
    if not np.isfinite(v) or v < 0:
        raise FactError(f"amount for {q.id} must be zero or more")
    if v > 1e9:
        raise FactError(f"amount for {q.id} is implausibly large ({v:,.0f})")
    return v


class Intake:
    def __init__(self, bundle: Bundle, settings: Settings | None = None):
        self.b, self.cfg = bundle, settings or Settings()
        self.qi = {q.id: j for j, q in enumerate(bundle.pop.questions)}
        self.policy = VOIPolicy(question_cost=self.cfg.question_cost, budget=self.cfg.max_questions)
        self.facts: dict[str, dict] = {}  # question id -> {"evidence", "reading"}, in the order given
        self.raw: dict[str, dict] = {}  # question id -> the fact as the state stores it (see state.py)
        self.history: list[dict] = []
        self.s = self._session()

    def _session(self) -> Session:
        return Session(self.b.pop, self.b.loss, min_ess=self.cfg.min_ess, outcome=self.b.model)

    # ------------------------------------------------------------------------------- reading facts
    def _choice(self, q, value) -> int:
        opts = [o.lower() for o in q.options]
        if isinstance(value, bool):
            value = "yes" if value else "no"
        text = str(value).strip().lower().rstrip(".")
        if text in opts:
            return opts.index(text)
        if opts == ["no", "yes"] and (text in YES or text in NO):
            return 1 if text in YES else 0
        try:
            n = float(text.replace(",", "").replace("$", ""))
        except ValueError:
            n = None
        if n is not None and q.edges is None:  # counts: "3" -> "3", 0 -> "none", 9 -> "7 or more"
            if n == 0 and "none" in opts:
                return opts.index("none")
            if float(n).is_integer() and str(int(n)) in opts:
                return opts.index(str(int(n)))
            for k, o in enumerate(opts):
                m = re.fullmatch(r"(\d+)(?: or more|\+)", o)
                if m and n >= int(m.group(1)):
                    return k
        raise FactError(f'"{value}" is not an option for {q.id}; use one of: {", ".join(q.options)}'
                        + (' (or give "amount")' if q.edges is not None else ""))

    def _amount(self, q, a: dict) -> tuple[np.ndarray, str]:
        if q.edges is None:
            raise FactError(f'{q.id} is not an amount question; give "choice", one of: {", ".join(q.options)}')
        if not isinstance(a, dict):
            a = {"value": a}
        low = a.get("low", a.get("value"))
        high = a.get("high", a.get("value"))
        if low is None and high is None:
            raise FactError(f'amount for {q.id} needs "value" or "low"/"high"')
        low = _dollars(q, low if low is not None else high)
        high = _dollars(q, high if high is not None else low)
        if low > high:  # "between 30 and 20 thousand": the person means the range
            low, high = high, low
        period = str(a.get("period") or "none").lower()
        if period not in PERIOD_PER_YEAR and period != "none":
            raise FactError(f'period must be one of year, month, week, none (got "{period}")')
        slots = {"low": low, "high": high, "period": period, "approximate": bool(a.get("approximate")),
                 "zero": low == 0 and high == 0}
        ev = StructuredReader().evidence(q, "", raw={"slots": slots})
        # say what the code did with it, in the question's own unit
        factor = 1.0 if q.per is None else PERIOD_PER_YEAR.get(period, PERIOD_PER_YEAR[q.per]) / PERIOD_PER_YEAR[q.per]
        lo, hi = low * factor, high * factor
        if slots["approximate"]:
            lo, hi = lo * 0.85, hi * 1.15
        unit = {"year": " a year", "month": " a month", None: ""}[q.per]
        stated = (_money(low) if low == high else f"{_money(low)}-{_money(high)}") + ("" if period == "none" else f" a {period}")
        conv = _money(lo) + unit if lo == hi else f"{_money(lo)}-{_money(hi)}{unit}"
        top = np.argsort(-ev)[:2]
        spread = ", ".join(f'"{q.options[k]}" {ev[k] / ev.sum():.0%}' for k in top if ev[k] / ev.sum() >= 0.1)
        canon = {"question": q.id, "amount": {"low": low, "high": high, "period": period,
                                              "approximate": slots["approximate"]}}
        if factor == 1.0 and not slots["approximate"]:
            return ev, f"{stated} -> {spread}", canon
        return ev, f"{stated} -> {conv} -> {spread}", canon

    def _evidence(self, q, f: dict) -> tuple[np.ndarray, str, dict]:
        """(likelihood over the options, how it was read, the fact as the state stores it)"""
        kinds = [k for k in ("choice", "amount") if f.get(k) is not None] + (["dont_know"] if f.get("dont_know") else [])
        if len(kinds) > 1:
            raise FactError(f'give exactly one of "choice", "amount" or "dont_know" (got {", ".join(kinds)})')
        if f.get("dont_know"):
            return np.ones(len(q.options)), "don't know (no evidence; the belief is unchanged)",                 {"question": q.id, "dont_know": True}
        if f.get("amount") is not None:
            return self._amount(q, f["amount"])
        if f.get("choice") is not None:
            k = self._choice(q, f["choice"])
            ev = np.zeros(len(q.options))
            ev[k] = 1.0
            return ev, q.options[k], {"question": q.id, "choice": q.options[k]}
        raise FactError('give one of "choice", "amount" or "dont_know"')

    def record(self, facts: list[dict]) -> list[dict]:
        """Record typed facts; returns one reading per fact (or an error saying how to fix it).
        Valid facts go in even when others in the same call are rejected."""
        out, new, replaced = [], [], False
        for f in facts:
            qid = f.get("question") if isinstance(f, dict) else None
            if not isinstance(qid, str) or qid not in self.qi:
                out.append({"question": qid, "error": f"unknown question; use an id from: {', '.join(self.qi)}"})
                continue
            try:
                ev, reading, canon = self._evidence(self.b.pop.questions[self.qi[qid]], f)
            except FactError as e:
                out.append({"question": qid, "error": str(e)})
                continue
            q = self.b.pop.questions[self.qi[qid]]
            r = {"question": qid, "reading": reading}
            if qid in self.facts:
                replaced = True
                r["replaces"] = self.facts.pop(qid)["reading"]
                self.raw.pop(qid)
            self.facts[qid] = {"evidence": ev, "reading": reading}
            self.raw[qid] = canon
            new.append(qid)
            out.append(r)
        before = self.s.risk()
        # Several facts at once (a replayed state, a correction, a volunteered batch): weigh each in turn,
        # re-score the households through the outcome model once at the end.
        if replaced:  # a correction: replay every fact on a fresh belief
            self.s = self._session()
            for qid, f in self.facts.items():
                self.s.observe(self.qi[qid], f["evidence"], source="fact", rescore=False)
        else:
            for qid in new:
                self.s.observe(self.qi[qid], self.facts[qid]["evidence"], source="fact", rescore=False)
        if new:
            self.s.rescore()
        conflicts = {e["q"] for e in self.s.log if e.get("conflict")}
        for r in out:
            if "reading" in r and r["question"] in conflicts:
                r["warning"] = "no household on record fits this together with the earlier answers; check it with the person"
        if new:
            self.history.append({"facts": new, "cost_before": round(before), "cost_after": round(self.s.risk())})
        return out

    # ------------------------------------------------------------------------------- state
    def state(self) -> str:
        """Everything this intake knows, as a short string (state.py); from_state rebuilds it exactly."""
        return encode(list(self.raw.values()))

    @classmethod
    def from_state(cls, bundle: Bundle, token: str, settings: Settings | None = None) -> "Intake":
        return cls.resume(bundle, token, [], settings)[0]

    @classmethod
    def resume(cls, bundle: Bundle, token: str, new: list, settings: Settings | None = None) -> tuple["Intake", list]:
        """Rebuild from a state and record new facts in one batch, so the households are re-scored once, not
        twice (the stateless service's hot path). Returns the intake and the readings of the new facts."""
        it = cls(bundle, settings)
        old = decode(bundle.pop.questions, token) if token else []
        readings = it.record(old + list(new)) if (old or new) else []
        bad = [r for r in readings[:len(old)] if "error" in r]
        if bad:  # only possible if the bundle changed under a state written for another one
            raise FactError(f"state no longer reads: {bad[0]['error']}")
        return it, readings[len(old):]

    # ------------------------------------------------------------------------------- what next
    def question_view(self, j: int) -> dict:
        q = self.b.pop.questions[j]
        v = {"id": q.id, "text": q.text}
        if q.edges is not None:
            unit = {"year": "dollars a year", "month": "dollars a month", None: "dollars in total"}[q.per]
            v.update(kind="amount", unit=unit, ranges=list(q.options))
        else:
            v.update(kind="choice", options=list(q.options))
        return v

    def catalogue(self) -> list[dict]:
        return [self.question_view(j) for j in range(len(self.b.pop.questions))]

    def next_step(self) -> dict:
        s = self.s
        risk = s.risk()
        progress = {"facts_recorded": len(self.facts), "expected_mistake_cost": round(risk),
                    "similar_households": round(s.ess()), "support": self.support()}
        j = self.policy.next_question(s, len(self.facts))
        if j is not None:
            single = dict(s.voi_table())
            v = single.get(j, 0.0)
            by = np.maximum(s.voi_by_decision(j), 0)
            share = by / by.sum() if by.sum() > 0 else by
            settles = [{"id": self.b.pop.decisions[d], "programme": self.b.labels[self.b.pop.decisions[d]],
                        "share": round(float(share[d]), 2)}
                       for d in np.argsort(-share)[:3] if share[d] >= 0.1]
            why = {"worth_per_year": round(v), "question_cost": self.cfg.question_cost, "mostly_settles": settles}
            if v <= self.cfg.question_cost:
                why["note"] = "worth asking only together with another question; no single one pays on its own"
            return {"action": "ask", "question": self.question_view(j), "why": why, "progress": progress}
        hand_off = risk > self.cfg.handoff_cost
        out = {"action": "hand_off" if hand_off else "finish", "progress": progress, **self.assessment(brief=True)}
        if hand_off:
            out["hand_off_reason"] = (f"No remaining question is worth asking, but the doubt that is left would still cost "
                                      f"about {_money(risk)} a year in mistakes. A benefits navigator or caseworker "
                                      f"should go through it with the person.")
        return out

    def support(self) -> dict:
        """How many training households fit these answers before the support floor stepped in. Below the floor
        the outcome model is extrapolating ("thin"): fewer households like this person than the engine trusts."""
        sp = self.s.support()
        return {"households_like_this": round(sp["raw_ess"], 1),
                "level": "supported" if sp["raw_ess"] >= self.cfg.min_ess else "thin",
                "conflicting_answers": sp["conflicts"]}

    def _tier(self, recommend: bool, p: float, thin: bool) -> str:
        tier = ("likely" if p >= 0.5 else "worth checking") if recommend else ("unlikely" if p < 0.1 else "not recommended")
        if thin and tier in ("unlikely", "not recommended") and round(p, 2) >= self.cfg.unsure_floor:
            tier = "can't tell"  # too few households like this person to call this negative (D33)
        return tier

    def _thin(self, s: Session) -> bool:
        return self.cfg.abstain_below is not None and s.support()["raw_ess"] < self.cfg.abstain_below

    def assessment(self, brief: bool = False) -> dict:
        rows = []
        thin = self._thin(self.s)
        for dv in self.s.decisions():
            p = dv.p_eligible
            tier = self._tier(dv.recommend, p, thin)
            row = {"id": dv.decision, "programme": self.b.labels[dv.decision], "tier": tier, "probability": round(p, 2)}
            if dv.recommend or not brief:
                lo, hi = max(dv.value_low, 0), max(dv.value_high, 0)
                row["value_if_eligible_per_year"] = [int(round(lo, -1)), int(round(hi, -1))]  # whole dollars
            rows.append(row)
        rows.sort(key=lambda r: (r["tier"] not in ("likely", "worth checking"), -r["probability"]))
        recs = [r for r in rows if r["tier"] in ("likely", "worth checking")]
        unsure = [{k: r[k] for k in ("id", "programme", "tier", "probability")} for r in rows if r["tier"] == "can't tell"]
        out = {"recommend": recs, "unsure": unsure}  # always present, empty when the data covers the person
        if not brief:
            out["all_programmes"] = rows
            out["facts"] = [{"question": k, "reading": f["reading"]} for k, f in self.facts.items()]
            out["expected_mistake_cost"] = round(self.s.risk())
            out["support"] = self.support()
        out["note"] = self.b.info.get("caveat", "")
        return out

    # ------------------------------------------------------------------------------- what would change it
    def _neighbours(self, q, k: int) -> list[int]:
        """The answers next to option k: one bracket either side for amounts and counts, the other option for
        yes/no, every other option for an unordered choice. The state is not a neighbour of anything."""
        n = len(q.options)
        if q.id == "state":
            return []
        if q.edges is not None or all(_COUNT.fullmatch(o.lower()) for o in q.options):
            return [j for j in (k - 1, k + 1) if 0 <= j < n]
        return [j for j in range(n) if j != k]

    def triage(self, already=()) -> dict:
        """For a caseload list: what this person stands to gain and what to ask them first. expected_value is the
        sum, over the programmes the engine would recommend now and the household doesn't already get, of
        probability times typical yearly value (the dollars a call could unlock); next is the question worth
        most, with its dollar worth, or None. opened_by_rule: programmes the plan's federal rules open because of
        what the household already gets (SNAP -> WIC's income test, for one), not counted in expected_value."""
        already = list(dict.fromkeys(already))
        opened = []
        if already:
            plan = self.plan(already)  # validates the ids
            opened = [{"id": p["id"], "because": p["links"][0]["already"], "rule": p["links"][0]["source"]}
                      for s in plan["steps"] for p in s["programmes"] if p.get("added_by_rule")]
        thin = self._thin(self.s)
        recs, ev = [], 0.0
        for dv in self.s.decisions():
            if dv.recommend and dv.decision not in already:
                typical = (max(dv.value_low, 0) + max(dv.value_high, 0)) / 2
                ev += dv.p_eligible * typical
                recs.append({"id": dv.decision, "tier": self._tier(True, dv.p_eligible, thin),
                             "probability": round(dv.p_eligible, 2)})
        recs.sort(key=lambda r: -r["probability"])
        nxt = self.next_step()
        ask = None
        if nxt["action"] == "ask":
            ask = {"id": nxt["question"]["id"], "text": nxt["question"]["text"], "worth_per_year": nxt["why"]["worth_per_year"]}
        out = {"expected_value": int(round(ev, -1)), "recommend": recs, "next": ask, "action": nxt["action"],
               "expected_mistake_cost": round(self.s.risk()), "support": self.support()["level"]}
        if already:
            out["already"] = already
            out["opened_by_rule"] = opened
        return out

    def _category_fits(self, pid: str) -> bool:
        for q, ruled_out in PLAN_CATEGORY.get(pid, []):
            choice = self.raw.get(q, {}).get("choice")
            if choice is not None and str(choice).lower() in ruled_out:  # unanswered: it may fit
                return False
        return True

    def plan(self, already=()) -> dict:
        """The recommended programmes in the order to apply (PLAN_STEPS), with the statutory links between
        them (PLAN_LINKS) where both ends are recommended or already received, and one practical note per step
        (PLAN_NOTES). `already`: programme ids the household gets now; they leave the steps, and a programme
        whose income test one of them settles is added as "worth checking" when its category can fit."""
        already = list(dict.fromkeys(already))
        unknown = [a for a in already if a not in self.b.labels]
        if unknown:
            raise FactError(f"unknown programme {unknown[0]!r}; use an id from: {', '.join(self.b.labels)}")
        recs = {r["id"]: r for r in self.assessment(brief=True)["recommend"] if r["id"] not in already}
        steps = []
        for key, title, ids in PLAN_STEPS:
            items = []
            for pid in ids:
                if pid in already or pid not in self.b.labels:
                    continue
                links = []
                for target, sources, kind, text, cite in PLAN_LINKS:
                    have = [s for s in sources if (s in recs or s in already) and s != pid]
                    if target == pid and have:
                        links.append({"kind": kind, "from": have, "source": cite,
                                      "already": [s for s in have if s in already],
                                      "text": text.format(src=_either([_SHORT[s] for s in have]))})
                if pid in recs:
                    items.append({"id": pid, "programme": recs[pid]["programme"], "tier": recs[pid]["tier"],
                                  "links": links})
                elif any(link["already"] for link in links) and self._category_fits(pid):
                    items.append({"id": pid, "programme": self.b.labels[pid], "tier": "worth checking",
                                  "added_by_rule": True, "links": [l for l in links if l["already"]]})
            if items:
                in_step = {i["id"] for i in items}
                notes = [{"kind": k, "text": t} for k, (ids_, t) in PLAN_NOTES.items() if in_step & set(ids_)]
                steps.append({"step": key, "title": title, "programmes": items, "notes": notes})
        for n, s in enumerate(steps, 1):
            s["number"] = n
        return {"steps": steps, "already": [{"id": a, "programme": self.b.labels[a]} for a in already],
                "note": PLAN_NOTE}

    def cliff(self, question: str = "earned_income") -> dict:
        """Every option of one amount question (earnings by default), the other answers kept: each programme's
        tier, probability and typical value at that option, and the expected yearly value of all of them. A drop
        between neighbouring options is a benefits cliff. Each option reuses the belief (Session.what_if)."""
        if question not in self.qi:
            raise FactError(f"unknown question {question!r}")
        j = self.qi[question]
        q = self.b.pop.questions[j]
        if q.edges is None:
            raise FactError(f"{question} is not an amount question")
        f = self.facts.get(question)
        current = int(np.argmax(f["evidence"])) if f is not None and np.ptp(f["evidence"]) > 0 else None
        groups = {g: [d for d in ds if d in self.b.pop.decisions] for g, ds in CLIFF_GROUPS.items()}
        groups = {g: ds for g, ds in groups.items() if ds}  # only the groups this bundle has
        group_of = {d: g for g, ds in groups.items() for d in ds}
        options = []
        for k in range(len(q.options)):
            lik = np.zeros(len(q.options))
            lik[k] = 1.0
            s2 = self.s.what_if(j, lik)
            thin = self._thin(s2)
            rows, by_group = [], {g: 0.0 for g in groups}
            for dv in s2.decisions():
                typical = (max(dv.value_low, 0) + max(dv.value_high, 0)) / 2
                ev = dv.p_eligible * typical
                by_group[group_of.get(dv.decision, "food and cash")] += ev
                rows.append({"id": dv.decision, "tier": self._tier(dv.recommend, dv.p_eligible, thin),
                             "probability": round(dv.p_eligible, 2), "typical_value": int(round(typical, -1)),
                             "expected_value": int(round(ev, -1))})
            options.append({"option": q.options[k], "current": k == current,
                            "expected_value": int(round(sum(by_group.values()), -1)),
                            "by_group": {g: int(round(v, -1)) for g, v in by_group.items()},
                            "programmes": rows})
        return {"question": question, "unit": self.question_view(j).get("unit"), "options": options,
                "groups": groups, "note": CLIFF_NOTE}

    def explain(self) -> dict:
        """For every programme whose result would change: which single answer, moved to a neighbouring option
        with the others kept, changes it, and to what. Each case reuses the belief (Session.what_if)."""
        thin = self._thin(self.s)
        now = {dv.decision: (self._tier(dv.recommend, dv.p_eligible, thin), dv.p_eligible) for dv in self.s.decisions()}
        changes: dict[str, list] = {d: [] for d in now}
        tried = 0
        for qid, f in self.facts.items():
            ev = f["evidence"]
            if np.ptp(ev) == 0:
                continue  # "don't know": nothing to move
            j = self.qi[qid]
            q = self.b.pop.questions[j]
            k = int(np.argmax(ev))
            for alt in self._neighbours(q, k):
                lik = np.zeros(len(q.options))
                lik[alt] = 1.0
                s2 = self.s.what_if(j, lik)
                thin2 = self._thin(s2)
                tried += 1
                for dv in s2.decisions():
                    t = self._tier(dv.recommend, dv.p_eligible, thin2)
                    if _GROUP[t] != _GROUP[now[dv.decision][0]]:
                        changes[dv.decision].append({
                            "question": qid, "kind": "amount" if q.edges is not None else "household",
                            "from": q.options[k], "to": q.options[alt], "tier": t,
                            "probability": round(dv.p_eligible, 2),
                            "probability_change": round(dv.p_eligible - now[dv.decision][1], 2)})
        rows = [{"id": d, "programme": self.b.labels[d], "tier": now[d][0], "probability": round(now[d][1], 2),
                 "would_change": ch} for d, ch in changes.items() if ch]
        rows.sort(key=lambda r: (_GROUP[r["tier"]] not in ("likely", "worth checking"), -r["probability"]))
        return {"near_the_line": rows, "answers_tried": tried, "note": EXPLAIN_NOTE}

