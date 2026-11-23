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

YES = {"yes", "y", "true", "yeah", "yep"}
NO = {"no", "n", "false", "nope", "none"}


@dataclass
class Settings:
    question_cost: float = 15.0  # dollars a year a question must be expected to save before it's asked
    min_ess: float = 10.0  # support floor (see Session)
    handoff_cost: float = 2500.0  # expected mistakes left at the end above this go to a person
    max_questions: int = 18


class FactError(ValueError):
    pass


def _money(x: float) -> str:
    return f"${x:,.0f}"


class Intake:
    def __init__(self, bundle: Bundle, settings: Settings | None = None):
        self.b, self.cfg = bundle, settings or Settings()
        self.qi = {q.id: j for j, q in enumerate(bundle.pop.questions)}
        self.policy = VOIPolicy(question_cost=self.cfg.question_cost, budget=self.cfg.max_questions)
        self.facts: dict[str, dict] = {}  # question id -> {"evidence", "reading"}, in the order given
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
        low = float(low if low is not None else high)
        high = float(high if high is not None else low)
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
        if factor == 1.0 and not slots["approximate"]:
            return ev, f"{stated} -> {spread}"
        return ev, f"{stated} -> {conv} -> {spread}"

    def _evidence(self, q, f: dict) -> tuple[np.ndarray, str]:
        if f.get("dont_know"):
            return np.ones(len(q.options)), "don't know (no evidence; the belief is unchanged)"
        if f.get("amount") is not None:
            return self._amount(q, f["amount"])
        if f.get("choice") is not None:
            k = self._choice(q, f["choice"])
            ev = np.zeros(len(q.options))
            ev[k] = 1.0
            return ev, q.options[k]
        raise FactError('give one of "choice", "amount" or "dont_know"')

    def record(self, facts: list[dict]) -> list[dict]:
        """Record typed facts; returns one reading per fact (or an error saying how to fix it).
        Valid facts go in even when others in the same call are rejected."""
        out, new, replaced = [], [], False
        for f in facts:
            qid = f.get("question")
            if qid not in self.qi:
                out.append({"question": qid, "error": f"unknown question; use an id from: {', '.join(self.qi)}"})
                continue
            try:
                ev, reading = self._evidence(self.b.pop.questions[self.qi[qid]], f)
            except FactError as e:
                out.append({"question": qid, "error": str(e)})
                continue
            r = {"question": qid, "reading": reading}
            if qid in self.facts:
                replaced = True
                r["replaces"] = self.facts.pop(qid)["reading"]
            self.facts[qid] = {"evidence": ev, "reading": reading}
            new.append(qid)
            out.append(r)
        before = self.s.risk()
        if replaced:  # a correction: replay every fact on a fresh belief
            self.s = self._session()
            for qid, f in self.facts.items():
                self.s.observe(self.qi[qid], f["evidence"], source="fact")
        else:
            for qid in new:
                self.s.observe(self.qi[qid], self.facts[qid]["evidence"], source="fact")
        conflicts = {e["q"] for e in self.s.log if e.get("conflict")}
        for r in out:
            if r.get("question") in conflicts and "reading" in r:
                r["warning"] = "no household on record fits this together with the earlier answers; check it with the person"
        if new:
            self.history.append({"facts": new, "cost_before": round(before), "cost_after": round(self.s.risk())})
        return out

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
                    "similar_households": round(s.ess())}
        j = self.policy.next_question(s, len(self.facts))
        if j is not None:
            single = dict(s.voi_table())
            v = single.get(j, 0.0)
            by = np.maximum(s.voi_by_decision(j), 0)
            share = by / by.sum() if by.sum() > 0 else by
            settles = [{"programme": self.b.labels[self.b.pop.decisions[d]], "share": round(float(share[d]), 2)}
                       for d in np.argsort(-share)[:3] if share[d] >= 0.1]
            why = {"worth_per_year": round(v), "question_cost": self.cfg.question_cost, "mostly_settles": settles}
            if v <= self.cfg.question_cost:
                why["note"] = "worth asking only together with another question; no single one pays on its own"
            return {"action": "ask", "question": self.question_view(j), "why": why, "progress": progress}
        hand_off = risk > self.cfg.handoff_cost
        out = {"action": "hand_off" if hand_off else "finish", "progress": progress, **self.assessment(brief=True)}
        if hand_off:
            out["hand_off_reason"] = (f"No remaining question is worth asking, but the doubt that is left would still cost "
                                      f"about {_money(risk)} a year in mistakes. A caseworker should review.")
        return out

    def assessment(self, brief: bool = False) -> dict:
        rows = []
        for dv in self.s.decisions():
            p = dv.p_eligible
            tier = ("likely" if p >= 0.5 else "worth checking") if dv.recommend else ("unlikely" if p < 0.1 else "not recommended")
            row = {"programme": self.b.labels[dv.decision], "tier": tier, "probability": round(p, 2)}
            if dv.recommend or not brief:
                lo, hi = max(dv.value_low, 0), max(dv.value_high, 0)
                row["value_if_eligible_per_year"] = "small" if hi < 120 else [round(lo, -1), round(hi, -1)]
            rows.append(row)
        rows.sort(key=lambda r: (r["tier"] not in ("likely", "worth checking"), -r["probability"]))
        recs = [r for r in rows if r["tier"] in ("likely", "worth checking")]
        out = {"recommend": recs}
        if not brief:
            out["all_programmes"] = rows
            out["facts"] = [{"question": k, "reading": f["reading"]} for k, f in self.facts.items()]
            out["expected_mistake_cost"] = round(self.s.risk())
        out["note"] = self.b.info.get("caveat", "")
        return out

