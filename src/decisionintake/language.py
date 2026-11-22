"""Reading free-text answers into evidence over a question's options.

Three readers, so their effect on decisions can be compared:
  DecisionModelReader - a decision model (llama.cpp /v1/systemone, e.g. Kev-4B) gives, for every option,
                        the probability that the answer establishes it. Calibrated on held-out answers,
                        that becomes soft evidence: a hedged answer spreads over neighbouring options.
  LLMReader(hard)     - an instruct model picks one option (or "not stated"): one-hot evidence.
  LLMReader(selfconf) - the same, plus the model's own confidence, spread over the rest.
Nothing here knows the domain: it sees the question text, the option labels and the answer.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np

from .core import Question


def _post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def option_statement(q: Question, option: str) -> str:
    return f'The answer means the correct choice for "{q.text}" is: {option}.'


def _noul(statement: str, yes: str, no: str) -> dict:
    return {"type": "noul", "instructions": f"Answer this question about the text: {statement}",
            "criteria": {"true": yes, "false": no}}


@dataclass
class DecisionModelReader:
    """mode "options": one yes/no decision per option ("does the answer establish this option?").
    mode "thresholds": for ordered questions, one decision per step up the scale ("is it this option
    or higher?"); the chain is made monotone and its differences are the distribution. Unordered
    questions always use options. probabilities() returns per-option scores in both modes."""

    url: str = "http://127.0.0.1:9995"
    mode: str = "options"
    temperature: float = 1.0  # fitted on held-out answers: evidence ~ p ** (1 / temperature)
    name: str = "decision_model"

    def _state(self, q: Question, utterance: str) -> str:
        opts = "; ".join(q.options)
        return f"An intake assistant asked: {q.text}\nThe choices, in order: {opts}\nThe person answered: {utterance}"

    def probabilities(self, q: Question, utterance: str) -> np.ndarray:
        state = self._state(q, utterance)
        if self.mode == "thresholds" and q.ordered:
            questions = {
                f"t{k}": _noul(f'The answer is "{q.options[k]}" or a higher choice in the list.',
                               "Yes, the answer is at that choice or above it.",
                               "No, the answer is below that choice, or it's unclear.")
                for k in range(1, len(q.options))
            }
            r = _post(self.url + "/v1/systemone", {"state": state, "questions": questions})
            ge = np.array([1.0] + [float(r["answers"][f"t{k}"]["noul"]) for k in range(1, len(q.options))])
            ge = np.minimum.accumulate(ge)  # P(at least option k) can't rise with k
            p = ge - np.append(ge[1:], 0.0)
            # thresholds all near 0.5 mean the answer settled nothing: scale the scores down
            certainty = float(np.mean(np.abs(ge[1:] - 0.5)) * 2) if len(ge) > 1 else 1.0
            if p.max() <= 0:
                return np.zeros_like(p)
            return np.clip(p, 0, 1) * certainty / p.max()
        questions = {
            f"o{i}": _noul(option_statement(q, opt), "Yes, the answer establishes this.",
                           "No, it doesn't, or it's unclear.")
            for i, opt in enumerate(q.options)
        }
        r = _post(self.url + "/v1/systemone", {"state": state, "questions": questions})
        return np.array([float(r["answers"][f"o{i}"]["noul"]) for i in range(len(q.options))])

    def evidence(self, q: Question, utterance: str, p: np.ndarray | None = None) -> np.ndarray:
        if p is None:
            p = self.probabilities(q, utterance)
        p = np.clip(p, 1e-4, 1.0)
        shaped = p ** (1.0 / self.temperature)
        shaped = shaped / shaped.sum()
        # If no option is clearly established (an "I don't know"), the evidence says little.
        strength = float(p.max())
        b = len(q.options)
        return strength * shaped + (1.0 - strength) / b


@dataclass
class LLMReader:
    url: str = "http://127.0.0.1:9996"
    mode: str = "hard"  # "hard" or "selfconf"
    name: str = "llm"

    def ask(self, q: Question, utterance: str) -> tuple[int | None, float]:
        opts = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(q.options))
        want = '{"choice": N}' if self.mode == "hard" else '{"choice": N, "confidence": C}'
        prompt = (
            f"An intake assistant asked: {q.text}\nThe person answered: {utterance}\n\nOptions:\n{opts}\n\n"
            # (a 0 sentinel collided with option 1 "no": "No, not us." came back as "not stated")
            f"Which option number does the answer give? If the answer doesn't say (for example the person "
            f"doesn't know), use null for the choice. "
            + ("Also give your confidence that the choice is right, from 0 to 100. " if self.mode == "selfconf" else "")
            + f"Reply with JSON only, like {want}."
        )
        r = _post(self.url + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = r["choices"][0]["message"]["content"]
        # numbers may come back quoted ({"choice": "1"}); missing that read every quoted answer as "not stated"
        m = re.search(r'"choice"\s*:\s*"?(\d+)', text)
        c = re.search(r'"confidence"\s*:\s*"?(\d+(?:\.\d+)?)', text)
        choice = int(m.group(1)) if m else 0
        conf = float(c.group(1)) / 100.0 if c else 1.0
        if not 1 <= choice <= len(q.options):
            return None, 0.0
        return choice - 1, min(max(conf, 0.0), 1.0)

    def evidence(self, q: Question, utterance: str, parsed=None) -> np.ndarray:
        choice, conf = parsed if parsed is not None else self.ask(q, utterance)
        b = len(q.options)
        if choice is None:
            return np.ones(b)  # "not stated": no information
        if self.mode == "hard":
            v = np.zeros(b)
            v[choice] = 1.0
            return v
        v = np.full(b, (1.0 - conf) / max(b - 1, 1))
        v[choice] = conf
        return v


PERIOD_PER_YEAR = {"year": 1.0, "month": 12.0, "week": 52.0, "day": 260.0}


def interval_evidence(q: Question, low: float, high: float, floor: float = 0.02) -> np.ndarray:
    """Spread an amount interval [low, high] (already in the question's own unit) over the question's
    ranges, in proportion to how much of the interval each range covers. A point becomes a small
    interval so it can straddle a boundary."""
    edges = np.asarray(q.edges, float)
    lo_b = np.concatenate([[-np.inf], edges])
    hi_b = np.concatenate([edges, [np.inf]])
    low, high = min(low, high), max(low, high)
    if high - low < 1e-9:
        low, high = low * 0.97, high * 1.03 + 1e-9
    overlap = np.clip(np.minimum(hi_b, high) - np.maximum(lo_b, low), 0, None)
    ev = overlap / overlap.sum() if overlap.sum() > 0 else np.ones(len(q.options)) / len(q.options)
    return (1 - floor) * ev + floor / len(q.options)


@dataclass
class StructuredReader:
    """The LLM reads language into typed slots; code does the arithmetic; optionally a decision model
    gives calibrated probabilities for the questions that are judgments rather than amounts.

    Amount questions (q.edges set): the LLM returns low/high as stated, the period, and flags for
    approximate / zero / unknown. Code converts to the question's unit and spreads the interval over
    the ranges (approximate answers widened by 15% each way). Other questions: the decision model's
    per-option probabilities when dm_url is set, else the LLM's single pick."""

    url: str = "http://127.0.0.1:9996"
    dm_url: str | None = None
    name: str = "structured"

    def extract(self, q: Question, utterance: str) -> dict:
        prompt = (
            f"An intake assistant asked: {q.text}\nThe person answered: {utterance}\n\n"
            "Extract what the answer says, without converting anything. Reply with JSON only:\n"
            '{"low": number or null, "high": number or null, "period": "year" | "month" | "week" | "none", '
            '"approximate": true/false, "zero": true/false, "unknown": true/false}\n'
            "low and high: the smallest and largest dollar amount the answer allows, as stated (one amount: "
            "low = high; \"35 thousand\" = 35000). period: the time period the person stated, \"none\" if they "
            "stated none. approximate: they said about/around/roughly or gave a range. zero: the answer means "
            "none or nothing. unknown: they don't know or won't say. Words without numbers: low and high null."
        )
        r = _post(self.url + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 80,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = r["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    def raw(self, q: Question, utterance: str):
        if q.edges is not None:
            return {"slots": self.extract(q, utterance)}
        if self.dm_url:
            return {"p": DecisionModelReader(url=self.dm_url).probabilities(q, utterance).tolist()}
        return {"pick": list(LLMReader(url=self.url, mode="hard").ask(q, utterance))}

    def evidence(self, q: Question, utterance: str, raw=None) -> np.ndarray:
        raw = raw if raw is not None else self.raw(q, utterance)
        b = len(q.options)
        if "p" in raw:
            return DecisionModelReader().evidence(q, "", p=np.array(raw["p"]))
        if "pick" in raw:
            return LLMReader(mode="hard").evidence(q, "", parsed=tuple(raw["pick"]))
        s = raw.get("slots") or {}
        if s.get("unknown"):
            return np.ones(b)
        if s.get("zero"):
            return interval_evidence(q, 0.0, 0.0)
        low, high = s.get("low"), s.get("high")
        if low is None and high is None:
            return np.ones(b)  # words without an amount: no reliable reading
        low = float(low if low is not None else high)
        high = float(high if high is not None else low)
        period = str(s.get("period") or "none").lower()
        if q.per is None:  # a stock (savings): periods don't apply
            factor = 1.0
        else:
            per_year = PERIOD_PER_YEAR.get(period, PERIOD_PER_YEAR[q.per])  # unstated: assume the question's unit
            factor = per_year / PERIOD_PER_YEAR[q.per]
        low, high = low * factor, high * factor
        if s.get("approximate"):
            low, high = low * 0.85, high * 1.15
        return interval_evidence(q, low, high)


@dataclass
class CombinedReader:
    """Amounts: the structured reading (LLM slots + code arithmetic) and the decision model's calibrated
    per-range probabilities, multiplied as independent evidence (weights 1 - dm_weight and dm_weight).
    Everything else: the LLM's single pick, which read non-amount answers better than the decision model.
    dm_temperature and dm_weight were fitted on one set of held-out answers and frozen before testing
    on another (DECISIONS D23)."""

    url: str = "http://127.0.0.1:9996"
    dm_url: str = "http://127.0.0.1:9995"
    dm_temperature: float = 0.1
    dm_weight: float = 0.6
    name: str = "combined"

    def raw(self, q: Question, utterance: str):
        if q.edges is None:
            return {"pick": list(LLMReader(url=self.url, mode="hard").ask(q, utterance))}
        return {"slots": StructuredReader(url=self.url).extract(q, utterance),
                "p": DecisionModelReader(url=self.dm_url).probabilities(q, utterance).tolist()}

    def evidence(self, q: Question, utterance: str, raw=None) -> np.ndarray:
        raw = raw if raw is not None else self.raw(q, utterance)
        if "pick" in raw:
            return LLMReader(mode="hard").evidence(q, "", parsed=tuple(raw["pick"]))
        s = StructuredReader().evidence(q, "", raw={"slots": raw["slots"]})
        s = s / s.sum()
        d = DecisionModelReader(temperature=self.dm_temperature).evidence(q, "", p=np.array(raw["p"]))
        d = d / d.sum()
        ev = s ** (1 - self.dm_weight) * d ** self.dm_weight
        return ev / ev.sum()
