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
            f"Which option does the answer give? Use 0 if the answer doesn't say. "
            + ("Also give your confidence that the choice is right, from 0 to 100. " if self.mode == "selfconf" else "")
            + f"Reply with JSON only, like {want}."
        )
        r = _post(self.url + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = r["choices"][0]["message"]["content"]
        m = re.search(r'"choice"\s*:\s*(\d+)', text)
        c = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', text)
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
