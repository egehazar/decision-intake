"""An intake's facts as one short string, so a server can keep nothing between calls.

Every hosted call carries the state, the server rebuilds the belief from it (about 0.1 s), answers,
and hands back the new state. Nothing about the person is stored anywhere: no database, no session
table, no logs of answers. Any replica can serve any call, so the service scales to zero.

The string is readable, so a person or an assistant can see what it holds:

    di1;state=TX;household_size=3;children=1;earned_income=~2400/month;assets=?;#6f0c

    choice      the option's text, spaces as "_"            tenure=rent   household_size=7_or_more
    amount      a number or low..high, "~" if approximate,  earned_income=~2400/month
                "/period" unless it has none                housing_cost=1100..1300/month
    don't know  "?"                                         assets=?
    gets        programmes the household already gets, joined by "+"      gets=snap+medicaid
    check       "#" and four hex digits of a CRC of the rest, so a mangled copy is refused rather
                than silently read as different facts
"""

from __future__ import annotations

import re
import zlib

PREFIX = "di1"
MAX_LEN = 2000
_SAFE = re.compile(r"[A-Za-z0-9$,.+ ]+")
_AMOUNT = re.compile(r"(~?)(\d+(?:\.\d+)?)(?:\.\.(\d+(?:\.\d+)?))?(?:/(year|month|week))?")


class StateError(ValueError):
    pass


def _num(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _check(body: str) -> str:
    return f"{zlib.crc32(body.encode()) & 0xFFFF:04x}"


_PROGRAMME = re.compile(r"[a-z][a-z0-9_]*")


def encode(facts: list[dict], gets=()) -> str:
    parts = []
    for f in facts:
        qid = f["question"]
        if f.get("dont_know"):
            v = "?"
        elif "amount" in f:
            a = f["amount"]
            v = ("~" if a["approximate"] else "") + _num(a["low"])
            if a["high"] != a["low"]:
                v += ".." + _num(a["high"])
            if a["period"] != "none":
                v += "/" + a["period"]
        else:
            text = f["choice"]
            if not _SAFE.fullmatch(text):
                raise StateError(f"option {text!r} of {qid} can't be written into a state")
            v = text.replace(" ", "_")
        parts.append(f"{qid}={v}")
    if gets:
        bad = [g for g in gets if not _PROGRAMME.fullmatch(g)]
        if bad:
            raise StateError(f"programme {bad[0]!r} can't be written into a state")
        parts.append("gets=" + "+".join(gets))
    body = ";".join([PREFIX, *parts])
    return f"{body};#{_check(body)}"


def decode(questions, token: str) -> list[dict]:
    """The facts a state holds, in order, as typed facts the intake accepts. Raises StateError."""
    if not isinstance(token, str) or not token.strip():
        raise StateError("state is missing; call start_intake and pass the state from the most recent result")
    token = token.strip()
    if len(token) > MAX_LEN:
        raise StateError("state is too long")
    body, sep, check = token.rpartition(";#")
    if not sep or check != _check(body):
        raise StateError("state is damaged (its check doesn't match); pass the exact state string from the most "
                         "recent result")
    head, *parts = body.split(";")
    if head != PREFIX:
        raise StateError(f"unknown state version {head!r}")
    by_id = {q.id: q for q in questions}
    seen, facts = set(), []
    for p in parts:
        qid, eq, v = p.partition("=")
        if qid == "gets" and eq and "gets" not in seen:  # programmes the household already gets
            names = v.split("+")
            if not all(_PROGRAMME.fullmatch(n) for n in names):
                raise StateError("state holds an unreadable list of programmes")
            seen.add("gets")
            facts.append({"question": "gets", "programmes": names})
            continue
        q = by_id.get(qid)
        if not eq or q is None:
            raise StateError(f"state holds an unknown question {qid!r}")
        if qid in seen:
            raise StateError(f"state holds {qid} twice")
        seen.add(qid)
        if v == "?":
            facts.append({"question": qid, "dont_know": True})
            continue
        m = _AMOUNT.fullmatch(v) if q.edges is not None else None
        if m:
            low = float(m.group(2))
            high = float(m.group(3)) if m.group(3) else low
            facts.append({"question": qid, "amount": {"low": low, "high": high, "period": m.group(4) or "none",
                                                      "approximate": m.group(1) == "~"}})
            continue
        text = v.replace("_", " ")
        if text not in q.options:
            raise StateError(f"state holds {text!r} for {qid}, which is not one of its options")
        facts.append({"question": qid, "choice": text})
    return facts
