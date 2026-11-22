"""End to end with language: the engine interviews held-out households who answer in sentences.

For each household and reader the engine asks its questions; the household's answer to question j is
generated from its record with a style fixed by (household, question), so every reader hears the same
words. The reader turns the sentence into evidence; the engine decides what to ask next and when to
stop. Readers:
  typed        - the person picks an option from a list (no language; "don't know" gives no evidence)
  llm_hard     - instruct model picks one option
  llm_selfconf - instruct model picks one option and rates its confidence
  dm           - decision model gives a probability per option (calibrated temperature)

usage: python experiments/run_language.py N_HOUSEHOLDS QUESTION_COST MIN_ESS READER [READER...]
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
sys.path.insert(0, os.path.dirname(__file__))

from benefits.population import build, load_table  # noqa: E402
from benefits.utterances import answer, styles_for  # noqa: E402
from read_eval import CACHE, READERS, evidence_from_raw, load_cache, raw_reading  # noqa: E402

from decisionintake import Session, VOIPolicy  # noqa: E402
from decisionintake.simulate import split_by_group  # noqa: E402

LOCK = threading.Lock()


def main():
    n, lam, min_ess, names = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
    t = load_table()
    t["_other_income"] = np.maximum(t.market_income - t.earned_income - t.social_security, 0)
    pop, loss, meta = build(t)
    rest, test = split_by_group(pop, 0.15, seed=0)
    train = pop.subset(rest)
    rng = np.random.default_rng(11)
    hh = rng.choice(test, size=n, replace=False)
    cache = load_cache()
    os.makedirs("results", exist_ok=True)
    f = open(CACHE, "a", encoding="utf-8")

    def utterance(i, j):
        r = np.random.default_rng(int(i) * 131 + j)
        q = pop.questions[j]
        style = str(r.choice(styles_for(q.id)))
        return style, answer(q.id, t.iloc[int(i)], int(pop.answers[i, j]), style, r)

    def session(name, reader, i):
        s = Session(train, loss, min_ess=min_ess)
        pol = VOIPolicy(question_cost=lam)
        k = 0
        trace = []
        while True:
            j = pol.next_question(s, k)
            if j is None:
                break
            k += 1
            q = pop.questions[j]
            style, utt = utterance(i, j)
            if name == "typed":
                ev = np.ones(len(q.options)) if style == "dont_know" else np.eye(len(q.options))[int(pop.answers[i, j])]
            else:
                key = (name, q.id, utt)
                with LOCK:
                    raw = cache.get(key)
                if raw is None:
                    raw = raw_reading(reader, q, utt)
                    with LOCK:
                        cache[key] = raw
                        f.write(json.dumps({"reader": name, "qid": q.id, "utterance": utt, "raw": raw}) + "\n")
                        f.flush()
                ev = evidence_from_raw(reader, q, raw)
            s.observe(j, ev, source=name)
            trace.append({"q": q.id, "style": style, "utterance": utt, "truth": int(pop.answers[i, j]),
                          "top": int(np.argmax(ev)), "flat": bool(np.ptp(ev) == 0)})
        return {"i": int(i), "n_questions": k, "recommend": s.recommend_mask().astype(int).tolist(),
                "risk": s.risk(), "ess": s.ess(), "trace": trace}

    out = {}
    for name in names:
        reader = READERS[name]() if name != "typed" else None
        with ThreadPoolExecutor(4 if name != "typed" else 1) as ex:
            res = list(ex.map(lambda i: session(name, reader, i), hh))
        out[name] = res
        idx = np.array([r["i"] for r in res])
        rec = np.array([r["recommend"] for r in res], bool)
        nq = np.array([r["n_questions"] for r in res], float)
        w = pop.weight[idx] / pop.weight[idx].sum()
        el = pop.eligible[idx]
        miss, fr = loss.realised(pop.subset(idx))
        realised = (np.where(rec, 0.0, miss) + np.where(rec, fr, 0.0)).sum(1)
        print(f"{name:<13} n={len(res)}  q={w @ nq:5.2f}  loss=${w @ realised:7.0f}  "
              f"prog_recall={(w[:, None] * (el & rec)).sum() / (w[:, None] * el).sum():.3f}  "
              f"false_rec={(w[:, None] * (~el & rec)).sum() / max((w[:, None] * rec).sum(), 1e-12):.3f}", flush=True)
        with open(f"results/language_{name}_{n}_{int(lam)}_{int(min_ess)}.json", "w") as g:
            json.dump(res, g)


if __name__ == "__main__":
    main()
