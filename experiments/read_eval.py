"""How well does each reader turn a spoken answer into evidence?

Answers are generated from held-out households' exact records (adapters/benefits/utterances.py), so
the true option is known. For each reader: accuracy of its top option, log loss of the true option,
how often it is confidently wrong, and how often it says "not stated". The calibration split fits
the decision model's temperature; the evaluation split reports.

usage: python experiments/read_eval.py SPLIT N READER [READER...]
  SPLIT: calib | eval     READER: dm (decision model) | dm_small | llm_hard | llm_selfconf
Every model call is cached in results/read_cache.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))

from benefits.population import build, load_table  # noqa: E402
from benefits.utterances import answer, styles_for  # noqa: E402

from decisionintake.language import DecisionModelReader, LLMReader  # noqa: E402
from decisionintake.simulate import split_by_group  # noqa: E402

CACHE = "results/read_cache.jsonl"
READERS = {
    "dm": lambda: DecisionModelReader(url="http://127.0.0.1:9995", name="dm"),
    "dm_thr": lambda: DecisionModelReader(url="http://127.0.0.1:9995", mode="thresholds", name="dm_thr"),
    "dm_small": lambda: DecisionModelReader(url="http://127.0.0.1:9997", name="dm_small"),
    "llm_hard": lambda: LLMReader(url="http://127.0.0.1:9996", mode="hard", name="llm_hard"),
    "llm_selfconf": lambda: LLMReader(url="http://127.0.0.1:9996", mode="selfconf", name="llm_selfconf"),
}


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line)
            c[(r["reader"], r["qid"], r["utterance"])] = r["raw"]
    return c


def raw_reading(reader, q, utt):
    if isinstance(reader, DecisionModelReader):
        return reader.probabilities(q, utt).tolist()
    choice, conf = reader.ask(q, utt)
    return [choice, conf]


def evidence_from_raw(reader, q, raw):
    if isinstance(reader, DecisionModelReader):
        return reader.evidence(q, "", p=np.array(raw))
    return reader.evidence(q, "", parsed=(raw[0], raw[1]))


def items(split: str, n: int, seed: int = 3):
    t = load_table()
    t["_other_income"] = np.maximum(t.market_income - t.earned_income - t.social_security, 0)
    pop, loss, meta = build(t)
    rest, test = split_by_group(pop, 0.15, seed=0)
    sub_tr, sub_va = split_by_group(pop.subset(rest), 0.12, seed=1)
    idx = rest[sub_va] if split == "calib" else test
    rng = np.random.default_rng(seed if split == "eval" else seed + 100)
    out = []
    for _ in range(n):
        i = int(rng.choice(idx))
        j = int(rng.integers(len(pop.questions)))
        q = pop.questions[j]
        style = str(rng.choice(styles_for(q.id)))
        utt = answer(q.id, t.iloc[i], int(pop.answers[i, j]), style, rng)
        out.append({"i": i, "j": j, "qid": q.id, "style": style, "utterance": utt, "truth": int(pop.answers[i, j])})
    return pop, out


def main():
    split, n, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    pop, its = items(split, n)
    cache = load_cache()
    os.makedirs("results", exist_ok=True)
    for name in names:
        reader = READERS[name]()
        todo = [it for it in its if (name, it["qid"], it["utterance"]) not in cache]
        with open(CACHE, "a", encoding="utf-8") as f, ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(raw_reading, reader, pop.questions[it["j"]], it["utterance"]): it for it in todo}
            for k, fut in enumerate(futs):
                it = futs[fut]
                raw = fut.result()
                cache[(name, it["qid"], it["utterance"])] = raw
                f.write(json.dumps({"reader": name, "qid": it["qid"], "utterance": it["utterance"], "raw": raw}) + "\n")
                f.flush()
                if k % 50 == 0:
                    print(f"{name}: {k}/{len(todo)}", flush=True)
        report(name, reader, pop, its, cache)


def report(name, reader, pop, its, cache, by_style=True):
    rows = []
    for it in its:
        q = pop.questions[it["j"]]
        ev = evidence_from_raw(reader, q, cache[(name, it["qid"], it["utterance"])])
        p = ev / ev.sum()
        rows.append((it["style"], it["truth"], p))
    def summary(sel):
        acc = np.mean([np.argmax(p) == t for _, t, p in sel])
        nll = np.mean([-np.log(max(p[t], 1e-6)) for _, t, p in sel])
        conf_wrong = np.mean([(p.max() > 0.8) and (np.argmax(p) != t) for _, t, p in sel])
        flat = np.mean([p.max() - p.min() < 1e-9 for _, t, p in sel])
        return acc, nll, conf_wrong, flat, len(sel)
    acc, nll, cw, flat, k = summary(rows)
    print(f"\n{name}: n={k}  top-option accuracy {acc:.3f}  log loss {nll:.3f}  confidently wrong {cw:.3f}  no-info {flat:.3f}")
    if by_style:
        for st in sorted({r[0] for r in rows}):
            a, l, c, fl, kk = summary([r for r in rows if r[0] == st])
            print(f"   {st:<10} n={kk:4d}  acc {a:.3f}  log loss {l:.3f}  conf wrong {c:.3f}  no-info {fl:.3f}")


if __name__ == "__main__":
    main()
