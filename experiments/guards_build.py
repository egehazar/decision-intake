"""Build the 2026 rules file and check every rule against PolicyEngine's own results (DECISIONS D32).

A rule ships only if, among the training households it matches, PolicyEngine finds the programme in no
more than the rule's cap (2%, survey-weighted), so a capped probability is never below what the data shows. Then an audit on random valid profiles (every combination of answers a person
could give, not only common ones): how often v2 calls a ruled-out programme likely, with and without.

usage: python experiments/guards_build.py [BUNDLE] [OUT.json]   # default: the v2 bundle -> bundles/guards_2026.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decisionintake import bundle  # noqa: E402
from decisionintake.guards import CAP, DEFAULT_RULES, GuardedOutcome, Rules, build_rules  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAX_EXCEPTIONS = CAP  # a cap below the data's own rate would be miscalibrated


def random_profiles(questions, n, seed=0):
    """Uniform over each question's options, made consistent: at least one adult, no child under 5 without
    children, no married couple without two adults."""
    qi = {q.id: j for j, q in enumerate(questions)}
    rng = np.random.default_rng(seed)
    R = np.stack([rng.integers(0, len(q.options), n) for q in questions], 1).astype(np.uint8)
    hh = R[:, qi["household_size"]] + 1
    R[:, qi["children"]] = np.minimum(R[:, qi["children"]], np.maximum(hh - 1, 0))
    kids = R[:, qi["children"]]
    R[:, qi["child_under5"]] = np.where(kids == 0, 0, R[:, qi["child_under5"]])
    R[:, qi["married"]] = np.where(hh - kids < 2, 0, R[:, qi["married"]])
    return R


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bundles", "benefits_v2.npz")
    dest = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_RULES)
    b = bundle.load(src)
    pop = b.pop
    rules = build_rules(pop.questions)
    R = Rules(rules, pop.questions, pop.decisions)
    di = {d: j for j, d in enumerate(pop.decisions)}
    elig = pop.value > 0
    kept, report = [], []
    print(f"{len(rules)} candidate rules; checked against {len(pop):,} training households")
    for r, m in zip(rules, R.matches(pop.answers)):
        stats = {}
        ok = True
        for d in r["decisions"]:
            n = int(m.sum())
            share = float((pop.weight[m] * elig[m, di[d]]).sum() / pop.weight[m].sum()) if n else 0.0
            stats[d] = {"households": n, "eligible": int(elig[m, di[d]].sum()), "weighted_share": round(share, 5)}
            ok &= share <= MAX_EXCEPTIONS
        r = {**r, "checked": stats}
        report.append((r["id"], ok, stats))
        if ok:
            kept.append(r)
    for rid, ok, stats in report:
        s = "; ".join(f"{d}: {v['eligible']}/{v['households']} ({v['weighted_share']:.2%})" for d, v in stats.items())
        print(f"  {'keep' if ok else 'DROP'} {rid:24s} {s}")
    out = {"about": "Statutory limits applied on top of the outcome model (src/decisionintake/guards.py, "
                    "DECISIONS D32). Each rule was checked against PolicyEngine-US results on the training "
                    f"households: kept only if at most {MAX_EXCEPTIONS:.0%} of the households it matches get the "
                    "programme (survey-weighted).",
           "cap": CAP, "rules": kept}
    open(dest, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print(f"wrote {dest}: {len(kept)} rules, checked on {os.path.basename(src)}")

    # audit: random valid profiles
    R2 = Rules(kept, pop.questions, pop.decisions)
    P = random_profiles(pop.questions, 50_000)
    p0, _ = b.model.predict(pop, P)
    p1, _ = GuardedOutcome(b.model, R2).predict(pop, P)
    ruled = np.zeros_like(p0, bool)
    for (conds, ds, cap), m in zip(R2.compiled, R2.matches(P)):
        for d in ds:
            ruled[m, d] = True
    n = int(ruled.any(1).sum())
    print(f"audit, 50,000 random valid profiles: {n:,} have a ruled-out programme")
    for d, name in enumerate(pop.decisions):
        m = ruled[:, d]
        if m.sum():
            print(f"  {name:13s} {m.sum():6d} profiles: called likely (p>=0.5) by v2 {np.mean(p0[m, d] >= 0.5):6.1%}, "
                  f"worth checking or likely (p>=0.2) {np.mean(p0[m, d] >= 0.2):6.1%}; with the rules "
                  f"{np.mean(p1[m, d] >= 0.2):.1%}")
    untouched = ~ruled
    assert np.array_equal(p0[untouched], p1[untouched]), "the rules may only lower ruled-out probabilities"


if __name__ == "__main__":
    main()
