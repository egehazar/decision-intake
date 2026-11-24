"""After D35 (not registered): how production does on the households D34's adapter bug touched.

D34 found that the training answers left out service-business, partnership and farm self-employment, so
4,809 of 57,240 households had different earnings and 4,061 a different self-employed answer (in the
question brackets: earnings 2,391, self-employed 3,623, and other income, derived from earnings, 1,831).
Production still serves v2, trained on the old answers. This reads D35's saved interviews (both arms
answered from the corrected coding, as a real person would) and splits the 2,183 households by whether
the bug changed their answers, to see what that costs the people it touched and what guards them.

    production  v2 + the D32 rules + abstention (the candidate arm plus the can't-tell layer, as served)
    v2.1        retrained on the corrected answers + 19 rules + abstention (the release arm)

Rates are per program decision (a household has ten), unweighted, with 95% intervals from resampling whole
households (2,000 draws). Counts are given so the small subgroup can be read at its real size.

usage: python experiments/affected_d35.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "adapters"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import run_release as RR  # noqa: E402
from benefits.population import build, load_table  # noqa: E402
from common import splits  # noqa: E402

OUT = os.path.join(ROOT, "results", "affected_d35.json")


def main():
    RR.init()
    idx, test = RR.G["idx"], RR.G["test"]
    pop, _, _, _, _, test_rows = splits()
    old, _, _ = build(load_table(os.path.join(ROOT, "data/cache/d28/chunks/c*.csv")))
    assert len(old) == len(pop)
    qid = [q.id for q in pop.questions]
    je, js, jo = qid.index("earned_income"), qid.index("self_employed"), qid.index("other_income")
    # the same households in the same order (D34): only earnings, self-employment and, through them, other
    # income (market income minus earnings) may differ; outcomes and weights are identical
    touched = [je, js, jo]
    others = [j for j in range(len(qid)) if j not in touched]
    assert (old.answers[:, others] == pop.answers[:, others]).all() and (old.eligible == pop.eligible).all()
    changed = (old.answers[:, touched] != pop.answers[:, touched]).any(1)[test_rows][idx]
    says_se = (test.answers[idx, js] == pop.questions[js].options.index("yes"))

    miss, fr = RR.G["arms"]["candidate"][0].loss.realised(test)
    runs = {}
    for line in open(RR.OUT):
        r = json.loads(line)
        runs[(r["arm"], r["user"], r["i"])] = r
    w_all = test.weight[idx]
    el = test.eligible[idx]
    out = {"households": int(len(idx)), "changed_by_bug": int(changed.sum()),
           "changed_share_weighted": float(w_all[changed].sum() / w_all.sum()),
           "says_self_employed": int(says_se.sum()), "groups": {}}
    lines = [f"After D35, not registered: {len(idx)} households, {changed.sum()} whose answers D34's bug changed "
             f"({out['changed_share_weighted']:.1%} weighted); {says_se.sum()} answer 'self-employed: yes'."]
    for uname in ["clean", "realistic"]:
        lines.append(f"\n{uname} answers")
        out["groups"][uname] = {}
        arm_rows = {}
        for arm, label in (("candidate", "production"), ("release", "v2.1")):
            R_ = [runs[(arm, uname, int(i))] for i in idx]
            rec = np.array([r["rec"] for r in R_], bool)
            p = np.array([r["p"] for r in R_])
            raw = np.array([r["raw_ess"] for r in R_])
            q = np.array([r["q"] for r in R_], float)
            flag = ~rec & (raw < RR.ABSTAIN_BELOW)[:, None] & (p >= RR.UNSURE_FLOOR)
            if arm == "candidate":  # what the can't-tell layer adds: the same interviews without it
                arm_rows["production without can't-tell"] = (rec, np.zeros_like(flag), q)
            arm_rows[label] = (rec, flag, q)
        for gname, g in (("changed by the bug", changed), ("not changed", ~changed),
                         ("say self-employed", says_se), ("say not self-employed", ~says_se)):
            w = w_all[g] / w_all[g].sum()
            row = {"n": int(g.sum())}
            parts = []
            boots = np.random.default_rng(11).integers(0, g.sum(), size=(2000, g.sum()))  # households resampled
            for label, (rec, flag, q) in arm_rows.items():
                conf_neg = ~rec[g] & ~flag[g]
                wn_num, wn_den = int((conf_neg & el[g]).sum()), int(conf_neg.sum())
                me_den = int(el[g].sum())
                wrong_neg = wn_num / max(wn_den, 1)
                wrong_rec = (rec[g] & ~el[g]).sum() / max(rec[g].sum(), 1)
                missed_eligible = wn_num / max(me_den, 1)
                hh_num, hh_den = (conf_neg & el[g]).sum(1), conf_neg.sum(1)  # per household, for the bootstrap
                hh_el = el[g].sum(1)
                bw = np.array([hh_num[b].sum() / max(hh_den[b].sum(), 1) for b in boots])
                be = np.array([hh_num[b].sum() / max(hh_el[b].sum(), 1) for b in boots])
                wn_ci, me_ci = np.percentile(bw, [2.5, 97.5]), np.percentile(be, [2.5, 97.5])
                cost = (np.where(rec[g], 0, miss[idx][g]) + np.where(rec[g], fr[idx][g], 0)).sum(1) + RR.QCOST * q[g]
                row[label] = {"wrong_negative": float(wrong_neg), "wrong_negative_count": [wn_num, wn_den],
                              "wrong_negative_ci": [float(x) for x in wn_ci],
                              "wrong_recommendation": float(wrong_rec),
                              "eligible_called_unlikely": float(missed_eligible),
                              "eligible_called_unlikely_count": [wn_num, me_den],
                              "eligible_called_unlikely_ci": [float(x) for x in me_ci],
                              "households_with_a_wrong_negative": int((hh_num > 0).sum()),
                              "cant_tell_per_household": float(flag[g].sum(1).mean()), "cost": float(w @ cost)}
                parts.append(f"{label}: 'doesn't look likely' wrong {wn_num}/{wn_den} = {wrong_neg:.2%} "
                             f"[{wn_ci[0]:.2%}, {wn_ci[1]:.2%}]; eligible programs called unlikely {wn_num}/{me_den} = "
                             f"{missed_eligible:.1%} [{me_ci[0]:.1%}, {me_ci[1]:.1%}] in {(hh_num > 0).sum()} households; "
                             f"can't-tell {flag[g].sum(1).mean():.2f}/household, cost ${w @ cost:,.0f}")
            out["groups"][uname][gname] = row
            lines.append(f"  {gname} ({g.sum()}):\n    " + "\n    ".join(parts))
    text = "\n".join(lines)
    print(text)
    json.dump(out, open(OUT, "w"), indent=1)
    open(OUT.replace(".json", ".txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
