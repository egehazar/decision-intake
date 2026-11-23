# Contradiction ledger (D32)

Every training household a candidate rule matches but where PolicyEngine-US pays the credit, traced by tax unit (experiments/contradiction_ledger.py).

| rule | shipped | households matched | exceptions | dependants' wages | other earned income | pre-tax deductions | another tax unit | unexplained |
|---|---|---|---|---|---|---|---|---|
| eitc_income_joint_0 | yes | 3962 | 1 | 0 | 0 | 1 | 0 | 0 |
| eitc_income_joint_1 | yes | 1743 | 6 | 0 | 0 | 6 | 0 | 0 |
| eitc_income_joint_2 | yes | 1969 | 1 | 0 | 0 | 0 | 0 | 1 |
| eitc_income_joint_3 | yes | 789 | 2 | 2 | 0 | 0 | 0 | 0 |
| eitc_income_joint_4 | yes | 250 | 2 | 2 | 0 | 0 | 0 | 0 |
| eitc_income_single_0 | yes | 2688 | 3 | 0 | 0 | 3 | 0 | 0 |
| eitc_income_single_1 | yes | 161 | 4 | 3 | 0 | 1 | 0 | 0 |
| eitc_income_single_2 | yes | 113 | 8 | 5 | 0 | 3 | 0 | 0 |
| eitc_income_single_3 | yes | 19 | 2 | 2 | 0 | 0 | 0 | 0 |
| no_earnings_joint_0 | yes | 2960 | 1 | 0 | 1 | 0 | 0 | 0 |
| no_earnings_joint_4 | yes | 9 | 5 | 0 | 5 | 0 | 0 | 0 |
| no_earnings_single_0 | yes | 1614 | 4 | 0 | 4 | 0 | 0 | 0 |
| no_earnings_single_1 | yes | 18 | 6 | 0 | 6 | 0 | 0 | 0 |
| no_earnings_single_4 | yes | 6 | 6 | 0 | 6 | 0 | 0 | 0 |
| eitc_income_single_4 | no | 6 | 2 | 2 | 0 | 0 | 0 | 0 |
| no_earnings_joint_1 | no | 39 | 5 | 0 | 5 | 0 | 0 | 0 |
| no_earnings_joint_2 | no | 51 | 21 | 0 | 21 | 0 | 0 | 0 |
| no_earnings_joint_3 | no | 21 | 13 | 0 | 13 | 0 | 0 | 0 |
| no_earnings_single_2 | no | 14 | 8 | 0 | 8 | 0 | 0 | 0 |
| no_earnings_single_3 | no | 8 | 9 | 0 | 9 | 0 | 0 | 0 |

Other earned income, by largest source: sstb_self_employment_income 73, partnership_self_employment_net_earnings 5

## What it says

- **The shipped rules (14):** 51 exceptions among 16,301 matched households (0.31%).
  - 22 are self-employment the data coded as "no earnings, not self-employed" (below): a bug in the
    adapter, not in the rules.
  - 14 are a dependant's wages, which count in the household's earnings but not in the filer's earned
    income (26 U.S.C. 32(c)(2)). The rules reason about the household, the credit about the filer.
  - 14 are pre-tax deductions: the credit uses taxable wages (W-2 box 1), so 401(k)-style contributions can
    bring gross earnings above the limit back under it.
  - 1 is unexplained.
- **The dropped rules (6):** all 58 exceptions are the same coding bug. In PolicyEngine-US, self-employment
  income from a specified service business (`sstb_self_employment_income`, 73 of the 78 traced cases),
  partnership earnings and farm income are separate from `self_employment_income`. The adapter counted only
  the last in "earnings from work" and "anyone self-employed", so these households look like they have no
  earnings while their EITC is computed on real earnings. A person in that position would answer that they
  are self-employed and earn money. Fixed in the adapter (DECISIONS D34): the rules are re-checked on the
  corrected data.
