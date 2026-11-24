# Model card: DecisionIntake, US benefits screening

What it is, what it was tested on, where it is wrong, and what it must not be used for. Decision
numbers (D28, ...) point to [DECISIONS.md](DECISIONS.md). A result marked **registered** had its
pass/fail rule written there before it ran; anything else is marked as such.

| | |
|---|---|
| version | engine 1.1.0; model v3 (`bundles/benefits_v3.npz`, sha256 `527ffdb595b0…`) = v2's network unchanged (`bundles/benefits_v2.npz`, sha256 `f012a9a0e5ce…`) plus an add-on network for five programmes (D43); rules `bundles/guards_2026.json` (14 rules, canonical-JSON sha256 `0cbf5e1845fc…`); `/healthz` reports the hashes that are deployed |
| rules year | 2026, from PolicyEngine-US 2.152.15 |
| last updated | 2026-11-24 (D43) |
| contact | [GitHub issues](https://github.com/egehazar/decision-intake/issues) |

## What it does

Given what a person has said about their household, the engine picks the next question worth asking
(or stops), and estimates for fifteen US programs how likely the household is to qualify and what each is
worth a year. It recommends a program when applying is worth the expected cost of a wrong
recommendation, says "can't tell" where too few households like the person are on record, and advises
that a person review the case when the doubt left over is costly.

Programs: SNAP, WIC, TANF, SSI, Medicaid, CHIP, the Earned Income Tax Credit, the refundable part of the
Child Tax Credit, the ACA premium tax credit, and free or reduced-price school meals; since D43 also state
earned income tax credits, state child tax credits, Head Start, Early Head Start and housing assistance
(vouchers and public housing, where waiting lists are often long).

**It is a screener, not an eligibility decision.** Only the program decides. Immigration status is not
asked and can change eligibility.

## Parts

| part | what | where |
|---|---|---|
| belief | 48,657 households, re-weighted after every answer; a support floor keeps at least 10 effective households behind every estimate | `core.py` |
| outcome model | wide-and-deep network in plain numpy (512/256 hidden units, dropout 0.4, weight decay 0.3, out-of-fold Platt scaling), ten programs at once | `neural.py`, D28 |
| add-on network | the same architecture and settings, for the five D43 programmes, trained on the corrected answers; it sits beside v2's network and does not change it | `bundle.StackedOutcome`, D43 |
| rules | 14 statutory limits on top of the model (2026 EITC phase-out ends; no EITC or refundable CTC without earnings), each checked against PolicyEngine's results before shipping | `guards.py`, D32 |
| question choice | expected value of information per question, with pair look-ahead; a question must save more than $15 a year to be asked | `policies.py` |
| decision rule | a wrong recommendation costs $1,000; a missed program costs its yearly value (at least $120); advise a person above $2,500 of expected mistakes | `core.py`, `service.py` |
| can't tell | where fewer than 10 training households fit the answers, a program not recommended but at 5% or more is "can't tell", never "unlikely" | `service.py`, D33 |

## Data

- Rules: [PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) 2.152.15, tax and benefit year 2026.
- Households: PolicyEngine's [Populace](https://huggingface.co/datasets/policyengine/populace-us) 2024
  (`populace-us-2024-spm-20260915`, MIT licence), a synthetic population calibrated to official
  statistics. Not real people. 57,240 households; 48,657 train the model, 8,583 are held out by source
  household. **Every held-out household has now been used by a test**, so the next model's release test
  needs new data (D39).
- Labels: PolicyEngine's computed eligibility and value per program, with take-up assumed.

## Known limitation: v2 was trained on a coding error (D34, D39)

The contradiction ledger found that this project's adapter coded self-employment from service businesses,
partnerships and farms (which PolicyEngine keeps in separate variables) as no earnings. It changed the
earnings of 8.4% of households and the self-employed answer of 7.1% (in the question brackets, 2,391, 3,623
and, through earnings, 1,831 other-income answers of 57,240). v2 was trained before the fix;
test households are scored on the corrected answers, as a person would give them.

What it costs, on the 144 of D35's 2,183 households the fix touched (clean answers, after D35, not
registered). Rates are per program decision (each household has ten), with 95% intervals from resampling
whole households; the subgroup is small, so read the counts:

| on those households | wrong "doesn't look likely" | eligible programs called unlikely |
|---|---|---|
| v2 without the can't-tell layer | 35 of 1,166: 3.00% [1.68, 4.73] | 35 of 179: 19.6% [11.5, 28.4] |
| **v2 as served** | **14 of 999: 1.40% [0.59, 2.47]** | **14 of 179: 7.8% [3.6, 12.8], in 11 households** |
| v2.1, retrained on the corrected answers | 11 of 986: 1.12% [0.40, 2.16] | 11 of 179: 6.1% [2.3, 10.8], in 8 households |
| (v2 as served, every other household) | 96 of 14,585: 0.66% [0.51, 0.82] | 96 of 2,486: 3.9% [3.0, 4.8] |

These households are harder for both models; the served and corrected models differ by 3 decisions here,
which is within the noise.

v2.1 is not served: it failed its registered release test (D35) and was worse on realistic answers for
everyone else ($2,117 against $1,863 a household). A guard for the self-employed did not help (33 of 3,213
wrong negatives to 31 of 3,199) and is not used. All of these figures are in `results/affected_d35.txt`. The next candidate needs that weakness explained and a test on new data.

## How well it does

| test | registered? | result |
|---|---|---|
| D28, 2,000 households, clean answers | yes | 8.4 questions. Against the best fixed 8-question form −$768 a household a year [−1,241, −398], the strongest decision tree −$293 [−485, −136], the previous engine −$281, v1's 18-question form −$483 (Holm p < 0.002 each). Against the same model asking all 18: −$79 [−188, +37], no detectable difference |
| D28, realistic answers (misreports, "don't know") | yes | no question policy separates from another |
| D28, tiers | descriptive | "likely" calls right 78% of the time; "worth checking" 28%; the least-sure 20% of households carry 45% of the mistake cost; the engine's estimate of its own mistakes is 19% high on clean answers, 28% low with misreports |
| D29, Claude Haiku interviewing 200 households in templated everyday sentences | yes | mistakes $2,305 with the engine against $3,884 alone (−41%; total −$1,591 [−2,469, −775]); survey-weighted −$294 [−1,831, +1,146], not significant. Programs found 68% against 51% at the same precision, 41% against 40% (most recommendations are wrong in both arms; the engine arm wrongly recommends Medicaid in 71 households against 20). Told to see a caseworker 20% against 60% |
| D30, Claude given the predictor but choosing its own questions | yes | the engine's question choice is worth −$891 a household [−1,538, −324] unweighted; −$516 [−1,441, +359] weighted |
| D31b, Claude through the stateless hosted tools, 100 development households | yes, on development households | as good as sessions in memory: −$49 [−286, +137]; 2 miscopied states refused by the checksum and resent |
| D32, 2,000 households, the rules on and off | yes | costs nothing: −$3.7 [−9.4, +0.0] clean, −$15 [−33, −1] realistic; with clean answers no eligible program suppressed |
| D35, 2,183 households, the release test | yes | the can't-tell layer passed: wrong "doesn't look likely" 2.17% → 0.62% with v2.1 [−1.76, −1.35 points], 1.06 labels a household. v2.1 failed its cost criterion (−$90 [−288, +59] against a +$25 margin sized too tightly) and does not ship |
| D35, the same run, the model that serves | no, read after the result | wrong "doesn't look likely" 2.17% → 0.71% [−1.66, −1.27], 1.05 labels a household, no recommendation changed. Per program (release arm): EITC 5.0% → 0.8%, refundable CTC 2.8% → 0.3%, SNAP 3.6% → 1.3% |
| D51, 991 held-out households that get SNAP | yes | receipt of SNAP as evidence (likelihood 1 for households the rules model makes SNAP-eligible, 0.2 for the rest): with five fields, mistakes on the other 14 programmes −$1,600 [−2,402, −895] a household a year (4,620 against 6,221); with all 18 answers, no decision changes. Served since engine 1.1.0 |
| D43, 2,000 held-out households, five programmes added | yes (labels for the five never computed before) | five added: −$1,144 [−1,797, −627]; original ten: −$38 [−122, +23] against a +$25 margin, both pass (Holm). Of the five, eligible programmes found 67.6% against 52.7% for v2 plus a simple rule; 9.2 questions against 8.4. Realistic answers: −$771 [−1,165, −432] on the five, no detectable difference on the ten |
| D38, Claude through the public connector, 20 development households | no | 20 of 20 finished; recall 73%, precision 38%, against 73% and 36% in memory |

Not yet measured: results by state or household type, and anything with real applicants.

On validation households in clear-cut groups (household make-up by income cells where at least 30
training households all agree), the model is on the wrong side of 50% for 0.1% to 2.4% of them,
depending on the program (D32). On the training population its mean predicted probability is within
0.005 of PolicyEngine's rate for every program, a sanity check rather than evidence of accuracy.

## Where it is wrong

- **Thin support.** After about 8 answers most people are matched by fewer than 10 training households,
  and there the model's negatives are wrong several times as often (2.5% against 0.4%, D35). That is
  what the can't-tell layer is for.
- **Unusual answer combinations.** Where few households look like the person, the network extrapolates.
  Before the rules file, a lone parent with no earnings at all, or earning $78,000, got the EITC as
  "likely". Across 50,000 random valid profiles, v2 called a statutorily ruled-out EITC likely in 4.5%
  and a ruled-out refundable child credit in 20.4%; the rules bring both to 0% (D32). Rules exist only
  where they hold in PolicyEngine's own results; six candidates failed that check. When answers are
  misreported a rule can fire wrongly: on 2,000 households with realistic answers the rules suppressed 8
  recommendations the household should have had, and removed 23 wrong ones.
- **Confident misses.** Among the self-employed, the wrong negatives that remain mostly sit under a 5%
  probability, where no caution layer reaches them (D39).
- **Household, not tax or SNAP unit.** Questions are about the household. Households with several tax
  units (an adult child filing separately, unmarried partners, roommates) are where it is weakest.
- **What it doesn't ask.** Immigration status, citizenship, work requirements, sanctions, resources
  beyond bank accounts, and state rules PolicyEngine does not model.
- **One year of rules.** 2026. A new year needs a new population build, model and registered test.

## Use

Intended: a first screen that tells a person which programs are worth applying for, inside a benefits
navigator's tool, a screener site, or an AI assistant (through MCP), with a person to turn to.

Not intended: deciding eligibility, denying anyone help, ranking or scoring people, or any use where a
"not recommended" or "can't tell" stops someone from applying. Integrations must show the `note` with
every result and never present a dollar range as an amount someone will receive.

## Privacy

The service keeps nothing. Each call carries the facts so far as a short string (`state`); the service
rebuilds the belief from it, answers, and forgets it. No database, no sessions, no answers or states in
logs (route, status and timing only), no cookies, no analytics. The public screener keeps the answers in
the browser tab (sessionStorage) so a person can go back and change them; "Clear my answers" or closing
the tab removes them.

## Accessibility

axe-core 4.10.2 finds no WCAG 2.1 A/AA violations on any screen of the public screener, in light and dark
mode; a keyboard-only pass and a pass over the accessibility tree found no blockers. These are automated
and scripted checks, not a test with screen-reader users.
