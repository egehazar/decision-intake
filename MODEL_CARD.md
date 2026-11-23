# Model card: DecisionIntake, US benefits screening (v2 + 2026 rules)

What this is, what it was tested on, where it is wrong, and what it must not be used for. Every
number links to a decision record in [DECISIONS.md](DECISIONS.md), each registered before its test ran.

## What it does

Given what a person has said about their household, the engine picks the next question worth asking
(or stops), and estimates for ten US programmes how likely the household is to qualify and what each
is worth a year. It recommends a programme when applying is worth the expected cost of a wrong
recommendation, and hands the case to a person when the doubt left over is costly.

Programmes: SNAP, WIC, TANF, SSI, Medicaid, CHIP, the Earned Income Tax Credit, the refundable part of
the Child Tax Credit, the ACA premium tax credit, and free or reduced-price school meals.

**It is a screener, not an eligibility decision.** Only the programme decides. Immigration status is
not asked and can change eligibility.

## Parts

| part | what | where |
|---|---|---|
| belief | 48,657 households, re-weighted after every answer; a support floor keeps at least 10 effective households behind every estimate | `core.py` |
| outcome model | wide-and-deep network in plain numpy (512/256 hidden units, dropout 0.4, weight decay 0.3, out-of-fold Platt scaling), ten programmes at once | `neural.py`, D28 |
| rules | 14 statutory limits on top of the model (2026 EITC phase-out ends; no EITC or refundable CTC without earnings), each checked against PolicyEngine's results before shipping | `guards.py`, `bundles/guards_2026.json`, D32 |
| question choice | expected value of information per question, with pair look-ahead; a question must save more than $15 a year to be asked | `policies.py` |
| decision rule | a wrong recommendation costs $1,000; a missed programme costs its yearly value (at least $120); hand off above $2,500 of expected mistakes | `core.py`, `service.py` |

## Data

- Rules: [PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) 2.152.15, tax and benefit year 2026.
- Households: PolicyEngine's [Populace](https://huggingface.co/datasets/policyengine/populace-us) 2024
  (`populace-us-2024-spm-20260915`, MIT licence), a synthetic population calibrated to official
  statistics. Not real people. 57,240 households; 48,657 train the model, 8,583 are held out by source
  household and drawn on only by registered tests.
- Labels: PolicyEngine's computed eligibility and value per programme, with take-up assumed.

## How well it does

All on held-out households the model never saw, registered before running:

| test | result |
|---|---|
| D28, 2,000 households, crisp answers | 8.4 questions; $1,017 of mistakes a household against $1,356 for the full 18-question form (−$339 [−523, −173]); beats the strongest decision tree (−$293) and the best fixed 8-question form (−$768); matches its own 18-question form |
| D28, "likely" tier | 78% of "likely" calls are right; the least-sure 20% of households carry 45% of the mistake cost |
| D29, Claude Haiku interviewing 200 households in everyday sentences | 41% fewer costly mistakes with the engine ($2,305 against $3,884); recall 68% against 51% at the same precision; hand-offs 20% against 60% |
| D30, same, predictor given to Claude but not the question choice | the engine's question choice alone is worth −$891 a household [−1,538, −324] |

Calibration on the training population (mean predicted probability against PolicyEngine's rate,
survey-weighted) is within 0.005 for every programme. On validation households in clear-cut groups
(household make-up by income cells where at least 30 training households all agree), the model is on
the wrong side of 50% for 0.1% to 2.4% of them, depending on the programme (D32).

## Where it is wrong

- **Unusual answer combinations.** Where few households look like the person, the network
  extrapolates. Before the rules file, a lone parent with no earnings at all, or earning $78,000, got
  the EITC as "likely". Across 50,000 random valid profiles, v2 called a statutorily ruled-out EITC
  likely in 4.5% and a ruled-out refundable child credit in 20.4%; the rules bring both to 0% and
  change nothing else (D32). Rules exist only where they hold in PolicyEngine's own results; six
  candidate rules failed that check and are not used.
- **Messy answers.** With simulated realistic answers (people who misremember, volunteer, or don't
  know), the engine and the alternatives do not separate (D28 secondary). The AI-assistant tests
  (D29, D30) use everyday-sentence answers and do separate.
- **Household, not tax unit.** Questions are about the household. Households with several tax units
  (an adult child filing separately, unmarried partners) are where the tax credits are hardest to call.
- **What it doesn't ask.** Immigration status, citizenship, work requirements, sanctions, resources
  beyond bank accounts, and state programme rules that PolicyEngine does not model.
- **One year of rules.** 2026 rules. A new rules year needs a new population build, a new model and a
  new registered test.

## Use

Intended: a first screen that tells a person which programmes are worth applying for, inside a benefits
navigator's tool, a screener site, or an AI assistant (through MCP), with hand-offs to a person.

Not intended: deciding eligibility, denying anyone help, ranking or scoring people, or any use where a
"not recommended" stops someone from applying.

## Privacy

The hosted service keeps nothing. Each call carries the facts so far as a short string (`state`), the
service rebuilds the belief from it, answers, and forgets it. No database, no sessions, no answers or
states in logs (route, status and timing only), no cookies, no analytics. Checked against the agent
tests in D31.
