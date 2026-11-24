# DecisionIntake

**An open-source decision engine for benefits screening: it picks the question most worth asking next,
stops when another question isn't worth the person's time, and says "we couldn't tell" instead of
guessing.**

- **Screener** (English and Spanish, ten US programs): https://decision-intake.vercel.app
- **MCP connector** for any AI assistant: `https://decision-intake.vercel.app/mcp`
- **Developer guide** (JSON API, MCP, examples in JS, Python and curl): https://decision-intake.vercel.app/docs.html
- **Walkthrough** (watch it interview a test household, with everything it believes on screen):
  https://decision-intake.vercel.app/walkthrough/

It covers SNAP, WIC, TANF, SSI, Medicaid, CHIP, the EITC, the refundable child tax credit, the ACA
premium tax credit and school meals, with 2026 rules from
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us). **It is a screening estimate, not an
eligibility decision,** and it has been tested on simulated households, not real applicants
([limits](#limits)).

## Try it in a minute

Add the hosted connector to Claude Code and ask it to screen a household:

```bash
claude mcp add --transport http decisionintake https://decision-intake.vercel.app/mcp
```

Or call the JSON API directly. Each response carries `state`, the facts so far; send it back with the
next answer. The server keeps nothing.

```bash
curl -s https://decision-intake.vercel.app/v1/step -H 'Content-Type: application/json' \
  -d '{"facts": [{"question": "state", "choice": "TX"}, {"question": "children", "choice": "2"},
                 {"question": "earned_income", "amount": {"value": 2400, "period": "month", "approximate": true}}]}'
# -> {"state": "di1;state=TX;children=2;earned_income=~2400/month;#....",
#     "readings": [... how each fact was read ...],
#     "next": {"action": "ask", "question": {...}, "why": {"worth_per_year": ..., "mostly_settles": [...]}}}
```

## What the tests show

Every test below was written down in [DECISIONS.md](DECISIONS.md) before it ran, on held-out
households, unless marked otherwise. The households are a calibrated synthetic population (57,240
households from PolicyEngine's [Populace](https://huggingface.co/datasets/policyengine/populace-us)
data, scored by PolicyEngine-US). "Cost of mistakes" is a dollar loss defined in advance: a program
the household qualifies for but isn't told about costs its yearly value, a wrong recommendation costs
$1,000, and each question costs $15 of the person's time.

**At a glance.** Cost is dollars a household a year (mistakes plus $15 a question; lower is better), on
held-out simulated households; intervals are 95%.

| question | comparison | households | registered | result |
|---|---|---|---|---|
| Does choosing questions beat a fixed form? | v2 against the best fixed 8-question form, same model | 2,000 | yes, D28 | **−$768** [−1,241, −398] |
| ... and a decision tree? | against the strongest tree we built, same model | 2,000 | yes, D28 | **−$293** [−485, −136] |
| Does asking 8.4 instead of 18 lose accuracy? | against the same model asking all 18 | 2,000 | secondary, D28 | −$79 [−188, +37], not distinguishable from zero |
| Does that hold with noisy answers? | the same comparisons with misreports and "don't know" | 2,000 | secondary, D28 | no question policy separates |
| Does it help an AI assistant? | Claude Haiku with the MCP tools against without | 200 | yes, D29 | mistakes 41% lower ($3,884 → $2,305); total **−$1,591** [−2,469, −775] unweighted; weighted −$294 [−1,831, +1,146] |
| Is that the question choice? | same predictor, Claude choosing its own questions | 200 | yes, D30 | **−$891** [−1,538, −324] unweighted; weighted −$516 [−1,441, +359] |
| Do the statutory rules cost anything? | rules on against off | 2,000 | yes, D32 | −$3.7 [−9.4, +0.0] |
| Does "can't tell" cut wrong negatives? | wrong "doesn't look likely" | 2,183 | yes with v2.1 (D35); after the result for the served model | 2.17% → 0.62% [−1.76, −1.35 points]; served: 2.17% → 0.71% |

**Choosing questions this way beats fixed forms and decision trees (D28, 2,000 households).** With the
same outcome model deciding, a screening averaged 8.4 questions:

| DecisionIntake v2 (8.4 questions) against | difference in cost a household a year [95% CI] |
|---|---|
| the best fixed 8-question form | **−$768** [−1,241, −398] |
| the strongest decision tree we could build (11.1 questions) | **−$293** [−485, −136] |
| the previous engine, v1 (8.5 questions) | **−$281** [−437, −142] |
| v1's full 18-question form | **−$483** [−667, −317] |
| the same model asking all 18 questions (secondary) | −$79 [−188, +37]: not distinguishable from zero |

All four registered comparisons hold after Holm correction (p < 0.002) and when whole source households
are resampled. Against asking everything, the difference was not statistically distinguishable from zero
(a non-significant result, not a test of equivalence). The
limit: these are clean answers. When simulated people misreport and say "don't know", no question
policy separates from another (D28, part 2).

**It makes an AI assistant better at the job (D29, D30, 200 households).** Claude Haiku interviewed
simulated people who answer in everyday sentences generated from templates ("somewhere between 25 and
30 thousand"), once on its own and once with this engine's MCP tools:

| per household | Claude alone | Claude + DecisionIntake |
|---|---|---|
| cost of mistakes | $3,884 | **$2,305** (41% lower) |
| eligible programs found | 51% | **68%** |
| share of recommendations that were right | 40% | 41% |
| told to see a caseworker | 60% | **20%** |

Total cost −$1,591 a household [−2,469, −775]. Holding the predictor equal and letting Claude pick its
own questions, it stopped after 3.6 and did $891 worse [−1,538, −324] (D30), so more than half the gain
is the question choice. Both results are unweighted; survey-weighted, neither is distinguishable from
zero (−$294 [−1,831, +1,146] and −$516 [−1,441, +359]). Most recommendations in both arms are wrong,
largely because the loss recommends a $10,000 program at a 9% chance; those are labelled "worth
checking", and the engine arm over-recommends Medicaid (71 households against 20).

**It says when it can't tell (D33, D35, 2,183 households).** Where fewer than 10 training households
match the answers, a negative is wrong 2.5% of the time, against 0.4% where they do. So there, a
program not recommended but still at 5% or more shows as "we couldn't tell, you can still apply". That
rule was fixed before the release test and passed its registered criterion: wrong "doesn't look likely"
answers 2.17% → 0.62%, about one label a household, with the retrained model that test was for. On the
model that serves, the same run gives 2.17% → 0.71% (read after the result).

**What did not work** is in the log too: v2.1, retrained after we found a bug in our own data pipeline,
failed its release test and does not ship (D35, D39); multicalibration made validation worse and was
dropped (D32); v1's linear model, not the question policy, was what held the first engine back (D28).

## Limits

- **Simulated, not real.** Every number above is agreement with PolicyEngine-US on a synthetic population,
  assuming everyone takes up what they are entitled to. Nobody has used it to apply yet.
- **A screening estimate, not an eligibility decision.** Only the program decides; the screener sends
  people to the official place to apply, and its "talk to someone" advice is advice, not a referral.
- **Known model limitation (D39).** The model that serves (v2) was trained before we fixed how
  self-employment from service businesses, partnerships and farms was coded (D34). On the 8% of
  households that fix touched, it calls an eligible program unlikely 7.8% of the time, against 3.9%
  elsewhere and 6.1% for the corrected model; the can't-tell layer is what brings it down from 19.6%.
  The corrected model was worse for everyone else on realistic answers, so it waits for a new test.
- **Households, not tax or SNAP units.** It asks about the household; households with several tax
  filers or unrelated adults are where its rules and predictions are weakest.
- **Not asked:** immigration status, which changes eligibility. **Not reviewed:** the Spanish, which is
  AI-assisted and labelled so on the page.
- **Its own error estimate** is 19% too high on clean answers and 28% too low with misreports (D28).

## How it works

```
person's words ──► an LLM, or a form ──► typed facts (option, amount + period, "don't know")
                                               │
                                               ▼
                  belief: weights over 48,657 households that could be this person
                                               │
              a neural outcome model scores each candidate (P(eligible) and dollars, ten programs)
              statutory limits cap it where it would extrapolate (EITC phase-out, no earnings)
                                               │
                                               ▼
           value of information: the expected drop in the cost of mistakes, for every question
                                               │
               ASK the best question while it is worth more than a question costs
               FINISH with "likely" / "worth checking" programs and "can't tell" where data is thin
               HAND OFF (advise a navigator) when the doubt left is still costly
```

- **Language in, arithmetic out.** An LLM fills typed slots; code turns "about $2,900 a month" into
  evidence over yearly ranges. That cut confidently wrong readings from 10% to 0.8% against letting the
  LLM pick a range (D26).
- **A belief over real-data households,** updated through each question's model of how people misreport,
  with a support floor so it can't become certain by ruling almost everyone out (D12).
- **A wide-and-deep network in plain numpy:** two ReLU layers shared by all ten programs next to
  hand-listed linear terms, an eligibility head and a dollars head, out-of-fold Platt scaling. Trains in
  about 3 minutes on a CPU (D28).
- **Rules it may not break** (`bundles/guards_2026.json`), each checked against PolicyEngine's own results
  before shipping. They cost nothing on 2,000 fresh households (−$3.7 [−9.4, +0.0], registered, D32).
  Tracing every exception through PolicyEngine tax unit by tax unit explained 108 of 109, and found the
  bug in our own pipeline that D34 fixed ([contradiction ledger](results/contradiction_ledger.md)).

## Three ways to run it

**Hosted** (https://decision-intake.vercel.app). The screener, the JSON API (`/v1/questions`,
`/v1/step`, `/v1/assessment`, described at `/openapi.json`) and MCP at `/mcp`: the stateless 2026-07-28
revision and the 2025 handshake revisions, three tools (`start_intake`, `record_facts`,
`get_assessment`) that pass a `state` string. About 300 ms of server time a step. No database, no
sessions, no answers in logs. `/healthz` reports the commit and the model and rules hashes; a GitHub
Actions job checks the page, a whole screening, the deployed model and rules, and both MCP eras after every
deploy and on a 30-minute schedule. Per IP, 120 requests a minute on the API and MCP routes.

**Local**, from a clone (not on PyPI; the model file ships in the repo, 1.5 MB, numpy only):

```bash
pip install -e .
python -m decisionintake.web --port 8080         # screener, API and MCP, as hosted (~110 ms a step)
claude mcp add decisionintake -- python -m decisionintake.mcp_server   # MCP over stdio, four tools
```

The stdio server keeps sessions in memory and adds a `next_step` tool; the hosted one folds it into
`record_facts`.

**In the browser.** The walkthrough runs the same engine and network in JavaScript on a 1,500-household
resample of the training data and plays held-out households through it, so you can watch the belief and
the value of every question change after each answer.

The screener has one question per screen, labelled controls, focus moved to each new question, and no
color-only meaning; axe-core finds no WCAG 2.1 A/AA violations on any screen in light or dark mode.

## Reproduce

```bash
# 1. environment (Python 3.11+; tested on 3.13 with numpy 2.5.5 and policyengine-us 2.152.15)
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]" policyengine-us==2.152.15 tables h5py
pytest -q                                              # 92 tests, on the shipped model; no data needed

# 2. data: Populace national build populace-us-2024-spm-20260915 (827 MB, sha256 6496cc43...aee84) into
#    data/raw/populace_us_2024.h5, then the outcome table (about 10 minutes on 12 cores)
bash adapters/benefits/build_all.sh 30 5

# 3. tuning on the validation split only, then the registered test (D28; WORKERS processes, ~1.5 GB each)
python experiments/outcome_v2.py && python experiments/val_v2.py 400 WORKERS
python experiments/run_v2.py 2000 WORKERS && python experiments/run_v2.py 2000 1 analyse

# 4. the shipped model and rules, the agent test (D29; needs Claude Code), the release test (D35)
python experiments/export_bundle.py bundles/benefits_v2.npz
python experiments/guards_build.py
python experiments/agent_e2e.py prep test 200 && python experiments/agent_e2e.py run test engine haiku 200 3
python experiments/agent_e2e.py run test alone haiku 200 3 && python experiments/agent_e2e.py analyse test haiku
python experiments/run_release.py WORKERS && python experiments/run_release.py WORKERS analyse
```

Every experiment has its script in `experiments/`, named in DECISIONS.md; seeds are fixed, so a rerun
gives the same households and draws. On Windows, PolicyEngine's longest file path exceeds 260
characters unless the package sits under a short prefix: copy it to one and set `PE_SHORT_PATH` (D11).

## How it was built

AI agents (Claude, with reviews by GPT) wrote the code and ran the experiments under Ege Hazar's
direction. Each test's pass/fail rule was written before it ran and is followed even when it goes
against us (v2.1 is not shipped because its rule failed). The [decision log](DECISIONS.md) has every
choice, result and dead end; the [model card](MODEL_CARD.md) has data, tests and what it must not be
used for; [DEPLOY.md](DEPLOY.md) has the production runbook. MIT licensed.
