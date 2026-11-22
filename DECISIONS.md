# Decision log

Every design choice in DecisionIntake, why it was made, and what would change it. Written so Ege can
explain and defend each one. AI agents (Claude, with reviews from Fable and GPT) did the hands-on work;
Ege directed it.

## D1. What the project is (2026-11-22)

A small engine that tells an AI assistant **what to ask next, which answers to trust, and when it knows
enough to act or should hand off to a person**. US public-benefits screening is the first real use.

- Rejected first: "beat the BeNYfits benchmark". It's benchmarking again, and as a product it would be
  a worse copy of the official NYC screener.
- Rejected: a consumer benefits app. MyFriendBen and others already run real screeners on PolicyEngine
  (an Urban Institute evaluation covered ~100,000 sessions). A prettier screener isn't an achievement.
- Kept: the part those products don't have, a principled way to choose questions and to stop.

## D2. Rules come from PolicyEngine-US, data from Populace (2026-11-22)

- We write no eligibility rules. PolicyEngine-US is maintained by professionals and used in production.
- The household population is **Populace** (`policyengine/populace-us`, build
  `populace-us-2024-spm-20260915`, sha256 `6496cc43…aee84`, checked). It replaced the archived
  `policyengine-us-data` in June 2026.
- Honest caveat: Populace is a **calibrated synthetic population** built from CPS, IRS, SIPP and SCF,
  with imputation and geographic copies of households. It isn't a set of real, fully observed households.
  Every result is "agreement with PolicyEngine on a synthetic population", not real-world accuracy.
- Copies of one source household must not land on both sides of a train/test split, or results look
  better than they are. The file has no lineage column, so we group records by their non-geographic
  attributes and split by group (`split_by_group`).

## D3. The belief is a weighted population (particles) (2026-11-22)

- Each record in the training population is a "particle" with known answers to every question and known
  results for every programme (precomputed once with PolicyEngine). A conversation only reweights them,
  so each turn is fast array maths.
- Known weakness (raised by GPT): an unusual household can leave very few matching particles, and the
  belief then looks confident only because almost everything was eliminated. So the engine reports
  **effective sample size** and must hand off when it collapses, rather than trust its own numbers.

## D4. The objective is decision loss, not "dollars found" (2026-11-22)

- GPT's catch: if success is "share of eligible dollars found", recommending every programme scores
  perfectly. So each programme has two costs: **missing** one the household can get (its value) and
  **recommending** one it can't (a fixed cost for wasted time and a refusal).
- Results are always reported as several numbers side by side: dollars found, programmes found, wrong
  recommendations, questions asked, households with any miss, worst-served groups. Not one score.

## D5. How the next question is chosen (2026-11-22)

- Value of information: for each question, the expected drop in decision loss after hearing the answer,
  over everyone the belief still allows. Ask the best one while it's worth more than a question costs.
- This isn't new maths. What we have to show is that it works in a real assistant: with messy language,
  facts volunteered out of order, and a reason to stop or hand off.

## D6. The baselines, and an honest prediction (2026-11-22)

- Baselines: the best fixed form (one order for everyone), entropy-greedy (reduces uncertainty, blind to
  stakes), a **decision tree trained on the same loss**, and the full questionnaire.
- Prediction written down before any result: on perfectly clean answers, our engine and the trained tree
  should be close, because a greedy tree on this loss *is* the same question rule, computed in advance.
  If the engine wins, it should win where a fixed tree can't adapt: facts volunteered in the first
  sentence, answers that are uncertain or wrong, and "I don't know".
- GPT's go/no-go bar: at least 25% fewer questions than the tree at the same quality. If the engine
  doesn't clear it, we publish that.

## D7. No overclaiming of guarantees (2026-11-22)

- Fable proposed a formal guarantee on missed dollars. GPT showed it doesn't hold as stated (the
  conditions behind it don't fit a system that picks questions as it goes). We report measured stopping
  behaviour on held-out households instead, and claim nothing we can't show.

## D8. Amounts mean entitlement, under 2026 rules (2026-11-22)

- The dataset carries "takes up if eligible" flags, so PolicyEngine's normal amounts are what people
  *claim*. A screener cares what they're *entitled to*, so every take-up flag is set to true and we use
  the `*_if_takes_up` amounts. Rules year is 2026; the 2024 data is uprated by PolicyEngine itself.
- A programme counts as "eligible" from $120 a year ($10 a month). Below that a recommendation helps nobody.

## D9. How Medicaid is valued (2026-11-22)

- PolicyEngine's per-person Medicaid cost divides each state's total spending by a total computed over
  *the simulated sample*. In a national run that's fine. On a sample it explodes (we saw $10^13). So we
  value each eligible person at PolicyEngine's own per-capita cost for their group (MACPAC: about $3.5k
  for a child, $7k for an expansion adult, $21k aged or disabled). Health coverage is then worth real
  dollars in the loss, and dominates dollar totals; that's why results are also reported per programme.

## D10. The ten programmes and eighteen questions (2026-11-22)

- Programmes: SNAP, WIC, TANF, SSI, Medicaid, CHIP, EITC, refundable CTC, ACA premium tax credit,
  free/reduced school meals. Federal or nationwide, all computed by PolicyEngine.
- Questions are ones a person can answer in a sentence, numbers in ranges: state, household size,
  children, child under 5, pregnancy or baby, anyone 65+, disability, married couple, earnings, self-
  employment, Social Security, other income, rent or mortgage, rent/own, childcare, employer insurance,
  savings, college student.
- Never asked: immigration status. It changes eligibility, so it stays an unasked source of error, and
  results are reported separately for households with a non-citizen.
- Each question has a report-error model (income: 10% chance of naming the next range up or down). The
  engine uses it to weigh answers; the "realistic" simulated user draws answers from it.

## D11. Engineering notes (2026-11-22)

- Windows' 260-character path limit breaks PolicyEngine (its longest file needs a short install prefix).
  We run against a copy of the package at `C:\Users\egeha\pe` rather than change a system setting.
- The value of every question is computed with one matrix product per turn (a one-hot answer matrix
  times the weighted losses), then each question's error model is applied to small per-option totals.
  A test checks this equals the direct computation.
