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

## D12. Tuning round 1: the support floor, and look-ahead that didn't pay (2026-11-22)

Validation split only (400 households), clean answers, wrong-recommendation cost $300:

| setting | questions | actual loss | engine's own estimate |
|---|---|---|---|
| ask all 18, no floor | 18 | $3,419 | $216 |
| ask all 18, floor 10 | 18 | $965 | $971 |
| fixed form 8, floor 30 | 8 | $1,136 | $1,335 |
| engine, floor 30 | 8.6 | $995 | $1,054 |
| engine, no floor | 5.7 | $3,049 | $184 |

- Without a floor the belief collapses onto a few households and the engine believes it is 16 times
  more certain than it is. The floor fixes both the loss and the self-assessment. This is the first
  result worth showing: an engine that knows how much it doesn't know.
- The floor is crude: it softens all evidence equally, so extra answers can make things worse
  (engine at a lower question cost asked 13 questions and did worse than at 9). A better fix would
  soften unimportant answers more than important ones. Not done yet.
- Pair look-ahead asked one more question for no gain in this domain. It's kept off by default and
  reported as a negative result; it does matter when facts only count together (tested on a toy case).
- With a $300 cost for a wrong recommendation, 70% of recommendations were wrong: the loss told the
  engine to list anything with a few percent chance. That cost was too low. Round 2 tries $1,000 and
  $3,000 and reports the trade-off rather than picking one silently.

## D13. The real ceiling problem, and the fix: smoothed outcomes (2026-11-22)

Round 2 (wrong-recommendation cost $1,000 and $3,000) showed that even with all 18 answers, decisions
were poor: about 69% of programmes found with more than half of recommendations wrong. No question
policy can beat that ceiling, and the engine did not beat the fixed form. The trained tree was worse
still (it stopped after ~3.5 questions; multi-way splits like 51 states leave test households in
branches with a handful of training records).

First guess, wrong: income brackets too coarse. Adding income as a percentage of the poverty line,
with brackets at real programme cutoffs, changed nothing.

The actual cause: the belief turned answers into a prediction by averaging the raw yes/no outcomes of
the few training households matching all 18 answers, a noisy nearest-neighbour vote that also matches
on answers that don't matter for a given programme. Per-programme ranking (AUC, all answers given):
SNAP 0.82, Medicaid 0.85, SSI 0.82, ACA credit 0.72.

Fix: keep the belief over answers, but replace each training household's raw outcome with a learned
estimate of P(eligible | its answers) and of the dollars if eligible (regularised logistic and ridge
regression on one-hot answers plus income interactions; out-of-fold, so no household's estimate saw
its own label). This is Rao-Blackwellization. Ranking with the same answers: SNAP 0.95, Medicaid
0.94, SSI 0.92, CHIP 0.94, EITC 0.94, TANF 0.99, ACA credit 0.84, the rest 0.98-0.99.
Scoring always uses each test household's true PolicyEngine outcome, never the smoothed one.

## D14. Completion: score candidates on the person's own answers (2026-11-22)

Smoothing raised the model's ranking (D13) but the engine still lost most of it: deciding straight
from the model with all answers known gave $1,507 loss and 81% of programmes found, while the
engine asking all 18 questions gave $2,557 and 66%. The belief can only weight training households
that roughly match the person; in 18 dimensions none match exactly, and each neighbour's smoothed
outcome was computed at *its own* answers (another state, another income bracket).

Fix: re-score every candidate household on a hybrid, the person's answers for every question
answered so far and the household's own answers only where the person hasn't answered. Training
households now only fill in unknowns. The outcome model is a sum of indicator terms, so re-scoring is
a table lookup; the pre-answer scores are cached and only households still carrying weight are
re-scored (0.12 s per full interview, 0.29 s with the engine choosing questions).

Asking all 18 on validation, wrong-recommendation cost $1,000: $2,257 -> $1,673 loss, 61% -> 80% of
programmes found, 51% -> 42% of recommendations wrong. Within reach of the model's own ceiling.

## D15. Round 4: the engine was overconfident, so it stopped early (2026-11-22)

With completion, on validation ($1,000 per wrong recommendation): asking all 18 gave $1,679 loss;
a fixed 6- or 9-question form about $1,840; the engine stopped after about 6 questions at $2,230 to
$2,310, worse than the fixed form. It believed its remaining loss was about $1,200 when it was about
$2,300, so it stopped too soon.

Two causes, two fixes, both in smoothing.py:
- Probabilities too sharp: during interviews households are re-scored by a model fitted on those same
  households. Platt scaling, fitted on out-of-fold predictions, is folded into the model. Validation
  reliability afterwards: SNAP 0.62 predicted -> 0.61 observed, Medicaid 0.80 -> 0.84, school meals
  0.82 -> 0.85 (rare programmes stay noisy in their top bins).
- Dollars biased low: a log-scale fit estimates medians (Medicaid $14.9k vs $16.4k true). A smearing
  correction then overshot by up to 40% (SNAP $4.9k vs $3.8k). Fitting dollars on the dollar scale
  is unbiased by construction: now within about 5% for most programmes.

Then a cleaner test of question choice alone: engine and fixed form each ask exactly k questions
(experiments/k_compare.py), so the stopping rule is not mixed in.

## D16. Question choice alone, and the settings frozen before the test run (2026-11-22)

Validation, 400 households, $1,000 per wrong recommendation, engine and fixed form each asking
exactly k questions (support floor 10):

| k | engine | fixed form |
|---|---|---|
| 3 | $2,477 | $2,716 |
| 5 | $2,062 | $2,046 |
| 7 | $1,724 | $2,295 |
| 9 | $1,685 | $1,756 |
| 12 | $1,635 | $1,544 |

With its own stopping rule (question cost $5), the engine asked 7.4 questions for $1,604, as good as
asking all 18 ($1,684). With realistic answers it was also ahead in the middle range (k=7: $1,830 vs
$2,255). The fixed form's own curve isn't smooth (k=7 worse than k=5), so differences of a few hundred
dollars on 400 households are noise; the test run uses more households and paired intervals.

Frozen for the test run (experiments/run_final.py), chosen on validation only: wrong-recommendation
cost $1,000; support floor 10; question costs $50/$15/$5 for the engine's own stopping; tree minimum
leaf 10; k in {3, 5, 7, 9, 12}; outcome model trained on train + validation. The test split is
scored once; any later change is reported as a change, with both results.

## D17. Test run cut to 600 households, before any result was looked at (2026-11-22)

The 1,200-household run would have taken about four hours. It was stopped at 12:44 UTC with 5,360
interviews done; none of their outcomes had been read. The run now scores the first 600 households
of the same fixed random order of test households, reusing every interview already computed for
them (same seeds, identical results). Settings are otherwise unchanged from D16.

GPT's second review (2026-11-22, notes/gpt_reply_build_r2.txt): CONTINUE. Its points, all accepted:
completion treats a possibly-misreported answer as truth (ablation needed); the fixed baseline
should be optimised per question budget; the tree should decide with the same calibrated model;
45% wrong recommendations comes largely from a dollar-scaled threshold, so report per-programme
precision/recall and a risk-coverage curve; no "matches the full questionnaire" headline; the
language-layer comparison is the top priority. Everything it asked for after the freeze is reported
as a secondary experiment.

## D18. Readers: each component does only what it is reliable at (2026-11-22)

First readings showed both 4B models fail at arithmetic hidden in language. "Around $2,900 a month"
(about $35k a year) became $60-80k at 100% confidence from the instruct model (confidently wrong),
and a near-flat spread from the decision model (uninformative, but not confidently wrong).

So a fifth reader splits the work:
- The LLM reads language into typed slots: low and high amount as stated, the period, approximate,
  zero, don't know. It never converts anything.
- Code does the arithmetic: per-month to per-year, then spreads the amount interval over the
  question's ranges in proportion to overlap (an approximate answer widened 15% each way).
- For questions that are judgments rather than amounts ("my dad lives with us, he's 72" for
  "anyone 65 or older?"), the decision model's per-option probabilities are the evidence
  (structured_dm), or the LLM's single pick (structured).

Six readers are compared on held-out generated answers (experiments/read_eval.py): llm_hard,
llm_selfconf, dm (per option), dm_thr (threshold chain on ordered questions), structured,
structured_dm. If the decision model adds nothing over the LLM's pick, that is the result (GPT r2).

## D19. Fairer baselines and ablations, added after the freeze (2026-11-22, secondary)

experiments/run_secondary.py, same test households, labelled secondary: the best fixed set per
question budget (beam search on training data); a tree that decides with the same calibrated belief
and completion as the engine, with rare branches falling back to the parent; the engine without
completion (does completion hurt when answers may be misreported?); final probabilities saved for
per-programme precision-recall curves.

## D20. Two reader bugs that would have sunk the LLM baseline (2026-11-22)

The first run of the instruct-model reader said "not stated" for 59% of answers, including plain
"No, not us." Two causes, both mine:
- The model returns numbers quoted ({"choice": "1"}); the parser only accepted bare numbers, so every
  quoted answer became "not stated".
- The prompt used 0 as the "not stated" sentinel while option 1 was "no", inviting confusion. It now
  asks for null.
Also, the generated "exact" answer for a state was its abbreviation, and "OK." (Oklahoma) reads as
"okay"; it now says the full name. All cached readings from before the fix were discarded and the
evaluation re-run. The accuracy metric was also wrong: a flat ("no information") reading counted as
correct whenever the truth was option 1. Readers are now scored on how often they commit, accuracy
when they commit, how often they are confidently wrong, and log loss.

## D21. Held-out test result (2026-11-22)

Frozen run, 600 test households, full table in results/final_600_analysis.txt and WRITEUP.md.
- Clean answers: the engine stops at 8.2 questions; loss vs asking all 18: +$67 [-363, +496] per
  household per year, indistinguishable from zero. Expected $1,329, made $1,250.
- Equal question counts, clean: better than the best single fixed order at k = 5, 7, 9 (-$569 to
  -$811, all intervals below zero), not at 3 or 12. Better than the tree at 5, 9, 12.
- Facts volunteered first: better than the fixed order at 3, 5, 9 and than the tree at every k.
- Realistic answers: no significant difference from the fixed order at any k. My D6 prediction
  (the engine wins most on messy answers) was wrong on this test.
- Hand-off: the least-certain 10% carry 22-31% of mistake cost.
GPT's go/no-go bar (>=25% fewer questions than an optimised tree at equal quality) can't be scored
cleanly because the frozen tree is weak (it stops at ~5 questions); the fair tree is in the
secondary run. Against asking everything, the engine asks 54% fewer questions at equal quality.

## D22. Secondary results: the fair tree matches the engine on clean answers (2026-11-22)

Same 600 test households, run after the freeze (results/secondary_600_analysis.txt):
- A greedy tree that decides with the same calibrated belief and completion as the engine is not
  significantly different from the engine on clean answers at any k (engine minus tree: -$142 to
  +$247). The engine's win over the frozen tree came from that tree's decision rule, not from its
  questions. This is what D6 predicted: on clean answers a greedy tree on the same loss is the same
  rule, computed in advance. GPT's bar (>=25% fewer questions than an optimised tree) is NOT met on
  clean answers. Where the engine can differ is what a precomputed tree can't do: use facts
  volunteered out of order, absorb misreports and "don't know", change its question budget or cost
  without regrowing, and say how unsure it is. The realistic-answer tree comparison tests that.
- Against the best fixed set chosen per question budget, the engine still wins with clean answers at
  k = 5, 7, 9 and with realistic answers at k = 7 (-$729 [-1,401, -132]) and 9 (-$598 [-1,167, -71]).
- Completion helps: clean, k = 9: -$736 [-1,469, -162].
- Precision by programme (engine, 9 questions, clean): SNAP 0.81, WIC 0.96, school meals 0.88,
  EITC 0.76, CTC 0.71; Medicaid 0.47, ACA credit 0.23, SSI 0.42, CHIP 0.16. Medicaid and the ACA
  credit make up 69% of wrong recommendations. Showing only recommendations at >= 30% probability:
  precision 0.73 with 77% of eligible programmes still found; at >= 50%: 0.84 and 55%.
- The secondary run first crashed on a bug in the tree's fallback branch ("don't know" after a
  backed-off answer filtered on it and found no households). Fixed; the missing runs were resumed.
- Update after the resumed run: the fair tree also matches the engine with REALISTIC answers (engine
  minus tree -$224 to +$244, no interval excludes zero). So on clean and messy answers alike, a
  well-built greedy tree asking as many questions does as well. The completion ablation follows
  GPT's warning: completion helps clearly on clean answers (k = 9: -$736) but its benefit shrinks and
  is not significant when answers may be misreported (k = 9: -$423 [-1,158, +112]). The last open
  comparison is facts volunteered first, the one setting where a precomputed tree can only skip.
- Facts volunteered first (3 facts before any question): the engine beats the fair tree at k = 3
  (-$1,072 [-1,924, -409]), 5 (-$524 [-996, -123]) and 9 (-$376 [-769, -62]); 7 and 12 are
  borderline. It beats the per-budget fixed sets at every k. This is the defensible advantage: not
  smarter questions on a clean form (a fair tree matches it there), but using whatever the person has
  already said, in any order, which a precomputed tree can only skip along its own branches.
