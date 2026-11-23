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

## D23. Readers: what each component is good at, measured (2026-11-22)

240 held-out generated answers (results_read_eval.log), after the D20 fixes:
- Plain LLM pick: commits on 70% (abstains on every "don't know"), 95.2% accurate when it commits,
  confidently wrong 3.3%, log loss 0.80. Fails on monthly amounts and vague answers.
- LLM pick + self-rated confidence: no better (95.2%, 3.3%, 0.77). Self-confidence adds nothing.
- Structured (LLM slots, code arithmetic): 97.6% when committing, confidently wrong 1.7%, log loss
  0.51; all monthly and range answers right.
- Decision model (Kev-4B), per option, uncalibrated: never confidently wrong, but diffuse (log loss
  1.00, 75% top-option accuracy) and it leans one way on "don't know" instead of going flat.
- Split by question type, with the decision model's sharpness cross-fitted: on non-amount questions
  the LLM pick is better (0.35 vs 0.41), so handing those to the decision model is the wrong design;
  on amounts the decision model (0.84) beats the LLM pick (1.76) and roughly ties structured (0.86).
- Multiplying the structured reading and the calibrated decision model on amounts (product of
  independent evidence, weight 0.5-0.75 on the decision model, sharpness 0.1; cross-fitted 5 times):
  log loss 0.785, better than either alone.
Frozen before confirmation: CombinedReader = LLM pick for non-amount questions; for amounts,
structured^(0.4) x decision-model(T=0.1)^(0.6). Tested on a NEW set of 120 amount answers
(read_eval.py amounts 120), which none of these choices saw.

End to end (40 test households, unweighted, experiments/analyze_language.py): misreading rate when a
reader commits, typed 0%, plain LLM 8.9%, structured 3.7%; loss differences vs typed answers are
within noise at this size (structured +$93 [-697, +743]). Survey-weighted means were dominated by one
household carrying 16% of the weight, so this comparison is reported unweighted.

## D24. The volunteered-facts advantage is not robust (2026-11-22)

experiments/run_opener_robust.py, 300 test households, engine minus fair tree at k = 3, 5, 9 under
other opener assumptions: one volunteered fact; two facts from only the most natural openers (state,
household size, children, 65+, married, rent/own); three facts that may be misreported. No
interval excludes zero (e.g. natural openers: +$19, -$200, +$149). The D22 advantage appeared when
three facts were drawn from a pool that includes income and disability, two of the most informative
facts; with natural openers it disappears. The intervals are wide, so this does not show the
advantage is zero, but it cannot be claimed as a property of the engine.

Corrected conclusion: the engine reaches an 18-question form's decision quality with about 8
questions, and a well-built precomputed tree does as well in almost every setting tested. What the
engine adds over a tree is flexibility, not accuracy: any question budget or question cost without
regrowing, facts accepted in any order, evidence that can be uncertain, an honest estimate of its
own remaining error, and a hand-off signal. README and WRITEUP were corrected to say so.
- Decision model with the threshold chain on ordered questions ("is it this range or higher?"):
  log loss 0.93 vs 1.00 per option, 72.9% top-option accuracy; better than asking per option, still
  far behind the structured reader (0.51). Asking ordered questions as thresholds helps a classifier
  a little; doing the arithmetic in code helps a lot more.

## D25. Pre-registered: does the engine's flexibility matter in numbers? (2026-11-22, before any run)

GPT's round-3 review: a good tree matches the engine on a plain intake, so the engine's case rests
on adapting at run time to conditions the tree wasn't built for. Test that, with everything fixed
here before any result is seen. Script: experiments/run_adapt.py.

- Households: 600 test households never used in any earlier run (test split minus the first 1,200
  of the fixed order every earlier run drew from), chosen with seed 2027.
- Base question cost $15 (the demo's setting). Every policy stops on its own when no question is
  worth its cost; at most 18 questions. Loss settings frozen as before ($1,000 per wrong
  recommendation, floor 10).
- Policies:
  - engine: unchanged, reads each question's cost at run time.
  - tree: a cost-sensitive greedy tree grown once at the uniform $15 cost; same loss, same pair
    look-ahead, rare-branch backoff, final decisions from the same calibrated belief as the
    engine (so only the choice of questions differs). A question that gets no answer (refused)
    doesn't narrow its branch: the tree carries on from that node's households.
  - tree_regrown: the same tree regrown with the scenario's question costs (only where costs change).
- Scenarios, drawn per household from a seeded generator, identical for every policy:
  - S0 plain: nothing volunteered, every question answered, equal costs.
  - S1 volunteered: 0-3 facts (equally likely) volunteered before any question, drawn uniformly from
    all 18 questions, answered exactly.
  - S2 refusals: 2 questions drawn uniformly from the 18 are refused; asking one still costs a
    question and gives no evidence.
  - S3 unequal costs: x3 for earnings, self-employment, other income, Social Security, savings,
    disability, pregnancy/infant (sensitive or slow to answer); x0.5 for state, household size,
    children, 65+, married, rent/own; x1 the rest.
  - S4 all of S1-S3 together. PRIMARY.
- Metric: realised loss plus the question costs actually paid, $/household/year, survey-weighted,
  paired bootstrap 95% CI (2,000 resamples). Also mean questions and loss alone.
- Comparisons: engine minus tree in S0-S4; engine minus tree_regrown in S3 and S4.
- Reading, fixed now: engine below the tree in S4 with the CI below zero = flexibility pays (A);
  engine level with the regrown tree but below the original = it saves rebuilding trees (B);
  level with the original tree everywhere = flexibility not shown on this task (C). Every scenario
  is reported whatever it shows.
- Not tested: a budget cut mid-session. Both policies choose greedily one question at a time, so a
  cut truncates both the same way; the equal-budget comparisons (D22) already cover it.

## D25 result. Flexibility does not show; a level gap does (2026-11-22)

results/adapt_600.jsonl (600 fresh test households, $/household/year, loss + question costs):

| scenario | engine | tree | engine minus tree [95% CI] | minus regrown tree |
|---|---|---|---|---|
| S0 plain | $1,459 (8.0 q) | $2,166 (7.5 q) | -$707 [-1,252, -233] | |
| S1 volunteered | $1,472 | $2,126 | -$655 [-1,220, -193] | |
| S2 refusals | $1,488 | $2,129 | -$642 [-1,113, -303] | |
| S3 unequal costs | $1,465 | $2,234 | -$769 [-1,313, -297] | -$762 [-1,284, -338] |
| S4 all three | $1,461 | $2,098 | -$637 [-1,097, -281] | -$598 [-1,059, -244] |

Read literally, the rule written in advance says outcome A ("flexibility pays"). That rule was badly
written: it compared levels, and the engine is just as far ahead in S0, where nothing changes. The
question GPT asked is whether the gap GROWS when conditions change. Post hoc, paired on the same
households (experiments/adapt_did.py): change in the gap relative to S0 is +$52 (S1), +$65 (S2),
-$62 [-144, +17] (S3), +$70 (S4), every interval including zero. So: flexibility is not shown on
this task (outcome C in spirit). Neither policy is hurt much by the changes. Regrowing the tree for
the new costs changes nothing. The level gap is a separate finding, examined in D27.
Speed: the engine takes about 24 ms to choose a question on this CPU; the tree's lookup is ~0.3 ms.

## D26. The decision model's edge on amounts did not replicate (2026-11-22)

Confirmation of the combination frozen in D23 (structured^0.4 x decision model^0.6 on amounts), 120
fresh amount answers (results_read_amounts.log): structured reader log loss 0.617, confidently wrong
0.8%; combined 0.625, confidently wrong 2.5%; decision model alone 1.06; plain LLM pick 1.83 (10%
confidently wrong). The cross-fitted 0.785 vs 0.86 on 76 answers was noise. The decision model adds
nothing measurable anywhere in this system, so it is dropped from the recommended design: the reader
is the structured one (the LLM fills typed slots, code does the arithmetic). The structured reader's
lead over a plain LLM pick on amounts did replicate (0.62 vs 1.83).

## D27. Post hoc: why the tree fell behind, and a tree that doesn't (2026-11-22)

Not pre-registered. On 300 further fresh test households, plain intakes (experiments/diag_adapt.py,
diag_adapt2.py; results/diag_adapt*_300.jsonl), $/household/year, loss + question costs:

| policy | questions | total | engine minus it [95% CI] |
|---|---|---|---|
| engine, own stopping at $15 | 8.1 | $1,396 | |
| D25 tree (min leaf 30) | 6.9 | $2,093 | -$697 [-1,192, -285] |
| same, min leaf 10 | 6.9 | $2,251 | -$856 [-1,719, -252] |
| tree that falls back to broader data, min leaf 30 | 9.8 | $1,809 | -$414 [-829, -104] |
| tree that falls back to broader data, min leaf 100 | 10.9 | $1,566 | -$170 [-505, +113] |
| at exactly 8 questions: engine vs fallback tree | 7.9 / 7.6 | | -$215 [-689, +230] |

Why the D25 tree lost: 79% of its intakes ended because its branch held fewer than 60 training
households, not because no question was worth asking. A tree splits its data; after about seven
questions there is too little left to choose the next question from. The same happened in D22: the
fair tree given 9 or 12 questions asked 6.9 and 7.0 on average. The engine never splits its data (it
reweights all of it, keeps at least ten households' worth of support, and lets the outcome model
fill in), so it keeps finding questions worth asking.

GreedyTree.deep_backoff fixes that: an answer that would leave too few training households doesn't
narrow the tree's household set (the session's belief still uses it). With a larger minimum leaf
(100; two values tried, both reported, chosen after seeing test results) the tree comes within noise
of the engine, using about 11 questions to the engine's 8.

Conclusion: the engine's measured edge is getting to the same place with fewer questions and no
tree-building choices to tune. It is not a large accuracy gap over a well-built tree, and on this
task it is not flexibility (D25). A precomputed tree is also about 80 times faster per question.

## D28. v2: a neural outcome model, and its pre-registered confirmation (2026-11-22, 21:36 UTC, before any test run)

Where the mistakes come from: even with all 18 answers, the linear outcome model (D13) left about
$1,200-1,500 of mistakes per household, most of them Medicaid and ACA-credit calls. The rules nest
deeper than the hand-listed interactions (income against household size against state; employer
coverage against income), so the outcome model is the ceiling every question policy works against.

v2 (src/decisionintake/neural.py): a wide-and-deep network in plain numpy. The wide part is the
same hand-listed indicator terms as the linear model; the deep part is two ReLU layers (512, 256) on
the one-hot answers, shared by all ten programmes, with an eligibility head and a dollars head.
Dropout 0.4, decoupled weight decay 0.3, Adam with a cosine schedule, early stopping on a held-out
10% of source households inside the training data, then out-of-fold Platt scaling folded into the
last layer (as D15). It trains in about 3 minutes on the CPU, against 5 for the linear model.

Validation only (test untouched):
- Model alone, all answers known, 5,867 validation households (experiments/outcome_v2.py and a
  settings sweep): decision loss $1,510 linear -> $1,166-1,263 across the regularised settings; the
  setting above was picked on log loss (1.449 -> 1.268), the less noisy measure. Bigger networks
  without regularisation overfit badly (log loss up to 12); those runs are why it is regularised.
- Engine, 400 validation households, $15 per question (experiments/val_v2.py, results/val_v2_400_analysis.txt):

| user | policy | linear: questions / loss / total | neural: questions / loss / total | neural minus linear, total [95% CI] |
|---|---|---|---|---|
| clean | engine | 8.0 / $1,328 / $1,448 | 7.7 / $827 / $943 | -$505 [-917, -176] |
| clean | all 18 | 18 / $1,236 / $1,506 | 18 / $785 / $1,055 | -$450 [-816, -173] |
| realistic | engine | 7.9 / $1,658 / $1,776 | 7.7 / $1,284 / $1,399 | -$377 [-744, -55] |
| realistic | all 18 | 16 / $1,447 / $1,687 | 16 / $997 / $1,237 | -$450 [-822, -152] |

Frozen now for the test run (experiments/run_v2.py): the model settings above, trained on train +
validation; engine unchanged (support floor 10, $1,000 per wrong recommendation, own stopping at $15
a question, pair look-ahead); every other setting as in D16/D25.

Pre-registered confirmation:
- Households: 2,000 test households no earlier run touched (the test split minus the 1,200 of D21-D24,
  the 600 of D25 and the 300 of D27), drawn with seed 2029.
- Users: clean answers (primary), and realistic ones (2 volunteered facts, misreports, 10% don't
  know; secondary, reported in full).
- Metric: decision loss + $15 per question asked, $/household/year, survey-weighted, paired bootstrap
  95% CI (4,000 resamples); Holm correction over the four primary comparisons.
- Primary comparisons (clean answers), v2 engine minus:
  - P1 the v1 engine (linear model), same stopping rule;
  - P2 the v1 full questionnaire (all 18 questions, linear model);
  - P3 the strongest tree found so far: falls back to broader data, minimum leaf 100, cost-sensitive,
    pair look-ahead, deciding with the same v2 model as the engine (D27's best);
  - P4 the best fixed 8-question form (beam search on the training population), deciding with v2.
- Secondary: v2 engine minus v2 full questionnaire; minus the D25 tree on v2; v2 minus v1 with all
  18 answers; the same comparisons for realistic users; precision of the "likely" tier (recommended
  at 50% or more) and of "worth checking"; the engine's own estimate of its mistakes against the
  realised ones; the share of mistake cost carried by the 10% and 20% of households it is least
  sure about.
- Reading, fixed now: P1 and P2 below zero after Holm = v2 is a real improvement and beats the old
  full form with fewer than half the questions. P3/P4 below zero = it beats the strongest baselines;
  intervals spanning zero are reported as "no conclusive difference", whichever side the point
  estimate falls on. Everything is reported whatever it shows.

## D28 result, part 1: P1 and P2 confirmed (2026-11-22, 22:00 UTC)

The run (results/v2_2000.jsonl) was started once at 21:37 UTC, stopped before any result was read,
and restarted from scratch at 21:44 with two changes that don't alter any result: re-scoring skips
records whose inputs didn't change (checked identical on 24 interviews and by tests/test_neural.py),
and the primary comparisons run first. At 21:58 the system stopped it for low memory, after the v2
engine, the v1 engine and the v1 full questionnaire had finished all 2,000 households on clean
answers. Those three give P1 and P2 (results/v2_2000_analysis.txt):

| policy | questions | decision loss | total (loss + $15 a question) |
|---|---|---|---|
| v2 engine | 8.4 | $1,017 | $1,144 |
| v1 engine | 8.5 | $1,297 | $1,424 |
| v1 full questionnaire | 18 | $1,356 | $1,626 |

- P1, v2 minus v1 engine: -$281 [-437, -142], Holm p < 0.002 (no bootstrap resample crossed zero,
  times the four comparisons). Loss alone -$280.
- P2, v2 minus v1 full questionnaire: -$483 [-667, -317], Holm p < 0.002. Loss alone -$339
  [-523, -173]: fewer mistakes, not only fewer questions.
- Resampling whole source households (82 of the 1,918 give two records; GPT round 4) gives the same
  intervals within $5.
- Descriptive, v2 engine: recommendations at 50% or more ("likely") are right 78% of the time and
  find 67% of all eligible programmes; "worth checking" ones are right 28% of the time. Its own
  estimate of its mistakes is 19% high (cautious). The 10% of households it is least sure about carry
  27% of the mistake cost, the 20% carry 45%.

By the reading fixed in D28: v2 is a real improvement, and it beats the old full form with fewer
than half the questions. P3 (strongest tree), P4 (best fixed 8-question set) and the secondary
comparisons still need the run resumed (it continues from the saved interviews).

## D29. Pre-registered: does the engine make a real AI assistant better? (2026-11-23, 09:30 UTC, before any test run)

Everything so far simulated the person's answers as typed options. D29 puts a real language model in the
loop, end to end, through the product's own interface:

- **The product.** `decisionintake.bundle` packs the D28 engine (questions, 48,657 training households,
  loss, the v2 network with its features stored as data) into one 1.5 MB file that needs only numpy.
  `decisionintake.service.Intake` takes typed facts from any assistant (an option, an amount with its
  period and "approximate", or "don't know"; amount arithmetic in code; a correction replays every
  fact) and answers ask / finish / hand off, with what the next question is worth and which programmes
  it mostly settles. `decisionintake.mcp_server` serves it over the Model Context Protocol (stdio,
  JSON-RPC, four tools). Tests show the typed intake asks exactly the questions the simulated
  interviews asked and reaches the same decisions.
- **The assistant.** Claude Haiku 5.5 through Claude Code in print mode (`claude -p`, version 2.1.330),
  with no built-in tools, only the MCP server of its arm.
- **The person.** Each household speaks in templated sentences generated from its own record (the
  generator and seeds of D23: exact figures, round numbers, monthly figures for yearly questions,
  ranges, vague words, implied facts, "I'm not sure"), and opens by volunteering three facts. The
  assistant reaches the person only through an `ask_person` tool (a message plus the ids of the
  catalogue questions it asks, so the template can answer) and ends with `submit_recommendations`.
- **Two arms, same model, same households, same words.** *engine*: the assistant has DecisionIntake's
  tools and is told to follow them (it reads the person's words into typed facts; the engine decides
  what to ask, when to stop and what to recommend). *alone*: the assistant decides everything itself,
  from its own knowledge of the 2026 rules, with the same question catalogue and the same costs.

Development, on 100 fresh test households set aside for it (seed 2032; not used below), with the
prompts before the two fixes listed next: engine arm 7.0 questions, $1,857 of mistakes, 16% handed off;
alone 8.2 questions, $3,408, 55% handed off; engine minus alone, total cost, -$1,568 [-3,119, -187]
(results/agent/dev/haiku_analysis.txt). Medicaid explains most of it: the alone arm missed it in 11
households, the engine in 1; the engine recommended it wrongly more often, 32 against 11, which its
dollar-weighted threshold accepts.

Fixed after development, before the test: (1) the engine's final recommendations carry programme ids,
because in 3 of the dev households the agent mistyped the engine's list (one submitted CHIP for four
other programmes and said so itself); (2) both arms are told to list what they would recommend even
when they hand off, because the alone arm often handed off with an empty list, which would have
counted against it.

Frozen now (commit e9f3af7): experiments/agent_e2e.py as committed (prompts, tools, scoring), the D28
bundle (data/bundle/benefits_v2.npz from experiments/export_bundle.py), question cost $15, $1,000 per
wrong recommendation, hand-off above $2,500 of expected mistakes.

Pre-registered test:
- Households: 200 fresh test households (the test split minus D21-D28's 4,100 and the 100 development
  households), seed 2033.
- Primary: engine arm minus alone arm, total cost (realised decision loss against PolicyEngine + $15 for
  each distinct catalogue question the assistant asked beyond the three volunteered facts),
  $/household/year, unweighted mean over households, paired bootstrap 95% CI (4,000 resamples).
  Confirmed if the whole interval is below zero. Unweighted because survey weights vary about a
  hundredfold and, at 200 households, a few heavy ones would decide the result; development showed it
  (the weighted interval was almost four times wider).
- A run that ends without a submission is rerun once; if it fails again it counts as recommending
  nothing, with the questions it asked.
- Secondary, reported in full: survey-weighted difference; questions; cost of missed programmes and of
  wrong recommendations; hand-off rate; the difference on households neither arm handed off;
  per-programme misses and wrong recommendations; how often the engine-arm agent's submission differs
  from the engine's own list.
- Exploratory: the same two arms with Claude Sonnet 5.5 on the first 100 test households.
- Prediction: the engine arm costs less, and the interval excludes zero.
