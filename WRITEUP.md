# DecisionIntake: a decision engine for AI intake

AI agents (Claude, with reviews by GPT and Fable) did the hands-on work; Ege Hazar directed it.
Everything marked "validation" was used for tuning and is not an independent result.

**In brief.** DecisionIntake decides what an AI assistant should ask next, how far to trust each
answer, and when to act or hand off to a person. Version 2 pairs the decision engine with a
wide-and-deep outcome network written in plain numpy. In a test registered before it ran, on 2,000
synthetic households it had never seen, scored against PolicyEngine's benefit rules, it asked 8.4
questions instead of 18 and made 25% fewer costly mistakes than the full questionnaire of version 1
(−$339 a household a year [−523, −173]); against version 1's engine with the same questioning, 22%
fewer (−$280 [−437, −140]). The largest gain came from better prediction, not cleverer questioning:
the outcome model was the ceiling every question policy worked against. Language is read by an LLM
that fills typed slots while code does the arithmetic, which cut confidently wrong readings of
amounts from 10% to 0.8%. The whole engine also runs in a browser.

## The problem

An assistant helping someone find public benefits has to decide, turn by turn, what to ask. A form asks
everyone everything. A language model asks whatever seems natural and has no idea when it knows
enough. Neither weighs what a question is worth: how many costly mistakes it is expected to prevent,
across every programme at once.

DecisionIntake is a small engine that makes that call. It keeps a belief over households that could
be this person, scores each candidate with an outcome model trained on PolicyEngine's rules, and asks
the question expected to cut the cost of mistakes the most. It stops when no question is worth asking,
and hands off to a person when the remaining risk is too high. The engine knows nothing about benefits;
the benefits adapter supplies questions, programmes and a population.

## Version 2: the outcome model was the ceiling

By version 1 the engine asked about 8 questions and did as well as asking all 18, but "as well"
still meant about $1,200-1,500 of mistakes per household, mostly Medicaid and ACA-credit calls. With
every answer known, a linear model on the answers and a hand-picked list of interactions (income
against household size, children, age, and so on) could not get closer to PolicyEngine. Eligibility
rules nest deeper than that: income against household size against state, employer coverage against
income. Every question policy works against that ceiling, so that is where the next gain had to come
from.

**The model.** A wide-and-deep network (`src/decisionintake/neural.py`), written in numpy with its own
backward pass. The wide part is the same hand-listed indicator terms as the linear model; the deep
part is two ReLU layers (512 and 256 units) on the one-hot answers, shared by all ten programmes, with
an eligibility head and a dollars head trained only on households eligible for each programme.
Dropout 0.4, decoupled weight decay, Adam with a cosine schedule, early stopping on a held-out tenth
of source households inside the training data, then out-of-fold Platt scaling folded into the last
layer so the engine can take its probabilities at face value. It trains in about 3 minutes on the CPU.
Bigger networks without regularisation overfit badly (validation log loss up to 12); the settings
were picked on validation log loss from eight configurations.

**Validation** (400 households, $15 a question): the engine's mistakes fell from $1,328 to $827 a
household, and with realistic answers (misreports, "don't know", facts volunteered up front) the
total cost fell by $377 [−744, −55]. Then everything was frozen and the test registered (D28).

**The pre-registered test.** 2,000 test households that no earlier run had touched, clean answers,
Holm correction over the primary comparisons:

| policy | questions | mistakes | mistakes + $15 a question |
|---|---|---|---|
| **v2 engine** | **8.4** | **$1,017** | **$1,144** |
| v1 engine | 8.5 | $1,297 | $1,424 |
| v1 full questionnaire | 18 | $1,356 | $1,626 |

v2 minus v1 engine: −$281 [−437, −142]; v2 minus the v1 full questionnaire: −$483 [−667, −317],
−$339 [−523, −173] in mistakes alone. Both p < 0.002 after Holm. Resampling whole source households
(82 of the 1,918 source households contribute two test records) moves the intervals by under $5.

**What it does with what it knows.**
- Recommendations at 50% or more ("likely") are right 78% of the time and find two thirds of all
  eligible programmes. Below that, a recommendation means "worth checking": missing a $10,000
  programme costs ten times a wrong recommendation, so the objective recommends it at a 9% chance,
  and the page says so.
- The engine's estimate of its own remaining mistakes is 19% above what it actually gets wrong (it
  errs on the cautious side), and the 20% of households it is least sure about carry 45% of all
  mistake cost. That concentration is what makes a hand-off worth having.

**Engineering notes.** After every answer each candidate household is re-scored on the person's own
answers (completion, D14); only households whose inputs changed are scored again, which is exact
(tests/test_neural.py checks it against scoring everything). The browser demo runs the same network
in JavaScript from base64 float32 weights and reproduces numpy's probabilities to 5×10⁻⁶.

## Building the engine: five problems and their fixes

The textbook version did not work. Each failure below was found on a validation split and fixed
before the test set was touched.

1. **The belief collapsed.** Conditioning on many answers left a handful of matching households, and
   the engine became 16 times more confident than it should have been (it believed it would lose $184
   a household; it lost $3,049). A support floor, which softens the evidence just enough to keep the
   belief resting on at least ten households, fixed both the loss and the self-assessment.
2. **Decisions were poor even with every question answered.** The belief predicted eligibility by
   averaging the raw yes/no outcomes of a few neighbouring households in 18 dimensions. Replacing each
   household's raw outcome with a learned estimate of eligibility given its answers (out-of-fold, so no
   household's estimate saw its own label) raised the ranking quality sharply: SNAP AUC 0.82 to 0.95,
   Medicaid 0.85 to 0.94, SSI 0.82 to 0.92.
3. **The engine still lost most of that.** Neighbours were scored at *their* answers, so a neighbour
   from another income bracket brought that bracket's eligibility along. Completion fixed it: each
   candidate is re-scored on the person's own answers wherever the person has answered, and supplies
   only the unknowns. With all questions asked, programmes found rose from 61% to 80%.
4. **It stopped too early because it was overconfident.** Its probabilities were too sharp (the model
   re-scored households it was fitted on) and its dollar estimates too low (a log-scale fit estimates
   medians). Platt scaling on out-of-fold predictions and fitting dollars on the dollar scale fixed both.
5. **Two-question look-ahead added nothing here.** It matters when facts only count together (there is
   a test for that); in this domain one question at a time was enough. It is off by default.

## Version 1 on held-out households (D21, D22)

600 households the engine never saw, settings frozen on validation beforehand (D16, D17), wrong
recommendation costed at $1,000, 95% intervals by paired bootstrap. Loss is the cost of mistakes per
household per year: missed programmes at their value, wrong recommendations at $1,000.

**Stopping by itself, against asking every question:**

| answers | engine: questions, loss | all questions: questions, loss | difference [95% CI] |
|---|---|---|---|
| clean | 8.2, $1,250 | 18, $1,183 | +$67 [-363, +496] |
| volunteers 3 facts first | 5.6, $1,030 | 15, $1,183 | -$153 [-472, +50] |
| realistic (misreports, 10% don't know) | 8.0, $1,870 | 16, $1,674 | +$196 [-85, +532] |

With clean answers the engine asks fewer than half the questions, and the test cannot tell its loss
apart from asking everything. That is not proof of equal quality: the interval still allows the
engine to be up to $496 a household worse ($532 with realistic answers), and no margin of "close
enough" was fixed in advance. On average its estimate of its own loss is close: it expected $1,329 of
mistakes per household and made $1,250 (on realistic answers it is about 12% optimistic).

**At equal question counts** (engine minus baseline; negative means the engine is better):

| answers | k | vs best fixed order | vs tree on the same loss |
|---|---|---|---|
| clean | 5 | -$674 [-1,380, -166] | -$499 [-939, -20] |
| clean | 7 | -$569 [-1,006, -149] | -$312 [-839, +228] |
| clean | 9 | -$811 [-1,379, -288] | -$549 [-1,040, -43] |
| clean | 3 / 12 | not significant | 12: -$577 [-1,071, -83] |
| volunteers facts | 3 / 5 / 9 | -$747 / -$572 / -$449, all significant | significant at every k |
| realistic | any | not significant | 9: -$798 [-1,551, -40] |

Against these two baselines the engine is ahead with clean answers in the middle range. Both
baselines turned out weaker than they should be (the tree decided from a handful of households in its
leaf; a single fixed order is not the best fixed set for each budget), and the next paragraphs replace
them with stronger ones.

**A fair tree changes the story, and sharpens it** (secondary experiment, run after the freeze at GPT's
request). The frozen tree decided with its own leaves; a tree that decides with the same calibrated
belief as the engine, and only differs in how it picks questions, matches the engine on clean answers
and on realistic answers at every question budget (no interval excludes zero; the tree often used
fewer questions than its budget, which turned out to matter, see below). That was the prediction
written down before any result: on a fixed form, a greedy tree on the same loss *is* the same rule,
computed in advance. Where the engine pulls ahead is when the person has already volunteered facts:

| facts volunteered first | engine minus fair tree [95% CI] |
|---|---|
| 3 questions | -$1,072 [-1,924, -409] |
| 5 questions | -$524 [-996, -123] |
| 9 questions | -$376 [-769, -62] |

That looked like the engine's real advantage: a precomputed tree can only skip along its own branches,
while the engine re-plans around whatever it has been told. **It did not hold up.** With other opener
assumptions (one fact; two facts from the most natural openers such as state, household size and
children; three facts that may be misreported), on 300 test households, no difference was significant
(natural openers: +$19, -$200, +$149 at 3, 5 and 9 questions). The edge above came from openers that
could include income and disability, two of the most informative facts. So a well-built tree does as
well as the engine in every setting tested so far. What the engine might add is flexibility: it takes
any question budget or question cost without regrowing anything, accepts facts in any order and
evidence that is uncertain, and estimates its own remaining risk, which the hand-off uses. Whether that
flexibility is worth anything in numbers is tested below. My prediction (D6) was
that the engine would gain most when answers are messy; on this test it is the reverse: with
misreports and "don't know", its edge over a fixed form is not significant. The fixed order's own
curve is uneven (5 to 9 questions barely help it, 12 helps a lot), so part of the clean-answer margin
may come from a weak single order; per-budget fixed sets are tested as a secondary experiment.

**Hand-off.** Handing the 10% of households the engine is least sure about to a person would route
22-31% of all mistake cost to that person, two to three times what a random 10% would. The engine's
estimate of its own remaining loss correlates 0.35 with what it actually loses: useful for triage, not
a guarantee.

## Version 1 against decision trees, and run-time adaptation (D25, D27)

If a good tree matches the engine, the engine's case rests on adapting at run time: facts arriving in
any order, questions people won't answer, questions that cost more than others. Before running
anything I wrote down the test: 600 test households no earlier run had touched, both policies
deciding for themselves when to stop at $15 a question, scored on mistakes plus the questions paid
for, under four changes the tree wasn't built for.

| conditions | engine | tree | engine minus tree [95% CI] |
|---|---|---|---|
| plain intake | $1,459 (8.0 questions) | $2,166 (7.5) | -$707 [-1,252, -233] |
| 0-3 facts volunteered first | $1,472 | $2,126 | -$655 [-1,220, -193] |
| two questions refused | $1,488 | $2,129 | -$642 [-1,113, -303] |
| sensitive questions cost 3x, easy ones half | $1,465 | $2,234 | -$769 [-1,313, -297] |
| all three at once | $1,461 | $2,098 | -$637 [-1,097, -281] |

The rule I wrote down said a significant win in the last row means flexibility pays. That rule was
badly written. The engine is just as far ahead in the first row, where nothing changes, so the gap
says nothing about adapting. Measured as what the question actually asks, how much the gap grows
when conditions change, every change is within noise (between -$62 and +$70). **Flexibility is not
shown on this task.** Regrowing the tree for the new costs changed nothing either.

**Why the tree fell behind (not pre-registered, 300 more untouched households).** 79% of the tree's
intakes ended because its branch held fewer than 60 training households, not because no question
was worth asking. A tree splits its data, and after about seven questions there is too little left to
choose the next question from. The fair tree of the previous section had the same problem: given 9
or 12 questions, it asked 6.9 and 7.0. The engine never splits its data, it reweights all of it and
lets the outcome model fill in, so it keeps finding questions worth asking. A tree that falls back to
broader data when a branch runs thin, with a larger minimum branch size, comes within noise of the
engine (engine ahead by $170 [-505, +113]) but needs about 11 questions to the engine's 8. Its
settings were picked after seeing these households (two tried, both reported).

So the engine's measured edge over a tree is modest: it gets to the same place with fewer questions
and without tree-building choices to tune. A precomputed tree is about 80 times faster per question
(0.3 ms against 24 ms on this CPU).

## The language layer

The engine needs evidence, not just an answer: "about $2,900 a month" should land on one income
range, "somewhere between 25 and 40 thousand" should spread over two, and "I'd have to check" should
change nothing. Readers were compared on 240 answers generated from held-out households' records in
controlled styles (exact, approximate, monthly, range, vague, indirect, don't know), all on this
CPU-only machine with 4B-parameter models (Qwen3.5-4B as the language model, Kev-4B as the decision
model).

| reader | commits | accurate when it commits | confidently wrong | log loss |
|---|---|---|---|---|
| language model picks an option | 70% | 95.2% | 3.3% | 0.80 |
| ... and rates its own confidence | 69% | 95.2% | 3.3% | 0.77 |
| decision model, probability per option | 100% | 75.4% | 0.0% | 1.00 |
| structured: model fills slots, code does arithmetic | 70% | 97.6% | 1.7% | 0.51 |

Three things came out of this:

- **Let code do the arithmetic.** Both small models fail at arithmetic hidden in language. Asked about
  "around $2,900 a month", the language model picked $60-80k a year with full confidence (the truth is
  about $35k). When it only fills typed slots (amount as stated, period, approximate, zero, don't know)
  and code converts and spreads the amount over the ranges, monthly and range answers are all read
  right and confident mistakes halve.
- **Self-rated confidence is worthless here.** Asking the language model how sure it is changed
  nothing measurable.
- **The decision model earns its place only where it is good.** On yes/no and category questions the
  language model's pick reads better (log loss 0.35 vs 0.41, decision model calibrated by cross-fitting),
  so handing those to the decision model would be worse. On amounts the calibrated decision model
  roughly ties the structured reading, and multiplying the two as independent evidence beat either
  (0.785 vs 0.84 and 0.86, cross-fitted on 76 answers). **It did not replicate.** On 120 fresh amount
  answers, with the combination frozen beforehand, the structured reader alone scored 0.617 and the
  combination 0.625, with three times the confident mistakes (2.5% vs 0.8%). So the decision model
  adds nothing measurable anywhere in this system, and the recommended reader is the structured one.
  Its lead over the plain pick on amounts did replicate (0.62 vs 1.83, 10% confidently wrong).

In full interviews (40 held-out households answering in sentences), the structured reader misread
3.7% of the answers it committed to, against 8.9% for the plain pick. At 40 households the effect on
decision loss is within noise for every reader, including typed answers, so the claim is about
reading accuracy, not about benefits found.

The browser demo uses the same structured design but with Claude filling the slots, because a page
can't run the 4B models. The numbers above are for the 4B models, so they don't measure the demo's
exact pipeline.

## What this is not

- A research prototype. Nobody should use it to decide whether to apply for a benefit. The demo
  keeps nothing about the household it is told about; a free-text description is sent to Claude to
  be read, and nothing else leaves the page.
- Not an eligibility determination, and not tested on real people. Populace is a calibrated synthetic
  population; every number is agreement with PolicyEngine on it, under 2026 rules with full take-up.
- Taken together, 43% of v2's recommendations are for programmes the household can't get. That follows
  from the objective (missing a $10,000 programme costs ten times a wrong recommendation, so it is
  recommended at a 9% chance), which is why the engine shows two tiers: "likely" (right 78% of the
  time) and "worth checking". `experiments/failure_case.py` traces a v1 case: a married couple in
  California with one child, earning $80-100k, no employer insurance. After nine questions the belief
  gives the ACA premium credit a 36% chance; worth about $10,000 a year when it applies, it pays to
  recommend above 9%, and PolicyEngine says the household gets nothing (`results/failure_case_*.txt`).
  Per-programme operating points, chosen on validation, are what a real screener would add.
- No immigration question, by design; it changes Medicaid, CHIP and ACA eligibility, so it remains an
  unasked source of error.
- The answers in the language test are generated from templates, which are cleaner than real speech.
