# DecisionIntake

**A decision engine for AI assistants: what to ask next, how far to trust each answer, and when it
knows enough to act or should hand off to a person.**

First use: US public-benefits screening across ten programmes, with rules from
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) and 57,240 households from
PolicyEngine's [Populace](https://huggingface.co/datasets/policyengine/populace-us) data.

On 2,000 households it had never seen, in a test registered before it ran (DECISIONS D28):

| | DecisionIntake v2 | the full 18-question form (v1) |
|---|---|---|
| questions asked | **8.4** | 18 |
| cost of mistakes per household per year | **$1,017** | $1,356 |
| difference [95% CI] | **−$339** [−523, −173] | |

**About half the questions, 25% fewer costly mistakes.** Against the previous engine, with the same
questioning, it cuts mistakes by 22% (−$280 [−437, −140]). Both results hold after Holm correction
(p < 0.002) and with source households resampled as clusters.

> Built by AI agents (Claude, with reviews by GPT and Fable) under Ege Hazar's direction. Every
> design choice, every result, and every idea that didn't make it is logged in [DECISIONS.md](DECISIONS.md).

## What it does

```
person's words ──► LLM reader ──► typed evidence (amount, period, approximate, don't know)
                                        │
                                        ▼
                  belief: weights over 48,657 households that could be this person
                                        │
              neural outcome model scores each candidate on the person's own answers
                    (P(eligible) and dollars, ten programmes at once)
                                        │
                                        ▼
           value of information: the expected drop in the cost of mistakes for every question
                                        │
               ASK the best one while it is worth more than a question costs
               ACT with tiered recommendations ("likely" / "worth checking")
               HAND OFF when the doubt that is left is still costly
```

- **Reads language, doesn't trust it with arithmetic.** An LLM fills typed slots; code converts
  "about $2,900 a month" into evidence spread over the right yearly ranges. On fresh amount answers
  this cut confidently wrong readings from 10% to 0.8% against letting the LLM pick a range (D26).
- **Believes over real-data households.** Weights over training households, updated through each
  question's model of how people misreport. A support floor keeps it from becoming certain by ruling
  almost everyone out (D12).
- **Predicts with a wide-and-deep network written in plain numpy.** Two ReLU layers shared by all ten
  programmes, next to the hand-listed linear terms, with an eligibility head and a dollars head;
  dropout, decoupled weight decay, early stopping on held-out source households, out-of-fold Platt
  scaling folded into the weights. Trains in about 3 minutes on a CPU (D28).
- **Prices every question in dollars.** For each unasked question, the expected reduction in the
  cost of mistakes across all ten programmes, in one matrix product per turn, with pair look-ahead
  when facts only matter together (D5, D11).
- **Knows what it doesn't know.** Its estimate of its own remaining mistakes is within 19% of what it
  actually gets wrong (on the cautious side), and the 20% of households it is least sure about carry
  45% of all mistake cost, which is what the hand-off routes to a person.
- **Runs in a browser.** The demo page runs the whole engine and the neural model in JavaScript; its
  probabilities match numpy to 5×10⁻⁶ and it scores 1,500 households in well under a second.

## Results

Held-out test, settings frozen on a validation split first, $/household/year, 95% paired bootstrap
intervals. Loss is the cost of mistakes: a missed programme at its value, a wrong recommendation at
$1,000. Full tables in [WRITEUP.md](WRITEUP.md) and `results/`.

| v2 engine (8.4 questions) against | difference in mistakes + $15 a question | in mistakes alone |
|---|---|---|
| v1 engine, same questioning (8.5 questions) | **−$281** [−437, −142] | −$280 [−437, −140] |
| v1 full questionnaire (18 questions) | **−$483** [−667, −317] | −$339 [−523, −173] |

- **Tiered verdicts.** Recommendations it calls "likely" (50% or more) are right 78% of the time and
  find two thirds of all eligible programmes. Lower-probability ones are shown as "worth checking":
  the dollar-weighted objective recommends a $10,000 programme at a 9% chance, which is the right
  call for the person but should be labelled as such.
- **Against decision trees** (v1, pre-registered, D25): a cost-sensitive greedy tree on the same loss
  cost $700 a household more (−$707 [−1,252, −233]). A tree we then rebuilt to fall back to broader
  data comes within noise of the engine but needs about 11 questions to its 8 (D27).
- **The biggest gain came from prediction, not question choice.** Even with every answer, the old
  linear model left most of the mistakes; the outcome model was the ceiling every question policy
  worked against. Replacing it is what moved the numbers (D28).

## Reproduce

```bash
# 1. environment (Python 3.13)
py -m uv venv --python 3.13 .venv
py -m uv pip install --python .venv/Scripts/python.exe policyengine-us==2.152.15 tables h5py pytest -e .

# 2. data: Populace national build populace-us-2024-spm-20260915 (827 MB, sha256 6496cc43...aee84)
#    into data/raw/populace_us_2024.h5, then the outcome table (about 10 minutes on 12 cores)
bash adapters/benefits/build_all.sh 30 5

# 3. tests (30), and tuning on the validation split only
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe experiments/outcome_v2.py            # outcome models, validation
.venv/Scripts/python.exe experiments/val_v2.py 400 WORKERS     # engine with each model, validation

# 4. the pre-registered test (D28): 2,000 fresh test households; WORKERS processes, ~1.5 GB each
.venv/Scripts/python.exe experiments/run_v2.py 2000 WORKERS
.venv/Scripts/python.exe experiments/run_v2.py 2000 1 analyse

# 5. the browser demo
.venv/Scripts/python.exe experiments/export_demo_v2.py 1500 demo/population.json
cd demo && ../.venv/Scripts/python.exe -m http.server 8765    # then open http://127.0.0.1:8765
```

Earlier experiments (D21-D27: v1 test, trees, run-time adaptation, the language layer) have their
own scripts in `experiments/`, listed in DECISIONS.md. Seeds are fixed everywhere, so a rerun gives
the same households and draws. Tested with Python 3.13, numpy 2.5.5, policyengine-us 2.152.15 on a
12-core CPU without a GPU. On Windows, PolicyEngine's longest file path exceeds the 260-character
limit unless the package sits under a short prefix; the build script handles it (D11).

## Scope

- **A research system, not an eligibility determination.** Results are agreement with PolicyEngine's
  2026 rules on a calibrated synthetic population, assuming everyone takes up what they're entitled
  to. A real screener sends people to the official application.
- **The language results are for 4B models on a CPU.** The demo uses Claude to fill the same slots;
  spoken answers in the tests come from templates, which are cleaner than real speech.
- **No immigration questions,** by design. Immigration status changes eligibility, so it remains an
  unasked source of error.
