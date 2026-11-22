# DecisionIntake

An engine that tells an AI assistant **what to ask next, which answers to trust, and when it knows
enough to act or should hand off to a person**. Its first real use is US public-benefits screening:
ten programmes, rules from [PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us), and a
population of 57,240 households from PolicyEngine's [Populace](https://huggingface.co/datasets/policyengine/populace-us)
dataset.

The engine itself knows nothing about benefits. A domain supplies questions (with answer options and
a model of how people misreport them), decisions (here: recommend a programme or not, with the cost of
each kind of mistake), and a population of records with known answers and known outcomes.

> Built by AI agents (Claude, with reviews by GPT and Fable) under Ege Hazar's direction.
> Every design choice and the reason for it is in [DECISIONS.md](DECISIONS.md).

## How it works

```
person's words ──► reader ──► evidence over a question's options
                                  │
                                  ▼
             belief: weights over households that could be this person
                                  │
            outcome model scores each candidate on the person's own answers
            (the candidate only fills in what the person hasn't said yet)
                                  │
                                  ▼
       for every question: expected drop in the cost of mistakes if we ask it
                                  │
          ASK the best one while it's worth more than a question costs
          ACT when nothing is worth asking
          HAND OFF when the remaining expected cost of mistakes is still high
```

- **Belief.** Weights over training households, updated by each answer through that question's
  misreport model. A support floor keeps the belief resting on enough households that it can't
  become confident just by eliminating almost everyone (D12).
- **Outcome model.** A regularised logistic and ridge regression on answers and the interactions the
  rules use (income against household size, and so on), trained on PolicyEngine's results. Each
  candidate household is scored on a hybrid of the person's answers and its own (D13, D14).
- **Value of information.** For each unasked question, the expected reduction in the cost of mistakes
  across all ten programmes at once, computed with one matrix product per turn (D5, D11).
- **Readers.** Free-text answers become evidence through a reader. The best one tested lets a language
  model fill typed slots (amount as stated, per month or year, approximate, don't know) and does the
  arithmetic in code, so "about $2,900 a month" lands on the right yearly range and a hedged answer
  becomes evidence spread over several options instead of a guess. A small decision model that gives
  a probability per option was also tested; it only helped on amounts (D23).

## Results (held-out test, 600 households never used for training or tuning)

Settings were frozen on a validation split first (DECISIONS D16). Loss is the cost of mistakes per
household per year: a missed programme at its value, a wrong recommendation at $1,000. Intervals are
95% paired bootstrap. Full tables: [WRITEUP.md](WRITEUP.md), `results/final_600_analysis.txt`,
`results/secondary_600_analysis.txt`.

- **About half the questions, no conclusive difference in loss.** With clean answers the engine stops
  after 8.2 questions instead of 18; the difference in loss is +$67 [-363, +496]. The test can't tell
  the two apart, which is not the same as showing they're equal: the interval allows the engine to be
  up to $496 a household worse. On average it expected $1,329 of mistakes per household and made
  $1,250.
- **A well-built tree does as well.** A greedy tree on the same loss, deciding with the same calibrated
  model, matches the engine at every question budget with clean or messy answers. In one test, with
  three facts volunteered from a pool including income and disability, the engine was ahead by
  $376-$1,072 per household; with more natural openers (state, household size, children) that edge
  was not significant (DECISIONS D24). What the engine can add over a tree is flexibility, not
  accuracy: any budget or question cost without regrowing, facts in any order, uncertain evidence,
  and an estimate of its own remaining risk.
- **It beats fixed forms** (a fixed set chosen for each question budget) with clean answers at 5-9
  questions, and with realistic answers at 7 and 9.
- **Hand-off.** The 10% of households it is least sure about carry 22-31% of all mistake cost.
- **Not yet a screener people should rely on.** About 45% of recommendations are for programmes the
  household can't get, almost all Medicaid and the ACA credit, where the dollar-weighted objective
  recommends at low probability. Showing only recommendations at 30% or more: 73% precision, 77% of
  eligible programmes still found.

## Reproduce

```bash
# 1. environment (Python 3.13)
py -m uv venv --python 3.13 .venv
py -m uv pip install --python .venv/Scripts/python.exe policyengine-us tables h5py pytest -e .

# 2. data: Populace national build populace-us-2024-spm-20260915 (827 MB, sha256 6496cc43...aee84)
#    into data/raw/populace_us_2024.h5, then the outcome table (about 10 minutes on 12 cores)
bash adapters/benefits/build_all.sh 30 5

# 3. tests, tuning (validation split only), the final comparison (test split)
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe experiments/tune.py 400 10
.venv/Scripts/python.exe experiments/run_core.py 2000 10
```

On Windows, PolicyEngine's longest file path exceeds the 260-character limit unless the package sits
under a short prefix; the build script points `PYTHONPATH` at such a copy (D11).

## What this is not

- **A research prototype.** Nobody should use it to decide whether to apply for a benefit. The demo
  keeps nothing about the household; a free-text description is sent to Claude to be read. The
  language results above were measured with 4B models on a CPU, not with Claude.
- **Not an eligibility determination.** Results are PolicyEngine's estimates under 2026 rules,
  assuming everyone takes up what they're entitled to. A real screener must send people to the
  official application.
- **Not tested on real people.** Populace is a calibrated synthetic population built from census and
  tax data. Every number here is agreement with PolicyEngine on that population; spoken answers are
  generated from templates, which are cleaner than real speech.
- **No immigration questions,** by design. Immigration status changes eligibility, so it remains an
  unasked source of error, reported separately.
