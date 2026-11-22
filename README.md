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
- **Readers.** Free-text answers become evidence through a reader: a small decision model that gives
  a probability for each option (or for each step of an ordered scale), or an instruct model that
  picks one option. Hedged answers become spread-out evidence instead of a guess.

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

- **Not an eligibility determination.** Results are PolicyEngine's estimates under 2026 rules,
  assuming everyone takes up what they're entitled to. A real screener must send people to the
  official application.
- **Not tested on real people.** Populace is a calibrated synthetic population built from census and
  tax data. Every number here is agreement with PolicyEngine on that population; spoken answers are
  generated from templates, which are cleaner than real speech.
- **No immigration questions,** by design. Immigration status changes eligibility, so it remains an
  unasked source of error, reported separately.
