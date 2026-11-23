# DecisionIntake

**A decision engine for AI assistants: what to ask next, how far to trust each answer, and when it
knows enough to act or should hand off to a person.**

**Live demo:** https://claude.ai/artifact/R4AEXkghaed3zDZFiwxufi (talk to it through Claude, or watch a
held-out household's interview, with everything the engine believes shown as it decides).

First use: US public-benefits screening across ten programmes, with rules from
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) and 57,240 households from
PolicyEngine's [Populace](https://huggingface.co/datasets/policyengine/populace-us) data.

On 2,000 households it had never seen, in a test registered before it ran (DECISIONS D28):

| | DecisionIntake v2 | the full 18-question form (v1) |
|---|---|---|
| questions asked | **8.4** | 18 |
| cost of mistakes per household per year | **$1,017** | $1,356 |
| difference [95% CI] | **−$339** [−523, −173] | |

**About half the questions, 25% fewer costly mistakes.** It beat every alternative registered
against it: the previous engine (−22% mistakes), the strongest decision tree we could build, and the
best fixed 8-question form. All four comparisons hold after Holm correction (p < 0.002) and with
source households resampled as clusters, and it matches its own 18-question form with 8.4 questions.

**And it makes a real AI assistant better.** In a second registered test (D29), Claude Haiku interviewed
200 fresh households that answer in everyday sentences ("somewhere between 25 and 30 thousand",
"my dad lives with us, he's 72"), once deciding on its own and once with DecisionIntake's MCP tools:

| per household | Claude alone | Claude + DecisionIntake |
|---|---|---|
| costly mistakes | $3,884 | **$2,305** (−41%) |
| questions asked | 8.0 | **7.2** |
| cases handed to a caseworker | 60% | **20%** |

Total cost −$1,591 a household [−2,469, −775]. It isn't recommending more to find more: at the same
precision (41% against 40%) it finds 68% of the programmes people qualify for, against 51%. Claude
alone misses the expensive ones (Medicaid in 26 households against 8). A smaller exploratory run with
Claude Sonnet 5.5 points the same way: 76% of eligible programmes found against 50%, 32% fewer
mistakes.

**It's the decisions, not only the data (D30).** Given the same PolicyEngine-trained predictor and
decision rule but left to choose its own questions, Claude stopped after 3.6 questions and made $3,249
of mistakes. With the engine choosing what to ask and when to stop: $2,305, a further −$891 a
household [−1,538, −324], registered and confirmed.

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

## Plug it into any assistant (MCP)

The engine ships as one 1.5 MB file and a Model Context Protocol server with no dependencies beyond
numpy. Any MCP client (Claude Code, Claude Desktop, Cursor, an agent framework) can run an intake:
the assistant does the talking and reads the person's words into typed facts; the engine decides
what to ask, how far to trust it, and when to stop or hand off.

The trained engine ships in the repo (`bundles/benefits_v2.npz`, 1.5 MB), so this works straight after
cloning, with no data build:

```bash
pip install -e .                                                  # numpy is the only dependency
claude mcp add decisionintake -- python -m decisionintake.mcp_server
```

Any other MCP client takes the same command (`python -m decisionintake.mcp_server`, stdio transport),
or the hosted endpoint over HTTP (see below), where the same tools carry a `state` string instead of a
session and `next_step` is folded into `record_facts`.

| tool | what it does |
|---|---|
| `start_intake` | opens an intake; returns every question (so volunteered facts can be recorded) and the first step |
| `record_facts` | typed facts from anything the person said: an option, an amount as said (`{"value": 2400, "period": "month", "approximate": true}`) or "don't know"; returns how code read each one and the next step |
| `next_step` | ask (which question, what it's worth in dollars of mistakes prevented, which programmes it settles), finish with tiered recommendations, or hand off with the reason |
| `get_assessment` | every programme's probability, tier and yearly value range, and the facts so far |

**Tested end to end with a real model** (D29, `experiments/agent_e2e.py`). Claude Haiku 5.5 runs in
Claude Code's print mode with no built-in tools, only one MCP server: DecisionIntake's four tools in
one arm, nothing in the other. The simulated person speaks in sentences generated from a held-out
household's record (exact figures, round numbers, monthly figures for yearly questions, ranges, vague
words, implied facts, "I'm not sure") and opens with three facts unprompted. Both arms reach the
person through the same `ask_person` tool and end with `submit_recommendations`, scored against
PolicyEngine. Prompts and scoring were frozen on 100 separate development households before the 200
test households ran. The agent passed the engine's verdict on exactly in all 200 interviews; an
interview takes about 30 seconds and half a cent of model use. Where the engine arm is weaker:
households that qualify for nothing, where its "worth checking" calls cost $200 more than Claude's
caution, and the survey-weighted difference (−$294 [−1,831, +1,146]) is not significant. Details in
DECISIONS D29.

## Run it as a service

One process serves a public screener, a JSON API and a remote MCP endpoint, on the standard library
and numpy:

```bash
python -m decisionintake.web --port 8080      # screener at /, API at /v1/*, MCP at /mcp, /openapi.json
```

| route | what |
|---|---|
| `GET /` | the screener: one question at a time, results with yearly value ranges and the official place to apply, answers you can change |
| `POST /v1/step` | `{"state"?, "facts"?}` → `{"state", "readings", "next"}`: record what the person said, get ask / finish / hand off |
| `POST /v1/assessment` | every programme's probability, tier and value range, and the facts so far |
| `GET /v1/questions` | the programmes, every question and its options, the starting state |
| `POST /mcp` | MCP over Streamable HTTP: the stateless 2026-07-28 revision (`server/discover`, per-request metadata, header validation) and the handshake revisions 2025-11-25 back to 2024-11-05 on the same endpoint; add the URL as a connector in Claude or any MCP client |
| `GET /healthz`, `GET /openapi.json` | liveness; the API, described |
| `GET /docs.html` | the developer guide: the loop, facts, responses, the state string, MCP, code in JS, Python and curl; every example is a real response, regenerated by `experiments/build_docs.py` |

- **Keeps nothing.** The facts travel with each call as a short readable string
  (`di1;state=TX;household_size=3;earned_income=~2400/month;#6f0c`, with a check so a mangled copy
  is refused) and the service rebuilds the belief from it: 109 ms a request at the median, 175 ms at
  the 95th percentile on one core, no errors in 997 requests of whole screenings (`experiments/load_test.py`). No database, no sessions,
  no answers in logs. Any instance answers any call, so it scales to zero (D31).
- **Rules the model may not break.** Statutory limits on top of the network (`bundles/guards_2026.json`):
  where the answers pin down one tax unit, no EITC above its 2026 phase-out end and no EITC or
  refundable child credit without earnings. Every rule names its source and was checked against
  PolicyEngine's own results before shipping; every exception is traced to its cause
  ([contradiction ledger](results/contradiction_ledger.md)). On 2,000 fresh households the rules cost
  nothing (−$3.7 a household [−9.4, +0.0], registered, D32) and, with clean answers, suppressed no
  programme anyone qualified for. The ledger also caught a bug in this project's own data pipeline
  (self-employment that PolicyEngine keeps in separate variables was coded as no earnings in 8% of
  households); fixed and retrained in D34.
- **Says when it can't tell.** After about 8 answers, most people are matched by fewer than 10 of the
  48,657 training households, and that is where the model's negatives fail: on 2,183 fresh households
  a "not recommended" was wrong 2.5% of the time where the data was thin and 0.4% where it was not.
  So when support is thin, a programme that is not recommended but still has a 5% chance is shown as
  "we couldn't tell, you can still apply" rather than "unlikely". That cut wrong "doesn't look likely"
  answers from 2.17% to 0.71% [−1.66, −1.27 points] at about one such label a household, and changes
  no recommendation (D33, D35).
- **English and Spanish.** One switch in the header (Spanish is picked automatically for browsers that
  prefer it): every question, option, answer and result, with each agency's official Spanish page where
  it has one (Social Security, the IRS, CuidadoDeSalud.gov, InsureKidsNow), USA.gov's pages marked as
  English-only, and the National Hunger Hotline's Spanish line. AI-assisted translation, not yet
  reviewed by a native speaker.
- **Accessible.** axe-core 4.10.2 finds no WCAG 2.1 A/AA violations on any screen (start, question,
  results), in light and dark mode; one question per screen, labelled controls, focus moved to each
  new question, no colour-only meaning.
- **Deploys as is** to Vercel (`vercel.json`, `api/index.py`): the screener from the CDN, everything
  else from one Python function. CI runs the tests on Python 3.11 to 3.13 and smoke-tests the service
  over a real socket.
- **Documented** in a [model card](MODEL_CARD.md): data, tests, where it is wrong, what it must not be
  used for.

## Results

Held-out test, settings frozen on a validation split first, $/household/year, 95% paired bootstrap
intervals. Loss is the cost of mistakes: a missed programme at its value, a wrong recommendation at
$1,000. Full tables in [WRITEUP.md](WRITEUP.md) and `results/`.

| v2 engine (8.4 questions) against | difference in mistakes + $15 a question | in mistakes alone |
|---|---|---|
| v1 engine, same questioning (8.5 questions) | **−$281** [−437, −142] | −$280 [−437, −140] |
| v1 full questionnaire (18 questions) | **−$483** [−667, −317] | −$339 [−523, −173] |
| strongest decision tree, deciding with v2 (11.1 questions) | **−$293** [−485, −136] | −$253 [−446, −95] |
| best fixed 8-question form, deciding with v2 | **−$768** [−1,241, −398] | −$774 [−1,251, −405] |
| v2 asking all 18 questions (secondary) | −$79 [−188, +37] | +$64 [−44, +180] |

The four bold rows were registered before the run (D28) and all hold after Holm correction. The last
row is the trade-off in one line: asking 9.6 fewer questions costs no detectable accuracy. With noisy
answers (misreports, "don't know") no question policy separates from another (D28 part 2).

- **Tiered verdicts.** Recommendations it calls "likely" (50% or more) are right 78% of the time and
  find two thirds of all eligible programmes. Lower-probability ones are shown as "worth checking":
  the dollar-weighted objective recommends a $10,000 programme at a 9% chance, which is the right
  call for the person but should be labelled as such.
- **Against decision trees.** A cost-sensitive greedy tree on the same loss cost $700 a household
  more than the v1 engine (D25). The strongest tree we then built, which falls back to broader data
  where its branches run thin, came within noise of v1 (D27); deciding with the same v2 model, v2's
  engine beats it by $293 with 2.6 fewer questions.
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

# 3. tests (52), and tuning on the validation split only
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe experiments/outcome_v2.py            # outcome models, validation
.venv/Scripts/python.exe experiments/val_v2.py 400 WORKERS     # engine with each model, validation

# 4. the pre-registered test (D28): 2,000 fresh test households; WORKERS processes, ~1.5 GB each
.venv/Scripts/python.exe experiments/run_v2.py 2000 WORKERS
.venv/Scripts/python.exe experiments/run_v2.py 2000 1 analyse

# 5. the engine as one file, its MCP server, and the end-to-end agent test (D29; needs Claude Code)
.venv/Scripts/python.exe experiments/export_bundle.py bundles/benefits_v2.npz   # rebuilds the shipped bundle
.venv/Scripts/python.exe -m decisionintake.mcp_server                              # speaks MCP on stdio
.venv/Scripts/python.exe experiments/agent_e2e.py prep test 200
.venv/Scripts/python.exe experiments/agent_e2e.py run test engine haiku 200 3
.venv/Scripts/python.exe experiments/agent_e2e.py run test alone haiku 200 3
.venv/Scripts/python.exe experiments/agent_e2e.py analyse test haiku

# 6. the service, its rules file and their tests (D31, D32)
.venv/Scripts/python.exe experiments/guards_build.py          # rebuilds bundles/guards_2026.json, checks each rule
.venv/Scripts/python.exe experiments/run_guard.py 2000 WORKERS && .venv/Scripts/python.exe experiments/run_guard.py 2000 1 analyse
.venv/Scripts/python.exe experiments/agent_e2e.py run dev stateless haiku 100 5
.venv/Scripts/python.exe -m decisionintake.web --port 8080

# 7. the browser demo
.venv/Scripts/python.exe experiments/export_demo_v2.py 1500 demo/population.json
cd demo && ../.venv/Scripts/python.exe -m http.server 8765    # then open http://127.0.0.1:8765
```

Earlier experiments (D21-D27: v1 test, trees, run-time adaptation, the language layer) have their
own scripts in `experiments/`, listed in DECISIONS.md. Seeds are fixed everywhere, so a rerun gives
the same households and draws. Tested with Python 3.13, numpy 2.5.5, policyengine-us 2.152.15 on a
12-core CPU without a GPU. On Windows, PolicyEngine's longest file path exceeds the 260-character
limit unless the package sits under a short prefix: copy it to one and set `PE_SHORT_PATH` (D11).

## Scope

- **A research system, not an eligibility determination.** Results are agreement with PolicyEngine's
  2026 rules on a calibrated synthetic population, assuming everyone takes up what they're entitled
  to. A real screener sends people to the official application.
- **The language results are for 4B models on a CPU.** The demo uses Claude to fill the same slots;
  spoken answers in the tests come from templates, which are cleaner than real speech.
- **No immigration questions,** by design. Immigration status changes eligibility, so it remains an
  unasked source of error.
