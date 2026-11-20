# Agent ECG: warning early when a coding agent is going to fail

*Team RAMadan · TimeNet hackathon, Round 1 · living report, last updated 2026-11-20*

> **Status:** dataset contributed to TimeNet ([OpenTSLM/TimeNet#832](https://github.com/OpenTSLM/TimeNet/pull/832),
> awaiting review) · evaluation and CPU baselines done (this report) · OpenTSLM fine-tune: next, on
> Nebius compute.

## 1. In one paragraph

Coding agents work for dozens of steps on a task, and most of their runs fail. We want a monitor that
reads an agent's steps *while it works*, like an ECG reads a heartbeat, and says early whether the run
is heading for success. We turned 80,036 runs of SWE-agent on real GitHub issues into a TimeNet
dataset of **checkpoints**: the first 5, 10, 20 or 30 steps of a run that is still going. A model sees
per-step signals, the agent's own reasoning and the issue, and must predict whether the run will
resolve the issue, how it will end, and write a short status report. On held-out repositories, a
simple model of the agent's thoughts reaches AUROC 0.68-0.70 at every checkpoint, against 0.61 for a
model that reads only the issue: the agent's behaviour carries signal beyond how hard the issue is.
That is the bar our OpenTSLM model has to clear.

## 2. The problem

- **Agents fail often and expensively.** In this data, 83 % of runs do not resolve their issue, and
  failing runs are the long ones: they burn compute until they hit the context limit.
- **Today the verdict comes at the end** (tests pass or fail). A live signal would let an operator, or
  the agent itself, stop, restart or escalate a run that is going wrong.
- **An agent's run is a time series.** Each step has a command, a reply and a thought; their kinds,
  lengths and repetitions form a trace with patterns (loops, error bursts, a context that fills up).
  That is exactly what a time-series language model (TSLM) reads, alongside text.

## 3. The data

**Source.** [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
(CC BY 4.0; model outputs also under the Llama 3.1 License): 80,036 runs of SWE-agent (Llama 3.1 8B /
70B / 405B) on 3,591 issues from 1,276 repositories, each labelled with whether its final patch
passed the issue's tests (13,389 resolved, 16.7 %).

**Our TimeNet connector** (`nebius/swe_agent_trajectories`, PR #832) downloads the release at a pinned
revision and converts it:

- **Checkpoints, never whole runs.** The end of a run gives its outcome away (every submitted run ends
  with `submit`; 36 % of failed runs end at the context limit vs 5 % of resolved ones). So each run is
  cut after k = 5, 10, 20 and 30 steps, *only if it goes on after them*. No input shows the outcome,
  the exit status, the run's length, the patch or the test log.
- **Inputs per checkpoint:** seven per-step signals on an ordinal axis (thought, command and reply
  length; conversation length; repeated command; command kind: edit/create/run/read/search/submit/other;
  reply kind: linter rejection/timeout/traceback/command error/silent success/output), the agent's
  thoughts as text, and the issue text.
- **Three tasks per checkpoint:** *Will it pass?* (resolved / unresolved), *How will it end?* (6
  endings), and a rule-made *status report* (what happened so far + how many more steps, how it ends,
  whether it resolves).
- **Split by repository** (8/1/1, stratified on the outcome): no repository or issue is in two
  partitions, because each issue has 22 runs on average.

| Checkpoint | Runs still going | Resolved | train / validation / test checkpoints |
| --- | --- | --- | --- |
| 5 steps | 76,182 | 16.6 % | 60,985 / 7,557 / 7,640 |
| 10 steps | 59,708 | 13.8 % | 48,056 / 5,649 / 6,003 |
| 20 steps | 34,059 | 6.4 % | 27,587 / 3,107 / 3,365 |
| 30 steps | 22,479 | 4.2 % | 18,288 / 2,030 / 2,161 |

192,428 checkpoints and 577,284 tasks in all; test holds 143 repositories and 7,640 runs.

## 4. How we evaluate

The evaluation script (`agent-ecg evaluate`, `src/agent_ecg/metrics.py`) scores any model's predictions
on the held-out **test** split. Three choices matter:

1. **Never accuracy for pass/fail.** Always answering "fail" is right 83-96 % of the time. We report
   **AUROC** (with a 95 % interval from resampling whole test repositories), **average precision**
   (its chance level is the resolved rate), the **Brier score** (is the probability calibrated?) and
   **balanced accuracy** at a threshold chosen on validation.
2. **Per checkpoint.** The resolved rate falls from 17 % at 5 steps to 4 % at 30, so a pooled score
   mixes four different problems. We report k = 5, 10, 20, 30 separately (and pooled for reference).
3. **Score the status report by its prediction only.** Its first three sentences restate counts the
   model can read off its input; only the last sentence predicts. A word-overlap score on the full
   text stays high when the prediction is wrong, so we parse the last sentence and score its three
   facts: remaining steps (absolute error), ending (accuracy, macro-F1), pass/fail (balanced accuracy).
   We print word overlap on the full text next to it to show the gap.

## 5. Baselines

All baselines run on a laptop CPU, train on train, tune on validation, and are scored on test. Each
checkpoint is weighted by 1 / (checkpoints of its run) so long, failing runs do not count four times.

| Baseline | Inputs | Model |
| --- | --- | --- |
| majority | none | resolved rate of the checkpoint; most common ending; median remaining steps |
| signals | the 7 per-step signals, summarised (kind shares overall and in the last 5 steps, lengths, conversation size) | gradient-boosted trees |
| thoughts | the agent's thoughts so far | TF-IDF of words and word pairs + logistic regression |
| signals+thoughts | both | logistic regression |
| issue | the issue text only (never the agent) | TF-IDF + logistic regression |

**Why the issue-only baseline.** Some issues are simply harder than others, and each is attempted 22
times on average. A model could score well by judging the issue ("issues like this usually fail")
without watching the agent at all, and the agent's thoughts often repeat the issue. If the
agent-based baselines beat the issue-only one, the monitor really learns from the agent's behaviour.

### Results on test (143 repositories never seen in training)

**Will it pass?** AUROC per checkpoint, with 95 % intervals from resampling test repositories
(chance = 0.5). Average precision is in parentheses; its chance level is the resolved rate (16 / 14 / 7 / 6 %).

| Baseline | 5 steps | 10 steps | 20 steps | 30 steps |
| --- | --- | --- | --- | --- |
| majority | 0.50 (0.16) | 0.50 (0.14) | 0.50 (0.07) | 0.50 (0.06) |
| issue only | 0.61 [0.54, 0.69] (0.24) | 0.62 [0.55, 0.69] (0.22) | 0.61 [0.50, 0.70] (0.11) | 0.61 [0.46, 0.71] (0.09) |
| signals | 0.61 [0.56, 0.66] (0.24) | 0.63 [0.58, 0.67] (0.21) | 0.65 [0.56, 0.73] (0.12) | 0.67 [0.58, 0.77] (0.11) |
| thoughts | **0.68** [0.63, 0.73] (0.28) | **0.70** [0.65, 0.75] (0.26) | 0.70 [0.61, 0.77] (0.17) | **0.70** [0.60, 0.79] (0.13) |
| signals+thoughts | 0.68 [0.63, 0.73] (0.29) | 0.69 [0.64, 0.74] (0.25) | **0.71** [0.63, 0.78] (0.16) | 0.70 [0.62, 0.77] (0.15) |

**What this shows:**
- **The agent's behaviour carries signal beyond the issue.** Thoughts beat the issue-only model at
  every checkpoint, and the gain holds up in a paired test that resamples the same test repositories
  for both models (1,000 resamples; `agent-ecg compare`, saved in `results/compare_*.txt`):

  | AUROC gain over issue-only | 5 steps | 10 steps | 20 steps | 30 steps |
  | --- | --- | --- | --- | --- |
  | thoughts | **+0.07** [+0.01, +0.13] | **+0.09** [+0.02, +0.15] | **+0.09** [+0.02, +0.17] | **+0.09** [+0.01, +0.19] |
  | signals | -0.01 [-0.08, +0.07] | +0.02 [-0.05, +0.08] | +0.04 [-0.07, +0.13] | +0.06 [-0.05, +0.19] |

  The thoughts often restate the issue, so part of their score may still be issue difficulty; the
  gain is what watching the agent adds on top.
- **The step signals trend upward as the run goes on** (0.61 at 5 steps to 0.67 at 30) while the
  issue-only model stays flat at 0.61, which is the ECG idea: the longer you watch, the more the trace
  says. With this test set the trend is not yet significant (the intervals above include 0).
- **Combining signals and thoughts in a linear model adds nothing yet.** A model that reads the trace
  as a sequence, such as a TSLM, is the obvious next step.
- **The intervals are wide** (test has 143 repositories, and only 127 resolved runs at 30 steps), so
  differences under about 0.05 are not reliable.
- **Pooled scores mislead.** The majority baseline has a pooled AUROC of 0.60 although it knows
  nothing about any run, because the resolved rate differs by checkpoint. Hence per-checkpoint scores.

**How will it end?** (signals baseline vs majority, test) Macro-F1 over the six endings is 0.13 / 0.17
/ 0.22 / 0.20 at 5 / 10 / 20 / 30 steps, against 0.13 / 0.12 / 0.07 / 0.04 for always guessing
`submitted`. It mostly separates `submitted` from `context_limit`; `early_exit` (3 % of test
checkpoints) is predicted only 3 times.

**Status report.** The signals baseline predicts the remaining steps with a median error of 6 steps
(majority: 9). The table below shows why only the last sentence is scored: the majority baseline,
which predicts "will not resolve" for every run, gets 0.96 word-overlap F1 on the full text.

| Baseline | Word-overlap F1, full text | Word-overlap F1, last sentence | Pass/fail balanced accuracy | Remaining steps, median error |
| --- | --- | --- | --- | --- |
| majority | 0.963 | 0.877 | 0.500 | 9 |
| signals | 0.968 | 0.897 | 0.597 | 6 |

Full tables: [`results/baselines.test.md`](results/baselines.test.md); JSON per baseline in `results/`.

## 6. Next: fine-tuning OpenTSLM

- **Input:** the seven step signals as time series, the agent's thoughts and the issue as text.
- **Targets:** the three tasks above; the status report teaches the model to describe the trace before
  it predicts.
- **Plan:** a small run on the laptop to test the data pipeline, then the full fine-tune on Nebius GPUs;
  scored with the same script, against the baselines above.
- **Bar to clear:** AUROC above the thoughts baseline (0.68 / 0.70 / 0.70 / 0.70 at 5 / 10 / 20 / 30 steps), and clearly above the issue-only model, so the gain comes from watching the agent.

## 7. Limitations

- One agent framework (SWE-agent) and one model family (Llama 3.1; 93 % of runs from the 70B model).
  A monitor trained here may not transfer to other agents without new data.
- Signals and kinds come from fixed text rules (listed in the dataset card); the status report is a
  template, not an expert's assessment.
- The label is the release's test result, which can miss correct fixes or accept incomplete ones.
- No baseline yet combines the issue with the agent's behaviour, or reads the steps as a sequence.

## 8. Reproduce

```bash
git clone https://github.com/etienneRV/agent-ecg && cd agent-ecg && uv sync
uv run agent-ecg prepare     # ~10 min, 1.1 GB download
uv run agent-ecg baselines   # ~9 min, peak 2.3 GB RAM, on a laptop CPU
```
