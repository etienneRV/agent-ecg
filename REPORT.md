# Agent ECG: warning early when a coding agent is going to fail

*Team RAMadan · TimeNet hackathon, Round 1 · living report, last updated 2026-11-23*

> **Status:** dataset contributed to TimeNet ([OpenTSLM/TimeNet#832](https://github.com/OpenTSLM/TimeNet/pull/832),
> awaiting review) · baselines, three fine-tuned OpenTSLM models, a live-monitor replay and a second round of
> two runs, all scored on test (this report; every number in [results/SUMMARY.md](results/SUMMARY.md)).

## 1. In one paragraph

Coding agents work for dozens of steps on a task, and most of their runs fail. We want a monitor that
reads an agent's steps *while it works*, like an ECG reads a heartbeat, and says early whether the run
is heading for success. We turned 80,036 runs of SWE-agent on real GitHub issues into a TimeNet
dataset of **checkpoints**: the first 5, 10, 15, 20 or 30 steps of a run that is still going. A model sees
per-step signals, the agent's own reasoning and the issue, and must predict whether the run will
resolve the issue, how it will end, and write a short status report. On held-out repositories, a
simple model of the agent's thoughts reaches AUROC 0.68-0.71 at every checkpoint, against 0.61-0.62 for a
model that reads only the issue: the agent's behaviour carries signal beyond how hard the issue is.
That is the bar our OpenTSLM models had to clear. The best of them, ecg-full, comes within 0.03-0.04 AUROC of
the baseline that reads the same inputs at 5-20 steps (not a significant gap) but does not beat it. On pairs of
runs on the same issue, where the issue text cannot help, the two are level: the baseline's lead comes mostly from
telling hard issues from easy ones. As a live monitor it flags about a third of the runs that go on to fail, as the
signals baseline does.

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
  cut after k = 5, 10, 15, 20 and 30 steps, *only if it goes on after them*. No input shows the outcome,
  the exit status, the run's length, the patch or the test log.
- **Inputs per checkpoint:** nine per-step signals on an ordinal axis (thought, command and reply
  length; conversation length; repeated command; command kind: edit/create/run/read/search/submit/other;
  reply kind: linter rejection/timeout/traceback/command error/silent success/output; whether an edit
  changes the same file as the edit before; the kind of file an edit changes: source, test, or a file
  the agent created), the agent's thoughts as text, and the issue text.
- **Three tasks per checkpoint:** *Will it pass?* (resolved / unresolved), *How will it end?* (6
  endings), and a rule-made *status report* (what happened so far + how many more steps, how it ends,
  whether it resolves).
- **Split by repository** (8/1/1, stratified on the outcome): no repository or issue is in two
  partitions, because each issue has 22 runs on average.

| Checkpoint | Runs still going | Resolved | train / validation / test checkpoints |
| --- | --- | --- | --- |
| 5 steps | 76,182 | 16.6 % | 60,985 / 7,557 / 7,640 |
| 10 steps | 59,708 | 13.8 % | 48,056 / 5,649 / 6,003 |
| 15 steps | 43,550 | 9.2 % | 35,124 / 4,075 / 4,351 |
| 20 steps | 34,059 | 6.4 % | 27,587 / 3,107 / 3,365 |
| 30 steps | 22,479 | 4.2 % | 18,288 / 2,030 / 2,161 |

235,978 checkpoints and 707,934 tasks in all; test holds 143 repositories and 7,640 runs.

## 4. How we evaluate

The evaluation script (`agent-ecg evaluate`, `src/agent_ecg/metrics.py`) scores any model's predictions
on the held-out **test** split. Three choices matter:

1. **Never accuracy for pass/fail.** Always answering "fail" is right 83-96 % of the time. We report
   **AUROC** (with a 95 % interval from resampling whole test repositories), **average precision**
   (its chance level is the resolved rate), the **Brier score** (is the probability calibrated?) and
   **balanced accuracy** at a threshold chosen on validation.
2. **Per checkpoint.** The resolved rate falls from 17 % at 5 steps to 4 % at 30, so a pooled score
   mixes five different problems. We report k = 5, 10, 15, 20, 30 separately (and pooled for reference).
3. **Score the status report by its prediction only.** Its first four sentences restate counts the
   model can read off its input; only the last sentence predicts. A word-overlap score on the full
   text stays high when the prediction is wrong, so we parse the last sentence and score its three
   facts: remaining steps (absolute error), ending (accuracy, macro-F1), pass/fail (balanced accuracy).
   We print word overlap on the full text next to it to show the gap.
4. **Same-issue pairs.** AUROC over all runs also rewards judging the issue: some issues are harder
   than others, and each has 22 runs on average. So we also report, among pairs of runs on the same
   issue with one resolved and one not, the share where the model scores the resolved run higher
   (0.5 is chance). The issue is fixed within a pair, so only what the model reads from the run counts.

## 5. Baselines

All baselines run on a laptop CPU, train on train, tune on validation (by mean per-checkpoint
AUROC), and are scored on test. Each
checkpoint is weighted by 1 / (checkpoints of its run) so long, failing runs do not count five times.

| Baseline | Inputs | Model |
| --- | --- | --- |
| majority | none | resolved rate of the checkpoint; most common ending; median remaining steps |
| signals | the 9 per-step signals, summarised (kind shares overall and in the last 5 steps, lengths, conversation size) | gradient-boosted trees |
| thoughts | the agent's thoughts so far | TF-IDF of words and word pairs + logistic regression |
| signals+thoughts | both | logistic regression |
| issue | the issue text only (never the agent) | TF-IDF + logistic regression |
| issue+signals+thoughts | everything a full-context TSLM reads | logistic regression |

**Why the issue-only baseline.** Some issues are simply harder than others, and each is attempted 22
times on average. A model could score well by judging the issue ("issues like this usually fail")
without watching the agent at all, and the agent's thoughts often repeat the issue. If the
agent-based baselines beat the issue-only one, the monitor really learns from the agent's behaviour.

### Results on test (143 repositories never seen in training)

**Will it pass?** AUROC per checkpoint, with 95 % intervals from resampling test repositories
(chance = 0.5). Average precision is in parentheses; its chance level is the resolved rate (16 / 14 / 9 / 7 / 6 %).

| Baseline | 5 steps | 10 steps | 15 steps | 20 steps | 30 steps |
| --- | --- | --- | --- | --- | --- |
| majority | 0.50 (0.16) | 0.50 (0.14) | 0.50 (0.09) | 0.50 (0.07) | 0.50 (0.06) |
| issue only | 0.61 [0.54, 0.69] (0.25) | 0.62 [0.55, 0.69] (0.23) | 0.61 [0.52, 0.70] (0.15) | 0.62 [0.50, 0.70] (0.12) | 0.61 [0.46, 0.72] (0.10) |
| signals | 0.61 [0.57, 0.66] (0.24) | 0.65 [0.60, 0.68] (0.23) | 0.67 [0.61, 0.73] (0.19) | 0.71 [0.61, 0.79] (0.19) | **0.73** [0.63, 0.82] (0.18) |
| thoughts | 0.68 [0.63, 0.73] (0.28) | 0.70 [0.65, 0.75] (0.26) | 0.71 [0.64, 0.76] (0.22) | 0.70 [0.61, 0.77] (0.17) | 0.70 [0.60, 0.79] (0.13) |
| signals+thoughts | 0.67 [0.62, 0.73] (0.28) | 0.69 [0.64, 0.73] (0.24) | 0.71 [0.64, 0.77] (0.21) | 0.71 [0.63, 0.78] (0.16) | 0.70 [0.62, 0.77] (0.15) |
| issue+signals+thoughts | **0.69** [0.64, 0.75] (0.31) | **0.71** [0.65, 0.75] (0.29) | **0.72** [0.64, 0.77] (0.24) | **0.72** [0.64, 0.78] (0.16) | 0.70 [0.62, 0.77] (0.14) |

**Same-issue pairs** (share of resolved/unresolved pairs on the same issue where the resolved run
scores higher; 0.5 = chance; 27,526 / 17,091 / 6,964 / 3,203 / 1,223 pairs):

| Baseline | 5 steps | 10 steps | 15 steps | 20 steps | 30 steps |
| --- | --- | --- | --- | --- | --- |
| issue only | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| signals | 0.543 | **0.610** | **0.593** | **0.601** | **0.651** |
| thoughts | 0.544 | 0.562 | 0.576 | 0.572 | 0.576 |
| signals+thoughts | 0.549 | 0.579 | 0.583 | 0.555 | 0.590 |
| issue+signals+thoughts | **0.571** | 0.605 | 0.593 | 0.570 | 0.601 |

**Repository cross-validation** (`agent-ecg baselines --cv 5`): five folds over all 1,276
repositories, so every checkpoint is predicted once by a model that never saw its repository.
Mean ± sd of the fold AUROCs / same-issue pairs:

| Baseline | 5 steps | 10 steps | 15 steps | 20 steps | 30 steps |
| --- | --- | --- | --- | --- | --- |
| issue only | 0.616 ± 0.024 / 0.500 | 0.609 ± 0.029 / 0.500 | 0.602 ± 0.036 / 0.500 | 0.601 ± 0.037 / 0.500 | 0.603 ± 0.033 / 0.500 |
| signals | 0.663 ± 0.024 / 0.551 | 0.711 ± 0.015 / **0.615** | 0.729 ± 0.020 / **0.662** | 0.747 ± 0.018 / **0.686** | **0.775** ± 0.016 / **0.735** |
| thoughts | **0.700** ± 0.019 / 0.546 | **0.728** ± 0.019 / 0.593 | **0.735** ± 0.018 / 0.626 | **0.748** ± 0.020 / 0.649 | 0.769 ± 0.021 / 0.689 |

**What this shows:**
- **The agent's behaviour carries signal beyond the issue.** In a paired test that resamples the same
  test repositories for both models (1,000 resamples; `agent-ecg compare`, saved in `results/compare_*.txt`):

  | AUROC gain | 5 steps | 10 steps | 15 steps | 20 steps | 30 steps |
  | --- | --- | --- | --- | --- | --- |
  | thoughts over issue only | **+0.07** [+0.01, +0.13] | **+0.09** [+0.01, +0.15] | **+0.10** [+0.02, +0.18] | **+0.09** [+0.01, +0.17] | **+0.09** [+0.01, +0.17] |
  | signals over issue only | -0.00 [-0.07, +0.08] | +0.03 [-0.05, +0.10] | +0.06 [-0.02, +0.14] | +0.10 [-0.00, +0.20] | **+0.12** [+0.01, +0.23] |
  | signals over thoughts | **-0.07** [-0.11, -0.03] | **-0.06** [-0.09, -0.02] | -0.04 [-0.08, +0.01] | +0.01 [-0.05, +0.07] | +0.03 [-0.04, +0.11] |

- **The step signals are the ECG: the longer you watch, the more they say.** Their AUROC climbs from
  0.61 at 5 steps to 0.73 at 30 (cross-validated: 0.66 to 0.78) while the issue-only model stays flat
  at 0.60-0.62. By step 30 their gain over the issue is significant.
- **Within an issue, the signals beat the words.** The thoughts are strong across issues, but much of
  that is the issue showing through: they often restate it. On pairs of runs on the same issue the
  signals pick the resolved run 61 % of the time at step 10 and 65 % at step 30, against 56-58 % for
  the thoughts (cross-validated: 62 % and 74 % vs 59 % and 69 %).
- **The file-identity channels help a little.** On test the signals model rose from 0.63 / 0.65 / 0.67
  to 0.65 / 0.71 / 0.73 at 10 / 20 / 30 steps, but training also gained the 15-step checkpoints and
  the test split is noisy. The controlled comparison (cross-validation, same checkpoints, with and
  without the two channels) puts their gain at +0.006 / +0.010 at 10 / 20 steps.
- **Three other channels were tried and left out of the connector.** Under the same cross-validation
  (signals model without them: 0.676 / 0.705 / 0.688 at 5 / 10 / 20 steps), none helped: a repeated
  error line +0.001 / +0.002 / −0.007; whether a run command runs the tests or the agent's own script
  −0.003 / +0.002 / −0.006; phrases in the thoughts (changing approach, doubt, success, finding the
  cause, apologising) −0.003 / +0.002 / −0.002. File identity, for comparison: −0.001 / +0.006 / +0.010.
- **Fifteen more channels, from the raw step text, did not help either** ([results/channels.md](results/channels.md),
  `agent-ecg channels`; train + validation only). Test results, scripts fixed or still crashing,
  exception types and edit sizes, plus three re-expressions of the signals (failure streak, steps since
  a success, context growth): 8 of 15 passed screening, none passed confirmation (AUROC gain with a
  repository-bootstrap interval above 0). Gains were +0.001 to +0.006; a test summary appears in only
  2-3 % of runs by step 15. The 19 series stay as they are.
- **Adding the issue to the behaviour adds little.** issue+signals+thoughts is the best model on test
  at 5-20 steps by 0.01-0.02, within the intervals. It is the bar for a full-context TSLM.
- **The test split is a hard draw for the signals.** Cross-validation puts them at 0.711 ± 0.015 at 10
  steps; the single test split gives 0.645. Differences under about 0.05 on test are not reliable.
- **Pooled scores mislead.** The majority baseline has a pooled AUROC of 0.6 although it knows nothing
  about any run, because the resolved rate differs by checkpoint. Hence per-checkpoint scores.

**How will it end?** (signals baseline vs majority, test) Macro-F1 over the six endings is 0.13 / 0.18
/ 0.22 / 0.22 / 0.20 at 5 / 10 / 15 / 20 / 30 steps, against 0.08 / 0.10 / 0.12 / 0.14 / 0.15 for
always guessing the most common training ending (`context_limit`). It mostly separates `submitted`
from `context_limit`; `early_exit` (3 % of test checkpoints) is predicted 19 times.

**Status report.** The signals baseline predicts the remaining steps with a median error of 6 steps
(majority: 10). The table below shows why only the last sentence is scored: the majority baseline,
which predicts "will not resolve" for every run, gets 0.97 word-overlap F1 on the full text.

| Baseline | Word-overlap F1, full text | Word-overlap F1, last sentence | Pass/fail balanced accuracy | Remaining steps, median error |
| --- | --- | --- | --- | --- |
| majority | 0.968 | 0.874 | 0.500 | 10 |
| signals | 0.974 | 0.897 | 0.614 | 6 |

**The live monitor on the signals model** (`results/monitor.signals.md`, fitted on validation): the
critical zone is health 12-15 or less, where 95.0 / 96.6 / 99.0 / 98.9 / 99.3 % of validation runs
failed at 5 / 10 / 15 / 20 / 30 steps; it flags 14-17 % of the failing runs. `agent-ecg replay` picks
test runs from it: the top one (`results/replay.signals.falling.jsonl`) starts at health 94, falls to
31 at step 8 and turns critical at step 15, while the agent alternates `read` and `search` with
repeated commands; it ran into the context limit and did not resolve its issue.

Full tables: [`results/baselines.test.md`](results/baselines.test.md); JSON per baseline in `results/`.

## 6. Fine-tuning OpenTSLM, and the live monitor

- **Inputs:** the nine step signals as 19 labelled series (four z-normalised lengths, fifteen 0/1
  indicators), with, by export, nothing else (`--context none`, the ECG alone, 443-558 tokens), the
  agent's thoughts (`thoughts`), or the issue and the thoughts (`full`, 994-1,839 tokens).
- **No counts in the series labels.** An earlier export labelled each 0/1 series with its count ("2
  of 5 steps"), so the model could answer from the text and never read the series. The labels now
  say only what a 1 means.
- **Runs** (`GPU.md`), all starting from OpenTSLM's ECG model (`OpenTSLM/llama-3.2-1b-ecg-sp`,
  the last stage of its curriculum, which reads 12 labelled leads at once):
  1. **ecg: the ECG alone, pass/fail.** Whatever it gets right, it read from the trace; this is the
     demo's model.
  2. **ecg-thought: thoughts beside the ECG, pass/fail.** Against the thoughts baseline, it shows what the
     series add to the text.
  3. **ecg-full: everything, all three tasks**, against the issue+signals+thoughts baseline.
- **Training:** each answer counts equally in the loss, resolved runs get 30 % of the sampling
  weight, and the best checkpoint is chosen by validation pass/fail AUROC.
- **The monitor** (`src/agent_ecg/monitor.py`), fitted on validation predictions only: an isotonic
  map per checkpoint gives calibrated probabilities (the reweighting inflates the raw ones); health
  from 0 to 100 is the percentile of a run's score among validation runs still going at the same
  step; the alarm goes on when health is at most 15 in a zone where at least 95 % of validation runs
  failed, two steps in a row, and holds until health is back to 30. `agent-ecg replay` and `opentslm/replay.py` replay test runs step by
  step and write one JSON line per step for the terminal ECG.
- **Bar to clear:** for ecg, the signals baseline on the same inputs (0.61 / 0.65 / 0.67 / 0.71 / 0.73
  on test; same-issue pairs 0.61 at 10 steps); for ecg-thought, the thoughts baseline (0.68 / 0.70 / 0.71 /
  0.70 / 0.70); for ecg-full, issue+signals+thoughts (0.69 / 0.71 / 0.72 / 0.72 / 0.70). And always clearly
  above the issue-only model (0.61-0.62), so the gain comes from watching the agent.


### Results on test

| Model | Inputs | 5 steps | 10 steps | 15 steps | 20 steps | 30 steps |
|---|---|---|---|---|---|---|
| ecg | the ECG | 0.583 | 0.611 | 0.616 | 0.620 | 0.615 |
| ecg-thought | thoughts + ECG | 0.624 | 0.638 | 0.643 | 0.647 | 0.596 |
| ecg-full (3 tasks) | issue + thoughts + ECG | 0.657 | 0.677 | 0.684 | 0.679 | 0.624 |
| signals baseline (bar for ecg) | the signals | 0.611 | 0.645 | 0.674 | 0.712 | 0.728 |
| thoughts baseline (bar for ecg-thought) | thoughts | 0.680 | 0.703 | 0.710 | 0.701 | 0.701 |
| issue+signals+thoughts baseline (bar for ecg-full) | all | 0.691 | 0.709 | 0.720 | 0.719 | 0.700 |

- **No model clears its bar.** ecg-full comes closest: 0.03-0.04 below at 5-20 steps, inside the paired
  repository-bootstrap interval, and significantly below at 30. ecg-thought and ecg are significantly below theirs
  at most checkpoints. Averaging ecg-full with its baseline adds nothing (+0.004 at most).
- **On pairs of runs on the same issue, ecg-full is level with its baseline.** AUROC over all runs also rewards
  telling hard issues from easy ones, which the issue text alone does (0.61). On pairs of a resolved and an
  unresolved run on the same issue, only what a model reads from the run counts: there ecg-full and the
  issue+signals+thoughts baseline differ by -0.009 to +0.031 at 5-20 steps, every interval around zero
  (`agent-ecg compare --same-issue`, [results/same_issue/](results/same_issue/)). Over all runs the gap at those
  steps is 0.03-0.04, so most of the baseline's lead comes from telling issues apart (point estimates).
- **The ECG alone tells two attempts at the same issue apart:** the signals baseline picks the run that succeeds
  in 54 % of such pairs at 5 steps and 65 % at 30 (chance 50 %; above chance at 5, 10, 15 and 30 steps). ecg, which
  reads the same series, is below it on these pairs at every checkpoint, significantly at 10 and 30 steps.
- **Early the text carries the signal, later the behaviour does.** The signals baseline climbs from 0.611 to 0.728
  as runs get longer; ecg stays flat at about 0.61-0.62, so the TSLM does not use the longer trace.
- **As a live monitor** (600 unseen test runs replayed step by step), ecg-full flags 32 % of the runs that go on to
  fail, with a median alarm at step 6 and 18 % false alarms on resolved runs; the signals baseline flags 34 %, at
  step 9, with 15 %. They flag mostly different runs. Half of ecg-full's alarms fire at the earliest possible step
  and cluster by issue, so part of its head start comes from how hard the issue is.
- **ecg-full's other two tasks:** it predicts the ending as well as the signals baseline (accuracy 0.662 against
  0.655) and writes status reports in the right format 99.7 % of the time, but their forecast is weaker
  (remaining steps off by 9 at the median, against 6).
- **A second round of two runs** fixed what validation checks pointed to: ecg v2 adds three count channels the
  model was not forming (steps since the last success, tracebacks so far, code runs so far), and ecg-full v2 trains
  against overfitting and keeps the first thoughts of long runs. Judged by rules fixed before training, neither
  improved significantly: ecg v2 changed little, and ecg-full v2 gained +0.018 at 20 steps and +0.036 at 30, as
  expected, but within intervals of 0.05-0.12. ecg-full v2's gap to its baseline is no longer significant at any
  checkpoint.
- **Shortcut checks:** which agent LLM wrote a run does not drive any score (within 0.012 within each agent);
  run length is handled by per-checkpoint scores ([results/shortcut_checks.md](results/shortcut_checks.md)).

Every number, with intervals and sources: [results/SUMMARY.md](results/SUMMARY.md); training details: `GPU.md`.

### Why the baselines win

- **Early in a run, most of the signal is how hard the issue is,** and that shows in the words used. A linear
  model of word counts over the whole issue and every thought does this well. The TSLM reads a trimmed prompt and
  has 2,853 training issues to learn it from. On same-issue pairs, where difficulty cancels out, it is level.
- **Later, the behaviour carries the signal.** The trees get summaries we designed (the share of each kind of step,
  over the run and over its last five steps). The TSLM must form them from raw series, four steps per patch,
  through an encoder pretrained on heart ECG waveforms. ecg reads the ECG less well than the trees, and handing it
  the counts as extra series (ecg v2) did not help, although the same counts help a simple outside model.
- **The kept checkpoint was picked on a noisy score.** ecg-full's validation AUROC moved between 0.71 and 0.77 from
  one evaluation to the next (2,000 checkpoints each), and the best of 15 was kept: 0.768, against 0.66-0.68 on test at 5-20 steps.
  ecg-full v2, scored on 4,000 validation checkpoints, peaked at 0.739.
- **Others find the same.** Removing the language model from LLM-based forecasters does not hurt them
  ([Tan et al., NeurIPS 2024](https://arxiv.org/abs/2406.16964)), and tree-based models still beat deep networks on
  tabular data ([Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)). The signals baseline's inputs
  are such tabular summaries.

### Why predicting failure early is hard, and what remembering earlier attempts gains

Computed by `scripts/history_monitor.py`; every number is in [results/history_monitor.md](results/history_monitor.md).

- **The issue decides most of the outcome.** 77 % of the 3,586 issues are never solved in any attempt, 1 % always,
  22 % sometimes; which issue it is explains 52 % of the variance in outcomes. The success rate of the other
  attempts at the same issue ranks a test run with AUROC 0.93 at 5 steps (0.87 at 30); our model reaches 0.66.
- **The first steps reveal little of it.** On pairs of attempts at the same issue, every model ranks only 54 to 57 %
  of pairs right at 5 steps (section 6, same-issue pairs); the attempts drift apart later.
- **The training data is smaller than it looks for this.** 60,985 training runs, but 2,853 issues, of which only 659
  were ever solved: a 1B model learns which issues are solvable from a few hundred examples and memorizes them
  (its validation AUROC peaks early, and test is about 0.09 below validation).
- **OpenTSLM's encoder fits this data poorly.** It was pretrained on heart ECG waveforms, and it groups steps in
  patches of four, padded with zeros without a mask: in a 5-step snapshot, 3 of the 8 positions are padding that
  looks like "nothing happened". The ECG-only TSLM stays flat while trees on simple counts climb.
- **Memory across attempts works.** An attempt succeeds 16.6 % of the time, but 8.1 % after one failed attempt at the
  same issue, 3.4 % after three and 2.0 % after five. Combining our model with the outcome of three earlier attempts
  (drawn at random among the other attempts at the issue; logistic regression cross-fitted over two halves of the
  test repositories) lifts AUROC from 0.657 to 0.852 at 5 steps, +0.195 [+0.139, +0.250]; the gain is significant
  at every checkpoint. Three earlier attempts alone already reach 0.848: the history carries most of it.

## 7. Limitations

- One agent framework (SWE-agent) and one model family (Llama 3.1; 93 % of runs from the 70B model).
  A monitor trained here may not transfer to other agents without new data.
- Signals and kinds come from fixed text rules (listed in the dataset card); the status report is a
  template, not an expert's assessment.
- The label is the release's test result, which can miss correct fixes or accept incomplete ones.
- No baseline reads the steps as a sequence; that is the TSLM's job.
- The baselines win: no fine-tuned model beats the baseline that reads its inputs (on same-issue pairs, ecg-full
  is level with it). We did not vary the base model
  (Llama 3.2 1B), the starting checkpoint or the training setup, so we cannot say which of these limits the TSLM.
- Whether ecg-full relies on the ECG is untested: a test that shuffles its series between runs was not run. ecg,
  which reads only the ECG, scores above chance.
- We did not weight training by issue, mask the zero padding of OpenTSLM's four-step patches, try a per-step
  encoder, or run several seeds and a hyperparameter search; each could matter given the diagnosis above.
- The history-aware monitor needs several attempts at the same issue, and draws "earlier" attempts at random
  among the others, which assumes attempts are interchangeable.
- The test set is small for differences between models: paired intervals are 0.02-0.12 wide, so gains of
  +0.02-0.04, like the second round's, can be neither confirmed nor ruled out.

## 8. Reproduce

```bash
git clone https://github.com/etienneRV/agent-ecg && cd agent-ecg && uv sync
uv run agent-ecg prepare         # ~5 min, 1.1 GB download
uv run agent-ecg baselines       # ~21 min, peak 5.8 GB RAM, on a laptop CPU
uv run agent-ecg baselines --cv 5   # ~10 min, peak 4.0 GB
uv run agent-ecg replay --top 3  # the live monitor on test runs
uv run agent-ecg channels        # the channel hunt, ~70 min on 5 CPU threads
```

Fine-tuning needs a GPU: `GPU.md` (runs ecg, ecg-thought, ecg-full) and `opentslm/final_run.sh c2|a2` (the second
round). The website is built from `results/` with `uv run python scripts/charts.py --theme both` and
`uv run python scripts/site.py`.
