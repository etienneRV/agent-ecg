# Results at a glance

Where each number comes from, for the website and the pitch. Test AUROC is reported at 5 / 10 / 15 / 20 / 30
steps, always per checkpoint (runs of the same length compared with each other).

## Data

TimeNet connector for `nebius/SWE-agent-trajectories` (upstream PR OpenTSLM/TimeNet#832): 76,182 runs,
235,978 checkpoints at 5 / 10 / 15 / 20 / 30 steps, 9 per-step signals. The test split holds 143
repositories and 7,640 runs never seen in training. 83 % of runs fail: 83 % of those still going at 5 steps,
96 % at 30. Details: [REPORT.md](../REPORT.md).

The "ECG" a model reads: 19 series, one value per step. 4 lengths (characters of the thought, the command,
the reply, and the whole conversation) and 15 on/off events (what the agent did: edit, create, run, read,
search; how the environment replied: rejected edit, traceback, command error, timeout, silent success; and
patterns: repeated command, same file as the last edit, edit of a source, test or own file). It is a
multichannel event trace, not a waveform.

## Pass/fail prediction on test (AUROC)

| model | reads | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| baseline: signals | ECG signals | 0.611 | 0.645 | 0.674 | 0.712 | 0.728 |
| baseline: thoughts | agent's thoughts | 0.680 | 0.703 | 0.710 | 0.701 | 0.701 |
| baseline: issue + signals + thoughts | all | 0.691 | 0.709 | 0.720 | 0.719 | 0.700 |
| baseline: issue only | issue text | 0.613 | 0.619 | 0.612 | 0.616 | 0.614 |
| ecg | ECG | 0.583 | 0.611 | 0.616 | 0.620 | 0.615 |
| ecg-thought | thoughts + ECG | 0.624 | 0.638 | 0.643 | 0.647 | 0.596 |
| ecg-full (3 tasks) | issue + thoughts + ECG | 0.657 | 0.677 | 0.684 | 0.679 | 0.624 |

ecg, ecg-thought and ecg-full are TSLMs: OpenTSLM-SP on Llama 3.2 1B, fine-tuned from `OpenTSLM/llama-3.2-1b-ecg-sp` on one L40S.
Sources: [baselines.test.md](baselines.test.md), [baselines/](baselines/), [opentslm/](opentslm/),
[GPU.md](../GPU.md). Run folders: ecg = `ecg-only`, ecg-thought = `thoughts-outcome`, ecg-full = `full-multi`
(older commits call them C, B, A); the baselines are the CPU models.

- Each TSLM is below the baseline with the same inputs. ecg-full is closest: 0.03-0.04 below issue + signals +
  thoughts, not significant at 5-20 steps (paired repository bootstrap, `opentslm/full-multi/compare_vs_*`).
- ecg-full beats ecg-thought by 0.03-0.04 at every checkpoint (borderline at 10 and 15 steps); ecg-full differs in two ways (the
  issue, and three tasks), so the gain cannot be credited to either.
- **ecg-full + the baseline together add nothing:** an equal-weight average of their per-checkpoint ranks (nothing
  fitted on test) scores +0.002 / +0.004 / +0.001 / +0.001 / -0.018 over the baseline alone, every interval
  around zero ([ensemble/](ensemble/)).
- Early on, the text carries most of the signal (the issue and thoughts are complete from step 1); later, the
  behaviour does: the signals baseline climbs from 0.611 to 0.728 while ecg stays flat.

## Same-issue pairs: where the baselines' lead comes from

AUROC over all test runs rewards two things: telling hard issues from easy ones (the issue text alone scores 0.61)
and telling a good run from a bad one. The same-issue pair share keeps only the second: over pairs of a resolved and
an unresolved run on the same issue, the share where the resolved run scores higher (chance 0.5; the issue-only
baseline is 0.5 by construction). It is the question only a monitor can answer: is *this* attempt going well?
Paired repository bootstrap: `agent-ecg compare --same-issue` ([same_issue/](same_issue/)).

| model | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| baseline: issue + signals + thoughts | 0.571 | 0.605 | 0.593 | 0.570 | 0.601 |
| ecg-full | 0.569 | 0.596 | 0.597 | 0.601 | 0.539 |
| ecg-full v2 | 0.550 | 0.591 | 0.582 | 0.586 | 0.612 |
| baseline: signals (the ECG) | 0.543 | 0.610 | 0.593 | 0.601 | 0.651 |
| baseline: thoughts | 0.544 | 0.562 | 0.576 | 0.572 | 0.576 |
| ecg (the ECG) | 0.534 | 0.552 | 0.537 | 0.544 | 0.582 |
| baseline: issue only | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| pairs | 27,526 | 17,091 | 6,964 | 3,203 | 1,223 |

Gaps with 95 % intervals (bold: the interval excludes zero):

| comparison | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| ecg-full − baseline (all inputs), AUROC over all runs | -0.034 [-0.086, +0.014] | -0.032 [-0.079, +0.019] | -0.036 [-0.079, +0.009] | -0.041 [-0.092, +0.009] | **-0.077 [-0.158, -0.001]** |
| ecg-full − baseline (all inputs), same-issue pairs | -0.001 [-0.035, +0.035] | -0.009 [-0.041, +0.017] | +0.004 [-0.039, +0.045] | +0.031 [-0.018, +0.091] | -0.062 [-0.138, +0.041] |
| ecg-full v2 − baseline (all inputs), same-issue pairs | -0.021 [-0.050, +0.010] | -0.014 [-0.048, +0.017] | -0.011 [-0.046, +0.023] | +0.017 [-0.025, +0.044] | +0.011 [-0.071, +0.076] |
| signals − chance, same-issue pairs | **+0.043 [+0.001, +0.090]** | **+0.110 [+0.056, +0.157]** | **+0.093 [+0.003, +0.178]** | +0.101 [-0.023, +0.236] | **+0.151 [+0.016, +0.314]** |
| signals − thoughts, same-issue pairs | -0.001 [-0.030, +0.028] | **+0.049 [+0.005, +0.094]** | +0.017 [-0.028, +0.067] | +0.029 [-0.057, +0.133] | +0.075 [-0.028, +0.206] |
| ecg − signals, same-issue pairs | -0.008 [-0.032, +0.020] | **-0.058 [-0.109, -0.014]** | -0.056 [-0.123, +0.005] | -0.057 [-0.170, +0.038] | **-0.069 [-0.162, -0.007]** |

- **On same-issue pairs, ecg-full is level with the baseline that reads the same inputs** at every checkpoint
  (ecg-full v2 too). Over all runs the gap is 0.03-0.04 at 5-20 steps; on same-issue pairs it is -0.009 to +0.031.
  Most of the baseline's lead therefore comes from telling hard issues from easy ones (point estimates: the
  intervals of the two gaps overlap).
- **The ECG alone tells two attempts at the same issue apart,** above chance at 5, 10, 15 and 30 steps, and ahead of
  the thoughts at 10 steps; its lead over the thoughts is larger later (+0.03, +0.08) but not significant.
- **The TSLM reads the ECG less well than trees do:** ecg is below the signals baseline on same-issue pairs,
  significantly at 10 and 30 steps.
- Why the baselines win, in four points: [REPORT.md, section 6](../REPORT.md#why-the-baselines-win).
- Figures (`uv run python scripts/paper_figures.py`): [figures/results.png](figures/results.png) (four panels, PDF and SVG
  alongside) and [figures/social.png](figures/social.png) (16:9, for posts).

## Why early prediction is hard, and a monitor that remembers earlier attempts

From `scripts/history_monitor.py` ([history_monitor.md](history_monitor.md)):

- 77 % of issues are never solved in any attempt, 1 % always, 22 % sometimes; the issue explains 52 % of the
  variance in outcomes. Training: 60,985 runs, 2,853 issues, 659 ever solved.
- The other attempts' success rate at the same issue ranks a test run with AUROC 0.930 / 0.922 / 0.908 / 0.905 /
  0.870 at 5 / 10 / 15 / 20 / 30 steps; ecg-full alone: 0.657 / 0.677 / 0.684 / 0.679 / 0.624.
- Chance an attempt succeeds after m failed attempts at the same issue: 16.6 % (m = 0), 8.1 % (1), 3.4 % (3),
  2.0 % (5).
- ecg-full + 3 earlier attempts (cross-fitted logistic regression): 0.852 / 0.843 / 0.833 / 0.841 / 0.816, a gain of
  +0.195 [+0.139, +0.250] at 5 steps and significant at every checkpoint; 3 earlier attempts alone: 0.848 / 0.834 /
  0.820 / 0.821 / 0.822.

## ecg-full's other two tasks (full test split, 23,520 checkpoints)

- **Ending** (6 classes): accuracy 0.662, macro-F1 0.222; signals baseline 0.655 / 0.225; majority 0.494 / 0.110.
  About level with the signals baseline.
- **Status report** (generated text): 99.7 % follow the format. The forecast sentence is weaker than the
  signals baseline: remaining steps, mean error 17.8 (median 9) vs 14.3 (median 6); pass/fail balanced accuracy
  0.568 vs 0.614.
- Source: [opentslm/full-multi.alltasks/report.test.json](opentslm/full-multi.alltasks/report.test.json) (the
  generated reports are in its `predictions.test.parquet`); the earlier 4,000-checkpoint subset agrees within
  0.005 ([opentslm/full-multi.alltasks-4000/](opentslm/full-multi.alltasks-4000/)).

## The live monitor (600 unseen test runs, replayed step by step)

| monitor | failing runs flagged | median alarm step | false alarms on resolved runs | alarms that were right* |
|---|---|---|---|---|
| ecg-full | 32 % | 6 | 18 % | 96 % |
| signals baseline | 34 % | 9 | 15 % | 97 % |

\* at the natural 7 % share of resolved runs among runs that last 20 steps.

They flag mostly different runs, but combining them adds no accuracy (above). Half of ecg-full's alarms fire at the
earliest possible step and cluster by issue: ecg-full partly reads how hard the issue is. Demo run resting on
behaviour: `train-00010-of-00012:721` (ecg-full alarms at step 12, the signals monitor never, and ecg-full spares the 5 other
runs of the same issue); at step 12 it is re-editing the same file after the linter rejected a syntax error
([frame](opentslm/scan/demo.A.showcase.step12.svg), replays for both monitors:
[ecg-full](opentslm/scan/replay.showcase.A.jsonl), [signals](opentslm/scan/replay.showcase.signals.jsonl); play it with
`agent-ecg demo results/opentslm/scan/replay.showcase.A.jsonl --run-id train-00010-of-00012:721`). Source:
[opentslm/scan/summary.md](opentslm/scan/summary.md); signals baseline monitor replays:
[replay.signals.falling.jsonl](replay.signals.falling.jsonl), [demo.falling.step15.svg](demo.falling.step15.svg).

## Second round: ecg v2 and ecg-full v2

Each run fixes one problem found in the first round and was judged by a rule fixed before training (GPU.md section 6): ecg v2 needs a gain over ecg whose 95 % interval is above zero at two of 5 / 10 / 15
steps; ecg-full v2 at two of 15 / 20 / 30 (where its fix acts), with 5 / 10 not worse. Neither passed, so ecg and
ecg-full stay the reported models. Bold: the interval excludes zero (paired repository bootstrap on test).

| run | what changed | test AUROC 5 / 10 / 15 / 20 / 30 | gain over its original, with 95 % interval (5 / 10 / 15 / 20 / 30) | verdict |
|---|---|---|---|---|
| ecg v2 | ecg + 3 count channels (steps since the last success, tracebacks so far, code runs so far) | 0.600 / 0.614 / 0.608 / 0.616 / 0.642 | +0.017 [-0.003, +0.037]; +0.004 [-0.008, +0.015]; -0.008 [-0.025, +0.008]; -0.003 [-0.026, +0.018]; +0.027 [-0.001, +0.054] | tried, no gain |
| ecg-full v2 | ecg-full with 1 epoch, the natural share of successes, LoRA dropout 0.1, finer early stopping, and the first 512 + latest 1,536 thought tokens | 0.662 / 0.687 / 0.681 / 0.696 / 0.659 | +0.005 [-0.042, +0.055]; +0.009 [-0.043, +0.055]; -0.003 [-0.062, +0.057]; +0.018 [-0.047, +0.089]; +0.036 [-0.087, +0.162] | tried, no gain |

- **ecg v2:** giving the TSLM the counts did not improve it, apart from a borderline +0.027 at 30 steps. Adding
  the same counts to ecg's score in a simple outside model raises validation AUROC by 0.05-0.07, so the
  information is there, but this fine-tuned model did not learn to use it. We did not vary the base model, the
  starting checkpoint or the training setup, so we cannot say which of these limits it.
- **ecg-full v2:** the expected pattern (+0.018 at 20 steps, +0.036 at 30, nothing lost at 5-10) and a validation
  score that held near its peak (0.739 to 0.725, where ecg-full fell from 0.768 to 0.732), but intervals of 0.05 to 0.12 either way cannot separate it from ecg-full. Its
  gap to the issue + signals + thoughts baseline is no longer significant at any checkpoint (ecg-full's was, at
  30 steps):

| run | gap to its baseline (5 / 10 / 15 / 20 / 30) |
|---|---|
| ecg v2 | -0.011 [-0.045, +0.026]; -0.031 [-0.082, +0.017]; **-0.066 [-0.125, -0.005]**; **-0.096 [-0.161, -0.025]**; **-0.086 [-0.152, -0.029]** |
| ecg-full v2 | -0.029 [-0.076, +0.013]; -0.022 [-0.067, +0.017]; -0.039 [-0.080, +0.004]; -0.023 [-0.073, +0.030]; -0.041 [-0.120, +0.029] |

Second round on H100: 1.25 h (ecg v2) and 2.69 h (ecg-full v2). Results:
[opentslm/ecg-counts/](opentslm/ecg-counts/) and [opentslm/full-multi-a2/](opentslm/full-multi-a2/).

## Checks

- **Shortcuts** ([shortcut_checks.md](shortcut_checks.md)): which agent LLM wrote a run does not drive any score
  (within 0.012); issue difficulty partly does; run length is handled by per-checkpoint scores.
- **Channel hunt** ([channels.md](channels.md)): 15 more per-step channels (test results, exception types, edit
  sizes, ...) were screened on train + validation; none improved prediction.
- **Not run:** a test that shuffles ecg-full's series between runs, so we do not claim ecg-full relies on the ECG;
  ecg, which reads only the ECG, scores above chance.

## Compute

Training 2.8 h (ecg), 5.3 h (ecg-thought), 7.1 h (ecg-full): 15.2 GPU-hours on L40S, for training alone. With OpenTSLM's
eager attention, ecg-full fit in memory only with gradient checkpointing, at 7.8 s per step (projected ~17 h; that run was
stopped after 10 minutes). With PyTorch SDPA (same math) it fit without checkpointing and trained in 7.1 h ([GPU.md](../GPU.md)).

| GPU | Training | Scoring, replays and status reports | Smoke tests, aborted run | Total |
|---|---|---|---|---|
| L40S (first round) | 15.2 h | 9.7 h: ecg 0.5, ecg-thought 1.1, ecg-full pass/fail 1.4, its endings and status reports 6.2, the 600-run live-monitor replay 0.5 | 0.4 h | about 25 GPU-hours |
| H100 (second round) | 3.9 h: ecg v2 1.25, ecg-full v2 2.69 | 1.2 h | 0.1 h | about 5 GPU-hours |

Times come from the run logs. ecg-full's status reports took longest: 0.9 h on a 4,000-checkpoint subset, then 5.3 h on
the full test split (3.0 h scoring the endings, 2.3 h writing 23,520 reports). The baselines and the channel hunt ran on CPU.
