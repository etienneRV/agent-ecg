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
| CPU: signals | ECG signals | 0.611 | 0.645 | 0.674 | 0.712 | 0.728 |
| CPU: thoughts | agent's thoughts | 0.680 | 0.703 | 0.710 | 0.701 | 0.701 |
| CPU: issue + signals + thoughts | all | 0.691 | 0.709 | 0.720 | 0.719 | 0.701 |
| CPU: issue only | issue text | 0.613 | 0.619 | 0.612 | 0.616 | 0.614 |
| TSLM C | ECG | 0.583 | 0.611 | 0.616 | 0.620 | 0.615 |
| TSLM B | thoughts + ECG | 0.624 | 0.638 | 0.643 | 0.647 | 0.596 |
| TSLM A (3 tasks) | issue + thoughts + ECG | 0.657 | 0.677 | 0.684 | 0.679 | 0.624 |

TSLM = OpenTSLM-SP on Llama 3.2 1B, fine-tuned from `OpenTSLM/llama-3.2-1b-ecg-sp` on one L40S.
Sources: [baselines.test.md](baselines.test.md), [baselines/](baselines/), [opentslm/](opentslm/),
[GPU.md](../GPU.md).

- Each TSLM is below the CPU model with the same inputs. A is closest: 0.03-0.04 below issue + signals +
  thoughts, not significant at 5-20 steps (paired repository bootstrap, `opentslm/full-multi/compare_vs_*`).
- A beats B by 0.03-0.04 at every checkpoint (borderline at 10 and 15 steps); A differs in two ways (the
  issue, and three tasks), so the gain cannot be credited to either.
- **A + the CPU model together add nothing:** an equal-weight average of their per-checkpoint ranks (nothing
  fitted on test) scores +0.002 / +0.004 / +0.001 / +0.001 / -0.018 over the CPU model alone, every interval
  around zero ([ensemble/](ensemble/)).
- Early on, the text carries most of the signal (the issue and thoughts are complete from step 1); later, the
  behaviour does: the CPU signals model climbs from 0.611 to 0.728 while C stays flat.

## A's other two tasks (full test split, 23,520 checkpoints)

- **Ending** (6 classes): accuracy 0.662, macro-F1 0.222; CPU signals 0.655 / 0.225; majority 0.494 / 0.110.
  About level with the CPU signals model.
- **Status report** (generated text): 99.7 % follow the format. The forecast sentence is weaker than the CPU
  signals model: remaining steps, mean error 17.8 (median 9) vs 14.3 (median 6); pass/fail balanced accuracy
  0.568 vs 0.614.
- Source: [opentslm/full-multi.alltasks/report.test.json](opentslm/full-multi.alltasks/report.test.json) (the
  generated reports are in its `predictions.test.parquet`); the earlier 4,000-checkpoint subset agrees within
  0.005 ([opentslm/full-multi.alltasks-4000/](opentslm/full-multi.alltasks-4000/)).

## The live monitor (600 unseen test runs, replayed step by step)

| monitor | failing runs flagged | median alarm step | false alarms on resolved runs | alarms that were right* |
|---|---|---|---|---|
| TSLM A | 32 % | 6 | 18 % | 96 % |
| CPU signals | 34 % | 9 | 15 % | 97 % |

\* at the natural 7 % share of resolved runs among runs that last 20 steps.

They flag mostly different runs, but combining them adds no accuracy (above). Half of A's alarms fire at the
earliest possible step and cluster by issue: A partly reads how hard the issue is. Demo run resting on
behaviour: `train-00010-of-00012:721` (A alarms at step 12, the signals monitor never, and A spares the 5 other
runs of the same issue); at step 12 it is re-editing the same file after the linter rejected a syntax error
([frame](opentslm/scan/demo.A.showcase.step12.svg), replays for both monitors:
[A](opentslm/scan/replay.showcase.A.jsonl), [signals](opentslm/scan/replay.showcase.signals.jsonl); play it with
`agent-ecg demo results/opentslm/scan/replay.showcase.A.jsonl --run-id train-00010-of-00012:721`). Source:
[opentslm/scan/summary.md](opentslm/scan/summary.md); CPU monitor replays:
[replay.signals.falling.jsonl](replay.signals.falling.jsonl), [demo.falling.step15.svg](demo.falling.step15.svg).

## Checks

- **Shortcuts** ([shortcut_checks.md](shortcut_checks.md)): which agent LLM wrote a run does not drive any score
  (within 0.012); issue difficulty partly does; run length is handled by per-checkpoint scores.
- **Channel hunt** ([channels.md](channels.md)): 15 more per-step channels (test results, exception types, edit
  sizes, ...) were screened on train + validation; none improved prediction.
- **Not yet run:** a test that shuffles A's series between runs. Until then, say that C reads the ECG; do not
  claim A relies on it.

## Compute

Training 2.8 h (C), 5.3 h (B), 7.1 h (A): 15.2 GPU-hours on L40S. OpenTSLM's eager attention ran out of memory
on A's long prompts; switching to PyTorch SDPA (same math) cut A from ~17 h to 7 h ([GPU.md](../GPU.md)).
