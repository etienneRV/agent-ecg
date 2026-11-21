# Channel hunt

Train + validation only (212,458 checkpoints, 1132 repositories); the test split is never read. Inputs: the stored per-step signals, plus the text of steps 1..t (thought, command, reply) read from the release shards, with the run id as a join key only. Proxy: the signals baseline's gradient-boosted trees, with and without a channel's summaries (mean, mean of the last 5 steps, last value, maximum), on identical seeded repository folds. Thresholds were fixed before any result was seen. Compute: 71 min on x86_64 CPU, 8 cores, OMP_NUM_THREADS=5.

## Stage 0: checks

| k | baseline OOF AUROC | fold sd | results/cv5 (all splits) |
|---|---|---|---|
| 5 | 0.663 | 0.019 | 0.661 |
| 10 | 0.713 | 0.011 | 0.710 |
| 15 | 0.731 | 0.011 | 0.729 |
| 20 | 0.748 | 0.006 | 0.748 |
| 30 | 0.765 | 0.024 | 0.776 |

Noise control (i.i.d. random values per step): k=5 -0.000 [-0.003, +0.002], k=10 -0.001 [-0.003, +0.001], k=15 -0.001 [-0.003, +0.001], k=20 -0.002 [-0.005, +0.001], k=30 +0.003 [-0.001, +0.006]. Passes the stage B rule: **no**.

## Stage A: screening (fold seed 0, a 30% sample of repositories)

AUROC gain over the baseline per checkpoint (the candidate's fold sd in brackets), mean gain at 5-15 steps, and the same-issue change averaged over checkpoints.

| # | channel | kind | what it measures | k=5 | k=10 | k=15 | k=20 | k=30 | mean 5-15 | same-issue change |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `steps_since_success` **→ B** | re-expression | steps since the last successful command (silent success, or output after a run or an edit) | +0.0029 (0.019) | +0.0067 (0.024) | +0.0090 (0.031) | +0.0101 (0.027) | +0.0020 (0.047) | +0.0062 | -0.0043 |
| 2 | `same_exception` **→ B** | new information | 1 when a traceback raises the same exception type as the last one | +0.0041 (0.016) | +0.0051 (0.027) | +0.0024 (0.030) | +0.0011 (0.023) | +0.0058 (0.048) | +0.0039 | +0.0020 |
| 3 | `context_ratio` **→ B** | re-expression | conversation length relative to its length after the first step | +0.0029 (0.021) | +0.0026 (0.028) | +0.0031 (0.033) | -0.0023 (0.024) | -0.0033 (0.054) | +0.0029 | -0.0004 |
| 4 | `exception_changed` **→ B** | new information | 1 when a traceback raises a different exception type than the last one | +0.0003 (0.023) | +0.0047 (0.022) | +0.0036 (0.024) | -0.0010 (0.018) | +0.0001 (0.050) | +0.0029 | +0.0021 |
| 5 | `fail_streak` **→ B** | re-expression | consecutive steps whose reply is a traceback, command error, rejected edit or timeout | +0.0017 (0.020) | +0.0058 (0.028) | +0.0002 (0.032) | -0.0003 (0.020) | -0.0017 (0.041) | +0.0026 | +0.0026 |
| 6 | `edit_span` **→ B** | new information | lines an edit replaces | +0.0014 (0.023) | +0.0028 (0.023) | +0.0021 (0.032) | -0.0003 (0.036) | +0.0026 (0.050) | +0.0021 | +0.0075 |
| 7 | `edit_unviewed` **→ B** | new information | 1 when an edit replaces lines of the open file that no earlier reply showed | +0.0030 (0.019) | +0.0034 (0.024) | -0.0009 (0.028) | +0.0004 (0.027) | -0.0001 (0.055) | +0.0018 | +0.0001 |
| 8 | `tests_failed_delta` **→ B** | new information | change in failed tests since the previous test summary (negative: fewer failures) | +0.0004 (0.016) | +0.0022 (0.025) | +0.0002 (0.032) | -0.0006 (0.025) | -0.0010 (0.051) | +0.0009 | -0.0015 |
| 9 | `script_still_crashing` | new information | 1 when a script that crashed still crashes after an edit | -0.0020 (0.017) | +0.0004 (0.027) | +0.0042 (0.028) | +0.0026 (0.020) | +0.0038 (0.053) | +0.0009 | -0.0018 |
| 10 | `tests_all_passed` | new information | 1 when a test summary reports passes and no failures | +0.0032 (0.014) | +0.0011 (0.028) | -0.0024 (0.031) | -0.0016 (0.025) | -0.0120 (0.054) | +0.0006 | -0.0018 |
| 11 | `success_marker` | new information | 1 when a run of code prints "completed successfully" | +0.0024 (0.015) | +0.0006 (0.030) | -0.0014 (0.033) | -0.0027 (0.024) | -0.0118 (0.058) | +0.0006 | -0.0045 |
| 12 | `script_fixed` | new information | 1 when a script that crashed runs cleanly after an edit | +0.0043 (0.022) | -0.0028 (0.024) | -0.0006 (0.031) | +0.0031 (0.025) | +0.0061 (0.051) | +0.0003 | -0.0001 |
| 13 | `tests_passed` | new information | passed tests in a test summary at this step | +0.0007 (0.014) | -0.0011 (0.029) | +0.0005 (0.030) | -0.0005 (0.026) | +0.0063 (0.051) | +0.0000 | -0.0001 |
| 14 | `edit_size` | new information | lines an edit writes | -0.0011 (0.022) | -0.0019 (0.028) | +0.0012 (0.027) | -0.0042 (0.032) | -0.0036 (0.047) | -0.0006 | -0.0061 |
| 15 | `tests_failed` | new information | failed tests (failures and errors) in a test summary at this step | +0.0009 (0.013) | -0.0008 (0.026) | -0.0064 (0.025) | -0.0023 (0.020) | -0.0062 (0.056) | -0.0021 | +0.0036 |

Shuffled control of the top candidate (`steps_since_success`, values moved between checkpoints of the same k): k=5 -0.003 [-0.008, +0.002], k=10 -0.003 [-0.008, +0.002], k=15 -0.007 [-0.016, +0.001], k=20 -0.006 [-0.017, +0.003], k=30 -0.008 [-0.022, +0.007]. Passes the stage B rule: **no**.

## Stage B: confirmation (fold seeds 1 and 2, all repositories, pooled)

AUROC gain with a paired repository-bootstrap 95 % interval (1,000 resamples, the same for both seeds and both models). Raw-series proxy: the raw per-step values of the 9 signals plus the channel, padded to 30 steps (fold seed 1).

| channel | kind | k=5 | k=10 | k=15 | k=20 | k=30 | same-issue change (5 / 10 / 15 / 20 / 30) | early-stop recall change | raw proxy, mean gain 5-15 | shuffled control passes | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `steps_since_success` | re-expression | +0.000 [-0.002, +0.003] | +0.003 [-0.001, +0.006] | +0.001 [-0.002, +0.004] | +0.004 [+0.000, +0.008] | +0.004 [-0.001, +0.010] | -0.001 / +0.000 / +0.000 / -0.000 / -0.001 | +0.017 / +0.019 / -0.014 / +0.009 / +0.008 | +0.0022 | no | fails: AUROC rule |
| `same_exception` | new information | -0.000 [-0.002, +0.002] | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.004] | +0.000 [-0.002, +0.003] | -0.004 [-0.008, -0.001] | +0.001 / +0.000 / -0.001 / -0.001 / -0.005 | +0.002 / +0.003 / -0.005 / -0.009 / -0.009 | -0.0008 | no | fails: AUROC rule |
| `context_ratio` | re-expression | -0.001 [-0.003, +0.001] | -0.001 [-0.004, +0.001] | +0.000 [-0.002, +0.003] | -0.000 [-0.003, +0.003] | -0.001 [-0.004, +0.002] | +0.001 / +0.001 / +0.001 / +0.000 / -0.000 | -0.001 / -0.006 / +0.003 / +0.001 / +0.014 | +0.0003 | no | fails: AUROC rule |
| `exception_changed` | new information | -0.001 [-0.003, +0.001] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.004] | -0.001 [-0.004, +0.002] | -0.001 [-0.005, +0.002] | -0.001 / -0.001 / -0.001 / -0.002 / -0.001 | -0.000 / +0.007 / -0.013 / -0.030 / +0.009 | +0.0003 | no | fails: AUROC rule |
| `fail_streak` | re-expression | -0.001 [-0.003, +0.001] | +0.001 [-0.001, +0.004] | +0.001 [-0.001, +0.004] | -0.001 [-0.004, +0.002] | -0.002 [-0.006, +0.002] | -0.000 / -0.001 / +0.003 / -0.002 / +0.001 | +0.002 / +0.005 / -0.009 / -0.021 / +0.000 | -0.0004 | no | fails: AUROC rule |
| `edit_span` | new information | +0.001 [-0.001, +0.003] | +0.001 [-0.002, +0.004] | +0.000 [-0.003, +0.003] | -0.000 [-0.003, +0.003] | -0.001 [-0.004, +0.003] | +0.002 / -0.000 / -0.002 / -0.002 / +0.006 | -0.011 / +0.009 / +0.005 / -0.007 / +0.007 | +0.0005 | no | fails: AUROC rule |
| `edit_unviewed` | new information | -0.001 [-0.004, +0.001] | -0.000 [-0.002, +0.002] | -0.001 [-0.003, +0.002] | -0.002 [-0.004, +0.001] | -0.003 [-0.006, +0.001] | -0.000 / +0.000 / -0.002 / -0.000 / +0.001 | +0.003 / +0.001 / -0.000 / +0.000 / -0.003 | -0.0012 | no | fails: AUROC rule |
| `tests_failed_delta` | new information | -0.000 [-0.002, +0.001] | +0.000 [-0.001, +0.002] | +0.000 [-0.002, +0.002] | -0.000 [-0.003, +0.002] | -0.002 [-0.006, +0.001] | +0.001 / -0.001 / -0.001 / -0.003 / +0.000 | +0.001 / +0.005 / +0.012 / +0.009 / +0.008 | +0.0003 | no | fails: AUROC rule |

Baseline on these folds (mean of both seeds): AUROC 0.658 / 0.706 / 0.726 / 0.745 / 0.766; same-issue pairs 0.547 / 0.612 / 0.663 / 0.700 / 0.740; early-stop recall (flagged group fails 95 % of the time at 5-10 steps, 99 % at 15-30) 0.029 / 0.283 / 0.119 / 0.281 / 0.501.

## Stage C: combination

Skipped: 0 channel(s) passed; stage C needs 2 or more.

## Verdict

**No channel reliably helps.** 8 candidates advanced from screening (`steps_since_success`, `same_exception`, `context_ratio`, `exception_changed`, `fail_streak`, `edit_span`, `edit_unviewed`, `tests_failed_delta`), but none met the fixed stage B rule on fresh folds. Keep the 19 series as they are for Sunday's training round. The proxies read features derived from the channels, while the TSLM reads the series itself, so this is a shortlist with evidence, not a guarantee. Only a **new information** channel is likely to help a model that already reads the full sequence of signals; a **re-expression** mostly restates what it can already learn.
