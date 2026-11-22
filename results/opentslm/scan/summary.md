# Live-monitor scan: ecg-full vs the signals baseline

Both monitors replayed the same 600 test runs step by step (step 5 to the run's last visible step), as the
live monitor would see them: all 239 resolved test runs with at least 20 steps, and 361 random failing ones
(seed 0, `run_ids.txt`). Neither monitor chose the runs. Alarm: the `critical` state (health at most 15 where
validation runs failed at least 95 % of the time, two steps in a row). Reproduce with
`opentslm/scan_summary.py`.

| monitor | failing / resolved runs | failing runs flagged | median alarm step | resolved runs flagged (false alarms) | alarms that were right, at the natural 7.1 % resolved |
|---|---|---|---|---|---|
| ecg-full (issue + thoughts + ECG, TSLM) | 361 / 239 | 32.4 % | 6 | 18.4 % | 95.8 % |
| signals (baseline) | 361 / 239 | 34.1 % | 9 | 14.6 % | 96.8 % |

- **They catch different runs.** Of the failing runs, 51 are flagged by both, 66 only by ecg-full and 72 only by
  the signals monitor. When both flag a run, ecg-full is earlier in 27, the signals monitor in 12, the same step in 12.
- **Part of ecg-full's head start comes from the issue.** Half of ecg-full's alarms (80 of 161) fire at step 6, the
  earliest possible, against 28 % for the signals monitor, and ecg-full's alarms cluster by issue: of the 114
  issues with two or more runs in the sample, ecg-full flags every run of 20 and splits 27; the signals monitor
  flags every run of 14 and splits 52.
- **Demo runs that rest on behaviour:** `train-00010-of-00012:721` (scverse__scanpy-2859): ecg-full goes critical
  at step 12, the signals monitor never, and ecg-full spares the 5 other runs of the same issue in the sample.
  Steady under both: `train-00000-of-00012:1738` (resolved). Avoid the serge-sans-paille__gast-59 runs as ecg-full
  showcases: ecg-full flags every run of that issue at step 6.
