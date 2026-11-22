# OpenTSLM runs

Each folder holds one fine-tuned model's results on the test split: `report.test.json` (every score),
`predictions.test.parquet`, `compare_vs_*.test.txt` (paired AUROC gains), `config.json` (training settings),
`history.json` (validation curve) and `monitor.json` (the live monitor fitted on validation). Folder names are
the training runs' names; the docs and the website use the model names.

| Folder | Model | Reads | Trained on |
| --- | --- | --- | --- |
| `ecg-only/` | ecg | the 19 per-step series | L40S |
| `thoughts-outcome/` | ecg-thought | thoughts + series | L40S |
| `full-multi/` | ecg-full (pass/fail, ending, status report) | issue + thoughts + series | L40S |
| `ecg-counts/` | ecg v2 (second round) | series + 3 count channels | H100 |
| `full-multi-a2/` | ecg-full v2 (second round) | issue + first and latest thoughts + series | H100 |
| `full-multi.alltasks/` | ecg-full's ending and status-report predictions on the full test split | | |
| `full-multi.alltasks-4000/` | the same on a 4,000-checkpoint subset, scored first | | |
| `scan/` | the live-monitor replay of 600 test runs (ecg-full against the signals baseline) | | |

Older commits call ecg, ecg-thought and ecg-full C, B and A, and the second-round runs C2 and A2. How each run was
trained: [GPU.md](../../GPU.md); what the numbers mean: [results/SUMMARY.md](../SUMMARY.md).
