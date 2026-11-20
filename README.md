# agent-ecg

An "ECG" for coding agents: read the first steps of a SWE-agent run and warn early whether it will
resolve its GitHub issue. Built on the TimeNet dataset `nebius/swe_agent_trajectories`
([OpenTSLM/TimeNet#832](https://github.com/OpenTSLM/TimeNet/pull/832)). Team RAMadan, TimeNet
hackathon Round 1.

**Read [REPORT.md](REPORT.md)** for the problem, the data, the evaluation protocol and the results.

## Setup

```bash
uv sync
```

Until #832 is merged, the connector is installed from its PR branch (see `pyproject.toml`).

## Run

```bash
uv run agent-ecg prepare      # build the TimeNet release (1.1 GB download, ~5 min) and flatten it into data/
uv run agent-ecg baselines    # fit the five CPU baselines, score them on test, write results/
uv run agent-ecg evaluate my_predictions.parquet          # score any model on test
uv run agent-ecg evaluate my_predictions.csv --split validation --json report.json
uv run agent-ecg compare mine.parquet data/predictions/thoughts.test.parquet   # paired AUROC gain per checkpoint
```

`prepare` writes three tables under `data/` (git-ignored):

| File | One row per | Holds |
| --- | --- | --- |
| `checkpoints.parquet` | checkpoint (first k steps of a run still going) | ids, `split`, `k`, the 7 per-step signals as lists, targets `resolved`, `ending`, `status`, `remaining` |
| `thoughts.parquet` | run | the agent's thoughts, step by step; a checkpoint at k sees the first k |
| `issues.parquet` | issue annotation | the issue text (`issue_id` links it to checkpoints) |

## Scoring a model

A predictions file has one row per checkpoint of the split, keyed by `run_id` and `k`, with any of:

| Column | Task | Scored by |
| --- | --- | --- |
| `p_resolved` | Will it pass? (probability) | AUROC with a 95 % repository-bootstrap interval, average precision, Brier score |
| `pred_resolved` | Will it pass? (yes/no; default `p_resolved >= 0.5`) | balanced accuracy |
| `pred_ending` | How will it end? (6 labels) | accuracy, macro-F1 |
| `pred_status` | Status report (text) | the **last sentence only**: parse rate, remaining-steps error, ending, pass/fail |

Every score is reported per checkpoint (k = 5, 10, 20, 30) and pooled. Why: most runs fail (83 % at
5 steps, 96 % at 30), so accuracy rewards always saying "fail", and the first three sentences of a
status report restate counts the model can read off its input. The report also prints word-overlap
F1 on the full text next to the prediction-only score, to show the gap.

## Layout

```
src/agent_ecg/
  data.py        build the release, flatten it, load it
  status.py      read and write the status report's sentences
  metrics.py     the evaluation (all tasks)
  baselines.py   majority, signals, thoughts, signals+thoughts, issue
  cli.py         agent-ecg prepare | baselines | evaluate | compare
tests/           metric and parser tests (uv run pytest)
results/         baseline scores (Markdown + JSON)
```
