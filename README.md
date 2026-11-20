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

## Training data for OpenTSLM

```bash
uv run agent-ecg opentslm-export   # after prepare; writes data/opentslm/{train,validation,test}.parquet
```

Each row is one checkpoint in [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM)'s sample format:
`pre_prompt` (setting, issue, the agent's thoughts so far), 15 labelled per-step `time_series`, and
the three tasks' answers. `post_prompts.json` holds each task's question. To train, copy
[`opentslm/agent_ecg_dataset.py`](opentslm/agent_ecg_dataset.py) into an OpenTSLM checkout; its
docstring explains the options (tasks, one checkpoint per run, sample caps).

**Length limits** (Llama 3.2 tokens), chosen from the measured lengths so a prompt stays near 2,000 tokens:

| Part | Kept | Why |
| --- | --- | --- |
| Issue | first 512 tokens | about 80 % of issues fit whole; the start states the problem |
| Thoughts | latest steps first, up to 1,024 tokens, each step at most 256 | recent behaviour matters most for a monitor; the time series still cover every step |
| Time series | all steps | 4 values make one token, so 15 series x 30 steps cost about 120 tokens |

Measured on the export: a full sample is 1,340-1,460 tokens at the median and at most 2,341. At 30
steps the thoughts budget drops the oldest 15 steps' thoughts for the median checkpoint; raise
`THOUGHT_TOKENS` in `opentslm_data.py` if the GPU has memory to spare.

## Fine-tuning OpenTSLM

See **[GPU.md](GPU.md)**: setup, a 20-step smoke test, the full run, and scoring.
`opentslm/train.py` fine-tunes OpenTSLM-SP (encoder, projector and LoRA) on all three tasks;
`opentslm/predict.py` writes a predictions file for `agent-ecg evaluate`, with pass/fail
probabilities from the model's likelihood of "resolved" vs "unresolved".

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
  opentslm_data.py  checkpoints -> OpenTSLM samples (trimming, series, prompts)
  cli.py         agent-ecg prepare | baselines | evaluate | compare | opentslm-export
opentslm/        train.py, predict.py, common.py (GPU only), and a dataset class for OpenTSLM's own trainer
tests/           metric, parser and converter tests (uv run pytest)
results/         baseline scores (Markdown + JSON)
```

## Licence

The code is MIT-licensed (see [LICENSE](LICENSE)). The data is not part of this repository: the
nebius/SWE-agent-trajectories release is CC BY 4.0, its model outputs also fall under the Llama 3.1
License, and the base models you fine-tune (Llama 3.2, Gemma 3) keep their own licences.

