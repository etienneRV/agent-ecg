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
uv run agent-ecg baselines    # fit the six baselines, score them on test, write results/ and data/models/
uv run agent-ecg baselines --cv 5   # repository cross-validation over all checkpoints: steadier AUROCs
uv run agent-ecg replay --top 3     # the live monitor on test runs, step by step (signals model)
uv run agent-ecg evaluate my_predictions.parquet          # score any model on test
uv run agent-ecg evaluate my_predictions.csv --split validation --json report.json
uv run agent-ecg compare mine.parquet data/predictions/thoughts.test.parquet   # paired AUROC gain per checkpoint
```

`prepare` writes three tables under `data/` (git-ignored):

| File | One row per | Holds |
| --- | --- | --- |
| `checkpoints.parquet` | checkpoint (first k steps of a run still going) | ids, `split`, `k`, the 9 per-step signals as lists, targets `resolved`, `ending`, `status`, `remaining` |
| `thoughts.parquet` | run | the agent's thoughts, step by step; a checkpoint at k sees the first k |
| `issues.parquet` | issue annotation | the issue text (`issue_id` links it to checkpoints) |

## Training data for OpenTSLM

```bash
uv run agent-ecg opentslm-export                     # issue + thoughts + series -> data/opentslm/
uv run agent-ecg opentslm-export --context thoughts  # thoughts + series        -> data/opentslm-thoughts/
uv run agent-ecg opentslm-export --context none      # the series alone (ECG)   -> data/opentslm-none/
```

Two options build the final runs' data (GPU.md section 6), and need `--out`:
- `--count-channels` adds ecg v2's three count channels: steps since the last success, tracebacks so far, code runs so far; fixed scales, past steps only.
- `--head-thought-tokens 512 --thought-tokens 1536` keeps the first thoughts as well as the latest (ecg-full v2).

`meta.json` records the settings, and `opentslm/replay.py` rebuilds prompts from them.

Each row is one checkpoint in [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM)'s sample format:
`pre_prompt` (the setting, then the issue and the agent's thoughts so far, as the context asks), 19
labelled per-step `time_series`, and the three tasks' answers. `post_prompts.json` holds each task's
question and `meta.json` the context.

**The series** are four lengths (thought, command, reply, whole conversation), z-normalised with
their mean and standard deviation in the label, and fifteen 0/1 indicators: a repeated command, five
command kinds, five reply kinds, an edit of the same file as the edit before, and the kind of file an
edit changes (source, test, a file the agent created). An indicator's label says what a 1 means and
nothing else. With a count in the label ("2 of 5 steps"), the model could answer from the text and
never learn to read the series. To train, copy
[`opentslm/agent_ecg_dataset.py`](opentslm/agent_ecg_dataset.py) into an OpenTSLM checkout; its
docstring explains the options (tasks, one checkpoint per run, sample caps).

**Length limits** (Llama 3.2 tokens), chosen from the measured lengths so a prompt stays near 2,000 tokens:

| Part | Kept | Why |
| --- | --- | --- |
| Issue | first 512 tokens | about 80 % of issues fit whole; the start states the problem |
| Thoughts | latest steps first, up to 1,024 tokens, each step at most 256 | recent behaviour matters most for a monitor; the time series still cover every step |
| Time series | all steps | 4 values make one token, so 19 series x 30 steps cost about 150 tokens |

Measured on the exports (the whole prompt with the pass/fail question, one token per four series
values), median / 99th percentile:

| Context | 5 steps | 10 steps | 15 steps | 20 steps | 30 steps |
| --- | --- | --- | --- | --- | --- |
| none (the ECG alone) | 443 / 445 | 463 / 464 | 482 / 483 | 501 / 502 | 558 / 558 |
| thoughts | 753 / 1,151 | 1,117 / 1,529 | 1,458 / 1,581 | 1,557 / 1,620 | 1,635 / 1,708 |
| full (issue and thoughts) | 994 / 1,496 | 1,389 / 1,966 | 1,649 / 2,085 | 1,747 / 2,128 | 1,839 / 2,213 |

At 30 steps the thoughts budget drops the oldest 15 steps' thoughts for the median checkpoint; raise
`THOUGHT_TOKENS` in `opentslm_data.py` if the GPU has memory to spare.

## Fine-tuning OpenTSLM

See **[GPU.md](GPU.md)**: setup, a 20-step smoke test, the recommended runs, scoring and the replay.
`opentslm/train.py` fine-tunes OpenTSLM-SP (encoder, projector and LoRA) on any export and any subset
of the three tasks; `opentslm/predict.py` writes a predictions file for `agent-ecg evaluate`, with
pass/fail probabilities from the model's likelihood of "resolved" vs "unresolved", calibrated on
validation (`--calibrate-from`); `opentslm/replay.py` replays runs step by step for the demo.

## The live monitor

`src/agent_ecg/monitor.py` turns any model's pass/fail scores into what a monitor shows, fitted on
validation predictions only:

- **Health, 0 to 100:** the percentile of a run's score among validation runs still going at the same
  step. 96 % of the runs still going at step 30 fail, so an absolute probability would sit near zero
  for every long run; the percentile shows whether this run looks worse than runs of its age.
- **Calibrated probability:** an isotonic map per checkpoint. Training gives resolved runs 30 % of the
  weight, which inflates the raw scores; the map gives honest probabilities back (Brier score).
- **Critical:** health at or below 15 where validation runs failed at least 95 % of the time, for two
  steps in a row. The alarm then holds until health is back to 30, so it does not flicker, and the
  recommendation quotes the measured failure rate.
- **States:** `stable` (health 60 or more), `watch` (30 to 60), `low` (under 30), `critical`.

`agent-ecg replay` (signals model) and `opentslm/replay.py` (a fine-tuned TSLM) score every prefix of
a test run, from step 5 to its last visible step, and write one JSON line per step: `step`,
`health`, `state`, `score`, `p_calibrated`, `action`, `reply`, `events` (repeated command, edit
rejected, traceback, same file again, ...), the rule-made `status` description, and a
`recommendation` once the state is critical. A terminal ECG display can read it line by line.
Between checkpoints (step 7, 12, ...) health is interpolated from the two nearest fitted ones.

## The terminal ECG (demo)

```bash
uv run agent-ecg replay --top 3 --out results/replay.signals.falling.jsonl        # test runs that fall to critical
uv run agent-ecg demo results/replay.signals.falling.jsonl                         # play the first one, 2 s per step
uv run agent-ecg demo results/replay.signals.falling.jsonl --at 15 --svg ecg.svg   # one frame as an image
```

The left pane shows the agent's real thought, command and reply at each step, read from the run's
raw trajectory on Hugging Face (one ~90 MB shard, downloaded once). The right pane shows the monitor:
a heartbeat whose beats shrink with health, the health line, the chance of failure, the state, and the
recommendation once the alarm is on. The end card says what really happened to the run. Every number
on the right comes from the replay file, computed one step at a time, so the screen is what a live
monitor would have shown; it is still a replay of a recorded run, and the header says so. For a
fine-tuned TSLM, pass the JSONL that `opentslm/replay.py --out` writes.
[`results/demo.falling.step15.svg`](results/demo.falling.step15.svg) is the frame at the alarm of a
test run (`bids-standard__pybids-997`): the agent loops between opening and searching the same file.

## Scoring a model

A predictions file has one row per checkpoint of the split, keyed by `run_id` and `k`, with any of:

| Column | Task | Scored by |
| --- | --- | --- |
| `p_resolved` | Will it pass? (probability) | AUROC with a 95 % repository-bootstrap interval, same-issue pairs, average precision, Brier score |
| `p_calibrated` | the same, calibrated on validation (optional) | Brier score |
| `pred_resolved` | Will it pass? (yes/no; default `p_resolved >= 0.5`) | balanced accuracy |
| `pred_ending` | How will it end? (6 labels) | accuracy, macro-F1 |
| `pred_status` | Status report (text) | the **last sentence only**: parse rate, remaining-steps error, ending, pass/fail |

Every score is reported per checkpoint (k = 5, 10, 15, 20, 30) and pooled. Why: most runs fail (83 %
at 5 steps, 96 % at 30), so accuracy rewards always saying "fail", and the first four sentences of a
status report restate counts the model can read off its input. The report also prints word-overlap
F1 on the full text next to the prediction-only score, to show the gap.

**Same-issue pairs** answer the objection that a monitor only learns which issues are hard. Among
pairs of runs on the same issue, one resolved and one not, it is the share where the model scores the
resolved run higher (ties count half; 0.5 is chance). The issue is the same within a pair, so only
what the model reads from the run itself counts.

## Layout

```
src/agent_ecg/
  data.py              build the release, flatten it, load it
  status.py            read and write the status report's sentences
  metrics.py           the evaluation (all tasks), including same-issue pairs
  baselines.py         majority, signals, thoughts, signals+thoughts, issue, issue+signals+thoughts; repository CV
  channels.py          the channel hunt: candidate per-step channels, screened on train + validation
  monitor.py           calibration, health, critical alerts, step-by-step replay
  demo.py              the terminal ECG: a replayed run beside its monitor
  opentslm_data.py     checkpoints -> OpenTSLM samples (trimming, series, prompts, contexts)
  opentslm_samples.py  choose, weight and label the samples for training (no PyTorch, so tested anywhere)
  cli.py               agent-ecg prepare | baselines | evaluate | compare | opentslm-export | replay | demo | channels
opentslm/          train.py, predict.py, replay.py, common.py (GPU only), scan_summary.py, and a dataset class
                   for OpenTSLM's own trainer
tests/             metric, parser, converter, monitor, demo and baseline tests (uv run pytest)
results/           every score on test: baselines, the TSLMs, the live-monitor scan (start at results/SUMMARY.md)
scripts/           charts.py (presentation charts) and site.py (the project website), both read from results/
figures/           the charts for the video (1920 x 1080, dark), see figures/README.md
site/              the project website, published to GitHub Pages by .github/workflows/pages.yml
```

## Licence

The code is MIT-licensed (see [LICENSE](LICENSE)). The data is not part of this repository: the
nebius/SWE-agent-trajectories release is CC BY 4.0, its model outputs also fall under the Llama 3.1
License, and the base models you fine-tune (Llama 3.2, Gemma 3) keep their own licences.

