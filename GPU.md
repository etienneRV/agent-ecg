# Fine-tuning OpenTSLM on a GPU

Step-by-step for the Nebius machine. One GPU with 24 GB or more (L40S, A100, H100) is enough for
OpenTSLM-SP on Llama 3.2 1B: only ~14 M parameters train (encoder, projector, LoRA).

## 0. Before you start

- Your Hugging Face account needs access to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  (request it on the page; Meta approves it), and a **read** token from https://huggingface.co/settings/tokens.
- Backup without approval: accept the terms of [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)
  and add `--llm google/gemma-3-270m` below (smaller, faster, likely weaker).

## 1. Set up (~15 min, mostly the 1.1 GB download)

```bash
git clone https://github.com/etienneRV/agent-ecg && cd agent-ecg
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                          # agent-ecg itself
uv pip install -r opentslm/requirements-gpu.txt  # OpenTSLM (pinned), PyTorch, transformers, peft
export HF_TOKEN=hf_...                           # your read token
uv run agent-ecg prepare                         # build the TimeNet dataset (~10 min)
uv run agent-ecg opentslm-export                 # OpenTSLM samples into data/opentslm (~2 min)
uv run agent-ecg baselines                       # baseline predictions to compare against (~9 min, CPU)
```

From here on, use `uv run --no-sync`, so uv keeps the packages installed in the second step.

## 2. Smoke test (~5 min): does everything run?

```bash
uv run --no-sync python opentslm/train.py --smoke
```

It trains 20 steps on a small sample and prints the loss, seconds per step, peak GPU memory and an
estimate for the full run. The loss should drop. Then check prediction end to end:

```bash
uv run --no-sync python opentslm/predict.py runs/opentslm-sp-smoke --split test --max 50
uv run agent-ecg evaluate runs/opentslm-sp-smoke/predictions.test.parquet --subset
```

If memory runs out: `--batch-size 4 --grad-accum 4`, or add `--gradient-checkpointing`.

## 3. Full runs

Run them inside `tmux` (or with `nohup`) so they survive a dropped connection. Two runs answer the
main question, whether the status report helps or distracts:

```bash
# A: all three tasks, each answer weighted equally
uv run --no-sync python opentslm/train.py --name llama1b-multi 2>&1 | tee runs/llama1b-multi.log
# B: pass/fail only
uv run --no-sync python opentslm/train.py --name llama1b-outcome --tasks outcome 2>&1 | tee runs/llama1b-outcome.log
```

How training works:
- **Each answer counts equally.** The loss averages each answer's tokens first, then the answers, so
  a 3-token "unresolved" weighs as much as a 110-token status report. (A plain token average would
  give the status reports ~97 % of the signal.) Shift the balance with `--task-weights outcome=2`.
- **The best checkpoint is chosen by pass/fail AUROC on validation** (2,000 checkpoints, mean over
  5/10/20/30 steps), the score we report, not by the loss. `history.json` logs both.
- Defaults: 2 epochs of 60,000 items, batch 8 x 2 accumulation, resolved runs at 30 % of the
  sampling weight, validation every 500 steps. Check the smoke test's time estimate against the
  voucher hours and lower `--samples-per-epoch` or `--epochs` if needed. If time allows only one run,
  run B: it is what the AUROC table measures.

## 4. Predict and score

```bash
R=runs/llama1b-outcome   # or runs/llama1b-multi
uv run --no-sync python opentslm/predict.py $R --split validation
uv run --no-sync python opentslm/predict.py $R --split test --threshold-from $R/predictions.validation.parquet
uv run agent-ecg evaluate $R/predictions.test.parquet --json $R/report.test.json
uv run agent-ecg compare $R/predictions.test.parquet data/predictions/thoughts.test.parquet   # vs the best baseline
uv run agent-ecg compare $R/predictions.test.parquet data/predictions/issue.test.parquet      # vs the issue alone
```

Pass/fail and ending scoring is fast (two short forward passes per checkpoint). The status reports
are generated text and take longest; for a quick look, add `--tasks outcome,ending`. A pass/fail-only
model can only answer `--tasks outcome`.

Copy `report.test.json`, the `compare` output and the predictions file into `results/` and commit
them; checkpoints stay out of git (`runs/` is ignored).

## Knobs worth trying

| Idea | Flag |
| --- | --- |
| Start from a model that already reads time series | `--init-from OpenTSLM/llama-3.2-1b-tsqa-sp` |
| Bigger base model | `--llm meta-llama/Llama-3.2-3B` |
| Weight pass/fail more in multi-task training | `--task-weights outcome=3` |
| More or fewer successes in training | `--resolved-share 0.5` (0 = natural 12 %) |
| One checkpoint per run (less repetition) | `--one-per-run` |
| Longer thought history | raise `THOUGHT_TOKENS` in `src/agent_ecg/opentslm_data.py`, re-run `opentslm-export` |
| More training | `--epochs 3 --samples-per-epoch 120000` |

Every run is scored by the same `agent-ecg evaluate`, so results stay comparable. The bar: AUROC above
the thoughts baseline (0.68 / 0.70 / 0.70 / 0.70 at 5 / 10 / 20 / 30 steps).
