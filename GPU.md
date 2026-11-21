# Fine-tuning OpenTSLM on a GPU

Step-by-step for the Nebius machine. One GPU with 24 GB or more (L40S, A100, H100) is enough for
OpenTSLM-SP on Llama 3.2 1B: only ~14 M parameters train (encoder, projector, LoRA).

## 0. Before you start

- Your Hugging Face account needs access to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  (request it on the page; Meta approves it), and a **read** token from https://huggingface.co/settings/tokens.
- Without that approval: `--llm unsloth/Llama-3.2-1B` is the same model, not gated, but then train
  from scratch (no `--init-from`: the OpenTSLM checkpoints name the gated id).

## 1. Set up (~20 min, mostly the 1.1 GB download)

```bash
git clone https://github.com/etienneRV/agent-ecg && cd agent-ecg
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                          # agent-ecg itself
uv pip install -r opentslm/requirements-gpu.txt  # OpenTSLM (pinned), PyTorch, transformers, peft
export HF_TOKEN=hf_...                           # your read token
uv run --no-sync agent-ecg prepare                         # build the TimeNet dataset (~5 min)
uv run --no-sync agent-ecg opentslm-export --context none      # the ECG alone -> data/opentslm-none (~1 min)
uv run --no-sync agent-ecg opentslm-export --context thoughts  # thoughts + ECG -> data/opentslm-thoughts
uv run --no-sync agent-ecg opentslm-export                     # issue + thoughts + ECG -> data/opentslm
uv run --no-sync agent-ecg baselines                       # CPU baselines + the signals monitor (~15 min)
```

From here on, use `uv run --no-sync`, so uv keeps the packages installed in the second step.

## 2. Smoke test (~5 min): does everything run?

```bash
uv run --no-sync python opentslm/train.py --smoke --data data/opentslm-none --tasks outcome \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp
```

It trains 20 steps on a small sample and prints the loss, seconds per step, peak GPU memory and an
estimate for the full run. The loss should drop. Then check prediction end to end:

```bash
R=runs/opentslm-sp-smoke
uv run --no-sync python opentslm/predict.py $R --split validation --tasks outcome --max 200
uv run --no-sync python opentslm/predict.py $R --split test --tasks outcome --max 50 --calibrate-from $R/predictions.validation.parquet
uv run --no-sync agent-ecg evaluate $R/predictions.test.parquet --subset
```

If memory runs out: `--batch-size 4 --grad-accum 4`, or add `--gradient-checkpointing`.

Not on a Mac: OpenTSLM loads Llama in bfloat16, which has no fast CPU path there, and Apple's MPS
backend grew past 30 GB on this smoke test. The scripts were checked on a Mac CPU in float32 instead
(20 steps at 4-5 s each, 7.9 GB): use the GPU for real runs.

## 3. The runs, in order

Run them inside `tmux` (or with `nohup`) so they survive a dropped connection. All three start from
OpenTSLM's ECG model, the last stage of its curriculum: it already reads 12 labelled leads at once,
each labelled "..., it has mean x and std y:" like our 19 series.

```bash
mkdir -p runs
# C: the ECG alone, pass/fail. The demo's model: whatever it says, it read from the series.
uv run --no-sync python opentslm/train.py --name ecg-only --data data/opentslm-none --tasks outcome \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp 2>&1 | tee runs/ecg-only.log
# B: the agent's thoughts beside the ECG, pass/fail
uv run --no-sync python opentslm/train.py --name thoughts-outcome --data data/opentslm-thoughts --tasks outcome \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp 2>&1 | tee runs/thoughts-outcome.log
# A: issue, thoughts and ECG; all three tasks
uv run --no-sync python opentslm/train.py --name full-multi --data data/opentslm \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp 2>&1 | tee runs/full-multi.log
```

Why this order:
- **C is the demo** and the cleanest TSLM result: the prompt holds no issue and no thoughts (about
  450-560 tokens), so any skill comes from reading the trace. It is also the fastest to train.
- **B adds what the agent writes.** The thoughts baseline is the strongest CPU baseline, so B shows
  what the series add to the text.
- **A adds the issue and the status report.** It needs the most GPU time (~1,400 tokens a sample);
  run it last, if the hours allow.

How training works:
- **Each answer counts equally.** The loss averages each answer's tokens first, then the answers, so
  a 3-token "unresolved" weighs as much as a 110-token status report. Shift the balance with
  `--task-weights outcome=2`.
- **The best checkpoint is chosen by pass/fail AUROC on validation** (2,000 checkpoints, mean over
  the five checkpoints), the score we report, not by the loss. `history.json` logs both.
- **Resolved runs get 30 % of the sampling weight** (`--resolved-share 0.3`), so the model sees
  enough successes. That inflates its probabilities; step 4 calibrates them back on validation.
- Defaults: 2 epochs of 60,000 items, batch 8 x 2 accumulation, validation every 500 steps. Check the
  smoke test's time estimate against the voucher hours and lower `--samples-per-epoch` or `--epochs`
  if needed.

## 4. Predict, calibrate and score

```bash
R=runs/ecg-only    # or runs/thoughts-outcome, runs/full-multi
uv run --no-sync python opentslm/predict.py $R --split validation --tasks outcome
uv run --no-sync python opentslm/predict.py $R --split test --tasks outcome \
    --threshold-from $R/predictions.validation.parquet --calibrate-from $R/predictions.validation.parquet
uv run --no-sync agent-ecg evaluate $R/predictions.test.parquet --json $R/report.test.json
uv run --no-sync agent-ecg compare $R/predictions.test.parquet data/predictions/signals.test.parquet  # vs the same inputs on CPU
uv run --no-sync agent-ecg compare $R/predictions.test.parquet data/predictions/issue.test.parquet    # vs the issue alone
```

- `predict.py` reads the samples the run trained on (`config.json`), so an ECG-only model is
  scored on ECG-only prompts.
- `--calibrate-from` fits the monitor on the validation predictions: an isotonic map per checkpoint
  (`p_calibrated`, scored by its Brier score), health from 0 to 100, and the critical threshold. It
  saves `monitor.json` in the run folder and prints how often a critical alert was right on validation.
- `evaluate` adds **same-issue pairs**: among runs on the same issue, one resolved and one not, how
  often the model ranks the resolved one higher. It answers "does it only learn which issues are hard?".
- For model A, drop `--tasks outcome` to also predict the ending and the status report (slow: the
  reports are generated text).

Copy `report.test.json`, the `compare` output and the predictions file into `results/` and commit
them; checkpoints stay out of git (`runs/` is ignored).

## 5. The demo: replay a run as the live monitor

```bash
uv run --no-sync agent-ecg replay --top 3 --out results/replay.signals.jsonl   # picks test runs that fall to critical
uv run --no-sync python opentslm/replay.py runs/ecg-only --run-id <a run id printed above> \
    --out results/replay.ecg-only.jsonl
uv run --no-sync agent-ecg replay --pick resolved --top 3   # contrast: resolved runs that stay steady
```

Each JSON line is one step: `step`, `health` (0-100), `state` (`stable` / `watch` / `low` /
`critical`), `p_calibrated`, the step's `action` and `reply`, `events` (repeated command, edit
rejected, same file again, ...), the rule-made `status`, and a `recommendation` ("Stop the agent to
save credits: ... failed 97 % of the time") once the state is critical. That is what the terminal ECG
shows. The alarm goes on when health is at most 15 in a zone where at least 95 % of validation runs
failed, two steps in a row, and holds until health is back to 30.

## Results to fill in

| Run | Text beside the ECG | Tasks | AUROC 5 / 10 / 15 / 20 / 30 steps (test) | Same-issue pairs, 10 steps | Brier, calibrated, 10 steps |
| --- | --- | --- | --- | --- | --- |
| CPU: signals (same inputs as C) | none | pass/fail | 0.611 / 0.645 / 0.674 / 0.712 / 0.728 | 0.610 | 0.1196 |
| CPU: thoughts (the bar for B) | thoughts | pass/fail | 0.680 / 0.703 / 0.710 / 0.701 / 0.701 | 0.562 | 0.1139 |
| CPU: issue+signals+thoughts (the bar for A) | issue, thoughts | pass/fail | 0.691 / 0.709 / 0.720 / 0.719 / 0.700 | 0.605 | 0.1126 |
| C ecg-only | none | pass/fail | | | |
| B thoughts-outcome | thoughts | pass/fail | | | |
| A full-multi | issue, thoughts | all three | | | |

The test split is a hard draw for the signals: repository cross-validation (`agent-ecg baselines
--cv 5`) gives them 0.663 / 0.711 / 0.729 / 0.747 / 0.775 and same-issue pairs of 0.615 at 10 steps.
Compare a TSLM with the CPU rows on the same test split, as above.

## Knobs worth trying

| Idea | Flag |
| --- | --- |
| Another starting point (3-axis motion, an earlier curriculum stage) | `--init-from OpenTSLM/llama-3.2-1b-har-sp` |
| Bigger base model | `--llm meta-llama/Llama-3.2-3B --init-from OpenTSLM/llama-3.2-3b-ecg-sp` |
| Weight pass/fail more in multi-task training | `--task-weights outcome=3` |
| More or fewer successes in training | `--resolved-share 0.5` (0 = natural 12 %) |
| One checkpoint per run (less repetition) | `--one-per-run` |
| Longer thought history | raise `THOUGHT_TOKENS` in `src/agent_ecg/opentslm_data.py`, re-run `opentslm-export` |
| More training | `--epochs 3 --samples-per-epoch 120000` |

Every run is scored by the same `agent-ecg evaluate`, so results stay comparable.
