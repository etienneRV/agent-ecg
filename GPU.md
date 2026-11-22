# Fine-tuning OpenTSLM on a GPU

Step-by-step for the Nebius machine. OpenTSLM-SP on Llama 3.2 1B trains only ~14 M parameters (encoder,
projector, LoRA), so one GPU is enough: 24 GB for ecg (ECG only, 22.0 GB peak) and ecg-thought (thoughts + ECG, 20.7 GB
at batch 4 x 4), 32 GB or more for ecg-full (issue + thoughts + ECG, 27.7 GB at batch 4 x 4). We used L40S (46 GB).

## 0. Before you start

- Your Hugging Face account needs access to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  (request it on the page; Meta approves it), and a **read** token from https://huggingface.co/settings/tokens.
- Without that approval, add `--llm unsloth/Llama-3.2-1B` to every `train.py` command: the same
  weights, not gated, and `--init-from OpenTSLM/llama-3.2-1b-ecg-sp` still works (`load_model` puts the
  checkpoint on whichever copy of Llama 3.2 1B you name). `predict.py` and `replay.py` read the model
  from the run's `config.json`.

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
uv run --no-sync agent-ecg baselines                       # the baselines + the signals monitor (~15 min)
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

If memory runs out: `--batch-size 4 --grad-accum 4` (the same effective batch), or add
`--gradient-checkpointing` (about 30 % slower).

**Attention.** OpenTSLM loads Llama with eager attention, which keeps a [batch, heads, length, length]
matrix per layer for the backward pass, so memory grows with the square of the prompt length.
`opentslm/common.py` switches the loaded model to PyTorch's SDPA (`ATTENTION = "sdpa"`): the same
attention, computed without that matrix. The OpenTSLM package itself is unchanged. Measured on one
L40S (46 GB), from `OpenTSLM/llama-3.2-1b-ecg-sp`:

| Run | Prompt tokens (p99) | Eager attention | SDPA |
| --- | --- | --- | --- |
| ecg, ECG only | ~560 | batch 8: 21.9 GB, 1.2 s/step | - |
| ecg-full, issue + thoughts + ECG | ~2,300 | out of memory at batch 8 and 4; 4 x 4 with checkpointing: 7.8 s/step (~17 h) | out of memory at batch 8; 4 x 4: 27.1 GB, 3.1 s/step (~8 h) |

So train ecg-full with `--batch-size 4 --grad-accum 4`, and ecg-thought too (its prompts reach ~1,700 tokens, so batch 8
would need an estimated ~40 GB); ecg fits the default.

Not on a Mac: OpenTSLM loads Llama in bfloat16, which has no fast CPU path there, and Apple's MPS
backend grew past 30 GB on this smoke test. The scripts were checked on a Mac CPU in float32 instead
(20 steps at 4-5 s each, 7.9 GB): use the GPU for real runs.

## 3. The runs, in order

The three models are **ecg** (the ECG only; run folder `ecg-only`), **ecg-thought** (thoughts + ECG;
`thoughts-outcome`) and **ecg-full** (issue + thoughts + ECG, three tasks; `full-multi`); the second-round
runs are **ecg v2** (`ecg-counts`) and **ecg-full v2** (`full-multi-a2`). Older commits call them C, B, A, C2
and A2. The CPU models each is compared with are the **baselines**.

Run them inside `tmux` (or with `nohup`) so they survive a dropped connection. All three start from
OpenTSLM's ECG model, the last stage of its curriculum: it already reads 12 labelled leads at once,
each labelled "..., it has mean x and std y:" like our 19 series.

```bash
mkdir -p runs
# ecg: the ECG alone, pass/fail. The demo's model: whatever it says, it read from the series.
uv run --no-sync python opentslm/train.py --name ecg-only --data data/opentslm-none --tasks outcome \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp 2>&1 | tee runs/ecg-only.log
# ecg-thought: the agent's thoughts beside the ECG, pass/fail
uv run --no-sync python opentslm/train.py --name thoughts-outcome --data data/opentslm-thoughts --tasks outcome \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp --batch-size 4 --grad-accum 4 2>&1 | tee runs/thoughts-outcome.log
# ecg-full: issue, thoughts and ECG; all three tasks
uv run --no-sync python opentslm/train.py --name full-multi --data data/opentslm \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp --batch-size 4 --grad-accum 4 2>&1 | tee runs/full-multi.log
```

Why this order:
- **ecg is the demo** and the cleanest TSLM result: the prompt holds no issue and no thoughts (about
  450-560 tokens), so any skill comes from reading the trace. It is also the fastest to train.
- **ecg-thought adds what the agent writes.** The thoughts baseline is the strongest baseline, so ecg-thought shows
  what the series add to the text.
- **ecg-full adds the issue and the status report.** It needs the most GPU time (~1,400 tokens a sample);
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
uv run --no-sync agent-ecg compare $R/predictions.test.parquet data/predictions/signals.test.parquet  # vs the baseline with the same inputs
uv run --no-sync agent-ecg compare $R/predictions.test.parquet data/predictions/issue.test.parquet    # vs the issue alone
```

- `predict.py` reads the samples the run trained on (`config.json`), so an ECG-only model is
  scored on ECG-only prompts.
- `--calibrate-from` fits the monitor on the validation predictions: an isotonic map per checkpoint
  (`p_calibrated`, scored by its Brier score), health from 0 to 100, and the critical threshold. It
  saves `monitor.json` in the run folder and prints how often a critical alert was right on validation.
- `evaluate` adds **same-issue pairs**: among runs on the same issue, one resolved and one not, how
  often the model ranks the resolved one higher. It answers "does it only learn which issues are hard?".
- For ecg-full, drop `--tasks outcome` to also predict the ending and the status report (slow: the
  reports are generated text).

Copy `report.test.json`, the `compare` output and the predictions file into `results/` and commit
them; checkpoints stay out of git (`runs/` is ignored).

## 5. The demo: replay a run as the live monitor

```bash
uv run --no-sync agent-ecg replay --top 3 --out results/replay.signals.jsonl   # picks test runs that fall to critical
uv run --no-sync python opentslm/replay.py runs/ecg-only --run-id <a run id printed above> \
    --out results/replay.ecg-only.jsonl
uv run --no-sync agent-ecg replay --pick resolved --top 3   # contrast: resolved runs that stay steady
uv run --no-sync agent-ecg demo results/replay.ecg-only.jsonl   # the terminal ECG, for the demo video
```

Each JSON line is one step: `step`, `health` (0-100), `state` (`stable` / `watch` / `low` /
`critical`), `p_calibrated`, the step's `action` and `reply`, `events` (repeated command, edit
rejected, same file again, ...), the rule-made `status`, and a `recommendation` ("Stop the agent to
save credits: ... failed 97 % of the time") once the state is critical. That is what the terminal ECG
shows. The alarm goes on when health is at most 15 in a zone where at least 95 % of validation runs
failed, two steps in a row, and holds until health is back to 30.

## 6. Final runs: ecg v2 and ecg-full v2

Two runs in parallel, one per problem found in ecg, ecg-thought and ecg-full (details: `results/SUMMARY.md`). Everything not
listed stays as in the original run. Each export records its settings in `meta.json`, and
`opentslm/replay.py` rebuilds prompts from them.

**ecg v2 = ecg + 3 count channels.** ecg and ecg-thought ignore simple counts: added to their own scores, the three channels
raise AUROC by 0.03-0.07 on validation. Each channel has a fixed scale, capped at 1 and stated in its
label, and uses only past steps:
- steps since the last successful command (/30);
- tracebacks so far (/10);
- code runs so far (/10).

The prompt grows from 558 to 650 tokens at 30 steps (median).

```bash
uv run --no-sync agent-ecg opentslm-export --context none --count-channels --out data/opentslm-none-counts
uv run --no-sync python opentslm/train.py --name ecg-counts --data data/opentslm-none-counts --tasks outcome \
    --init-from OpenTSLM/llama-3.2-1b-ecg-sp 2>&1 | tee runs/ecg-counts.log
```

**ecg-full v2 = ecg-full against overfitting, keeping the early thoughts.** ecg-thought and ecg-full overfit: ecg-thought's training loss fell to 0
while its validation loss doubled, and ecg-full peaked at step 2,500 of 7,500. ecg-full also lost the first thoughts of
long runs (97 % of 30-step checkpoints), and a word-count model fed ecg-full's window falls from 0.70 to 0.63
AUROC at 30 steps. The changes:
- one epoch at the natural share of successes, so each run is seen about once;
- LoRA dropout 0.1;
- finer early stopping;
- the first 512 thought tokens kept as well as the latest 1,536. 0 % of checkpoints lose their first step; the median prompt is 2,700 tokens at 30 steps, so plan about 40 % more time per step.

```bash
uv run --no-sync agent-ecg opentslm-export --head-thought-tokens 512 --thought-tokens 1536 --out data/opentslm-a2
uv run --no-sync python opentslm/train.py --name full-multi-a2 --data data/opentslm-a2 \
    --epochs 1 --resolved-share 0.17 --lora-dropout 0.1 --eval-every 250 --val-checkpoints 4000 \
    --batch-size 4 --grad-accum 4 --init-from OpenTSLM/llama-3.2-1b-ecg-sp 2>&1 | tee runs/full-multi-a2.log
```

On an 80 GB GPU, the default batch (8 x 2) likely fits; the smoke test tells. Then score both as in section 4.

**How we judge them** (fixed before the runs; test, paired repository bootstrap with `agent-ecg compare`):
- **ecg v2 beats ecg** if the 95 % interval of the gain is above zero at two of steps 5 / 10 / 15.
- **ecg-full v2 beats ecg-full** if the interval is above zero at two of steps 15 / 20 / 30 (where the thought window bites:
  38 / 77 / 97 % of ecg-full's checkpoints lost early thoughts), with steps 5 / 10 not worse beyond noise.
  The same test runs still going at step 30 should also no longer score lower as the run goes on
  (ecg-full: 0.68 at step 5, 0.62 at step 30).
- A winner replaces its original in the results and the demo. Otherwise we report "tried, no gain",
  with the validation numbers. Test intervals are about ±0.05 wide, so a real +0.02 can miss the bar.

**Run them with one command each**, one per GPU (in `tmux`); `DRY_RUN=1` prints the steps first:

```bash
opentslm/final_run.sh c2    # first GPU
opentslm/final_run.sh a2    # second GPU
```

It builds the export, smoke-tests the largest batch that leaves 20 % of the GPU free (8 x 2, else 4 x 4,
else 4 x 4 with gradient checkpointing), trains, scores validation and test as in section 4, and writes the
two comparisons the rule above needs: against the original run (`results/opentslm/...`) and against the
baseline (`results/baselines/...`). Everything lands in `runs/<name>/` and `runs/<name>.final.log`; a trained run
is never overwritten.

### Design notes: why ecg-full v2 keeps the first and the latest thoughts

ecg-full keeps the latest 1,024 tokens of thoughts, so long runs lose their first steps, where the agent reads the
issue and plans (6 % of checkpoints at 10 steps, 77 % at 20, 97 % at 30), while the thoughts baseline reads
everything. Two fixes were weighed:

- **A tiered budget** (the latest 10 steps in full, older steps cut to their first 32 tokens), proposed first.
- **The first 512 plus the latest 1,536 tokens** (head + tail), chosen. It is one export setting, and the
  defaults reproduce ecg, ecg-thought and ecg-full byte for byte.

A validation check settled it (`opentslm/thought_window_check.py`, results in
[results/thought_window_check.json](results/thought_window_check.json)): the thoughts baseline, fitted on
8,000 training runs with a fixed C and scored on all 22,418 validation checkpoints, with each variant cutting
the thoughts exactly as the export does. Test was not read.

| Thoughts kept | AUROC at 5 / 10 / 15 / 20 / 30 steps (validation) |
| --- | --- |
| latest 1,024 (ecg-thought, ecg-full) | 0.681 / 0.728 / 0.720 / 0.722 / 0.683 |
| latest 2,048 | 0.683 / 0.729 / 0.721 / 0.732 / 0.752 |
| first 512 + latest 1,024 | 0.683 / 0.730 / 0.720 / 0.729 / 0.749 |
| **first 512 + latest 1,536 (ecg-full v2)** | **0.683 / 0.730 / 0.721 / 0.733 / 0.757** |
| tiered (latest 10 steps in full, older cut to 32 tokens) | 0.684 / 0.730 / 0.714 / 0.720 / 0.729 |
| all thoughts, uncut | 0.683 / 0.730 / 0.719 / 0.732 / 0.753 |

ecg-full v2's window is best or tied at every checkpoint and recovers what keeping every thought gives: +0.07 over ecg-full's
window at 30 steps, +0.01 at 20, nothing at 5-15, where nothing was cut. The tiered design loses information
by cutting older thoughts to 32 tokens. This is a word-count model, not the TSLM, and the top variants differ
by under 0.01.

Both designs follow published findings:

1. Keeping the beginning and the end of a long text beat keeping either alone for BERT classification
   ([Sun et al. 2019](https://arxiv.org/abs/1905.05583)); later work finds the beginning alone can do as well,
   depending on the data ([Investigating text shortening strategy in BERT, 2024](https://arxiv.org/abs/2403.12799)).
2. Language models use the start and the end of a long input best and the middle worst, and do worse on longer
   inputs ([Liu et al. 2023, *Lost in the Middle*](https://arxiv.org/abs/2307.03172)).
3. Coding agents manage their own history this way: SWE-agent keeps its reasoning and recent tool outputs and
   masks older outputs. Inside SWE-agent on SWE-bench Verified this halved cost while matching the solve rate,
   and the best window was setup-specific
   ([Lindenbauer et al. 2025, *The Complexity Trap*](https://arxiv.org/abs/2508.21433)).

None of these covers our case (a fine-tuned model classifying a run from its thoughts), which is why the
choice rests on the validation check above. (The first comparison, in PR #4, scored test; the validation
check alone picks the same window, so the choice no longer depends on test.)

## Results

| Run | Text beside the ECG | Tasks | AUROC 5 / 10 / 15 / 20 / 30 steps (test) | Same-issue pairs, 10 steps | Brier, calibrated, 10 steps |
| --- | --- | --- | --- | --- | --- |
| baseline: signals (same inputs as ecg) | none | pass/fail | 0.611 / 0.645 / 0.674 / 0.712 / 0.728 | 0.610 | 0.1196 |
| baseline: thoughts (the bar for ecg-thought) | thoughts | pass/fail | 0.680 / 0.703 / 0.710 / 0.701 / 0.701 | 0.562 | 0.1139 |
| baseline: issue+signals+thoughts (the bar for ecg-full) | issue, thoughts | pass/fail | 0.691 / 0.709 / 0.720 / 0.719 / 0.700 | 0.605 | 0.1126 |
| ecg (`ecg-only`) | none | pass/fail | 0.583 / 0.611 / 0.616 / 0.620 / 0.615 | 0.552 | 0.1195 |
| ecg-thought (`thoughts-outcome`) | thoughts | pass/fail | 0.624 / 0.638 / 0.643 / 0.647 / 0.596 | 0.520 | 0.1189 |
| ecg-full (`full-multi`) | issue, thoughts | all three | 0.657 / 0.677 / 0.684 / 0.679 / 0.624 | 0.596 | 0.1184 |

Each run is below its baseline. ecg-full is closest: 0.03-0.04 under issue+signals+thoughts, inside the paired
repository-bootstrap interval at 5-20 steps and outside it at 30. ecg-thought and ecg are significantly below theirs at
most checkpoints. ecg-full beats ecg-thought by 0.03-0.04 at every checkpoint (borderline at 10 and 15 steps), but ecg-full differs
in two ways at once (the issue, and three tasks instead of one), so the gain cannot be credited to either.
ecg trained with eager attention, ecg-thought and ecg-full with SDPA (same math). Compute: one L40S per run, training 2.8 h
(ecg), 5.3 h (ecg-thought) and 7.1 h (ecg-full), 15.2 GPU-hours in all; scoring pass/fail on validation and test took
0.5-1.4 h per run, and ecg-full's generated status reports on the full test split about 3 h more. Reports, predictions, `compare` outputs and
configs: [results/opentslm/](results/opentslm/). As a live monitor (600 test runs replayed step by step), ecg-full
and the signals monitor each flag about a third of the failing runs, mostly different ones; see
[results/opentslm/scan/summary.md](results/opentslm/scan/summary.md). Which agent LLM wrote a run does
not drive any score (within each agent, AUROC changes by at most 0.012); issue difficulty partly does:
[results/shortcut_checks.md](results/shortcut_checks.md). The baselines' test predictions are in
[results/baselines/](results/baselines/), so `agent-ecg compare` runs without rebuilding them.

The test split is a hard draw for the signals: repository cross-validation (`agent-ecg baselines
--cv 5`) gives them 0.663 / 0.711 / 0.729 / 0.747 / 0.775 and same-issue pairs of 0.615 at 10 steps.
Compare a TSLM with the baseline rows on the same test split, as above.

## Knobs worth trying

| Idea | Flag |
| --- | --- |
| Another starting point (3-axis motion, an earlier curriculum stage) | `--init-from OpenTSLM/llama-3.2-1b-har-sp` |
| Bigger base model | `--llm meta-llama/Llama-3.2-3B --init-from OpenTSLM/llama-3.2-3b-ecg-sp` |
| Weight pass/fail more in multi-task training | `--task-weights outcome=3` |
| More or fewer successes in training | `--resolved-share 0.5` (0 = the natural share, 17 % of training runs) |
| One fixed checkpoint per run | `--one-per-run` (does not reduce repetition: the sampler already weights runs equally) |
| Longer thought history | raise `THOUGHT_TOKENS` in `src/agent_ecg/opentslm_data.py`, re-run `opentslm-export` |
| More training | `--epochs 3 --samples-per-epoch 120000` |

Every run is scored by the same `agent-ecg evaluate`, so results stay comparable.
