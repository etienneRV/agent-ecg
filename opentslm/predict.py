"""Predict with a fine-tuned OpenTSLM-SP and write a file ``agent-ecg evaluate`` scores.

    python opentslm/predict.py runs/llama1b-outcome --split validation
    python opentslm/predict.py runs/llama1b-outcome --split test --threshold-from runs/llama1b-outcome/predictions.validation.parquet
    uv run agent-ecg evaluate runs/llama1b-outcome/predictions.test.parquet

- **Will it pass?** ``p_resolved`` compares the model's likelihood of answering "resolved" and
  "unresolved" (softmax over the two summed log-probabilities), so AUROC can be computed.
  ``pred_resolved`` cuts it at the threshold that maximises balanced accuracy on validation
  (``--threshold-from``), or at 0.5.
- **How will it end?** ``pred_ending`` is the most likely of the six endings, scored the same way.
- **Status report:** ``pred_status`` is the greedy continuation (skip with ``--tasks outcome,ending``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import candidate_logprobs, embed_prompts, generate, load_model, model_sample, p_resolved, pick_device

from agent_ecg import opentslm_samples as samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the training run folder (config.json, best_model.pt)")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--data", type=Path, default=Path("data/opentslm"))
    parser.add_argument("--tasks", default="outcome,ending,status")
    parser.add_argument("--max", type=int, help="only this many checkpoints (score with `evaluate --subset`)")
    parser.add_argument("--batch-size", type=int, default=16, help="checkpoints per status-generation batch")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="status reports are at most 116 tokens")
    parser.add_argument("--threshold-from", type=Path, help="validation predictions to choose the decision threshold")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    device = pick_device(args.device)
    model = load_model(config["llm"], device, init_from=config.get("init_from"), lora=not config.get("no_lora", False))
    model.load_from_file(str(args.run / "best_model.pt"))
    model.eval()
    prompts = samples.post_prompts(args.data)
    tasks = [t for t in args.tasks.split(",") if t]
    table = samples.load(args.data, args.split, cap=args.max)
    rows = table.to_dict("records")
    out = pd.DataFrame({"run_id": table["run_id"], "k": table["k"]})
    started = time.perf_counter()

    if "outcome" in tasks or "ending" in tasks:
        probabilities, endings = [], []
        for i, row in enumerate(rows):
            if "outcome" in tasks:
                probabilities += p_resolved(model, [row], prompts)
            if "ending" in tasks:
                (prompt,) = embed_prompts(model, [model_sample(row, "ending", prompts)])
                scores = candidate_logprobs(model, prompt, list(samples.ENDINGS))
                endings.append(samples.ENDINGS[int(scores.argmax())])
            if (i + 1) % 500 == 0:
                print(
                    f"  scored {i + 1}/{len(rows)} ({(time.perf_counter() - started) / (i + 1):.2f} s each)", flush=True
                )
        if probabilities:
            out["p_resolved"] = probabilities
            cut = 0.5
            if args.threshold_from:
                val = pd.read_parquet(args.threshold_from).merge(
                    samples.load(args.data, "validation")[["run_id", "k", "resolved"]], on=["run_id", "k"]
                )
                cut = samples.threshold(val["resolved"].to_numpy(bool), val["p_resolved"].to_numpy())
            out["pred_resolved"] = out["p_resolved"] >= cut
            print(f"decision threshold {cut:.4f}")
        if endings:
            out["pred_ending"] = endings

    if "status" in tasks:
        reports = []
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            reports += generate(
                model, embed_prompts(model, [model_sample(r, "status", prompts) for r in chunk]), args.max_new_tokens
            )
            print(f"  generated {len(reports)}/{len(rows)}", flush=True)
        out["pred_status"] = reports

    path = args.run / f"predictions.{args.split}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} checkpoints, {(time.perf_counter() - started) / 60:.1f} min)")
    if "p_resolved" in out:
        merged = out.merge(table[["run_id", "k", "resolved"]], on=["run_id", "k"])
        from sklearn.metrics import roc_auc_score

        for k, group in merged.groupby("k"):
            if 0 < group["resolved"].sum() < len(group):
                print(f"  quick check, AUROC at k={k}: {roc_auc_score(group['resolved'], group['p_resolved']):.3f}")


if __name__ == "__main__":
    main()
