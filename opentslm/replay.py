"""Replay runs step by step with a fine-tuned OpenTSLM-SP: what the live monitor would have shown.

    python opentslm/predict.py runs/ecg-only --split validation --tasks outcome
    uv run agent-ecg replay --top 3          # prints test runs that fall to critical mid-run
    python opentslm/replay.py runs/ecg-only --run-id <run id> --out runs/ecg-only/replay.jsonl

The monitor (calibration, health, critical threshold) is fitted on the run folder's validation
predictions, so the replayed test runs never shaped it. Each step builds the sample the way
``agent-ecg opentslm-export`` does, from a prefix of the run's longest checkpoint and with the context
(issue, thoughts or neither) the model trained with. The output has one JSON line per step, the same
format as ``agent-ecg replay --out``: step, health, state, score, p_calibrated, action, reply, events,
status, and a recommendation once the state is critical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_model, p_resolved, pick_device

from agent_ecg import data, monitor, opentslm_data
from agent_ecg import opentslm_samples as samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the training run folder (config.json, best_model.pt)")
    parser.add_argument("--run-id", action="append", required=True, help="a run to replay (repeatable)")
    parser.add_argument("--tables", type=Path, default=Path("data"), help="the tables `agent-ecg prepare` wrote")
    parser.add_argument("--data", type=Path, help="the exported samples; default: the folder the run trained on")
    parser.add_argument(
        "--predictions", type=Path, help="validation predictions; default: <run>/predictions.validation.parquet"
    )
    parser.add_argument("--out", type=Path, help="write the replays as JSONL, one line per step")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    exported = args.data or Path(config["data"])
    meta = exported / "meta.json"
    context = json.loads(meta.read_text())["context"] if meta.exists() else "full"
    post = samples.post_prompts(exported)
    predictions = pd.read_parquet(args.predictions or args.run / "predictions.validation.parquet")
    gold = pd.read_parquet(exported / "validation.parquet", columns=["run_id", "k", "resolved"])
    fitted = monitor.Monitor.fit_predictions(predictions, gold)
    print(f"monitor fitted on {len(predictions)} validation checkpoints ({context} context):\n{fitted.summary()}")

    table = data.load_checkpoints(args.tables)
    longest = table.loc[table.groupby("run_id")["k"].idxmax()].set_index("run_id", drop=False)
    missing = [run for run in args.run_id if run not in longest.index]
    if missing:
        raise SystemExit(f"unknown run ids: {missing}")
    prompts = opentslm_data.Prompts(args.tables, context)
    device = pick_device(args.device)
    model = load_model(config["llm"], device, init_from=config.get("init_from"), lora=not config.get("no_lora", False))
    model.load_from_file(str(args.run / "best_model.pt"))
    model.eval()

    def score(cuts: list[dict]) -> list[float]:
        rows = []
        for cut in cuts:
            text, _ = prompts(cut["run_id"], cut["issue_id"], cut["k"])
            series_text, series = opentslm_data.time_series(cut)
            rows.append({"pre_prompt": text, "time_series_text": series_text, "time_series": series})
        return p_resolved(model, rows, post)

    lines = []
    for run_id in args.run_id:
        row = longest.loc[run_id].to_dict()
        steps = monitor.replay(row, score, fitted)
        print()
        print(monitor.render(row, steps))
        lines += [json.dumps({"run_id": run_id, **step}) for step in steps]
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
