"""Pick the scan's runs, or summarise two replays of them (results/opentslm/scan/summary.md).

    python opentslm/scan_summary.py pick data/checkpoints.parquet ids.txt
    ARGS=$(sed 's/^/--run-id /' ids.txt | tr '\n' ' ')
    uv run agent-ecg replay $ARGS --out signals.jsonl
    uv run python opentslm/replay.py runs/full-multi $ARGS --out A.jsonl
    python opentslm/scan_summary.py score data/checkpoints.parquet ids.txt A=A.jsonl signals=signals.jsonl
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_STEPS, TOTAL, SEED = 20, 600, 0


def pick(table_path: str, out: str) -> None:
    t = pd.read_parquet(table_path, columns=["run_id", "split", "k", "resolved"])
    runs = t[t["split"] == "test"].groupby("run_id").agg(k=("k", "max"), resolved=("resolved", "first"))
    long = runs[runs["k"] >= MIN_STEPS]
    good = long[long["resolved"]].index.tolist()
    bad = long[~long["resolved"]].index.tolist()
    rng = np.random.default_rng(SEED)
    bad = sorted(rng.choice(bad, size=min(len(bad), TOTAL - len(good)), replace=False).tolist())
    ids = sorted(good) + bad
    Path(out).write_text("\n".join(ids) + "\n")
    print(
        f"{len(long)} test runs with >= {MIN_STEPS} steps ({len(good)} resolved); scanning all {len(good)} "
        f"resolved + {len(bad)} random failing (seed {SEED})"
    )


def load(path: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["run_id"]].append(r)
    return out


def score(table_path: str, ids_path: str, replays: list[str]) -> None:
    t = pd.read_parquet(table_path, columns=["run_id", "split", "k", "resolved"])
    runs = t[t["split"] == "test"].groupby("run_id").agg(k=("k", "max"), resolved=("resolved", "first"))
    natural = float(runs.loc[runs["k"] >= MIN_STEPS, "resolved"].mean())  # resolved share among long test runs
    resolved = dict(zip(t["run_id"], t["resolved"], strict=True))
    ids = [line.strip() for line in Path(ids_path).read_text().splitlines() if line.strip()]
    first: dict[str, dict[str, int | None]] = {}
    lines = []
    for spec in replays:
        name, path = spec.split("=", 1)
        runs = load(path)
        first[name] = {
            rid: next((r["step"] for r in runs[rid] if r["state"] == "critical"), None) for rid in ids if rid in runs
        }
        f = [rid for rid in first[name] if not resolved[rid]]
        g = [rid for rid in first[name] if resolved[rid]]
        caught = [first[name][rid] for rid in f if first[name][rid] is not None]
        false = [rid for rid in g if first[name][rid] is not None]
        recall, fpr = len(caught) / len(f), len(false) / len(g)
        alarm = recall * (1 - natural) + fpr * natural
        precision = recall * (1 - natural) / alarm if alarm else float("nan")
        lines.append(
            f"| {name} | {len(f)} / {len(g)} | {len(caught) / len(f):.1%} | "
            f"{np.median(caught) if caught else float('nan'):.0f} | {len(false) / len(g):.1%} | "
            f"{precision:.1%} |"
        )
    print(
        "| monitor | failing / resolved runs replayed | failing runs flagged critical | median alarm step | "
        f"resolved runs flagged (false alarms) | alarms that were right, at the natural {natural:.1%} resolved |"
    )
    print("|---|---|---|---|---|---|")
    print("\n".join(lines))
    if len(first) == 2:
        (na, a), (nb, b) = first.items()
        both = [rid for rid in ids if rid in a and rid in b and not resolved[rid]]
        only_a = sum(a[r] is not None and b[r] is None for r in both)
        only_b = sum(b[r] is not None and a[r] is None for r in both)
        pairs = [(a[r], b[r]) for r in both if a[r] is not None and b[r] is not None]
        earlier_a = sum(x < y for x, y in pairs)
        earlier_b = sum(y < x for x, y in pairs)
        print(
            f"\nFailing runs: flagged only by {na}: {only_a}; only by {nb}: {only_b}; by both: {len(pairs)} "
            f"({na} earlier in {earlier_a}, {nb} earlier in {earlier_b}, same step in {len(pairs) - earlier_a - earlier_b})."
        )
        showcase = sorted(
            ((b[r] - a[r], r, a[r], b[r]) for r in both if a[r] is not None and b[r] is not None and a[r] < b[r]),
            reverse=True,
        )[:5]
        early_only = [(r, a[r]) for r in both if a[r] is not None and b[r] is None][:5]
        print(
            f"Showcase, {na} earlier than {nb} (run, {na} step, {nb} step): "
            + ", ".join(f"{r} ({x}, {y})" for _, r, x, y in showcase)
        )
        print(f"Showcase, flagged only by {na} (run, step): " + ", ".join(f"{r} ({x})" for r, x in early_only))
        steady = [r for r in ids if resolved[r] and r in a and a[r] is None and b.get(r) is None][:5]
        print("Resolved runs neither flags: " + ", ".join(steady))


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        pick(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4:])
