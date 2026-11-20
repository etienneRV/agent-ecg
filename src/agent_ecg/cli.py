"""``agent-ecg prepare | baselines | evaluate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agent_ecg import data, metrics

RESULTS = Path("results")


def _prepare(args: argparse.Namespace) -> None:
    release = args.release or data.build(args.registry)
    print(f"release: {release}")
    data.export(Path(release), args.data)
    table = data.load_checkpoints(args.data)
    print(f"{len(table)} checkpoints, {table['run_id'].nunique()} runs -> {args.data}")
    share = metrics.describe_matches(table.sample(2000, random_state=0))
    print(f"descriptions rebuilt from the signals match the targets for {share:.1%} of a 2,000-checkpoint sample")


def _baselines(args: argparse.Namespace) -> None:
    from agent_ecg import baselines

    gold = data.load_checkpoints(args.data)
    results = baselines.run(gold, data.load_thoughts(args.data), data.load_issues(args.data), args.data / "predictions")
    RESULTS.mkdir(exist_ok=True)
    sections = []
    for name, by_split in results.items():
        report = metrics.evaluate(gold, by_split[args.split], args.split)
        (RESULTS / f"{name.replace('+', '_')}.{args.split}.json").write_text(json.dumps(report, indent=2))
        sections.append((name, report))
    (RESULTS / f"baselines.{args.split}.md").write_text(metrics.concat_reports(sections))
    print(metrics.concat_reports(sections))


def _evaluate(args: argparse.Namespace) -> None:
    gold = data.load_checkpoints(args.data)
    reader = pd.read_csv if args.predictions.suffix == ".csv" else pd.read_parquet
    report = metrics.evaluate(gold, reader(args.predictions), args.split, subset=args.subset)
    print(metrics.to_markdown(report, args.predictions.stem))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


def _opentslm_export(args: argparse.Namespace) -> None:
    from agent_ecg import opentslm_data

    stats = opentslm_data.export(args.data, args.out)
    print(f"OpenTSLM samples -> {args.out}")
    print("| k | issue cut | issue tokens p50 / p99 | thought tokens p50 / p99 | runs with early steps dropped |")
    print("|---|---|---|---|---|")
    for k, group in stats.groupby("k"):
        print(
            f"| {k} | {group['issue_cut'].mean():.1%} | {group['issue_tokens'].median():.0f} / {group['issue_tokens'].quantile(0.99):.0f} "
            f"| {group['thought_tokens'].median():.0f} / {group['thought_tokens'].quantile(0.99):.0f} "
            f"| {(group['steps_dropped'] > 0).mean():.1%} (median dropped when cut: {group.loc[group['steps_dropped'] > 0, 'steps_dropped'].median():.0f}) |"
        )


def _compare(args: argparse.Namespace) -> None:
    gold = data.load_checkpoints(args.data)
    read = lambda path: pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)  # noqa: E731
    result = metrics.paired_auroc_gain(gold, read(args.a), read(args.b), args.split)
    print(f"AUROC of {args.a.stem} minus {args.b.stem} ({args.split}, paired repository bootstrap)")
    for k, row in result.items():
        low, high = row["gain_95"]
        print(
            f"  {k}: {row['gain']:+.3f} [{low:+.3f}, {high:+.3f}]  share of resamples not better: {row['p_not_better']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-ecg", description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="where the flattened tables live")
    commands = parser.add_subparsers(required=True)

    prepare = commands.add_parser("prepare", help="build the TimeNet release and flatten it")
    prepare.add_argument("--registry", type=Path, default=Path("data/registry"), help="local TimeNet registry")
    prepare.add_argument("--release", type=Path, help="use an already built release directory instead")
    prepare.set_defaults(run=_prepare)

    run = commands.add_parser("baselines", help="fit the four baselines and score them")
    run.add_argument("--split", default="test", choices=("validation", "test"))
    run.set_defaults(run=_baselines)

    evaluate = commands.add_parser("evaluate", help="score a predictions file (.parquet or .csv)")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--split", default="test", choices=("validation", "test"))
    evaluate.add_argument("--json", type=Path, help="also write the report as JSON here")
    evaluate.add_argument("--subset", action="store_true", help="score only the checkpoints the file covers")
    evaluate.set_defaults(run=_evaluate)

    export = commands.add_parser("opentslm-export", help="write OpenTSLM training samples (needs `prepare` first)")
    export.add_argument("--out", type=Path, default=Path("data/opentslm"))
    export.set_defaults(run=_opentslm_export)

    compare = commands.add_parser("compare", help="paired AUROC gain of predictions A over B, per checkpoint")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--split", default="test", choices=("validation", "test"))
    compare.set_defaults(run=_compare)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
