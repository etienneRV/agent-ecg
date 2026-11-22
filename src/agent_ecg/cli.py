"""``agent-ecg prepare | baselines | evaluate | opentslm-export | compare | replay | demo | channels``."""

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


def _channels(args: argparse.Namespace) -> None:
    from agent_ecg import channels

    channels.run(
        args.data,
        stage=args.stage,
        sample=args.sample or None,
        raw=args.raw,
        workers=args.workers,
        deadline=args.deadline,
    )


def _cv_markdown(result: dict, folds: int) -> str:
    ks = sorted(next(iter(result.values())))
    lines = [
        f"Repository {folds}-fold cross-validation over the checkpoints of all splits. "
        "Each cell: AUROC (mean ± sd over folds) / same-issue pairs.",
        "",
        "| model | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---" * (len(ks) + 1) + "|",
    ]
    for model, by_k in result.items():
        cells = [
            f"{by_k[k]['auroc_fold_mean']:.3f} ± {by_k[k]['auroc_fold_sd']:.3f} / {by_k[k]['same_issue_pairs']:.3f}"
            for k in ks
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _baselines(args: argparse.Namespace) -> None:
    from agent_ecg import baselines
    from agent_ecg.monitor import Monitor

    gold = data.load_checkpoints(args.data)
    thoughts, issues = data.load_thoughts(args.data), data.load_issues(args.data)
    RESULTS.mkdir(exist_ok=True)
    if args.cv:
        result = baselines.cross_validate(gold, thoughts, issues, folds=args.cv)
        (RESULTS / f"cv{args.cv}.json").write_text(json.dumps(result, indent=2))
        text = _cv_markdown(result, args.cv)
        (RESULTS / f"cv{args.cv}.md").write_text(text)
        print(text)
        return
    results = baselines.run(gold, thoughts, issues, args.data / "predictions")
    sections = []
    for name, by_split in results.items():
        report = metrics.evaluate(gold, by_split[args.split], args.split)
        (RESULTS / f"{name.replace('+', '_')}.{args.split}.json").write_text(json.dumps(report, indent=2))
        sections.append((name, report))
    (RESULTS / f"baselines.{args.split}.md").write_text(metrics.concat_reports(sections))
    print(metrics.concat_reports(sections))
    summary = Monitor.from_json(args.data / "models" / "signals.monitor.json").summary()
    (RESULTS / "monitor.signals.md").write_text(
        f"Critical alerts of the signals monitor (fitted on validation)\n\n{summary}\n"
    )
    print(f"critical alerts of the signals monitor, on validation:\n{summary}")


def _replay(args: argparse.Namespace) -> None:
    import numpy as np

    from agent_ecg import baselines, monitor

    models = args.data / "models"
    if not (models / "signals.pkl").exists():
        raise SystemExit("run `agent-ecg baselines` first: it saves the signals model and monitor the replay uses")
    score = baselines.signals_scorer(models)
    fitted = monitor.Monitor.from_json(models / "signals.monitor.json")
    gold = data.load_checkpoints(args.data)
    split = gold[gold["split"] == args.split]
    longest = split.loc[split.groupby("run_id")["k"].idxmax()]
    if args.run_id:
        rows = longest[longest["run_id"].isin(args.run_id)].to_dict("records")
        missing = set(args.run_id) - {row["run_id"] for row in rows}
        if missing:
            raise SystemExit(f"not in the {args.split} split: {sorted(missing)}")
    else:
        # Runs with at least 20 visible steps, so the line has room to move.
        pool = longest[(longest["resolved"] == (args.pick == "resolved")) & (longest["k"] >= 20)]
        rows = pool.sample(min(args.candidates, len(pool)), random_state=0).to_dict("records")
    cuts = [monitor.prefixes(row) for row in rows]
    scores = np.asarray(score([cut for run in cuts for cut in run]))
    ends = np.cumsum([len(run) for run in cuts])
    traces = [monitor.trace(run, scores[end - len(run) : end], fitted) for run, end in zip(cuts, ends, strict=True)]
    chosen = list(zip(rows, traces, strict=True))
    if not args.run_id:
        chosen = _pick(chosen, args.pick)[: args.top]
        print(f"{len(chosen)} {args.pick} {args.split} runs out of {len(rows)} candidates, best first")
        for row, steps in chosen:
            print(f"  {row['run_id']}  {monitor.sparkline([s['health'] for s in steps])}")
    for row, steps in chosen[: 1 if not args.run_id else None]:
        print()
        print(monitor.render(row, steps))
    if args.out:
        lines = [json.dumps({"run_id": row["run_id"], **s}) for row, steps in chosen for s in steps]
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out} ({len(chosen)} runs)")


def _demo(args: argparse.Namespace) -> None:
    from agent_ecg import demo

    replays = demo.load_replay(args.replay)
    if not replays:
        raise SystemExit(f"{args.replay} holds no replay")
    run_id = args.run_id or next(iter(replays))
    if run_id not in replays:
        raise SystemExit(f"{run_id} is not in {args.replay}, which holds {sorted(replays)}")
    monitor = replays[run_id]
    run = demo.fetch_run(run_id)
    if args.at is not None or args.svg is not None:
        step = args.at or demo.first_alarm(monitor) or max(monitor)
        print(
            demo.render(
                run, monitor, step, svg=args.svg, width=args.width, height=args.height, chance=not args.no_chance
            )
        )
        if args.svg:
            print(f"wrote {args.svg} (step {step})")
        return
    demo.play(run, monitor, pace=args.pace, chance=not args.no_chance)


def _pick(chosen: list, pick: str) -> list:
    """Order replays for a demo: ``falling`` runs start stable and turn critical mid-run (ideally near
    step 15, after a long fall); ``resolved`` runs never alert and keep the highest lowest health."""

    def falling(steps: list[dict]) -> tuple | None:
        first = next((s["step"] for s in steps if s["state"] == "critical"), None)
        if steps[0]["state"] != "stable" or first is None or first < 10:
            return None
        drop = steps[0]["health"] - min(s["health"] for s in steps)
        return (drop >= 50, -abs(first - 15), drop)

    def steady(steps: list[dict]) -> tuple | None:
        if any(s["state"] == "critical" for s in steps):
            return None
        return (min(s["health"] for s in steps),)

    rank = falling if pick == "falling" else steady
    ranked = [(rank(steps), i) for i, (_, steps) in enumerate(chosen)]
    return [chosen[i] for key, i in sorted((r for r in ranked if r[0] is not None), reverse=True)]


def _evaluate(args: argparse.Namespace) -> None:
    gold = data.load_checkpoints(args.data)
    reader = pd.read_csv if args.predictions.suffix == ".csv" else pd.read_parquet
    report = metrics.evaluate(gold, reader(args.predictions), args.split, subset=args.subset)
    print(metrics.to_markdown(report, args.predictions.stem))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


def _opentslm_export(args: argparse.Namespace) -> None:
    from agent_ecg import opentslm_data

    default = "data/opentslm" if args.context == "full" else f"data/opentslm-{args.context}"
    custom = args.count_channels or args.head_thought_tokens or args.thought_tokens != opentslm_data.THOUGHT_TOKENS
    if custom and args.out is None:
        raise SystemExit("pass --out for an export with non-default settings, so it never overwrites runs A-C's data")
    out = args.out or Path(default)
    stats = opentslm_data.export(
        args.data,
        out,
        context=args.context,
        head_thought_tokens=args.head_thought_tokens,
        thought_tokens=args.thought_tokens,
        count_channels=args.count_channels,
    )
    print(json.dumps(opentslm_data.export_settings(out / "meta.json")))
    print(f"OpenTSLM samples ({args.context} context) -> {out}")
    print(
        "| k | prompt tokens p50 / p99 | issue cut | issue tokens p50 / p99 | thought tokens p50 / p99 "
        "| runs with steps left out | runs that lose their first step |"
    )
    print("|---|---|---|---|---|---|---|")
    for k, group in stats.groupby("k"):
        dropped = group.loc[group["steps_dropped"] > 0, "steps_dropped"]
        print(
            f"| {k} | {group['prompt_tokens'].median():.0f} / {group['prompt_tokens'].quantile(0.99):.0f} "
            f"| {group['issue_cut'].mean():.1%} | {group['issue_tokens'].median():.0f} / {group['issue_tokens'].quantile(0.99):.0f} "
            f"| {group['thought_tokens'].median():.0f} / {group['thought_tokens'].quantile(0.99):.0f} "
            f"| {(group['steps_dropped'] > 0).mean():.1%} (median left out: {dropped.median() if len(dropped) else 0:.0f}) "
            f"| {group['first_step_lost'].mean():.1%} |"
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

    run = commands.add_parser("baselines", help="fit the baselines and score them, or cross-validate them")
    run.add_argument("--split", default="test", choices=("validation", "test"))
    run.add_argument(
        "--cv", type=int, metavar="FOLDS", help="instead, repository cross-validation over all checkpoints"
    )
    run.set_defaults(run=_baselines)

    replay = commands.add_parser(
        "replay", help="replay runs step by step with the signals model, as the live monitor (needs `baselines`)"
    )
    replay.add_argument("--run-id", action="append", help="a run to replay (repeatable); default: pick runs")
    replay.add_argument(
        "--pick",
        default="falling",
        choices=("falling", "resolved"),
        help="without --run-id: failed runs that turn critical mid-run, or resolved runs that stay steady",
    )
    replay.add_argument("--split", default="test", choices=("validation", "test"))
    replay.add_argument("--candidates", type=int, default=3000, help="runs to replay when picking")
    replay.add_argument("--top", type=int, default=5, help="picked runs to list and write")
    replay.add_argument("--out", type=Path, help="write the replays as JSONL, one line per step")
    replay.set_defaults(run=_replay)

    show = commands.add_parser("demo", help="play a replayed run in the terminal: the agent's steps beside the ECG")
    show.add_argument("replay", type=Path, help="a JSONL file from `agent-ecg replay --out` or opentslm/replay.py")
    show.add_argument("--run-id", help="the run to play; default: the first in the file")
    show.add_argument("--pace", type=float, default=2.0, help="seconds per step")
    show.add_argument("--no-chance", action="store_true", help="hide the chance-of-failure line in the monitor panel")
    show.add_argument("--at", type=int, help="print the frame at this step instead of playing")
    show.add_argument("--svg", type=Path, help="save that frame as an SVG image (default step: the first alarm)")
    show.add_argument("--width", type=int, default=150, help="frame width for --at/--svg")
    show.add_argument("--height", type=int, default=44, help="frame height for --at/--svg")
    show.set_defaults(run=_demo)

    evaluate = commands.add_parser("evaluate", help="score a predictions file (.parquet or .csv)")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--split", default="test", choices=("validation", "test"))
    evaluate.add_argument("--json", type=Path, help="also write the report as JSON here")
    evaluate.add_argument("--subset", action="store_true", help="score only the checkpoints the file covers")
    evaluate.set_defaults(run=_evaluate)

    export = commands.add_parser("opentslm-export", help="write OpenTSLM training samples (needs `prepare` first)")
    export.add_argument(
        "--context",
        default="full",
        choices=("full", "thoughts", "none"),
        help="text beside the series: issue and thoughts, thoughts only, or none (the ECG-only variant)",
    )
    export.add_argument("--out", type=Path, help="default: data/opentslm (full) or data/opentslm-<context>")
    export.add_argument(
        "--count-channels", action="store_true", help="run C2: add 3 count channels (fixed scales) to the 19 series"
    )
    export.add_argument(
        "--head-thought-tokens", type=int, default=0, help="also keep the earliest thoughts, up to this many tokens"
    )
    export.add_argument(
        "--thought-tokens", type=int, default=1024, help="the latest thoughts kept, up to this many tokens"
    )
    export.set_defaults(run=_opentslm_export)

    compare = commands.add_parser("compare", help="paired AUROC gain of predictions A over B, per checkpoint")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--split", default="test", choices=("validation", "test"))
    compare.set_defaults(run=_compare)

    hunt = commands.add_parser(
        "channels", help="screen and confirm new per-step channels on train+validation (results/channels.md)"
    )
    hunt.add_argument("--stage", default="all", choices=("all", "0", "A", "B", "C"))
    hunt.add_argument("--sample", type=float, default=0.3, help="share of repositories for screening; 0 = all")
    hunt.add_argument("--raw", type=Path, help="folder with the release's parquet shards (default: download them)")
    hunt.add_argument("--workers", type=int, default=4, help="shards read in parallel")
    hunt.add_argument("--deadline", default="2026-11-22T09:00:00+01:00", help="start no new fit after this time")
    hunt.set_defaults(run=_channels)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
