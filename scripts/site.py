"""Build the project website (site/index.html) from the result files and the charts.

    uv run python scripts/charts.py --theme both   # first: the figures
    uv run python scripts/site.py                   # then: site/index.html

Every number on the page is read from results/ or data/checkpoints.parquet; rerun after they change.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from string import Template

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts

ROOT = charts.ROOT
RESULTS = charts.RESULTS
SITE = ROOT / "site"
SITE_URL = "https://etiennerv.github.io/agent-ecg/"  # GitHub Pages address of the repo; used for link previews
FIGS = ("hero_ecg", "auroc_by_step", "monitor_numbers", "failing_vs_resolved", "training_curves")
PAPER_FIGS = ("same_issue", "paper_results")  # scripts/paper_figures.py
ENDINGS = {"context_limit": "ran out of context", "submitted": "submitted a patch", "early_exit": "exited early"}


def pct(x: float) -> str:
    return f"{100 * x:.0f}&nbsp;%"


def figure(name: str, alt: str, caption: str) -> str:
    return (
        f'<figure class="chart"><img src="figures/{name}.svg" alt="{alt}" loading="lazy">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def data_numbers() -> dict:
    t = pd.read_parquet(charts.CHECKPOINTS, columns=["run_id", "repository", "split", "k", "resolved"])
    by_k = t.groupby("k").agg(runs=("run_id", "nunique"), resolved=("resolved", "mean"))
    test = t[t["split"] == "test"]
    rows = "".join(
        f"<tr><td>{k}</td><td>{int(r.runs):,}</td><td>{pct(1 - r.resolved)}</td></tr>" for k, r in by_k.iterrows()
    )
    return {
        "n_runs": f"{t['run_id'].nunique():,}",
        "n_ckpt": f"{len(t):,}",
        "fail_first": pct(1 - by_k["resolved"].iloc[0]),
        "fail_last": pct(1 - by_k["resolved"].iloc[-1]),
        "k_first": int(by_k.index[0]),
        "k_last": int(by_k.index[-1]),
        "test_runs": f"{test['run_id'].nunique():,}",
        "test_repos": f"{test['repository'].nunique()}",
        "data_rows": rows,
    }


def auroc_table() -> dict:
    tslm = {k: charts.model_auroc(p / "report.test.json") for k, p in charts.tslm_runs().items()}
    names = charts.tslm_labels()
    cpu = {
        n: charts.model_auroc(RESULTS / f"{n}.test.json")
        for n in ("signals", "thoughts", "issue_signals_thoughts", "issue")
    }
    models = [
        ("Baseline", "signals", "the ECG signals", cpu["signals"]),
        ("Baseline", "thoughts", "the agent's thoughts", cpu["thoughts"]),
        ("Baseline", "issue + signals + thoughts", "everything", cpu["issue_signals_thoughts"]),
        ("Baseline", "issue only", "the issue text", cpu["issue"]),
        ("TSLM", names["C"], "the ECG", tslm["C"]),
        ("TSLM", names["B"], "thoughts + ECG", tslm["B"]),
        ("TSLM", f"{names['A']} (3 tasks)", "issue + thoughts + ECG", tslm["A"]),
    ]
    best = {k: max(m[3][k][0] for m in models) for k in charts.STEPS}
    rows = []
    for kind, name, reads, m in models:
        cells = "".join(f'<td class="{"best" if m[k][0] == best[k] else ""}">{m[k][0]:.3f}</td>' for k in charts.STEPS)
        rows.append(
            f'<tr class="{kind.lower()}"><th><span class="tag">{kind}</span> {name}</th><td class="reads">{reads}</td>{cells}</tr>'
        )
    early = [k for k in charts.STEPS if k <= 20]
    gaps = [cpu["issue_signals_thoughts"][k][0] - tslm["A"][k][0] for k in early]
    sig = cpu["signals"]
    return {
        "label_a": names["A"],
        "label_b": names["B"],
        "label_c": names["C"],
        "mon_name": charts.MONITOR_MODEL,
        "auroc_head": "".join(f"<th>{k}</th>" for k in charts.STEPS),
        "auroc_rows": "".join(rows),
        "gap_lo": f"{min(gaps):.2f}",
        "gap_hi": f"{max(gaps):.2f}",
        "early": f"{early[0]}&ndash;{early[-1]}",
        "sig_first": f"{sig[charts.STEPS[0]][0]:.3f}",
        "sig_last": f"{sig[charts.STEPS[-1]][0]:.3f}",
        "sig_first2": f"{sig[charts.STEPS[0]][0]:.2f}",
        "sig_last2": f"{sig[charts.STEPS[-1]][0]:.2f}",
        "c_lo": f"{min(v[0] for v in tslm['C'].values()):.2f}",
        "c_hi": f"{max(v[0] for v in tslm['C'].values()):.2f}",
        "iss_lo": f"{min(v[0] for v in cpu['issue'].values()):.2f}",
        "iss_hi": f"{max(v[0] for v in cpu['issue'].values()):.2f}",
    }


def monitor() -> dict:
    m = charts.scan_summary()
    a, s = m["A"], m["signals"]
    row = lambda name, r: (  # noqa: E731
        f"<tr><th>{name}</th><td>{r['flagged']:.1f}&nbsp;%</td><td>{r['median_step']}</td>"
        f"<td>{r['false_alarms']:.1f}&nbsp;%</td><td>{r['right']:.1f}&nbsp;%</td></tr>"
    )
    return {
        "mon_total": a["failing"] + a["resolved"],
        "mon_failing": a["failing"],
        "mon_resolved": a["resolved"],
        "mon_rows": row(charts.MONITOR_MODEL, a) + row(charts.SIGNALS_BASELINE, s),
        "a_right": f"{a['right']:.0f}&nbsp;%",
        "a_flagged": f"{a['flagged']:.0f}&nbsp;%",
        "a_false": f"{a['false_alarms']:.0f}&nbsp;%",
        "a_median": a["median_step"],
        "s_median": s["median_step"],
        "natural": f"{m['natural']:.1f}&nbsp;%",
        "both": m["both"],
        "a_only": m["a_only"],
        "s_only": m["s_only"],
        "hold": charts.HOLD,
    }


def showcase() -> dict:
    a = charts.replay(charts.SCAN / "replay.showcase.A.jsonl", charts.SHOWCASE)
    s = charts.replay(charts.SCAN / "replay.showcase.signals.jsonl", charts.SHOWCASE)
    row = charts.run_row(charts.SHOWCASE)
    step = charts.alarm_step(a)
    return {
        "show_issue": row["instance_id"],
        "show_steps": len(row["action_kind"]),
        "show_alarm": step,
        "show_end": ENDINGS.get(row["ending"], row["ending"].replace("_", " ")),
        "show_signals": "never alarms" if charts.alarm_step(s) is None else f"alarms at step {charts.alarm_step(s)}",
    }


def compute() -> dict:
    text = (RESULTS / "SUMMARY.md").read_text()
    hit = re.search(
        r"Training ([\d.]+) h \(ecg\), ([\d.]+) h \(ecg-thought\), ([\d.]+) h \(ecg-full\): ([\d.]+) GPU-hours on (\w+)",
        text,
    )
    if not hit:
        raise SystemExit("could not find the compute line in results/SUMMARY.md; update compute()")
    c, b, a, total, gpu = hit.groups()
    # all GPU time per round, from the ledger table under "Compute" (training, scoring, replays, smoke tests)
    ledger = {
        rnd: re.search(rf"^\| (\w+) \({rnd} round\) \|.*\| about ([\d.]+) GPU-hours \|$", text, flags=re.M)
        for rnd in ("first", "second")
    }
    if not all(ledger.values()):
        raise SystemExit("could not find the compute ledger in results/SUMMARY.md; update compute()")
    cfg = json.loads((charts.tslm_runs()["A"] / "config.json").read_text())
    return {
        "h_c": c,
        "h_b": b,
        "h_a": a,
        "gpu_hours": total,
        "gpu": gpu,
        "gpu_total": ledger["first"].group(2),
        "r2_gpu_total": ledger["second"].group(2),
        "base": cfg["init_from"],
        "llm": cfg["llm"],
    }


def other_tasks() -> dict:
    """A's ending and status-report scores (full test split if present, else the 4,000-checkpoint subset)."""
    base = RESULTS / "opentslm"
    path = next(
        p
        for p in (base / "full-multi.alltasks", base / "full-multi.alltasks-4000")
        if (p / "report.test.json").exists()
    )
    a = json.loads((path / "report.test.json").read_text())
    models = [
        (charts.MONITOR_MODEL, "tslm", a),
        (charts.SIGNALS_BASELINE, "baseline", json.loads((RESULTS / "signals.test.json").read_text())),
        ("majority", "baseline", json.loads((RESULTS / "majority.test.json").read_text())),
    ]
    rows = []
    for name, kind, r in models:
        e, st = r["ending"]["all (pooled)"], r["status"]["all (pooled)"]
        rows.append(
            f'<tr class="{kind}"><th>{name}</th><td>{e["accuracy"]:.3f}</td><td>{e["macro_f1"]:.3f}</td>'
            f"<td>{100 * st['parse_rate']:.1f}&nbsp;%</td><td>{st['remaining_mae']:.1f} ({st['remaining_median_ae']:.0f})</td>"
            f"<td>{st['resolved_balanced_accuracy']:.3f}</td></tr>"
        )
    full = path.name == "full-multi.alltasks"
    return {
        "task_rows": "".join(rows),
        "task_scope": f"the full test split ({a['checkpoints']:,} checkpoints)"
        if full
        else f"a {a['checkpoints']:,}-checkpoint test subset",
    }


def _gains(path: Path) -> dict[int, tuple[float, float, float]]:
    """Per checkpoint: (gain, interval low, interval high) from an `agent-ecg compare` output."""
    out = {}
    for line in path.read_text().splitlines():
        hit = re.match(r"\s*k=(\d+): ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if hit:
            out[int(hit[1])] = (float(hit[2]), float(hit[3]), float(hit[4]))
    if set(out) != set(charts.STEPS):
        raise SystemExit(f"could not parse {path}")
    return out


def same_issue() -> dict:
    """Pairs of runs on the same issue (`agent-ecg compare --same-issue`, results/same_issue/): only what a model
    reads from the run counts there, not how hard the issue is."""
    folder = RESULTS / "same_issue"
    a = _gains(folder / "A_vs_issue_signals_thoughts.test.txt")
    sig = _gains(folder / "signals_vs_chance.test.txt")
    early = [k for k in charts.STEPS if k <= 20]
    if not all(a[k][1] <= 0 <= a[k][2] for k in early):
        raise SystemExit(
            "the site says ecg-full is level with its baseline on same-issue pairs; the intervals disagree"
        )
    signed = lambda x: f"{x:+.3f}".replace("-", "&minus;")  # noqa: E731
    history = json.loads((RESULTS / "opentslm" / "full-multi" / "history.json").read_text())
    return {
        "si_lo": signed(min(a[k][0] for k in early)),
        "si_hi": signed(max(a[k][0] for k in early)),
        "si_sig_first": f"{(0.5 + sig[charts.STEPS[0]][0]) * 100:.0f}&nbsp;%",
        "si_sig_last": f"{(0.5 + sig[charts.STEPS[-1]][0]) * 100:.0f}&nbsp;%",
        "a_val_evals": len(history),
        "a_val_best": f"{max(h['val_auroc'] for h in history):.2f}",
    }


def second_round() -> dict:
    """The second-round runs (ecg v2, ecg-full v2): their gain over the original model and the pre-agreed verdict."""
    names = charts.tslm_labels()
    op = RESULTS / "opentslm"
    runs = [  # (name, folder, original's folder, its baseline, steps the rule counts, steps that must not be worse)
        (f"{names['C']} v2", "ecg-counts", "ecg-only", "signals", (5, 10, 15), ()),
        (f"{names['A']} v2", "full-multi-a2", "full-multi", "issue_signals_thoughts", (15, 20, 30), (5, 10)),
    ]
    rows, verdicts, gains, base = [], {}, {}, {}
    for name, folder, original, baseline, counted, not_worse in runs:
        g = _gains(op / folder / f"compare_vs_{original}_predictions.test.txt")
        base[name] = _gains(op / folder / f"compare_vs_baselines_{baseline}.test.txt")
        passed = sum(g[k][1] > 0 for k in counted) >= 2 and all(g[k][2] > 0 for k in not_worse)
        verdicts[name], gains[name] = passed, g
        cells = "".join(
            f'<td class="{"best" if lo > 0 or hi < 0 else ""}">{d:+.3f}<br><span class="ci">[{lo:+.3f}, {hi:+.3f}]</span></td>'
            for d, lo, hi in (g[k] for k in charts.STEPS)
        )
        rows.append(
            f'<tr class="tslm"><th>{name}</th>{cells}<td class="reads">{"better" if passed else "tried, no gain"}</td></tr>'
        )
    c2, a2 = (r[0] for r in runs)
    first = _gains(op / "full-multi" / "compare_vs_issue_signals_thoughts.test.txt")
    sig = lambda g: [k for k in charts.STEPS if g[k][1] > 0 or g[k][2] < 0]  # noqa: E731
    if not any(verdicts.values()):
        verdict = f"Neither run passed its rule, so {names['C']} and {names['A']} stay the models reported above."
    else:
        won = [n for n, v in verdicts.items() if v]
        verdict = f"{' and '.join(won)} passed {'its rule' if len(won) == 1 else 'their rules'}."
    gap = (
        f"Its gap to the issue + signals + thoughts baseline is no longer significant at any checkpoint "
        f"({names['A']}'s was, at {', '.join(map(str, sig(first)))} steps)."
        if not sig(base[a2]) and sig(first)
        else f"Its gap to the issue + signals + thoughts baseline stays significant at {', '.join(map(str, sig(base[a2]))) or 'no'} steps."
    )
    summary = (
        f"{verdict} {c2} changed little: giving the model the counts did not help it use them. "
        f"{a2} moved as expected where its fix acts ({gains[a2][20][0]:+.3f} at 20 steps, {gains[a2][30][0]:+.3f} at 30) "
        f"and its validation score held near its peak instead of declining, but the test set is too small to separate it "
        f"from {names['A']}. {gap}"
    )
    text = (RESULTS / "SUMMARY.md").read_text()
    hit = re.search(r"Second round on (\w+): ([\d.]+) h \(ecg v2\) and ([\d.]+) h \(ecg-full v2\)", text)
    if not hit:
        raise SystemExit("results/SUMMARY.md lacks the second round's compute line; update second_round()")
    return {
        "r2_head": "".join(f"<th>{k} steps</th>" for k in charts.STEPS),
        "r2_rows": "".join(rows),
        "r2_summary": summary,
        "label_c2": c2,
        "label_a2": a2,
        "r2_gpu": hit[1],
        "h_c2": hit[2],
        "h_a2": hit[3],
    }


def main() -> None:
    missing = [n for n in FIGS if not (SITE / "figures" / f"{n}.svg").exists()]
    if missing:
        raise SystemExit("figures missing; run: uv run python scripts/charts.py --theme both")
    if not all((SITE / "figures" / f"{n}.svg").exists() for n in PAPER_FIGS):
        raise SystemExit("figures missing; run: uv run python scripts/paper_figures.py")
    values = {
        **data_numbers(),
        **auroc_table(),
        **monitor(),
        **showcase(),
        **compute(),
        **other_tasks(),
        **same_issue(),
        **second_round(),
    }
    values["fig_hero"] = figure(
        "hero_ecg",
        "Health of one agent run over 30 steps, with the 19 input series below",
        f"One example, chosen to illustrate: an unseen test run on <code>{values['show_issue']}</code>. {charts.MONITOR_MODEL}'s health falls through watch and low "
        f"while the agent keeps re-editing the same file after the linter rejected its edits, and the alarm fires at "
        f"step {values['show_alarm']}. The run went on to fail (it {values['show_end']}). The signals baseline "
        f"(grey, dashed) {values['show_signals']} on this run. Below it are the 19 series the model reads, one row each.",
    )
    values["fig_auroc"] = figure(
        "auroc_by_step",
        "Test AUROC at 5 to 30 steps for three TSLMs and their baselines",
        "Each panel pairs a TSLM (blue) with the baseline that reads the same inputs (grey). The dashed line reads "
        "the issue text only; the shading is a 95&nbsp;% interval from resampling test repositories.",
    )
    values["fig_monitor"] = figure(
        "monitor_numbers",
        "Stat cards and a Venn diagram comparing the two live monitors",
        f"The Venn counts are the failing runs each monitor flags (of {values['mon_failing']}); the circles are not to scale.",
    )
    values["fig_runs"] = figure(
        "failing_vs_resolved",
        "Health over time for a failing and a resolved run, both monitors",
        f"The showcase failing run and a resolved run. {charts.MONITOR_MODEL} stays stable on the resolved run until a dip to watch "
        "at the end; the signals baseline reads it as low early on.",
    )
    values["fig_training"] = figure(
        "training_curves",
        "Validation AUROC over training steps for the three TSLMs",
        "Validation AUROC during training; the ringed point is the checkpoint we kept. Validation runs are not the test set.",
    )
    values["fig_same_issue"] = figure(
        "same_issue",
        "Two panels: the gap between ecg-full and its baseline over all runs and on same-issue pairs, and the share of same-issue pairs each model ranks right",
        f"(a) {values['label_a']} minus the issue + signals + thoughts baseline at each checkpoint: pink over all test "
        "runs (AUROC), blue on pairs of a resolved and an unresolved run on the same issue. Bars: 95&nbsp;% intervals "
        "from resampling test repositories. (b) The share of such pairs where a model scores the resolved run higher. "
        "The issue text alone scores 0.50 by construction; the band is the ECG baseline's 95&nbsp;% interval. "
        "Sources: <code>results/same_issue/</code>.",
    )
    values["fig_paper"] = figure(
        "paper_results",
        "Four panels: test AUROC by steps seen, the same-issue gap, the same-issue pair share, and the live monitor's rates",
        "(a) Test AUROC by steps seen, with 95&nbsp;% intervals for the two models with the same inputs. (b) and (c) as in "
        "the Results section. (d) The live monitor on the replayed test runs. Style after figures4papers "
        '(Chen Liu). Download: <a href="figures/paper_results.pdf">PDF</a>.',
    )
    values["site_url"] = SITE_URL
    html = Template((Path(__file__).parent / "site_template.html").read_text()).substitute(values)
    # GitHub Pages serves site/index.html as is, so it gets the full document around the template;
    # the claude.ai preview adds that wrapper itself, so it gets the bare template (build/, git-ignored).
    head, sep, body = html.partition("</style>\n")
    if not sep:
        raise SystemExit("site_template.html must end its head part with </style>")
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head}{sep}</head>\n<body>\n{body}</body>\n</html>\n"
    )
    (SITE / "index.html").write_text(page)
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "site-fragment.html").write_text(html)
    print(f"wrote {(SITE / 'index.html').relative_to(ROOT)} and {len(FIGS)} figures")


if __name__ == "__main__":
    main()
