"""Build the project website (site/index.html) from the result files and the charts.

    uv run python scripts/charts.py --theme both   # first: the figures
    uv run python scripts/site.py                   # then: site/index.html + site/figures/

Every number on the page is read from results/ or data/checkpoints.parquet; rerun after they change.
"""

from __future__ import annotations

import json
import re
import shutil
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
    cfg = json.loads((charts.tslm_runs()["A"] / "config.json").read_text())
    return {"h_c": c, "h_b": b, "h_a": a, "gpu_hours": total, "gpu": gpu, "base": cfg["init_from"], "llm": cfg["llm"]}


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


def main() -> None:
    missing = [n for n in FIGS if not (ROOT / "figures" / "site" / f"{n}.svg").exists()]
    if missing:
        raise SystemExit("figures missing; run: uv run python scripts/charts.py --theme both")
    values = {**data_numbers(), **auroc_table(), **monitor(), **showcase(), **compute(), **other_tasks()}
    values["fig_hero"] = figure(
        "hero_ecg",
        "Health of one agent run over 30 steps, with the 19 input series below",
        f"An unseen test run on <code>{values['show_issue']}</code>. {charts.MONITOR_MODEL}'s health falls through watch and low "
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
    values["site_url"] = SITE_URL
    html = Template((Path(__file__).parent / "site_template.html").read_text()).substitute(values)
    shutil.rmtree(SITE / "figures", ignore_errors=True)
    (SITE / "figures").mkdir(parents=True)
    for n in FIGS:
        svg = (ROOT / "figures" / "site" / f"{n}.svg").read_text()
        # web hosts refuse SVGs with a DOCTYPE; matplotlib writes one we do not need
        (SITE / "figures" / f"{n}.svg").write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg))
    preview = ROOT / "figures" / "social_preview.png"
    if preview.exists():
        shutil.copy(preview, SITE / "social_preview.png")
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
