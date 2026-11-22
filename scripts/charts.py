"""Presentation charts for the video and the website, read from the result files.

    uv run python scripts/charts.py                 # dark theme  -> figures/*.svg, figures/*.png
    uv run python scripts/charts.py --theme light   # light theme -> figures/light/
    uv run python scripts/charts.py --theme both

Every number is read from results/ (and the showcase run's channels from data/checkpoints.parquet); nothing
is typed in. Rerun after the result files change. See figures/README.md for what each chart says.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch

from agent_ecg import opentslm_data
from agent_ecg.monitor import CRITICAL, HOLD, STABLE, WATCH

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCAN = RESULTS / "opentslm" / "scan"
CHECKPOINTS = ROOT / "data" / "checkpoints.parquet"
SHOWCASE = "train-00010-of-00012:721"
RESOLVED = "train-00000-of-00012:1738"
STEPS = (5, 10, 15, 20, 30)

W, H, DPI = 16, 9, 120  # 1920 x 1080
# Website versions (figures/site/): no built-in title, subtitle or footer (the page has its own headings and
# captions), cropped to the content, and a taller hero so its 19 row labels can be bigger.
BARE = False
HERO_H = 11


# ---------------------------------------------------------------- theme


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    ink: str
    ink2: str
    muted: str
    grid: str
    stable: str
    watch: str
    low: str
    critical: str
    tslm: str
    cpu: str
    glow: bool

    def state(self, s: str) -> str:
        return {"stable": self.stable, "watch": self.watch, "low": self.low, "critical": self.critical}[s]


DARK = Theme(
    "dark",
    bg="#07090a",
    panel="#11161a",
    ink="#eef2f0",
    ink2="#b8c2be",
    muted="#7c8884",
    grid="#1c2427",
    stable="#3ddc84",
    watch="#f5c542",
    low="#ff8c2a",
    critical="#ff4d4d",
    tslm="#4cc9f0",
    cpu="#9aa3a0",
    glow=True,
)
LIGHT = Theme(
    "light",
    bg="#ffffff",
    panel="#f3f5f4",
    ink="#111816",
    ink2="#3d4744",
    muted="#6b7571",
    grid="#e3e7e5",
    stable="#16924c",
    watch="#b98900",
    low="#d9620a",
    critical="#d32323",
    tslm="#0a74b8",
    cpu="#6b7280",
    glow=False,
)

FONT = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
SIZE = {"title": 34, "subtitle": 20, "label": 19, "tick": 17, "small": 15, "note": 14, "big": 64}


def setup(t: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "svg.fonttype": "path",  # text as outlines: the SVG looks the same without the font installed
            "svg.hashsalt": "agent-ecg",  # stable element ids across reruns
            "figure.facecolor": t.bg,
            "axes.facecolor": t.bg,
            "savefig.facecolor": t.bg,
            "text.color": t.ink,
            "axes.labelcolor": t.ink2,
            "axes.edgecolor": t.grid,
            "xtick.color": t.muted,
            "ytick.color": t.muted,
            "xtick.labelcolor": t.ink2,
            "ytick.labelcolor": t.ink2,
            "xtick.labelsize": SIZE["tick"],
            "ytick.labelsize": SIZE["tick"],
            "axes.labelsize": SIZE["label"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def figure(t: Theme, title: str, subtitle: str = "", height: float = H) -> plt.Figure:
    fig = plt.figure(figsize=(W, height), dpi=DPI)
    if BARE:
        return fig
    fig.text(0.05, 0.935, title, fontsize=SIZE["title"], fontweight="bold", color=t.ink, va="baseline")
    if subtitle:
        fig.text(0.05, 0.89, subtitle, fontsize=SIZE["subtitle"], color=t.ink2, va="baseline")
    return fig


def footer(fig: plt.Figure, t: Theme, text: str) -> None:
    if BARE:
        return
    fig.text(0.05, 0.022, text, fontsize=SIZE["note"], color=t.muted, va="bottom")


def style_axis(ax: plt.Axes, t: Theme) -> None:
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.grid)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(length=0, pad=8)


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if BARE:
        fig.savefig(out / f"{name}.svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  {out.relative_to(ROOT) / name}.svg")
        return
    fig.savefig(out / f"{name}.png", dpi=DPI)
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})  # no timestamp: reruns give the same file
    plt.close(fig)
    print(f"  {out.relative_to(ROOT) / name}.{{png,svg}}")


# ---------------------------------------------------------------- data


def replay(path: Path, run_id: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame([r for r in rows if r["run_id"] == run_id]).sort_values("step")
    if df.empty:
        raise SystemExit(f"{run_id} not in {path}")
    return df.reset_index(drop=True)


def alarm_step(df: pd.DataFrame) -> int | None:
    hit = df.loc[df["state"] == "critical", "step"]
    return int(hit.iloc[0]) if len(hit) else None


ORDER = ("stable", "watch", "low", "critical")


def lowest_state(df: pd.DataFrame) -> str:
    return max(df["state"], key=ORDER.index)


def run_row(run_id: str) -> pd.Series:
    t = pd.read_parquet(CHECKPOINTS, filters=[("run_id", "==", run_id)])
    return t.sort_values("k").iloc[-1]  # the longest checkpoint holds every visible step


SHORT = {  # chart names for the 19 series, in opentslm_data.time_series order
    **{
        k: v
        for k, v in zip(
            opentslm_data.LENGTHS,
            ("thought length", "command length", "reply length", "conversation length"),
            strict=True,
        )
    },
    "repeated": "repeated command",
    **{
        f"action:{k}": v
        for k, v in zip(opentslm_data.ACTIONS, ("edit", "create file", "run code", "read", "search"), strict=True)
    },
    **{
        f"reply:{k}": v
        for k, v in zip(
            opentslm_data.REPLIES,
            ("rejected edit", "traceback", "command error", "timeout", "silent success"),
            strict=True,
        )
    },
    "same_file": "same file again",
    **{
        f"target:{k}": v
        for k, v in zip(opentslm_data.EDIT_TARGETS, ("edits source", "edits test", "edits own file"), strict=True)
    },
}


def leads(row: pd.Series) -> list[tuple[str, bool, np.ndarray]]:
    """The 19 series the TSLM reads, as (name, is_length, values)."""
    _, values = opentslm_data.time_series(row)
    keys = (
        list(opentslm_data.LENGTHS)
        + ["repeated"]
        + [f"action:{k}" for k in opentslm_data.ACTIONS]
        + [f"reply:{k}" for k in opentslm_data.REPLIES]
        + ["same_file"]
        + [f"target:{k}" for k in opentslm_data.EDIT_TARGETS]
    )
    if len(keys) != len(values):
        raise SystemExit(f"expected {len(keys)} series, opentslm_data gives {len(values)}")
    n_len = len(opentslm_data.LENGTHS)
    return [(SHORT[k], i < n_len, np.asarray(v, float)) for i, (k, v) in enumerate(zip(keys, values, strict=True))]


def model_auroc(path: Path) -> dict[int, tuple[float, float, float]]:
    out = json.loads(path.read_text())["outcome"]
    return {k: (out[f"k={k}"]["auroc"], *out[f"k={k}"]["auroc_95"]) for k in STEPS if f"k={k}" in out}


RUNS_FILE = ROOT / "figures" / "runs.json"
# The live-monitor scan (results/opentslm/scan) and ecg-full's full-test report were made with ecg-full (run folder full-multi),
# whatever figures/runs.json says, so the monitor charts name that model directly.
MONITOR_MODEL = "ecg-full"
SIGNALS_BASELINE = "signals baseline"


def _runs() -> dict[str, dict]:
    runs = {k: v for k, v in json.loads(RUNS_FILE.read_text()).items() if not k.startswith("_")}
    if set(runs) != {"A", "B", "C"}:
        raise SystemExit(f"{RUNS_FILE.relative_to(ROOT)} must name runs for A, B and C")
    for slot, r in runs.items():
        if not (RESULTS / "opentslm" / r["dir"] / "report.test.json").exists():
            raise SystemExit(f"TSLM {slot}: results/opentslm/{r['dir']}/report.test.json not found")
    return runs


def tslm_runs() -> dict[str, Path]:
    """TSLM slot (A: issue + thoughts + ECG, B: thoughts + ECG, C: ECG only) -> run directory, from figures/runs.json."""
    return {slot: RESULTS / "opentslm" / r["dir"] for slot, r in _runs().items()}


def tslm_labels() -> dict[str, str]:
    """TSLM slot -> the name shown on charts and the site (A, or A2 for a second-round run)."""
    return {slot: r["label"] for slot, r in _runs().items()}


def scan_summary() -> dict:
    text = (SCAN / "summary.md").read_text()
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and re.match(r"\d+ / \d+", cells[1]):
            key = "A" if cells[0].startswith("ecg-full") else "signals" if cells[0].startswith("signals") else None
            if key is None:
                continue
            failing, resolved = (int(x) for x in cells[1].split("/"))
            rows[key] = {
                "failing": failing,
                "resolved": resolved,
                "flagged": float(cells[2].rstrip(" %")),
                "median_step": int(cells[3]),
                "false_alarms": float(cells[4].rstrip(" %")),
                "right": float(cells[5].rstrip(" %")),
            }
    natural = re.search(r"natural ([\d.]+) % resolved", text)
    venn = re.search(r"(\d+) are flagged by both, (\d+) only by ecg-full and (\d+) only by\s+the signals monitor", text)
    if set(rows) != {"A", "signals"} or not natural or not venn:
        raise SystemExit("could not parse results/opentslm/scan/summary.md; update scan_summary()")
    both, a_only, s_only = (int(x) for x in venn.groups())
    return {**rows, "natural": float(natural.group(1)), "both": both, "a_only": a_only, "s_only": s_only}


# ---------------------------------------------------------------- drawing helpers


def bands(ax: plt.Axes, t: Theme, label: bool = True, x: float | None = None) -> None:
    """Faint state bands with their thresholds (stable >= 60, watch >= 30, low below, critical zone <= 15)."""
    alpha = 0.07 if t.glow else 0.08
    for lo, hi, s in (
        (STABLE, 100, "stable"),
        (WATCH, STABLE, "watch"),
        (CRITICAL, WATCH, "low"),
        (0, CRITICAL, "critical"),
    ):
        ax.axhspan(lo, hi, color=t.state(s), alpha=alpha, lw=0, zorder=0)
        if label:
            ax.text(
                1.008,
                (lo + hi) / 2,
                s,
                transform=ax.get_yaxis_transform(),
                color=t.state(s),
                fontsize=SIZE["small"],
                fontweight="bold",
                va="center",
                ha="left",
            )
    for y in (STABLE, WATCH, CRITICAL):
        ax.axhline(y, color=t.grid, lw=1, zorder=0.5)


def health_line(ax: plt.Axes, t: Theme, df: pd.DataFrame, lw: float = 3.2) -> None:
    """Health as a heart-monitor trace, each segment coloured by the state it leads into."""
    x, y = df["step"].to_numpy(float), df["health"].to_numpy(float)
    segs = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    colors = [t.state(s) for s in df["state"].iloc[1:]]
    if t.glow:
        ax.add_collection(
            LineCollection(segs, colors=colors, linewidths=lw * 4, alpha=0.12, capstyle="round", zorder=3)
        )
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round", zorder=4))
    ax.scatter(x, y, s=22, c=[t.state(s) for s in df["state"]], zorder=5, linewidths=0)


def cpu_line(ax: plt.Axes, t: Theme, df: pd.DataFrame) -> None:
    ax.plot(df["step"], df["health"], color=t.cpu, lw=1.8, ls=(0, (5, 3)), zorder=2.5, alpha=0.95)


def health_axis(ax: plt.Axes, t: Theme, steps: tuple[int, int]) -> None:
    style_axis(ax, t)
    ax.set_ylim(0, 104)
    ax.set_xlim(steps[0] - 0.5, steps[1] + 0.5)
    ax.set_yticks([0, CRITICAL, WATCH, STABLE, 100])
    ax.set_ylabel("health", labelpad=10)


def alarm(ax: plt.Axes, t: Theme, df: pd.DataFrame, text: str) -> int | None:
    step = alarm_step(df)
    if step is None:
        return None
    h = float(df.loc[df["step"] == step, "health"].iloc[0])
    ax.scatter([step], [h], s=420, facecolors="none", edgecolors=t.critical, linewidths=3, zorder=6)
    ax.annotate(
        text,
        (step, h),
        xytext=(step + 1.6, h + 30),
        color=t.critical,
        fontsize=SIZE["label"],
        fontweight="bold",
        ha="left",
        va="center",
        zorder=7,
        arrowprops={"arrowstyle": "-", "color": t.critical, "lw": 2, "shrinkA": 4, "shrinkB": 14},
    )
    return step


def legend_line(fig: plt.Figure, t: Theme, x: float, y: float, items: list[tuple[str, dict]]) -> None:
    """A one-row legend in figure coordinates: a short line sample and its label for each item."""
    ax = fig.add_axes([x, y - 0.015, 0.9 - x, 0.03])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pos = 0.0
    for label, kw in items:
        if kw.get("segments"):
            n = len(kw["segments"])
            for i, c in enumerate(kw["segments"]):
                ax.plot(
                    [pos + i * 0.04 / n, pos + (i + 1) * 0.04 / n], [0.5, 0.5], color=c, lw=4, solid_capstyle="butt"
                )
        else:
            ax.plot([pos, pos + 0.04], [0.5, 0.5], **{k: v for k, v in kw.items() if k != "segments"})
        txt = ax.text(pos + 0.05, 0.5, label, va="center", fontsize=SIZE["small"], color=t.ink2)
        box = txt.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        pos = box.x1 + 0.035


# ---------------------------------------------------------------- charts


def hero_ecg(t: Theme, out: Path) -> None:
    a = replay(SCAN / "replay.showcase.A.jsonl", SHOWCASE)
    s = replay(SCAN / "replay.showcase.signals.jsonl", SHOWCASE)
    row = run_row(SHOWCASE)
    n = len(row["action_kind"])
    outcome = "resolved" if row["resolved"] else "failed"
    fig = figure(
        t,
        "Agent ECG: a coding agent's run, read step by step",
        f"{row['instance_id']} · {n} steps · the run {outcome}. Top: monitor health. Bottom: the 19 series the model reads.",
        height=HERO_H if BARE else H,
    )
    left, right = 0.17, 0.92
    ax = fig.add_axes([left, 0.565, right - left, 0.265])
    bands(ax, t)
    cpu_line(ax, t, s)
    health_line(ax, t, a)
    health_axis(ax, t, (1, n))
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    step = alarm(ax, t, a, "")
    if step is not None:
        ax.texts[-1].set_text(f"step {step}: stop the agent")
    ax.text(
        a["step"].iloc[0] - 0.6,
        45,
        f"monitor\nstarts at\nstep {a['step'].iloc[0]}",
        ha="right",
        va="center",
        fontsize=SIZE["note"],
        color=t.muted,
        linespacing=1.2,
    )
    legend_line(
        fig,
        t,
        left,
        0.855,
        [
            (f"{MONITOR_MODEL} (issue + thoughts + ECG)", {"segments": [t.stable, t.watch, t.low, t.critical]}),
            (
                SIGNALS_BASELINE + ("" if alarm_step(s) else ": no alarm"),
                {"color": t.cpu, "lw": 2, "ls": (0, (5, 3))},
            ),
        ],
    )

    # the leads
    series = leads(row)
    lx = fig.add_axes([left, 0.135, right - left, 0.41])
    style_axis(lx, t)
    rows = len(series)
    lx.set_ylim(rows - 0.4, -0.6)
    lx.set_xlim(0.5, n + 0.5)
    for i, (name, is_len, v) in enumerate(series):
        if i % 2 == 0:
            lx.axhspan(i - 0.5, i + 0.5, color=t.panel, lw=0, zorder=0)
        steps = np.arange(1, len(v) + 1)
        if is_len:
            span = np.ptp(v) or 1.0
            yv = i + 0.38 - 0.76 * (v - v.min()) / span
            lx.plot(steps, yv, color=t.ink2, lw=1.6, zorder=2)
        else:
            on = steps[v > 0]
            lx.vlines(on, i - 0.36, i + 0.36, color=t.ink, lw=4.5, zorder=2)
        lx.text(
            -0.005,
            i,
            name,
            transform=lx.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=SIZE["small"] if BARE else SIZE["note"] - 1,
            color=t.ink2,
        )
    for b in (3.5, 4.5, 9.5, 14.5, 15.5):  # lengths | repeat | actions | replies | same file | targets
        lx.axhline(b, color=t.grid, lw=1, zorder=1)
    if step is not None:
        lx.axvline(step, color=t.critical, lw=1.5, alpha=0.7, zorder=1)
        ax.axvline(step, color=t.critical, lw=1.5, alpha=0.4, zorder=1)
    lx.set_yticks([])
    lx.set_xticks([1, 5, 10, 15, 20, 25, n] if n not in (25, 26) else [1, 5, 10, 15, 20, n])
    lx.text(
        1.0, -0.12, "agent step", transform=lx.transAxes, ha="right", va="top", fontsize=SIZE["small"], color=t.ink2
    )
    lx.spines["left"].set_visible(False)
    footer(
        fig,
        t,
        "Health: monitor replay on an unseen test run (results/opentslm/scan). Series: data/checkpoints.parquet, as the TSLM reads them.",
    )
    save(fig, out, "hero_ecg")


def failing_vs_resolved(t: Theme, out: Path) -> None:
    runs = [(SHOWCASE, "failing run"), (RESOLVED, "resolved run")]
    fig = figure(
        t,
        "A failing run and a resolved run, through both monitors",
        "Health 0-100 at each step; the same axes in both panels.",
    )
    left, right = 0.1, 0.9
    tops = [0.475, 0.115]
    last = 0
    data = []
    for run_id, kind in runs:
        a = replay(SCAN / "replay.showcase.A.jsonl", run_id)
        s = replay(SCAN / "replay.showcase.signals.jsonl", run_id)
        row = run_row(run_id)
        if bool(row["resolved"]) != (kind == "resolved run"):
            raise SystemExit(f"{run_id} is not a {kind}")
        data.append((run_id, kind, a, s, row))
        last = max(last, int(a["step"].max()))
    for (_, kind, a, s, row), y0 in zip(data, tops, strict=True):
        ax = fig.add_axes([left, y0, right - left, 0.27])
        bands(ax, t)
        cpu_line(ax, t, s)
        health_line(ax, t, a)
        health_axis(ax, t, (5, last))
        step = alarm(ax, t, a, "")
        if step is not None:
            ax.texts[-1].set_text(f"step {step}: stop the agent")
        sa, ss = alarm_step(a), alarm_step(s)
        say = lambda name, st, df: f"{name} alarms at step {st}" if st else f"{name} lowest: {lowest_state(df)}"  # noqa: E731
        ax.text(
            0,
            1.06,
            f"{kind.capitalize()}  ·  {row['instance_id']}",
            transform=ax.transAxes,
            fontsize=SIZE["label"],
            fontweight="bold",
            color=t.ink,
            va="bottom",
        )
        ax.text(
            1,
            1.06,
            f"{say(MONITOR_MODEL, sa, a)}  ·  {say(SIGNALS_BASELINE, ss, s)}",
            transform=ax.transAxes,
            fontsize=SIZE["small"],
            color=t.ink2,
            va="bottom",
            ha="right",
        )
        if y0 == tops[0]:
            ax.set_xticklabels([])
        else:
            ax.text(
                1.0,
                -0.2,
                "agent step",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=SIZE["small"],
                color=t.ink2,
            )
    legend_line(
        fig,
        t,
        left,
        0.84,
        [
            (f"{MONITOR_MODEL}, coloured by state", {"segments": [t.stable, t.watch, t.low, t.critical]}),
            (SIGNALS_BASELINE, {"color": t.cpu, "lw": 2, "ls": (0, (5, 3))}),
        ],
    )
    footer(
        fig,
        t,
        "Source: results/opentslm/scan/replay.showcase.{A,signals}.jsonl (unseen test runs, replayed step by step).",
    )
    save(fig, out, "failing_vs_resolved")


def auroc_by_step(t: Theme, out: Path) -> None:
    tslm = {k: model_auroc(p / "report.test.json") for k, p in tslm_runs().items()}
    names = tslm_labels()
    cpu = {
        n: model_auroc(RESULTS / f"{n}.test.json")
        for n in ("signals", "signals_thoughts", "issue_signals_thoughts", "issue")
    }
    pairs = [
        ("C", "ECG only", "signals", "signals baseline"),
        ("B", "thoughts + ECG", "signals_thoughts", "signals + thoughts baseline"),
        ("A", "issue + thoughts + ECG", "issue_signals_thoughts", "issue + signals + thoughts baseline"),
    ]
    early = [k for k in STEPS if k <= 20]
    gaps = [cpu["issue_signals_thoughts"][k][0] - tslm["A"][k][0] for k in early]
    all_below = all(tslm[m][k][0] < cpu[c][k][0] for m, _, c, _ in pairs for k in STEPS)
    sub = (
        "Each TSLM stays below the baseline with the same inputs; " if all_below else ""
    ) + f"{names['A']} comes within {min(gaps):.2f}-{max(gaps):.2f} ({early[0]}-{early[-1]} steps)."
    fig = figure(t, "Predicting pass / fail from the first steps (test AUROC)", sub)
    lo = min(min(v[0] for v in m.values()) for m in [*tslm.values(), *cpu.values()])
    hi = max(max(v[0] for v in m.values()) for m in [*tslm.values(), *cpu.values()])
    ylim = (min(0.47, lo - 0.03), max(0.76, hi + 0.03))
    pw, gap, left = 0.22, 0.095, 0.075
    for i, (m, reads, c, cname) in enumerate(pairs):
        ax = fig.add_axes([left + i * (pw + gap), 0.16, pw, 0.495])
        style_axis(ax, t)
        ax.axhline(0.5, color=t.muted, lw=1.2, ls=(0, (1, 3)))
        ax.text(STEPS[-1], 0.5 + 0.004, "chance", color=t.muted, fontsize=SIZE["note"], ha="right", va="bottom")
        iss = [cpu["issue"][k][0] for k in STEPS]
        ax.plot(STEPS, iss, color=t.muted, lw=1.6, ls=(0, (6, 4)))
        for model, color, lw, ms in ((cpu[c], t.cpu, 2.6, 8), (tslm[m], t.tslm, 3.4, 9)):
            y = [model[k][0] for k in STEPS]
            ax.fill_between(
                STEPS, [model[k][1] for k in STEPS], [model[k][2] for k in STEPS], color=color, alpha=0.08, lw=0
            )
            ax.plot(STEPS, y, color=color, lw=lw, marker="o", ms=ms, mec=t.bg, mew=2, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_xlim(3, 32)
        ax.set_xticks(STEPS)
        ax.set_xlabel("steps seen", labelpad=6)
        if i == 0:
            ax.set_ylabel("AUROC", labelpad=10)
        else:
            ax.set_yticklabels([])
        ax.yaxis.grid(True, color=t.grid, lw=1)
        ax.set_axisbelow(True)
        ax.text(
            0,
            1.205,
            f"{names[m]}",
            transform=ax.transAxes,
            fontsize=SIZE["label"],
            fontweight="bold",
            va="bottom",
        )
        ax.text(
            0,
            1.03,
            f"reads {reads}\nvs {cname}",
            transform=ax.transAxes,
            fontsize=SIZE["small"],
            color=t.ink2,
            va="bottom",
            linespacing=1.3,
        )
        # direct labels at the right end, nudged apart
        ends = {
            names[m]: (tslm[m][STEPS[-1]][0], t.tslm),
            "baseline": (cpu[c][STEPS[-1]][0], t.cpu),
            "issue only": (iss[-1], t.muted),
        }
        placed: list[float] = []
        for label, (yv, col) in sorted(ends.items(), key=lambda kv: -kv[1][0]):
            yy = yv
            for p in placed:
                if abs(yy - p) < 0.02:
                    yy = p - 0.02
            placed.append(yy)
            ax.text(
                30.9,
                yy,
                label,
                color=col,
                fontsize=SIZE["small"],
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=False,
            )
    legend_line(
        fig,
        t,
        left,
        0.835,
        [
            ("TSLM (OpenTSLM, Llama 3.2 1B)", {"color": t.tslm, "lw": 3.4, "marker": "o", "ms": 7}),
            ("Baseline, same inputs", {"color": t.cpu, "lw": 2.6, "marker": "o", "ms": 7}),
            ("Issue-only baseline", {"color": t.muted, "lw": 1.6, "ls": (0, (6, 4))}),
        ],
    )
    test = pd.read_parquet(CHECKPOINTS, columns=["run_id", "repository", "split"]).query("split == 'test'")
    footer(
        fig,
        t,
        f"{test['run_id'].nunique():,} unseen test runs from {test['repository'].nunique()} repositories; shaded: 95 % interval. Sources: results/*.test.json, results/opentslm/*/report.test.json.",
    )
    save(fig, out, "auroc_by_step")


def monitor_numbers(t: Theme, out: Path) -> None:
    m = scan_summary()
    a, s = m["A"], m["signals"]
    fig = figure(
        t,
        f"Two live monitors, {a['failing'] + a['resolved']} unseen runs replayed step by step",
        f"{a['failing']} failing and {a['resolved']} resolved test runs; the alarm fires after {HOLD} critical steps in a row.",
    )
    cards = [
        (
            "alarms that were right",
            f"{a['right']:.0f} %",
            f"{s['right']:.0f} %",
            f"at the test share of resolved runs ({m['natural']:.1f} %)",
        ),
        (
            "failing runs flagged",
            f"{a['flagged']:.0f} %",
            f"{s['flagged']:.0f} %",
            f"false alarms: {a['false_alarms']:.0f} % / {s['false_alarms']:.0f} % of resolved",
        ),
        (
            "median alarm step",
            f"{a['median_step']}",
            f"{s['median_step']}",
            "head start partly from the issue",
        ),
    ]
    x0, cw, ch, gap = 0.05, 0.44, 0.205, 0.035
    for i, (label, va, vs, note) in enumerate(cards):
        y = 0.62 - i * (ch + gap)
        fig.patches.append(
            FancyBboxPatch(
                (x0, y),
                cw,
                ch,
                boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure,
                facecolor=t.panel,
                edgecolor=t.grid,
                lw=1.2,
            )
        )
        fig.text(x0 + 0.02, y + ch - 0.04, label, fontsize=SIZE["label"], fontweight="bold", color=t.ink, va="center")
        fig.text(x0 + 0.02, y + 0.075, va, fontsize=SIZE["big"] * 0.85, fontweight="bold", color=t.tslm, va="center")
        fig.text(x0 + 0.02, y + 0.025, MONITOR_MODEL, fontsize=SIZE["small"], color=t.ink2, va="center")
        fig.text(x0 + 0.22, y + 0.075, vs, fontsize=SIZE["big"] * 0.85, fontweight="bold", color=t.cpu, va="center")
        fig.text(x0 + 0.22, y + 0.025, SIGNALS_BASELINE, fontsize=SIZE["small"], color=t.ink2, va="center")
        fig.text(x0 + cw - 0.02, y + ch - 0.04, note, fontsize=SIZE["note"], color=t.muted, va="center", ha="right")
    # Venn
    ax = fig.add_axes([0.53, 0.1, 0.43, 0.74])
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.75, 1.9)
    r, d = 1.15, 0.72
    for cx, col in ((-d, t.tslm), (d, t.cpu)):
        ax.add_patch(Circle((cx, 0), r, facecolor=col, alpha=0.22, edgecolor=col, lw=3))
    ax.text(
        -d - 0.48,
        0.08,
        f"{m['a_only']}",
        fontsize=SIZE["big"] * 0.75,
        fontweight="bold",
        color=t.ink,
        ha="center",
        va="center",
    )
    ax.text(-d - 0.48, -0.35, f"{MONITOR_MODEL} only", fontsize=SIZE["small"], color=t.ink2, ha="center", va="center")
    ax.text(
        0, 0.08, f"{m['both']}", fontsize=SIZE["big"] * 0.75, fontweight="bold", color=t.ink, ha="center", va="center"
    )
    ax.text(0, -0.35, "both", fontsize=SIZE["small"], color=t.ink2, ha="center", va="center")
    ax.text(
        d + 0.48,
        0.08,
        f"{m['s_only']}",
        fontsize=SIZE["big"] * 0.75,
        fontweight="bold",
        color=t.ink,
        ha="center",
        va="center",
    )
    ax.text(d + 0.48, -0.35, "signals only", fontsize=SIZE["small"], color=t.ink2, ha="center", va="center")
    ax.text(
        -d, r + 0.16, MONITOR_MODEL, fontsize=SIZE["label"], fontweight="bold", color=t.tslm, ha="center", va="bottom"
    )
    ax.text(
        d, r + 0.16, SIGNALS_BASELINE, fontsize=SIZE["label"], fontweight="bold", color=t.cpu, ha="center", va="bottom"
    )
    ax.text(
        0,
        1.78,
        f"Failing runs flagged (of {a['failing']})",
        fontsize=SIZE["label"],
        fontweight="bold",
        color=t.ink,
        ha="center",
        va="bottom",
    )
    ax.text(
        0,
        -r - 0.22,
        "They catch mostly different runs.",
        fontsize=SIZE["small"],
        color=t.ink2,
        ha="center",
        va="top",
        linespacing=1.35,
    )
    footer(
        fig,
        t,
        "Source: results/opentslm/scan/summary.md (replays: opentslm/scan_summary.py).",
    )
    save(fig, out, "monitor_numbers")


def training_curves(t: Theme, out: Path) -> None:
    runs = tslm_runs()
    reads = {"A": "issue + thoughts + ECG", "B": "thoughts + ECG", "C": "ECG only"}
    cfg = json.loads((runs["A"] / "config.json").read_text())
    fig = figure(
        t,
        "Training the TSLMs: validation AUROC",
        f"Pass/fail AUROC on {cfg['val_checkpoints']:,} validation checkpoints, measured every {cfg['eval_every']} training steps.",
    )
    ax = fig.add_axes([0.08, 0.15, 0.74, 0.65])
    style_axis(ax, t)
    ax.yaxis.grid(True, color=t.grid, lw=1)
    ax.set_axisbelow(True)
    styles = {
        "A": (t.tslm, "-", 3.4, "o"),
        "B": (t.tslm, (0, (6, 3)), 2.6, "s"),
        "C": (t.tslm, (0, (1.5, 2.5)), 2.6, "D"),
    }
    alphas = {"A": 1.0, "B": 0.75, "C": 0.55}
    ends: dict[str, float] = {}
    for k in ("A", "B", "C"):
        h = pd.DataFrame(json.loads((runs[k] / "history.json").read_text())).dropna(subset=["val_auroc"])
        col, ls, lw, mk = styles[k]
        ax.plot(h["step"], h["val_auroc"], color=col, ls=ls, lw=lw, marker=mk, ms=7, alpha=alphas[k], mec=t.bg, mew=1.5)
        best = h.loc[h["val_auroc"].idxmax()]
        ax.scatter([best["step"]], [best["val_auroc"]], s=260, facecolors="none", edgecolors=t.ink, lw=2, zorder=5)
        ends[k] = float(h.iloc[-1]["val_auroc"])
    span = np.diff(ax.get_ylim())[0]
    placed: list[float] = []
    for k in sorted(ends, key=lambda k: -ends[k]):
        y = ends[k]
        for p in placed:
            if abs(y - p) < 0.13 * span:
                y = p - 0.13 * span
        placed.append(y)
        ax.text(
            1.01,
            y,
            f"{tslm_labels()[k]}\n{reads[k]}",
            transform=ax.get_yaxis_transform(),
            fontsize=SIZE["small"],
            color=t.ink2,
            va="center",
            linespacing=1.2,
        )
    ax.set_xlabel("training step", labelpad=6)
    ax.set_ylabel("validation AUROC", labelpad=10)
    footer(
        fig, t, "Ringed point: the checkpoint kept (best validation AUROC). Source: results/opentslm/*/history.json."
    )
    save(fig, out, "training_curves")


CHARTS = [hero_ecg, failing_vs_resolved, auroc_by_step, monitor_numbers, training_curves]


def social_preview(t: Theme) -> None:
    """The 1200 x 630 card LinkedIn and others show for a link to the site: title and the showcase run's health."""
    a = replay(SCAN / "replay.showcase.A.jsonl", SHOWCASE)
    fig = plt.figure(figsize=(10, 5.25), dpi=DPI)
    fig.text(0.06, 0.86, "Agent ECG", fontsize=40, fontweight="bold", color=t.ink, va="baseline")
    fig.text(
        0.06, 0.765, "Warning early when a coding agent is going to fail", fontsize=19, color=t.ink2, va="baseline"
    )
    ax = fig.add_axes([0.06, 0.1, 0.82, 0.56])
    bands(ax, t)
    health_line(ax, t, a, lw=4)
    health_axis(ax, t, (int(a["step"].min()), int(a["step"].max())))
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    step = alarm(ax, t, a, "")
    if step is not None:
        ax.texts[-1].set_text(f"step {step}: stop the agent")
        ax.texts[-1].set_fontsize(20)
    fig.savefig(ROOT / "figures" / "social_preview.png", dpi=DPI)
    plt.close(fig)
    print("  figures/social_preview.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--theme", choices=["dark", "light", "both"], default="dark")
    args = p.parse_args()
    themes = [DARK, LIGHT] if args.theme == "both" else [DARK if args.theme == "dark" else LIGHT]
    for t in themes:
        setup(t)
        out = ROOT / "figures" if t.name == "dark" else ROOT / "figures" / "light"
        print(f"{t.name}:")
        for chart in CHARTS:
            chart(t, out)
        if t.name == "light":
            global BARE
            BARE = True
            for chart in CHARTS:
                chart(t, ROOT / "figures" / "site")
            BARE = False
            social_preview(t)


if __name__ == "__main__":
    main()
