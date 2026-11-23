"""Publication-style figures of the main results, for the paper, slides and posts.

    uv run python scripts/paper_figures.py      # -> results/figures/{results,social}.{png,pdf,svg}
                                                #    and the website's copies in site/figures/

The house style follows figures4papers' scientific-figure-making guide (Chen Liu,
https://github.com/ChenLiu-1996/figures4papers): Helvetica/Arial, no top or right spines, frameless
legends, bold panel letters, a semantic palette (blue for our model, red for the baseline it is held
against, neutrals for references), y-limits fitted to the data, 300 dpi PNG plus PDF and SVG.

Every number comes from results/: the test reports (AUROC and its interval per checkpoint), the paired
comparisons (`agent-ecg compare`, with and without `--same-issue`) and the live-monitor scan.
"""

from __future__ import annotations

import io
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts

RESULTS = charts.RESULTS
OUT = RESULTS / "figures"
SITE_FIGS = charts.ROOT / "site" / "figures"
STEPS = charts.STEPS

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#C9DAEE",
    "red_strong": "#B64342",
    "red_2": "#E9A6A1",
    "red_1": "#F6CFCB",
    "teal": "#42949E",
    "neutral": "#CFCECE",
    "gray": "#767676",
    "ink": "#272727",
}
STYLE = {
    "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 16,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2,
    "axes.edgecolor": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "agent-ecg",  # stable element ids: reruns give the same SVG files
    "pdf.fonttype": 42,
}


# ---------------------------------------------------------------- data


def gains(path: Path) -> dict[int, tuple[float, float, float]]:
    """Per checkpoint: (gain, interval low, interval high) from an `agent-ecg compare` output."""
    out = {}
    for line in path.read_text().splitlines():
        hit = re.match(r"\s*k=(\d+): ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if hit:
            out[int(hit[1])] = (float(hit[2]), float(hit[3]), float(hit[4]))
    if set(out) != set(STEPS):
        raise SystemExit(f"could not parse {path}")
    return out


def same_issue_share(name: str) -> dict[int, tuple[float, float, float]]:
    """Same-issue pair share with its interval: the gain over a constant score, which ties every pair (0.5)."""
    return {
        k: (0.5 + g, 0.5 + lo, 0.5 + hi)
        for k, (g, lo, hi) in gains(RESULTS / "same_issue" / f"{name}_vs_chance.test.txt").items()
    }


def load() -> dict:
    tslm = charts.tslm_runs()
    return {
        "auroc": {
            "full": charts.model_auroc(RESULTS / "issue_signals_thoughts.test.json"),
            "signals": charts.model_auroc(RESULTS / "signals.test.json"),
            "issue": charts.model_auroc(RESULTS / "issue.test.json"),
            "A": charts.model_auroc(tslm["A"] / "report.test.json"),
        },
        "gap_all": gains(RESULTS / "opentslm" / "full-multi" / "compare_vs_issue_signals_thoughts.test.txt"),
        "gap_same": gains(RESULTS / "same_issue" / "A_vs_issue_signals_thoughts.test.txt"),
        "same": {
            "signals": same_issue_share("signals"),
            "thoughts": same_issue_share("thoughts"),
            "A": same_issue_share("A"),
        },
        "monitor": charts.scan_summary(),
        "label_a": charts.tslm_labels()["A"],
    }


# ---------------------------------------------------------------- drawing helpers


def letter(ax, text: str, x: float = -0.13) -> None:
    ax.text(x, 1.08, text, transform=ax.transAxes, fontsize=26, fontweight="bold", va="bottom", ha="left")


def series(
    ax,
    values: dict,
    color: str,
    label: str,
    *,
    dashed: bool = False,
    band: bool = True,
    z: int = 3,
    alpha: float = 0.11,
) -> None:
    x = np.array(STEPS)
    y = np.array([values[k][0] for k in STEPS])
    if band:
        lo, hi = np.array([values[k][1] for k in STEPS]), np.array([values[k][2] for k in STEPS])
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=z - 1)
    ax.plot(x, y, color=color, lw=3, ls="--" if dashed else "-", marker="o", ms=7, label=label, zorder=z)


def steps_axis(ax) -> None:
    ax.set_xticks(STEPS)
    ax.set_xlim(3, 32)
    ax.set_xlabel("Steps of the run seen")


def banner(fig, text: str, y: float = 0.035) -> None:
    """The take-home message on a light arrow across the bottom, as figures4papers does."""
    arrow = FancyArrow(
        0.04,
        y,
        0.90,
        0,
        width=0.05,
        head_width=0.075,
        head_length=0.025,
        length_includes_head=True,
        transform=fig.transFigure,
        color=PALETTE["blue_light"],
        lw=0,
        zorder=0,
    )
    fig.add_artist(arrow)
    fig.text(0.48, y, text, ha="center", va="center", fontsize=19, color=PALETTE["ink"])


# ---------------------------------------------------------------- panels


def panel_auroc(ax, d: dict) -> None:
    a = d["auroc"]
    series(ax, a["full"], PALETTE["red_strong"], "Simple model, same inputs as ours")
    series(ax, a["A"], PALETTE["blue_main"], f"Our model ({d['label_a']})", z=4)
    series(ax, a["signals"], PALETTE["teal"], "Simple model, actions only (the ECG)", band=False)
    series(ax, a["issue"], PALETTE["gray"], "Simple model, issue text only", dashed=True, band=False)
    steps_axis(ax)
    ax.set_ylabel("AUROC (0.5 = coin flip)")
    ax.set_ylim(0.47, 0.78)
    ax.set_title("Will the run succeed? All test runs", fontsize=17, pad=12)
    ax.legend(loc="lower left", fontsize=12.5, handlelength=2.2)


def panel_gap(ax, d: dict) -> None:
    x = np.array(STEPS, float)
    for key, offset, color, label in (
        ("gap_all", -0.9, PALETTE["red_2"], "All test runs"),
        ("gap_same", 0.9, PALETTE["blue_main"], "Two attempts at the same issue"),
    ):
        g = d[key]
        y = np.array([g[k][0] for k in STEPS])
        err = np.array([[g[k][0] - g[k][1] for k in STEPS], [g[k][2] - g[k][0] for k in STEPS]])
        ax.errorbar(
            x + offset,
            y,
            yerr=err,
            fmt="o",
            ms=11,
            color=color,
            ecolor=color,
            elinewidth=3,
            capsize=6,
            capthick=2.5,
            label=label,
            zorder=3,
            mec=PALETTE["ink"] if key == "gap_same" else color,
            mew=1.2,
        )
    ax.axhline(0, color=PALETTE["ink"], lw=1.8, zorder=1)
    ax.axhspan(0, 0.12, color=PALETTE["blue_light"], alpha=0.35, lw=0, zorder=0)
    ax.text(25, 0.105, "ours ahead", ha="center", va="top", fontsize=13, color=PALETTE["blue_main"])
    ax.text(25, -0.155, "simple model ahead", ha="center", va="bottom", fontsize=13, color=PALETTE["red_strong"])
    steps_axis(ax)
    ax.set_ylim(-0.17, 0.12)
    ax.set_ylabel("Ours minus simple model (same inputs)")
    ax.set_title("The simple model's lead comes from spotting hard issues", fontsize=17, pad=12)
    ax.legend(loc="lower left", fontsize=12.5)


def panel_same(ax, d: dict) -> None:
    s = d["same"]
    series(ax, s["signals"], PALETTE["teal"], "Simple model, actions only (the ECG)")
    series(ax, s["A"], PALETTE["blue_main"], f"Our model ({d['label_a']})", band=False, z=4)
    series(ax, s["thoughts"], PALETTE["red_2"], "Simple model, thoughts only", band=False)
    ax.axhline(0.5, color=PALETTE["gray"], lw=2.5, ls="--", zorder=1)
    ax.text(
        31.5,
        0.503,
        "issue text alone: 0.50 by design",
        ha="right",
        va="bottom",
        fontsize=12,
        color=PALETTE["gray"],
    )
    steps_axis(ax)
    ax.set_ylim(0.45, 0.83)
    ax.text(5.3, 0.465, "shaded: 95 % interval of the actions-only model", fontsize=12, color=PALETTE["teal"])
    ax.set_ylabel("Pairs ranked right")
    ax.set_title("Two attempts at the same issue: which one will succeed?", fontsize=17, pad=12)
    ax.legend(loc="upper left", fontsize=12.5)


def panel_monitor(ax, d: dict) -> None:
    m = d["monitor"]
    groups = ("Failing runs flagged", "False alarms on runs\nthat succeeded")
    models = (
        (f"Our model: half its catches by step {m['A']['median_step']}", m["A"], PALETTE["blue_main"]),
        (f"Simple model (actions only): by step {m['signals']['median_step']}", m["signals"], PALETTE["neutral"]),
    )
    width = 0.36
    for i, (label, row, color) in enumerate(models):
        values = (row["flagged"], row["false_alarms"])
        xs = np.arange(len(groups)) + (i - 0.5) * width
        bars = ax.bar(xs, values, width, color=color, edgecolor=PALETTE["ink"], lw=2, label=label, zorder=2)
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.8,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 46)
    ax.set_ylabel("Share of runs (%)")
    ax.set_title(f"As a live monitor ({m['A']['failing'] + m['A']['resolved']} unseen runs)", fontsize=17, pad=12)
    ax.legend(loc="upper right", fontsize=12.5)


# ---------------------------------------------------------------- figures


def save(fig, name: str) -> None:
    """The paper copies: 300 dpi PNG, PDF and SVG with editable text, without timestamps (reruns give the same files)."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.svg", facecolor="white", metadata={"Date": None})
    print(f"wrote {OUT.relative_to(charts.ROOT)}/{name}.png|pdf|svg")


def save_web(fig, name: str) -> None:
    """The website's copy: SVG with text as outlines, so it looks the same without the fonts installed, and no
    DOCTYPE, which web hosts refuse (as scripts/charts.py does)."""
    SITE_FIGS.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with plt.rc_context({"svg.fonttype": "path", "svg.hashsalt": "agent-ecg"}):
        fig.savefig(buf, format="svg", facecolor="white", metadata={"Date": None}, bbox_inches="tight", pad_inches=0.15)
    (SITE_FIGS / f"{name}.svg").write_text(re.sub(r"<!DOCTYPE[^>]*>\s*", "", buf.getvalue()))
    print(f"wrote {SITE_FIGS.relative_to(charts.ROOT)}/{name}.svg")


def results_figure(d: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.15, hspace=0.42, wspace=0.24)
    for ax, draw, tag in zip(axes.flat, (panel_auroc, panel_gap, panel_same, panel_monitor), "abcd", strict=True):
        draw(ax, d)
        letter(ax, tag)
    banner(
        fig,
        "On two attempts at the same issue, our model is level with the best simple model, whose lead comes from spotting hard issues.",
        y=0.045,
    )
    save(fig, "results")
    save_web(fig, "paper_results")
    plt.close(fig)
    shutil.copyfile(OUT / "results.pdf", SITE_FIGS / "paper_results.pdf")  # the site links it for download


def social_figure(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.73, bottom=0.2, wspace=0.28)
    panel_gap(axes[0], d)
    panel_same(axes[1], d)
    for ax, tag in zip(axes, "ab", strict=True):
        letter(ax, tag)
    fig.text(
        0.04,
        0.935,
        "Agent ECG: is this coding-agent run going to fail?",
        fontsize=28,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.04,
        0.87,
        "7,640 unseen SWE-agent runs on 143 repositories. Error bars: 95 % intervals from resampling repositories.",
        fontsize=16,
        color=PALETTE["gray"],
    )
    banner(fig, "The simple model wins by spotting hard issues. On the same issue, our model is level.", y=0.06)
    save(fig, "social")
    plt.close(fig)


def site_same_issue(d: dict) -> None:
    """The website's same-issue chart: the social card's two panels, without its title and banner (the page has
    its own heading and caption)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.28)
    panel_gap(axes[0], d)
    panel_same(axes[1], d)
    for ax, tag in zip(axes, "ab", strict=True):
        letter(ax, tag)
    save_web(fig, "same_issue")
    plt.close(fig)


def site_auroc(d: dict) -> None:
    """The website's accuracy chart: each of our three models next to the simple model that reads the same inputs."""
    tslm = {k: charts.model_auroc(p / "report.test.json") for k, p in charts.tslm_runs().items()}
    names = charts.tslm_labels()
    simple = {
        n: charts.model_auroc(RESULTS / f"{n}.test.json")
        for n in ("signals", "signals_thoughts", "issue_signals_thoughts", "issue")
    }
    panels = (
        ("C", "Reads the agent's actions (the ECG)", "signals"),
        ("B", "Reads its thoughts and actions", "signals_thoughts"),
        ("A", "Reads the issue, thoughts and actions", "issue_signals_thoughts"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.74, bottom=0.14, wspace=0.08)
    for ax, (slot, title, base), tag in zip(axes, panels, "abc", strict=True):
        ax.axhline(0.5, color=PALETTE["gray"], lw=1.6, ls=":", zorder=1)
        ax.text(31.5, 0.506, "coin flip", ha="right", va="bottom", fontsize=12, color=PALETTE["gray"])
        series(ax, simple["issue"], "#A9A9A9", "Simple model, issue text only", dashed=True, band=False)
        series(ax, simple[base], PALETTE["gray"], "Simple model, same inputs", alpha=0.07)
        series(ax, tslm[slot], PALETTE["blue_main"], "Our model", z=4, alpha=0.09)
        steps_axis(ax)
        ax.set_ylim(0.47, 0.78)
        ax.set_title(title, fontsize=16.5, pad=30)
        ax.text(
            0.5,
            1.035,
            f"our model: {names[slot]}",
            transform=ax.transAxes,
            ha="center",
            fontsize=12.5,
            color=PALETTE["gray"],
        )
        letter(ax, tag, x=-0.13 if tag == "a" else -0.04)
    axes[0].set_ylabel("AUROC (0.5 = coin flip)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 1, 0]  # ours, same inputs, issue only
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=3,
        fontsize=14,
        bbox_to_anchor=(0.53, 0.995),
    )
    save_web(fig, "auroc_paper")
    plt.close(fig)


def site_monitor(d: dict) -> None:
    """The website's live-monitor chart: three rates side by side, and which failing runs each monitor caught."""
    m = d["monitor"]
    a, s = m["A"], m["signals"]
    neither = a["failing"] - m["a_only"] - m["both"] - m["s_only"]
    fig = plt.figure(figsize=(16, 6.4))
    rates = (
        ("Failing runs\nflagged", a["flagged"], s["flagged"], "{:.0f}%", 46, "higher is better"),
        (
            "False alarms on runs\nthat succeeded",
            a["false_alarms"],
            s["false_alarms"],
            "{:.0f}%",
            46,
            "lower is better",
        ),
        ("Half the catches\ncame by", a["median_step"], s["median_step"], "step {:.0f}", 12.5, "lower is earlier"),
    )
    for i, (title, ours, simple, fmt, top, hint) in enumerate(rates):
        ax = fig.add_axes([0.035 + i * 0.19, 0.2, 0.14, 0.56])
        bars = ax.bar(
            [0, 1],
            [ours, simple],
            width=0.62,
            color=[PALETTE["blue_main"], PALETTE["neutral"]],
            edgecolor=PALETTE["ink"],
            lw=2,
            zorder=2,
        )
        for bar, v in zip(bars, (ours, simple), strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + top * 0.03,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=17,
                fontweight="bold",
            )
        ax.set_xticks([0, 1], ["Ours", "Simple\nmodel"])
        ax.tick_params(axis="x", length=0, labelsize=14)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=16, pad=14)
        ax.text(0.5, -0.25, hint, transform=ax.transAxes, ha="center", fontsize=12.5, color=PALETTE["gray"])
    ax = fig.add_axes([0.6, 0.02, 0.38, 0.92])
    ax.set_aspect("equal")
    ax.axis("off")
    cx = 0.62
    ax.add_patch(Circle((-cx, 0), 1, facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], lw=2.5, zorder=1))
    ax.add_patch(
        Circle((cx, 0), 1, facecolor=PALETTE["neutral"], edgecolor=PALETTE["gray"], lw=2.5, alpha=0.55, zorder=2)
    )
    for x, n, label in ((-1.08, m["a_only"], "ours only"), (0, m["both"], "both"), (1.08, m["s_only"], "simple only")):
        ax.text(
            x, 0.1, f"{n}", ha="center", va="center", fontsize=34, fontweight="bold", color=PALETTE["ink"], zorder=3
        )
        ax.text(x, -0.3, label, ha="center", va="center", fontsize=14, color=PALETTE["ink"], zorder=3)
    ax.text(-cx, 1.2, "Our model", ha="center", fontsize=16, fontweight="bold", color=PALETTE["blue_main"])
    ax.text(cx, 1.2, "Simple model", ha="center", fontsize=16, fontweight="bold", color=PALETTE["gray"])
    ax.text(0, 1.62, f"Which of the {a['failing']} failing runs each one caught", ha="center", fontsize=17)
    ax.text(0, -1.38, f"{neither} were caught by neither", ha="center", fontsize=14, color=PALETTE["gray"])
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.6, 1.85)
    save_web(fig, "monitor_paper")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(STYLE)
    d = load()
    results_figure(d)
    social_figure(d)
    site_same_issue(d)
    site_monitor(d)
    site_auroc(d)


if __name__ == "__main__":
    main()
