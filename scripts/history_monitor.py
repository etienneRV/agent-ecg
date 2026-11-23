"""Why early prediction is hard here, and what a monitor that remembers earlier attempts gains.

    uv run python scripts/history_monitor.py      # -> results/history_monitor.md

Two parts, both computed from data/checkpoints.parquet and results/:

1. Where the outcome's variance lives: how many issues are never, always or sometimes solved, the share of
   the variance that the issue alone explains, and how well the success rate of the other attempts at the
   same issue (leave one out) ranks a run.
2. A history-aware monitor. Coding agents are usually run several times on the same issue, and earlier
   attempts finish before a new one starts, so a monitor can know how they went. For each test run we draw
   m earlier attempts at random among the other attempts at the same issue and combine their success rate
   with our model's chance of success (logistic regression per checkpoint, cross-fitted over two halves of
   the test repositories, so the combination is never scored on the repositories it was fitted on).
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_ecg import metrics  # noqa: E402

RESULTS = ROOT / "results"
OURS = RESULTS / "opentslm" / "full-multi" / "predictions.test.parquet"
SIMPLE = RESULTS / "baselines" / "issue_signals_thoughts.test.parquet"
STEPS = (5, 10, 15, 20, 30)
HISTORY = (1, 3, 5)
SEED = 0


def final_runs(table: pd.DataFrame) -> pd.DataFrame:
    """One row per run (its longest checkpoint), with its outcome."""
    return (
        table.sort_values("k").groupby("run_id").tail(1)[["run_id", "instance_id", "repository", "split", "resolved"]]
    )


def variance_part(runs: pd.DataFrame) -> dict:
    g = runs.groupby("instance_id")["resolved"].agg(["size", "mean"])
    p = runs["resolved"].mean()
    between = ((g["mean"] - p) ** 2 * g["size"]).sum() / g["size"].sum()
    train = runs[runs["split"] == "train"].groupby("instance_id")["resolved"].max()
    return {
        "runs": len(runs),
        "issues": len(g),
        "attempts_median": float(g["size"].median()),
        "attempts_mean": float(g["size"].mean()),
        "never": float((g["mean"] == 0).mean()),
        "always": float((g["mean"] == 1).mean()),
        "mixed": float(((g["mean"] > 0) & (g["mean"] < 1)).mean()),
        "success": float(p),
        "issue_share": float(between / (p * (1 - p))),
        "train_runs": int((runs["split"] == "train").sum()),
        "train_issues": len(train),
        "train_solved": int(train.sum()),
    }


def sibling_auroc(table: pd.DataFrame) -> dict[int, float]:
    """AUROC of the leave-one-out success rate of the other test runs on the same issue, per checkpoint."""
    out = {}
    for k, g in table[table["split"] == "test"].groupby("k"):
        s = g.groupby("instance_id")["resolved"].transform("sum")
        n = g.groupby("instance_id")["resolved"].transform("size")
        loo = ((s - g["resolved"]) / (n - 1)).where(n > 1)
        keep = loo.notna()
        out[int(k)] = float(roc_auc_score(g.loc[keep, "resolved"], loo[keep]))
    return out


def after_failures(runs: pd.DataFrame, ms=(0, 1, 3, 5)) -> dict[int, float]:
    """Chance an attempt succeeds given that m other attempts at the same issue, drawn at random, all failed."""
    groups = runs.groupby("instance_id")["resolved"].agg(["size", "sum"])
    out = {}
    for m in ms:
        num = den = 0.0
        for n, s in groups.itertuples(index=False):
            if n <= m:
                continue
            fail_if_success = comb(n - s, m) / comb(n - 1, m) if s else 0.0
            fail_if_failure = comb(n - s - 1, m) / comb(n - 1, m) if n - s else 0.0
            num += s * fail_if_success
            den += s * fail_if_success + (n - s) * fail_if_failure
        out[m] = num / den
    return out


def earlier_attempts(runs: pd.DataFrame, m: int, rng: np.random.Generator) -> pd.DataFrame:
    """For each test run: how many of m other attempts at the same issue, drawn at random, succeeded."""
    test = runs[runs["split"] == "test"]
    rows = []
    for _, g in test.groupby("instance_id"):
        ids, ok = g["run_id"].to_numpy(), g["resolved"].to_numpy(bool)
        for i, run in enumerate(ids):
            others = np.delete(np.arange(len(ids)), i)
            pick = rng.choice(others, size=min(m, len(others)), replace=False) if len(others) else others
            rows.append((run, int(ok[pick].sum()), len(pick)))
    return pd.DataFrame(rows, columns=["run_id", "earlier_solved", "earlier_seen"])


def combine(gold: pd.DataFrame, model: pd.DataFrame, history: pd.DataFrame, prior: float) -> pd.DataFrame:
    """Our model's chance and the earlier attempts' success rate, combined by a logistic regression per checkpoint,
    fitted on one half of the test repositories and applied to the other half (and the other way round)."""
    table = gold.merge(model[["run_id", "k", "p_resolved"]], on=["run_id", "k"]).merge(history, on="run_id")
    p = table["p_resolved"].clip(1e-4, 1 - 1e-4)
    rate = (table["earlier_solved"] + 2 * prior) / (table["earlier_seen"] + 2)  # shrunk toward the base rate
    table["x_model"], table["x_history"] = np.log(p / (1 - p)), np.log(rate / (1 - rate))
    repos = np.sort(table["repository"].unique())
    half = set(np.random.default_rng(SEED).permutation(repos)[: len(repos) // 2])
    table["fold"] = table["repository"].isin(half).astype(int)
    out = []
    for _, g in table.groupby("k"):
        g = g.copy()
        for fold in (0, 1):
            fit, apply = g[g["fold"] != fold], g[g["fold"] == fold]
            lr = LogisticRegression().fit(fit[["x_model", "x_history"]], fit["resolved"])
            g.loc[apply.index, "combined"] = lr.predict_proba(apply[["x_model", "x_history"]])[:, 1]
        out.append(g)
    table = pd.concat(out)
    return table[["run_id", "k", "combined"]].rename(columns={"combined": "p_resolved"})


def cell(gain: tuple[float, float, float]) -> str:
    g, lo, hi = gain
    mark = "**" if lo > 0 or hi < 0 else ""
    return f"{mark}{g:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}"


def main() -> None:
    table = pd.read_parquet(
        ROOT / "data" / "checkpoints.parquet", columns=["run_id", "instance_id", "repository", "split", "k", "resolved"]
    )
    runs = final_runs(table)
    v = variance_part(runs)
    sib = sibling_auroc(table)
    fails = after_failures(runs)
    gold = table[table["split"] == "test"].copy()
    ours = pd.read_parquet(OURS)
    simple = pd.read_parquet(SIMPLE)
    rng = np.random.default_rng(SEED)
    histories = {m: earlier_attempts(runs, m, rng) for m in HISTORY}

    def auroc(pred: pd.DataFrame, k: int) -> float:
        g = gold[gold["k"] == k].merge(pred[["run_id", "k", "p_resolved"]], on=["run_id", "k"])
        return float(roc_auc_score(g["resolved"], g["p_resolved"]))

    prior = float(runs["resolved"].mean())
    combos = {m: combine(gold, ours, histories[m], prior) for m in HISTORY}
    combo_simple = combine(gold, simple, histories[3], prior)
    history_only = {
        m: gold[["run_id", "k"]]
        .merge(histories[m], on="run_id")
        .assign(p_resolved=lambda t: (t["earlier_solved"] + 2 * prior) / (t["earlier_seen"] + 2))
        for m in HISTORY
    }
    gain3 = metrics.paired_auroc_gain(gold, combos[3], ours, "test")

    lines = [
        "# Why early prediction is hard here, and what remembering earlier attempts gains",
        "",
        "Computed by `scripts/history_monitor.py` from `data/checkpoints.parquet` and `results/`.",
        "",
        "## Where the outcome's variance lives",
        "",
        f"- {v['runs']:,} runs on {v['issues']:,} issues ({v['attempts_median']:.0f} attempts per issue at the median, {v['attempts_mean']:.1f} on average).",
        f"- {v['never']:.0%} of issues are never solved in any attempt, {v['always']:.0%} always, {v['mixed']:.0%} sometimes.",
        f"- Which issue it is explains {v['issue_share']:.0%} of the variance in outcomes (overall success {v['success']:.1%}).",
        f"- Training split: {v['train_runs']:,} runs but {v['train_issues']:,} issues, of which {v['train_solved']:,} were ever solved.",
        "",
        "Leave-one-out success rate of the other attempts at the same issue, as a score (test AUROC):",
        "",
        "| steps seen | " + " | ".join(str(k) for k in STEPS) + " |",
        "|---|" + "---|" * len(STEPS),
        "| sibling success rate | " + " | ".join(f"{sib[k]:.3f}" for k in STEPS) + " |",
        "| our model (ecg-full) | " + " | ".join(f"{auroc(ours, k):.3f}" for k in STEPS) + " |",
        "",
        "Chance an attempt succeeds, given that m other attempts at the same issue (drawn at random) all failed:",
        "",
        "| m | " + " | ".join(str(m) for m in fails) + " |",
        "|---|" + "---|" * len(fails),
        "| chance of success | " + " | ".join(f"{100 * p:.1f} %" for p in fails.values()) + " |",
        "",
        "## A monitor that remembers earlier attempts (test AUROC)",
        "",
        "For each test run, m earlier attempts are drawn at random among the other attempts at the same issue "
        f"(seed {SEED}); their success rate is combined with the model's chance of success by a logistic "
        "regression per checkpoint, cross-fitted over two halves of the test repositories.",
        "",
        "| model | " + " | ".join(str(k) for k in STEPS) + " |",
        "|---|" + "---|" * len(STEPS),
        "| our model alone | " + " | ".join(f"{auroc(ours, k):.3f}" for k in STEPS) + " |",
    ]
    for m in HISTORY:
        lines.append(
            f"| {m} earlier attempt{'s' if m > 1 else ''} alone | "
            + " | ".join(f"{auroc(history_only[m], k):.3f}" for k in STEPS)
            + " |"
        )
    for m in HISTORY:
        lines.append(
            f"| our model + {m} earlier attempt{'s' if m > 1 else ''} | "
            + " | ".join(f"{auroc(combos[m], k):.3f}" for k in STEPS)
            + " |"
        )
    lines += [
        "| simple model (same inputs) + 3 earlier attempts | "
        + " | ".join(f"{auroc(combo_simple, k):.3f}" for k in STEPS)
        + " |",
        "",
        "Gain of our model + 3 earlier attempts over our model alone (paired repository bootstrap; bold: the interval excludes zero):",
        "",
        "| steps seen | " + " | ".join(str(k) for k in STEPS) + " |",
        "|---|" + "---|" * len(STEPS),
        "| gain | " + " | ".join(cell((gain3[f"k={k}"]["gain"], *gain3[f"k={k}"]["gain_95"])) for k in STEPS) + " |",
        "",
    ]
    out = RESULTS / "history_monitor.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
