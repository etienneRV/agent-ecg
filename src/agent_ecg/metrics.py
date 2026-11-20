"""Score a model's predictions on the three tasks of each checkpoint.

A predictions file has one row per checkpoint, keyed by ``run_id`` and ``k``, with any of:

``p_resolved``
    Probability that the run will resolve its issue ("Will it pass?").
``pred_resolved``
    A yes/no decision for the same question. Without it, ``p_resolved >= 0.5`` is used.
``pred_ending``
    One of the six ending labels ("How will it end?").
``pred_status``
    The status report text. Only its last sentence, the prediction, is scored.

Most runs fail (83 % at 5 steps, 96 % at 30), so every score is reported per checkpoint, accuracy is
never used for pass/fail, and AUROC comes with a 95 % interval from resampling whole repositories.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from agent_ecg.status import describe, parse_status, split_status

KEY = ["run_id", "k"]
_BOOTSTRAP = 200


def _auroc_interval(resolved: np.ndarray, score: np.ndarray, groups: np.ndarray, seed: int = 0) -> list[float]:
    """A 95 % interval for AUROC from resampling whole repositories, as the split groups them."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    values = []
    for _ in range(_BOOTSTRAP):
        rows = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        if 0 < resolved[rows].sum() < len(rows):
            values.append(roc_auc_score(resolved[rows], score[rows]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [np.nan] * 2


def score_outcome(table: pd.DataFrame) -> dict:
    """Pass/fail: AUROC (with interval), average precision, Brier score, balanced accuracy."""
    resolved = table["resolved"].to_numpy(bool)
    result: dict = {"n": len(table), "resolved_rate": float(resolved.mean())}
    if "p_resolved" in table:
        p = table["p_resolved"].to_numpy(float)
        both = 0 < resolved.sum() < len(resolved)
        result["auroc"] = float(roc_auc_score(resolved, p)) if both else np.nan
        result["auroc_95"] = _auroc_interval(resolved, p, table["repository"].to_numpy()) if both else [np.nan] * 2
        result["auprc"] = float(average_precision_score(resolved, p)) if both else np.nan
        result["brier"] = float(brier_score_loss(resolved, p))
    decision = table["pred_resolved"] if "pred_resolved" in table else table.get("p_resolved")
    if decision is not None:
        decision = decision.to_numpy(bool) if decision.dtype == bool else decision.to_numpy(float) >= 0.5
        result["balanced_accuracy"] = float(balanced_accuracy_score(resolved, decision))
    return result


def score_ending(table: pd.DataFrame) -> dict:
    """How will it end: accuracy and macro-F1 over the endings that occur."""
    gold, pred = table["ending"].astype(str), table["pred_ending"].astype(str)
    return {
        "n": len(table),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=sorted(gold.unique()), average="macro", zero_division=0)),
    }


def _tokens(text: str) -> list[str]:
    return text.lower().replace(",", " ").replace(".", " ").split()


def token_f1(gold: str, pred: str) -> float:
    """Word-overlap F1, a stand-in for ROUGE-1: shows how much a text score rewards copied words."""
    gold_tokens, pred_tokens = Counter(_tokens(gold)), Counter(_tokens(pred))
    common = sum((gold_tokens & pred_tokens).values())
    if common == 0:
        return 0.0
    precision, recall = common / sum(pred_tokens.values()), common / sum(gold_tokens.values())
    return 2 * precision * recall / (precision + recall)


def score_status(table: pd.DataFrame) -> dict:
    """The status report: its prediction's three facts, plus word overlap to show why that is the score."""
    gold = [parse_status(text) for text in table["status"]]
    pred = [parse_status(text) for text in table["pred_status"].fillna("")]
    parsed = np.array([p.parsed for p in pred])
    remaining_gold = np.array([g.remaining for g in gold], float)
    remaining_pred = np.array([p.remaining if p.parsed else np.nan for p in pred], float)
    error = np.abs(remaining_pred - remaining_gold)
    resolved_gold = np.array([g.resolved for g in gold], bool)
    # An unparsed prediction counts as wrong: "not resolve" for resolved runs and vice versa.
    resolved_pred = np.array([p.resolved if p.parsed else not g.resolved for g, p in zip(gold, pred, strict=True)])
    ending_gold = [g.ending for g in gold]
    ending_pred = [p.ending if p.parsed else "unparsed" for p in pred]
    descriptions = [split_status(t)[0] for t in table["pred_status"].fillna("")]
    return {
        "n": len(table),
        "parse_rate": float(parsed.mean()),
        "remaining_mae": float(np.nanmean(error)) if parsed.any() else np.nan,
        "remaining_median_ae": float(np.nanmedian(error)) if parsed.any() else np.nan,
        "ending_accuracy": float(accuracy_score(ending_gold, ending_pred)),
        "ending_macro_f1": float(
            f1_score(ending_gold, ending_pred, labels=sorted(set(ending_gold)), average="macro", zero_division=0)
        ),
        "resolved_balanced_accuracy": float(balanced_accuracy_score(resolved_gold, resolved_pred)),
        "description_exact": float(
            np.mean([d == split_status(g)[0] for d, g in zip(descriptions, table["status"], strict=True)])
        ),
        "token_f1_full_text": float(
            np.mean([token_f1(g, p) for g, p in zip(table["status"], table["pred_status"].fillna(""), strict=True)])
        ),
        "token_f1_prediction": float(
            np.mean(
                [
                    token_f1(split_status(g)[1], split_status(p)[1])
                    for g, p in zip(table["status"], table["pred_status"].fillna(""), strict=True)
                ]
            )
        ),
    }


def _by_checkpoint(table: pd.DataFrame, score) -> dict:
    result = {f"k={k}": score(group) for k, group in table.groupby("k")}
    result["all (pooled)"] = score(table)
    return result


def evaluate(gold: pd.DataFrame, predictions: pd.DataFrame, split: str = "test", subset: bool = False) -> dict:
    """Score predictions against the gold checkpoints of one split, per checkpoint and pooled.

    With ``subset``, only the checkpoints the predictions cover are scored (for quick checks; report
    full-split scores).

    Raises:
        ValueError: If the predictions miss a checkpoint of the split or repeat one.
    """
    gold = gold[gold["split"] == split]
    if subset:
        gold = gold.merge(predictions[KEY], on=KEY)
    if predictions.duplicated(KEY).any():
        raise ValueError("predictions repeat a (run_id, k) pair")
    table = gold.merge(predictions, on=KEY, how="left", validate="one_to_one", suffixes=("", "_pred"))
    report: dict = {"split": split if not subset else f"{split} (subset)", "checkpoints": len(table)}
    columns = [c for c in ("p_resolved", "pred_resolved", "pred_ending", "pred_status") if c in predictions]
    if not columns:
        raise ValueError("predictions hold none of p_resolved, pred_resolved, pred_ending, pred_status")
    missing = table[columns].isna().all(axis=1).sum()
    if missing:
        raise ValueError(f"predictions miss {missing} of the {len(table)} {split} checkpoints")
    if "p_resolved" in columns or "pred_resolved" in columns:
        report["outcome"] = _by_checkpoint(table, score_outcome)
    if "pred_ending" in columns:
        report["ending"] = _by_checkpoint(table, score_ending)
    if "pred_status" in columns:
        report["status"] = _by_checkpoint(table, score_status)
    return report


def _cell(value) -> str:
    if isinstance(value, list):
        return f"[{value[0]:.3f}, {value[1]:.3f}]"
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def to_markdown(report: dict, title: str | None = None) -> str:
    """Render a report as Markdown tables, one per task."""
    lines = [f"### {title}", ""] if title else []
    for task in ("outcome", "ending", "status"):
        if task not in report:
            continue
        rows = report[task]
        metrics = list(next(iter(rows.values())))
        lines += [f"**{task}** ({report['split']}, {report['checkpoints']} checkpoints)", ""]
        lines += ["| checkpoint | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
        lines += [f"| {name} | " + " | ".join(_cell(row[m]) for m in metrics) + " |" for name, row in rows.items()]
        lines.append("")
    return "\n".join(lines)


def describe_matches(table: pd.DataFrame) -> float:
    """Share of checkpoints whose description, rebuilt from the signals, equals the target's."""
    return float(np.mean([describe(row) == split_status(row["status"])[0] for _, row in table.iterrows()]))


def concat_reports(reports: Iterable[tuple[str, dict]]) -> str:
    return "\n".join(to_markdown(report, name) for name, report in reports)


def paired_auroc_gain(gold: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, split: str = "test", n: int = 1000) -> dict:
    """AUROC of model ``a`` minus model ``b`` per checkpoint, with a 95 % interval from resampling the same
    repositories for both, and the share of resamples where ``a`` does not beat ``b``."""
    table = gold[gold["split"] == split].merge(a[[*KEY, "p_resolved"]], on=KEY, validate="one_to_one")
    table = table.merge(b[[*KEY, "p_resolved"]], on=KEY, validate="one_to_one", suffixes=("_a", "_b"))
    rng = np.random.default_rng(0)
    result = {}
    for k, group in table.groupby("k"):
        y, pa_, pb = (
            group["resolved"].to_numpy(bool),
            group["p_resolved_a"].to_numpy(),
            group["p_resolved_b"].to_numpy(),
        )
        repos = group["repository"].to_numpy()
        members = [np.flatnonzero(repos == r) for r in np.unique(repos)]
        gains = []
        for _ in range(n):
            rows = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
            if 0 < y[rows].sum() < len(rows):
                gains.append(roc_auc_score(y[rows], pa_[rows]) - roc_auc_score(y[rows], pb[rows]))
        result[f"k={k}"] = {
            "gain": float(roc_auc_score(y, pa_) - roc_auc_score(y, pb)),
            "gain_95": [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
            "p_not_better": float(np.mean(np.array(gains) <= 0)),
        }
    return result
