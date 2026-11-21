"""Choose, weight and label the OpenTSLM samples; no PyTorch here, so it is tested on any machine.

``opentslm/train.py`` and ``opentslm/predict.py`` use these to turn the exported checkpoints
(``agent-ecg opentslm-export``) into training items, one per (checkpoint, task).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from timenet_connectors.datasets.nebius.swe_agent_trajectories.tasks import ENDING_LABELS

TASKS = ("outcome", "ending", "status")
OUTCOMES = ("resolved", "unresolved")
ENDINGS = tuple(label.name for label in ENDING_LABELS.labels)


def load(data: Path, split: str, *, one_per_run: bool = False, cap: int | None = None, seed: int = 0) -> pd.DataFrame:
    """The exported checkpoints of one split, optionally one random checkpoint per run and at most ``cap``."""
    table = pd.read_parquet(data / f"{split}.parquet")
    if one_per_run:
        table = table.groupby("run_id").sample(1, random_state=seed)
    if cap is not None and cap < len(table):
        table = table.sample(cap, random_state=seed)
    return table.reset_index(drop=True)


def post_prompts(data: Path) -> dict[str, str]:
    return json.loads((data / "post_prompts.json").read_text())


def answer(row: pd.Series | dict, task: str) -> str:
    """The target text of one task for one checkpoint."""
    if task == "outcome":
        return OUTCOMES[0] if row["resolved"] else OUTCOMES[1]
    if task == "ending":
        return str(row["ending"])
    return str(row["status"])


def items(table: pd.DataFrame, tasks: tuple[str, ...] | list[str]) -> list[tuple[int, str]]:
    """One training item per (checkpoint row, task)."""
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown tasks {sorted(unknown)}; choose from {TASKS}")
    return [(row, task) for row in range(len(table)) for task in tasks]


def weights(table: pd.DataFrame, chosen: list[tuple[int, str]], resolved_share: float | None) -> np.ndarray:
    """Sampling weight of each item.

    Each checkpoint counts 1 / (checkpoints of its run in the table), so long runs, which fail more
    often, do not count up to five times. With ``resolved_share``, resolved runs are reweighted to make
    up that share of the total weight; the model then sees successes often enough to learn them, at the
    cost of calibration (its probabilities overstate success).
    """
    per_run = 1.0 / table.groupby("run_id")["k"].transform("size").to_numpy(float)
    resolved = table["resolved"].to_numpy(bool)
    w = np.array([per_run[row] for row, _ in chosen])
    if resolved_share is not None:
        is_resolved = np.array([resolved[row] for row, _ in chosen])
        if not 0 < resolved_share < 1 or not is_resolved.any() or is_resolved.all():
            raise ValueError("resolved_share needs both outcomes in the table and a value between 0 and 1")
        w[is_resolved] *= resolved_share / w[is_resolved].sum()
        w[~is_resolved] *= (1 - resolved_share) / w[~is_resolved].sum()
    return w / w.sum()


def mean_auroc(resolved: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
    """Mean of the per-checkpoint AUROCs (checkpoints with both outcomes), the model-selection score.

    Pooling across checkpoints would reward knowing the checkpoint, not the run (see REPORT.md).
    """
    scores = [
        roc_auc_score(resolved[k == c], p[k == c]) for c in np.unique(k) if 0 < resolved[k == c].sum() < (k == c).sum()
    ]
    return float(np.mean(scores)) if scores else float("nan")


def parse_weights(text: str) -> dict[str, float]:
    """``"outcome=2,status=0.5"`` -> {"outcome": 2.0, "status": 0.5}; unnamed tasks weigh 1."""
    weights = {}
    for part in filter(None, text.split(",")):
        task, _, value = part.partition("=")
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r} in task weights")
        weights[task] = float(value)
    return weights


def threshold(resolved: np.ndarray, p: np.ndarray) -> float:
    """The cut on p that maximises balanced accuracy, chosen on validation."""
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    scores = [balanced_accuracy_score(resolved, p >= c) for c in candidates]
    return float(candidates[int(np.argmax(scores))])
