import numpy as np
import pandas as pd
import pytest

from agent_ecg import opentslm_samples as s
from agent_ecg.metrics import evaluate


def _table() -> pd.DataFrame:
    # Run a has 4 checkpoints (unresolved), runs b and c one each (b resolved).
    return pd.DataFrame(
        {
            "run_id": ["a", "a", "a", "a", "b", "c"],
            "k": [5, 10, 20, 30, 5, 5],
            "resolved": [False, False, False, False, True, False],
            "ending": ["context_limit"] * 4 + ["submitted", "early_exit"],
            "status": ["report"] * 6,
        }
    )


def test_items_and_answers() -> None:
    table = _table()
    chosen = s.items(table, ["outcome", "ending"])
    assert len(chosen) == 12 and chosen[:2] == [(0, "outcome"), (0, "ending")]
    assert s.answer(table.iloc[4], "outcome") == "resolved" and s.answer(table.iloc[0], "outcome") == "unresolved"
    assert s.answer(table.iloc[5], "ending") == "early_exit"
    with pytest.raises(ValueError, match="unknown tasks"):
        s.items(table, ["outcome", "vibes"])


def test_each_run_counts_once_and_resolved_share_is_met() -> None:
    table = _table()
    chosen = s.items(table, ["outcome"])
    natural = s.weights(table, chosen, None)
    # Run a's four checkpoints together weigh as much as run b's one.
    assert natural[:4].sum() == pytest.approx(natural[4])
    balanced = s.weights(table, chosen, 0.5)
    assert balanced[4] == pytest.approx(0.5) and balanced.sum() == pytest.approx(1)


def test_threshold_maximises_balanced_accuracy() -> None:
    resolved = np.array([False] * 90 + [True] * 10)
    p = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.3, 0.9, 10)]
    cut = s.threshold(resolved, p)
    assert 0.3 <= cut <= 0.41


def test_evaluate_can_score_a_subset() -> None:
    gold = _table().assign(split="test", repository=["r1", "r1", "r1", "r1", "r2", "r3"])
    pred = gold[["run_id", "k"]].iloc[[0, 4, 5]].assign(p_resolved=[0.1, 0.9, 0.2])
    with pytest.raises(ValueError, match="miss"):
        evaluate(gold, pred)
    report = evaluate(gold, pred, subset=True)
    assert report["checkpoints"] == 3 and report["outcome"]["k=5"]["auroc"] == 1.0


def test_mean_auroc_averages_checkpoints_and_skips_one_class_groups() -> None:
    resolved = np.array([True, False, True, False, False, False])
    k = np.array([5, 5, 10, 10, 20, 20])
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.5])
    # k=5 perfect (1.0), k=10 inverted (0.0), k=20 has no resolved run and is skipped.
    assert s.mean_auroc(resolved, k, p) == 0.5
    assert np.isnan(s.mean_auroc(np.zeros(3, bool), np.array([5, 5, 5]), np.ones(3)))


def test_task_weights_parse() -> None:
    assert s.parse_weights("outcome=2,status=0.5") == {"outcome": 2.0, "status": 0.5}
    assert s.parse_weights("") == {}
    with pytest.raises(ValueError, match="unknown task"):
        s.parse_weights("vibes=1")
