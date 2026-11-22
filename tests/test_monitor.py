import itertools

import numpy as np
import pytest

from agent_ecg import monitor
from agent_ecg.monitor import Monitor, states


def _validation(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores that rank runs well but overstate success, as training with --resolved-share does."""
    rng = np.random.default_rng(seed)
    k = rng.choice([5, 10], size=n)
    resolved = rng.random(n) < np.where(k == 5, 0.25, 0.15)
    score = 1 / (1 + np.exp(-(1.5 * resolved + rng.normal(size=n))))
    return resolved, score, k


def test_calibration_is_monotone_and_lowers_the_brier_score_on_new_runs() -> None:
    fitted = Monitor.fit(*_validation(seed=0))
    resolved, score, k = _validation(seed=1)
    calibrated = np.array([fitted.calibrate(s, c) for s, c in zip(score, k, strict=True)])
    assert np.mean((calibrated - resolved) ** 2) < np.mean((score - resolved) ** 2)
    assert np.all(np.diff([fitted.calibrate(s, 5) for s in np.linspace(0, 1, 50)]) >= 0)


def test_health_is_the_percentile_among_runs_of_the_same_age() -> None:
    resolved, score, k = _validation()
    fitted = Monitor.fit(resolved, score, k)
    health = np.array([fitted.health(s, c) for s, c in zip(score, k, strict=True)])
    assert health.min() >= 0 and health.max() <= 100
    assert abs(np.median(health[k == 5]) - 50) < 2
    middle = float(np.median(score))
    low, high = sorted((fitted.health(middle, 5), fitted.health(middle, 10)))
    assert low <= fitted.health(middle, 7) <= high  # interpolated between checkpoints


def test_a_critical_alert_is_right_at_least_95_percent_of_the_time() -> None:
    resolved, score, k = _validation()
    fitted = Monitor.fit(resolved, score, k)
    for c in (5, 10):
        threshold = fitted.critical[c]
        assert threshold is not None
        flagged = (k == c) & (score <= threshold)
        assert 30 <= flagged.sum() <= 0.15 * (k == c).sum()
        assert (~resolved[flagged]).mean() >= 0.95 and fitted.precision[c] >= 0.95
        assert fitted.critical_health(c) <= monitor.CRITICAL


def test_without_a_reliable_alert_a_checkpoint_never_turns_critical() -> None:
    rng = np.random.default_rng(0)
    fitted = Monitor.fit(rng.random(2000) < 0.3, rng.random(2000), np.full(2000, 5))
    assert fitted.critical[5] is None and fitted.critical_health(5) == -1
    assert fitted.alert_precision(5) is None


def test_the_alarm_needs_two_critical_steps_and_holds_until_health_recovers() -> None:
    health = [70, 40, 10, 10, 20, 10, 45, 10]
    flags = [False, False, True, True, False, True, False, True]
    expected = ["stable", "watch", "low", "critical", "critical", "critical", "watch", "low"]
    assert states(health, flags) == expected


def _thrashing_run(steps: int = 20) -> dict:
    """Ten calm steps, then the agent edits the same file again and again and the linter rejects it."""
    calm, bad = steps // 2, steps - steps // 2
    return {
        "run_id": "r1",
        "repository": "o/r",
        "instance_id": "o__r-1",
        "ending": "context_limit",
        "resolved": False,
        "issue_id": 0,
        "k": steps,
        "thought_chars": [100] * steps,
        "action_chars": [20] * steps,
        "observation_chars": [500] * steps,
        "context_chars": [5000 + 700 * i for i in range(steps)],
        "repeated_action": [False] * calm + [True] * bad,
        "action_kind": ["read"] * calm + ["edit"] * bad,
        "observation_kind": ["output"] * calm + ["edit_rejected"] * bad,
        "same_file_edit": [False] * calm + [False] + [True] * (bad - 1),
        "edit_target": ["none"] * calm + ["source"] * bad,
    }


def _rejections(cuts: list[dict]) -> list[float]:
    return [max(0.0, 1 - 2 * cut["observation_kind"].count("edit_rejected") / cut["k"]) for cut in cuts]


def test_a_thrashing_run_falls_from_stable_to_critical(tmp_path) -> None:
    rng = np.random.default_rng(0)
    score = rng.random(8000)
    fitted = Monitor.fit(rng.random(8000) < score**3, score, rng.choice([5, 10, 15, 20], size=8000))
    fitted.to_json(tmp_path / "monitor.json")
    fitted = Monitor.from_json(tmp_path / "monitor.json")

    steps = monitor.replay(_thrashing_run(), _rejections, fitted)
    assert [s["step"] for s in steps] == list(range(5, 21))
    assert steps[0]["state"] == "stable" and steps[0]["health"] > 90
    health = [s["health"] for s in steps]
    assert all(a >= b for a, b in itertools.pairwise(health[5:]))  # it only falls once it thrashes
    assert steps[-1]["state"] == "critical" and "Stop the agent" in steps[-1]["recommendation"]
    first_critical = next(s for s in steps if s["state"] == "critical")
    assert "recommendation" not in steps[steps.index(first_critical) - 1]
    assert all(s["state"] == "critical" for s in steps[steps.index(first_critical) :])  # no flicker
    assert steps[10]["events"] == ["repeated command", "edit rejected", "same file again"]
    assert steps[-1]["status"].startswith("In its first 20 steps the agent edited files 10 times")
    assert "critical" in monitor.render(_thrashing_run(), steps)


def test_a_run_shorter_than_the_first_step_cannot_be_replayed() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        monitor.prefixes(_thrashing_run(steps=4))


def test_a_step_next_to_a_checkpoint_that_never_alerts_never_turns_critical() -> None:
    # k=10 alerts at a real threshold, k=15 never does: steps 11-14 lie between them and must not alert.
    grid = np.linspace(0, 1, 101).tolist()
    fitted = Monitor(
        quantiles={10: grid, 15: grid},
        calibration={10: ([0.0, 1.0], [0.0, 1.0]), 15: ([0.0, 1.0], [0.0, 1.0])},
        critical={10: 0.10, 15: None},
        precision={10: 0.97, 15: None},
        recall={10: 0.3, 15: None},
    )
    assert fitted.critical_health(10) > 0
    assert all(fitted.critical_health(k) == -1 for k in range(11, 16))
