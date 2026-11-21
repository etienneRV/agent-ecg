"""Turn a model's pass/fail scores into the live monitor: calibration, health, alerts and a replay.

Every piece is fitted on validation predictions, never on test:

- **calibration:** an isotonic map per checkpoint from a model's score to the resolved rate observed
  on validation. Training reweights resolved runs (``--resolved-share``), which inflates the scores;
  the map gives probabilities back.
- **health, 0 to 100:** the percentile of a run's score among validation runs at the same checkpoint.
  A health of 10 means the run looks worse than 90 % of the runs still going at that point. It is
  relative to runs of the same age on purpose: 96 % of the runs still going at step 30 fail, so an
  absolute probability would sit near zero for every long run, whatever the agent does.
- **critical:** a health of at most ``CRITICAL`` whose validation runs failed at least ``target`` of
  the time (95 % by default), so an alert is right that often on repositories the model never saw.
  Where the bottom 15 % fail less often, the threshold is lowered until they do; where no group of at
  least ``min_flagged`` runs does, that checkpoint never alerts.

Between two checkpoints, health and the critical limit are interpolated linearly. :func:`replay`
scores a run step by step, as a live monitor would see it, and :func:`states` turns each step into
``stable``, ``watch``, ``low`` or ``critical``. The alarm goes on after ``HOLD`` critical steps in a
row and stays on until health is back to ``WATCH``, so it does not flicker around its threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from agent_ecg import data
from agent_ecg.status import describe

STABLE, WATCH, CRITICAL = 60.0, 30.0, 15.0
HOLD = 2  # critical steps in a row before the alert
_PERCENTILES = np.linspace(0, 100, 101)
_REPLY_EVENTS = {
    "edit_rejected": "edit rejected",
    "traceback": "traceback",
    "command_error": "command error",
    "timeout": "timeout",
}


@dataclass
class Monitor:
    """What a monitor needs per checkpoint k, fitted on validation predictions."""

    quantiles: dict[int, list[float]] = field(default_factory=dict)
    calibration: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    critical: dict[int, float | None] = field(default_factory=dict)
    precision: dict[int, float | None] = field(default_factory=dict)
    recall: dict[int, float | None] = field(default_factory=dict)

    @classmethod
    def fit(
        cls, resolved: np.ndarray, score: np.ndarray, k: np.ndarray, target: float = 0.95, min_flagged: int = 30
    ) -> Monitor:
        """Fit the per-checkpoint pieces on validation runs.

        Args:
            resolved: Whether each validation run resolved its issue.
            score: The model's score for each, higher meaning more likely to resolve.
            k: The checkpoint of each.
            target: The failure rate a critical alert must reach on validation.
            min_flagged: The fewest validation runs a critical threshold may rest on.

        Returns:
            The fitted monitor.
        """
        resolved, score, k = np.asarray(resolved, bool), np.asarray(score, float), np.asarray(k)
        monitor = cls()
        for c in sorted(int(v) for v in np.unique(k)):
            y, p = resolved[k == c], score[k == c]
            monitor.quantiles[c] = np.quantile(p, _PERCENTILES / 100).tolist()
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y.astype(float))
            monitor.calibration[c] = (iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
            order = np.argsort(p, kind="stable")
            ranked, fails = p[order], np.cumsum(~y[order])
            count = np.arange(1, len(p) + 1)
            # Cut only after the last run of a score, so "score <= threshold" flags exactly `count` runs.
            last_of_score = np.r_[ranked[1:] != ranked[:-1], True]
            ok = last_of_score & (count >= min_flagged) & (count <= CRITICAL / 100 * len(p)) & (fails >= target * count)
            if ok.any():
                last = int(np.flatnonzero(ok)[-1])
                monitor.critical[c] = float(ranked[last])
                monitor.precision[c] = float(fails[last] / count[last])
                monitor.recall[c] = float(fails[last] / (~y).sum())
            else:
                monitor.critical[c] = monitor.precision[c] = monitor.recall[c] = None
        return monitor

    @classmethod
    def fit_predictions(cls, predictions: pd.DataFrame, gold: pd.DataFrame, **kwargs) -> Monitor:
        """Fit on a predictions file (``run_id``, ``k``, ``p_resolved``) and the matching gold rows."""
        merged = predictions[["run_id", "k", "p_resolved"]].merge(
            gold[["run_id", "k", "resolved"]], on=["run_id", "k"], validate="one_to_one"
        )
        if len(merged) != len(predictions):
            raise ValueError(f"only {len(merged)} of {len(predictions)} predictions have a gold checkpoint")
        return cls.fit(merged["resolved"], merged["p_resolved"], merged["k"], **kwargs)

    def _around(self, k: int) -> list[tuple[int, float]]:
        """The fitted checkpoints around step ``k`` with linear weights."""
        fitted = sorted(self.quantiles)
        if k <= fitted[0]:
            return [(fitted[0], 1.0)]
        if k >= fitted[-1]:
            return [(fitted[-1], 1.0)]
        low = max(c for c in fitted if c <= k)
        high = min(c for c in fitted if c >= k)
        if low == high:
            return [(low, 1.0)]
        t = (k - low) / (high - low)
        return [(low, 1 - t), (high, t)]

    def nearest(self, k: int) -> int:
        """The fitted checkpoint closest to step ``k`` (the earlier one on a tie)."""
        return min(self.quantiles, key=lambda c: (abs(c - k), c))

    def _health_at(self, score: float, c: int) -> float:
        return float(np.interp(score, self.quantiles[c], _PERCENTILES))

    def health(self, score: float, k: int) -> float:
        """The score's percentile among validation runs at step ``k``, 0 to 100."""
        return sum(w * self._health_at(score, c) for c, w in self._around(k))

    def calibrate(self, score: float, k: int) -> float:
        """The calibrated probability of resolving at step ``k``."""
        return sum(w * float(np.interp(score, *self.calibration[c])) for c, w in self._around(k))

    def critical_health(self, k: int) -> float:
        """The health at or under which step ``k`` is critical; -1 where it never is."""
        limits = []
        for c, w in self._around(k):
            threshold = self.critical[c]
            limits.append(w * (self._health_at(threshold, c) if threshold is not None else -1.0))
        return sum(limits)

    def alert_precision(self, k: int) -> float | None:
        """The validation failure rate behind an alert at step ``k`` (nearest checkpoint that alerts)."""
        for c in sorted(self.quantiles, key=lambda c: (abs(c - k), c)):
            if self.precision[c] is not None:
                return self.precision[c]
        return None

    def to_json(self, path: Path) -> None:
        """Write the monitor as JSON."""
        payload = {
            "quantiles": self.quantiles,
            "calibration": self.calibration,
            "critical": self.critical,
            "precision": self.precision,
            "recall": self.recall,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> Monitor:
        """Read a monitor written by :meth:`to_json`."""
        raw = json.loads(path.read_text())

        def by_k(values: dict) -> dict:
            return {int(c): v for c, v in values.items()}

        return cls(
            quantiles=by_k(raw["quantiles"]),
            calibration={c: (x, y) for c, (x, y) in by_k(raw["calibration"]).items()},
            critical=by_k(raw["critical"]),
            precision=by_k(raw["precision"]),
            recall=by_k(raw["recall"]),
        )

    def summary(self) -> str:
        """One line per checkpoint: the critical threshold's health, precision and recall."""
        lines = ["| k | critical at health <= | failed when flagged | failing runs flagged |", "|---|---|---|---|"]
        for c in sorted(self.quantiles):
            if self.critical[c] is None:
                lines.append(f"| {c} | never | - | - |")
            else:
                lines.append(
                    f"| {c} | {self.critical_health(c):.0f} | {self.precision[c]:.1%} | {self.recall[c]:.1%} |"
                )
        return "\n".join(lines)


def states(health: Sequence[float], critical: Sequence[bool], hold: int = HOLD) -> list[str]:
    """Each step's state: ``critical`` from ``hold`` critical steps in a row until health is back to
    ``WATCH``, else by health."""
    out, streak, alarm = [], 0, False
    for h, c in zip(health, critical, strict=True):
        streak = streak + 1 if c else 0
        if streak >= hold:
            alarm = True
        elif alarm and h >= WATCH:
            alarm = False
        if alarm:
            out.append("critical")
        elif h >= STABLE:
            out.append("stable")
        elif h >= WATCH:
            out.append("watch")
        else:
            out.append("low")
    return out


def prefix(row: Mapping, k: int) -> dict:
    """The first ``k`` steps of a checkpoint row: every per-step signal cut to ``k``."""
    cut = dict(row)
    for name in data.SIGNALS:
        cut[name] = list(row[name])[:k]
    cut["k"] = k
    return cut


def prefixes(row: Mapping, first: int = data.CHECKPOINTS[0]) -> list[dict]:
    """Every prefix of a run from ``first`` steps to its last visible step."""
    steps = len(row["action_kind"])
    if steps < first:
        raise ValueError(f"the run shows {steps} steps, fewer than the first replayed step {first}")
    return [prefix(row, k) for k in range(first, steps + 1)]


def events(cut: Mapping) -> list[str]:
    """What stood out in the latest step of a prefix: errors, repeats, edits of the same or a test file."""
    out = []
    if cut["repeated_action"][-1]:
        out.append("repeated command")
    reply = cut["observation_kind"][-1]
    if reply in _REPLY_EVENTS:
        out.append(_REPLY_EVENTS[reply])
    if cut["same_file_edit"][-1]:
        out.append("same file again")
    if cut["edit_target"][-1] == "test":
        out.append("edited a test")
    return out


def trace(cuts: Sequence[Mapping], scores: Sequence[float], monitor: Monitor) -> list[dict]:
    """The monitor's view of each prefix, from the model's scores of those prefixes.

    Returns:
        One dict per step: health, state, score, calibrated probability, the step's action, reply and
        events, the status description, and a recommendation once the state is critical.
    """
    health = [monitor.health(s, cut["k"]) for s, cut in zip(scores, cuts, strict=True)]
    flags = [h <= monitor.critical_health(cut["k"]) for h, cut in zip(health, cuts, strict=True)]
    out = []
    for cut, score, h, state in zip(cuts, scores, health, states(health, flags), strict=True):
        k = int(cut["k"])
        entry = {
            "step": k,
            "health": round(h),
            "state": state,
            "score": float(score),
            "p_calibrated": monitor.calibrate(score, k),
            "action": cut["action_kind"][-1],
            "reply": cut["observation_kind"][-1],
            "events": events(cut),
            "status": describe(cut),
        }
        if state == "critical":
            precision = monitor.alert_precision(k) or 0.0
            entry["recommendation"] = (
                f"Stop the agent to save credits: it fell into the zone where, on repositories the model "
                f"never saw, runs at this point failed {precision:.0%} of the time."
            )
        out.append(entry)
    return out


def replay(
    row: Mapping, scorer: Callable[[list[dict]], Sequence[float]], monitor: Monitor, first: int = data.CHECKPOINTS[0]
) -> list[dict]:
    """Score a run step by step, from step ``first`` to its last visible step, as a live monitor would.

    Args:
        row: The run's longest checkpoint row (``checkpoints.parquet``); every checkpoint of a run is a
            prefix of it, and it never shows the run's last step.
        scorer: The model: a list of prefix rows in, one pass/fail score per row out.
        monitor: The monitor fitted on validation predictions of the same model.
        first: The first step to score; the models train on checkpoints from step 5.

    Returns:
        One dict per step, as :func:`trace` makes them.
    """
    cuts = prefixes(row, first)
    return trace(cuts, scorer(cuts), monitor)


def sparkline(values: Sequence[float]) -> str:
    """A one-line chart of values from 0 to 100."""
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(len(bars) - 1, max(0, int(v / 100 * len(bars))))] for v in values)


def render(row: Mapping, steps: Sequence[Mapping]) -> str:
    """A replay as Markdown: the run, its health line, one row per step, and the alert if one fired."""
    outcome = "resolved" if row["resolved"] else "did not resolve"
    lines = [
        f"run {row['run_id']} · {row['repository']} · {row['instance_id']} · it ended {row['ending']} and {outcome} the issue",
        f"health {sparkline([s['health'] for s in steps])} (steps {steps[0]['step']}-{steps[-1]['step']})",
        "",
        "| step | health | state | action | reply | events |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s['step']} | {s['health']} | {s['state']} | {s['action']} | {s['reply']} | {', '.join(s['events'])} |"
        for s in steps
    ]
    alert = next((s for s in steps if "recommendation" in s), None)
    if alert is not None:
        lines += ["", f"step {alert['step']}: {alert['recommendation']}"]
    return "\n".join(lines)
