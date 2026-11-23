"""The numbers of results/history_monitor.md (written by scripts/history_monitor.py), for the site and figures."""

from __future__ import annotations

import re
from pathlib import Path

HISTORY_MD = Path(__file__).resolve().parents[1] / "results" / "history_monitor.md"


def _row(text: str, label: str) -> list[float]:
    line = next(line for line in text.splitlines() if line.startswith(f"| {label} |"))
    return [float(c.strip().rstrip(" %").strip("*").split(" ")[0]) for c in line.split("|")[2:-1]]


def history_numbers() -> dict:
    """Never-solved share, chance of success after m failed attempts, and step-5 scores with and without history."""
    text = HISTORY_MD.read_text()
    never = int(re.search(r"- (\d+)% of issues are never solved", text).group(1))
    share = int(re.search(r"explains (\d+)% of the variance", text).group(1))
    train = re.search(r"Training split: ([\d,]+) runs but ([\d,]+) issues, of which ([\d,]+) were ever solved", text)
    after = dict(zip((0, 1, 3, 5), _row(text, "chance of success"), strict=True))
    return {
        "never": never,
        "issue_share": share,
        "train_runs": train.group(1),
        "train_issues": train.group(2),
        "train_solved": train.group(3),
        "sibling": _row(text, "sibling success rate"),
        "after": after,
        "ours": _row(text, "our model alone"),
        "with": {m: _row(text, f"our model + {m} earlier attempt{'s' if m > 1 else ''}") for m in (1, 3, 5)},
    }
