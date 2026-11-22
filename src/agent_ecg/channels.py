"""Hunt for new per-step channels that improve early failure prediction, on CPU and without the test split.

A *channel* is one value per step of a checkpoint, computed only from steps 1..t at step t. Two kinds:

- **new information:** read from the raw text of each step (the agent's thought and command, the
  environment's reply), which the stored signals leave out: test results, whether a script stopped
  crashing, exception types, edit sizes. The text comes from the release's parquet shards, read once
  each; only the ``trajectory`` column is read, and the run id (``<shard>:<row>``) is a join key, never
  a feature. The final patch, the exit status, the test log and the outcome are never read.
- **re-expression:** computed from the stored per-step signals (:data:`agent_ecg.data.SIGNALS`), as a
  comparison group.

A proxy model scores a channel: the signals baseline's gradient-boosted trees on
:func:`agent_ecg.baselines.features`, with and without the channel's summaries (mean, mean of the last
five steps, last value, maximum), on identical seeded repository folds over train + validation.

- **Stage 0** (all repositories): reproduce the baseline, and run a noise control.
- **Stage A** (fold seed 0, a 30 % sample of repositories by default): screen every candidate; the top 8
  by mean AUROC gain at 5 / 10 / 15 steps advance if that mean is positive. The top candidate's shuffled
  control runs after it.
- **Stage B** (fold seeds 1 and 2, all repositories, pooled): a finalist passes if the paired
  repository-bootstrap 95 % interval of its AUROC gain is above 0 at two or more of 5 / 10 / 15 steps,
  no checkpoint's interval is entirely below 0, its same-issue pair change is at least -0.005 at three or
  more checkpoints, and its shuffled control does not pass. A second proxy reads raw per-step values.
- **Stage C:** greedy forward selection, only if two or more channels pass.

The thresholds are fixed in :data:`RULES` and were set before any result was seen. With ``--deadline``,
every stage stops starting new fits after that time and writes what it has; by default there is no limit.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from agent_ecg import data
from agent_ecg.baselines import _SEED, _weights, features

CHECKPOINTS = data.CHECKPOINTS
EARLY = (5, 10, 15)
FOLDS = 5
BOOTSTRAP = 1000
MAX_STEPS = max(CHECKPOINTS)
DEADLINE = None  # no limit by default; the hackathon run used --deadline 2026-11-22T09:00:00+01:00
RULES = {
    "advance": 8,  # finalists from stage A
    "ci_above_zero_at_least": 2,  # of the early checkpoints
    "same_issue_floor": -0.005,
    "same_issue_at_least": 3,  # of the five checkpoints
    "early_stop_precision": {5: 0.95, 10: 0.95, 15: 0.99, 20: 0.99, 30: 0.99},
    "baseline_far_off": 0.03,  # stage 0 stops if the baseline differs more than this from results/cv5
    "stage_c_needs_passing": 2,
    "screening_sample": 0.3,
}
_RECENT = 5
_ERRORS = frozenset({"traceback", "command_error", "edit_rejected", "timeout"})

NEW, REEXPRESSION = "new information", "re-expression"
DESCRIPTIONS: dict[str, str] = {}
CATEGORY: dict[str, str] = {}


# --- re-expressions of the stored signals (the comparison group) --------------------------------------

SignalChannel = Callable[[Mapping[str, Sequence]], list[float]]
SIGNAL_CHANNELS: dict[str, SignalChannel] = {}


def signal_channel(description: str) -> Callable[[SignalChannel], SignalChannel]:
    def register(fn: SignalChannel) -> SignalChannel:
        SIGNAL_CHANNELS[fn.__name__], DESCRIPTIONS[fn.__name__], CATEGORY[fn.__name__] = fn, description, REEXPRESSION
        return fn

    return register


def _streak(flags: Sequence[bool]) -> list[float]:
    """Length of the run of True values ending at each step."""
    out, run = [], 0
    for flag in flags:
        run = run + 1 if flag else 0
        out.append(float(run))
    return out


def _since(flags: Sequence[bool]) -> list[float]:
    """Steps since the last True value (0 on such a step); before the first one, the step number."""
    out, last = [], -1
    for t, flag in enumerate(flags):
        if flag:
            last = t
        out.append(float(t - last if last >= 0 else t + 1))
    return out


@signal_channel("consecutive steps whose reply is a traceback, command error, rejected edit or timeout")
def fail_streak(s: Mapping[str, Sequence]) -> list[float]:
    return _streak([str(kind) in _ERRORS for kind in s["observation_kind"]])


@signal_channel("steps since the last successful command (silent success, or output after a run or an edit)")
def steps_since_success(s: Mapping[str, Sequence]) -> list[float]:
    ok = [
        str(r) == "silent_success" or (str(r) == "output" and str(a) in ("run", "edit"))
        for a, r in zip(s["action_kind"], s["observation_kind"], strict=True)
    ]
    return _since(ok)


@signal_channel("conversation length relative to its length after the first step")
def context_ratio(s: Mapping[str, Sequence]) -> list[float]:
    context = [float(v) for v in s["context_chars"]]
    return [c / max(context[0], 1.0) for c in context]


# --- new information from the text of each step -------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """What a monitor sees of one step: the agent's thought and command, and the environment's reply."""

    thought: str
    command: str
    reply: str


TextChannel = Callable[[Sequence[Step]], list[float]]
TEXT_CHANNELS: dict[str, TextChannel] = {}


def text_channel(description: str) -> Callable[[TextChannel], TextChannel]:
    def register(fn: TextChannel) -> TextChannel:
        TEXT_CHANNELS[fn.__name__], DESCRIPTIONS[fn.__name__], CATEGORY[fn.__name__] = fn, description, NEW
        return fn

    return register


_PYTEST_LINE = re.compile(
    r"^=*\s*((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?)(?:, )?)+) in [\d.]+s", re.M
)
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?)")
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in", re.M)
_UNITTEST_FAILED = re.compile(r"^FAILED \(([^)]*)\)", re.M)
_TRACEBACK = "Traceback (most recent call last)"
_EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))\b", re.M)
_EDIT = re.compile(r"^edit\s+(\d+):(\d+)\s*\n?(.*?)\n?end_of_edit\s*$", re.S)
_OPEN_FILE = re.compile(r"\(Open file: ([^)\n]*)\)")
_SHOWN_LINE = re.compile(r"^(\d+):", re.M)
_SUCCESS = re.compile(r"completed successfully", re.I)
_RUN_WORDS = ("python", "python3", "pytest")


def test_counts(reply: str) -> tuple[int, int] | None:
    """Passed and failed (failures plus errors) tests from a pytest or unittest summary, if the reply has one."""
    found = _PYTEST_LINE.findall(reply)
    if found:
        counts = Counter()
        for word_count in _PYTEST_COUNT.findall(found[-1]):
            counts["passed" if word_count[1] == "passed" else "failed"] += int(word_count[0])
        return counts["passed"], counts["failed"]
    ran = _UNITTEST_RAN.findall(reply)
    if ran:
        total, failed = int(ran[-1]), 0
        detail = _UNITTEST_FAILED.findall(reply)
        if detail:
            failed = sum(int(n) for n in re.findall(r"(?:failures|errors)=(\d+)", detail[-1]))
        return max(total - failed, 0), failed
    return None


def _is_run(command: str) -> bool:
    words = command.split(maxsplit=1)
    return bool(words) and words[0] in _RUN_WORDS


def _exception(reply: str) -> str | None:
    if _TRACEBACK not in reply:
        return None
    found = _EXCEPTION.findall(reply)
    return found[-1] if found else None


@text_channel("failed tests (failures and errors) in a test summary at this step")
def tests_failed(steps: Sequence[Step]) -> list[float]:
    return [float(c[1]) if (c := test_counts(s.reply)) else 0.0 for s in steps]


@text_channel("passed tests in a test summary at this step")
def tests_passed(steps: Sequence[Step]) -> list[float]:
    return [float(c[0]) if (c := test_counts(s.reply)) else 0.0 for s in steps]


@text_channel("1 when a test summary reports passes and no failures")
def tests_all_passed(steps: Sequence[Step]) -> list[float]:
    return [float(bool(c) and c[0] > 0 and c[1] == 0) if (c := test_counts(s.reply)) else 0.0 for s in steps]


@text_channel("change in failed tests since the previous test summary (negative: fewer failures)")
def tests_failed_delta(steps: Sequence[Step]) -> list[float]:
    out, previous = [], None
    for s in steps:
        counts = test_counts(s.reply)
        if counts is None:
            out.append(0.0)
            continue
        out.append(float(counts[1] - previous) if previous is not None else 0.0)
        previous = counts[1]
    return out


def _reruns(steps: Sequence[Step]) -> Iterator[tuple[bool, bool | None]]:
    """For each step: whether it reruns a script after an edit, and whether that script crashed last time
    (None when it does not)."""
    last_crash: dict[str, bool] = {}
    edited_since: dict[str, bool] = {}
    for s in steps:
        if s.command.startswith("edit"):
            edited_since = dict.fromkeys(edited_since, True)
        if _is_run(s.command):
            crashed = _TRACEBACK in s.reply
            before = last_crash.get(s.command) if edited_since.get(s.command) else None
            yield crashed, before
            last_crash[s.command], edited_since[s.command] = crashed, False
        else:
            yield False, None


@text_channel("1 when a script that crashed runs cleanly after an edit")
def script_fixed(steps: Sequence[Step]) -> list[float]:
    return [float(before is True and not crashed) for crashed, before in _reruns(steps)]


@text_channel("1 when a script that crashed still crashes after an edit")
def script_still_crashing(steps: Sequence[Step]) -> list[float]:
    return [float(before is True and crashed) for crashed, before in _reruns(steps)]


@text_channel('1 when a run of code prints "completed successfully"')
def success_marker(steps: Sequence[Step]) -> list[float]:
    return [float(_is_run(s.command) and bool(_SUCCESS.search(s.reply))) for s in steps]


def _exception_events(steps: Sequence[Step]) -> Iterator[tuple[bool, bool]]:
    """For each step: (same exception type as the last traceback, a different one)."""
    last = None
    for s in steps:
        current = _exception(s.reply)
        yield (current is not None and current == last, current is not None and last is not None and current != last)
        if current is not None:
            last = current


@text_channel("1 when a traceback raises a different exception type than the last one")
def exception_changed(steps: Sequence[Step]) -> list[float]:
    return [float(changed) for _, changed in _exception_events(steps)]


@text_channel("1 when a traceback raises the same exception type as the last one")
def same_exception(steps: Sequence[Step]) -> list[float]:
    return [float(same) for same, _ in _exception_events(steps)]


def _edit(command: str) -> tuple[int, int, int] | None:
    """An edit's first and last replaced line and how many lines it writes."""
    match = _EDIT.match(command)
    if match is None:
        return None
    first, last, body = int(match.group(1)), int(match.group(2)), match.group(3)
    return first, last, (len(body.split("\n")) if body.strip() else 0)


@text_channel("lines an edit replaces")
def edit_span(steps: Sequence[Step]) -> list[float]:
    return [float(max(e[1] - e[0] + 1, 0)) if (e := _edit(s.command)) else 0.0 for s in steps]


@text_channel("lines an edit writes")
def edit_size(steps: Sequence[Step]) -> list[float]:
    return [float(e[2]) if (e := _edit(s.command)) else 0.0 for s in steps]


@text_channel("1 when an edit replaces lines of the open file that no earlier reply showed")
def edit_unviewed(steps: Sequence[Step]) -> list[float]:
    out, current, seen = [], None, {}
    for s in steps:
        edit = _edit(s.command)
        # An edit changes the file open before it: the one the previous reply's footer named.
        unseen = (
            edit is not None
            and current is not None
            and any(line not in seen.get(current, ()) for line in range(edit[0], edit[1] + 1))
        )
        out.append(float(unseen))
        footer = _OPEN_FILE.findall(s.reply)
        if footer:
            current = None if footer[-1].strip() == "n/a" else footer[-1].strip()
        if current is not None:
            seen.setdefault(current, set()).update(int(n) for n in _SHOWN_LINE.findall(s.reply))
    return out


CANDIDATES = (*TEXT_CHANNELS, *SIGNAL_CHANNELS)


# --- reading the text -----------------------------------------------------------------------------------

DATASET, REVISION = data.RAW_DATASET, data.RAW_REVISION
TEXT_COLUMNS = ["trajectory"]  # the only column read from the shards


def parse_steps(trajectory: list | str) -> list[Step]:
    """The first 30 steps of a release trajectory, split as the connector splits them."""
    from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import split_message

    messages = json.loads(trajectory) if isinstance(trajectory, str) else trajectory
    body = messages[2:]  # system prompt and issue
    steps = []
    for t in range(min((len(body) + 1) // 2, MAX_STEPS)):
        thought, command = split_message(body[2 * t].get("text") or "")
        reply = (body[2 * t + 1].get("text") or "") if 2 * t + 1 < len(body) else ""
        steps.append(Step(thought, command, reply))
    return steps


def text_values(steps: Sequence[Step]) -> np.ndarray:
    """Every text channel of a run's steps: [channels, 30], NaN after the run's last step."""
    out = np.full((len(TEXT_CHANNELS), MAX_STEPS), np.nan, dtype=np.float32)
    for i, fn in enumerate(TEXT_CHANNELS.values()):
        values = fn(steps)
        out[i, : len(values)] = values
    return out


def _shard_values(path: str, stem: str, rows: list[int]) -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq

    wanted, ids, values = set(rows), [], []
    parquet, offset = pq.ParquetFile(path), 0
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if wanted & set(range(offset, offset + size)):
            trajectories = parquet.read_row_group(group, columns=TEXT_COLUMNS).column("trajectory").to_pylist()
            for i, trajectory in enumerate(trajectories):
                if offset + i in wanted:
                    ids.append(f"{stem}:{offset + i}")
                    values.append(text_values(parse_steps(trajectory)))
        offset += size
    return ids, np.stack(values) if values else np.zeros((0, len(TEXT_CHANNELS), MAX_STEPS), np.float32)


def read_text_channels(run_ids: Sequence[str], raw: Path | None, workers: int) -> tuple[list[str], np.ndarray]:
    """Text channels for the given runs, reading each shard once, ``workers`` shards at a time."""
    from huggingface_hub import hf_hub_download

    by_shard: dict[str, list[int]] = {}
    for run_id in run_ids:
        stem, _, row = run_id.partition(":")
        by_shard.setdefault(stem, []).append(int(row))
    jobs = []
    for stem, rows in sorted(by_shard.items()):
        path = (raw / f"{stem}.parquet") if raw else None
        if path is None or not path.is_file():
            path = Path(hf_hub_download(DATASET, f"data/{stem}.parquet", repo_type="dataset", revision=REVISION))
        jobs.append((str(path), stem, rows))
    ids, blocks = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for shard_ids, block in pool.map(_shard_values, *zip(*jobs, strict=True)):
            ids += shard_ids
            blocks.append(block)
    return ids, np.concatenate(blocks)


# --- the data the hunt may see -------------------------------------------------------------------------


def development_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Train and validation checkpoints only: the test split is never used."""
    return table[table["split"] != "test"].reset_index(drop=True)


def signals_of(row: Mapping) -> dict[str, list]:
    """The per-step signals of a checkpoint row, and nothing else, as plain lists."""
    return {name: list(row[name]) for name in data.SIGNALS}


def signal_series(table: pd.DataFrame, name: str) -> list[list[float]]:
    """One signal channel's per-step values for every checkpoint."""
    fn = SIGNAL_CHANNELS[name]
    columns = [table[n].to_numpy() for n in data.SIGNALS]
    return [fn(dict(zip(data.SIGNALS, (list(c[i]) for c in columns), strict=True))) for i in range(len(table))]


def summarise(name: str, values: list[list[float]]) -> pd.DataFrame:
    """The summaries a candidate adds to the proxy: mean, mean of the last five steps, last value, maximum."""
    rows = np.array([(np.mean(v), np.mean(v[-_RECENT:]), v[-1], np.max(v)) for v in values], dtype=np.float32).reshape(
        -1, 4
    )
    return pd.DataFrame(rows, columns=[f"{name}__{s}" for s in ("mean", "recent", "last", "max")])


def noise_series(table: pd.DataFrame, seed: int) -> list[list[float]]:
    """Negative control: i.i.d. uniform values, one per step."""
    rng = np.random.default_rng(seed)
    return [rng.random(int(k)).tolist() for k in table["k"]]


def shuffle_rows(frame: pd.DataFrame, k: np.ndarray, seed: int) -> pd.DataFrame:
    """Negative control: move each checkpoint's values to another checkpoint with the same k."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    for c in np.unique(k):
        idx = np.flatnonzero(k == c)
        order[idx] = rng.permutation(idx)
    return frame.iloc[order].reset_index(drop=True)


def fold_assignment(repositories: np.ndarray, seed: int, folds: int = FOLDS) -> np.ndarray:
    """A seeded random assignment of whole repositories to folds."""
    unique, inverse = np.unique(repositories, return_inverse=True)
    fold_of_repo = np.empty(len(unique), dtype=int)
    fold_of_repo[np.random.default_rng(seed).permutation(len(unique))] = np.arange(len(unique)) % folds
    return fold_of_repo[inverse]


def out_of_fold(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, fold: np.ndarray, categorical: np.ndarray | None = None
) -> np.ndarray:
    """Out-of-fold P(resolved) from the signals baseline's model, one fit per fold."""
    out = np.empty(len(y))
    for f in range(FOLDS):
        train, test = fold != f, fold == f
        kwargs = {} if categorical is None else {"categorical_features": categorical}
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=_SEED, **kwargs)
        clf.fit(X[train], y[train], sample_weight=w[train])
        out[test] = clf.predict_proba(X[test])[:, 1]
    return out


# --- scoring --------------------------------------------------------------------------------------------


def _weighted_auc(inverse: np.ndarray, n_unique: int, y: np.ndarray, w: np.ndarray) -> float:
    """AUROC with row weights (repository multiplicities), from scores ranked once; ties count half."""
    pos = np.bincount(inverse, weights=w * y, minlength=n_unique)
    neg = np.bincount(inverse, weights=w * (1.0 - y), minlength=n_unique)
    below = np.cumsum(neg) - neg
    den = pos.sum() * neg.sum()
    return float((pos * (below + 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def early_stop_recall(resolved: np.ndarray, p: np.ndarray, precision: float) -> float:
    """Largest share of failing runs flaggable as "will fail" (lowest p) with the flagged group failing
    at least ``precision`` of the time."""
    fail = ~np.asarray(resolved, bool)
    order = np.argsort(p, kind="stable")
    ranked, fails = p[order], np.cumsum(fail[order])
    count = np.arange(1, len(p) + 1)
    last_of_score = np.r_[ranked[1:] != ranked[:-1], True]
    ok = last_of_score & (fails >= precision * count)
    return float(fails[np.flatnonzero(ok)[-1]] / fail.sum()) if ok.any() and fail.any() else 0.0


class Checkpoint:
    """Everything needed to score predictions at one checkpoint k, with repository resamples."""

    def __init__(self, table: pd.DataFrame, k: int, n_boot: int = BOOTSTRAP) -> None:
        self.mask = table["k"].to_numpy() == k
        sub = table[self.mask]
        self.k = k
        self.y = sub["resolved"].to_numpy(bool)
        self.yf = self.y.astype(float)
        repos, self.repo = np.unique(sub["repository"].to_numpy(), return_inverse=True)
        rng = np.random.default_rng(1000 + k)
        draws = rng.integers(len(repos), size=(n_boot, len(repos)))
        self.counts = np.stack([np.bincount(d, minlength=len(repos)) for d in draws]).astype(float)
        # Issues with both outcomes, for same-issue pairs; an issue lives in one repository.
        issues = sub["instance_id"].to_numpy()
        self.pairs_idx: list[tuple[np.ndarray, np.ndarray]] = []
        issue_repo, pairs = [], []
        for _, idx in pd.Series(np.arange(len(sub))).groupby(issues, sort=False):
            idx = idx.to_numpy()
            good, bad = idx[self.y[idx]], idx[~self.y[idx]]
            if len(good) and len(bad):
                self.pairs_idx.append((good, bad))
                issue_repo.append(self.repo[idx[0]])
                pairs.append(len(good) * len(bad))
        self.issue_repo, self.pairs = np.array(issue_repo, dtype=int), np.array(pairs, float)
        self._memo: dict[tuple[str, int], tuple[np.ndarray, object]] = {}

    def _remember(self, what: str, p: np.ndarray, make: Callable[[], object]):
        # Keyed by the array's identity; the array is kept alongside so its id cannot be reused.
        key = (what, id(p))
        if key not in self._memo:
            self._memo[key] = (p, make())
        return self._memo[key][1]

    def auroc(self, p: np.ndarray) -> float:
        return float(roc_auc_score(self.y, p[self.mask]))

    def boot_auroc(self, p: np.ndarray) -> np.ndarray:
        """AUROC under every repository resample."""

        def make() -> np.ndarray:
            unique, inverse = np.unique(p[self.mask], return_inverse=True)
            return np.array([_weighted_auc(inverse, len(unique), self.yf, c[self.repo]) for c in self.counts])

        return self._remember("boot_auroc", p, make)

    def issue_wins(self, p: np.ndarray) -> np.ndarray:
        def make() -> np.ndarray:
            q = p[self.mask]
            wins = []
            for good, bad in self.pairs_idx:
                diff = q[good][:, None] - q[bad][None, :]
                wins.append(float((diff > 0).sum() + 0.5 * (diff == 0).sum()))
            return np.array(wins)

        return self._remember("issue_wins", p, make)

    def same_issue(self, p: np.ndarray) -> float:
        return float(self.issue_wins(p).sum() / self.pairs.sum()) if len(self.pairs) else float("nan")

    def boot_same_issue(self, p: np.ndarray) -> np.ndarray:
        weights = self.counts[:, self.issue_repo]
        return (weights @ self.issue_wins(p)) / (weights @ self.pairs)

    def fold_aurocs(self, p: np.ndarray, fold: np.ndarray) -> list[float]:
        q, f = p[self.mask], fold[self.mask]
        return [
            float(roc_auc_score(self.y[f == i], q[f == i]))
            for i in range(FOLDS)
            if 0 < self.y[f == i].sum() < (f == i).sum()
        ]

    def recall(self, p: np.ndarray) -> float:
        return early_stop_recall(self.y, p[self.mask], RULES["early_stop_precision"][self.k])


def compare(
    points: dict[int, Checkpoint],
    candidate: list[np.ndarray],
    baseline: list[np.ndarray],
    folds: list[np.ndarray],
    with_ci: bool = True,
) -> dict[int, dict]:
    """Candidate vs baseline per checkpoint, pooled over seeds (one OOF vector per seed in each list):
    point gains are means over seeds, intervals come from resampling the same repositories for every
    seed and both models."""
    out = {}
    for k, cp in points.items():
        pairs = list(zip(candidate, baseline, strict=True))
        entry = {
            "auroc_baseline": float(np.mean([cp.auroc(b) for b in baseline])),
            "auroc_candidate": float(np.mean([cp.auroc(c) for c in candidate])),
            "gain": float(np.mean([cp.auroc(c) - cp.auroc(b) for c, b in pairs])),
            "fold_sd": float(np.mean([np.std(cp.fold_aurocs(c, f)) for c, f in zip(candidate, folds, strict=True)])),
            "same_issue_baseline": float(np.mean([cp.same_issue(b) for b in baseline])),
            "same_issue_change": float(np.mean([cp.same_issue(c) - cp.same_issue(b) for c, b in pairs])),
            "early_stop_recall_baseline": float(np.mean([cp.recall(b) for b in baseline])),
            "early_stop_recall_change": float(np.mean([cp.recall(c) - cp.recall(b) for c, b in pairs])),
        }
        if with_ci:
            boot = np.mean([cp.boot_auroc(c) - cp.boot_auroc(b) for c, b in pairs], axis=0)
            boot_pairs = np.mean([cp.boot_same_issue(c) - cp.boot_same_issue(b) for c, b in pairs], axis=0)
            entry["gain_95"] = [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))]
            entry["same_issue_change_95"] = [
                float(np.nanquantile(boot_pairs, 0.025)),
                float(np.nanquantile(boot_pairs, 0.975)),
            ]
        out[k] = entry
    return out


def auroc_rule(result: dict[int, dict]) -> bool:
    """The stage B AUROC part: interval above 0 at enough early checkpoints, and nowhere entirely below 0."""
    above = sum(result[k]["gain_95"][0] > 0 for k in EARLY)
    below = any(entry["gain_95"][1] < 0 for entry in result.values())
    return above >= RULES["ci_above_zero_at_least"] and not below


def same_issue_rule(result: dict[int, dict]) -> bool:
    held = sum(entry["same_issue_change"] >= RULES["same_issue_floor"] for entry in result.values())
    return held >= RULES["same_issue_at_least"]


def stage_b_rule(result: dict[int, dict]) -> bool:
    """Every part of the stage B rule except the shuffled control, which needs its own result."""
    return auroc_rule(result) and same_issue_rule(result)


def early_mean_gain(result: dict[int, dict]) -> float:
    return float(np.mean([result[k]["gain"] for k in EARLY]))


# --- the raw-series proxy -------------------------------------------------------------------------------

_ENUMS = {
    "action_kind": ("edit", "create", "run", "read", "search", "submit", "other"),
    "observation_kind": ("edit_rejected", "timeout", "traceback", "command_error", "silent_success", "output"),
    "edit_target": ("none", "source", "test", "own_script"),
}


def raw_matrix(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw per-step values of the baseline signals, padded to 30 steps with NaN.

    Returns:
        The matrix and a mask of its categorical columns (the enum signals as codes).
    """
    blocks, categorical = [], []
    for name in data.SIGNALS:
        block = np.full((len(table), MAX_STEPS), np.nan, dtype=np.float32)
        codes = {v: i for i, v in enumerate(_ENUMS[name])} if name in _ENUMS else None
        for i, values in enumerate(table[name].to_numpy()):
            vals = [codes[str(v)] for v in values] if codes else [float(v) for v in values]
            block[i, : len(vals)] = vals
        blocks.append(block)
        categorical += [codes is not None] * MAX_STEPS
    return np.hstack(blocks), np.array(categorical)


def raw_block(values: list[list[float]]) -> np.ndarray:
    """One channel's per-step values, padded to 30 steps with NaN."""
    block = np.full((len(values), MAX_STEPS), np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        block[i, : len(v)] = v
    return block


# --- proposals for the TSLM -----------------------------------------------------------------------------

# How a passing channel would enter opentslm_data.time_series(): 0/1 stays 0/1; anything else gets a
# fixed scale named in the label, never per-run statistics.
PROPOSALS: dict[str, tuple[str, Callable[[float], float]]] = {
    "fail_streak": ("Replies in a row with an error, divided by 5, capped at 1:", lambda v: min(v / 5, 1.0)),
    "steps_since_success": (
        "Steps since the last successful command, divided by 10, capped at 1:",
        lambda v: min(v / 10, 1.0),
    ),
    "context_ratio": (
        "Conversation length over its length after step 1, divided by 10, capped at 1:",
        lambda v: min(v / 10, 1.0),
    ),
    "tests_failed": ("Failed tests in a test summary, divided by 10, capped at 1:", lambda v: min(v / 10, 1.0)),
    "tests_passed": ("Passed tests in a test summary, divided by 50, capped at 1:", lambda v: min(v / 50, 1.0)),
    "tests_all_passed": ("1 marks a test run where every test passed:", lambda v: v),
    "tests_failed_delta": (
        "Change in failed tests since the previous test run, divided by 10, clipped to -1 and 1:",
        lambda v: max(-1.0, min(v / 10, 1.0)),
    ),
    "script_fixed": ("1 marks a rerun after an edit of a script that crashed and now runs cleanly:", lambda v: v),
    "script_still_crashing": ("1 marks a rerun after an edit of a script that still crashes:", lambda v: v),
    "success_marker": ('1 marks a run of code that printed "completed successfully":', lambda v: v),
    "exception_changed": ("1 marks a traceback with a different exception type than the last one:", lambda v: v),
    "same_exception": ("1 marks a traceback with the same exception type as the last one:", lambda v: v),
    "edit_span": ("Lines an edit replaces, divided by 50, capped at 1:", lambda v: min(v / 50, 1.0)),
    "edit_size": ("Lines an edit writes, divided by 50, capped at 1:", lambda v: min(v / 50, 1.0)),
    "edit_unviewed": ("1 marks an edit of lines the agent had not seen:", lambda v: v),
}


def token_cost(label: str) -> dict[str, int]:
    """Extra prompt tokens per sample for one channel: its label plus one token per four values."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        from agent_ecg.opentslm_data import TOKENIZER

        tokenizer = Tokenizer.from_file(hf_hub_download(*TOKENIZER))
        label_tokens = len(tokenizer.encode(label, add_special_tokens=False).ids)
    except Exception:  # offline: about 1.3 tokens per word
        label_tokens = round(1.3 * len(label.split()))
    return {"label": label_tokens, **{f"k={k}": label_tokens + math.ceil(k / 4) for k in CHECKPOINTS}}


# --- the stages -----------------------------------------------------------------------------------------


class DeadlineReached(Exception):
    pass


class Hunt:
    """Shared state of one run over a set of repositories: rows, cached features and the work folder."""

    def __init__(
        self,
        table: pd.DataFrame,
        text: tuple[list[str], np.ndarray],
        work: Path,
        deadline: datetime | None,
        started: float,
        sample: float | None = None,
    ) -> None:
        self.started, self.work, self.deadline, self.sample = started, work, deadline, sample
        if sample:
            repos = np.unique(table["repository"])
            keep = np.random.default_rng(7).choice(repos, size=int(len(repos) * sample), replace=False)
            table = table[table["repository"].isin(keep)].reset_index(drop=True)
        self.table = table
        self.y = table["resolved"].to_numpy(bool)
        self.k = table["k"].to_numpy()
        self.w = _weights(table)
        index = {run_id: i for i, run_id in enumerate(text[0])}
        self.text_rows = np.array([index[r] for r in table["run_id"]])
        self.text = text[1]
        self.points = {k: Checkpoint(table, k) for k in CHECKPOINTS}
        self.log(f"{len(table):,} checkpoints from {table['repository'].nunique()} repositories ({self.tag or 'all'})")
        self.base = self._cached("baseline_features", lambda: features(table).astype(np.float32))
        self._summaries: dict[str, pd.DataFrame] = {}
        self._oof: dict[tuple, np.ndarray] = {}

    @property
    def tag(self) -> str:
        return f"sample{self.sample}" if self.sample else ""

    def log(self, message: str) -> None:
        print(f"[{(time.perf_counter() - self.started) / 60:6.1f} min] {message}", flush=True)

    def _cached(self, name: str, make: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.work / f"{name}{'-' + self.tag if self.tag else ''}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if len(frame) == len(self.table):
                return frame
        frame = make().reset_index(drop=True)
        frame.to_parquet(path, index=False)
        return frame

    def series(self, name: str) -> list[list[float]]:
        if name in SIGNAL_CHANNELS:
            return signal_series(self.table, name)
        c = list(TEXT_CHANNELS).index(name)
        return [self.text[r, c, :k].tolist() for r, k in zip(self.text_rows, self.k, strict=True)]

    def summaries(self, name: str) -> pd.DataFrame:
        if name not in self._summaries:
            self._summaries[name] = self._cached(f"summary_{name}", lambda: summarise(name, self.series(name)))
        return self._summaries[name]

    def folds(self, seed: int) -> np.ndarray:
        fold = fold_assignment(self.table["repository"].to_numpy(), seed)
        path = self.work / f"folds_seed{seed}{'-' + self.tag if self.tag else ''}.parquet"
        if not path.exists():
            pd.DataFrame({"repository": self.table["repository"], "fold": fold}).drop_duplicates().to_parquet(path)
        return fold

    def oof(
        self, key: tuple, X: Callable[[], np.ndarray], seed: int, categorical: np.ndarray | None = None
    ) -> np.ndarray:
        """OOF predictions for a feature set and fold seed, cached in memory and on disk."""
        if key in self._oof:
            return self._oof[key]
        name = "-".join(str(part) for part in key) + (f"-{self.tag}" if self.tag else "")
        path = self.work / f"oof_{name}.npy"
        if path.exists():
            p = np.load(path)
        else:
            if self.deadline is not None and datetime.now().astimezone() > self.deadline:
                raise DeadlineReached(name)
            begun = time.perf_counter()
            p = out_of_fold(X(), self.y, self.w, self.folds(seed), categorical)
            np.save(path, p)
            self.log(f"fitted {name} ({(time.perf_counter() - begun) / 60:.1f} min)")
        self._oof[key] = p
        return p

    def X(self, *frames: pd.DataFrame) -> np.ndarray:
        return pd.concat([self.base, *frames], axis=1).to_numpy(np.float32)

    def baseline(self, seed: int) -> np.ndarray:
        return self.oof(("baseline", seed), lambda: self.base.to_numpy(np.float32), seed)

    def candidate(self, names: Sequence[str], seed: int) -> np.ndarray:
        return self.oof(("+".join(names), seed), lambda: self.X(*(self.summaries(n) for n in names)), seed)

    def shuffled(self, name: str, seed: int) -> np.ndarray:
        return self.oof(
            (f"shuffled_{name}", seed), lambda: self.X(shuffle_rows(self.summaries(name), self.k, 10_000 + seed)), seed
        )

    def noise(self, seed: int) -> np.ndarray:
        frame = self._cached("summary_noise", lambda: summarise("noise", noise_series(self.table, seed=99)))
        return self.oof(("noise", seed), lambda: self.X(frame), seed)


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def stage0(h: Hunt, report: dict) -> bool:
    base = h.baseline(0)
    cv_path = Path("results/cv5.json")
    cv = json.loads(cv_path.read_text())["signals"] if cv_path.exists() else {}
    rows = {}
    for k, cp in h.points.items():
        ref = cv.get(str(k), {}).get("auroc")
        rows[k] = {"auroc": cp.auroc(base), "fold_sd": float(np.std(cp.fold_aurocs(base, h.folds(0)))), "cv5": ref}
    far = [k for k, r in rows.items() if r["cv5"] and abs(r["auroc"] - r["cv5"]) > RULES["baseline_far_off"]]
    noise = compare(h.points, [h.noise(0)], [base], [h.folds(0)])
    report["stage0"] = {
        "baseline": rows,
        "baseline_far_off_at": far,
        "noise": noise,
        "noise_passes": stage_b_rule(noise),
    }
    h.log(
        f"stage 0: baseline AUROC {[round(r['auroc'], 3) for r in rows.values()]}; noise passes: {stage_b_rule(noise)}"
    )
    return not far and not stage_b_rule(noise)


def stage_a(h: Hunt, report: dict, save: Callable[[], None]) -> list[str]:
    base, fold = h.baseline(0), h.folds(0)
    results = report.setdefault("stage_a", {}).setdefault("results", {})
    for name in CANDIDATES:
        if name not in results:
            results[name] = compare(h.points, [h.candidate([name], 0)], [base], [fold], with_ci=False)
            h.log(f"stage A {name}: mean gain at 5-15 steps {early_mean_gain(results[name]):+.4f}")
            save()
    ranked = sorted(CANDIDATES, key=lambda n: early_mean_gain(results[n]), reverse=True)
    finalists = [n for n in ranked[: RULES["advance"]] if early_mean_gain(results[n]) > 0]
    shuffled = compare(h.points, [h.shuffled(ranked[0], 0)], [base], [fold])
    report["stage_a"] |= {
        "sample": h.sample,
        "ranked": ranked,
        "finalists": finalists,
        "top_shuffled": {"channel": ranked[0], "result": shuffled, "passes": stage_b_rule(shuffled)},
    }
    h.log(f"stage A finalists: {finalists}; shuffled {ranked[0]} passes: {stage_b_rule(shuffled)}")
    save()
    return finalists


def stage_b(h: Hunt, report: dict, finalists: list[str], save: Callable[[], None]) -> list[str]:
    seeds = (1, 2)
    section = report.setdefault("stage_b", {"seeds": list(seeds), "raw_proxy_seed": 1, "results": {}})
    bases, folds = [h.baseline(s) for s in seeds], [h.folds(s) for s in seeds]
    raw_base_X, raw_cat = None, None
    for name in finalists:
        if name in section["results"]:
            continue
        summary = compare(h.points, [h.candidate([name], s) for s in seeds], bases, folds)
        shuffled = compare(h.points, [h.shuffled(name, s) for s in seeds], bases, folds)
        if raw_base_X is None:
            raw_base_X, raw_cat = raw_matrix(h.table)
        base_X = raw_base_X
        raw_base = h.oof(("raw_baseline", 1), lambda base_X=base_X: base_X, 1, raw_cat)
        cat = np.r_[raw_cat, np.zeros(MAX_STEPS, dtype=bool)]
        raw_c = h.oof(
            (f"raw_{name}", 1),
            lambda base_X=base_X, name=name: np.hstack([base_X, raw_block(h.series(name))]),
            1,
            cat,
        )
        raw = compare(h.points, [raw_c], [raw_base], [folds[0]])
        passes = stage_b_rule(summary) and not stage_b_rule(shuffled)
        section["results"][name] = {
            "summary": summary,
            "shuffled": shuffled,
            "shuffled_passes": stage_b_rule(shuffled),
            "auroc_rule": auroc_rule(summary),
            "same_issue_rule": same_issue_rule(summary),
            "passes": passes,
            "raw": raw,
            "raw_early_mean_gain": early_mean_gain(raw),
            "summary_only": passes and not any(raw[k]["gain_95"][0] > 0 for k in EARLY),
        }
        h.log(
            f"stage B {name}: passes {passes} (AUROC rule {auroc_rule(summary)}, same-issue rule "
            f"{same_issue_rule(summary)}, shuffled passes {stage_b_rule(shuffled)})"
        )
        save()
    passing = [n for n in finalists if section["results"].get(n, {}).get("passes")]
    passing.sort(key=lambda n: early_mean_gain(section["results"][n]["summary"]), reverse=True)
    section["passing"] = passing
    section["complete"] = all(n in section["results"] for n in finalists)
    save()
    return passing


def stage_c(h: Hunt, report: dict, passing: list[str], save: Callable[[], None]) -> list[str]:
    if len(passing) < RULES["stage_c_needs_passing"]:
        report["stage_c"] = {
            "skipped": f"{len(passing)} channel(s) passed; stage C needs 2 or more",
            "final_set": passing,
        }
        save()
        return passing
    seeds = (1, 2)
    folds = [h.folds(s) for s in seeds]
    chosen: list[str] = []
    steps = []
    for name in passing:
        before = [h.candidate(chosen, s) if chosen else h.baseline(s) for s in seeds]
        result = compare(h.points, [h.candidate([*chosen, name], s) for s in seeds], before, folds)
        kept = auroc_rule(result)
        steps.append({"channel": name, "on_top_of": list(chosen), "result": result, "kept": kept})
        if kept:
            chosen.append(name)
        h.log(f"stage C {name}: kept {kept}")
    final = compare(h.points, [h.candidate(chosen, s) for s in seeds], [h.baseline(s) for s in seeds], folds)
    report["stage_c"] = {"steps": steps, "final_set": chosen, "final_vs_baseline": final}
    save()
    return chosen


# --- the report -----------------------------------------------------------------------------------------


def _gains(result: dict) -> str:
    return ", ".join(f"k={k} {v['gain']:+.3f} {_ci(v['gain_95'])}" for k, v in result.items())


def to_markdown(report: dict) -> str:
    meta = report["meta"]
    lines = [
        "# Channel hunt",
        "",
        f"Train + validation only ({meta['checkpoints']:,} checkpoints, {meta['repositories']} repositories); the test "
        "split is never read. Inputs: the stored per-step signals, plus the text of steps 1..t (thought, command, "
        "reply) read from the release shards, with the run id as a join key only. Proxy: the signals baseline's "
        "gradient-boosted trees, with and without a channel's summaries (mean, mean of the last 5 steps, last value, "
        "maximum), on identical seeded repository folds. Thresholds were fixed before any result was seen. "
        f"Compute: {meta['minutes']:.0f} min on {meta['machine']}.",
        "",
    ]
    if meta.get("stopped"):
        lines += [
            f"**Stopped at the deadline ({meta['deadline']}) before:** {meta['stopped']}. Results below are partial.",
            "",
        ]
    s0 = report.get("stage0")
    if s0:
        lines += [
            "## Stage 0: checks",
            "",
            "| k | baseline OOF AUROC | fold sd | results/cv5 (all splits) |",
            "|---|---|---|---|",
        ]
        for k, r in s0["baseline"].items():
            lines.append(
                f"| {k} | {r['auroc']:.3f} | {r['fold_sd']:.3f} | {r['cv5']:.3f} |"
                if r["cv5"]
                else f"| {k} | {r['auroc']:.3f} | {r['fold_sd']:.3f} | - |"
            )
        lines += [
            "",
            f"Noise control (i.i.d. random values per step): {_gains(s0['noise'])}. Passes the stage B rule: "
            f"**{'yes, so the pipeline is broken' if s0['noise_passes'] else 'no'}**.",
            "",
        ]
    sa = report.get("stage_a")
    if sa and "ranked" in sa:
        share = f"a {sa['sample']:.0%} sample of repositories" if sa.get("sample") else "all repositories"
        lines += [
            f"## Stage A: screening (fold seed 0, {share})",
            "",
            "AUROC gain over the baseline per checkpoint (the candidate's fold sd in brackets), mean gain at 5-15 "
            "steps, and the same-issue change averaged over checkpoints.",
            "",
            "| # | channel | kind | what it measures | k=5 | k=10 | k=15 | k=20 | k=30 | mean 5-15 | same-issue change |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, name in enumerate(sa["ranked"], 1):
            r = sa["results"][name]
            cells = " | ".join(f"{r[k]['gain']:+.4f} ({r[k]['fold_sd']:.3f})" for k in CHECKPOINTS)
            pairs = np.mean([r[k]["same_issue_change"] for k in CHECKPOINTS])
            mark = " **→ B**" if name in sa["finalists"] else ""
            lines.append(
                f"| {i} | `{name}`{mark} | {CATEGORY[name]} | {DESCRIPTIONS[name]} | {cells} | "
                f"{early_mean_gain(r):+.4f} | {pairs:+.4f} |"
            )
        top = sa["top_shuffled"]
        lines += [
            "",
            f"Shuffled control of the top candidate (`{top['channel']}`, values moved between checkpoints of the same "
            f"k): {_gains(top['result'])}. Passes the stage B rule: "
            f"**{'yes, so the pipeline is broken' if top['passes'] else 'no'}**.",
            "",
        ]
    sb = report.get("stage_b")
    if sb and sb.get("results"):
        lines += [
            "## Stage B: confirmation (fold seeds 1 and 2, all repositories, pooled)",
            "",
            "AUROC gain with a paired repository-bootstrap 95 % interval (1,000 resamples, the same for both seeds and "
            "both models). Raw-series proxy: the raw per-step values of the 9 signals plus the channel, padded to 30 "
            "steps (fold seed 1).",
            "",
            "| channel | kind | k=5 | k=10 | k=15 | k=20 | k=30 | same-issue change (5 / 10 / 15 / 20 / 30) | "
            "early-stop recall change | raw proxy, mean gain 5-15 | shuffled control passes | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, r in sb["results"].items():
            s = r["summary"]
            gains = " | ".join(f"{s[k]['gain']:+.3f} {_ci(s[k]['gain_95'])}" for k in CHECKPOINTS)
            pairs = " / ".join(f"{s[k]['same_issue_change']:+.3f}" for k in CHECKPOINTS)
            recall = " / ".join(f"{s[k]['early_stop_recall_change']:+.3f}" for k in CHECKPOINTS)
            if r["passes"]:
                verdict_cell = "**passes**" + (" (summary-only)" if r["summary_only"] else "")
            else:
                why = [
                    w
                    for w, bad in (
                        ("AUROC rule", not r["auroc_rule"]),
                        ("same-issue rule", not r["same_issue_rule"]),
                        ("shuffled control passes", r["shuffled_passes"]),
                    )
                    if bad
                ]
                verdict_cell = "fails: " + ", ".join(why)
            lines.append(
                f"| `{name}` | {CATEGORY[name]} | {gains} | {pairs} | {recall} | {r['raw_early_mean_gain']:+.4f} | "
                f"{'yes' if r['shuffled_passes'] else 'no'} | {verdict_cell} |"
            )
        first = next(iter(sb["results"].values()))["summary"]
        lines += [
            "",
            "Baseline on these folds (mean of both seeds): AUROC "
            + " / ".join(f"{first[k]['auroc_baseline']:.3f}" for k in CHECKPOINTS)
            + "; same-issue pairs "
            + " / ".join(f"{first[k]['same_issue_baseline']:.3f}" for k in CHECKPOINTS)
            + "; early-stop recall (flagged group fails 95 % of the time at 5-10 steps, 99 % at 15-30) "
            + " / ".join(f"{first[k]['early_stop_recall_baseline']:.3f}" for k in CHECKPOINTS)
            + ".",
            "",
        ]
    sc = report.get("stage_c")
    if sc is not None:
        lines += ["## Stage C: combination", ""]
        if "skipped" in sc:
            lines += [f"Skipped: {sc['skipped']}.", ""]
        else:
            lines += [
                f"- `{s['channel']}` on top of {s['on_top_of'] or 'the baseline'}: {'kept' if s['kept'] else 'dropped'}"
                for s in sc["steps"]
            ]
            lines += ["", f"Final set {sc['final_set']} vs the baseline: {_gains(sc['final_vs_baseline'])}.", ""]
    lines += ["## Verdict", "", report.get("verdict", "(not reached)"), ""]
    return "\n".join(lines)


def verdict(report: dict) -> str:
    s0 = report.get("stage0", {})
    if s0.get("baseline_far_off_at") or s0.get("noise_passes"):
        return "Stage 0 failed, so nothing after it ran. The pipeline needs fixing before any channel can be judged."
    sa = report.get("stage_a", {})
    if sa.get("top_shuffled", {}).get("passes"):
        return "The shuffled control of the top candidate passed the stage B rule, so the pipeline is broken; no channel is recommended."
    sb = report.get("stage_b", {})
    if not sb.get("complete"):
        return (
            "Stage B did not finish before the deadline; the table above is partial, and no channel is recommended yet."
        )
    caveat = (
        "The proxies read features derived from the channels, while the TSLM reads the series itself, so this is a "
        "shortlist with evidence, not a guarantee. Only a **new information** channel is likely to help a model that "
        "already reads the full sequence of signals; a **re-expression** mostly restates what it can already learn."
    )
    chosen = report.get("stage_c", {}).get("final_set", [])
    if not chosen:
        finalists = sa.get("finalists", [])
        return (
            "**No channel reliably helps.** "
            + (
                f"{len(finalists)} candidates advanced from screening ({', '.join(f'`{n}`' for n in finalists)}), "
                "but none met the fixed stage B rule on fresh folds. "
                if finalists
                else "No candidate had a positive mean gain at 5-15 steps in screening. "
            )
            + "Keep the 19 series as they are for Sunday's training round. "
            + caveat
        )
    parts = [
        f"**Add {len(chosen)} channel(s) on Sunday:** " + ", ".join(f"`{n}` ({CATEGORY[n]})" for n in chosen) + ".",
        "",
    ]
    parts += [
        "| channel | kind | label for time_series() | extra tokens per sample (k=5 / 10 / 15 / 20 / 30) |",
        "|---|---|---|---|",
    ]
    for name in chosen:
        label, _ = PROPOSALS[name]
        cost = token_cost(label)
        parts.append(
            f'| `{name}` | {CATEGORY[name]} | "{label}" | '
            + " / ".join(str(cost[f"k={k}"]) for k in CHECKPOINTS)
            + " |"
        )
    if any(CATEGORY[n] == NEW for n in chosen):
        parts += [
            "",
            "A text channel needs the connector to compute it first: a parser in the connector's `source.py` (the "
            "functions here can be copied), one new signal in `connector.py`, then a rebuild (about 4 minutes) and "
            "`agent-ecg prepare`.",
        ]
    parts += [
        "",
        "In `opentslm_data.time_series()`, append each channel after the 15 indicators, scaled as its label says "
        "(`PROPOSALS[name][1]`), with the label as written: no per-run mean, std or total. Then rerun `opentslm-export`.",
        "",
        caveat,
    ]
    return "\n".join(parts)


def _intkeys(value):
    """Undo JSON's string keys for checkpoints ("5" -> 5)."""
    if isinstance(value, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intkeys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_intkeys(v) for v in value]
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run(
    data_dir: Path,
    stage: str = "all",
    sample: float | None = RULES["screening_sample"],
    raw: Path | None = None,
    workers: int = 4,
    deadline: str | None = DEADLINE,
    work: Path = Path("data/channels"),
) -> dict:
    """Run the hunt up to ``stage`` and write ``results/channels.json`` and ``results/channels.md`` after
    every step, so a stopped run still leaves a report. Fitted predictions are cached in ``work``, so a
    rerun resumes where the last one stopped."""
    import platform

    started = time.perf_counter()
    work.mkdir(parents=True, exist_ok=True)
    out_json, out_md = Path("results/channels.json"), Path("results/channels.md")
    report: dict = _intkeys(json.loads(out_json.read_text())) if out_json.exists() and stage != "all" else {}
    previous = report.get("meta", {}).get("minutes", 0.0) if stage != "all" else 0.0
    due = datetime.fromisoformat(deadline) if deadline else None

    table = development_rows(data.load_checkpoints(data_dir))
    missing = [name for name in data.SIGNALS if name not in table]
    if missing:
        raise SystemExit(f"{data_dir} lacks {missing}: rerun `agent-ecg prepare` with the current connector")
    text_path = work / "text_channels.npz"
    if text_path.exists():
        saved = np.load(text_path, allow_pickle=False)
        text = (saved["run_ids"].tolist(), saved["values"])
    else:
        print("reading the step texts from the release shards ...", flush=True)
        text = read_text_channels(sorted(table["run_id"].unique()), raw, workers)
        np.savez(text_path, run_ids=np.array(text[0]), values=text[1])
    print(f"text channels for {len(text[0]):,} runs ({(time.perf_counter() - started) / 60:.1f} min)", flush=True)

    def save() -> None:
        report["meta"] = {
            "checkpoints": len(table),
            "repositories": int(table["repository"].nunique()),
            "screening_sample": sample,
            "fold_seeds": {"screening": 0, "confirmation": [1, 2]},
            "rules": _jsonable(RULES),
            "bootstrap": BOOTSTRAP,
            "deadline": deadline,
            "stopped": report.get("meta", {}).get("stopped"),
            "minutes": previous + (time.perf_counter() - started) / 60,
            "machine": f"{platform.machine()} CPU, {os.cpu_count()} cores, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'all')}",
        }
        report["verdict"] = verdict(report)
        Path("results").mkdir(exist_ok=True)
        out_json.write_text(json.dumps(_jsonable(report), indent=2))
        out_md.write_text(to_markdown(_intkeys(_jsonable(report))))

    full = Hunt(table, text, work, due, started)
    screen = Hunt(table, text, work, due, started, sample=sample) if sample else full
    try:
        ok = stage0(full, report) if stage in ("all", "0") else True
        save()
        if ok and stage in ("all", "A"):
            stage_a(screen, report, save)
            ok = not report["stage_a"]["top_shuffled"]["passes"]
        if ok and stage in ("all", "B"):
            passing = stage_b(full, report, report["stage_a"]["finalists"], save)
            if stage == "all" or stage == "C":
                stage_c(full, report, passing, save)
        elif ok and stage == "C":
            stage_c(full, report, report["stage_b"]["passing"], save)
    except DeadlineReached as stop:
        report.setdefault("meta", {})["stopped"] = str(stop)
        full.log(f"deadline reached before fitting {stop}; writing what there is")
    save()
    full.log(f"wrote {out_json} and {out_md}")
    return report
