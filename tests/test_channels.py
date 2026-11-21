import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from agent_ecg import channels, data
from agent_ecg.channels import Step
from agent_ecg.metrics import same_issue_pairs

ACTIONS = ("edit", "create", "run", "read", "search", "submit", "other")
REPLIES = ("edit_rejected", "timeout", "traceback", "command_error", "silent_success", "output")
TARGETS = ("none", "source", "test", "own_script")


def _signals(k: int, seed: int) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    actions = list(rng.choice(ACTIONS, k))
    targets = [str(rng.choice(TARGETS[1:])) if a == "edit" and rng.random() < 0.95 else "none" for a in actions]
    return {
        "thought_chars": rng.integers(0, 800, k).tolist(),
        "action_chars": rng.integers(1, 400, k).tolist(),
        "observation_chars": rng.integers(0, 6000, k).tolist(),
        "context_chars": (np.cumsum(rng.integers(100, 5000, k)) + 3000).tolist(),
        "repeated_action": (rng.random(k) < 0.3).tolist(),
        "action_kind": [str(a) for a in actions],
        "observation_kind": [str(v) for v in rng.choice(REPLIES, k)],
        "same_file_edit": [t != "none" and bool(rng.random() < 0.6) for t in targets],
        "edit_target": targets,
    }


def _footer(path: str) -> str:
    return f"\n(Open file: {path})\n(Current directory: /repo)\nbash-$"


# Replies drawn from what the release's replies look like.
_REPLIES = [
    "[File: /repo/a.py (120 lines total)]\n(10 more lines above)\n11:x = 1\n12:y = 2\n13:z = 3" + _footer("/repo/a.py"),
    "Traceback (most recent call last):\n  File \"r.py\", line 1\nAttributeError: 'X' has no attribute 'y'"
    + _footer("/repo/a.py"),
    "Traceback (most recent call last):\n  File \"r.py\", line 1\nModuleNotFoundError: No module named 'z'"
    + _footer("n/a"),
    "========================= 2 failed, 18 passed in 0.66s =========================" + _footer("/repo/a.py"),
    "============================== 20 passed in 0.18s ===============================" + _footer("/repo/a.py"),
    "Ran 5 tests in 0.01s\n\nFAILED (failures=1, errors=1)" + _footer("/repo/a.py"),
    "Ran 5 tests in 0.01s\n\nOK" + _footer("/repo/a.py"),
    "Your proposed edit has introduced new syntax error(s).\n\nERRORS:\n- F821 undefined name 'os'\n- E999 IndentationError"
    + _footer("/repo/a.py"),
    'No matches found for "foo" in /repo' + _footer("/repo/a.py"),
    'Found 12 matches for "parse" in /repo:' + _footer("/repo/a.py"),
    "File missing.py not found" + _footer("n/a"),
    "Script completed successfully, no errors." + _footer("/repo/a.py"),
    "Your command ran successfully and did not produce any output." + _footer("/repo/a.py"),
]
_COMMANDS = [
    "open a.py",
    "python reproduce.py",
    "edit 11:12\n    def f(x):\n        return x\nend_of_edit",
    "edit 50:60\nreturn 1\nend_of_edit",
    "pytest tests/test_a.py",
    "python -m unittest tests.test_a",
    "search_dir foo",
    "rm reproduce.py",
    "goto 100",
    "submit",
]


def _steps(k: int, seed: int) -> list[Step]:
    rng = np.random.default_rng(seed)
    return [Step(f"thought {t}", str(rng.choice(_COMMANDS)), str(rng.choice(_REPLIES))) for t in range(k)]


class _Recording(Mapping):
    """A signals mapping that records every key a channel reads and refuses any other column."""

    def __init__(self, signals: dict[str, list]) -> None:
        self.signals, self.read = signals, set()

    def __getitem__(self, key: str) -> list:
        self.read.add(key)
        if key not in data.SIGNALS:
            raise KeyError(f"forbidden column {key}")
        return self.signals[key]

    def __iter__(self):
        return iter(self.signals)

    def __len__(self) -> int:
        return len(self.signals)


def _run(name: str, k: int, seed: int) -> list[float]:
    if name in channels.SIGNAL_CHANNELS:
        return channels.SIGNAL_CHANNELS[name](_signals(k, seed))
    return channels.TEXT_CHANNELS[name](_steps(k, seed))


@pytest.mark.parametrize("name", channels.CANDIDATES)
def test_a_channel_gives_one_finite_value_per_step(name: str) -> None:
    for seed, k in enumerate((1, 5, 17, 30)):
        values = _run(name, k, seed)
        assert len(values) == k
        assert all(isinstance(v, float) and np.isfinite(v) for v in values)


@pytest.mark.parametrize("name", channels.CANDIDATES)
def test_a_channel_never_looks_ahead(name: str) -> None:
    for seed in range(20):
        if name in channels.SIGNAL_CHANNELS:
            full, fn = _signals(30, seed), channels.SIGNAL_CHANNELS[name]
            prefix = lambda k, full=full: {n: v[:k] for n, v in full.items()}  # noqa: E731
        else:
            full, fn = _steps(30, seed), channels.TEXT_CHANNELS[name]
            prefix = lambda k, full=full: full[:k]  # noqa: E731
        whole = fn(full)
        for k in range(1, 31):
            assert fn(prefix(k)) == pytest.approx(whole[:k]), (name, seed, k)


@pytest.mark.parametrize("name", list(channels.SIGNAL_CHANNELS))
def test_a_signal_channel_reads_only_the_step_signals(name: str) -> None:
    recording = _Recording(_signals(12, 3))
    channels.SIGNAL_CHANNELS[name](recording)
    assert recording.read <= set(data.SIGNALS)


def test_only_the_trajectory_is_read_from_the_shards() -> None:
    assert channels.TEXT_COLUMNS == ["trajectory"]


def test_signals_of_drops_targets_and_identifiers() -> None:
    row = _signals(5, 0) | {"resolved": True, "ending": "submitted", "status": "x", "run_id": "r", "split": "test"}
    assert set(channels.signals_of(row)) == set(data.SIGNALS)


def test_every_channel_has_a_kind_and_a_tslm_proposal() -> None:
    assert set(channels.PROPOSALS) == set(channels.CANDIDATES) == set(channels.CATEGORY)
    assert len(channels.CANDIDATES) <= 15  # the stage A cap
    for name, (label, scale) in channels.PROPOSALS.items():
        values = [scale(v) for v in _run(name, 30, 1)]
        assert all(-1.0 <= v <= 1.0 for v in values), name
        # A fixed text: nothing about the run (a mean, a std, a total) is formatted into it.
        assert label.endswith(":") and "{" not in label, name


def test_test_summaries_are_read() -> None:
    assert channels.test_counts(_REPLIES[3]) == (18, 2)
    assert channels.test_counts(_REPLIES[4]) == (20, 0)
    assert channels.test_counts(_REPLIES[5]) == (3, 2)
    assert channels.test_counts(_REPLIES[6]) == (5, 0)
    assert channels.test_counts(_REPLIES[0]) is None
    steps = [Step("", "pytest", _REPLIES[3]), Step("", "open a.py", _REPLIES[0]), Step("", "pytest", _REPLIES[4])]
    assert channels.tests_failed(steps) == [2, 0, 0]
    assert channels.tests_all_passed(steps) == [0, 0, 1]
    assert channels.tests_failed_delta(steps) == [0, 0, -2]


def test_a_script_fixed_after_an_edit_is_seen() -> None:
    crash, clean = _REPLIES[1], _REPLIES[11]
    steps = [
        Step("", "python r.py", crash),
        Step("", "python r.py", crash),  # no edit in between: neither fixed nor still crashing
        Step("", "edit 1:1\nx\nend_of_edit", _REPLIES[0]),
        Step("", "python r.py", crash),
        Step("", "edit 1:1\ny\nend_of_edit", _REPLIES[0]),
        Step("", "python r.py", clean),
    ]
    assert channels.script_still_crashing(steps) == [0, 0, 0, 1, 0, 0]
    assert channels.script_fixed(steps) == [0, 0, 0, 0, 0, 1]
    assert channels.success_marker(steps) == [0, 0, 0, 0, 0, 1]


def test_exception_types_are_compared() -> None:
    attribute, module = _REPLIES[1], _REPLIES[2]
    steps = [Step("", "python r.py", r) for r in (attribute, attribute, module, _REPLIES[0], module)]
    assert channels.same_exception(steps) == [0, 1, 0, 0, 1]
    assert channels.exception_changed(steps) == [0, 0, 1, 0, 0]


def test_edits_are_measured_against_what_was_shown() -> None:
    steps = [
        Step("", "open a.py", _REPLIES[0]),  # shows lines 11-13 of /repo/a.py
        Step("", "edit 11:12\n    def f(x):\n        return x\nend_of_edit", _REPLIES[0]),
        Step("", "edit 50:60\nreturn 1\nend_of_edit", _REPLIES[0]),
    ]
    assert channels.edit_span(steps) == [0, 2, 11]
    assert channels.edit_size(steps) == [0, 2, 1]
    assert channels.edit_unviewed(steps) == [0, 0, 1]


def test_steps_are_split_like_the_connector() -> None:
    trajectory = [
        {"role": "system", "text": "", "system_prompt": "You are..."},
        {"role": "user", "text": "ISSUE:\nbug\n\nINSTRUCTIONS:\nfix", "system_prompt": None},
        {"role": "ai", "text": "Let me look.\n\n```\nopen a.py\n```", "system_prompt": None},
        {"role": "user", "text": _REPLIES[0], "system_prompt": None},
        {"role": "ai", "text": "Done.\n\n```\nsubmit\n```", "system_prompt": None},
    ]
    steps = channels.parse_steps(trajectory)
    assert steps == [Step("Let me look.", "open a.py", _REPLIES[0]), Step("Done.", "submit", "")]
    assert channels.text_values(steps).shape == (len(channels.TEXT_CHANNELS), 30)


def test_the_test_split_is_dropped() -> None:
    table = pd.DataFrame({"split": ["train", "validation", "test", "train"], "k": [5, 5, 5, 10]})
    assert list(channels.development_rows(table)["split"]) == ["train", "validation", "train"]


def test_folds_hold_whole_repositories_and_follow_the_seed() -> None:
    repos = np.array([f"r{i % 40}" for i in range(400)])
    fold = channels.fold_assignment(repos, seed=1)
    assert set(fold) == set(range(channels.FOLDS))
    assert pd.Series(fold).groupby(repos).nunique().max() == 1
    assert (channels.fold_assignment(repos, seed=1) == fold).all()
    assert (channels.fold_assignment(repos, seed=2) != fold).any()


def test_controls_are_wired() -> None:
    k = np.array([5] * 50 + [10] * 50)
    frame = pd.DataFrame({"x": np.arange(100.0)})
    shuffled = channels.shuffle_rows(frame, k, seed=0)
    # Values move only between checkpoints of the same k, and the order changes.
    assert sorted(shuffled["x"][:50]) == list(range(50)) and sorted(shuffled["x"][50:]) == list(range(50, 100))
    assert (shuffled["x"].to_numpy() != frame["x"].to_numpy()).any()
    table = pd.DataFrame({"k": [5, 10, 30]})
    noise = channels.noise_series(table, seed=3)
    assert [len(v) for v in noise] == [5, 10, 30]
    assert noise == channels.noise_series(table, seed=3)


def test_weighted_auroc_matches_sklearn_and_counts_repeats() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.2
    p = np.round(rng.random(300), 2)  # with ties
    unique, inverse = np.unique(p, return_inverse=True)
    assert channels._weighted_auc(inverse, len(unique), y.astype(float), np.ones(300)) == pytest.approx(
        roc_auc_score(y, p)
    )
    w = rng.integers(0, 3, 300).astype(float)
    rows = np.repeat(np.arange(300), w.astype(int))
    assert channels._weighted_auc(inverse, len(unique), y.astype(float), w) == pytest.approx(
        roc_auc_score(y[rows], p[rows])
    )


def _table(n_repos: int = 30, per_issue: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {
            "k": 5,
            "repository": f"repo{r}",
            "instance_id": f"repo{r}__{issue}",
            "resolved": bool(rng.random() < 0.3),
            "run_id": f"{r}-{issue}-{i}",
        }
        for r in range(n_repos)
        for issue in range(2)
        for i in range(per_issue)
    ]
    return pd.DataFrame(rows)


def test_checkpoint_scores_match_the_shared_metrics() -> None:
    table = _table()
    p = np.random.default_rng(1).random(len(table))
    cp = channels.Checkpoint(table, 5, n_boot=50)
    assert cp.auroc(p) == pytest.approx(roc_auc_score(table["resolved"], p))
    expected, _ = same_issue_pairs(table["resolved"].to_numpy(), p, table["instance_id"].to_numpy())
    assert cp.same_issue(p) == pytest.approx(expected)
    assert cp.boot_auroc(p).shape == (50,) and cp.boot_same_issue(p).shape == (50,)


def test_a_better_candidate_has_a_positive_interval_and_the_same_one_none() -> None:
    table = _table(n_repos=60)
    y = table["resolved"].to_numpy()
    base = np.random.default_rng(2).random(len(table))
    points = {5: channels.Checkpoint(table, 5, n_boot=200)}
    fold = np.zeros(len(table), dtype=int)
    gain = channels.compare(points, [base + 2.0 * y], [base], [fold])[5]
    assert gain["gain"] > 0.3 and gain["gain_95"][0] > 0
    same = channels.compare(points, [base.copy()], [base], [fold])[5]
    assert same["gain"] == 0 and same["gain_95"] == [0.0, 0.0]


def test_early_stop_recall_is_the_largest_share_at_the_precision() -> None:
    resolved = np.array([False] * 9 + [True] + [False] * 10)
    p = np.arange(20, dtype=float)  # the 10th lowest is the one resolved run
    assert channels.early_stop_recall(resolved, p, 0.99) == pytest.approx(9 / 19)
    assert channels.early_stop_recall(resolved, p, 0.95) == pytest.approx(1.0)


def test_raw_matrix_pads_to_thirty_steps_and_codes_the_enums() -> None:
    table = pd.DataFrame([_signals(5, 0), _signals(30, 1)])
    X, categorical = channels.raw_matrix(table)
    assert X.shape == (2, 30 * len(data.SIGNALS)) and categorical.sum() == 3 * 30
    assert np.isnan(X[0, 5:30]).all() and not np.isnan(X[1, :30]).any()


_DATA = Path("data/checkpoints.parquet")


@pytest.mark.skipif(not _DATA.exists(), reason="needs `agent-ecg prepare`")
def test_signal_channels_are_prefix_consistent_across_real_checkpoints() -> None:
    table = pd.read_parquet(_DATA)
    if any(name not in table for name in data.SIGNALS):
        pytest.skip("data/ predates the current connector")
    table = channels.development_rows(table)
    sample = table[table.groupby("run_id")["k"].transform("size") >= 3].sort_values(["run_id", "k"]).head(300)
    for _, group in sample.groupby("run_id"):
        rows = [channels.signals_of(row) for _, row in group.iterrows()]
        for name, fn in channels.SIGNAL_CHANNELS.items():
            longest = fn(rows[-1])
            for row in rows[:-1]:
                assert fn(row) == pytest.approx(longest[: len(row["action_kind"])]), name


_SHARD = os.environ.get("AGENT_ECG_RAW_SHARD")  # a local train-000NN-of-00012.parquet


@pytest.mark.skipif(not _SHARD, reason="set AGENT_ECG_RAW_SHARD to a raw release shard")
def test_text_channels_on_real_runs_keep_the_connector_step_count() -> None:
    from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import MAX_STEPS

    stem = Path(_SHARD).stem
    ids, values = channels._shard_values(_SHARD, stem, list(range(200)))
    assert ids == [f"{stem}:{i}" for i in range(200)]
    import pyarrow.parquet as pq

    trajectories = pq.read_table(_SHARD, columns=["trajectory"]).slice(0, 200).column("trajectory").to_pylist()
    for row, trajectory in zip(values, trajectories, strict=True):
        steps = channels.parse_steps(trajectory)
        n = min((len(trajectory) - 2 + 1) // 2, MAX_STEPS)
        assert len(steps) == n and not np.isnan(row[:, :n]).any() and np.isnan(row[:, n:]).all()
        for t in range(1, n + 1):
            assert channels.text_values(steps[:t])[:, :t] == pytest.approx(row[:, :t])
