"""Six CPU baselines a time-series language model has to beat.

``majority``
    Always "will not resolve", with the training resolved rate of the checkpoint as its probability.
    Ending: the most common one; remaining steps: the training median.
``signals``
    Gradient-boosted trees on summary features of the nine per-step signals.
``thoughts``
    Logistic regression on hashed word and word-pair counts of the agent's thoughts so far.
``signals+thoughts``
    Logistic regression on both.
``issue``
    Logistic regression on hashed word and word-pair counts of the issue text alone, plus the
    checkpoint. It never sees the agent: if the agent-based baselines beat it, they learn from the
    agent's behaviour and not only from how hard the issue is.
``issue+signals+thoughts``
    Logistic regression on everything a full-context model reads: the bar for the TSLM.

All train on the train split, pick their decision threshold and regularisation on validation (mean per-checkpoint AUROC), and
predict validation and test. Each checkpoint is weighted by 1 / (checkpoints of its run), so long runs,
which fail more often, do not count five times. The status report of a baseline is the description
rebuilt from the signals plus a prediction sentence from its three predicted facts. The signals model
and its monitor (fitted on validation, see :mod:`agent_ecg.monitor`) are saved for ``agent-ecg replay``.

:func:`cross_validate` scores the same models by repository cross-validation over every checkpoint,
a steadier number than the single test split of 143 repositories.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from agent_ecg import opentslm_samples as samples
from agent_ecg.data import CHECKPOINTS
from agent_ecg.metrics import same_issue_pairs
from agent_ecg.monitor import Monitor
from agent_ecg.status import describe, predict_sentence

ACTIONS = ("edit", "create", "run", "read", "search", "submit", "other")
REPLIES = ("edit_rejected", "timeout", "traceback", "command_error", "silent_success", "output")
TARGETS = ("source", "test", "own_script")
_RECENT = 5
_SEED = 0
CV_MODELS = ("signals", "issue", "thoughts")


def features(table: pd.DataFrame) -> pd.DataFrame:
    """Summary features of each checkpoint's signals: what the agent did, how replies went, how long."""

    def one(row: pd.Series) -> dict:
        actions, replies = list(row["action_kind"]), list(row["observation_kind"])
        repeated = np.asarray(row["repeated_action"], float)
        thought, action = np.asarray(row["thought_chars"], float), np.asarray(row["action_chars"], float)
        reply, context = np.asarray(row["observation_chars"], float), np.asarray(row["context_chars"], float)
        k = len(actions)
        result = {"k": k, "repeated": repeated.mean(), "repeated_recent": repeated[-_RECENT:].mean()}
        for kind in ACTIONS:
            result[f"action_{kind}"] = actions.count(kind) / k
            result[f"action_{kind}_recent"] = actions[-_RECENT:].count(kind) / _RECENT
        for kind in REPLIES:
            result[f"reply_{kind}"] = replies.count(kind) / k
            result[f"reply_{kind}_recent"] = replies[-_RECENT:].count(kind) / _RECENT
        same, targets = np.asarray(row["same_file_edit"], float), list(row["edit_target"])
        result["same_file_edit"], result["same_file_edit_recent"] = same.mean(), same[-_RECENT:].mean()
        for kind in TARGETS:
            result[f"edit_{kind}"] = targets.count(kind) / k
            result[f"edit_{kind}_recent"] = targets[-_RECENT:].count(kind) / _RECENT
        result |= {
            "context_last": context[-1],
            "context_growth": (context[-1] - context[0]) / k,
            "opening": context[0] - thought[0] - action[0] - reply[0],
            "thought_mean": thought.mean(),
            "thought_recent": thought[-_RECENT:].mean(),
            "action_mean": action.mean(),
            "reply_mean": reply.mean(),
            "reply_max": reply.max(),
        }
        return result

    return pd.DataFrame([one(row) for _, row in table.iterrows()], index=table.index)


def thought_text(table: pd.DataFrame, thoughts: dict[str, list[str]]) -> list[str]:
    """The thoughts a checkpoint sees, joined: the first k of its run."""
    return ["\n".join(thoughts[run][:k]) for run, k in zip(table["run_id"], table["k"], strict=True)]


def _weights(table: pd.DataFrame) -> np.ndarray:
    return 1.0 / table.groupby("run_id")["k"].transform("size").to_numpy(float)


def _log(message: str, start: float) -> None:
    print(f"  [{time.perf_counter() - start:6.1f}s] {message}", flush=True)


def _predictions(table: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame({"run_id": table["run_id"], "k": table["k"], "p_resolved": p, "pred_resolved": p >= threshold})


def _with_status(
    frame: pd.DataFrame, table: pd.DataFrame, ending: np.ndarray | list, remaining: np.ndarray
) -> pd.DataFrame:
    frame = frame.copy()
    frame["pred_ending"] = list(ending)
    frame["pred_status"] = [
        f"{describe(row)} {predict_sentence(int(r), e, bool(y))}"
        for (_, row), r, e, y in zip(table.iterrows(), remaining, ending, frame["pred_resolved"], strict=True)
    ]
    return frame


def run(
    checkpoints: pd.DataFrame, thoughts: dict[str, list[str]], issues: dict[int, str], out: Path
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fit every baseline and write its validation and test predictions to ``out``.

    Returns:
        For each baseline, its predictions keyed by split.
    """
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    parts = {split: checkpoints[checkpoints["split"] == split] for split in ("train", "validation", "test")}
    train, valid = parts["train"], parts["validation"]
    y_train, y_valid = train["resolved"].to_numpy(bool), valid["resolved"].to_numpy(bool)
    w_train = _weights(train)
    results: dict[str, dict[str, pd.DataFrame]] = {}

    # majority
    rate = train.groupby("k")["resolved"].mean()
    ending = train["ending"].mode()[0]
    median_remaining = int(train["remaining"].median())
    results["majority"] = {
        split: _with_status(
            pd.DataFrame(
                {"run_id": t["run_id"], "k": t["k"], "p_resolved": t["k"].map(rate).to_numpy(), "pred_resolved": False}
            ),
            t,
            [ending] * len(t),
            np.full(len(t), median_remaining),
        )
        for split, t in parts.items()
        if split != "train"
    }
    _log("majority", start)

    # signals
    X = {split: features(t) for split, t in parts.items()}
    _log("signal features", start)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=_SEED)
    clf.fit(X["train"], y_train, sample_weight=w_train)
    end_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=_SEED)
    end_clf.fit(X["train"], train["ending"], sample_weight=w_train)
    steps = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=_SEED)
    steps.fit(X["train"], np.log1p(train["remaining"]), sample_weight=w_train)
    threshold = samples.threshold(y_valid, clf.predict_proba(X["validation"])[:, 1])
    # The replay scores any prefix of a run with this model, through a monitor fitted on validation.
    models = out.parent / "models"
    models.mkdir(parents=True, exist_ok=True)
    with (models / "signals.pkl").open("wb") as handle:
        pickle.dump({"model": clf, "columns": list(X["train"].columns)}, handle)
    Monitor.fit(y_valid, clf.predict_proba(X["validation"])[:, 1], valid["k"].to_numpy()).to_json(
        models / "signals.monitor.json"
    )
    results["signals"] = {}
    for split in ("validation", "test"):
        p = clf.predict_proba(X[split])[:, 1]
        remaining = np.maximum(1, np.round(np.expm1(steps.predict(X[split])))).astype(int)
        results["signals"][split] = _with_status(
            _predictions(parts[split], p, threshold), parts[split], end_clf.predict(X[split]), remaining
        )
    _log("signals", start)

    # thoughts, and signals+thoughts
    hasher = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), alternate_sign=False, norm=None)
    tfidf = TfidfTransformer(sublinear_tf=True)
    T = {"train": tfidf.fit_transform(hasher.transform(thought_text(train, thoughts)))}
    for split in ("validation", "test"):
        T[split] = tfidf.transform(hasher.transform(thought_text(parts[split], thoughts)))
    _log("thought features", start)
    scaler = StandardScaler().fit(X["train"])
    both = {s: sparse.hstack([T[s], sparse.csr_matrix(scaler.transform(X[s]))]).tocsr() for s in T}
    # The issue alone, with the checkpoint as one-hot columns so its probabilities fit each k.
    issue_tfidf = TfidfTransformer(sublinear_tf=True)
    issue_matrix: dict = {}
    for split, table in parts.items():
        counts = hasher.transform(table["issue_id"].map(issues))
        words = issue_tfidf.fit_transform(counts) if split == "train" else issue_tfidf.transform(counts)
        k = sparse.csr_matrix(np.stack([(table["k"] == c).to_numpy(float) for c in CHECKPOINTS], axis=1))
        issue_matrix[split] = sparse.hstack([words, k]).tocsr()
    _log("issue features", start)
    everything = {s: sparse.hstack([both[s], issue_matrix[s]]).tocsr() for s in T}
    fits = (
        ("thoughts", T),
        ("signals+thoughts", both),
        ("issue", issue_matrix),
        ("issue+signals+thoughts", everything),
    )
    for name, matrix in fits:
        best: tuple[float, LogisticRegression | None] = (-1.0, None)
        for c in (0.1, 0.3, 1.0):
            model = LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=_SEED)
            model.fit(matrix["train"], y_train, sample_weight=w_train)
            # Mean per-checkpoint AUROC: pooled AUROC would reward fitting the checkpoint, not the run.
            auroc = samples.mean_auroc(y_valid, valid["k"].to_numpy(), model.predict_proba(matrix["validation"])[:, 1])
            _log(f"{name} C={c}: validation AUROC {auroc:.3f}", start)
            if auroc > best[0]:
                best = (auroc, model)
        model = best[1]
        assert model is not None
        threshold = samples.threshold(y_valid, model.predict_proba(matrix["validation"])[:, 1])
        results[name] = {
            split: _predictions(parts[split], model.predict_proba(matrix[split])[:, 1], threshold)
            for split in ("validation", "test")
        }
    # Calibrated probabilities from an isotonic map per checkpoint, fitted on validation (so only the
    # test split measures them fairly).
    for name, by_split in results.items():
        if name == "majority":
            continue
        fitted = Monitor.fit(y_valid, by_split["validation"]["p_resolved"].to_numpy(), valid["k"].to_numpy())
        for frame in by_split.values():
            frame["p_calibrated"] = [
                fitted.calibrate(p, k) for p, k in zip(frame["p_resolved"], frame["k"], strict=True)
            ]
    for name, by_split in results.items():
        for split, frame in by_split.items():
            frame.to_parquet(out / f"{name.replace('+', '_')}.{split}.parquet", index=False)
    return results


def signals_scorer(models: Path) -> Callable[[list[dict]], np.ndarray]:
    """The signals model that :func:`run` saved, as a replay scorer: prefix rows in, P(resolved) out."""
    with (models / "signals.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    model, columns = saved["model"], saved["columns"]

    def score(cuts: list[dict]) -> np.ndarray:
        return model.predict_proba(features(pd.DataFrame(cuts))[columns])[:, 1]

    return score


def _k_columns(table: pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.stack([(table["k"] == c).to_numpy(float) for c in CHECKPOINTS], axis=1))


def _text_model(counts: sparse.csr_matrix, extra: sparse.csr_matrix, train: np.ndarray):
    """TF-IDF fitted on the training rows only, then the checkpoint columns."""
    tfidf = TfidfTransformer(sublinear_tf=True).fit(counts[train])
    return sparse.hstack([tfidf.transform(counts), extra]).tocsr()


def cross_validate(
    checkpoints: pd.DataFrame,
    thoughts: dict[str, list[str]],
    issues: dict[int, str],
    folds: int = 5,
    models: tuple[str, ...] = CV_MODELS,
) -> dict[str, dict[int, dict[str, float]]]:
    """Score baselines by repository cross-validation over every checkpoint of all three splits.

    The test split holds 143 repositories, so its AUROC moves by several points with the repositories
    it drew. Here each fold holds out a fifth of all repositories, and every checkpoint is predicted
    once, by a model that never saw its repository. Settings are fixed (no tuning): the signals model
    as in :func:`run`, the text models with C = 0.3.

    Returns:
        For each model and checkpoint: pooled AUROC, the mean and spread of the per-fold AUROCs, and
        the same-issue pair score.
    """
    unknown = set(models) - set(CV_MODELS)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; choose from {CV_MODELS}")
    table = checkpoints.reset_index(drop=True)
    y, k, w = table["resolved"].to_numpy(bool), table["k"].to_numpy(), _weights(table)
    hasher = HashingVectorizer(n_features=2**20, ngram_range=(1, 2), alternate_sign=False, norm=None)
    inputs: dict[str, object] = {}
    if "signals" in models:
        inputs["signals"] = features(table).to_numpy(float)
    if "issue" in models:
        inputs["issue"] = hasher.transform(table["issue_id"].map(issues))
    if "thoughts" in models:
        inputs["thoughts"] = hasher.transform(thought_text(table, thoughts))
    oof = {m: np.zeros(len(table)) for m in models}
    fold_of = np.zeros(len(table), dtype=int)
    for fold, (train, test) in enumerate(GroupKFold(folds).split(table, groups=table["repository"])):
        fold_of[test] = fold
        for m in models:
            if m == "signals":
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=_SEED)
                matrix = inputs[m]
            else:
                clf = LogisticRegression(C=0.3, max_iter=2000, solver="liblinear", random_state=_SEED)
                matrix = _text_model(inputs[m], _k_columns(table), train)
            clf.fit(matrix[train], y[train], sample_weight=w[train])
            oof[m][test] = clf.predict_proba(matrix[test])[:, 1]
    result: dict[str, dict[int, dict[str, float]]] = {}
    for m in models:
        result[m] = {}
        for c in sorted(int(v) for v in np.unique(k)):
            at = k == c
            per_fold = [
                roc_auc_score(y[at & (fold_of == f)], oof[m][at & (fold_of == f)])
                for f in range(folds)
                if 0 < y[at & (fold_of == f)].sum() < (at & (fold_of == f)).sum()
            ]
            pairs, _ = same_issue_pairs(y[at], oof[m][at], table["instance_id"].to_numpy()[at])
            result[m][c] = {
                "auroc": float(roc_auc_score(y[at], oof[m][at])),
                "auroc_fold_mean": float(np.mean(per_fold)),
                "auroc_fold_sd": float(np.std(per_fold)),
                "same_issue_pairs": pairs,
            }
    return result
