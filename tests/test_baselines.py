import numpy as np
import pandas as pd

from agent_ecg import baselines


def _runs(repos: int = 10, issues: int = 2, runs: int = 3) -> tuple[pd.DataFrame, dict, dict]:
    rng = np.random.default_rng(0)
    rows, thoughts, texts = [], {}, {}
    for repo in range(repos):
        for issue in range(issues):
            issue_id = repo * issues + issue
            texts[issue_id] = f"issue {issue_id} in repository {repo}"
            for run in range(runs):
                run_id = f"{repo}-{issue}-{run}"
                resolved = bool(rng.random() < 0.4)
                actions = list(rng.choice(baselines.ACTIONS, size=10))
                full = {
                    "thought_chars": list(rng.integers(50, 300, 10)),
                    "action_chars": list(rng.integers(5, 50, 10)),
                    "observation_chars": list(rng.integers(0, 900, 10)),
                    "context_chars": list(np.cumsum(rng.integers(100, 1000, 10)) + 4000),
                    "repeated_action": list(rng.random(10) < 0.2),
                    "action_kind": actions,
                    "observation_kind": list(rng.choice(baselines.REPLIES, size=10)),
                    "same_file_edit": [a == "edit" and bool(rng.random() < 0.5) for a in actions],
                    "edit_target": ["source" if a == "edit" else "none" for a in actions],
                }
                thoughts[run_id] = [f"thought {i} of run {run_id}" for i in range(10)]
                for k in (5, 10):
                    rows.append(
                        {
                            "run_id": run_id,
                            "instance_id": f"{repo}-{issue}",
                            "repository": f"repo{repo}",
                            "issue_id": issue_id,
                            "k": k,
                            "resolved": resolved,
                            **{name: values[:k] for name, values in full.items()},
                        }
                    )
    return pd.DataFrame(rows), thoughts, texts


def test_features_count_file_identity() -> None:
    table, _, _ = _runs()
    features = baselines.features(table)
    assert {"same_file_edit", "edit_source", "edit_own_script_recent", "edit_test"} <= set(features.columns)
    edits = table["action_kind"].map(lambda kinds: kinds.count("edit") / len(kinds))
    assert np.allclose(features["edit_source"], edits)


def test_cross_validation_scores_every_model_and_checkpoint() -> None:
    table, thoughts, texts = _runs()
    result = baselines.cross_validate(table, thoughts, texts, folds=5, models=("signals", "issue"))
    assert set(result) == {"signals", "issue"} and set(result["signals"]) == {5, 10}
    for by_k in result.values():
        for scores in by_k.values():
            assert 0 <= scores["auroc"] <= 1 and scores["auroc_fold_sd"] >= 0
