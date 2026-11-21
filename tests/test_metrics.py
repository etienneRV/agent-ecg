import numpy as np
import pandas as pd
import pytest

from agent_ecg.metrics import evaluate, token_f1
from agent_ecg.status import describe, parse_status, predict_sentence, split_status

DESCRIPTION = (
    "In its first 5 steps the agent edited files 0 times, created 0 files, ran code 0 times, read or "
    "searched the repository 4 times and repeated an earlier command 0 times. The linter rejected 0 "
    "edits, 1 reply showed a Python traceback, 1 reply a command error and 0 commands timed out. 0 edits "
    "changed the same file as the edit before, and 0 edits changed a file the agent had created. The "
    "conversation holds 12,857 characters."
)
PREDICTION = (
    "The run will go on for 41 more steps, then it will reach the context limit and SWE-agent will "
    "submit its changes, and the agent will not resolve the issue."
)
STATUS = f"{DESCRIPTION} {PREDICTION}"


def test_the_prediction_sentence_is_read_back() -> None:
    assert split_status(STATUS) == (DESCRIPTION, PREDICTION)
    parsed = parse_status(STATUS)
    assert (parsed.remaining, parsed.ending, parsed.resolved) == (41, "context_limit", False)
    assert (
        parse_status(
            "The run will go on for 1 more step, then the agent will submit its changes, "
            "and the agent will resolve the issue."
        ).resolved
        is True
    )
    assert not parse_status("No idea.").parsed


def test_predict_sentence_round_trips() -> None:
    for ending in ("submitted", "context_limit", "early_exit", "submitted_no_patch", "format_errors", "cost_limit"):
        parsed = parse_status(predict_sentence(7, ending, True))
        assert (parsed.remaining, parsed.ending, parsed.resolved) == (7, ending, True)


def test_describe_rebuilds_the_description() -> None:
    signals = {
        "action_kind": ["other", "search", "search", "search", "read"],
        "observation_kind": ["traceback", "output", "output", "output", "command_error"],
        "repeated_action": [False] * 5,
        "context_chars": [9000, 10000, 11000, 12000, 12857],
        "same_file_edit": [False] * 5,
        "edit_target": ["none"] * 5,
    }
    assert describe(signals) == DESCRIPTION
    edits = signals | {
        "action_kind": ["create", "edit", "edit", "edit", "read"],
        "same_file_edit": [False, False, True, False, False],
        "edit_target": ["none", "own_script", "own_script", "source", "none"],
    }
    assert "1 edit changed the same file as the edit before, and 2 edits changed a file the agent" in describe(edits)


def test_word_overlap_rewards_a_copied_description_with_a_wrong_prediction() -> None:
    wrong = STATUS.replace("will not resolve", "will resolve").replace("41", "3")
    assert token_f1(STATUS, wrong) > 0.95
    assert token_f1(split_status(STATUS)[1], split_status(wrong)[1]) < 0.95
    assert parse_status(wrong).resolved is True


def _gold(n_repos: int = 20, per_repo: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for repo in range(n_repos):
        for i in range(per_repo):
            resolved = bool(rng.random() < 0.3)
            rows.append(
                {
                    "run_id": f"r{repo}-{i}",
                    "k": 5,
                    "split": "test",
                    "repository": f"repo{repo}",
                    "resolved": resolved,
                    "ending": "submitted" if resolved else "context_limit",
                    "status": f"{DESCRIPTION} {predict_sentence(10, 'submitted' if resolved else 'context_limit', resolved)}",
                }
            )
    return pd.DataFrame(rows)


def test_a_perfect_model_scores_one() -> None:
    gold = _gold()
    pred = gold[["run_id", "k"]].assign(
        p_resolved=gold["resolved"].astype(float), pred_ending=gold["ending"], pred_status=gold["status"]
    )
    report = evaluate(gold, pred)
    outcome, ending, status = report["outcome"]["k=5"], report["ending"]["k=5"], report["status"]["k=5"]
    assert outcome["auroc"] == 1.0 and outcome["balanced_accuracy"] == 1.0 and outcome["brier"] == 0.0
    assert ending["macro_f1"] == 1.0
    assert status["parse_rate"] == 1.0 and status["remaining_mae"] == 0.0 and status["description_exact"] == 1.0


def test_always_fail_has_chance_auroc_and_balanced_accuracy() -> None:
    gold = _gold()
    pred = gold[["run_id", "k"]].assign(p_resolved=0.2)
    outcome = evaluate(gold, pred)["outcome"]["k=5"]
    assert outcome["auroc"] == 0.5 and outcome["balanced_accuracy"] == 0.5


def test_missing_predictions_are_an_error() -> None:
    gold = _gold()
    with pytest.raises(ValueError, match="miss"):
        evaluate(gold, gold[["run_id", "k"]].iloc[:5].assign(p_resolved=0.5))


def test_a_better_model_has_a_positive_paired_gain() -> None:
    from agent_ecg.metrics import paired_auroc_gain

    gold = _gold()
    perfect = gold[["run_id", "k"]].assign(p_resolved=gold["resolved"].astype(float))
    noise = gold[["run_id", "k"]].assign(p_resolved=np.random.default_rng(1).random(len(gold)))
    result = paired_auroc_gain(gold, perfect, noise, n=200)["k=5"]
    assert result["gain"] > 0.3 and result["gain_95"][0] > 0 and result["p_not_better"] == 0.0


def test_the_prediction_is_found_whatever_its_case() -> None:
    lower = STATUS.replace("The run will go on for", "the run will go on for")
    assert parse_status(lower).remaining == 41


def test_a_gap_in_one_prediction_column_is_an_error() -> None:
    gold = _gold()
    pred = gold[["run_id", "k"]].assign(p_resolved=0.5, pred_ending=gold["ending"])
    pred.loc[3, "p_resolved"] = np.nan
    with pytest.raises(ValueError, match="miss p_resolved for 1"):
        evaluate(gold, pred)


def test_compare_needs_full_files() -> None:
    from agent_ecg.metrics import paired_auroc_gain

    gold = _gold()
    full = gold[["run_id", "k"]].assign(p_resolved=0.5)
    with pytest.raises(ValueError, match="cover 10 of the 200"):
        paired_auroc_gain(gold, full, full.iloc[:10], n=10)


def test_same_issue_pairs_compare_runs_on_one_issue_only() -> None:
    from agent_ecg.metrics import same_issue_pairs

    resolved = np.array([True, False, True, False, False])
    issues = np.array(["a", "a", "b", "b", "c"])
    # Issue b looks easy (0.9 for both runs) and issue a hard (0.2, 0.1); only order within an issue counts.
    score = np.array([0.2, 0.1, 0.9, 0.9, 0.5])
    share, pairs = same_issue_pairs(resolved, score, issues)
    assert (share, pairs) == (0.75, 2)  # a: the resolved run ranks higher; b: a tie counts half
    assert np.isnan(same_issue_pairs(resolved[[0, 2]], score[[0, 2]], issues[[0, 2]])[0])


def test_the_outcome_report_has_same_issue_pairs_and_calibrated_brier() -> None:
    gold = _gold().assign(instance_id=lambda t: t["run_id"].str.split("-").str[0])
    pred = gold[["run_id", "k"]].assign(p_resolved=gold["resolved"].astype(float) * 0.5 + 0.5, p_calibrated=0.3)
    outcome = evaluate(gold, pred)["outcome"]["k=5"]
    assert outcome["same_issue_pairs"] == 1.0 and outcome["n_pairs"] > 0
    assert outcome["brier_calibrated"] > 0
