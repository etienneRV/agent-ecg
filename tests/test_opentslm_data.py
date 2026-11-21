import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from agent_ecg import opentslm_data
from agent_ecg.opentslm_data import POST_PROMPTS, Trimmer, time_series


@pytest.fixture
def trim() -> Trimmer:
    words = [f"w{i}" for i in range(50)] + ["[UNK]"]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return Trimmer(tokenizer)


def test_the_issue_keeps_its_head(trim: Trimmer) -> None:
    ids = trim.ids(["w1 w2 w3 w4 w5"])[0]
    assert trim.head(ids, 3).text == "w1 w2 w3 [...]" and trim.head(ids, 3).cut
    assert trim.head(ids, 9).text == "w1 w2 w3 w4 w5" and not trim.head(ids, 9).cut


def test_thoughts_keep_the_latest_steps(trim: Trimmer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opentslm_data, "THOUGHT_TOKENS", 6)
    monkeypatch.setattr(opentslm_data, "PER_THOUGHT_TOKENS", 3)
    steps = trim.ids(["w1 w1", "w2 w2", "w3 w3 w3 w3 w3", "w4 w4"])
    text, used, dropped = trim.thoughts(steps, k=4)
    # Step 4 (2 tokens) and step 3 (cut to 3) fit; step 2 would exceed 6 tokens.
    assert dropped == 2 and used == 5
    assert text.splitlines() == [
        "(Thoughts of steps 1-2 are left out to fit the length limit.)",
        "Step 3: w3 w3 w3 [...]",
        "Step 4: w4 w4",
    ]
    # Only the first k thoughts are visible to a checkpoint.
    assert trim.thoughts(steps, k=2)[0] == "Step 1: w1 w1\nStep 2: w2 w2"


ROW = {
    "thought_chars": [10, 20, 30, 40, 50],
    "action_chars": [5, 5, 5, 5, 5],
    "observation_chars": [100, 0, 300, 50, 10],
    "context_chars": [1000, 1200, 1500, 1600, 1700],
    "repeated_action": [False, False, True, False, True],
    "action_kind": ["search", "create", "edit", "run", "edit"],
    "observation_kind": ["output", "output", "edit_rejected", "traceback", "output"],
    "same_file_edit": [False, False, False, False, True],
    "edit_target": ["none", "none", "own_script", "none", "own_script"],
}


def test_nineteen_labelled_series_without_counts() -> None:
    texts, values = time_series(pd.Series(ROW))
    assert len(texts) == len(values) == 19
    assert all(len(v) == 5 and np.isfinite(v).all() for v in values)
    assert texts[0] == "Characters of the agent's thought at each step, it has mean 30.0 and std 14.1:"
    assert np.allclose(values[1], 0)  # a constant length normalises to zeros
    # A 0/1 series says what a 1 means, never how many there are: the model has to read the series.
    assert not any("of 5 steps" in text for text in texts)
    assert values[texts.index("1 marks a step where the agent edited a file:")] == [0, 0, 1, 0, 1]
    assert values[texts.index("1 marks an edit of the same file as the edit before:")] == [0, 0, 0, 0, 1]
    assert values[texts.index("1 marks a step where the agent edited a file it had created:")] == [0, 0, 1, 0, 1]
    assert values[texts.index("1 marks a step where the agent edited a source file:")] == [0] * 5


def test_the_context_decides_the_text_beside_the_series() -> None:
    from agent_ecg.opentslm_data import SETTING, SETTING_NO_ISSUE, post_prompts, pre_prompt

    full = pre_prompt("Fix the parser.", "Step 1: look around")
    assert full.startswith(SETTING) and "Issue:\nFix the parser." in full and "Step 1: look around" in full
    assert "Issue:" not in pre_prompt(None, "Step 1: look around")
    alone = pre_prompt(None, None)
    assert alone == f"{SETTING_NO_ISSUE}\n\nPer-step signals of the run so far:"
    assert post_prompts("full") == POST_PROMPTS
    assert all("above" not in prompt for prompt in post_prompts("none").values())


def _tables(path, trim: Trimmer) -> None:
    status = "In its first 5 steps ... The run will go on for 3 more steps, then it will submit."
    rows = [
        {
            "run_id": run,
            "instance_id": "i1",
            "repository": "o/r",
            "model_name": "m",
            "k": 5,
            "split": split,
            "issue_id": 0,
            **ROW,
            "resolved": split == "train",
            "ending": "submitted",
            "status": status,
            "remaining": 3,
        }
        for run, split in (("r1", "train"), ("r2", "validation"))
    ]
    pd.DataFrame(rows).to_parquet(path / "checkpoints.parquet", index=False)
    thoughts = [["w1 w2", "w3", "w4 w4", "w5", "w6"]] * 2
    pd.DataFrame({"run_id": ["r1", "r2"], "thoughts": thoughts}).to_parquet(path / "thoughts.parquet", index=False)
    pd.DataFrame({"issue_id": [0], "instance_id": ["i1"], "text": ["w7 w8 w9"]}).to_parquet(
        path / "issues.parquet", index=False
    )


@pytest.mark.parametrize("context", ["full", "thoughts", "none"])
def test_export_writes_one_context(tmp_path, trim: Trimmer, context: str) -> None:
    import json

    _tables(tmp_path, trim)
    stats = opentslm_data.export(tmp_path, tmp_path / "out", tokenizer=trim.tokenizer, context=context)
    train = pd.read_parquet(tmp_path / "out" / "train.parquet")
    assert json.loads((tmp_path / "out" / "meta.json").read_text()) == {"context": context}
    prompt = train["pre_prompt"][0]
    assert ("Issue:\nw7 w8 w9" in prompt) == (context == "full")
    assert ("Step 5: w6" in prompt) == (context != "none")
    assert len(train["time_series"][0]) == 19 and (stats["prompt_tokens"] > 0).all()
    assert set(stats["split"]) == {"train", "validation"}


def test_post_prompts_name_the_answers() -> None:
    assert POST_PROMPTS["outcome"].endswith("resolved or unresolved.\nAnswer:")
    assert "context_limit" in POST_PROMPTS["ending"] and "Will the agent's" not in POST_PROMPTS["ending"]
