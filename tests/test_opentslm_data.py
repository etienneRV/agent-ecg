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


def test_fifteen_labelled_series() -> None:
    row = pd.Series(
        {
            "thought_chars": [10, 20, 30, 40, 50],
            "action_chars": [5, 5, 5, 5, 5],
            "observation_chars": [100, 0, 300, 50, 10],
            "context_chars": [1000, 1200, 1500, 1600, 1700],
            "repeated_action": [False, False, True, False, True],
            "action_kind": ["search", "read", "edit", "run", "edit"],
            "observation_kind": ["output", "output", "edit_rejected", "traceback", "output"],
        }
    )
    texts, values = time_series(row)
    assert len(texts) == len(values) == 15
    assert all(len(v) == 5 and np.isfinite(v).all() for v in values)
    assert texts[0] == "Characters of the agent's thought at each step, it has mean 30.0 and std 14.1:"
    assert np.allclose(values[1], 0)  # a constant length normalises to zeros
    edits = texts.index("1 marks a step where the agent edited a file (2 of 5 steps):")
    assert values[edits] == [0, 0, 1, 0, 1]


def test_post_prompts_name_the_answers() -> None:
    assert POST_PROMPTS["outcome"].endswith("resolved or unresolved.\nAnswer:")
    assert "context_limit" in POST_PROMPTS["ending"] and "Will the agent's" not in POST_PROMPTS["ending"]
