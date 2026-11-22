import json

import pytest

from agent_ecg import demo
from agent_ecg.demo import RawRun, Step, braille, heartbeat, outcome, parse_steps


def _reply(text: str, opened: str = "n/a") -> str:
    return f"{text}\n\n(Open file: {opened})\n(Current directory: /repo)\nbash-$"


def test_steps_pair_each_agent_message_with_its_reply() -> None:
    messages = [
        {"role": "system", "text": ""},
        {"role": "user", "text": "ISSUE: ..."},
        {"role": "ai", "text": "Let me look.\n```\nopen src/a.py\n```"},
        {"role": "user", "text": _reply("[File: /repo/src/a.py (10 lines total)]", "/repo/src/a.py")},
        {"role": "ai", "text": "Done.\n```\nsubmit\n```"},
    ]
    steps = parse_steps(messages)
    assert [s.command for s in steps] == ["open src/a.py", "submit"]
    assert steps[0].thought == "Let me look." and steps[0].open_file == "/repo/src/a.py"
    assert steps[0].reply == "[File: /repo/src/a.py (10 lines total)]"  # the footer is gone
    assert steps[1].reply == "" and steps[1].open_file is None


def test_braille_draws_a_rising_line_from_bottom_left_to_top_right() -> None:
    rows = braille([0, 50, 100], width=10, height=2, low=0, high=100)
    assert [len(r) for r in rows] == [10, 10]
    assert rows[1][0] != chr(0x2800) and rows[0][0] == chr(0x2800)  # starts at the bottom
    assert rows[0][-1] != chr(0x2800) and rows[1][-1] == chr(0x2800)  # ends at the top
    gaps = braille([None, None, 50, 50], width=8, height=1, low=0, high=100)[0]
    assert gaps[:3] == chr(0x2800) * 3 and gaps[-1] != chr(0x2800)  # nothing drawn before the first value


def test_the_heartbeat_spike_follows_health() -> None:
    strong, weak = heartbeat([100], beats=1), heartbeat([10], beats=1)
    assert len(strong) == len(demo._BEAT) and max(strong) > 3 * max(weak)
    assert len(heartbeat([None, 80, 60, 40], beats=3)) == 3 * len(demo._BEAT)


def _run(resolved: bool = False, steps: int = 12) -> RawRun:
    step = Step("thinking", "edit 1:2", "Your proposed edit has introduced new syntax error(s).", "/repo/a.py")
    return RawRun(
        "train-00000-of-00012:1", "o__r-1", "swe-agent-llama-70b", resolved, "submitted (exit_context)", [step] * steps
    )


def _monitor(critical_from: int | None) -> dict[int, dict]:
    out = {}
    for k in range(5, 11):
        critical = critical_from is not None and k >= critical_from
        out[k] = {
            "step": k,
            "health": 10 if critical else 80,
            "state": "critical" if critical else "stable",
            "p_calibrated": 0.02 if critical else 0.2,
            "events": ["edit rejected"],
            **({"recommendation": "Stop the agent to save credits."} if critical else {}),
        }
    return out


def test_the_end_card_compares_the_alarm_with_what_happened() -> None:
    text = outcome(_run(steps=40), _monitor(critical_from=8))
    assert text.startswith("ALARM at step 8. The real run went on to step 40 (32 more steps)")
    assert "saved 32 of its 40 steps (80%)" in text
    assert outcome(_run(resolved=True), _monitor(None)).startswith("No alarm. The real run took 12 steps")


def test_a_frame_renders_the_agent_and_the_monitor(tmp_path) -> None:
    run, monitor = _run(steps=40), _monitor(critical_from=8)
    text = demo.render(run, monitor, 9, svg=tmp_path / "frame.svg", width=140, height=44)
    assert "replay, one step at a time" in text and "HEALTH  10" in text and "CRITICAL" in text
    assert "Stop the agent" in text and "step  9" in text
    assert (tmp_path / "frame.svg").read_text().startswith("<svg")
    assert "warming up" in demo.render(run, monitor, 3, width=140, height=44)
    end = demo.render(run, monitor, 10, width=140, height=44)
    assert "what really happened" in end


def test_replay_files_load_by_run_and_step(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps({"run_id": "a", "step": k, "state": "stable"}) for k in (5, 6)) + "\n")
    assert list(demo.load_replay(path)["a"]) == [5, 6]


def test_typical_failure_counts_validation_runs_still_going(tmp_path):
    import pandas as pd

    table = tmp_path / "checkpoints.parquet"
    pd.DataFrame(
        {
            "split": ["validation"] * 4 + ["train"],
            "k": [5, 5, 5, 10, 5],
            "remaining": [3, 20, 30, 25, 40],  # runs of 8, 25 and 35 steps; the k=10 row repeats the second run
            "resolved": [True, False, True, False, False],
        }
    ).to_parquet(table)
    assert demo.typical_failure(6, table) == pytest.approx(1 / 3)  # all three validation runs still going, one fails
    assert demo.typical_failure(10, table) == pytest.approx(0.5)  # the 8-step run has ended
    assert demo.typical_failure(40, table) is None  # no validation run lasts that long
    assert demo.typical_failure(6, tmp_path / "missing.parquet") is None
