"""Convert the checkpoints into OpenTSLM samples.

OpenTSLM (github.com/OpenTSLM/OpenTSLM) trains on samples of the form::

    {"pre_prompt": str, "time_series_text": [str, ...], "time_series": [[float, ...], ...],
     "post_prompt": str, "answer": str}

Each time series enters the model as its own short sequence (one token per 4 values), labelled by
its text. Here a checkpoint gives one sample per task, sharing everything but the post-prompt and the
answer:

- **pre_prompt:** the setting, the issue (first ``ISSUE_TOKENS`` tokens) and the agent's thoughts so
  far, kept from the latest step backwards up to ``THOUGHT_TOKENS`` tokens, each at most
  ``PER_THOUGHT_TOKENS``. Older steps are dropped with a note; the time series still cover them.
- **time series:** 15 per-step series. Four lengths, z-normalised with their mean and standard
  deviation in the label (as OpenTSLM's HAR and sleep datasets do), and eleven 0/1 indicators (a
  repeated command, five command kinds, five reply kinds), left as they are.
- **post_prompt and answer:** the TimeNet task's prompt and target, for "Will it pass?"
  (``resolved`` / ``unresolved``), "How will it end?" (one of six labels) and the status report.

Token counts use the Llama 3.2 tokenizer, OpenTSLM's default base model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from timenet_connectors.datasets.nebius.swe_agent_trajectories.tasks import (
    ENDING_LABELS,
    ENDING_PROMPT,
    RESOLVE_PROMPT,
    STATUS_PROMPT,
)
from tokenizers import Tokenizer

from agent_ecg import data

TOKENIZER = ("unsloth/Llama-3.2-1B", "tokenizer.json")  # same tokenizer as meta-llama/Llama-3.2-1B, not gated
ISSUE_TOKENS = 512
THOUGHT_TOKENS = 1024
PER_THOUGHT_TOKENS = 256

SETTING = (
    "You are monitoring a software engineering agent (SWE-agent) while it works. The agent is trying "
    "to resolve the GitHub issue below, one step at a time: at each step it "
    "writes a thought, runs one command, and reads the environment's reply."
)
POST_PROMPTS = {
    "outcome": f"{RESOLVE_PROMPT} Answer with one word: resolved or unresolved.\nAnswer:",
    "ending": (
        f"{ENDING_PROMPT} Answer with one of: {', '.join(label.name for label in ENDING_LABELS.labels)}.\nAnswer:"
    ),
    "status": f"{STATUS_PROMPT}\nReport:",
}
LENGTHS = {
    "thought_chars": "Characters of the agent's thought at each step",
    "action_chars": "Characters of the agent's command at each step",
    "observation_chars": "Characters of the environment's reply at each step",
    "context_chars": "Characters of the whole conversation after each step",
}
ACTIONS = {
    "edit": "edited a file",
    "create": "created a file",
    "run": "ran code (python or pytest)",
    "read": "read a file (open, goto, scroll, cat)",
    "search": "searched the repository (search, find, grep, ls)",
}
REPLIES = {
    "edit_rejected": "the linter rejected the agent's edit",
    "traceback": "the reply showed a Python traceback",
    "command_error": "the reply was a command error",
    "timeout": "the command timed out",
    "silent_success": "the command succeeded and printed nothing",
}


def time_series(row: pd.Series) -> tuple[list[str], list[list[float]]]:
    """The 15 labelled per-step series of a checkpoint."""
    texts, values = [], []
    for name, label in LENGTHS.items():
        series = np.asarray(row[name], float)
        mean, std = series.mean(), series.std()
        texts.append(f"{label}, it has mean {mean:.1f} and std {std:.1f}:")
        values.append(((series - mean) / max(std, 1e-6)).tolist())
    flags = [
        (
            "repeated_action",
            "1 marks a step whose command repeats an earlier one",
            np.asarray(row["repeated_action"], float),
        )
    ]
    actions, replies = np.asarray(row["action_kind"]), np.asarray(row["observation_kind"])
    flags += [
        (kind, f"1 marks a step where the agent {what}", (actions == kind).astype(float))
        for kind, what in ACTIONS.items()
    ]
    flags += [(kind, f"1 marks a step where {what}", (replies == kind).astype(float)) for kind, what in REPLIES.items()]
    for _, label, series in flags:
        texts.append(f"{label} ({int(series.sum())} of {len(series)} steps):")
        values.append(series.tolist())
    return texts, values


@dataclass(frozen=True)
class Trimmed:
    text: str
    tokens: int
    cut: bool


class Trimmer:
    """Cut the issue and the thoughts to the token budgets."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def head(self, ids: list[int], budget: int) -> Trimmed:
        if len(ids) <= budget:
            return Trimmed(self.tokenizer.decode(ids), len(ids), False)
        return Trimmed(self.tokenizer.decode(ids[:budget]) + " [...]", budget, True)

    def thoughts(self, step_ids: list[list[int]], k: int) -> tuple[str, int, int]:
        """The latest of the first ``k`` thoughts that fit the budget.

        Returns:
            The text, its token count, and how many of the earliest steps were dropped.
        """
        kept: list[str] = []
        used = 0
        first = k
        for step in range(k - 1, -1, -1):
            piece = self.head(step_ids[step], PER_THOUGHT_TOKENS)
            if used + piece.tokens > THOUGHT_TOKENS and kept:
                break
            kept.append(f"Step {step + 1}: {piece.text}")
            used += piece.tokens
            first = step
        lines = kept[::-1]
        if first > 0:
            lines.insert(0, f"(Thoughts of steps 1-{first} are left out to fit the length limit.)")
        return "\n".join(lines), used, first


def pre_prompt(issue: str, thoughts: str) -> str:
    return f"{SETTING}\n\nIssue:\n{issue}\n\nThe agent's thoughts so far:\n{thoughts}\n\nPer-step signals of the run so far:"


def export(data_dir: Path, out: Path, tokenizer: Tokenizer | None = None) -> pd.DataFrame:
    """Write ``{split}.parquet`` under ``out``: one row per checkpoint with its OpenTSLM inputs and targets.

    Returns:
        Per-checkpoint trimming statistics.
    """
    from huggingface_hub import hf_hub_download

    tokenizer = tokenizer or Tokenizer.from_file(hf_hub_download(*TOKENIZER))
    trim = Trimmer(tokenizer)
    checkpoints = data.load_checkpoints(data_dir)
    thoughts = data.load_thoughts(data_dir)
    issue_text = data.load_issues(data_dir)
    issue_ids = dict(zip(issue_text, trim.ids(list(issue_text.values())), strict=True))
    issues = {i: trim.head(ids, ISSUE_TOKENS) for i, ids in issue_ids.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "post_prompts.json").write_text(json.dumps(POST_PROMPTS, indent=2))
    stats = []
    for split, table in checkpoints.groupby("split"):
        rows = []
        step_ids: dict[str, list[list[int]]] = {}
        for _, row in table.iterrows():
            run = row["run_id"]
            if run not in step_ids:
                step_ids[run] = trim.ids(thoughts[run])
            text, used, dropped = trim.thoughts(step_ids[run], row["k"])
            issue = issues[row["issue_id"]]
            series_text, series = time_series(row)
            rows.append(
                {
                    "run_id": run,
                    "k": row["k"],
                    "repository": row["repository"],
                    "pre_prompt": pre_prompt(issue.text, text),
                    "time_series_text": series_text,
                    "time_series": series,
                    "resolved": row["resolved"],
                    "ending": row["ending"],
                    "status": row["status"],
                }
            )
            stats.append(
                {
                    "split": split,
                    "k": row["k"],
                    "issue_tokens": issue.tokens,
                    "issue_cut": issue.cut,
                    "thought_tokens": used,
                    "steps_dropped": dropped,
                }
            )
            if len(step_ids) > 2000:
                step_ids.clear()
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet", index=False)
        print(f"  {split}: {len(rows)} checkpoints", flush=True)
    return pd.DataFrame(stats)
