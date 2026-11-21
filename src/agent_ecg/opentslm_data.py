"""Convert the checkpoints into OpenTSLM samples.

OpenTSLM (github.com/OpenTSLM/OpenTSLM) trains on samples of the form::

    {"pre_prompt": str, "time_series_text": [str, ...], "time_series": [[float, ...], ...],
     "post_prompt": str, "answer": str}

Each time series enters the model as its own short sequence (one token per 4 values), labelled by
its text. Here a checkpoint gives one sample per task, sharing everything but the post-prompt and the
answer:

- **pre_prompt:** the setting, then, by ``context``, the issue (first ``ISSUE_TOKENS`` tokens) and
  the agent's thoughts so far (``full``), the thoughts only (``thoughts``), or neither (``none``: the
  ECG-only variant, where the model reads the step series and nothing else). Thoughts are kept from
  the latest step backwards up to ``THOUGHT_TOKENS`` tokens, each at most ``PER_THOUGHT_TOKENS``.
  Older steps are dropped with a note; the time series still cover them.
- **time series:** 19 per-step series. Four lengths, z-normalised with their mean and standard
  deviation in the label (as OpenTSLM's HAR and sleep datasets do), and fifteen 0/1 indicators (a
  repeated command, five command kinds, five reply kinds, an edit of the same file as the edit
  before, and three kinds of file an edit changes), left as they are. An indicator's label says what
  a 1 means and nothing more: a count in the label would let the model answer from the text and skip
  the series.
- **post_prompt and answer:** the TimeNet task's prompt and target, for "Will it pass?"
  (``resolved`` / ``unresolved``), "How will it end?" (one of six labels) and the status report.
  Without the issue in the prompt, "the GitHub issue above" becomes "a GitHub issue".

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

CONTEXTS = ("full", "thoughts", "none")
_STEP_HABIT = (
    "one step at a time: at each step it writes a thought, runs one command, and reads the environment's reply."
)
SETTING = (
    f"You are monitoring a software engineering agent (SWE-agent) while it works. The agent is trying "
    f"to resolve the GitHub issue below, {_STEP_HABIT}"
)
SETTING_NO_ISSUE = (
    f"You are monitoring a software engineering agent (SWE-agent) while it works. The agent is trying "
    f"to resolve a GitHub issue, {_STEP_HABIT}"
)
POST_PROMPTS = {
    "outcome": f"{RESOLVE_PROMPT} Answer with one word: resolved or unresolved.\nAnswer:",
    "ending": (
        f"{ENDING_PROMPT} Answer with one of: {', '.join(label.name for label in ENDING_LABELS.labels)}.\nAnswer:"
    ),
    "status": f"{STATUS_PROMPT}\nReport:",
}
_ISSUE_ABOVE = "the GitHub issue above"


def post_prompts(context: str = "full") -> dict[str, str]:
    """The post-prompt of each task; without the issue in the prompt, it no longer points "above"."""
    if context == "full":
        return dict(POST_PROMPTS)
    missing = [task for task, prompt in POST_PROMPTS.items() if _ISSUE_ABOVE not in prompt]
    if missing:
        raise ValueError(f"the connector's prompts for {missing} no longer say {_ISSUE_ABOVE!r}; update post_prompts()")
    return {task: prompt.replace(_ISSUE_ABOVE, "a GitHub issue") for task, prompt in POST_PROMPTS.items()}


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
EDIT_TARGETS = {
    "source": "the agent edited a source file",
    "test": "the agent edited a test file",
    "own_script": "the agent edited a file it had created",
}


def time_series(row: pd.Series | dict) -> tuple[list[str], list[list[float]]]:
    """The 19 labelled per-step series of a checkpoint."""
    texts, values = [], []
    for name, label in LENGTHS.items():
        series = np.asarray(row[name], float)
        mean, std = series.mean(), series.std()
        texts.append(f"{label}, it has mean {mean:.1f} and std {std:.1f}:")
        values.append(((series - mean) / max(std, 1e-6)).tolist())
    flags = [("1 marks a step whose command repeats an earlier one", np.asarray(row["repeated_action"], float))]
    actions, replies = np.asarray(row["action_kind"]), np.asarray(row["observation_kind"])
    targets = np.asarray(row["edit_target"])
    flags += [
        (f"1 marks a step where the agent {what}", (actions == kind).astype(float)) for kind, what in ACTIONS.items()
    ]
    flags += [(f"1 marks a step where {what}", (replies == kind).astype(float)) for kind, what in REPLIES.items()]
    flags.append(("1 marks an edit of the same file as the edit before", np.asarray(row["same_file_edit"], float)))
    flags += [(f"1 marks a step where {what}", (targets == kind).astype(float)) for kind, what in EDIT_TARGETS.items()]
    for label, series in flags:
        texts.append(f"{label}:")
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

    def lengths(self, texts: list[str], chunk: int = 4096) -> list[int]:
        """Token counts, a chunk at a time, so the ids of a whole split never sit in memory together."""
        out: list[int] = []
        for start in range(0, len(texts), chunk):
            out += [
                len(e.ids) for e in self.tokenizer.encode_batch(texts[start : start + chunk], add_special_tokens=False)
            ]
        return out

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


def pre_prompt(issue: str | None, thoughts: str | None) -> str:
    """The text before the series: the setting, then the issue and the thoughts when they are given."""
    parts = [SETTING if issue is not None else SETTING_NO_ISSUE]
    if issue is not None:
        parts.append(f"Issue:\n{issue}")
    if thoughts is not None:
        parts.append(f"The agent's thoughts so far:\n{thoughts}")
    parts.append("Per-step signals of the run so far:")
    return "\n\n".join(parts)


class Prompts:
    """The pre-prompt of any prefix of a run, by context: the same text for the export and for a replay."""

    def __init__(self, data_dir: Path, context: str = "full", tokenizer: Tokenizer | None = None) -> None:
        """Load what the context needs from the tables that ``agent-ecg prepare`` wrote.

        Args:
            data_dir: The flattened tables.
            context: What text joins the series: ``full`` (issue and thoughts), ``thoughts`` or ``none``.
            tokenizer: The tokenizer that counts the budgets; Llama 3.2's by default.

        Raises:
            ValueError: If the context is not one of ``CONTEXTS``.
        """
        if context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS}, got {context!r}")
        if tokenizer is None:
            from huggingface_hub import hf_hub_download

            tokenizer = Tokenizer.from_file(hf_hub_download(*TOKENIZER))
        self.context = context
        self.trim = Trimmer(tokenizer)
        self.with_issue, self.with_thoughts = context == "full", context in ("full", "thoughts")
        self.thoughts = data.load_thoughts(data_dir) if self.with_thoughts else {}
        self.issues: dict[int, Trimmed] = {}
        if self.with_issue:
            text = data.load_issues(data_dir)
            ids = dict(zip(text, self.trim.ids(list(text.values())), strict=True))
            self.issues = {i: self.trim.head(v, ISSUE_TOKENS) for i, v in ids.items()}
        self._step_ids: dict[str, list[list[int]]] = {}

    def __call__(self, run_id: str, issue_id: int, k: int) -> tuple[str, dict]:
        """The pre-prompt of the first ``k`` steps of a run, and what was cut to fit the budgets."""
        thoughts, used, dropped = None, 0, 0
        if self.with_thoughts:
            if run_id not in self._step_ids:
                if len(self._step_ids) >= 2000:
                    self._step_ids.clear()
                self._step_ids[run_id] = self.trim.ids(self.thoughts[run_id])
            thoughts, used, dropped = self.trim.thoughts(self._step_ids[run_id], k)
        issue = self.issues[issue_id] if self.with_issue else None
        stats = {
            "issue_tokens": issue.tokens if issue else 0,
            "issue_cut": issue.cut if issue else False,
            "thought_tokens": used,
            "steps_dropped": dropped,
        }
        return pre_prompt(issue.text if issue else None, thoughts), stats


def export(data_dir: Path, out: Path, tokenizer: Tokenizer | None = None, context: str = "full") -> pd.DataFrame:
    """Write ``{split}.parquet`` under ``out``: one row per checkpoint with its OpenTSLM inputs and targets.

    Args:
        data_dir: The flattened tables that ``agent-ecg prepare`` wrote.
        out: Where the samples go, with ``post_prompts.json`` and ``meta.json`` (the context).
        tokenizer: The tokenizer that counts the budgets; Llama 3.2's by default.
        context: What text joins the series: ``full`` (issue and thoughts), ``thoughts`` or ``none``.

    Returns:
        Per-checkpoint trimming statistics, with each prompt's length in tokens.
    """
    prompts = Prompts(data_dir, context, tokenizer)
    checkpoints = data.load_checkpoints(data_dir)
    post = post_prompts(context)
    out.mkdir(parents=True, exist_ok=True)
    (out / "post_prompts.json").write_text(json.dumps(post, indent=2))
    (out / "meta.json").write_text(json.dumps({"context": context}, indent=2))
    stats = []
    for split, table in checkpoints.groupby("split"):
        rows, split_stats = [], []
        for _, row in table.iterrows():
            text, trimmed = prompts(row["run_id"], row["issue_id"], row["k"])
            series_text, series = time_series(row)
            rows.append(
                {
                    "run_id": row["run_id"],
                    "k": row["k"],
                    "repository": row["repository"],
                    "pre_prompt": text,
                    "time_series_text": series_text,
                    "time_series": series,
                    "resolved": row["resolved"],
                    "ending": row["ending"],
                    "status": row["status"],
                }
            )
            split_stats.append({"split": split, "k": row["k"], **trimmed})
        # The prompt's length: its text in tokens (pre-prompt, series labels, the pass/fail question),
        # plus one token per four values of each series.
        texts = ["\n".join([r["pre_prompt"], *r["time_series_text"], post["outcome"]]) for r in rows]
        for entry, length, r in zip(split_stats, prompts.trim.lengths(texts), rows, strict=True):
            entry["prompt_tokens"] = length + sum(-(-len(values) // 4) for values in r["time_series"])
        stats += split_stats
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet", index=False)
        print(f"  {split}: {len(rows)} checkpoints", flush=True)
    return pd.DataFrame(stats)
