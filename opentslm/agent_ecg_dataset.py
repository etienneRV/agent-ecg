"""OpenTSLM dataset for the agent-ecg checkpoints.

Copy this file into an OpenTSLM checkout at
``src/opentslm/time_series_datasets/agent_ecg/AgentECGQADataset.py`` and point ``AGENT_ECG_DATA`` at
the folder that ``agent-ecg opentslm-export`` wrote (``data/opentslm`` by default).

Each checkpoint gives one sample per task (outcome, ending, status). Environment variables choose the
training set, because OpenTSLM's trainer builds datasets as ``Dataset(split, EOS_TOKEN)``:

``AGENT_ECG_TASKS``          comma-separated tasks (default ``outcome,ending,status``)
``AGENT_ECG_ONE_PER_RUN``    ``1``: train on one random checkpoint per run (default ``0``)
``AGENT_ECG_MAX_TRAIN``      cap on training checkpoints (default: all)
``AGENT_ECG_MAX_EVAL``       cap on validation and test checkpoints (default: all)

Every sample keeps ``run_id``, ``k`` and ``task``, so predictions can be scored with ``agent-ecg evaluate``.
"""

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

TASKS = ("outcome", "ending", "status")
POST_PROMPTS = {}  # filled from the exported files' metadata on first load


def _load(split: str) -> list[dict]:
    root = Path(os.environ.get("AGENT_ECG_DATA", "data/opentslm"))
    table = pd.read_parquet(root / f"{split}.parquet")
    prompts = pd.read_json(root / "post_prompts.json", typ="series").to_dict()
    POST_PROMPTS.update(prompts)
    if split == "train":
        if os.environ.get("AGENT_ECG_ONE_PER_RUN", "0") == "1":
            table = table.groupby("run_id").sample(1, random_state=0)
        cap = os.environ.get("AGENT_ECG_MAX_TRAIN")
    else:
        cap = os.environ.get("AGENT_ECG_MAX_EVAL")
    if cap:
        table = table.sample(min(int(cap), len(table)), random_state=0)
    tasks = [t for t in os.environ.get("AGENT_ECG_TASKS", ",".join(TASKS)).split(",") if t]
    answers = {
        "outcome": lambda r: "resolved" if r["resolved"] else "unresolved",
        "ending": lambda r: r["ending"],
        "status": lambda r: r["status"],
    }
    return [{**row, "task": task, "answer": answers[task](row)} for row in table.to_dict("records") for task in tasks]


class AgentECGQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> tuple[list, list, list]:
        return _load("train"), _load("validation"), _load("test")

    def _get_answer(self, row) -> str:
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        return row["pre_prompt"]

    def _get_post_prompt(self, row) -> str:
        return POST_PROMPTS[row["task"]]

    def _get_text_time_series_prompt_list(self, row) -> list[TextTimeSeriesPrompt]:
        return [
            TextTimeSeriesPrompt(text, list(values))
            for text, values in zip(row["time_series_text"], row["time_series"], strict=True)
        ]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample.update(run_id=row["run_id"], k=int(row["k"]), task=row["task"])
        return sample
