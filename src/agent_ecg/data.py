"""Build the TimeNet release once and flatten it into three small tables.

``checkpoints.parquet``
    One row per checkpoint (the first k steps of a run that is still going): ids, split, the seven
    per-step signals as lists, and the three targets (``resolved``, ``ending``, ``status``).
``thoughts.parquet``
    One row per run: the agent's thoughts for the steps of its latest checkpoint. A checkpoint at k
    steps sees the first k of them.
``issues.parquet``
    One row per issue annotation: the issue text a run's agent read.

Every value comes from the TimeNet release; nothing here reads the raw Hugging Face files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET = "nebius/swe_agent_trajectories"
SIGNALS = (
    "thought_chars",
    "action_chars",
    "observation_chars",
    "context_chars",
    "repeated_action",
    "action_kind",
    "observation_kind",
)
CHECKPOINTS = (5, 10, 20, 30)


def build(registry: Path) -> Path:
    """Build the release into ``registry`` (or reuse it) and return its directory."""
    import os

    import timenet

    # An isolated build installs timenet-connectors from PyPI, which lacks this connector until
    # OpenTSLM/TimeNet#832 is released; build with the connector installed here instead.
    os.environ.setdefault("TIMENET_ISOLATION", "off")
    return timenet.build(DATASET, local_registry=registry)


def export(release: Path, out: Path) -> None:
    """Flatten a built release into the three tables under ``out``."""
    from timenet.reader import TimeFReader
    from timenet.registry import open_local_version
    from timenet.types.text import read_value

    from agent_ecg.status import parse_status

    out.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = {}
    thoughts: dict[str, list[str]] = {}
    issues: dict[int, dict] = {}
    with TimeFReader(open_local_version(release)) as reader:
        dataset = reader.read()
        for task in dataset.iter_tasks():
            record = task.inputs[0]
            row = rows.get(id(record))
            if row is None:
                meta = record.metadata
                issue, thought = task.input_annotations
                if id(issue) not in issues:
                    issues[id(issue)] = {
                        "issue_id": len(issues),
                        "instance_id": issue.metadata["instance_id"],
                        "text": read_value(issue.value),
                    }
                row = {
                    "run_id": meta["run_id"],
                    "instance_id": meta["instance_id"],
                    "repository": meta["repository"],
                    "model_name": meta["model_name"],
                    "k": meta["checkpoint_steps"],
                    "split": str(task.split),
                    "issue_id": issues[id(issue)]["issue_id"],
                }
                for signal in record.sources[0].signals:
                    row[signal.name] = signal.to_arrow().to_pylist()
                # Checkpoints of a run come in ascending order; keep the longest prefix of thoughts.
                thoughts[meta["run_id"]] = read_value(thought.value)
                rows[id(record)] = row
            target = read_value(task.targets[0])
            label_set = getattr(task, "label_set", None)
            if label_set is None:
                row["status"] = target
                row["remaining"] = parse_status(target).remaining
            elif label_set.name == "run_outcome":
                row["resolved"] = target == "resolved"
            else:
                row["ending"] = target
    table = pd.DataFrame(rows.values())
    table.to_parquet(out / "checkpoints.parquet", index=False)
    pd.DataFrame({"run_id": list(thoughts), "thoughts": list(thoughts.values())}).to_parquet(
        out / "thoughts.parquet", index=False
    )
    pd.DataFrame(issues.values()).to_parquet(out / "issues.parquet", index=False)


def load_checkpoints(data: Path) -> pd.DataFrame:
    """The checkpoint table: one row per checkpoint, with its signals and targets."""
    return pd.read_parquet(data / "checkpoints.parquet")


def load_issues(data: Path) -> dict[int, str]:
    """The issue texts, keyed by ``issue_id``."""
    table = pd.read_parquet(data / "issues.parquet")
    return dict(zip(table["issue_id"].astype(int), table["text"], strict=True))


def load_thoughts(data: Path) -> dict[str, list[str]]:
    """The thoughts of each run, keyed by run id, for its latest checkpoint's steps."""
    table = pd.read_parquet(data / "thoughts.parquet")
    return dict(zip(table["run_id"], (list(t) for t in table["thoughts"]), strict=True))
