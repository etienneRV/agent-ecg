"""Build the TimeNet release once and flatten it into three small tables.

``checkpoints.parquet``
    One row per checkpoint (the first k steps of a run that is still going): ids, split, the nine
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
# The raw Hugging Face dataset behind the release, at the revision the connector pins (the demo and the channel
# hunt read step text from it).
RAW_DATASET, RAW_REVISION = "nebius/SWE-agent-trajectories", "68195a1450865274106246d0d0296a1d6807b88e"
SIGNALS = (
    "thought_chars",
    "action_chars",
    "observation_chars",
    "context_chars",
    "repeated_action",
    "action_kind",
    "observation_kind",
    "same_file_edit",
    "edit_target",
)
CHECKPOINTS = (5, 10, 15, 20, 30)


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
    rows: dict[str, dict] = {}
    thoughts: dict[str, list[str]] = {}
    issues: dict[tuple[str, str], dict] = {}
    with TimeFReader(open_local_version(release)) as reader:
        dataset = reader.read()
        for task in dataset.iter_tasks():
            record = task.inputs[0]
            row = rows.get(str(record.id))
            if row is None:
                meta = record.metadata
                issue, thought = task.input_annotations
                text = read_value(issue.value)
                issue_key = (issue.metadata["instance_id"], text)
                if issue_key not in issues:
                    issues[issue_key] = {"issue_id": len(issues), "instance_id": issue_key[0], "text": text}
                row = {
                    "run_id": meta["run_id"],
                    "instance_id": meta["instance_id"],
                    "repository": meta["repository"],
                    "model_name": meta["model_name"],
                    "k": meta["checkpoint_steps"],
                    "split": str(task.split),
                    "issue_id": issues[issue_key]["issue_id"],
                }
                for signal in record.sources[0].signals:
                    row[signal.name] = signal.to_arrow().to_pylist()
                # Keep the longest prefix of thoughts a run's checkpoints show, whatever their order.
                seen = read_value(thought.value)
                if len(seen) > len(thoughts.get(meta["run_id"], ())):
                    thoughts[meta["run_id"]] = seen
                rows[str(record.id)] = row
            target = read_value(task.targets[0])
            label_set = getattr(task, "label_set", None)
            if label_set is None:
                row["status"] = target
                row["remaining"] = parse_status(target).remaining
            elif _label_set_name(label_set) == "run_outcome":
                row["resolved"] = target == "resolved"
            else:
                row["ending"] = target
    table = pd.DataFrame(rows.values())
    table.to_parquet(out / "checkpoints.parquet", index=False)
    pd.DataFrame({"run_id": list(thoughts), "thoughts": list(thoughts.values())}).to_parquet(
        out / "thoughts.parquet", index=False
    )
    pd.DataFrame(issues.values()).to_parquet(out / "issues.parquet", index=False)


def _label_set_name(label_set) -> str | None:
    """A label set's name: since the connector API update it is the set's ``name`` annotation."""
    from timenet.types.text import read_value

    return next((read_value(a.value) for a in label_set.annotations if a.key == "name"), None)


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
