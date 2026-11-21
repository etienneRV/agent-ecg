"""The terminal ECG: a recorded run, replayed step by step beside the monitor that watches it.

    uv run agent-ecg replay --top 3 --out results/replay.signals.falling.jsonl
    uv run agent-ecg demo results/replay.signals.falling.jsonl                    # play the first run
    uv run agent-ecg demo results/replay.signals.falling.jsonl --at 15 --svg ecg.svg   # one frame, for a post

The left pane shows what the agent did at each step: its thought, its command and the start of the
reply, read from the run's raw trajectory on Hugging Face (the revision the connector pins). The right
pane shows the monitor: a heartbeat whose beats shrink and change colour with health, the health line,
the chance of failure, the state, and the recommendation once the alarm is on. Every number on the
right comes from the replay file, computed one step at a time: at step k the model had seen only the
first k steps. The end card says what really happened to the run.

It is a replay of a recorded run, not a live one, and the header says so.
"""

from __future__ import annotations

import io
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import (
    action_kind,
    observation_kind,
    open_file,
    split_message,
)

DATASET = "nebius/SWE-agent-trajectories"
REVISION = "68195a1450865274106246d0d0296a1d6807b88e"  # the revision the connector pins
_COLUMNS = ["instance_id", "model_name", "target", "trajectory", "exit_status"]
_FOOTER = re.compile(r"\s*\(Open file: [^)\n]*\)\s*\(Current directory: [^)\n]*\)\s*bash-\$\s*$")
_NO_FILE = "n/a"
_BAD_REPLIES = {"edit_rejected", "traceback", "command_error", "timeout"}
STYLES = {"stable": "green", "watch": "yellow", "low": "dark_orange", "critical": "bold red", None: "grey50"}
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))  # braille dot bits by [row][column]
_BEAT = (0, 0, 0.06, 0.1, 0.06, 0, 0, -0.12, 1.0, -0.35, 0, 0, 0.12, 0.2, 0.12, 0, 0, 0)  # P wave, QRS, T wave
_AGENT_STEPS = 3


@dataclass(frozen=True)
class Step:
    """One step of a run: the agent's thought and command, and the environment's reply."""

    thought: str
    command: str
    reply: str
    open_file: str | None


@dataclass(frozen=True)
class RawRun:
    """A run as the release recorded it."""

    run_id: str
    instance_id: str
    model_name: str
    resolved: bool
    exit_status: str
    steps: list[Step]


def parse_steps(messages: Sequence[Mapping]) -> list[Step]:
    """The steps of a trajectory: each agent message with the reply to it (the last one has none).

    Raises:
        ValueError: If the messages after the system prompt and the issue do not alternate agent, reply.
    """
    body = list(messages)[2:]
    steps = []
    for i in range(0, len(body), 2):
        agent, reply = body[i], body[i + 1] if i + 1 < len(body) else None
        if agent.get("role") != "ai" or (reply is not None and reply.get("role") != "user"):
            raise ValueError(f"message {i + 2} breaks the agent/reply alternation")
        thought, command = split_message(agent.get("text") or "")
        text = (reply or {}).get("text") or ""
        opened = open_file(text)
        steps.append(
            Step(thought, command, _FOOTER.sub("", text).strip(), None if opened in (None, _NO_FILE) else opened)
        )
    return steps


def fetch_run(run_id: str) -> RawRun:
    """Read one run from the release's parquet shard (downloaded once, then cached by huggingface_hub).

    Raises:
        ValueError: If the run id is not ``<shard>:<row>`` or the row is not in the shard.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shard, _, row = run_id.partition(":")
    if not shard.startswith("train-") or not row.isdigit():
        raise ValueError(f"run ids look like train-00011-of-00012:5254, got {run_id!r}")
    path = hf_hub_download(DATASET, f"data/{shard}.parquet", repo_type="dataset", revision=REVISION)
    parquet, index = pq.ParquetFile(path), int(row)
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if index < size:
            record = parquet.read_row_group(group, columns=_COLUMNS).slice(index, 1).to_pylist()[0]
            break
        index -= size
    else:
        raise ValueError(f"{run_id}: the shard holds fewer rows")
    trajectory = record["trajectory"]
    messages = json.loads(trajectory) if isinstance(trajectory, str) else trajectory
    return RawRun(
        run_id=run_id,
        instance_id=record["instance_id"],
        model_name=record["model_name"],
        resolved=bool(record["target"]),
        exit_status=record["exit_status"],
        steps=parse_steps(messages),
    )


def load_replay(path: Path) -> dict[str, dict[int, dict]]:
    """A replay file (``agent-ecg replay --out`` or ``opentslm/replay.py --out``), by run and step."""
    runs: dict[str, dict[int, dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            runs.setdefault(entry["run_id"], {})[int(entry["step"])] = entry
    return runs


def braille(values: Sequence[float | None], width: int, height: int, low: float, high: float) -> list[str]:
    """Draw a series as a line in ``width`` x ``height`` braille characters (2 x 4 dots each).

    The series spans the full width; a ``None`` leaves its part of the width blank.
    """
    columns, rows = width * 2, height * 4
    grid = [[0] * width for _ in range(height)]
    n = len(values)
    previous = None
    for cx in range(columns):
        x = cx * (n - 1) / (columns - 1) if n > 1 else 0.0
        left, right = int(x), min(int(x) + 1, n - 1)
        if n == 0 or values[left] is None or values[right] is None:
            previous = None
            continue
        value = values[left] + (values[right] - values[left]) * (x - left)  # type: ignore[operator]
        y = round((high - min(max(value, low), high)) / (high - low) * (rows - 1))
        for dot in range(
            min(y, previous if previous is not None else y), max(y, previous if previous is not None else y) + 1
        ):
            grid[dot // 4][cx // 2] |= _DOTS[dot % 4][cx % 2]
        previous = y
    return ["".join(chr(0x2800 + bits) for bits in row) for row in grid]


def heartbeat(health: Sequence[float | None], beats: int) -> list[float]:
    """An ECG-like trace, one beat per step: the spike follows health (grey and medium before scoring)."""
    trace: list[float] = []
    for h in list(health)[-beats:]:
        amplitude = 0.55 if h is None else 0.12 + 0.88 * h / 100
        trace += [v * amplitude if v in (1.0, -0.35) else v for v in _BEAT]
    return trace


def first_alarm(monitor: Mapping[int, Mapping]) -> int | None:
    """The first step the replay marks critical."""
    return next((k for k in sorted(monitor) if monitor[k]["state"] == "critical"), None)


def outcome(run: RawRun, monitor: Mapping[int, Mapping]) -> str:
    """What really happened, against what the monitor said."""
    n, alarm = len(run.steps), first_alarm(monitor)
    result = "resolved the issue" if run.resolved else "did not resolve the issue"
    if alarm is None:
        return f"No alarm. The real run took {n} steps, ended '{run.exit_status}' and {result}."
    text = (
        f"ALARM at step {alarm}. The real run went on to step {n} ({n - alarm} more steps), "
        f"ended '{run.exit_status}' and {result}."
    )
    if not run.resolved:
        text += f" Stopping at the alarm would have saved {n - alarm} of its {n} steps ({(n - alarm) / n:.0%})."
    return text


def _clip(text: str, lines: int, width: int) -> str:
    kept = [line if len(line) <= width else line[: width - 1] + "…" for line in text.splitlines()[:lines]]
    more = len(text.splitlines()) - lines
    return "\n".join(kept + ([f"… {more} more lines"] if more > 0 else []))


def _step_text(index: int, step: Step, current: bool, width: int) -> Text:
    kind, reply = str(action_kind(step.command)), str(observation_kind(step.reply))
    text = Text()
    text.append(f"step {index:>2}  ", style="bold white" if current else "grey62")
    text.append(kind, style="cyan")
    text.append(" → ")
    text.append(reply, style="red" if reply in _BAD_REPLIES else "green")
    if step.open_file:
        text.append(f"   {step.open_file}", style="grey50")
    text.append("\n")
    if step.thought:
        text.append(
            _clip(" ".join(step.thought.split()), 1, 3 * width) + "\n", style="italic" if current else "italic grey58"
        )
    if step.command:
        text.append(_clip("$ " + step.command, 3, width) + "\n", style="bold cyan" if current else "cyan")
    if step.reply:
        text.append(_clip(step.reply, 5 if current else 2, width) + "\n", style="grey70" if current else "grey46")
    return text


def frame(
    run: RawRun, monitor: Mapping[int, Mapping], step: int, width: int = 150, height: int = 44, end: bool = False
) -> Layout:
    """The screen at ``step``: the agent's latest steps on the left, the monitor on the right."""
    entry = monitor.get(step)
    state = entry["state"] if entry else None
    style = STYLES[state]
    horizon = max(30, max(monitor))
    health = [monitor[k]["health"] if k in monitor else None for k in range(1, step + 1)]
    pane = max(20, width * 2 // 5 - 6)

    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=1))
    layout["body"].split_row(Layout(name="agent", ratio=3), Layout(name="monitor", ratio=2))
    layout["monitor"].split_column(Layout(name="beat", size=8), Layout(name="line", size=9), Layout(name="status"))

    header = Text()
    header.append(" AGENT ECG ", style="bold black on white")
    header.append(f"  {run.model_name} on {run.instance_id}", style="bold")
    header.append("  ·  repository never seen in training  ·  replay, one step at a time", style="grey62")
    layout["header"].update(Panel(header, border_style="grey35"))

    first = max(1, step - _AGENT_STEPS + 1)
    blocks = [_step_text(i, run.steps[i - 1], i == step, width * 3 // 5 - 6) for i in range(first, step + 1)]
    layout["agent"].update(Panel(Group(*blocks), title=f"agent · step {step} of the run", border_style="grey35"))

    beat = braille(heartbeat(health, beats=max(3, pane * 2 // len(_BEAT))), pane, 6, -0.45, 1.05)
    layout["beat"].update(Panel(Text("\n".join(beat), style=style), title="heartbeat", border_style=style))
    line = braille(health + [None] * (horizon - step), pane, 7, 0, 100)
    layout["line"].update(
        Panel(Text("\n".join(line), style=style), title=f"health, steps 1-{horizon}", border_style="grey35")
    )

    status = Text()
    if entry is None:
        status.append("warming up\n", style="bold grey62")
        status.append("the monitor starts scoring at step 5\n", style="grey62")
    else:
        status.append(f"HEALTH {entry['health']:>3}", style=f"{style} bold")
        status.append("  /100\n", style="grey62")
        status.append(f"chance of failure {1 - entry['p_calibrated']:.0%}\n", style="bold")
        status.append(f"state  {state.upper()}\n", style=style)
        if entry.get("events"):
            status.append("this step: " + ", ".join(entry["events"]) + "\n", style="grey70")
    blocks = [status]
    if entry and entry.get("recommendation"):
        blocks.append(Panel(Text(entry["recommendation"], style="bold"), border_style="red", title="stop the agent"))
    if end:
        blocks.append(Panel(Text(outcome(run, monitor)), border_style="white", title="what really happened"))
    layout["status"].update(Panel(Group(*blocks), title="monitor", border_style=style))

    layout["footer"].update(Text(f" run {run.run_id}  ·  Ctrl+C to stop", style="grey50"))
    return layout


def play(run: RawRun, monitor: Mapping[int, Mapping], pace: float = 2.0, console: Console | None = None) -> None:
    """Play the run full screen, one step every ``pace`` seconds, then leave the end card on screen."""
    console = console or Console()
    last = max(monitor)
    with Live(console=console, screen=True, auto_refresh=False) as live:
        for step in range(1, last + 1):
            live.update(frame(run, monitor, step, console.width, console.height), refresh=True)
            time.sleep(pace)
        live.update(frame(run, monitor, last, console.width, console.height, end=True), refresh=True)
        time.sleep(max(3.0, 2 * pace))
    console.print(frame(run, monitor, last, console.width, console.height, end=True), height=console.height)


def render(
    run: RawRun, monitor: Mapping[int, Mapping], step: int, svg: Path | None = None, width: int = 150, height: int = 44
) -> str:
    """One frame as plain text, and optionally as an SVG image (for a post or the README)."""
    console = Console(record=True, width=width, height=height, file=io.StringIO(), force_terminal=True)
    console.print(frame(run, monitor, step, width, height, end=step == max(monitor)), height=height)
    if svg is not None:
        console.save_svg(str(svg), title="Agent ECG", clear=False)
    return console.export_text()
