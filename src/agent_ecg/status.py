"""Read and write the status report of the TimeNet task.

The report has two parts. The *description* (first four sentences) counts what the agent did so far;
it is a fixed function of the inputs. The *prediction* (last sentence) says how many more steps the run
will take, how it will end, and whether the agent will resolve the issue:

    The run will go on for 41 more steps, then it will reach the context limit and SWE-agent will
    submit its changes, and the agent will not resolve the issue.

Only the prediction is worth scoring; see :mod:`agent_ecg.metrics`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from timenet_connectors.datasets.nebius.swe_agent_trajectories.source import EXIT_STATUSES

PREDICTION_START = "The run will go on for"
_PREDICTION = re.compile(
    r"The run will go on for (?P<remaining>\d+) more steps?, then (?P<phrase>.+?), "
    r"and the agent will (?P<outcome>resolve|not resolve) the issue\.",
    re.IGNORECASE | re.DOTALL,
)
# Each ending phrase of the template, mapped to its ending label. Several exit statuses share a label.
PHRASE_TO_ENDING = {status.phrase.lower(): str(status.ending) for status in EXIT_STATUSES.values()}


@dataclass(frozen=True)
class Prediction:
    """The facts of a status report's last sentence; ``None`` where the text does not say."""

    remaining: int | None
    ending: str | None
    resolved: bool | None

    @property
    def parsed(self) -> bool:
        return self.remaining is not None and self.ending is not None and self.resolved is not None


def split_status(text: str) -> tuple[str, str]:
    """Split a report into its description and its prediction (from the last "The run will go on for")."""
    at = text.lower().rfind(PREDICTION_START.lower())
    if at < 0:
        return text.strip(), ""
    return text[:at].strip(), text[at:].strip()


def parse_status(text: str) -> Prediction:
    """Read the three facts of a report's prediction sentence."""
    _, prediction = split_status(text or "")
    match = _PREDICTION.search(prediction)
    if match is None:
        return Prediction(None, None, None)
    return Prediction(
        remaining=int(match["remaining"]),
        ending=PHRASE_TO_ENDING.get(" ".join(match["phrase"].lower().split())),
        resolved=match["outcome"].lower() == "resolve",
    )


def _count(number: int, singular: str, plural: str | None = None) -> str:
    return f"{number} {singular if number == 1 else plural or singular + 's'}"


def describe(signals: Mapping[str, Sequence]) -> str:
    """Write the description part of a report from a checkpoint's signals, as the connector does."""
    k = len(signals["action_kind"])
    actions, replies = Counter(signals["action_kind"]), Counter(signals["observation_kind"])
    repeats = int(sum(bool(value) for value in signals["repeated_action"]))
    same_file = int(sum(bool(value) for value in signals["same_file_edit"]))
    own = Counter(signals["edit_target"])["own_script"]
    return (
        f"In its first {k} steps the agent edited files {_count(actions['edit'], 'time')}, "
        f"created {_count(actions['create'], 'file')}, ran code {_count(actions['run'], 'time')}, "
        f"read or searched the repository {_count(actions['read'] + actions['search'], 'time')} "
        f"and repeated an earlier command {_count(repeats, 'time')}. "
        f"The linter rejected {_count(replies['edit_rejected'], 'edit')}, "
        f"{_count(replies['traceback'], 'reply', 'replies')} showed a Python traceback, "
        f"{_count(replies['command_error'], 'reply', 'replies')} a command error "
        f"and {_count(replies['timeout'], 'command')} timed out. "
        f"{_count(same_file, 'edit')} changed the same file as the edit before, "
        f"and {_count(own, 'edit')} changed a file the agent had created. "
        f"The conversation holds {int(signals['context_chars'][-1]):,} characters."
    )


def predict_sentence(remaining: int, ending: str, resolved: bool) -> str:
    """Write a prediction sentence in the template's words, for a model that predicts the three facts."""
    phrase = next(status.phrase for status in EXIT_STATUSES.values() if str(status.ending) == ending)
    return (
        f"{PREDICTION_START} {_count(remaining, 'more step')}, then {phrase}, "
        f"and the agent will {'resolve' if resolved else 'not resolve'} the issue."
    )
