"""Which instruction a wire frame carries, shared by every server backend.

Kept free of any policy library so the lerobot server and the openpi server (which run in
different environments) resolve prompts identically.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from policy_inference_spec.protocol import (
    CHUNK_ID_KEY,
    CONDITIONING_METADATA_KEY,
    DONE_KEY,
    DONE_REASON_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PREFIX_CHANGE_START_KEY,
    PREV_SKIPPED_ACTION_START_KEY,
    REWARD_ACTION_INDEX_KEY,
    REWARD_KEY,
    START_METADATA_KEY,
    SUBTASK_KEY,
    TASK_KEY,
)


PROMPT_SOURCES = ("auto", "subtask", "task")

# Where an instruction hides when ``task`` and ``subtask`` turn out to name the
# *construction* the operator assembled in the web UI ("simple_list", "generic_subtask")
# instead of saying anything to the policy. These two dicts are free-form on the wire --
# the protocol only checks that they hold JSON-like values -- so the sentence can only be
# found by the name of the field carrying it.
_PROMPT_METADATA_KEYS = (CONDITIONING_METADATA_KEY, START_METADATA_KEY)
# In preference order. "message" is where the Ultra web UI actually puts the sentence
# ({"start_metadata": {"message": "stack cubes"}}) when the run is driven by a list
# construction. Deliberately excludes "task"/"subtask": a metadata copy of those is a
# copy of the same construction name, and picking it up would silently reintroduce the
# bug this search exists to work around.
_PROMPT_FIELD_NAMES = (
    "instruction", "language_instruction", "prompt", "message", "text", "description",
)

# Keys that change every request. They are worth logging in a per-request line, but not
# in the metadata summary, whose whole value is that it only reappears when something
# about the run's configuration actually changed.
_VOLATILE_FRAME_KEYS = frozenset({
    CHUNK_ID_KEY, PREFIX_CHANGE_START_KEY, PREV_SKIPPED_ACTION_START_KEY,
    REWARD_KEY, REWARD_ACTION_INDEX_KEY, DONE_KEY, DONE_REASON_KEY,
})


def _frame_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    """Everything in a frame that is configuration rather than observation or timing.

    This is the only view of what the robot is actually saying about the run: images and
    state are excluded because they are megabytes of no help, the per-request bookkeeping
    because it changes constantly, and what is left is the handful of strings and
    free-form dicts in which an instruction can legally arrive.
    """
    return {
        key: value
        for key, value in frame.items()
        if key not in _VOLATILE_FRAME_KEYS
        and not key.startswith("observation/images/")
        and key != JOINT_STATE_KEY
        and not isinstance(value, (bytes, bytearray, np.ndarray))
    }


def _metadata_summary(frame: dict[str, Any], *, limit: int = 1200) -> str:
    summary = repr(_frame_metadata(frame))
    return summary if len(summary) <= limit else summary[:limit] + "...<truncated>"


def _walk_metadata_strings(
    node: Any, path: str, out: list[tuple[str, str, str]], *, depth: int = 0
) -> None:
    if depth > 4:
        return
    items: Iterable[tuple[str, Any]]
    if isinstance(node, Mapping):
        items = node.items()
    elif isinstance(node, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(node))
    else:
        return
    for key, value in items:
        child = f"{path}.{key}"
        if isinstance(value, str):
            out.append((key, child, value))
        else:
            _walk_metadata_strings(value, child, out, depth=depth + 1)


def _search_metadata_prompt(frame: dict[str, Any]) -> tuple[str, str] | None:
    """First conventionally-named instruction string in the free-form metadata dicts.

    The scan is ordered by field name rather than by tree position: the name is the only
    evidence available about which of several strings was meant for the policy, so a
    nested ``instruction`` outranks a shallower ``text``.
    """
    found: list[tuple[str, str, str]] = []
    for meta_key in _PROMPT_METADATA_KEYS:
        _walk_metadata_strings(frame.get(meta_key), meta_key, found)
    for name in _PROMPT_FIELD_NAMES:
        for leaf, path, value in found:
            if leaf == name and value.strip():
                return value, path
    return None


def _is_metadata_origin(origin: str) -> bool:
    """Whether a ``_frame_prompt`` origin label points into the free-form metadata."""
    return origin.removeprefix("auto:").split(".", 1)[0] in _PROMPT_METADATA_KEYS


def _lookup_prompt_path(frame: dict[str, Any], path: str) -> str | None:
    """Read a dotted path (``conditioning_metadata.instruction``) out of a frame."""
    node: Any = frame
    for segment in path.split("."):
        if isinstance(node, Mapping):
            node = node.get(segment)
        elif isinstance(node, (list, tuple)) and segment.isdigit() and int(segment) < len(node):
            node = node[int(segment)]
        else:
            return None
    return node if isinstance(node, str) else None


def _frame_prompt(frame: dict[str, Any], *, source: str = "auto") -> tuple[str, str]:
    """Pick the instruction to condition the policy on; return it and where it came from.

    The robot sends two strings and neither is guaranteed to be an instruction. ``task``
    is the top-level selection in the web UI and ``subtask`` the step under it, but when
    the operator builds the run out of UI constructions those hold the constructions'
    names ("simple_list", "generic_subtask") -- tokens no checkpoint has been trained on
    -- and the sentence the policy needs rides in the free-form metadata instead.

    ``auto`` therefore prefers a conventionally-named metadata field, then the subtask,
    then the task. Any other source is taken literally: ``task``, ``subtask``, or a
    dotted path into the frame. The returned label names the winner exactly, because a
    prompt silently coming from somewhere unexpected is the failure this resolves.

    Uses ``.get`` throughout because terminal (done) frames carry telemetry, not a
    validated inference frame, and need not include any of these keys.
    """
    task = str(frame.get(TASK_KEY) or "")
    subtask = str(frame.get(SUBTASK_KEY) or "")
    if source == "task":
        return task, TASK_KEY
    if source == "subtask":
        return subtask, SUBTASK_KEY
    if source != "auto":
        return _lookup_prompt_path(frame, source) or "", source
    metadata_hit = _search_metadata_prompt(frame)
    if metadata_hit is not None:
        value, path = metadata_hit
        return value, f"auto:{path}"
    if subtask:
        return subtask, f"auto:{SUBTASK_KEY}"
    return task, f"auto:{TASK_KEY}"


# Frame keys a --prompt-source path may start at. Restricting the root turns a typo
# ("subtaks.text") into a startup error instead of a run conditioned on an empty string.
_PROMPT_PATH_ROOTS = (
    CONDITIONING_METADATA_KEY, START_METADATA_KEY, TASK_KEY, SUBTASK_KEY,
    MODEL_ID_KEY, POLICY_ID_KEY,
)


def _check_prompt_source(value: str) -> str:
    if value in PROMPT_SOURCES or value.split(".", 1)[0] in _PROMPT_PATH_ROOTS:
        return value
    raise argparse.ArgumentTypeError(
        f"unknown prompt source {value!r}: expected one of {PROMPT_SOURCES}, or a dotted "
        f"path into the frame starting with one of {_PROMPT_PATH_ROOTS}"
    )
