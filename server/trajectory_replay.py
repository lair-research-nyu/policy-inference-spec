"""Open-loop trajectory replay for server.server_openpi (``--replay PATH``).

Instead of running the model, the server answers each inference request with the next
slice of a recorded commanded-joint trajectory, so the robot re-executes a recording
through the same server, client, chunk truncation and action limiter it uses for a policy.
Observations are ignored except the commanded state, which anchors the approach ramp.

Compared with ``server.debug --source rrd`` (``RrdActionSource``):

* the cursor follows what the robot actually executed. An async client (one that sends
  ``action_prefix`` / ``prefix_change_start``) keeps executing at the control rate while it
  re-requests, so the new chunk's row 0 is the recording row ``hz * (time since the anchor)``:
  the cursor runs on the wall clock. ``prefix_change_start`` is the client's latency
  estimate -- when the new chunk takes over -- not how far it moved since the last request
  (the Ultra client re-requests every ~20 ms with d=2; advancing by d played the recording at
  2x). The rows the client says it is already executing (its ``action_prefix``) are pinned.
  A sync client (no prefix) ran the whole previous chunk, so the cursor advances by its
  length; time spent waiting for the reply is not recording time.
  ``RrdActionSource`` advances one full chunk per request.
* it starts and resumes with an approach ramp: a smooth move from the live commanded
  state to the recording's pose, at a capped joint speed, before the recording plays.
  Every recording starts somewhere other than where the robot happens to be.
* it needs only the action streams, so client-side rollout logs (no cameras) replay too.
* it reads ``.npz`` files written by ``scripts/extract_rrd_actions.py``, and turns an
  ``.rrd`` into one by running that script under this repo's venv: the lair-openpi venv's
  rerun 0.23 cannot open recordings written by 0.30.

Resume rules match the RTC prefix's (``server_openpi._resolve_prefix``): the first request
of a connection and the first after ``endpoint=done`` start the recording over; the first
after an intervention or a request gap of ``--resume-gap-s`` re-anchors with a ramp to
where the recording had got to. When the recording ends the final pose is held.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from policy_inference_spec.action_limiter import GRIPPER_DIMS
from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL
from policy_inference_spec.protocol import ACTION_PREFIX_KEY, PREFIX_CHANGE_START_KEY

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = REPO_ROOT / "scripts" / "extract_rrd_actions.py"
DEFAULT_EXTRACT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_REPLAY_HZ = 50.0
DEFAULT_APPROACH_DEG_S = 20.0
DEFAULT_APPROACH_GRIPPER_PER_S = 1.0  # gripper units (0..1) per second
# Below this (on every joint), the robot is already at the recording's pose: no ramp.
APPROACH_TOLERANCE_DEG = 0.5
# Smoothstep peaks at 1.5x its mean speed; size the ramp so the peak is the cap.
_SMOOTHSTEP_PEAK = 1.5


def load_trajectory(
    path: Path,
    *,
    hz: float = DEFAULT_REPLAY_HZ,
    start_s: float = 0.0,
    end_s: float | None = None,
    extract_python: Path = DEFAULT_EXTRACT_PYTHON,
) -> np.ndarray:
    """A (T, 25) commanded-joint trajectory at ``hz`` from an .npz or .rrd, trimmed."""
    path = path.expanduser()
    assert path.is_file(), f"no such replay file: {path}"
    if path.suffix == ".rrd":
        actions = _extract_rrd(path, hz=hz, python=extract_python)
    else:
        assert path.suffix == ".npz", f"--replay takes an .rrd or an .npz, got {path}"
        with np.load(path) as data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            file_hz = float(data["hz"]) if "hz" in data else hz
        assert math.isclose(file_hz, hz), (
            f"{path} was extracted at {file_hz:g} Hz but --replay-hz is {hz:g}; re-extract it "
            f"(scripts/extract_rrd_actions.py --hz {hz:g}) rather than replay it at the wrong speed"
        )
    assert actions.ndim == 2 and actions.shape[1] == DEFAULT_HARDWARE_MODEL.action_dim, actions.shape
    assert np.isfinite(actions).all(), f"{path}: trajectory has non-finite values"
    first = int(round(start_s * hz))
    last = len(actions) if end_s is None else int(round(end_s * hz))
    assert 0 <= first < last <= len(actions), (
        f"trim [{start_s}, {end_s}] s is outside the recording's {len(actions) / hz:.1f} s"
    )
    return actions[first:last]


def _extract_rrd(path: Path, *, hz: float, python: Path) -> np.ndarray:
    assert Path(python).exists(), (
        f"{python} not found: .rrd replay needs this repo's venv (rerun 0.30) to extract the "
        f"trajectory. Pass --replay-extract-python, or extract it yourself with "
        f"scripts/extract_rrd_actions.py and replay the .npz"
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "trajectory.npz"
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        LOGGER.info("extracting %s with %s", path, python)
        result = subprocess.run(
            [str(python), str(EXTRACT_SCRIPT), str(path), "--output", str(out), "--hz", str(hz)],
            env=env, cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"extracting {path} failed:\n{result.stderr[-3000:]}"
        with np.load(out) as data:
            return np.asarray(data["actions"], dtype=np.float32)


def approach_ramp(
    start: np.ndarray,
    target: np.ndarray,
    *,
    hz: float,
    joint_speed: float,
    gripper_speed: float,
) -> np.ndarray:
    """Rows moving from ``start`` to ``target`` (inclusive of target, exclusive of start),
    along a smoothstep whose peak speed is ``joint_speed`` rad/s on the joints and
    ``gripper_speed`` units/s on the grippers. Empty when already there."""
    delta = np.asarray(target, np.float64) - np.asarray(start, np.float64)
    joints = [i for i in range(len(delta)) if i not in GRIPPER_DIMS]
    if np.abs(delta[joints]).max() < np.deg2rad(APPROACH_TOLERANCE_DEG) and \
            np.abs(delta[list(GRIPPER_DIMS)]).max() < 0.02:
        return np.zeros((0, len(delta)), dtype=np.float32)
    duration = _SMOOTHSTEP_PEAK * max(
        np.abs(delta[joints]).max() / joint_speed,
        np.abs(delta[list(GRIPPER_DIMS)]).max() / gripper_speed,
    )
    n = max(2, math.ceil(duration * hz))
    s = np.arange(1, n + 1) / n
    s = s * s * (3 - 2 * s)
    return (start + s[:, None] * delta).astype(np.float32)


@dataclass(frozen=True)
class ReplayConfig:
    """The loaded trajectory and how to play it; shared by every connection."""

    trajectory: np.ndarray  # (T, 25), one row per control step
    name: str
    hz: float = DEFAULT_REPLAY_HZ
    approach_speed: float = float(np.deg2rad(DEFAULT_APPROACH_DEG_S))  # rad/s
    approach_gripper_speed: float = DEFAULT_APPROACH_GRIPPER_PER_S
    resume_gap_s: float | None = 1.0
    clock: Callable[[], float] = field(default=time.perf_counter, repr=False)  # tests swap it

    def new_state(self) -> "ReplayState":
        return ReplayState(self)


@dataclass
class ReplayChunk:
    chunk: np.ndarray       # (rows, 25)
    committed: int          # leading rows the robot is already executing (limiter leaves them be)
    debug: str
    # The command the robot ran just before row 0, when the client consumed the whole previous
    # chunk (no prefix). The limiter rate-limits row 0 against it, so a lag the limiter built
    # up in the last chunk is caught up gradually instead of released as a jump.
    lead: np.ndarray | None = None


@dataclass
class ReplayState:
    """Per-connection cursor into the trajectory.

    ``seq`` is what is being played since the last anchor: the approach ramp followed by
    the trajectory from ``rec_start``. ``pos`` is the ``seq`` row that the last returned
    chunk began with; ``seq[0]`` is due at ``anchor_s`` on the clock, one row per 1/hz.
    """

    cfg: ReplayConfig
    seq: np.ndarray | None = None
    ramp_len: int = 0
    rec_start: int = 0
    pos: int = 0
    last_sent: np.ndarray | None = None
    last_arrival_s: float | None = None
    anchor_s: float = 0.0
    anchor_pending: bool = True
    finished_logged: bool = False
    _rewind: bool = field(default=True)

    def rewind(self) -> None:
        """Episode over: the next request starts the recording from the top."""
        self.anchor_pending = True
        self._rewind = True

    def interrupt(self) -> None:
        """Intervention: the next request ramps back to where the recording had got to."""
        self.anchor_pending = True

    def rec_index(self, pos: int) -> int:
        """Trajectory row that ``seq[pos]`` plays (the ramp maps to its target)."""
        assert self.seq is not None
        return self.rec_start + max(0, min(pos, len(self.seq) - 1) - self.ramp_len)

    def next_chunk(self, frame: dict[str, Any], state: np.ndarray, *, horizon: int) -> ReplayChunk:
        now = self.cfg.clock()
        gap_s = None if self.last_arrival_s is None else now - self.last_arrival_s
        self.last_arrival_s = now
        d = int(frame.get(PREFIX_CHANGE_START_KEY) or 0)
        wire = frame.get(ACTION_PREFIX_KEY)
        wire = None if wire is None else np.asarray(wire, dtype=np.float32)
        gap = (self.cfg.resume_gap_s is not None and gap_s is not None
               and gap_s >= self.cfg.resume_gap_s and self.seq is not None)

        prev, lead, pinned = self.last_sent, None, None
        if self.anchor_pending or gap:
            why = "start" if self._rewind else ("resume-gap=%.1fs" % gap_s if gap else "resume")
            committed, debug = 0, self._anchor(state, why, now)
        elif d > 0:
            # Async: the robot kept executing while it asked; row 0 is "now" on the recording.
            assert prev is not None
            due = int(round((now - self.anchor_s) * self.cfg.hz))
            advance = min(max(due - self.pos, 0), len(prev))
            if advance != due - self.pos:  # cannot run ahead of / behind the chunk it holds
                self.anchor_s = now - (self.pos + advance) / self.cfg.hz
            self.pos += advance
            if wire is not None:
                pinned = wire[:min(d, len(wire), horizon)]
            else:
                pinned = prev[advance:advance + min(d, horizon)]
            committed = len(pinned)
            debug = f"advance={advance} d={d}{self._prefix_offset(wire, prev)}"
        else:
            # Sync: the client ran the whole previous chunk, then waited for this reply.
            assert prev is not None
            advance = len(prev)
            self.pos += advance
            self.anchor_s = now - self.pos / self.cfg.hz  # the wait is not recording time
            committed, lead = 0, prev[-1]
            debug = f"advance={advance}"

        assert self.seq is not None
        rows = self.seq[self.pos:self.pos + horizon]
        if len(rows) < horizon:  # past the end: hold the final pose
            rows = np.concatenate([rows, np.repeat(self.seq[-1:], horizon - len(rows), axis=0)])
            if not self.finished_logged:
                LOGGER.warning("REPLAY FINISHED %s: holding the final recorded pose", self.cfg.name)
                self.finished_logged = True
        rows = rows.astype(np.float32, copy=True)
        if committed:
            # The committed rows are the ones the robot is executing: what we sent (after the
            # limiter), not the recording. Otherwise a lag the limiter introduced would reappear
            # as a jump at this chunk's first free row.
            rows[:committed] = pinned
        # The caller replaces this with the limited chunk it actually sends.
        self.last_sent = rows
        rec = self.rec_index(self.pos)
        phase = f"ramp {self.pos}/{self.ramp_len}" if self.pos < self.ramp_len else \
            f"t={rec / self.cfg.hz:.2f}/{len(self.cfg.trajectory) / self.cfg.hz:.2f}s"
        return ReplayChunk(chunk=rows, committed=committed, lead=lead,
                           debug=f"replay {phase} row={rec} {debug}")

    def _anchor(self, state: np.ndarray, why: str, now: float) -> str:
        traj = self.cfg.trajectory
        if self._rewind or self.seq is None:
            rec_start = 0
        else:
            # Assume the robot finished the last chunk before the pause.
            rec_start = self.rec_index(self.pos + len(self.last_sent))
        ramp = approach_ramp(state, traj[rec_start], hz=self.cfg.hz,
                             joint_speed=self.cfg.approach_speed,
                             gripper_speed=self.cfg.approach_gripper_speed)
        self.seq = np.concatenate([ramp, traj[rec_start:]])
        self.ramp_len, self.rec_start, self.pos, self.anchor_s = len(ramp), rec_start, 0, now
        self.anchor_pending = self._rewind = self.finished_logged = False
        far = np.rad2deg(np.abs(traj[rec_start] - state)[[i for i in range(len(state)) if i not in GRIPPER_DIMS]].max())
        LOGGER.info("replay %s: anchored at recording t=%.2fs; furthest joint %.1fdeg away -> "
                    "%d-step (%.2fs) approach ramp", why, rec_start / self.cfg.hz, far,
                    len(ramp), len(ramp) / self.cfg.hz)
        return f"anchor({why}) ramp={len(ramp)}"

    @staticmethod
    def _prefix_offset(wire: np.ndarray | None, prev: np.ndarray) -> str:
        """Diagnostic for the log: which row of our last chunk the client's prefix starts at.

        It should be the advance (give or take a row of network jitter). Only reported when
        the chunk moves enough for the match to mean something."""
        if wire is None or len(wire) == 0 or float(np.abs(np.diff(prev, axis=0)).max(initial=0)) < 1e-4:
            return ""
        errs = []
        for k in range(len(prev)):
            n = min(len(wire), len(prev) - k)
            errs.append(float(np.abs(prev[k:k + n] - wire[:n]).max()))
        k = int(np.argmin(errs))
        return f" prefix@{k}" if errs[k] < 1e-3 else f" prefix@?(best {k}, |diff| {errs[k]:.3f})"
