"""Action sources for the debug server.

An ``ActionSource`` is a callable that takes a wire inference request frame
and returns an ``(H, action_dim)`` float32 chunk — the same payload a real
policy would produce. The debug server selects one at startup via ``--source``.

To add a new source: implement ``__call__(frame) -> np.float32 ndarray`` with
shape ``(horizon, action_dim)``. See ``pipeline.ActionSource`` for the Protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from policy_inference_spec.debug.state_slice import slice_state_to_commanded_positions
from policy_inference_spec.feature_engineering import SchemaName, get_feature_bundle_for_schema
from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL, HardwareModel
from policy_inference_spec.protocol import JOINT_STATE_KEY

LOGGER = logging.getLogger(__name__)

_POSE_ACTION_SCHEMAS = frozenset(
    {
        SchemaName.GEN2_28D_STATE_POSE_ACTIONS,
        SchemaName.GEN2_32D_STATE_POSE_ACTIONS,
    }
)


class ZerosSource:
    """Emits all-zero action chunks.

    Use when: you want to smoke-test the wire plumbing end-to-end without any
    policy or recording — verifies the server handshake, validators, and the
    client round-trip are healthy.

    Do NOT drive a real robot with this. A chunk of zeros commands every joint
    to zero radians, which for most arm configurations is a large, unsafe jump
    from the current pose.

    Usage::

        uv run python -m server.debug --source zeros --action-horizon 50
    """

    def __init__(
        self,
        horizon: int,
        *,
        hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
    ) -> None:
        assert horizon >= 1, f"horizon must be positive, got {horizon}"
        self._horizon = horizon
        self._action_dim = hardware_model.action_dim

    def __call__(self, frame: dict[str, Any]) -> npt.NDArray[np.float32]:
        return np.zeros((self._horizon, self._action_dim), dtype=np.float32)


class RrdActionSource:
    """Replays joint-position actions from an .rrd recording.

    Use when: you want the robot to execute a previously recorded trajectory
    verbatim, with no model in the loop — useful for reproducing a demo,
    sanity-checking a recording, or exercising the hardware on a known-good
    motion.

    Each ``__call__`` advances the recording by one action chunk
    (``horizon`` rows at native ``hz``). When the recording ends, the source
    holds the last chunk and logs a warning, so the robot is left commanding
    the final recorded pose rather than receiving garbage.

    Schema constraint: only ``gen2-28d-state`` and ``gen2-32d-state`` are
    accepted. The two ``-pose-actions`` variants produce 23-dim Cartesian
    actions that don't fit the 25-dim wire contract.

    Horizon constraint: ``hz`` must be an integer multiple of ``horizon``
    (at hz=50, horizon in {1, 2, 5, 10, 25, 50} works).

    The .rrd must contain all streams declared by the schema — the reused
    ``RerunReplayer`` waits for every video, observation, and action stream
    to appear before yielding, so a recording missing any declared stream
    will hang on startup.

    Usage::

        uv run python -m server.debug --source rrd \\
            --recording-path ~/recordings/demo.rrd --action-horizon 50
    """

    def __init__(
        self,
        recording_path: Path,
        *,
        horizon: int,
        schema: SchemaName = SchemaName.GEN2_32D_STATE,
        hz: int = 50,
        hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
    ) -> None:
        assert horizon >= 1, f"horizon must be positive, got {horizon}"
        assert hz > 0, f"hz must be positive, got {hz}"
        assert hz % horizon == 0, (
            f"hz ({hz}) must be an integer multiple of horizon ({horizon})"
        )
        assert schema not in _POSE_ACTION_SCHEMAS, (
            f"schema {schema.value} has pose-based actions (action_dim=23); "
            f"RrdActionSource requires a joint-position schema (gen2-28d-state or "
            f"gen2-32d-state) with action_dim=25"
        )
        recording_path = recording_path.expanduser()
        assert recording_path.is_file(), (
            f"recording_path must be an existing file, got {recording_path}"
        )
        bundle = get_feature_bundle_for_schema(schema)
        assert bundle.action_dim == hardware_model.action_dim, (
            f"schema {schema.value} action_dim={bundle.action_dim} != wire "
            f"action_dim={hardware_model.action_dim}"
        )

        from policy_inference_spec.replay_rrd import RerunReplayer

        publish_hz = hz / horizon
        self._replayer = RerunReplayer(recording_path, bundle, hz, publish_hz)
        self._iter = iter(self._replayer)
        self._bundle = bundle
        self._horizon = horizon
        self._action_dim = hardware_model.action_dim
        self._last_chunk: npt.NDArray[np.float32] | None = None
        self._exhausted = False
        LOGGER.info(
            "RrdActionSource ready path=%s schema=%s horizon=%d hz=%d publish_hz=%g",
            recording_path,
            schema.value,
            horizon,
            hz,
            publish_hz,
        )

    def __call__(self, frame: dict[str, Any]) -> npt.NDArray[np.float32]:
        if self._exhausted:
            assert self._last_chunk is not None
            return self._last_chunk
        try:
            sample = next(self._iter)
        except StopIteration:
            self._exhausted = True
            assert self._last_chunk is not None, (
                "rrd recording yielded no samples before exhaustion"
            )
            LOGGER.warning("rrd recording exhausted; holding last chunk")
            return self._last_chunk
        pieces: list[npt.NDArray[np.float32]] = []
        for feature in self._bundle.actions:
            piece = sample[feature.name]
            assert isinstance(piece, np.ndarray), (
                f"expected ndarray for {feature.name}, got {type(piece)}"
            )
            assert piece.shape == (self._horizon, feature.shape), (
                f"{feature.name} shape {piece.shape} != expected "
                f"{(self._horizon, feature.shape)}"
            )
            pieces.append(piece.astype(np.float32, copy=False))
        chunk = np.concatenate(pieces, axis=1)
        assert chunk.shape == (self._horizon, self._action_dim), (
            f"assembled chunk shape {chunk.shape} != "
            f"{(self._horizon, self._action_dim)}"
        )
        self._last_chunk = chunk
        return chunk


class GoToPoseSource:
    """Drives the robot to a target joint configuration in one chunk, then holds.

    Use when: you want to reposition the robot to a known-safe pose (home,
    idle, demo-start) without running a policy — great for setting up before
    a demo or after a failure.

    Behavior:
      - First call: slices the current 97-dim state into 25 commanded
        positions, linearly interpolates from there to ``target_pose`` across
        ``horizon`` steps, and emits that chunk once. Row ``H-1`` equals the
        target exactly.
      - Subsequent calls: return ``target_pose`` tiled ``H`` times, so the
        robot holds the target pose indefinitely.

    At the default horizon=50 @ 50 Hz, the motion takes 1 second.

    Safety gate: ``max_step_per_joint_rad`` caps ``max(|target - state|)``
    before emitting the chunk. A violation raises AssertionError on the
    first call rather than commanding a wild jump. Defaults to 0.5 rad
    (~28°). Set ``max_step_per_joint_rad=None`` or ``<= 0`` on the CLI
    to disable (e.g. for large, intentional resets).

    Usage::

        uv run python -m server.debug --source goto \\
            --target-pose-file home_pose.npy --action-horizon 50
    """

    def __init__(
        self,
        target_pose: npt.NDArray[np.float32],
        horizon: int,
        *,
        max_step_per_joint_rad: float | None = 0.5,
        hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
    ) -> None:
        assert horizon >= 1, f"horizon must be positive, got {horizon}"
        assert isinstance(target_pose, np.ndarray), (
            f"target_pose must be ndarray, got {type(target_pose)}"
        )
        assert target_pose.shape == (hardware_model.action_dim,), (
            f"target_pose must have shape ({hardware_model.action_dim},), "
            f"got {target_pose.shape}"
        )
        assert np.issubdtype(target_pose.dtype, np.floating), (
            f"target_pose must be floating, got {target_pose.dtype}"
        )
        if max_step_per_joint_rad is not None:
            assert max_step_per_joint_rad > 0, (
                f"max_step_per_joint_rad must be positive or None, "
                f"got {max_step_per_joint_rad}"
            )
        self._target = target_pose.astype(np.float32, copy=True)
        self._horizon = horizon
        self._action_dim = hardware_model.action_dim
        self._hardware_model = hardware_model
        self._max_step = max_step_per_joint_rad
        self._done = False
        self._hold_chunk = np.tile(self._target, (self._horizon, 1)).astype(np.float32)

    def __call__(self, frame: dict[str, Any]) -> npt.NDArray[np.float32]:
        if self._done:
            return self._hold_chunk
        state_97 = frame[JOINT_STATE_KEY]
        state_25 = slice_state_to_commanded_positions(
            state_97, hardware_model=self._hardware_model
        )
        delta = self._target - state_25
        if self._max_step is not None:
            max_abs = float(np.max(np.abs(delta)))
            assert max_abs <= self._max_step, (
                f"go-to-pose delta per-joint max |{max_abs:.4f}| exceeds "
                f"max_step_per_joint_rad={self._max_step}; pass "
                f"max_step_per_joint_rad=None to disable this safety check"
            )
        alpha = np.linspace(
            1.0 / self._horizon, 1.0, self._horizon, dtype=np.float32
        )
        chunk = state_25[None, :] + alpha[:, None] * delta[None, :]
        chunk = np.ascontiguousarray(chunk.astype(np.float32, copy=False))
        self._done = True
        LOGGER.info(
            "GoToPoseSource emitted interpolation chunk; holding target on "
            "subsequent calls (max |delta|=%.4f rad)",
            float(np.max(np.abs(delta))),
        )
        return chunk
