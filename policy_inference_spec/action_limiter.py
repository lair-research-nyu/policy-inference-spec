"""Per-chunk joint limits, applied to every action chunk before it leaves the server.

Why this exists. On 2026-09-01 a latent-space RL run (``--latent_horizon 1``: one noise row
repeated over pi0's 50-step chunk) made both arms swing on the first chunk. Measured
afterwards on the served checkpoint (``dsrl_pi0/tests/exp4/tails.py``), the repeated row
displaced the *whole chunk* from the current pose by up to 39 degrees on a single joint,
and by more than 10 degrees for 37% of latents; the base policy's own worst case is 9.7
degrees. The per-step motion inside those chunks was nearly normal (3.7 deg/step worst,
against 2.5 for the base policy), so the danger is a chunk that starts far from where the
robot is, not a jerk within the chunk. Nothing on the server or client path limited either.

Two limits, applied in order, on the joint dims only. Grippers are in their own units and
are never touched.

1. **Chunk-start offset.** The first step of the chunk may not be further than
   ``max_start_offset`` from the current commanded state on any joint. If it is, the whole
   chunk is translated by the excess: its shape is preserved and only its origin is pulled
   back to within reach of the pose the robot is actually in. For a noise-displaced chunk,
   whose displacement is constant over the chunk, this is exactly the right correction;
   for an intended move it costs at most the excess, which the next query re-plans from
   the new state.
2. **Per-step rate**, from the second step on. No joint may move more than ``max_step``
   between consecutive steps of the chunk. A step that asks for more is clipped and the
   rest of the chunk continues from the clipped position, so a jerk becomes a bounded
   approach. The first step is governed by the offset limit alone, because a chunk that
   starts up to ``max_start_offset`` from the state is what the base policy already does.

The defaults sit just above the base policy's own maxima on the cube-stack checkpoints
(chunk-start 10 deg, 3 deg per 50 Hz step, measured over 12 scenes x 64 latents), so in
normal operation neither limit fires. Whenever one does, the server logs which joints
and by how much; the learner's record still holds the chunk pi0 produced, so a fired
limit is visible in the logs and the executed motion is the limited one.

RTC exception: an accepted prefix represents commands already committed on the robot.
Those rows are immutable. The state-relative chunk-start correction is skipped, and
the per-step limit applies only to the free tail, including its boundary with the
last committed row. Rejected prefixes and resumes use the ordinary live-state limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

# Layout of the 25-dim commanded-position vector, in the order the policy emits it. The
# server asserts this matches its own ``_STATE_DIM_NAMES``.
GEN2_DIM_NAMES: tuple[str, ...] = (
    *(f"left_arm[{i}]" for i in range(7)), "left_gripper",
    *(f"right_arm[{i}]" for i in range(7)), "right_gripper",
    *(f"chest[{i}]" for i in range(6)),
    *(f"neck[{i}]" for i in range(3)),
)
GRIPPER_DIMS: tuple[int, ...] = (7, 15)
DEFAULT_LIMIT_DIMS: tuple[int, ...] = tuple(i for i in range(len(GEN2_DIM_NAMES)) if i not in GRIPPER_DIMS)

# Just above the base policy's measured maxima (9.7 deg chunk-start offset, 2.5 deg/step).
DEFAULT_MAX_START_OFFSET_DEG = 10.0
DEFAULT_MAX_STEP_DEG = 3.0


@dataclass(frozen=True)
class ActionLimits:
    """Limits in the action's own units (radians for the joints). ``None`` disables a limit."""

    dims: tuple[int, ...]
    max_start_offset: float | None
    max_step: float | None

    def __post_init__(self) -> None:
        assert len(self.dims) > 0, "no dims to limit"
        assert len(set(self.dims)) == len(self.dims), f"duplicate dims in {self.dims}"
        for limit, name in ((self.max_start_offset, "max_start_offset"), (self.max_step, "max_step")):
            assert limit is None or limit > 0, f"{name} must be positive or None, got {limit}"
        assert self.max_start_offset is not None or self.max_step is not None, "both limits are off"


@dataclass(frozen=True)
class LimiterReport:
    """What the limiter changed. Zero everywhere means the chunk went through untouched."""

    offset_excess: np.ndarray   # (action_dim,) how far each dim's chunk start was pulled back
    step_clips: int             # number of (step, dim) entries whose per-step motion was clipped
    max_step_excess: float      # the largest amount a single per-step motion was reduced by

    @property
    def triggered(self) -> bool:
        return bool(np.any(self.offset_excess != 0)) or self.step_clips > 0

    def describe(self, names: Sequence[str] = GEN2_DIM_NAMES) -> str:
        parts = []
        moved = np.nonzero(self.offset_excess)[0]
        if len(moved):
            worst = moved[np.argsort(-np.abs(self.offset_excess[moved]))][:4]
            parts.append("chunk start pulled back on " + ", ".join(
                f"{names[i]} by {np.rad2deg(self.offset_excess[i]):+.1f}deg" for i in worst)
                + (f" (+{len(moved) - len(worst)} more)" if len(moved) > len(worst) else ""))
        if self.step_clips:
            parts.append(f"{self.step_clips} per-step motions clipped, worst by "
                         f"{np.rad2deg(self.max_step_excess):.1f}deg/step")
        return "; ".join(parts) if parts else "untouched"


def resolve_limit_dims(names: Sequence[str]) -> tuple[int, ...]:
    """Dim names (``left_arm[3]``, ``chest[0]``, ...) or bare indices -> sorted dim indices.

    An empty selection means the default: every joint dim, no grippers.
    """
    if not names:
        return DEFAULT_LIMIT_DIMS
    dims: list[int] = []
    for name in names:
        if name.lstrip("-").isdigit():
            idx = int(name)
            assert 0 <= idx < len(GEN2_DIM_NAMES), f"dim index {idx} out of range 0..{len(GEN2_DIM_NAMES) - 1}"
        else:
            assert name in GEN2_DIM_NAMES, (
                f"unknown dim {name!r}; expected an index or one of: {', '.join(GEN2_DIM_NAMES)}"
            )
            idx = GEN2_DIM_NAMES.index(name)
        if idx not in dims:
            dims.append(idx)
    return tuple(sorted(dims))


def limits_from_degrees(
    max_start_offset_deg: float | int | None = DEFAULT_MAX_START_OFFSET_DEG,
    max_step_deg: float | int | None = DEFAULT_MAX_STEP_DEG,
    dims: Sequence[int] = DEFAULT_LIMIT_DIMS,
) -> ActionLimits:
    """Build limits from the numbers a person thinks in. A limit of 0 or None is off."""
    def rad(x: float | int | None) -> float | None:
        return None if x is None or x <= 0 else float(np.deg2rad(x))
    return ActionLimits(dims=tuple(dims), max_start_offset=rad(max_start_offset_deg), max_step=rad(max_step_deg))


def limit_chunk(
    chunk: np.ndarray,
    state: np.ndarray,
    limits: ActionLimits,
    *,
    committed_prefix_len: int = 0,
) -> tuple[np.ndarray, LimiterReport]:
    """Apply the limits to one chunk. Returns a new array; the input is not modified.

    ``chunk`` is (steps, action_dim) in robot units; ``state`` is the current commanded
    position, (action_dim,), in the same units. Only ``limits.dims`` are touched.

    With a valid RTC prefix, the first ``committed_prefix_len`` rows are already
    committed on the robot and must remain unchanged. Skip the state-relative offset
    correction (tracking lag is legitimate), and rate-limit the free tail starting
    from the last committed command. With no committed prefix, including after a
    resume, apply the usual chunk-start limit relative to the live state.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    assert chunk.ndim == 2, f"chunk must be (steps, action_dim), got {chunk.shape}"
    assert state.shape == (chunk.shape[1],), f"state {state.shape} does not match chunk {chunk.shape}"
    assert 0 <= committed_prefix_len <= chunk.shape[0], (
        f"committed prefix length {committed_prefix_len} exceeds chunk length {chunk.shape[0]}"
    )
    dims = list(limits.dims)
    assert max(dims) < chunk.shape[1], f"limit dims {dims} exceed action dim {chunk.shape[1]}"

    out = chunk.copy()
    offset_excess = np.zeros(chunk.shape[1], dtype=np.float32)

    if limits.max_start_offset is not None and committed_prefix_len == 0:
        m = limits.max_start_offset
        offset = out[0, dims] - state[dims]
        excess = offset - np.clip(offset, -m, m)
        out[:, dims] -= excess                       # translate the whole chunk, shape preserved
        offset_excess[dims] = excess

    step_clips = 0
    max_step_excess = 0.0
    if limits.max_step is not None:
        s = limits.max_step
        for t in range(max(1, committed_prefix_len), out.shape[0]):
            step = out[t, dims] - out[t - 1, dims]
            over = np.abs(step) - s
            hit = over > 0
            if hit.any():
                step_clips += int(hit.sum())
                max_step_excess = max(max_step_excess, float(over[hit].max()))
                out[t, dims] = out[t - 1, dims] + np.clip(step, -s, s)

    return out, LimiterReport(offset_excess=offset_excess, step_clips=step_clips, max_step_excess=max_step_excess)
