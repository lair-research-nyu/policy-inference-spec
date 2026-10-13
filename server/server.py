# START SERVER:
#   python -m server.server --checkpoint-path models/<ckpt> --host 0.0.0.0 --port 18090
#
# Consolidated policy inference server (merges old_minimal.py, old_minimal_debug.py,
# rtc_server.py). Four inference paths:
#   * default: synchronous select_action loop (works for act / pi0 / pi0_fast / pi05)
#   * --rtc:   inference-time Real-Time Chunking via predict_action_chunk with prefix-
#              guided inpainting (pi0 / pi05 flow-matching policies only). Adds a
#              pseudoinverse-guidance VJP to every denoise step.
#   * --rtc-train: train-time RTC (action-prefix conditioning, arXiv 2512.05964). Feeds
#              the committed prefix as an ordinary model input and runs one plain forward
#              pass — no guidance, no per-step VJP. Requires a checkpoint fine-tuned with
#              rtc_train_enabled=true (pi0 / pi05 only). Mutually exclusive with --rtc.
#   * --rl-agent-url: latent-space RL. An external agent picks the flow-matching
#              starting noise from the policy's own prefix embedding and learns from the
#              transitions; the robot client is unchanged and never sees a latent.
#              Mutually exclusive with --rtc and --temporal-ensemble-coeff, both of
#              which rewrite the chunk after denoising. See server/pi0_latent.py and
#              policy_inference_spec/rl.py.
#
# Wire contract for the RTC prefix (per inference request frame):
#   * ``prefix_change_start`` (int, optional): number of steps at the start of the
#     new chunk that are already committed and will execute before this prediction
#     is ready — the RTC ``inference_delay`` / freeze length. Defaults to 0.
#   * ``action_prefix`` (ndarray (T_prev, action_dim), optional): leftover actions
#     from the previous chunk, already aligned to the new chunk's first step and in
#     ROBOT action space -- the same units and dim (25) the server returns in
#     ``action``, per the upstream spec in README.md. The server converts it into
#     the policy's action space (:func:`_normalize_wire_prefix`) before use as
#     ``prev_chunk_left_over``; do NOT pre-normalize it on the client. For a checkpoint
#     trained on relative actions that conversion also subtracts the current state.
#     If omitted, the server falls back to its own memory of the previous returned
#     chunk and uses ``last_chunk[prefix_change_start:]`` (mirrors lerobot's
#     ``ActionQueue._replace_actions_queue``), converted the same way.
#   Both RTC paths read the same two fields; they differ only in how they use them.
#   --rtc (inference-time) freezes the first ``prefix_change_start`` steps and softly
#   guides the rest of the overlap. --rtc-train (train-time) conditions on a HARD prefix
#   of exactly ``prefix_change_start`` steps (``prev_chunk_left_over[:prefix_change_start]``)
#   — the same frozen prefix, so the two paths are directly comparable — and does nothing
#   with the softer overlap region.
#
# Pausing and resuming (teleop / DAgger interventions):
#   The prefix is a PROMISE that its rows are executing right now, while the request is in
#   flight. A pause breaks that promise -- the queued actions never run, and by the time
#   auto resumes a human has moved the arms somewhere else. Chaining across it commands the
#   robot back along a pre-pause trajectory for ``prefix_change_start`` steps, which is
#   felt as an abrupt shake that settles once the chunk's free postfix takes over. So:
#     * on pause, the client drops its leftover action queue;
#     * the first request after resume carries NEITHER ``action_prefix`` NOR
#       ``prefix_change_start`` (the wire format rejects one without the other, and rejects
#       prefix_change_start=0 alongside a prefix), which yields an unconditioned chunk
#       planned from the live observation;
#     * that chunk is executed from step 0 -- there is no in-flight prefix to skip;
#     * chaining resumes normally from the request after it.
#   The server can discard stale conditioning without client changes. It ignores the
#   prefix on any request that opens a connection, follows an ``endpoint=intervention``
#   frame, or arrives more than --resume-gap-s after the previous one (see _resume_reason)
#   -- a paused client is visible in its own request timing. What the server cannot do is
#   stop the client replaying the queue it kept across the pause; those actions never reach
#   the server. Such replay has not been established by the available logs; stale motion
#   for one inference delay is a possibility, not an inevitable residual after this fix.

# python -m server.server --checkpoint-path ... --host 0.0.0.0 --port 18090

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable, Sequence

import numpy as np
import torch
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.codec import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    server_handshake_for_hardware_model,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)
from policy_inference_spec.protocol import (
    ACTION_KEY,
    ACTION_PREFIX_KEY,
    CHUNK_ID_KEY,
    CONDITIONING_METADATA_KEY,
    DEFAULT_INFERENCE_SERVER_PORT,
    DONE_KEY,
    DONE_REASON_KEY,
    ENDPOINT_DONE,
    ENDPOINT_INTERVENTION,
    ENDPOINT_KEY,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PREFIX_CHANGE_START_KEY,
    PREV_SKIPPED_ACTION_START_KEY,
    Q_VALUE_KEY,
    REWARD_ACTION_INDEX_KEY,
    REWARD_KEY,
    RL_ENABLED_KEY,
    START_METADATA_KEY,
    STATUS_KEY,
    SUBTASK_KEY,
    TASK_KEY,
    ServerFeature,
    ServerHandshake,
)
from policy_inference_spec.action_limiter import (
    DEFAULT_MAX_START_OFFSET_DEG,
    DEFAULT_MAX_STEP_DEG,
    GEN2_DIM_NAMES,
    GRIPPER_DIMS,
    ActionLimits,
    limit_chunk,
    limits_from_degrees,
    resolve_limit_dims,
)
from policy_inference_spec.rl import (
    DEFAULT_AGENT_PORT,
    AgentObservation,
    ProducedActions,
    NullAgent,
    RemoteAgent,
    RLAgent,
    Transition,
)
from server.wire import _agent_images as _agent_images_for, _list_lan_ips, _wire_state_25
from server.prompts import (
    PROMPT_SOURCES,
    _PROMPT_FIELD_NAMES,
    _check_prompt_source,
    _frame_prompt,
    _is_metadata_origin,
    _metadata_summary,
)

import hashlib
import time
from itertools import count
import simplejpeg
from pathlib import Path
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import NormalizerProcessorStep, TransitionKey, UnnormalizerProcessorStep
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.configs.types import RTCAttentionSchedule

_REQUEST_COUNTER = count()


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=6).hexdigest()


def _image_fingerprint(value: Any) -> str:
    if isinstance(value, bytes):
        return f"len={len(value)} h={_hash_bytes(value)}"
    if isinstance(value, np.ndarray):
        return f"shape={tuple(value.shape)} h={_hash_bytes(value.tobytes())}"
    return f"type={type(value).__name__}"


def _policy_queue_info(policy: Any) -> str:
    parts: list[str] = []
    for attr in ("_action_queue", "_queues"):
        value = getattr(policy, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            for key, inner in value.items():
                try:
                    parts.append(f"{attr}[{key}]={len(inner)}")
                except TypeError:
                    parts.append(f"{attr}[{key}]=?")
        else:
            try:
                parts.append(f"{attr}={len(value)}")
            except TypeError:
                parts.append(f"{attr}=?")
    return ",".join(parts) if parts else "none"

DEFAULT_ACTION_HORIZON = 50
DEFAULT_REPLAY_ACTION_HORIZON = 50
DEFAULT_CHECKPOINT_PATH = None
DEFAULT_POLICY_TYPE = "pi0"  # one of: "act", "pi0", "pi0_fast", "pi05"

# RTC defaults. execution_horizon is the web app's "chunk execution limit":
# how many actions the client executes before requesting a fresh chunk. It
# shapes the RTC prefix-weight falloff and should match the client's setting.
DEFAULT_EXECUTION_HORIZON = 10
# Client control-loop rate. Only used to turn request-arrival gaps into control steps
# so the server can report the inference delay the client *should* be sending.
DEFAULT_ROLLOUT_HZ = 50.0
DEFAULT_RTC_SCHEDULE = "EXP"
DEFAULT_MAX_GUIDANCE_WEIGHT = 10.0
# Gap between consecutive requests, in seconds, above which the client is taken to have
# paused: the next request starts a fresh plan and its action_prefix is ignored. An async
# rollout re-requests as fast as it can, so its steady-state gap is one round trip -- 160 ms
# on the GEN2 field logs, against an inference time of 150 ms. A second is ~6x that and
# still far shorter than any hand-over to a teleoperator.
DEFAULT_RESUME_GAP_S = 1.0
# Largest per-joint distance, in degrees, between a wire action_prefix's first row and the
# state that arrived with it, before the prefix is dropped. OFF by default (0): a prefix
# that is far from the state is usually just an arm at speed, not a stale one. Measured on
# a 2026-09-04 rtc-train rollout at 50 Hz, right_arm[3] swung 14 deg between consecutive
# requests 157 ms apart (~89 deg/s) with no pause anywhere in the log, so a 10 deg limit
# fired on 4 of every 5 requests -- and each drop made the next chunk agree with its
# neighbours less, holding the gap open. Only useful against a client that pauses without
# any gap in its request stream; see _reject_stale_prefix.
DEFAULT_MAX_PREFIX_GAP_DEG = 0.0

# Edge length of the square images handed to the RL agent. Small on purpose: the agent's
# own conv encoder is tiny compared to the policy's, and the scene detail it cannot
# resolve at this size is already available to it through the policy's prefix embedding.
DEFAULT_RL_RESIZE_IMAGE = 128

# Where the RL observation dump goes when --rl-obs-dump-dir is not given. Every RL run
# gets a fresh timestamped subdirectory of this root: an offline sweep joins images to
# episodes on step_id, the server restarts that counter at 0, and the dump appends to
# steps.jsonl -- so two runs sharing one directory silently interleave, and the second
# run's images overwrite the first's. Sharing the learner's ~/exp and its
# %Y_%m_%d_%H_%M_%S naming makes the pairing obvious when it is time to view a run.
DEFAULT_RL_OBS_DUMP_ROOT = Path("~/exp/obsdump")

# Populated by _init_policy() once the checkpoint path / policy type are known.
POLICY: Any = None
POLICY_TYPE: str | None = None
POLICY_ID: str | None = None
_DATASET_STATS: dict | None = None
_PREPROCESSOR: Any = None
_POSTPROCESSOR: Any = None
# The postprocessor's action (un)normalizer, reused in the forward direction to convert
# wire action prefixes from robot space into the policy's action space. Holding the step
# itself -- rather than re-deriving mean/std -- guarantees the conversion is the exact
# inverse of the unnormalization applied to every chunk we return, including its
# normalization mode (MEAN_STD / MIN_MAX / QUANTILES) and eps.
_ACTION_NORMALIZER: Any = None
# The preprocessor's RelativeActionsProcessorStep when the checkpoint was trained on
# relative actions (action - state for non-gripper joints), else None. It caches the
# state of the request being served, so applying it to a robot-space prefix re-bases
# that prefix onto the same state the returned chunk is made absolute against.
_RELATIVE_ACTION_STEP: RelativeActionsProcessorStep | None = None
# RTC runtime config, populated by _init_policy(). execution_horizon is read per
# request so it can stay in sync with the client's chunk execution limit.
RTC_ENABLED: bool = False
EXECUTION_HORIZON: int = DEFAULT_EXECUTION_HORIZON
# Train-time RTC (action-prefix conditioning, arXiv 2512.05964), populated by
# _init_policy(). Mutually exclusive with RTC_ENABLED: this path conditions on a hard
# prefix and runs one plain forward pass, with no inference-time guidance processor.
RTC_TRAIN_ENABLED: bool = False
# Delay range the checkpoint was fine-tuned on, read from its config by
# _configure_rtc_train(). A prefix_change_start outside [min, max) conditions the model
# on a prefix length it never saw in training.
RTC_TRAIN_MIN_DELAY: int = 0
RTC_TRAIN_MAX_DELAY: int = 0
ROLLOUT_HZ: float = DEFAULT_ROLLOUT_HZ
# Request gap that marks a resume, in seconds; None = never infer a resume from timing.
# Set from --resume-gap-s. See _resume_reason.
RESUME_GAP_S: float | None = DEFAULT_RESUME_GAP_S
# Staleness threshold for wire action prefixes, in radians; None = accept any prefix.
# Set from --max-prefix-gap-deg. See _prefix_continuity_gap.
MAX_PREFIX_GAP: float | None = None
# Latent-space RL runtime, populated by _init_rl(). RL_AGENT defaults to None (plain
# inference); with an agent attached the server asks it for the denoiser's starting
# latent instead of sampling one, and reports transitions back for learning.
RL_BACKEND: Any = None
RL_AGENT: RLAgent | None = None
RL_RESIZE_IMAGE: int = DEFAULT_RL_RESIZE_IMAGE
# Optional side-channel that writes the wire's own camera JPEGs to disk, so an offline
# sweep can re-run pi0 on the exact observation the robot was steered from. The learner's
# 128px critic view is far too coarse for that; see server/rl_obs_dump.py.
RL_OBS_DUMP: Any = None
# State dims to overwrite with their training mean before normalization, as
# {index: mean}. Populated by _init_policy() from --pin-state-dims; empty = off.
PINNED_STATE_DIMS: dict[int, float] = {}

# Per-chunk joint limits applied to every chunk before it is returned; None = off.
# See policy_inference_spec.action_limiter for what they do and why they exist.
ACTION_LIMITS: ActionLimits | None = None


class TemporalEnsembleBuffer:
    """Rolling buffer for ACT temporal ensembling across overlapping action chunks.

    At each request (timestep t), a fresh chunk C_t of shape (H, action_dim) is
    added. The returned chunk is a weighted average of all stored predictions that
    cover each future position, with weight exp(-coeff * age) for predictions made
    `age` requests ago.  Older requests contribute to fewer positions naturally.

    Call reset() when the episode resets so stale predictions don't bleed across.
    """

    def __init__(self, action_horizon: int, action_dim: int, coeff: float) -> None:
        self._horizon = action_horizon
        self._dim = action_dim
        self._coeff = coeff
        self._chunks: list[np.ndarray] = []  # oldest → newest

    def reset(self) -> None:
        self._chunks.clear()

    def add_and_ensemble(self, chunk: np.ndarray) -> np.ndarray:
        """Store chunk (H, D) and return the exponentially-weighted ensemble."""
        assert chunk.shape == (self._horizon, self._dim), (
            f"expected ({self._horizon}, {self._dim}), got {chunk.shape}"
        )
        self._chunks.append(chunk)
        if len(self._chunks) > self._horizon:
            self._chunks.pop(0)
        n = len(self._chunks)
        result = np.zeros((self._horizon, self._dim), dtype=np.float32)
        for k in range(self._horizon):
            # Contributions: age 0 → chunks[n-1][k], age 1 → chunks[n-2][k+1], ...
            # Valid when k + age < H, so max_age = min(n-1, H-k-1).
            max_age = min(n - 1, self._horizon - k - 1)
            ages = np.arange(max_age + 1, dtype=np.float32)
            weights = np.exp(-self._coeff * ages)
            weights /= weights.sum()
            for age in range(max_age + 1):
                result[k] += weights[age] * self._chunks[n - 1 - age][k + age]
        return result


class _RtcState:
    """Per-connection memory of the last returned action chunk.

    ``last_chunk`` holds the final returned actions, after limiting and truncation, in
    ROBOT action space -- the same space as a wire ``action_prefix``. It is used when the
    client does not send an explicit prefix, and is converted into the policy's action
    space per request (:func:`_to_policy_action_space`). It cannot be stored already
    converted: for a relative-action checkpoint the policy-space value of a command
    depends on the state of the request that reads it, not the one that produced it.
    Inference-time RTC (:func:`_resolve_rtc_kwargs`) uses it as the guidance target;
    train-time RTC (:func:`_resolve_prefix_kwargs`) slices its hard prefix from it.
    """

    __slots__ = ("last_chunk", "last_arrival_s", "unconditioned_streak", "resume_pending",
                 "committed_prefix_len")

    def __init__(self) -> None:
        self.last_chunk: torch.Tensor | None = None  # (T, action_dim) robot space, on device
        # Effective prefix used for this prediction, after resume/rejection/clamping.
        # The output limiter must preserve these already committed rows.
        self.committed_prefix_len: int = 0
        # Wall-clock arrival of the previous request. Feeds the delay diagnostics, and the
        # resume detector reads the gap it implies.
        self.last_arrival_s: float | None = None
        # Diagnostics only: run length of consecutive unconditioned chunks.
        self.unconditioned_streak: int = 0
        # One-shot: the next request must ignore any prefix it carries, because nothing is
        # committed. True on a fresh connection and again after every reset().
        self.resume_pending: bool = True

    def reset(self) -> None:
        self.last_chunk = None
        self.committed_prefix_len = 0
        self.last_arrival_s = None
        self.unconditioned_streak = 0
        self.resume_pending = True


class _RlState:
    """Per-connection RL bookkeeping.

    Only the previous step's id is kept: the agent caches the observations it was sent,
    so a transition is just a pair of ids plus what happened in between. A new episode
    must start with no predecessor, or the first transition of the new episode would be
    stitched onto the last one of the old.
    """

    __slots__ = ("last_step_id",)

    def __init__(self) -> None:
        self.last_step_id: int | None = None

    def reset(self) -> None:
        self.last_step_id = None


def _load_config_tolerant(checkpoint_path: Path, config_cls: Any):
    """Load a PreTrainedConfig, dropping config.json keys the installed config
    dataclass doesn't define (e.g. fields added by a training fork). Safe only when
    the dropped fields are inert at inference (e.g. use_relative_actions=False)."""
    import dataclasses
    import json
    import tempfile
    from lerobot.configs.policies import PreTrainedConfig

    with open(checkpoint_path / "config.json") as f:
        raw = json.load(f)
    # `type` is draccus's choice discriminator, not a dataclass field — always keep it.
    valid = {f.name for f in dataclasses.fields(config_cls)} | {"type"}
    extra = sorted(set(raw) - valid)
    if not extra:
        return PreTrainedConfig.from_pretrained(str(checkpoint_path))
    LOGGER.warning(
        "Dropping config.json keys not defined on %s: %s", config_cls.__name__, extra
    )
    cleaned = {k: v for k, v in raw.items() if k in valid}
    with tempfile.TemporaryDirectory() as td:
        with open(Path(td) / "config.json", "w") as f:
            json.dump(cleaned, f)
        return PreTrainedConfig.from_pretrained(td)


def _assert_checkpoint_type(config, policy_type: str, checkpoint_path: Path) -> None:
    """Fail loudly when --policy-type disagrees with the checkpoint's own config.json.

    ``_load_config_tolerant`` ends in ``PreTrainedConfig.from_pretrained``, which dispatches
    on the ``type`` discriminator inside config.json — so the config it hands back is the
    *checkpoint's*, not the class the caller asked for. Handing that to a different policy
    class loads only the tensors whose names happen to collide, and both flow-matching
    classes swallow the mismatch: ``PI0Policy.from_pretrained`` wraps its strict load in a
    bare try/except that merely prints, and ``load_state_dict`` copies the matching tensors
    before raising. Serving a pi05 checkpoint as pi0 therefore comes up looking healthy
    while ``state_proj`` and ``action_time_mlp`` sit at random init and the AdaRMS timestep
    conditioning is dropped outright — the denoiser has no usable notion of flow time and
    every emitted chunk is incoherent.
    """
    ckpt_type = getattr(config, "type", None)
    assert ckpt_type == policy_type, (
        f"--policy-type {policy_type!r} but {checkpoint_path}/config.json declares "
        f"type={ckpt_type!r}. These architectures do not share a state dict, and the load "
        f"path does not raise on the mismatch — it leaves layers at random init. Re-run "
        f"with --policy-type {ckpt_type}."
    )


def _override_n_action_steps(config, n_action_steps: int | None):
    """Force the policy's execution horizon (the depth of select_action's queue).

    Checkpoints carry whatever n_action_steps the training config had, which need not
    match how we serve them. The synchronous path calls select_action exactly
    action_horizon times, so a queue shallower than that refills mid-request and runs a
    second denoise pass on the *same, now-stale* observation — stitching two
    independently sampled chunks together with a discontinuity at the refill point.
    Setting this equal to action_horizon makes one request exactly one inference."""
    if n_action_steps is None:
        return config
    chunk_size = getattr(config, "chunk_size", None)
    assert chunk_size is not None, (
        f"{type(config).__name__} has no chunk_size; cannot validate n_action_steps override"
    )
    assert 1 <= n_action_steps <= chunk_size, (
        f"n_action_steps must be in [1, chunk_size={chunk_size}], got {n_action_steps}. "
        "chunk_size is baked into the weights and cannot be raised without retraining."
    )
    if n_action_steps != config.n_action_steps:
        LOGGER.warning(
            "Overriding checkpoint n_action_steps %d -> %d (chunk_size=%d)",
            config.n_action_steps, n_action_steps, chunk_size,
        )
    config.n_action_steps = n_action_steps
    return config


def _load_policy(checkpoint_path: Path, policy_type: str, n_action_steps: int | None = None):
    if policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        config = _load_config_tolerant(checkpoint_path, ACTConfig)
        _assert_checkpoint_type(config, policy_type, checkpoint_path)
        config = _override_n_action_steps(config, n_action_steps)
        return ACTPolicy.from_pretrained(str(checkpoint_path), config=config)
    elif policy_type in ("pi0", "pi0_fast"):
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        config = _load_config_tolerant(checkpoint_path, PI0Config)
        _assert_checkpoint_type(config, policy_type, checkpoint_path)
        config = _override_n_action_steps(config, n_action_steps)
        return PI0Policy.from_pretrained(str(checkpoint_path), config=config)
    elif policy_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        config = _load_config_tolerant(checkpoint_path, PI05Config)
        _assert_checkpoint_type(config, policy_type, checkpoint_path)
        config = _override_n_action_steps(config, n_action_steps)
        return PI05Policy.from_pretrained(str(checkpoint_path), config=config)
    raise ValueError(f"Unsupported policy type: {policy_type!r}")


def _load_dataset_stats(checkpoint_path: Path) -> dict:
    # lerobot saves stats into the pretrained_model dir as dataset_stats.json
    import json
    stats_path = checkpoint_path / "stats.json"
    assert stats_path.exists(), (
        f"dataset_stats.json not found at {stats_path}. "
        "Pass dataset_stats manually if your checkpoint predates this convention."
    )
    with open(stats_path) as f:
        raw = json.load(f)
    import torch
    return {k: {sk: torch.tensor(sv) for sk, sv in sv_dict.items()} for k, sv_dict in raw.items()}


# Map GEN2 wire frame camera keys → lerobot observation keys
# Adjust the values to match the keys your policy was trained with.
_CAMERA_KEY_MAP = {
    "observation/images/main_image":        "observation.images.head",
    "observation/images/left_wrist_image":  "observation.images.left_wrist",
    "observation/images/right_wrist_image": "observation.images.right_wrist",
}

LOGGER = logging.getLogger(__name__)


def _configure_rtc(
    policy: Any,
    policy_type: str,
    *,
    execution_horizon: int,
    schedule: str,
    max_guidance_weight: float,
) -> None:
    """Enable Real-Time Chunking on a freshly loaded flow-matching policy.

    RTC is only meaningful for the flow-matching denoisers (pi0 / pi05). For
    other policy types we leave the config untouched and fall back to plain
    chunked inference.
    """
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if policy_type not in ("pi0", "pi05"):
        LOGGER.warning(
            "RTC requested but policy_type=%s has no flow-matching denoiser to "
            "inpaint; running without RTC guidance.", policy_type,
        )
        return
    if not hasattr(policy, "init_rtc_processor"):
        LOGGER.warning(
            "Installed %s has no RTC support (init_rtc_processor missing); "
            "running without RTC guidance.", type(policy).__name__,
        )
        return
    policy.config.rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule(schedule),
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
    )
    # Rebuild the processor and re-attach it to the inner model; the policy was
    # constructed before we set rtc_config, so its processor is currently None.
    policy.init_rtc_processor()
    assert policy._rtc_enabled(), "RTC config set but _rtc_enabled() is False"
    LOGGER.info(
        "RTC enabled: schedule=%s execution_horizon=%d max_guidance_weight=%.1f",
        schedule, execution_horizon, max_guidance_weight,
    )


def _configure_rtc_train(policy: Any, policy_type: str) -> None:
    """Verify a checkpoint is prepared for train-time RTC (action-prefix conditioning).

    Unlike inference-time RTC (:func:`_configure_rtc`), this installs no processor and
    mutates no config: the conditioning is baked into the weights by fine-tuning. All we
    do here is fail loudly if the checkpoint was *not* trained for it, because feeding a
    prefix to a model that never learned to read one produces plausible-looking but wrong
    actions — a silent failure that is very hard to notice on a robot.

    ``policy.config.rtc_train_enabled`` is True only when both (a) the checkpoint's
    config.json carried ``rtc_train_enabled: true`` and (b) the runtime lerobot defines
    the field (an unpatched lerobot drops it in ``_load_config_tolerant`` and also lacks
    the ``action_prefix`` sampling path), so this single check gates on both.
    """
    global RTC_TRAIN_MIN_DELAY, RTC_TRAIN_MAX_DELAY

    assert policy_type in ("pi0", "pi05"), (
        f"--rtc-train requires a flow-matching policy (pi0/pi05); got {policy_type}"
    )
    assert getattr(policy.config, "rtc_train_enabled", False), (
        "--rtc-train given, but this checkpoint is not prepared for it "
        "(config.rtc_train_enabled is False, or the runtime lerobot lacks train-time RTC "
        "support). This path only works with a checkpoint fine-tuned with "
        "rtc_train_enabled=true; re-train with the rtc-train config, or drop --rtc-train "
        "to serve synchronously."
    )
    RTC_TRAIN_MIN_DELAY = int(getattr(policy.config, "rtc_train_min_delay", 0))
    RTC_TRAIN_MAX_DELAY = int(getattr(policy.config, "rtc_train_max_delay", 0))
    LOGGER.info(
        "Train-time RTC enabled: action-prefix conditioning (arXiv 2512.05964), "
        "trained delays in [%d, %d), no inference-time guidance.",
        RTC_TRAIN_MIN_DELAY, RTC_TRAIN_MAX_DELAY,
    )


def _init_policy(
    checkpoint_path: Path,
    policy_type: str,
    *,
    rtc_enabled: bool = False,
    rtc_train_enabled: bool = False,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    rtc_schedule: str = DEFAULT_RTC_SCHEDULE,
    max_guidance_weight: float = DEFAULT_MAX_GUIDANCE_WEIGHT,
    pin_state_dims: Sequence[str] = (),
    n_action_steps: int | None = None,
) -> None:
    """Load the policy + processors for the given checkpoint and populate module globals."""
    global POLICY, POLICY_TYPE, POLICY_ID, _DATASET_STATS, _PREPROCESSOR, _POSTPROCESSOR, _ACTION_NORMALIZER
    global _RELATIVE_ACTION_STEP
    global RTC_ENABLED, RTC_TRAIN_ENABLED, EXECUTION_HORIZON, PINNED_STATE_DIMS
    assert not (rtc_enabled and rtc_train_enabled), (
        "rtc_enabled (inference-time RTC) and rtc_train_enabled (train-time RTC) are "
        "mutually exclusive"
    )
    POLICY_TYPE = policy_type
    POLICY_ID = str(checkpoint_path)
    EXECUTION_HORIZON = execution_horizon
    print("Loading policy...", flush=True)
    POLICY = _load_policy(checkpoint_path, policy_type, n_action_steps).eval().cuda()
    print("Policy loaded.", flush=True)
    if rtc_enabled:
        print("Configuring RTC...", flush=True)
        _configure_rtc(
            POLICY,
            policy_type,
            execution_horizon=execution_horizon,
            schedule=rtc_schedule,
            max_guidance_weight=max_guidance_weight,
        )
    if rtc_train_enabled:
        print("Configuring train-time RTC...", flush=True)
        _configure_rtc_train(POLICY, policy_type)
    RTC_ENABLED = bool(getattr(POLICY, "_rtc_enabled", lambda: False)())
    RTC_TRAIN_ENABLED = bool(rtc_train_enabled)
    print("Loading dataset stats...", flush=True)
    _DATASET_STATS = _load_dataset_stats(checkpoint_path)
    print("Dataset stats loaded.", flush=True)
    PINNED_STATE_DIMS = _resolve_state_pins(pin_state_dims, _DATASET_STATS)
    for idx, mean in PINNED_STATE_DIMS.items():
        std = float(_DATASET_STATS["observation.state"]["std"][idx])
        print(
            f"Pinning state dim {idx} ({_STATE_DIM_NAMES[idx]}) to training mean "
            f"{mean:.5f} (std {std:.5f}); incoming values for it are ignored.",
            flush=True,
        )
    print("Building pre/post processors...", flush=True)
    _PREPROCESSOR, _POSTPROCESSOR = make_pre_post_processors(
        policy_cfg=POLICY,
        pretrained_path=str(checkpoint_path),
        dataset_stats=_DATASET_STATS,
        preprocessor_overrides={"device_processor": {"device": str(POLICY.config.device)}},
    )
    print("Processors built.", flush=True)
    print("Resetting preprocessor...", flush=True)
    _PREPROCESSOR.reset()
    print("Resetting postprocessor...", flush=True)
    _POSTPROCESSOR.reset()
    _ACTION_NORMALIZER = _resolve_action_normalizer(_PREPROCESSOR, _POSTPROCESSOR)
    _RELATIVE_ACTION_STEP = _resolve_relative_action_step(_PREPROCESSOR, _POSTPROCESSOR)
    if _RELATIVE_ACTION_STEP is not None:
        print(
            "Relative actions: the policy predicts action - state for all dims except "
            f"{_RELATIVE_ACTION_STEP.exclude_joints}; RTC prefixes are re-based onto each "
            "request's state.",
            flush=True,
        )
    print("Init complete.", flush=True)


async def _init_rl(agent_url: str | None, *, resize_image: int = DEFAULT_RL_RESIZE_IMAGE,
                   obs_dump_dir: str | None = None) -> None:
    """Attach a latent-space RL agent. Call after _init_policy().

    With ``agent_url=None`` the server keeps behaving exactly as it does without RL: the
    policy samples its own noise and no RL code runs on the request path.
    """
    global RL_BACKEND, RL_AGENT, RL_RESIZE_IMAGE, RL_OBS_DUMP
    if agent_url is None:
        RL_BACKEND, RL_AGENT = None, None
        return

    from server.pi0_latent import FlowLatentBackend

    assert POLICY is not None, "_init_rl() must be called after _init_policy()"
    assert resize_image > 0, f"resize_image must be positive, got {resize_image}"
    RL_RESIZE_IMAGE = resize_image
    RL_BACKEND = FlowLatentBackend(POLICY, str(POLICY_TYPE))

    if obs_dump_dir:
        from server.rl_obs_dump import RlObsDump
        features = getattr(POLICY.config, "input_features", {}) or {}
        RL_OBS_DUMP = RlObsDump(
            obs_dump_dir,
            camera_keys=_CAMERA_KEY_MAP,
            trained_shapes={k.rsplit(".", 1)[-1]: tuple(int(v) for v in f.shape[1:])
                            for k, f in features.items() if "image" in k},
            checkpoint=str(POLICY_ID),
            policy_type=str(POLICY_TYPE),
        )

    agent = RemoteAgent(agent_url)
    handshake = await agent.connect()
    assert handshake.latent_shape == RL_BACKEND.latent_shape, (
        f"agent produces latents of shape {handshake.latent_shape} but {POLICY_TYPE} denoises "
        f"{RL_BACKEND.latent_shape}; the agent's action space must match the policy's noise space"
    )
    RL_AGENT = agent
    LOGGER.info(
        "Latent RL enabled: agent=%s latent_shape=%s resize_image=%d",
        agent.url, RL_BACKEND.latent_shape, RL_RESIZE_IMAGE,
    )


def _agent_images(frame: dict[str, Any], size: int) -> dict[str, np.ndarray]:
    """Decode the wire frame's JPEGs into small uint8 HWC images for the agent."""
    return _agent_images_for(frame, _CAMERA_KEY_MAP, size)


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=True,
        server_features=server_features
    )


def _decode_jpeg_to_tensor(jpeg_bytes: bytes, device: str) -> torch.Tensor:
    """JPEG bytes → float32 CHW tensor on device, values in [0, 1]."""
    img_hwc = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")  # uint8 HWC
    return torch.from_numpy(img_hwc.astype("float32") / 255.0).permute(2, 0, 1).to(device)


_WARNED_IMAGE_KEYS: set[str] = set()

# Layout of the 25-dim state/action vector, matching the slices taken from the wire frame.
_STATE_DIM_NAMES: tuple[str, ...] = (
    *(f"left_arm[{i}]" for i in range(7)), "left_gripper",
    *(f"right_arm[{i}]" for i in range(7)), "right_gripper",
    *(f"chest[{i}]" for i in range(6)),
    *(f"neck[{i}]" for i in range(3)),
)
assert _STATE_DIM_NAMES == GEN2_DIM_NAMES, "action_limiter's dim layout has drifted from the server's"


def _resolve_state_pins(names: Sequence[str], stats: dict) -> dict[int, float]:
    """Map state-dim names (or bare indices) to that dim's training mean.

    Accepts the names in ``_STATE_DIM_NAMES`` (``left_gripper``, ``chest[5]``, ...) or
    integer indices into the 25-dim state vector.
    """
    means = stats["observation.state"]["mean"]
    assert len(means) == len(_STATE_DIM_NAMES), (
        f"checkpoint stats have {len(means)} state dims, expected {len(_STATE_DIM_NAMES)}"
    )
    pins: dict[int, float] = {}
    for name in names:
        if name.lstrip("-").isdigit():
            idx = int(name)
            assert 0 <= idx < len(_STATE_DIM_NAMES), (
                f"state dim index {idx} out of range 0..{len(_STATE_DIM_NAMES) - 1}"
            )
        else:
            assert name in _STATE_DIM_NAMES, (
                f"unknown state dim {name!r}; expected an index or one of: "
                f"{', '.join(_STATE_DIM_NAMES)}"
            )
            idx = _STATE_DIM_NAMES.index(name)
        pins[idx] = float(means[idx])
    return pins


def _apply_state_pins(state_25: np.ndarray, req_id: int) -> np.ndarray:
    """Overwrite pinned dims with their training mean, before normalization.

    Escape hatch for state dimensions the demonstrator held constant during data
    collection. Their training std is near zero, so MEAN_STD normalization turns a
    physically trivial offset into a huge input: ``left_gripper`` has std 0.0035, i.e.
    0.35% of gripper travel per sigma, so 3.7% of travel lands 10.5 sigma out. Because
    ``PI0.state_proj`` is a bare ``nn.Linear`` with no clipping and the resulting state
    token is attended to by *every* action token, one such dim rotates the whole state
    embedding off-manifold and degrades the entire chunk, not just its own output.

    Pinning is only sound where the dim genuinely carried no information in training —
    check the std and the per-episode spread before adding one. It hides a real hardware
    fault rather than fixing it, so treat it as a stopgap while the robot is recalibrated.
    """
    if not PINNED_STATE_DIMS:
        return state_25
    pinned = state_25.copy()
    for idx, mean in PINNED_STATE_DIMS.items():
        pinned[idx] = mean
    LOGGER.info(
        "predict req=%d state pinned: %s", req_id,
        " ".join(
            f"{_STATE_DIM_NAMES[idx]} {state_25[idx]:.5f}->{mean:.5f}"
            for idx, mean in PINNED_STATE_DIMS.items()
        ),
    )
    return pinned


def _check_image_resolutions(obs: dict[str, Any], req_id: int) -> None:
    """Compare each decoded camera image against the resolution the checkpoint was
    trained on, and warn once per camera on mismatch.

    The server passes decoded JPEGs straight through, so a camera streaming at a
    different resolution than training is never rejected — pi0's
    ``resize_with_pad_torch`` silently rescales it into the 224x224 frame. Because it
    preserves aspect ratio and pads with black, a different *aspect* puts the scene at
    a different scale behind different-sized bars than the model saw in training. That
    survives coarse behaviour (reaching) but degrades anything alignment-sensitive
    (grasping), which makes it worth surfacing loudly.
    """
    expected = getattr(POLICY.config, "input_features", {}) or {}
    shapes = []
    for lerobot_key in _CAMERA_KEY_MAP.values():
        if lerobot_key not in obs:
            continue
        _, _, h, w = obs[lerobot_key].shape
        feature = expected.get(lerobot_key)
        want = tuple(feature.shape[1:]) if feature is not None else None
        shapes.append(f"{lerobot_key.rsplit('.', 1)[-1]}={h}x{w}")
        if want is not None and (h, w) != want:
            shapes[-1] += f"(TRAINED {want[0]}x{want[1]})"
            if lerobot_key not in _WARNED_IMAGE_KEYS:
                _WARNED_IMAGE_KEYS.add(lerobot_key)
                LOGGER.warning(
                    "predict req=%d IMAGE RESOLUTION MISMATCH %s: receiving %dx%d but "
                    "checkpoint was trained on %dx%d. Aspect %.3f vs %.3f — pi0 pads to "
                    "224x224 preserving aspect, so the scene lands at a different scale "
                    "behind different-sized black bars than in training.",
                    req_id, lerobot_key, h, w, want[0], want[1], w / h, want[1] / want[0],
                )
    LOGGER.info("predict req=%d image_shapes=[%s]", req_id, " ".join(shapes))


def _check_normalized_state(obs_processed: dict[str, Any], req_id: int) -> None:
    """Warn when the normalized state is far outside the training distribution.

    Restores the sanity check that existed before the server refactor. State is
    normalized MEAN_STD from the checkpoint's own stats, so in-distribution input is
    O(1); large values mean the 25-dim slice taken from the 97-dim wire state does not
    line up with the features the policy was trained on.
    """
    state = obs_processed.get("observation.state")
    if state is None:
        return
    norm_state = state.squeeze(0).cpu().float()
    abs_max = float(norm_state.abs().max())
    worst = int(norm_state.abs().argmax())
    LOGGER.info(
        "predict req=%d norm_state_abs_max=%.2f norm_state=%s",
        req_id, abs_max,
        np.array2string(norm_state.numpy(), precision=2, suppress_small=True, max_line_width=400),
    )
    # 5 sigma, not 10: dimensions the demonstrator held still during data collection get a
    # tiny training std, so a small absolute drift in a joint nobody is using still lands
    # far outside the training range. Those are exactly the ones worth catching early, and
    # they never reach a 10-sigma threshold from a plausible amount of drift.
    if abs_max > 5.0:
        LOGGER.warning(
            "predict req=%d POSSIBLE STATE MISMATCH: %s is %.1f sigma from the training "
            "mean (every other dim is within %.1f). A joint that was held constant during "
            "data collection has a near-zero training std, so even small physical drift in "
            "it reads as a large out-of-distribution input to the state encoder.",
            req_id, _STATE_DIM_NAMES[worst] if worst < len(_STATE_DIM_NAMES) else f"dim[{worst}]",
            abs_max,
            float(np.partition(norm_state.abs().numpy(), -2)[-2]),
        )


def _apply_action_processor(step: Any, action: torch.Tensor) -> torch.Tensor:
    """Run one normalize/unnormalize processor step over a bare action tensor."""
    return step({TransitionKey.ACTION: action})[TransitionKey.ACTION]


def _resolve_action_normalizer(preprocessor: Any, postprocessor: Any) -> Any:
    """Return the preprocessor's action normalizer, verified to invert the postprocessor.

    Reusing the pipeline's own step keeps wire-prefix conversion in lockstep with the
    checkpoint's normalization mode and stats instead of hard-coding MEAN_STD here. The
    round-trip check is the part that earns its keep: it fails at startup, not on the
    robot, if the two pipelines ever stop being inverses.
    """
    normalizers = [s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)]
    unnormalizers = [s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep)]
    assert len(normalizers) == 1, (
        f"expected exactly one NormalizerProcessorStep in the preprocessor, found {len(normalizers)}"
    )
    assert len(unnormalizers) == 1, (
        f"expected exactly one UnnormalizerProcessorStep in the postprocessor, found {len(unnormalizers)}"
    )
    normalizer = normalizers[0]
    probe = torch.randn(4, DEFAULT_HARDWARE_MODEL.action_dim, device=str(POLICY.config.device))
    round_trip = _apply_action_processor(normalizer, _apply_action_processor(unnormalizers[0], probe))
    max_err = float((round_trip - probe).abs().max())
    assert max_err < 1e-4, (
        "the preprocessor's action normalizer does not invert the postprocessor's "
        f"unnormalizer (max round-trip error {max_err:.3e}); wire action prefixes cannot "
        "be converted into the policy's action space safely"
    )
    return normalizer


def _resolve_relative_action_step(preprocessor: Any, postprocessor: Any) -> RelativeActionsProcessorStep | None:
    """Return the preprocessor's enabled relative-action step, or None for an absolute policy.

    Checked for consistency with the postprocessor: a relative policy whose outputs are
    never made absolute again (or vice versa) would drive the robot to joint *offsets*
    treated as positions, and the prefix conversion below would silently disagree with the
    chunks we return.
    """
    relative = [s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled]
    absolute = [s for s in postprocessor.steps if isinstance(s, AbsoluteActionsProcessorStep) and s.enabled]
    assert len(relative) == len(absolute) <= 1, (
        f"expected matching relative/absolute action steps, found {len(relative)} enabled "
        f"RelativeActionsProcessorStep(s) and {len(absolute)} enabled AbsoluteActionsProcessorStep(s)"
    )
    if not relative:
        return None
    assert absolute[0].relative_step is relative[0], (
        "the postprocessor's AbsoluteActionsProcessorStep is not linked to the preprocessor's "
        "RelativeActionsProcessorStep; returned chunks would be made absolute against a "
        "different state than RTC prefixes are made relative to"
    )
    return relative[0]


def _to_policy_action_space(actions: torch.Tensor) -> torch.Tensor:
    """Convert robot-space actions into the policy's action space for this request.

    The exact inverse of the postprocessor: subtract the request's state for the
    relative dims (relative-action checkpoints only), then normalize. The relative step
    must already have cached this request's state -- true whenever this runs after
    :func:`_prepare_policy_observation`, which every RTC path does.
    """
    assert _ACTION_NORMALIZER is not None, "_init_policy must run before a prefix is converted"
    if _RELATIVE_ACTION_STEP is not None:
        state = _RELATIVE_ACTION_STEP.get_cached_state()
        assert state is not None, "the preprocessor must run on this request before its prefix is converted"
        actions = _RELATIVE_ACTION_STEP({
            TransitionKey.OBSERVATION: {OBS_STATE: state},
            TransitionKey.ACTION: actions,
        })[TransitionKey.ACTION]
    return _apply_action_processor(_ACTION_NORMALIZER, actions)


def _normalize_wire_prefix(wire_prefix: Any, *, device: str) -> torch.Tensor:
    """Convert a wire ``action_prefix`` from robot action space into the policy's.

    Per the upstream spec (README.md) the wire carries robot-space actions -- the same
    units and dim the server returns in ``action``. Both RTC paths hand the prefix to
    lerobot as model-space actions, so this conversion is mandatory. Omitting it fails
    silently rather than loudly: under ACTION: MEAN_STD raw joint values land several
    sigma from anything the model saw in training, and it conditions on them just as
    willingly -- yielding confident, wrong actions instead of an error. The same holds
    for a relative-action checkpoint fed absolute positions.
    """
    assert _ACTION_NORMALIZER is not None, "_init_policy must run before a wire prefix arrives"
    prefix = torch.as_tensor(np.asarray(wire_prefix, dtype="float32"), device=device)
    normalized = _to_policy_action_space(prefix).to(device)
    # Same 5-sigma rule as _check_normalized_state, for the same reason. A client that
    # pre-normalizes its prefix lands here too: normalizing an already-normalized chunk
    # centres it on -mean/std, which for these stats is well past 5.
    abs_max = float(normalized.abs().max())
    if abs_max > 5.0:
        LOGGER.warning(
            "wire action_prefix sits %.1f sigma from the training mean after "
            "normalization. action_prefix is ROBOT-space per the spec -- a client that "
            "pre-normalizes it will look exactly like this.",
            abs_max,
        )
    return normalized


_PREFIX_JOINT_DIMS: tuple[int, ...] = tuple(
    i for i in range(len(GEN2_DIM_NAMES)) if i not in GRIPPER_DIMS
)


def _prefix_continuity_gap(
    frame: dict[str, Any], wire_prefix: Any
) -> tuple[float, int] | None:
    """How far the prefix's first row sits from the state that arrived with it.

    Returns ``(max_abs_gap_rad, dim_index)`` over the joint dims, or None when the frame
    carries no state to compare against.

    Both sides are in robot space and share the 25-dim layout, so this is a direct
    comparison. ``action_prefix[0]`` is the command for the control step immediately after
    the request is sent, so the gap is the servo's tracking error -- which is small only
    while the arm is slow. It grows with joint velocity, and on the GEN2 at 50 Hz it
    reaches 14 deg in ordinary motion, so it does NOT separate a stale prefix from a fast
    one. Read it as a diagnostic, and see _resume_reason for the detector that works.

    Grippers are excluded: they are in their own units (0-1, not radians) and legitimately
    travel their whole range in a single command, so they carry no continuity signal.
    """
    raw_state = frame.get(JOINT_STATE_KEY)
    if raw_state is None:
        return None
    prefix = np.asarray(wire_prefix, dtype="float32")
    if prefix.ndim != 2 or prefix.shape[0] == 0:
        return None
    state_25 = _wire_state_25(np.asarray(raw_state, dtype="float32"))
    if prefix.shape[1] != state_25.shape[0]:
        return None
    dims = np.asarray(_PREFIX_JOINT_DIMS)
    gaps = np.abs(prefix[0, dims] - state_25[dims])
    worst = int(np.argmax(gaps))
    return float(gaps[worst]), int(dims[worst])


def _resume_reason(rtc_state: _RtcState, gap_s: float | None, *, req_id: int) -> str | None:
    """Why this request must start a fresh plan and ignore any prefix, or None.

    Call exactly once per request: it consumes the one-shot ``resume_pending`` flag.

    A prefix is a promise that its rows are executing on the robot right now, while the
    request is in flight. Three things break that promise, and all three are visible to
    the server without any cooperation from the client:

    * ``resume_pending`` -- this connection has served no chunk yet, or an
      ``endpoint=intervention`` frame voided the one it did. Any prefix on such a request
      belongs to a plan this connection knows nothing about.
    * A long arrival gap. An async rollout re-requests as fast as it can, so its gaps are
      one round trip (160 ms in the field logs). A gap of seconds means the client stopped
      asking -- it was paused -- and whatever it sends next was computed before that.
    * Nothing else. In particular NOT "the prefix is far from the current state": that
      fires on ordinary fast motion, where the arm legitimately runs 15 deg behind its own
      command at 70 deg/s, and dropping the prefix there is self-reinforcing -- the
      unconditioned chunk that replaces it agrees with its neighbours even less, so the
      gap stays wide and the conditioning never comes back. See MAX_PREFIX_GAP.

    Dropping the prefix costs one unconditioned chunk planned from the live observation,
    which is exactly what a resume should do. Honouring it pins the new chunk's opening
    steps to a trajectory the robot has since left.
    """
    pending = rtc_state.resume_pending
    rtc_state.resume_pending = False
    if pending:
        return "resume"
    if RESUME_GAP_S is not None and gap_s is not None and gap_s >= RESUME_GAP_S:
        LOGGER.warning(
            "predict req=%d RESUME after a %.1fs gap in requests (steady state is one "
            "round trip). Ignoring the action_prefix: it was computed before the gap, so "
            "it does not describe what this robot is about to do. Planning an "
            "unconditioned chunk from the live observation instead.",
            req_id, gap_s,
        )
        return f"resume-gap={gap_s:.1f}s"
    return None


def _reject_stale_prefix(frame: dict[str, Any], wire_prefix: Any, *, req_id: int) -> bool:
    """True when the wire prefix sits further from the current state than MAX_PREFIX_GAP.

    Off by default, and deliberately so -- see the field log quoted on MAX_PREFIX_GAP.
    Useful only against a client that pauses without any gap in its request stream (one
    that keeps polling while held), where :func:`_resume_reason` has nothing to see. Set
    it well above the tracking error the arm shows at speed, or it will fire mid-motion.
    """
    if MAX_PREFIX_GAP is None:
        return False
    measured = _prefix_continuity_gap(frame, wire_prefix)
    if measured is None:
        return False
    gap, dim = measured
    if gap <= MAX_PREFIX_GAP:
        return False
    LOGGER.warning(
        "predict req=%d action_prefix dropped: its first row is %.1f deg from the current "
        "state on %s (--max-prefix-gap-deg %.1f). Planning an unconditioned chunk from the "
        "live observation instead. If the arm is simply moving fast this limit is too "
        "tight -- raise it or set it to 0.",
        req_id, np.rad2deg(gap), _STATE_DIM_NAMES[dim], np.rad2deg(MAX_PREFIX_GAP),
    )
    return True


def _resolve_rtc_kwargs(
    frame: dict[str, Any],
    rtc_state: _RtcState,
    *,
    device: str,
    execution_horizon: int,
    req_id: int = -1,
) -> tuple[dict[str, Any], str]:
    """Translate wire prefix fields + server state into RTC kwargs for
    ``predict_action_chunk``.

    Returns ``(kwargs, debug_str)``. ``kwargs`` is empty when there is no usable
    prefix (e.g. the first request of an episode), in which case the model runs
    an unguided chunk.
    """
    resume = _resume_reason(rtc_state, _consume_arrival_gap(rtc_state), req_id=req_id)
    if resume is not None:
        return {}, f"rtc=off({resume})"
    # inference_delay (a.k.a. freeze length): how many steps at the start of the
    # new chunk are already committed and will execute before this chunk is ready.
    raw_delay = frame.get(PREFIX_CHANGE_START_KEY)
    inference_delay = int(raw_delay) if raw_delay is not None else 0
    if inference_delay < 0:
        inference_delay = 0

    wire_prefix = frame.get(ACTION_PREFIX_KEY)
    if wire_prefix is not None and _reject_stale_prefix(frame, wire_prefix, req_id=req_id):
        return {}, "rtc=off(stale-prefix)"
    if wire_prefix is not None:
        # Client-supplied leftover, already aligned to the new chunk's first step and in
        # robot action space; normalize it into the space last_chunk (and the model) use.
        prev = _normalize_wire_prefix(wire_prefix, device=device)
        source = "wire"
    elif rtc_state.last_chunk is not None:
        # Fall back to our memory of the previous returned chunk, dropping the
        # steps already consumed during inference (mirrors lerobot's
        # ActionQueue._replace_actions_queue: original_actions[real_delay:]).
        # This branch is not staleness-checked -- the connection handler drops last_chunk
        # outright whenever the robot stops executing our chunks (intervention / done).
        t_prev = rtc_state.last_chunk.shape[0]
        prev = _to_policy_action_space(rtc_state.last_chunk[min(inference_delay, t_prev):])
        source = "state"
    else:
        return {}, "rtc=off(no-prefix)"

    if prev.shape[0] == 0:
        return {}, f"rtc=off(empty-prefix src={source} delay={inference_delay})"

    # Don't freeze more steps than we actually have a prefix for.
    inference_delay = min(inference_delay, prev.shape[0])
    kwargs = {
        "prev_chunk_left_over": prev,
        "inference_delay": inference_delay,
        "execution_horizon": execution_horizon,
    }
    debug = (
        f"rtc=on src={source} prefix_len={prev.shape[0]} "
        f"inference_delay={inference_delay} exec_horizon={execution_horizon}"
    )
    return kwargs, debug


def _predict_chunk_rtc(
    obs_processed: dict[str, Any],
    frame: dict[str, Any],
    rtc_state: _RtcState,
    *,
    device: str,
    action_horizon: int,
    req_id: int,
) -> tuple[np.ndarray, float]:
    """RTC path: one full denoise pass via predict_action_chunk, with prefix
    guidance when a usable prefix exists."""
    rtc_kwargs, rtc_debug = _resolve_rtc_kwargs(
        frame, rtc_state, device=device, execution_horizon=EXECUTION_HORIZON, req_id=req_id,
    )
    rtc_state.committed_prefix_len = int(rtc_kwargs.get("inference_delay", 0))
    t0 = time.perf_counter()
    # predict_action_chunk runs one full denoise pass and returns the entire
    # normalized chunk (1, chunk_size, action_dim). RTC guidance, when active, is
    # applied inside the denoise loop via the kwargs above. The method manages
    # its own grad context, so we don't wrap it in torch.no_grad().
    raw_chunk = POLICY.predict_action_chunk(obs_processed, **rtc_kwargs)  # (1, T, A)
    raw_chunk = raw_chunk.squeeze(0)  # (T, A), normalized output space
    # _inference_response remembers the final returned actions after limiting.
    # Postprocess (unnormalize) into robot action space. The postprocessor expects
    # a (T, 1, A) tensor, matching the per-step stacking in the synchronous path.
    action_full = (
        _POSTPROCESSOR(raw_chunk.unsqueeze(1)).squeeze(1).cpu().numpy().astype("float32")
    )
    inference_time_s = time.perf_counter() - t0

    # Honor the requested wire horizon (slice the model chunk down to it).
    action = action_full[:action_horizon]
    LOGGER.info(
        "predict req=%d %s chunk=%d returned=%d inference_s=%.3f",
        req_id, rtc_debug, raw_chunk.shape[0], action.shape[0], inference_time_s,
    )
    return action, inference_time_s


def _consume_arrival_gap(rtc_state: _RtcState) -> float | None:
    """Seconds since the previous request on this connection; None on the first.

    Call exactly once per request: it advances the arrival clock. The gap is the server's
    only view of the client's control loop -- an async rollout re-requests as fast as it
    can, so in steady state the gap IS the round trip, and a gap far larger than that
    means the client stopped asking (see :func:`_resume_reason`).
    """
    now = time.perf_counter()
    prev_arrival = rtc_state.last_arrival_s
    rtc_state.last_arrival_s = now
    return None if prev_arrival is None else now - prev_arrival


def _log_delay_diagnostics(
    frame: dict[str, Any], gap_s: float | None, *, req_id: int
) -> None:
    """Report the delay the client claims next to the one its own request rate implies.

    ``prefix_change_start`` is the number of control steps that elapse between a request
    being sent and its chunk being ready. The server cannot see the client's clock, but a
    rollout that re-requests as fast as it can turns the gap between successive arrivals
    into exactly that round trip, so ``gap * rollout_hz`` is what the client should be
    sending. A large disagreement means the client derives the delay from something other
    than its measured latency -- most often its leftover queue length, which is a
    different and usually much larger number.
    """
    if gap_s is None:
        return  # first request of the connection: no gap to measure yet

    implied = int(round(gap_s * ROLLOUT_HZ))
    raw_delay = frame.get(PREFIX_CHANGE_START_KEY)
    sent = int(raw_delay) if raw_delay is not None else 0
    wire_prefix = frame.get(ACTION_PREFIX_KEY)
    prefix_rows = 0 if wire_prefix is None else int(np.asarray(wire_prefix).shape[0])
    LOGGER.info(
        "predict req=%d delay sent=%d prefix_rows=%d arrival_gap=%.0fms "
        "implied_delay=%d@%.0fHz trained=[%d,%d)",
        req_id, sent, prefix_rows, gap_s * 1e3, implied, ROLLOUT_HZ,
        RTC_TRAIN_MIN_DELAY, RTC_TRAIN_MAX_DELAY,
    )
    # Only flag a disagreement big enough to matter: off by more than 3 steps AND more
    # than double (or less than half) the implied delay.
    if sent > 0 and abs(sent - implied) > 3 and abs(sent - implied) > implied:
        LOGGER.warning(
            "predict req=%d prefix_change_start=%d, but this client's own request rate "
            "implies ~%d steps of delay at %.0f Hz. prefix_change_start is the INFERENCE "
            "DELAY -- the steps that execute while the request is in flight -- not the "
            "number of actions left in the queue.",
            req_id, sent, implied, ROLLOUT_HZ,
        )


def _warn_on_untrained_delay(d: int, *, req_id: int) -> None:
    """Flag a prefix longer than anything the checkpoint was fine-tuned to read.

    Train-time RTC pins the first ``d`` chunk positions to the prefix at every denoise
    step. The model only learned to continue prefixes of the lengths it saw in training,
    so a ``d`` outside that range puts the pinned region out of distribution and makes
    the free postfix that follows it unreliable."""
    if RTC_TRAIN_MAX_DELAY <= 0:
        return  # trained range unknown (not the rtc-train path)
    if RTC_TRAIN_MIN_DELAY <= d < RTC_TRAIN_MAX_DELAY:
        return
    LOGGER.warning(
        "predict req=%d prefix_change_start=%d is outside the trained delay range "
        "[%d, %d). The model never saw a prefix of this length, so the %d pinned "
        "positions are out of distribution.",
        req_id, d, RTC_TRAIN_MIN_DELAY, RTC_TRAIN_MAX_DELAY, d,
    )


def _warn_on_unconditioned_chunk(rtc_state: _RtcState, d: int, *, req_id: int) -> None:
    """Flag a chunk that is an independent sample rather than a continuation.

    Legitimate whenever nothing is committed: the first request of an episode, and the
    first request after a pause or a human intervention (there the client MUST drop the
    prefix -- the abandoned chunk no longer describes what the robot is about to do).
    A sustained run of them is the bug: the client stopped sending the ``action_prefix`` /
    ``prefix_change_start`` pair altogether, and consecutive chunks become i.i.d. draws
    from a multimodal policy. An async rollout that re-plans every few control steps then
    executes only the opening steps of chunks that disagree about where to go, and the
    motion averages out to nothing while inference keeps running normally."""
    if rtc_state.last_chunk is None:
        return  # first request of the episode: nothing is committed yet
    rtc_state.unconditioned_streak += 1
    streak = rtc_state.unconditioned_streak
    if streak == 1:
        return  # one unconditioned chunk is how a pause/intervention is meant to resume
    if streak != 2 and streak % 25 != 0:
        return  # already reported; keep the log readable
    LOGGER.warning(
        "predict req=%d prefix_change_start=%d with a previous chunk in hand (%d in a "
        "row): these chunks are UNCONDITIONED, not continuations of one another. Past the "
        "first request of an episode -- and the first one after a pause or intervention, "
        "where dropping the prefix is correct -- the client must send action_prefix and "
        "prefix_change_start together on every request; the wire format rejects one "
        "without the other.",
        req_id, d, streak,
    )


def _chunk_diagnostics(
    raw_chunk: torch.Tensor,
    prev_chunk: torch.Tensor | None,
    prefix: torch.Tensor | None,
    *,
    d: int,
) -> str:
    """Numbers that separate "the policy is continuing its last plan" from "the policy is
    redrawing an independent plan every request". All in normalized action space, so the
    magnitudes are comparable across checkpoints and joints.

    * ``travel``   -- ||chunk[-1] - chunk[0]||: how far the chunk commits to moving.
      Collapses toward zero when the policy stops committing to anything.
    * ``max_step`` -- the largest single-step jump inside the chunk.
    * ``seam``     -- ||new[0] - prev[d]||: the discontinuity the robot feels where the
      new chunk is spliced onto the old one. This is the jerk RTC exists to remove.
    * ``overlap``  -- mean ||new[i] - prev[d+i]|| over the region both chunks cover.
      Small means consecutive chunks agree on the plan; large means they are effectively
      independent samples and the executed motion will cancel itself out.
    * ``pin_err``  -- max ||new[i] - prefix[i]|| over the pinned prefix. Train-time RTC
      overwrites those positions with the clean prefix at every denoise step, so this
      must be ~0. Anything larger means the conditioning did not actually take.
    """
    parts = []
    if raw_chunk.shape[0] > 1:
        travel = torch.linalg.vector_norm(raw_chunk[-1] - raw_chunk[0])
        max_step = torch.linalg.vector_norm(raw_chunk[1:] - raw_chunk[:-1], dim=-1).max()
        parts.append(f"travel={float(travel):.3f} max_step={float(max_step):.3f}")

    if prev_chunk is not None and prev_chunk.shape[-1] == raw_chunk.shape[-1]:
        # prev_chunk is indexed in the OLD chunk's frame; the new chunk starts d steps
        # later, so drop d to line the two up step for step.
        aligned = prev_chunk[min(d, prev_chunk.shape[0]):]
        overlap = min(aligned.shape[0], raw_chunk.shape[0])
        if overlap > 0:
            dist = torch.linalg.vector_norm(raw_chunk[:overlap] - aligned[:overlap], dim=-1)
            parts.append(
                f"seam={float(dist[0]):.3f} overlap={float(dist.mean()):.3f} n={overlap}"
            )

    if prefix is not None and prefix.shape[0] > 0 and prefix.shape[-1] == raw_chunk.shape[-1]:
        pinned = min(prefix.shape[0], raw_chunk.shape[0])
        pin_err = torch.linalg.vector_norm(
            raw_chunk[:pinned] - prefix[:pinned], dim=-1
        ).max()
        parts.append(f"pin_err={float(pin_err):.4f}")

    return " ".join(parts)


def _resolve_prefix_kwargs(
    frame: dict[str, Any],
    rtc_state: _RtcState,
    *,
    device: str,
    req_id: int = -1,
) -> tuple[dict[str, Any], str]:
    """Translate wire prefix fields + server state into train-time RTC kwargs for
    ``predict_action_chunk`` (``action_prefix`` / ``prefix_len``).

    Train-time RTC conditions on a HARD prefix of exactly ``d = prefix_change_start``
    committed steps, aligned to the new chunk's first step — the same frozen prefix
    inference-time RTC uses (:func:`_resolve_rtc_kwargs` produces the identical ``prev``),
    so the two paths are directly comparable. ``d == 0`` means nothing is committed yet,
    so the model runs a plain unconditioned chunk.

    Returns ``(kwargs, debug_str)``. ``kwargs`` is empty when there is no usable prefix
    (first request of an episode, or ``d == 0``), in which case the caller runs an
    unconditioned chunk.
    """
    gap_s = _consume_arrival_gap(rtc_state)
    _log_delay_diagnostics(frame, gap_s, req_id=req_id)
    resume = _resume_reason(rtc_state, gap_s, req_id=req_id)

    raw_delay = frame.get(PREFIX_CHANGE_START_KEY)
    d = int(raw_delay) if raw_delay is not None else 0
    if d <= 0:
        _warn_on_unconditioned_chunk(rtc_state, d, req_id=req_id)
        return {}, "rtc_train=off(delay=0)"
    rtc_state.unconditioned_streak = 0
    _warn_on_untrained_delay(d, req_id=req_id)

    if resume is not None:
        # Ignore both the wire prefix and our own remembered chunk: after a resume neither
        # describes what the robot is about to do.
        return {}, f"rtc_train=off({resume})"

    wire_prefix = frame.get(ACTION_PREFIX_KEY)
    if wire_prefix is not None and _reject_stale_prefix(frame, wire_prefix, req_id=req_id):
        return {}, "rtc_train=off(stale-prefix)"
    if wire_prefix is not None:
        # Client-supplied leftover, already aligned to the new chunk's first step and in
        # robot action space; normalize it into the space last_chunk (and the model) use.
        prev = _normalize_wire_prefix(wire_prefix, device=device)
        source = "wire"
    elif rtc_state.last_chunk is not None:
        # Fall back to our memory of the previous returned chunk, dropping the steps
        # already consumed during inference so prev[0] aligns to the new chunk's first
        # step (identical to _resolve_rtc_kwargs). Not staleness-checked, as there; the
        # connection handler drops last_chunk on intervention/done.
        t_prev = rtc_state.last_chunk.shape[0]
        prev = _to_policy_action_space(rtc_state.last_chunk[min(d, t_prev):])
        source = "state"
    else:
        return {}, "rtc_train=off(no-prefix)"

    # Never claim a longer prefix than we actually have actions for.
    d = min(d, prev.shape[0])
    if d == 0:
        return {}, f"rtc_train=off(empty-prefix src={source})"

    # The hard prefix is the first d aligned steps — exactly prev_chunk_left_over[:d],
    # the region inference-time RTC freezes. The model pads the action dim internally.
    kwargs = {"action_prefix": prev[:d].contiguous(), "prefix_len": d}
    debug = f"rtc_train=on src={source} prefix_len={d}"
    return kwargs, debug


def _predict_chunk_rtc_train(
    obs_processed: dict[str, Any],
    frame: dict[str, Any],
    rtc_state: _RtcState,
    *,
    device: str,
    action_horizon: int,
    req_id: int,
) -> tuple[np.ndarray, float]:
    """Train-time RTC path: one plain denoise pass via predict_action_chunk, conditioned
    on a hard action prefix. No pseudoinverse guidance and no per-step VJP — the prefix is
    an ordinary model input the checkpoint was fine-tuned to read."""
    prefix_kwargs, debug = _resolve_prefix_kwargs(
        frame, rtc_state, device=device, req_id=req_id
    )
    rtc_state.committed_prefix_len = int(prefix_kwargs.get("prefix_len", 0))
    # Hold the previous chunk; _chunk_diagnostics measures
    # the new chunk against it to tell a continuation from an independent redraw. Both
    # must be in this request's policy space for the distances to mean anything.
    prev_chunk = (
        None if rtc_state.last_chunk is None else _to_policy_action_space(rtc_state.last_chunk)
    )
    t0 = time.perf_counter()
    # predict_action_chunk runs one full denoise pass and returns the entire normalized
    # chunk (1, chunk_size, action_dim). sample_actions is @torch.no_grad, so — unlike the
    # guidance path — there is no grad machinery to manage.
    raw_chunk = POLICY.predict_action_chunk(obs_processed, **prefix_kwargs)  # (1, T, A)
    raw_chunk = raw_chunk.squeeze(0)  # (T, A), normalized output space
    # _inference_response remembers the final returned actions after limiting.
    # Postprocess (unnormalize) into robot action space; (T, 1, A) matches the per-step
    # stacking the other paths use.
    action_full = (
        _POSTPROCESSOR(raw_chunk.unsqueeze(1)).squeeze(1).cpu().numpy().astype("float32")
    )
    inference_time_s = time.perf_counter() - t0

    action = action_full[:action_horizon]
    # Measured after inference_time_s so the extra device syncs never inflate it.
    # Align the comparison on the delay the CLIENT claims, not on the prefix we ended up
    # using. When the prefix is dropped there is no prefix_len, and falling back to 0 would
    # measure the new chunk against prev_chunk[0:] -- a different, larger quantity than the
    # prev_chunk[d:] every other request reports, making seam/overlap incomparable across
    # exactly the requests worth comparing.
    claimed_delay = frame.get(PREFIX_CHANGE_START_KEY)
    diagnostics = _chunk_diagnostics(
        raw_chunk,
        prev_chunk,
        prefix_kwargs.get("action_prefix"),
        d=int(prefix_kwargs.get("prefix_len", max(int(claimed_delay or 0), 0))),
    )
    LOGGER.info(
        "predict req=%d %s chunk=%d returned=%d inference_s=%.3f diagnostics=pre-limit %s",
        req_id, debug, raw_chunk.shape[0], action.shape[0], inference_time_s, diagnostics,
    )
    return action, inference_time_s


def _predict_chunk_sync(
    obs_processed: dict[str, Any],
    *,
    action_horizon: int,
    req_id: int,
) -> tuple[np.ndarray, float]:
    """Synchronous path: drain select_action's internal queue action_horizon times."""
    queue_before = _policy_queue_info(POLICY)
    t0 = time.perf_counter()
    with torch.no_grad():
        actions = [POLICY.select_action(obs_processed) for _ in range(action_horizon)]
        action = _POSTPROCESSOR(torch.stack(actions, dim=0)).squeeze(1).cpu().numpy().astype("float32")
    inference_time_s = time.perf_counter() - t0
    queue_after = _policy_queue_info(POLICY)
    LOGGER.info(
        "predict req=%d queue_before=%s queue_after=%s inference_s=%.3f",
        req_id, queue_before, queue_after, inference_time_s,
    )
    return action, inference_time_s


async def _predict_chunk_rl(
    obs_processed: dict[str, Any],
    frame: dict[str, Any],
    rl_state: _RlState,
    *,
    state: np.ndarray,
    prompt: str,
    action_horizon: int,
    req_id: int,
) -> tuple[np.ndarray, float, float | None]:
    """RL path: condition once, let the agent choose the starting latent, then denoise.

    The agent is consulted *between* the two halves of inference because the latent it
    must choose depends on the policy's own view of the scene (the prefix embedding),
    which only exists after the conditioning pass.
    """
    assert RL_BACKEND is not None and RL_AGENT is not None, "RL path reached without an agent"
    t0 = time.perf_counter()

    # Queued, not written: the writer runs on its own thread so the robot never waits
    # on disk. The learner's dump keys on this same step id.
    if RL_OBS_DUMP is not None:
        RL_OBS_DUMP.submit(req_id, frame, task=prompt)

    prefix_rep, cache = RL_BACKEND.prefix(obs_processed)
    observation = AgentObservation(
        images=_agent_images(frame, RL_RESIZE_IMAGE),
        state=state.astype("float32"),
        prefix_rep=prefix_rep[0],
        task=prompt,
    )

    # Report the *previous* step's outcome before asking for a new latent, so the agent
    # decides with the freshest possible information. Reward and done ride in on this
    # request's frame because they describe what the last chunk accomplished.
    if rl_state.last_step_id is not None:
        # How many actions of the previous chunk actually ran. The client reports this
        # when it truncates a chunk; without it we can only assume the whole thing ran.
        executed_raw = frame.get(PREV_SKIPPED_ACTION_START_KEY)
        executed_steps = int(executed_raw) if executed_raw else action_horizon
        reward_raw = frame.get(REWARD_KEY)
        await RL_AGENT.observe(
            Transition(
                step_id=req_id,
                prev_step_id=rl_state.last_step_id,
                executed_steps=max(1, executed_steps),
                reward=None if reward_raw is None else float(reward_raw),
                done=bool(frame.get(DONE_KEY, False)),
            )
        )

    agent_action = await RL_AGENT.act(req_id, observation)
    raw_chunk = RL_BACKEND.denoise(cache, agent_action.latent[np.newaxis, ...])  # (1, T, A)
    rl_state.last_step_id = req_id

    # Postprocess (unnormalize) into robot action space; (T, 1, A) matches the per-step
    # stacking the other two paths use.
    chunk_tensor = torch.from_numpy(raw_chunk[0]).unsqueeze(1).to(str(POLICY.config.device))
    action_full = _POSTPROCESSOR(chunk_tensor).squeeze(1).cpu().numpy().astype("float32")
    inference_time_s = time.perf_counter() - t0

    action = action_full[:action_horizon]
    LOGGER.info(
        "predict req=%d rl=on latent_std=%.3f q=%s chunk=%d returned=%d inference_s=%.3f",
        req_id, float(agent_action.latent.std()), agent_action.q_value,
        raw_chunk.shape[1], action.shape[0], inference_time_s,
    )

    # Hand the produced chunk back to a recording agent. This is the only point in the
    # system where the latent and the motion it caused are both in scope: the agent knows
    # the latent, the robot knows the motion, and nothing else sees the pair. Sent after
    # the timing measurement so a learner's disk write cannot be mistaken for inference
    # latency, and a no-op unless the agent's handshake asked for it.
    #
    # ``action_full`` rather than ``action``: the truncation to action_horizon is the
    # client's business and is recoverable from the horizon we send alongside, whereas the
    # rows past it are the part of the denoiser's output that no one else can reconstruct.
    actions_taken = getattr(RL_AGENT, "actions_taken", None)
    if actions_taken is not None:
        await actions_taken(ProducedActions(
            step_id=req_id,
            actions=action_full,
            raw_actions=raw_chunk[0],
            action_horizon=action_horizon,
        ))

    return action, inference_time_s, agent_action.q_value



def _prepare_policy_observation(
    frame: dict[str, Any], prompt: str, req_id: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Apply the normal policy input path without sampling an action."""
    device = str(POLICY.config.device)
    raw_state = frame[JOINT_STATE_KEY].astype("float32")
    state_25 = _apply_state_pins(_wire_state_25(raw_state), req_id)
    raw_state_hash = _hash_bytes(raw_state.tobytes())
    image_fingerprints = " ".join(
        f"{wire_key.rsplit('/', 1)[-1]}={_image_fingerprint(frame[wire_key])}"
        for wire_key in _CAMERA_KEY_MAP
    )
    state_25_str = np.array2string(state_25, precision=3, suppress_small=True, max_line_width=400)
    LOGGER.info(
        "predict req=%d task=%r raw_state_hash=%s state_25=%s images=[%s]",
        req_id, prompt, raw_state_hash, state_25_str, image_fingerprints,
    )

    obs: dict[str, Any] = {
        "observation.state": torch.from_numpy(state_25).unsqueeze(0).to(device),
    }
    for wire_key, lerobot_key in _CAMERA_KEY_MAP.items():
        obs[lerobot_key] = _decode_jpeg_to_tensor(frame[wire_key], device).unsqueeze(0)

    _check_image_resolutions(obs, req_id)
    if POLICY_TYPE in ("pi0", "pi0_fast", "pi05"):
        obs["task"] = [prompt]

    obs_processed = _PREPROCESSOR(obs)
    _check_normalized_state(obs_processed, req_id)
    return obs_processed, state_25


async def _inference_response(
    frame: dict[str, Any],
    rtc_state: _RtcState,
    rl_state: _RlState,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    prompt: str = "",
    ensemble_buffer: TemporalEnsembleBuffer | None = None,
) -> dict[str, Any]:
    device = str(POLICY.config.device)
    req_id = next(_REQUEST_COUNTER)
    obs_processed, state_25 = _prepare_policy_observation(frame, prompt, req_id)

    q_value: float | None = None
    using_rtc = RL_AGENT is None and (RTC_TRAIN_ENABLED or RTC_ENABLED)
    if RL_AGENT is not None:
        action, inference_time_s, q_value = await _predict_chunk_rl(
            obs_processed, frame, rl_state,
            state=state_25, prompt=prompt, action_horizon=action_horizon, req_id=req_id,
        )
    elif RTC_TRAIN_ENABLED:
        action, inference_time_s = _predict_chunk_rtc_train(
            obs_processed, frame, rtc_state,
            device=device, action_horizon=action_horizon, req_id=req_id,
        )
    elif RTC_ENABLED:
        action, inference_time_s = _predict_chunk_rtc(
            obs_processed, frame, rtc_state,
            device=device, action_horizon=action_horizon, req_id=req_id,
        )
    else:
        action, inference_time_s = _predict_chunk_sync(
            obs_processed, action_horizon=action_horizon, req_id=req_id,
        )

    if ensemble_buffer is not None:
        action = ensemble_buffer.add_and_ensemble(action)
        LOGGER.info("predict req=%d temporal_ensemble applied", req_id)

    if ACTION_LIMITS is not None:
        # Against the robot's actual commanded state, not the pinned one the policy saw:
        # the limit is about where the robot is, not about what the policy was told.
        current = _wire_state_25(frame[JOINT_STATE_KEY].astype("float32"))
        action, limiter_report = limit_chunk(
            action, current, ACTION_LIMITS,
            committed_prefix_len=min(rtc_state.committed_prefix_len, len(action)) if using_rtc else 0,
        )
        if limiter_report.triggered:
            LOGGER.warning(
                "predict req=%d ACTION LIMITER fired: %s", req_id, limiter_report.describe(_STATE_DIM_NAMES),
            )

    action_0_str = np.array2string(action[0], precision=3, suppress_small=True, max_line_width=400)
    action_last_str = np.array2string(action[-1], precision=3, suppress_small=True, max_line_width=400)
    delta_first = float(np.linalg.norm(action[0] - state_25))
    LOGGER.info(
        "predict req=%d action[0]=%s action[-1]=%s ||action[0]-state_25||=%.4f",
        req_id, action_0_str, action_last_str, delta_first,
    )

    resp = {
        ACTION_KEY: action,
        INFERENCE_TIME_KEY: inference_time_s,
        POLICY_ID_KEY: POLICY_ID,
        RL_ENABLED_KEY: RL_AGENT is not None,
        Q_VALUE_KEY: 0.0 if q_value is None else float(q_value),
    }
    # A cancelled client request can leave its reply buffered on this websocket.
    # Echo the caller's ID so its existing receive loop can discard that reply on
    # resume. Without an ID, RemotePolicyClient accepts it as the current prediction
    # and can remain one response behind indefinitely.
    if CHUNK_ID_KEY in frame:
        resp[CHUNK_ID_KEY] = frame[CHUNK_ID_KEY]
    LOGGER.info("predict req=%d response chunk_id=%r", req_id, frame.get(CHUNK_ID_KEY))
    validate_wire_inference_response(resp)
    if using_rtc:
        # Cache exactly the actions we return, in robot space. A raw model chunk can
        # differ after limiting and may contain rows never sent to the client; neither is
        # a valid fallback for the next request's committed prefix. The next request
        # converts it into policy space against its own state (_to_policy_action_space).
        rtc_state.last_chunk = torch.as_tensor(action, device=device).detach().clone()
    return resp


async def handle_inference_connection(
    connection: ServerConnection,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    server_features: Iterable[str | ServerFeature] = (),
    one_shot: bool = False,
    task: str | None = None,
    prompt_source: str = "auto",
    temporal_ensemble_coeff: float | None = None,
) -> None:

    assert action_horizon >= 1, f"action_horizon must be positive, got {action_horizon}"
    _check_prompt_source(prompt_source)
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))

    # Per-connection state: a new robot episode opens a fresh connection (or
    # sends ENDPOINT_DONE at the end of the prior one), and must not inherit
    # the previous chunk.
    rtc_state = _RtcState()
    rl_state = _RlState()
    last_prompt: str | None = None
    last_metadata: str | None = None
    held_prompt: str | None = None

    def resolve_prompt(frame: dict[str, Any]) -> tuple[str, str]:
        """This frame's instruction, plus a label saying where it came from."""
        nonlocal held_prompt
        if task is not None:
            return task, "--task"
        prompt, origin = _frame_prompt(frame, source=prompt_source)
        if _is_metadata_origin(origin) and prompt:
            held_prompt = prompt
            return prompt, origin
        if held_prompt is not None:
            # The metadata carrying the instruction need not repeat on every request --
            # start_metadata is named for the start of an episode, and a terminal frame
            # carries telemetry rather than a full request. Falling back to the wire
            # fields when it is absent would swap the policy's conditioning from the real
            # instruction to a construction name partway through an episode, which is
            # worse than either, so the last instruction seen on this connection stands
            # until a new one arrives or the episode ends.
            return held_prompt, "held from an earlier request"
        return prompt, origin
    ensemble_buffer = (
        TemporalEnsembleBuffer(action_horizon, DEFAULT_HARDWARE_MODEL.action_dim, temporal_ensemble_coeff)
        if temporal_ensemble_coeff is not None
        else None
    )

    async for message in connection:
        assert isinstance(message, bytes), type(message)
        frame = deserialize_from_msgpack(message)
        if not isinstance(frame, dict):
            await connection.send(serialize_to_msgpack({"error": "expected dict frame"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_DONE:
            if RL_AGENT is not None:
                # Close the last latent with a real post-action observation when the
                # robot supplies one. Prefix extraction is required for critic state,
                # but no new latent is requested and no action is denoised.
                reason = frame.get(DONE_REASON_KEY, "")
                executed_raw = frame.get(PREV_SKIPPED_ACTION_START_KEY)
                if executed_raw is not None:
                    assert rl_state.last_step_id is not None, (
                        "terminal telemetry has no preceding RL action"
                    )
                    assert JOINT_STATE_KEY in frame and all(
                        wire_key in frame for wire_key in _CAMERA_KEY_MAP
                    ), "terminal telemetry must include joint state and every policy camera"
                    executed_steps = int(executed_raw)
                    assert executed_steps >= 1, (
                        f"terminal executed steps must be positive, got {executed_steps}"
                    )
                    terminal_step_id = next(_REQUEST_COUNTER)
                    terminal_prompt, _ = resolve_prompt(frame)
                    terminal_obs_processed, terminal_state = _prepare_policy_observation(
                        frame, terminal_prompt, terminal_step_id
                    )
                    assert RL_BACKEND is not None
                    if RL_OBS_DUMP is not None:
                        RL_OBS_DUMP.submit(terminal_step_id, frame,
                                           task=terminal_prompt, terminal=True)
                    prefix_rep, _ = RL_BACKEND.prefix(terminal_obs_processed)
                    final_observation = AgentObservation(
                        images=_agent_images(frame, RL_RESIZE_IMAGE),
                        state=terminal_state.astype("float32"),
                        prefix_rep=prefix_rep[0],
                        task=terminal_prompt,
                    )
                    await RL_AGENT.episode_end(
                        str(reason) if reason else "",
                        final_step_id=terminal_step_id,
                        final_observation=final_observation,
                        executed_steps=executed_steps,
                    )
                else:
                    LOGGER.warning(
                        "episode ended without terminal observation telemetry; "
                        "the learner will synthesize the last transition"
                    )
                    await RL_AGENT.episode_end(str(reason) if reason else "")
                rl_state.reset()

            # Reset only after terminal preprocessing: stateful preprocessors must see
            # the post-action observation in the same episode as its preceding action.
            POLICY.reset()
            _PREPROCESSOR.reset()
            _POSTPROCESSOR.reset()
            rtc_state.reset()
            held_prompt = None
            if ensemble_buffer is not None:
                ensemble_buffer.reset()
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_INTERVENTION:
            # A human has taken over. Everything we remember about "what the robot is
            # executing right now" is void from here on: the chunk we last sent was
            # abandoned mid-flight, and by the time auto resumes the operator will have
            # moved the arms somewhere we have never seen. Carrying that chunk across the
            # intervention is what makes a resume lurch -- it comes back as an RTC prefix
            # (pinning the next chunk's opening steps to a pre-teleop trajectory) or as
            # leftovers in select_action's queue.
            #
            # rl_state is deliberately untouched: its last_step_id links a transition the
            # learner still has to close, and the intervention itself is what happened in
            # between.
            LOGGER.info("intervention frame: dropping the last chunk; the next request "
                        "starts a fresh, unconditioned plan")
            POLICY.reset()
            _PREPROCESSOR.reset()
            _POSTPROCESSOR.reset()
            rtc_state.reset()
            if ensemble_buffer is not None:
                ensemble_buffer.reset()
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_TELEMETRY:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if ENDPOINT_KEY in frame:
            # Endpoint we don't implement; ack so the client keeps going.
            LOGGER.info("ignoring unhandled endpoint %r keys=%s", frame[ENDPOINT_KEY], sorted(frame))
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue
        validate_wire_inference_request_frame(frame)
        _ = frame[MODEL_ID_KEY]
        prompt, prompt_origin = resolve_prompt(frame)
        # The metadata is logged whenever it changes rather than once per connection:
        # a task list advancing to its next entry is a mid-connection change, and it is
        # the only place an operator can see what the robot is really saying about the
        # run. Everything the client may legally send that is not pixels, joints or
        # per-request bookkeeping shows up here, which is what makes an instruction
        # buried in free-form metadata findable at all.
        summary = _metadata_summary(frame)
        if summary != last_metadata:
            LOGGER.info("request metadata %s", summary)
            last_metadata = summary
        if prompt != last_prompt:
            log = LOGGER.info if prompt else LOGGER.warning
            log("prompt=%r from %s", prompt, prompt_origin)
            last_prompt = prompt
        resp = await _inference_response(
            frame,
            rtc_state,
            rl_state,
            action_horizon=action_horizon,
            prompt=prompt,
            ensemble_buffer=ensemble_buffer,
        )
        await connection.send(serialize_to_msgpack(resp))
        if one_shot:
            LOGGER.info("one-shot mode: sent one inference response; ignoring further requests on this connection")
            await asyncio.Future()


@asynccontextmanager
async def run_example_server(
    host: str = "127.0.0.1",
    port: int = 0,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    *, server_features: Iterable[str | ServerFeature] = (ServerFeature.REWARDS,),
    one_shot: bool = False,
    task: str | None = None,
    prompt_source: str = "auto",
    temporal_ensemble_coeff: float | None = None,
) -> AsyncIterator[str]:

    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(
            connection,
            action_horizon=action_horizon,
            server_features=server_features,
            one_shot=one_shot,
            task=task,
            prompt_source=prompt_source,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )

    async with websockets.serve(handler, host, port) as server:
        sock = next(iter(server.sockets))
        port = sock.getsockname()[1]
        yield f"ws://{host}:{port}/ws"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the consolidated policy inference server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_INFERENCE_SERVER_PORT, help="Bind port")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the pretrained policy checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--policy-type",
        choices=("act", "pi0", "pi0_fast", "pi05"),
        default=DEFAULT_POLICY_TYPE,
        help=f"Policy architecture to load (default: {DEFAULT_POLICY_TYPE}).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_REPLAY_ACTION_HORIZON,
        help="Number of action rows to emit per prediction. Defaults to 50 to match replay_rrd defaults.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        metavar="N",
        help="Override the checkpoint's n_action_steps (the depth of select_action's "
             "queue). Must be <= chunk_size, which is fixed by the weights. Pass "
             "--n-action-steps equal to --action-horizon so each request is exactly one "
             "denoise pass. Default: use the value stored in the checkpoint.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task prompt passed to the policy. If omitted, each request's own "
             "task/prompt from the robot's configuration is used.",
    )
    parser.add_argument(
        "--prompt-source",
        type=_check_prompt_source,
        default="auto",
        metavar="SOURCE",
        help="Where in each request to read the instruction from when --task is not "
             "given. 'auto' (default): a conventionally-named field "
             f"({'/'.join(_PROMPT_FIELD_NAMES)}) inside conditioning_metadata or "
             "start_metadata, else the subtask, else the task. 'subtask' / 'task': that "
             "wire field verbatim -- note that both hold the *name* of the UI "
             "construction ('simple_list', 'generic_subtask') when the run is driven by "
             "one, not an instruction. Anything else is read as a dotted path into the "
             "frame, e.g. 'conditioning_metadata.instruction'. Every request logs its "
             "metadata when it changes, so the right path can be read off the log.",
    )
    parser.add_argument(
        "--no-rewards",
        action="store_true",
        help="Do not advertise reward support in the handshake.",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Send one full action chunk per connection, then stop responding to further requests.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        metavar="M",
        help=(
            "Enable ACT temporal ensembling with exponential decay coefficient M. "
            "At each request the new chunk is blended with the last action_horizon chunks "
            "using weight exp(-M * age). Typical values: 0.01–0.1. "
            "Disabled by default (None = no ensembling)."
        ),
    )
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Enable inference-time Real-Time Chunking: predict_action_chunk with prefix-"
             "guided inpainting (pi0/pi05 only). Default is the synchronous select_action loop.",
    )
    parser.add_argument(
        "--rtc-train",
        action="store_true",
        help="Enable train-time RTC (action-prefix conditioning, arXiv 2512.05964): "
             "predict_action_chunk conditioned on a hard action prefix, one plain forward "
             "pass, no inference-time guidance. Requires a checkpoint fine-tuned with "
             "rtc_train_enabled=true (pi0/pi05 only). Mutually exclusive with --rtc.",
    )
    parser.add_argument(
        "--rollout-hz",
        type=float,
        default=DEFAULT_ROLLOUT_HZ,
        help="Client control-loop rate, used only to convert observed request-arrival "
             "gaps into control steps so the log can report the inference delay the "
             f"client should be sending (default: {DEFAULT_ROLLOUT_HZ:g}).",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=DEFAULT_EXECUTION_HORIZON,
        help="RTC execution horizon — the client's 'chunk execution limit'. Controls the "
             f"prefix-weight falloff (default: {DEFAULT_EXECUTION_HORIZON}).",
    )
    parser.add_argument(
        "--rtc-schedule",
        choices=tuple(s.value for s in RTCAttentionSchedule),
        default=DEFAULT_RTC_SCHEDULE,
        help=f"RTC prefix attention schedule (default: {DEFAULT_RTC_SCHEDULE}).",
    )
    parser.add_argument(
        "--max-guidance-weight",
        type=float,
        default=DEFAULT_MAX_GUIDANCE_WEIGHT,
        help=f"RTC max guidance weight (default: {DEFAULT_MAX_GUIDANCE_WEIGHT}).",
    )
    parser.add_argument(
        "--resume-gap-s",
        type=float,
        default=DEFAULT_RESUME_GAP_S,
        metavar="S",
        help=f"Treat a gap of this many seconds between requests as a pause: the next "
             f"request ignores its action_prefix and plans a fresh chunk from the live "
             f"observation. An async rollout's normal gap is one round trip, so this only "
             f"fires when the client stopped asking (default: {DEFAULT_RESUME_GAP_S:g}; "
             f"0 never infers a resume from timing).",
    )
    parser.add_argument(
        "--max-prefix-gap-deg",
        type=float,
        default=DEFAULT_MAX_PREFIX_GAP_DEG,
        metavar="DEG",
        help=f"Drop a wire action_prefix whose first row is further than this, on any joint, "
             f"from the state sent with it. OFF by default because an arm at speed runs "
             f"legitimately far behind its own command -- 14 deg between requests at 50 Hz "
             f"in the field logs -- so any tight value fires mid-motion and keeps firing. "
             f"Use only against a client that pauses without a gap in its request stream, "
             f"and set it well above the tracking error at speed "
             f"(default: {DEFAULT_MAX_PREFIX_GAP_DEG:g} = off).",
    )
    parser.add_argument(
        "--pin-state-dims",
        nargs="+",
        default=(),
        metavar="DIM",
        help="Overwrite these state dims with their training mean before normalization, by "
             "name (left_gripper, chest[5], ...) or index. Stopgap for dims the demonstrator "
             "held constant: their near-zero training std makes MEAN_STD normalization blow "
             "up small physical drift into a large out-of-distribution state token, which "
             "every action token attends to. Off by default. Example: "
             "--pin-state-dims left_gripper",
    )
    parser.add_argument(
        "--no-action-limiter",
        action="store_true",
        help="Disable the per-chunk joint limits (on by default). With them on, every chunk "
             "is pulled back so its first step is within --max-chunk-start-offset-deg of the "
             "current commanded pose on every joint, and no joint moves more than "
             "--max-step-deg between consecutive steps. Grippers are never limited. The "
             "defaults sit just above the base policy's own maxima, so they only fire on a "
             "chunk the policy would not normally produce; each firing is logged.",
    )
    parser.add_argument(
        "--max-chunk-start-offset-deg",
        type=float,
        default=DEFAULT_MAX_START_OFFSET_DEG,
        metavar="DEG",
        help=f"Largest allowed distance, per joint, between a chunk's first step and the "
             f"current commanded pose; the whole chunk is translated by any excess "
             f"(default: {DEFAULT_MAX_START_OFFSET_DEG:g}; 0 disables this limit).",
    )
    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=DEFAULT_MAX_STEP_DEG,
        metavar="DEG",
        help=f"Largest allowed motion of any joint between consecutive chunk steps, from the "
             f"second step on (default: {DEFAULT_MAX_STEP_DEG:g}; 0 disables this limit).",
    )
    parser.add_argument(
        "--limit-dims",
        nargs="*",
        default=[],
        metavar="DIM",
        help="Which action dims the limits apply to, by name (left_arm[3], chest[0], ...) or "
             "index. Default: every joint dim, no grippers.",
    )
    parser.add_argument(
        "--rl-agent-url",
        default=None,
        metavar="URL",
        help="Enable latent-space RL by delegating the denoiser's starting noise to an RL "
             f"agent at this WebSocket URL (e.g. ws://127.0.0.1:{DEFAULT_AGENT_PORT}). The agent "
             "also receives the resulting transitions. pi0/pi05 only. Off by default: without "
             "it the policy samples its own noise exactly as before.",
    )
    parser.add_argument(
        "--rl-obs-dump-dir",
        default=None,
        metavar="DIR",
        help="Write the wire's own camera JPEGs for every RL query to this exact "
             "directory, with a steps.jsonl index keyed by step id. This is what lets an "
             "offline sweep re-run pi0 on the exact observation the robot was steered "
             "from -- the learner only sees the 128px critic view, which perturbs the "
             "action chunk several times more than the latent itself does. Bytes are "
             "copied verbatim (no decode, no resize, no re-encode) and written on a "
             f"background thread. Default: a fresh {DEFAULT_RL_OBS_DUMP_ROOT}/<timestamp> "
             "per run, whenever --rl-agent-url is given.",
    )
    parser.add_argument(
        "--no-rl-obs-dump",
        action="store_true",
        help="Do not write the RL observation dump. Without it an offline sweep cannot "
             "resample latents on this run at all, so only pass it when disk is the "
             "problem (measured: ~57 kB per query, ~3 MB per 50-query episode, ~150 MB "
             "for an 83-episode run).",
    )
    parser.add_argument(
        "--rl-resize-image",
        type=int,
        default=DEFAULT_RL_RESIZE_IMAGE,
        help=f"Edge length of the square camera images sent to the RL agent "
             f"(default: {DEFAULT_RL_RESIZE_IMAGE}).",
    )
    return parser.parse_args(argv)


def _obs_dump_dir_from_args(args: argparse.Namespace, *, now: float | None = None) -> str | None:
    """Where the RL observation dump goes, or None when it is off.

    On by default because the alternative is discovering after the run that the only
    images kept are the learner's 128px critic view, which cannot be re-run through pi0
    -- and a robot session is not repeatable. An explicit --rl-obs-dump-dir is used
    verbatim so a caller can still put one run's images exactly where it wants them.
    """
    if args.no_rl_obs_dump or args.rl_obs_dump_dir == "":
        return None
    if args.rl_obs_dump_dir is not None:
        return args.rl_obs_dump_dir
    if args.rl_agent_url is None:
        return None  # No RL, no RL queries to dump.
    stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(now))
    return str(DEFAULT_RL_OBS_DUMP_ROOT.expanduser() / stamp)


def _action_limits_from_args(args: argparse.Namespace) -> ActionLimits | None:
    """None when disabled or when both limits are 0; otherwise the limits in radians."""
    if args.no_action_limiter:
        return None
    if args.max_chunk_start_offset_deg <= 0 and args.max_step_deg <= 0:
        return None
    return limits_from_degrees(
        max_start_offset_deg=args.max_chunk_start_offset_deg,
        max_step_deg=args.max_step_deg,
        dims=resolve_limit_dims(args.limit_dims),
    )


def _cli_server_features(no_rewards: bool) -> tuple[ServerFeature, ...]:
    if no_rewards:
        return ()
    return (ServerFeature.REWARDS,)


def _warn_on_stale_refill(args: argparse.Namespace) -> None:
    """Flag the case where the sync path drains select_action past its queue depth.

    Only the synchronous path uses the queue; the RTC and RL paths call
    predict_action_chunk directly and slice, so they are unaffected."""
    if args.rtc or args.rtc_train or args.rl_agent_url is not None:
        return
    n_action_steps = getattr(POLICY.config, "n_action_steps", None)
    if n_action_steps is None or args.action_horizon <= n_action_steps:
        return
    passes = -(-args.action_horizon // n_action_steps)  # ceil
    LOGGER.warning(
        "action_horizon=%d > n_action_steps=%d: each request runs %d denoise passes, and "
        "passes after the first re-infer on the same (stale) observation, seaming the "
        "chunk every %d steps. Pass --n-action-steps %d to make it one pass.",
        args.action_horizon, n_action_steps, passes, n_action_steps, args.action_horizon,
    )


async def _run_cli(argv: Sequence[str] | None = None) -> int:
    global ROLLOUT_HZ, ACTION_LIMITS, MAX_PREFIX_GAP, RESUME_GAP_S

    args = _parse_args(argv)
    assert args.action_horizon >= 1, f"action_horizon must be positive, got {args.action_horizon}"
    assert args.execution_horizon >= 1, f"execution_horizon must be positive, got {args.execution_horizon}"
    assert not (args.rtc and args.temporal_ensemble_coeff is not None), (
        "--rtc and --temporal-ensemble-coeff are mutually exclusive: both are chunk-consistency "
        "mechanisms and ensembling RTC-guided chunks would fight the prefix guidance"
    )
    assert not (args.rtc and args.rtc_train), (
        "--rtc (inference-time RTC) and --rtc-train (train-time RTC) are mutually exclusive: "
        "they are two different implementations of the same idea. Pick one to A/B against sync."
    )
    assert not (args.rtc_train and args.temporal_ensemble_coeff is not None), (
        "--rtc-train and --temporal-ensemble-coeff are mutually exclusive: ensembling would "
        "blend the prefix-conditioned chunk with stale ones and reintroduce the discontinuity "
        "the prefix conditioning exists to remove"
    )
    if args.rl_agent_url is not None:
        # Both of these rewrite the chunk after denoising, so the executed actions would
        # no longer be the ones the agent's latent produced and the critic would be
        # learning against a mapping that does not exist at execution time.
        assert not args.rtc, (
            "--rl-agent-url and --rtc are mutually exclusive: RTC inpaints the previous chunk "
            "into the denoise loop, breaking the latent -> action mapping the RL critic learns"
        )
        assert not args.rtc_train, (
            "--rl-agent-url and --rtc-train are mutually exclusive: the hard action prefix "
            "overrides the first steps of the chunk, so the executed actions are no longer the "
            "ones the agent's latent produced and the critic learns a mapping that does not exist"
        )
        assert args.temporal_ensemble_coeff is None, (
            "--rl-agent-url and --temporal-ensemble-coeff are mutually exclusive: ensembling "
            "blends chunks from different latents, so no single latent explains the executed actions"
        )
        assert args.policy_type in ("pi0", "pi05"), (
            f"--rl-agent-url requires a flow-matching policy whose sampler starts from noise; "
            f"got --policy-type {args.policy_type}"
        )
    server_features = _cli_server_features(args.no_rewards)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    assert args.rollout_hz > 0, f"rollout_hz must be positive, got {args.rollout_hz}"
    ROLLOUT_HZ = float(args.rollout_hz)
    RESUME_GAP_S = float(args.resume_gap_s) if args.resume_gap_s > 0 else None
    if RESUME_GAP_S is None:
        LOGGER.warning(
            "RESUME DETECTION OFF: a prefix sent after a pause will be conditioned on as "
            "though the robot had been executing it the whole time"
        )
    else:
        LOGGER.info(
            "resume detection on: a gap of %.1fs or more between requests drops that "
            "request's action_prefix and plans a fresh chunk", RESUME_GAP_S,
        )
    MAX_PREFIX_GAP = (
        float(np.deg2rad(args.max_prefix_gap_deg)) if args.max_prefix_gap_deg > 0 else None
    )
    if MAX_PREFIX_GAP is not None:
        LOGGER.warning(
            "position-based prefix check on at %.1f deg: this fires on ordinary fast "
            "motion and is self-reinforcing when it does. Watch for a run of "
            "'action_prefix dropped' lines with no pause in the log.",
            args.max_prefix_gap_deg,
        )
    ACTION_LIMITS = _action_limits_from_args(args)
    if ACTION_LIMITS is None:
        LOGGER.warning("ACTION LIMITER OFF: chunks go to the robot exactly as the policy emits them")
    else:
        LOGGER.info(
            "action limiter on: chunk start within %s deg of the current pose, at most %s deg per "
            "step, on %d dims (%s)",
            "-" if ACTION_LIMITS.max_start_offset is None else f"{np.rad2deg(ACTION_LIMITS.max_start_offset):g}",
            "-" if ACTION_LIMITS.max_step is None else f"{np.rad2deg(ACTION_LIMITS.max_step):g}",
            len(ACTION_LIMITS.dims), ", ".join(_STATE_DIM_NAMES[i] for i in ACTION_LIMITS.dims),
        )
    _init_policy(
        args.checkpoint_path,
        args.policy_type,
        rtc_enabled=args.rtc,
        rtc_train_enabled=args.rtc_train,
        execution_horizon=args.execution_horizon,
        rtc_schedule=args.rtc_schedule,
        max_guidance_weight=args.max_guidance_weight,
        pin_state_dims=args.pin_state_dims,
        n_action_steps=args.n_action_steps,
    )
    _warn_on_stale_refill(args)
    obs_dump_dir = _obs_dump_dir_from_args(args)
    if args.rl_agent_url is not None and obs_dump_dir is None:
        LOGGER.warning(
            "RL OBSERVATION DUMP OFF: this run keeps only the learner's 128px critic view, "
            "which is too coarse to re-run the policy on -- no offline latent resampling"
        )
    await _init_rl(args.rl_agent_url, resize_image=args.rl_resize_image,
                   obs_dump_dir=obs_dump_dir)
    if obs_dump_dir and RL_OBS_DUMP is not None:
        LOGGER.info(
            "when this run is over, view it with:\n"
            "  python -m viewer.server --run r=~/exp/<learner run> --obs-dump r=%s "
            "--checkpoint %s",
            obs_dump_dir, args.checkpoint_path,
        )
    async with run_example_server(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        server_features=server_features,
        one_shot=args.one_shot,
        task=args.task,
        prompt_source=args.prompt_source,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    ) as url:
        if args.task is not None:
            print(f"Task prompt: {args.task!r} (from --task)", flush=True)
        else:
            print(
                "Task prompt: from each request "
                f"(--prompt-source {args.prompt_source})",
                flush=True,
            )
        mode_suffix = f" rtc={'on' if RTC_ENABLED else 'off'}"
        if RTC_TRAIN_ENABLED:
            mode_suffix += " rtc_train=on"
        if RL_AGENT is not None:
            mode_suffix += f" rl={args.rl_agent_url}"
        if args.one_shot:
            mode_suffix += " one_shot=True"
        if args.temporal_ensemble_coeff is not None:
            mode_suffix += f" temporal_ensemble_coeff={args.temporal_ensemble_coeff}"
        print(f"Server listening on {url} (action_horizon={args.action_horizon}){mode_suffix}", flush=True)
        if args.host == "0.0.0.0":
            lan_ips = _list_lan_ips()
            if lan_ips:
                for ip in lan_ips:
                    print(f"  reachable at ws://{ip}:{args.port}/ws", flush=True)
            else:
                print("  (could not detect LAN IP; check `ip addr`)", flush=True)
        LOGGER.info(
            "Server listening on %s action_horizon=%d rtc=%s", url, args.action_horizon, RTC_ENABLED
        )
        await asyncio.Future()
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        try:
            raise SystemExit(asyncio.run(_run_cli(argv)))
        except KeyboardInterrupt:
            raise SystemExit(130)
    finally:
        # These runs end with Ctrl-C, and the dump's writer is a daemon thread: without
        # this, whatever is still queued when the interrupt lands is lost -- the tail of
        # the last episode, which is the part an offline sweep most wants. Also logs the
        # written/dropped counts, the only place a dropped frame is ever reported.
        if RL_OBS_DUMP is not None:
            RL_OBS_DUMP.close()


__all__ = [
    "handle_inference_connection",
    "main",
    "run_example_server",
    "server_handshake_config",
]


if __name__ == "__main__":
    main()
