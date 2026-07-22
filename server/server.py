# START SERVER:
#   python -m server.server --checkpoint-path models/<ckpt> --host 0.0.0.0 --port 18090
#
# Consolidated policy inference server (merges old_minimal.py, old_minimal_debug.py,
# rtc_server.py). Three inference paths:
#   * default: synchronous select_action loop (works for act / pi0 / pi0_fast / pi05)
#   * --rtc:   Real-Time Chunking via predict_action_chunk with prefix-guided
#              inpainting (pi0 / pi05 flow-matching policies only)
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
#     the policy's *normalized output action space*. Used as ``prev_chunk_left_over``.
#     If omitted, the server falls back to its own memory of the previous normalized
#     chunk and uses ``last_chunk[prefix_change_start:]`` (mirrors lerobot's
#     ``ActionQueue._replace_actions_queue``).

# python -m server.server --checkpoint-path ... --host 0.0.0.0 --port 18090

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
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
    DEFAULT_INFERENCE_SERVER_PORT,
    DONE_KEY,
    DONE_REASON_KEY,
    ENDPOINT_DONE,
    ENDPOINT_KEY,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PREFIX_CHANGE_START_KEY,
    PREV_SKIPPED_ACTION_START_KEY,
    Q_VALUE_KEY,
    REWARD_KEY,
    RL_ENABLED_KEY,
    STATUS_KEY,
    SUBTASK_KEY,
    TASK_KEY,
    ServerFeature,
    ServerHandshake,
)
from policy_inference_spec.rl import (
    DEFAULT_AGENT_PORT,
    AgentObservation,
    NullAgent,
    RemoteAgent,
    RLAgent,
    Transition,
)

import hashlib
import time
from itertools import count
import simplejpeg
from pathlib import Path
from lerobot.policies.factory import make_pre_post_processors
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


def _list_lan_ips() -> list[str]:
    """Best-effort enumeration of non-loopback IPv4 addresses on this machine."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                addrs.add(addr)
    except socket.gaierror:
        pass
    # UDP route-selection trick: no packet is sent, but getsockname returns the
    # IP the OS would use to reach the target. Catches the primary NIC even when
    # gethostname() resolves oddly (e.g. 127.0.1.1 on Debian).
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        if not primary.startswith("127."):
            addrs.add(primary)
    except OSError:
        pass
    finally:
        s.close()
    return sorted(addrs)


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
DEFAULT_RTC_SCHEDULE = "EXP"
DEFAULT_MAX_GUIDANCE_WEIGHT = 10.0

# Edge length of the square images handed to the RL agent. Small on purpose: the agent's
# own conv encoder is tiny compared to the policy's, and the scene detail it cannot
# resolve at this size is already available to it through the policy's prefix embedding.
DEFAULT_RL_RESIZE_IMAGE = 128

# Populated by _init_policy() once the checkpoint path / policy type are known.
POLICY: Any = None
POLICY_TYPE: str | None = None
POLICY_ID: str | None = None
_DATASET_STATS: dict | None = None
_PREPROCESSOR: Any = None
_POSTPROCESSOR: Any = None
# RTC runtime config, populated by _init_policy(). execution_horizon is read per
# request so it can stay in sync with the client's chunk execution limit.
RTC_ENABLED: bool = False
EXECUTION_HORIZON: int = DEFAULT_EXECUTION_HORIZON
# Latent-space RL runtime, populated by _init_rl(). RL_AGENT defaults to None (plain
# inference); with an agent attached the server asks it for the denoiser's starting
# latent instead of sampling one, and reports transitions back for learning.
RL_BACKEND: Any = None
RL_AGENT: RLAgent | None = None
RL_RESIZE_IMAGE: int = DEFAULT_RL_RESIZE_IMAGE


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
    """Per-connection memory of the last normalized action chunk.

    RTC guidance compares the committed prefix against the model's *normalized*
    output space, so we remember the raw ``predict_action_chunk`` output (before
    postprocessing) to use as the next request's guidance target when the client
    does not send an explicit ``action_prefix``.
    """

    __slots__ = ("last_chunk",)

    def __init__(self) -> None:
        self.last_chunk: torch.Tensor | None = None  # (T, action_dim) on device

    def reset(self) -> None:
        self.last_chunk = None


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


def _load_policy(checkpoint_path: Path, policy_type: str):
    if policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        config = _load_config_tolerant(checkpoint_path, ACTConfig)
        return ACTPolicy.from_pretrained(str(checkpoint_path), config=config)
    elif policy_type in ("pi0", "pi0_fast"):
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        config = _load_config_tolerant(checkpoint_path, PI0Config)
        return PI0Policy.from_pretrained(str(checkpoint_path), config=config)
    elif policy_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        config = _load_config_tolerant(checkpoint_path, PI05Config)
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


def _init_policy(
    checkpoint_path: Path,
    policy_type: str,
    *,
    rtc_enabled: bool = False,
    execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
    rtc_schedule: str = DEFAULT_RTC_SCHEDULE,
    max_guidance_weight: float = DEFAULT_MAX_GUIDANCE_WEIGHT,
) -> None:
    """Load the policy + processors for the given checkpoint and populate module globals."""
    global POLICY, POLICY_TYPE, POLICY_ID, _DATASET_STATS, _PREPROCESSOR, _POSTPROCESSOR
    global RTC_ENABLED, EXECUTION_HORIZON
    POLICY_TYPE = policy_type
    POLICY_ID = str(checkpoint_path)
    EXECUTION_HORIZON = execution_horizon
    print("Loading policy...", flush=True)
    POLICY = _load_policy(checkpoint_path, policy_type).eval().cuda()
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
    RTC_ENABLED = bool(getattr(POLICY, "_rtc_enabled", lambda: False)())
    print("Loading dataset stats...", flush=True)
    _DATASET_STATS = _load_dataset_stats(checkpoint_path)
    print("Dataset stats loaded.", flush=True)
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
    print("Init complete.", flush=True)


async def _init_rl(agent_url: str | None, *, resize_image: int = DEFAULT_RL_RESIZE_IMAGE) -> None:
    """Attach a latent-space RL agent. Call after _init_policy().

    With ``agent_url=None`` the server keeps behaving exactly as it does without RL: the
    policy samples its own noise and no RL code runs on the request path.
    """
    global RL_BACKEND, RL_AGENT, RL_RESIZE_IMAGE
    if agent_url is None:
        RL_BACKEND, RL_AGENT = None, None
        return

    from server.pi0_latent import FlowLatentBackend

    assert POLICY is not None, "_init_rl() must be called after _init_policy()"
    assert resize_image > 0, f"resize_image must be positive, got {resize_image}"
    RL_RESIZE_IMAGE = resize_image
    RL_BACKEND = FlowLatentBackend(POLICY, str(POLICY_TYPE))

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
    images: dict[str, np.ndarray] = {}
    for wire_key in _CAMERA_KEY_MAP:
        decoded = simplejpeg.decode_jpeg(frame[wire_key], colorspace="RGB")  # uint8 HWC
        tensor = torch.from_numpy(decoded).permute(2, 0, 1).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        name = wire_key.rsplit("/", 1)[-1]
        images[name] = resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    return images


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


def _resolve_rtc_kwargs(
    frame: dict[str, Any],
    rtc_state: _RtcState,
    *,
    device: str,
    execution_horizon: int,
) -> tuple[dict[str, Any], str]:
    """Translate wire prefix fields + server state into RTC kwargs for
    ``predict_action_chunk``.

    Returns ``(kwargs, debug_str)``. ``kwargs`` is empty when there is no usable
    prefix (e.g. the first request of an episode), in which case the model runs
    an unguided chunk.
    """
    # inference_delay (a.k.a. freeze length): how many steps at the start of the
    # new chunk are already committed and will execute before this chunk is ready.
    raw_delay = frame.get(PREFIX_CHANGE_START_KEY)
    inference_delay = int(raw_delay) if raw_delay is not None else 0
    if inference_delay < 0:
        inference_delay = 0

    wire_prefix = frame.get(ACTION_PREFIX_KEY)
    if wire_prefix is not None:
        # Client-supplied leftover, already aligned to the new chunk's first
        # step and in the policy's normalized output action space.
        prev = torch.as_tensor(np.asarray(wire_prefix, dtype="float32"), device=device)
        source = "wire"
    elif rtc_state.last_chunk is not None:
        # Fall back to our memory of the previous normalized chunk, dropping the
        # steps already consumed during inference (mirrors lerobot's
        # ActionQueue._replace_actions_queue: original_actions[real_delay:]).
        t_prev = rtc_state.last_chunk.shape[0]
        prev = rtc_state.last_chunk[min(inference_delay, t_prev):]
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
        frame, rtc_state, device=device, execution_horizon=EXECUTION_HORIZON,
    )
    t0 = time.perf_counter()
    # predict_action_chunk runs one full denoise pass and returns the entire
    # normalized chunk (1, chunk_size, action_dim). RTC guidance, when active, is
    # applied inside the denoise loop via the kwargs above. The method manages
    # its own grad context, so we don't wrap it in torch.no_grad().
    raw_chunk = POLICY.predict_action_chunk(obs_processed, **rtc_kwargs)  # (1, T, A)
    raw_chunk = raw_chunk.squeeze(0)  # (T, A), normalized output space
    # Remember the normalized chunk for the next request's RTC guidance target.
    rtc_state.last_chunk = raw_chunk.detach()
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
    return action, inference_time_s, agent_action.q_value


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

    raw_state = frame[JOINT_STATE_KEY].astype("float32")
    # Extract commanded positions from 97-dim wire state:
    # left_arm[0:8], right_arm[32:40], chest[64:70], neck[88:91]
    state_25 = np.concatenate([
        raw_state[0:8],
        raw_state[32:40],
        raw_state[64:70],
        raw_state[88:91],
    ])
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

    if POLICY_TYPE in ("pi0", "pi0_fast", "pi05"):
        obs["task"] = [prompt]

    obs_processed = _PREPROCESSOR(obs)

    q_value: float | None = None
    if RL_AGENT is not None:
        action, inference_time_s, q_value = await _predict_chunk_rl(
            obs_processed, frame, rl_state,
            state=state_25, prompt=prompt, action_horizon=action_horizon, req_id=req_id,
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
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(
    connection: ServerConnection,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    server_features: Iterable[str | ServerFeature] = (),
    one_shot: bool = False,
    task: str | None = None,
    temporal_ensemble_coeff: float | None = None,
) -> None:

    assert action_horizon >= 1, f"action_horizon must be positive, got {action_horizon}"
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))

    # Per-connection state: a new robot episode opens a fresh connection (or
    # sends ENDPOINT_DONE at the end of the prior one), and must not inherit
    # the previous chunk.
    rtc_state = _RtcState()
    rl_state = _RlState()
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
            # Episode finished on the robot: reset per-connection policy state so a
            # subsequent episode does not inherit the previous chunk.
            POLICY.reset()
            _PREPROCESSOR.reset()
            _POSTPROCESSOR.reset()
            rtc_state.reset()
            if ensemble_buffer is not None:
                ensemble_buffer.reset()
            if RL_AGENT is not None:
                # The agent labels and learns from the finished episode here. This can
                # block for a long time (operator input, gradient steps) and that is the
                # intended back-pressure: the robot cannot start the next episode until
                # the learner has caught up.
                reason = frame.get(DONE_REASON_KEY, "")
                await RL_AGENT.episode_end(str(reason) if reason else "")
                rl_state.reset()
            await connection.send(serialize_to_msgpack({"status": "ok"}))
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
        _ = frame[SUBTASK_KEY]
        prompt = task if task is not None else frame[TASK_KEY]
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
    temporal_ensemble_coeff: float | None = None,
) -> AsyncIterator[str]:

    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(
            connection,
            action_horizon=action_horizon,
            server_features=server_features,
            one_shot=one_shot,
            task=task,
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
        "--task",
        default=None,
        help="Task prompt passed to the policy. If omitted, each request's own "
             "task/prompt from the robot's configuration is used.",
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
        help="Enable Real-Time Chunking: predict_action_chunk with prefix-guided "
             "inpainting (pi0/pi05 only). Default is the synchronous select_action loop.",
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
        "--rl-agent-url",
        default=None,
        metavar="URL",
        help="Enable latent-space RL by delegating the denoiser's starting noise to an RL "
             f"agent at this WebSocket URL (e.g. ws://127.0.0.1:{DEFAULT_AGENT_PORT}). The agent "
             "also receives the resulting transitions. pi0/pi05 only. Off by default: without "
             "it the policy samples its own noise exactly as before.",
    )
    parser.add_argument(
        "--rl-resize-image",
        type=int,
        default=DEFAULT_RL_RESIZE_IMAGE,
        help=f"Edge length of the square camera images sent to the RL agent "
             f"(default: {DEFAULT_RL_RESIZE_IMAGE}).",
    )
    return parser.parse_args(argv)


def _cli_server_features(no_rewards: bool) -> tuple[ServerFeature, ...]:
    if no_rewards:
        return ()
    return (ServerFeature.REWARDS,)


async def _run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    assert args.action_horizon >= 1, f"action_horizon must be positive, got {args.action_horizon}"
    assert args.execution_horizon >= 1, f"execution_horizon must be positive, got {args.execution_horizon}"
    assert not (args.rtc and args.temporal_ensemble_coeff is not None), (
        "--rtc and --temporal-ensemble-coeff are mutually exclusive: both are chunk-consistency "
        "mechanisms and ensembling RTC-guided chunks would fight the prefix guidance"
    )
    if args.rl_agent_url is not None:
        # Both of these rewrite the chunk after denoising, so the executed actions would
        # no longer be the ones the agent's latent produced and the critic would be
        # learning against a mapping that does not exist at execution time.
        assert not args.rtc, (
            "--rl-agent-url and --rtc are mutually exclusive: RTC inpaints the previous chunk "
            "into the denoise loop, breaking the latent -> action mapping the RL critic learns"
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
    _init_policy(
        args.checkpoint_path,
        args.policy_type,
        rtc_enabled=args.rtc,
        execution_horizon=args.execution_horizon,
        rtc_schedule=args.rtc_schedule,
        max_guidance_weight=args.max_guidance_weight,
    )
    await _init_rl(args.rl_agent_url, resize_image=args.rl_resize_image)
    async with run_example_server(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        server_features=server_features,
        one_shot=args.one_shot,
        task=args.task,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    ) as url:
        if args.task is not None:
            print(f"Task prompt: {args.task!r} (from --task)", flush=True)
        else:
            print("Task prompt: from each request (robot's configured task)", flush=True)
        mode_suffix = f" rtc={'on' if RTC_ENABLED else 'off'}"
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
        raise SystemExit(asyncio.run(_run_cli(argv)))
    except KeyboardInterrupt:
        raise SystemExit(130)


__all__ = [
    "handle_inference_connection",
    "main",
    "run_example_server",
    "server_handshake_config",
]


if __name__ == "__main__":
    main()
