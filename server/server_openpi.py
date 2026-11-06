# START SERVER (from the repo root, in the lair-openpi venv -- see README "openpi server"):
#   PYTHONPATH=. ~/projects/lair-openpi/.venv/bin/python -m server.server_openpi \
#       --checkpoint-path models/openpi-ultra-cubestack-pi0-rtctrain-10000 --rtc-train \
#       --host 0.0.0.0 --port 18090
#
# WITH DSRL (learner first, from dsrl_pi0: python -m examples.train_agent_server ...):
#   PYTHONPATH=. ~/projects/lair-openpi/.venv/bin/python -m server.server_openpi \
#       --checkpoint-path models/openpi-ultra-cubestack-pi0-rtctrain-10000 \
#       --rl-agent-url ws://127.0.0.1:18091 --host 0.0.0.0 --port 18090
#   (no --rtc-train: a pinned prefix would override the rows the latent produced)
#
# REPLAY A RECORDING ON THE ROBOT (no model is loaded; see server/trajectory_replay.py):
#   PYTHONPATH=. ~/projects/lair-openpi/.venv/bin/python -m server.server_openpi \
#       --replay cubestack-recording.rrd --host 0.0.0.0 --port 18090
#
# The same wire protocol as server.server, with the lerobot forward pass replaced by
# openpi's (JAX), for checkpoints trained by lair-train-pipeline on the lair-openpi fork
# (https://github.com/lair-research-nyu/lair-openpi). The robot client is unchanged.
#
# Why a separate process and venv: openpi pins numpy<2 on Python 3.11 and an old lerobot,
# while server.server needs lerobot 0.6 on numpy 2. The two cannot share an environment,
# so this module imports nothing from server.server; the backend-free pieces both servers
# need live in server.prompts and server.wire.
#
# What carries over from server.server, with the same flags and semantics:
#   * handshake, request validation, chunk_id echo, done / intervention / telemetry frames
#   * prompt resolution (--task, --prompt-source)
#   * --rtc-train: train-time RTC. The committed prefix (wire ``action_prefix`` or, failing
#     that, our own memory of the previous chunk) is pinned at flow time 0 while the rest
#     of the chunk is denoised. Resume detection (--resume-gap-s, first request of a
#     connection, intervention frames) drops the prefix exactly as server.server does.
#   * the action limiter (--max-chunk-start-offset-deg / --max-step-deg / --limit-dims)
#   * --rl-agent-url: latent-space RL (DSRL). The agent picks the flow-matching starting
#     noise from the policy's prefix embedding and learns from the transitions; same
#     learner protocol, same agent images, same observation dump. See
#     server/openpi_latent.py. Mutually exclusive with --rtc-train, as in server.server.
#   * --kv-ema-alpha (openpi only, experimental): an exponential moving average of the
#     prefix KV cache over the requests of one connection; see server/openpi_kv_ema.py.
#     It resets with the same resume rules as the RTC prefix. Not with --rl-agent-url.
# Not carried over: inference-time RTC guidance (--rtc), temporal ensembling, state pins.
# None of them has an openpi implementation here.
#
# Model contract (lair-train-pipeline's defaults, which is how the cubestack checkpoint
# was trained): head -> base_0_rgb, left/right wrist -> left/right_wrist_0_rgb, 25-D
# absolute state and actions, mean/std normalization (pi0) or quantile (pi0.5) from the
# checkpoint's assets/<asset id>/norm_stats.json. pi0 vs pi0.5 and the padded action dim
# are read from the checkpoint's parameter tree; the chunk length is not stored there,
# so it comes from --model-action-horizon.

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Protocol, Sequence

import numpy as np
import simplejpeg
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.action_limiter import (
    DEFAULT_MAX_START_OFFSET_DEG,
    DEFAULT_MAX_STEP_DEG,
    GEN2_DIM_NAMES,
    ActionLimits,
    limit_chunk,
    limits_from_degrees,
    resolve_limit_dims,
)
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
    REWARD_KEY,
    RL_ENABLED_KEY,
    STATUS_KEY,
    ServerFeature,
    ServerHandshake,
)
from policy_inference_spec.rl import (
    DEFAULT_AGENT_PORT,
    AgentObservation,
    ProducedActions,
    RemoteAgent,
    RLAgent,
    Transition,
)
from server.openpi_kv_ema import TARGETS as KV_EMA_TARGETS
from server.openpi_kv_ema import TOKEN_MODES as KV_EMA_TOKEN_MODES
from server.openpi_kv_ema import KvEmaConfig, KvEmaMemory
from server.trajectory_replay import (
    DEFAULT_APPROACH_DEG_S,
    DEFAULT_EXTRACT_PYTHON,
    DEFAULT_REPLAY_HZ,
    ReplayConfig,
    ReplayState,
    load_trajectory,
)
from server.prompts import (
    _PROMPT_FIELD_NAMES,
    _check_prompt_source,
    _frame_prompt,
    _is_metadata_origin,
    _metadata_summary,
)
from server.wire import _agent_images, _list_lan_ips, _wire_state_25

LOGGER = logging.getLogger(__name__)

DEFAULT_ACTION_HORIZON = 50
DEFAULT_MODEL_ACTION_HORIZON = 50
DEFAULT_NUM_STEPS = 10
DEFAULT_RESUME_GAP_S = 1.0
# --max-step-deg under --replay: above the fastest step of the teleop recordings measured
# (GEN2-105-*.rrd: 9.2 deg/step), so a recording plays back exactly; still catches a glitch.
DEFAULT_REPLAY_MAX_STEP_DEG = 10.0
# lair-train-pipeline's model.rtc defaults: d ~ U[min_delay, max_delay).
DEFAULT_RTC_TRAIN_MIN_DELAY = 0
DEFAULT_RTC_TRAIN_MAX_DELAY = 15
# Same as server.server, so a learner configured for one works against the other.
DEFAULT_RL_RESIZE_IMAGE = 128
DEFAULT_RL_OBS_DUMP_ROOT = Path("~/exp/obsdump")

# Wire camera -> openpi image slot. Mirrors lair-train-pipeline's `robot.cameras`, which
# maps the dataset features the wire cameras were recorded as
# (observation.images.head / left_wrist / right_wrist) onto these slots.
CAMERA_SLOTS: dict[str, str] = {
    "observation/images/main_image": "base_0_rgb",
    "observation/images/left_wrist_image": "left_wrist_0_rgb",
    "observation/images/right_wrist_image": "right_wrist_0_rgb",
}

# The names server.server gives these cameras. The RL observation dump names its files
# after them, and the viewer reads those names, so an openpi run's dump looks the same.
DUMP_CAMERA_NAMES: dict[str, str] = {
    "observation/images/main_image": "observation.images.head",
    "observation/images/left_wrist_image": "observation.images.left_wrist",
    "observation/images/right_wrist_image": "observation.images.right_wrist",
}
assert DUMP_CAMERA_NAMES.keys() == CAMERA_SLOTS.keys()

_REQUEST_COUNTER = count()
# The JAX policy and its RNG are not re-entrant. Serialize forward passes -- not
# connections, so a new client always gets its handshake even while another is open.
_PREDICT_LOCK = threading.Lock()


class ChunkPolicy(Protocol):
    """What the connection handler needs from a backend: one chunk per observation."""

    policy_id: str
    action_horizon: int  # rows per predicted chunk
    rtc_train: bool
    rtc_train_min_delay: int
    rtc_train_max_delay: int
    kv_ema: KvEmaConfig | None  # None: the KV cache is used as computed

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
        *,
        prefix: np.ndarray | None,
        kv_memory: KvEmaMemory | None = None,
    ) -> np.ndarray:
        """(action_horizon, robot action dim) robot-space chunk. ``prefix`` is a robot-space
        (d, action dim) block of committed actions to pin at the start, or None.
        ``kv_memory`` is the connection's KV-cache average, required iff ``kv_ema`` is set."""
        ...

    def latent_backend(self) -> "LatentBackend":
        """The prefix / denoise split latent RL drives (only needed with --rl-agent-url)."""
        ...


class LatentBackend(Protocol):
    """What the RL path needs; server.openpi_latent.OpenpiLatentBackend is the real one."""

    latent_shape: tuple[int, int]

    def prefix(self, obs: dict[str, Any]) -> tuple[np.ndarray, Any]: ...
    def denoise(self, cache: Any, noise: np.ndarray) -> np.ndarray: ...
    def unnormalize(self, cache: Any, raw_chunk: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------------------
# openpi backend
# --------------------------------------------------------------------------------------


def _read_param_tree(params_dir: Path) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Parameter path -> global shape, from the orbax metadata (no arrays are loaded).

    Not ``_METADATA``'s ``write_shape``: that is the per-shard chunk, so a checkpoint saved
    FSDP-sharded (e.g. 4 GPUs) reports action_in_proj as (8, 1024) instead of (32, 1024).
    """
    import jax
    import orbax.checkpoint as ocp

    meta = ocp.PyTreeCheckpointer().metadata(params_dir.resolve())
    meta = getattr(meta, "item_metadata", meta)
    shapes = {}
    for key_path, leaf in jax.tree_util.tree_flatten_with_path(dict(meta))[0]:
        shapes[tuple(str(getattr(k, "key", k)) for k in key_path)] = tuple(leaf.shape)
    return shapes


def _model_spec_from_params(params_dir: Path) -> tuple[bool, int]:
    """(is pi0.5, padded model action dim), read off the weights.

    pi0 feeds the state through ``state_proj``; pi0.5 tokenizes it and instead has
    ``time_mlp_*`` for adaRMSNorm conditioning. Loading one as the other fails deep inside
    flax with a shape error, so decide here where the reason can be stated.
    """
    shapes = _read_param_tree(params_dir)
    names = {part for path in shapes for part in path}
    if "state_proj" in names:
        pi05 = False
    elif "time_mlp_in" in names:
        pi05 = True
    else:
        raise ValueError(f"{params_dir} is neither pi0 (state_proj) nor pi0.5 (time_mlp_in)")
    (action_dim, _) = next(
        shape for path, shape in shapes.items() if "action_in_proj" in path and path[-2] == "kernel"
    )
    return pi05, int(action_dim)


def _norm_stats_dir(checkpoint_path: Path, asset_id: str | None) -> Path:
    assets = checkpoint_path / "assets"
    if asset_id is not None:
        return assets / asset_id
    candidates = sorted(p.parent for p in assets.glob("*/norm_stats.json"))
    assert len(candidates) == 1, (
        f"expected exactly one assets/<asset id>/norm_stats.json under {checkpoint_path}, found "
        f"{[c.name for c in candidates]}; pick one with --asset-id"
    )
    return candidates[0]


class OpenpiChunkPolicy:
    """An openpi ``Policy`` built from a lair-train checkpoint, plus the wire mapping."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        rtc_train: bool,
        model_action_horizon: int = DEFAULT_MODEL_ACTION_HORIZON,
        rtc_train_min_delay: int = DEFAULT_RTC_TRAIN_MIN_DELAY,
        rtc_train_max_delay: int = DEFAULT_RTC_TRAIN_MAX_DELAY,
        num_steps: int = DEFAULT_NUM_STEPS,
        max_token_len: int | None = None,
        asset_id: str | None = None,
        kv_ema: KvEmaConfig | None = None,
    ) -> None:
        # Deferred so the connection handler can be imported (and tested) without JAX.
        import jax.numpy as jnp
        from openpi import transforms as _transforms
        from openpi.models import model as _model
        from openpi.models import pi0_config
        from openpi.policies import policy as _policy
        from openpi.shared import normalize as _normalize
        from openpi.training import config as _config

        checkpoint_path = Path(checkpoint_path)
        params_dir = checkpoint_path / "params"
        assert params_dir.is_dir(), f"{checkpoint_path} has no params/ -- not an openpi checkpoint"
        pi05, model_action_dim = _model_spec_from_params(params_dir)

        stats_dir = _norm_stats_dir(checkpoint_path, asset_id)
        norm_stats = _normalize.load(stats_dir)
        self.robot_action_dim = len(norm_stats["actions"].mean)
        assert self.robot_action_dim == DEFAULT_HARDWARE_MODEL.action_dim, (
            f"checkpoint predicts {self.robot_action_dim}-D actions, the wire carries "
            f"{DEFAULT_HARDWARE_MODEL.action_dim}"
        )
        assert len(norm_stats["state"].mean) == self.robot_action_dim, (
            "state and action norm stats differ in length; the 25-D wire state layout assumes they match"
        )

        model_config = pi0_config.Pi0Config(
            pi05=pi05,
            action_dim=model_action_dim,
            action_horizon=model_action_horizon,
            max_token_len=max_token_len,
            rtc_train_enabled=rtc_train,
            rtc_train_min_delay=rtc_train_min_delay,
            rtc_train_max_delay=rtc_train_max_delay,
        )
        # The same rule openpi's DataConfig applies at training time.
        use_quantiles = model_config.model_type != _model.ModelType.PI0
        model = model_config.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))
        model_transforms = _config.ModelTransformFactory()(model_config)
        self._policy = _policy.Policy(
            model,
            transforms=[
                _transforms.Normalize(norm_stats, use_quantiles=use_quantiles),
                *model_transforms.inputs,
            ],
            output_transforms=[
                *model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles),
            ],
            sample_kwargs={"num_steps": num_steps},
        )

        self.policy_id = checkpoint_path.name
        self.action_horizon = model_action_horizon
        self._num_steps = num_steps
        self._latent_backend = None
        self.rtc_train = rtc_train
        self.rtc_train_min_delay = rtc_train_min_delay if rtc_train else 0
        self.rtc_train_max_delay = rtc_train_max_delay if rtc_train else 0
        self.kv_ema = kv_ema
        self._kv_ema_sampler = None
        if kv_ema is not None:
            from server.openpi_kv_ema import KvEmaSampler

            self._kv_ema_sampler = KvEmaSampler(self._policy, kv_ema, num_steps=num_steps)
            LOGGER.warning("KV-CACHE EMA ON (experimental): %s", kv_ema.describe())
        LOGGER.info(
            "loaded %s: %s, model action dim %d (robot %d), horizon %d, %d denoise steps, "
            "%s normalization from %s, rtc_train=%s%s",
            checkpoint_path, "pi0.5" if pi05 else "pi0", model_action_dim, self.robot_action_dim,
            model_action_horizon, num_steps, "quantile" if use_quantiles else "mean/std", stats_dir,
            rtc_train, f" delays [{rtc_train_min_delay}, {rtc_train_max_delay})" if rtc_train else "",
        )

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
        *,
        prefix: np.ndarray | None,
        kv_memory: KvEmaMemory | None = None,
    ) -> np.ndarray:
        obs = openpi_obs(images, state, prompt)
        if self.rtc_train:
            # Always hand over a prefix, zero-length when nothing is committed. prefix_len
            # is a traced value, so both cases share one compiled sampler; an unconditioned
            # first request would otherwise leave the conditioned graph to compile in the
            # middle of a rollout. With prefix_len 0 the mask is all False and sampling is
            # the plain flow.
            padded = np.zeros((self.action_horizon, self.robot_action_dim), dtype=np.float32)
            d = 0
            if prefix is not None:
                d = min(len(prefix), self.action_horizon)
                padded[:d] = prefix[:d]
            # Under the training action key: Normalize (and delta actions, if a checkpoint
            # uses them) treat it exactly like a training action chunk.
            obs["actions"] = padded
            obs["prefix_len"] = d
        else:
            assert prefix is None, "a prefix reached a policy served without --rtc-train"
        if self._kv_ema_sampler is None:
            out = self._policy.infer(obs)
        else:
            assert kv_memory is not None, "--kv-ema-alpha needs the connection's KvEmaMemory"
            out = self._kv_ema_sampler.infer(obs, kv_memory)
        return np.asarray(out["actions"][:, : self.robot_action_dim], dtype=np.float32)

    def latent_backend(self):
        if self._latent_backend is None:
            from server.openpi_latent import OpenpiLatentBackend

            assert not self.rtc_train, (
                "latent RL needs a policy served without --rtc-train: a pinned prefix overrides "
                "the rows the agent's latent produced"
            )
            self._latent_backend = OpenpiLatentBackend(
                self._policy, robot_action_dim=self.robot_action_dim, num_steps=self._num_steps,
            )
        return self._latent_backend


def openpi_obs(images: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> dict[str, Any]:
    """The raw observation dict openpi's input transforms expect."""
    return {
        "state": np.asarray(state, dtype=np.float32),
        "image": images,
        "image_mask": {slot: np.True_ for slot in images},
        "prompt": prompt,
    }


def _decode_images(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    """Wire JPEGs -> openpi slot -> uint8 HWC. openpi resizes (with padding) to 224 itself."""
    return {
        slot: simplejpeg.decode_jpeg(frame[wire_key], colorspace="RGB")
        for wire_key, slot in CAMERA_SLOTS.items()
    }


# --------------------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------------------


@dataclass
class ServerOptions:
    action_horizon: int = DEFAULT_ACTION_HORIZON
    task: str | None = None
    prompt_source: str = "auto"
    resume_gap_s: float | None = DEFAULT_RESUME_GAP_S
    action_limits: ActionLimits | None = None
    one_shot: bool = False
    rl: RlRuntime | None = None
    expo: Any = None  # expo_ft.ultra.runtime.ExpoRuntime with --expo-ft, else None
    replay: ReplayConfig | None = None  # --replay: answer with a recording, not the model


@dataclass
class ReplayPolicy:
    """Stands in for the model under --replay: it only names the chunks (policy_id)."""

    policy_id: str
    action_horizon: int = DEFAULT_ACTION_HORIZON
    rtc_train: bool = False
    rtc_train_min_delay: int = 0
    rtc_train_max_delay: int = 0
    kv_ema: KvEmaConfig | None = None

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("--replay serves recorded chunks; the model is never called")


@dataclass
class RlRuntime:
    """Latent-space RL, attached with --rl-agent-url. See init_rl()."""

    backend: LatentBackend
    agent: RLAgent
    resize_image: int = DEFAULT_RL_RESIZE_IMAGE
    obs_dump: Any = None  # server.rl_obs_dump.RlObsDump, or None


class _RlState:
    """Per-connection RL bookkeeping (see server.server._RlState): only the previous
    step's id, so a new episode starts with no predecessor."""

    def __init__(self) -> None:
        self.last_step_id: int | None = None

    def reset(self) -> None:
        self.last_step_id = None


class _RtcState:
    """Per-connection memory of what the robot is executing (see server.server._RtcState).

    ``last_chunk`` is the final returned chunk, after limiting and truncation, in robot
    space -- the same space as a wire ``action_prefix``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_chunk: np.ndarray | None = None
        self.last_arrival_s: float | None = None
        self.unconditioned_streak = 0
        # One-shot: the next request ignores any prefix it carries, because nothing this
        # connection knows of is committed. True on a fresh connection and after reset().
        self.resume_pending = True


class _KvEmaState:
    """Per-connection KV-cache average (--kv-ema-alpha). Reset by the same events that
    make an RTC prefix stale: a new connection, ``done``, an intervention, or a request
    gap of --resume-gap-s -- after any of them the previous frames are not the recent past."""

    def __init__(self) -> None:
        self.memory = KvEmaMemory()
        self.last_arrival_s: float | None = None

    def reset(self) -> None:
        self.memory.reset()

    def before_request(self, opts: "ServerOptions", *, req_id: int) -> str:
        """Apply the gap rule; return a log label."""
        now = time.perf_counter()
        gap_s = None if self.last_arrival_s is None else now - self.last_arrival_s
        self.last_arrival_s = now
        if opts.resume_gap_s is not None and gap_s is not None and gap_s >= opts.resume_gap_s \
                and not self.memory.empty:
            LOGGER.warning("predict req=%d kv EMA reset after a %.1fs gap in requests", req_id, gap_s)
            self.memory.reset()
        return f"kv_ema={'start' if self.memory.empty else f'step{self.memory.steps}'}"


def _resolve_prefix(
    frame: dict[str, Any],
    rtc: _RtcState,
    policy: ChunkPolicy,
    opts: ServerOptions,
    *,
    req_id: int,
) -> tuple[np.ndarray | None, str]:
    """The robot-space prefix to pin for this request, or None; plus a log label.

    Same rules as server.server._resolve_prefix_kwargs: the prefix is the first
    ``d = prefix_change_start`` aligned steps of the wire ``action_prefix`` (else of our
    own last chunk past its first d rows), and it is dropped whenever it cannot be a
    promise about what the robot is executing right now -- first request of a connection,
    after an intervention, or after a request gap of --resume-gap-s.
    """
    now = time.perf_counter()
    gap_s = None if rtc.last_arrival_s is None else now - rtc.last_arrival_s
    rtc.last_arrival_s = now
    resume = rtc.resume_pending
    rtc.resume_pending = False

    d = int(frame.get(PREFIX_CHANGE_START_KEY) or 0)
    if d <= 0:
        if rtc.last_chunk is not None:
            rtc.unconditioned_streak += 1
            if rtc.unconditioned_streak == 2 or rtc.unconditioned_streak % 25 == 0:
                LOGGER.warning(
                    "predict req=%d %d unconditioned chunks in a row with a previous chunk in "
                    "hand: the client must send action_prefix and prefix_change_start on "
                    "every request after the first", req_id, rtc.unconditioned_streak,
                )
        return None, "rtc_train=off(delay=0)"
    rtc.unconditioned_streak = 0
    if not policy.rtc_train_min_delay <= d < policy.rtc_train_max_delay:
        LOGGER.warning(
            "predict req=%d prefix_change_start=%d is outside the trained delay range [%d, %d); "
            "the pinned positions are out of distribution",
            req_id, d, policy.rtc_train_min_delay, policy.rtc_train_max_delay,
        )
    if resume:
        return None, "rtc_train=off(resume)"
    if opts.resume_gap_s is not None and gap_s is not None and gap_s >= opts.resume_gap_s:
        LOGGER.warning(
            "predict req=%d RESUME after a %.1fs gap in requests: ignoring the action_prefix, "
            "which was computed before the gap, and planning from the live observation",
            req_id, gap_s,
        )
        return None, f"rtc_train=off(resume-gap={gap_s:.1f}s)"

    wire_prefix = frame.get(ACTION_PREFIX_KEY)
    if wire_prefix is not None:
        prev = np.asarray(wire_prefix, dtype=np.float32)
        source = "wire"
    elif rtc.last_chunk is not None:
        prev = rtc.last_chunk[min(d, len(rtc.last_chunk)):]
        source = "state"
    else:
        return None, "rtc_train=off(no-prefix)"
    d = min(d, len(prev))
    if d == 0:
        return None, f"rtc_train=off(empty-prefix src={source})"
    return prev[:d], f"rtc_train=on src={source} prefix_len={d}"


def _inference_response(
    frame: dict[str, Any],
    rtc: _RtcState,
    policy: ChunkPolicy,
    opts: ServerOptions,
    *,
    prompt: str,
    kv: _KvEmaState | None = None,
) -> dict[str, Any]:
    req_id = next(_REQUEST_COUNTER)
    state = _wire_state_25(frame[JOINT_STATE_KEY].astype(np.float32))
    images = _decode_images(frame)

    prefix, debug = (None, "rtc_train=disabled")
    if policy.rtc_train:
        prefix, debug = _resolve_prefix(frame, rtc, policy, opts, req_id=req_id)
    kv_kwargs = {}
    if getattr(policy, "kv_ema", None) is not None:
        assert kv is not None, "a --kv-ema-alpha policy needs the connection's _KvEmaState"
        debug += " " + kv.before_request(opts, req_id=req_id)
        kv_kwargs["kv_memory"] = kv.memory

    t0 = time.perf_counter()
    q_value = None
    with _PREDICT_LOCK:
        if opts.expo is not None:
            prev = frame.get(PREV_SKIPPED_ACTION_START_KEY)
            chunk = opts.expo.predict(images, state, prompt, prefix=prefix,
                                      prev_executed=None if prev is None else int(prev))
            info = opts.expo.last_info
            q_value = info["q_max"]
            debug += f" expo idx={info['idx']}{'(edit)' if info['edited'] else ''} q={q_value:.3f}"
        else:
            chunk = policy.predict(images, state, prompt, prefix=prefix, **kv_kwargs)
    inference_time_s = time.perf_counter() - t0

    committed = 0 if prefix is None else len(prefix)
    if committed:
        # The sampler returns the pinned rows verbatim; anything else means the prefix
        # never reached the model (wrong key, dropped by a transform).
        drift = float(np.abs(chunk[:committed] - prefix).max())
        if drift > 1e-2:
            LOGGER.warning("predict req=%d chunk does not reproduce its prefix: max |diff| %.4f",
                           req_id, drift)
    resp = _respond(frame, policy, opts, state=state, chunk=chunk, committed=committed,
                    inference_time_s=inference_time_s, req_id=req_id, prompt=prompt, debug=debug,
                    q_value=q_value)
    if policy.rtc_train:
        rtc.last_chunk = resp[ACTION_KEY].copy()
    if opts.expo is not None:
        opts.expo.record_sent(resp[ACTION_KEY])  # what the robot executes: after the limiter
    return resp


def _respond(
    frame: dict[str, Any],
    policy: ChunkPolicy,
    opts: ServerOptions,
    *,
    state: np.ndarray,
    chunk: np.ndarray,
    committed: int,
    inference_time_s: float,
    req_id: int,
    prompt: str,
    debug: str,
    q_value: float | None = None,
    lead: np.ndarray | None = None,
) -> dict[str, Any]:
    """Truncate, limit and package one chunk: the part every predict path shares.

    ``lead``: the command the robot ran just before this chunk's row 0, if known. Row 0 is
    then rate-limited against it like any other step, instead of offset-limited against
    the live state."""
    action = chunk[: opts.action_horizon]

    if opts.action_limits is not None and lead is not None and committed == 0:
        limited, report = limit_chunk(
            np.vstack([lead[None, :], action]), state, opts.action_limits, committed_prefix_len=1,
        )
        action = limited[1:]
    elif opts.action_limits is not None:
        action, report = limit_chunk(
            action, state, opts.action_limits, committed_prefix_len=min(committed, len(action)),
        )
    if opts.action_limits is not None and report.triggered:
        LOGGER.warning("predict req=%d ACTION LIMITER fired: %s", req_id, report.describe(GEN2_DIM_NAMES))

    LOGGER.info(
        "predict req=%d prompt=%r %s chunk=%d returned=%d inference_s=%.3f "
        "||action[0]-state||=%.4f",
        req_id, prompt, debug, len(chunk), len(action), inference_time_s,
        float(np.linalg.norm(action[0] - state)),
    )
    resp = {
        ACTION_KEY: action,
        INFERENCE_TIME_KEY: inference_time_s,
        POLICY_ID_KEY: policy.policy_id,
        RL_ENABLED_KEY: opts.rl is not None or opts.expo is not None,
        Q_VALUE_KEY: 0.0 if q_value is None else float(q_value),
    }
    # Echo the caller's ID so a reply left buffered by a cancelled request can be told
    # apart from the current one (see server.server._inference_response).
    if CHUNK_ID_KEY in frame:
        resp[CHUNK_ID_KEY] = frame[CHUNK_ID_KEY]
    validate_wire_inference_response(resp)
    return resp


def _replay_response(
    frame: dict[str, Any],
    replay: ReplayState,
    policy: ChunkPolicy,
    opts: ServerOptions,
    *,
    prompt: str,
) -> dict[str, Any]:
    """--replay: the next slice of the recording, through the same limiter and packaging."""
    req_id = next(_REQUEST_COUNTER)
    state = _wire_state_25(frame[JOINT_STATE_KEY].astype(np.float32))
    t0 = time.perf_counter()
    out = replay.next_chunk(frame, state, horizon=opts.action_horizon)
    resp = _respond(frame, policy, opts, state=state, chunk=out.chunk, committed=out.committed,
                    inference_time_s=time.perf_counter() - t0, req_id=req_id, prompt=prompt,
                    debug=out.debug, lead=out.lead)
    # What the robot executes is the limited chunk; the next request's prefix is its tail.
    replay.last_sent = np.asarray(resp[ACTION_KEY], dtype=np.float32).copy()
    return resp


def _locked(fn, *args):
    with _PREDICT_LOCK:
        return fn(*args)


def _agent_observation(frame: dict[str, Any], rl: RlRuntime, *, state: np.ndarray,
                       prefix_rep: np.ndarray, prompt: str) -> AgentObservation:
    return AgentObservation(
        images=_agent_images(frame, CAMERA_SLOTS, rl.resize_image),
        state=state.astype(np.float32),
        prefix_rep=prefix_rep[0],
        task=prompt,
    )


async def _inference_response_rl(
    frame: dict[str, Any],
    rl_state: _RlState,
    policy: ChunkPolicy,
    opts: ServerOptions,
    *,
    prompt: str,
) -> dict[str, Any]:
    """RL path (see server.server._predict_chunk_rl): condition once, let the agent choose
    the starting latent from the prefix embedding, then denoise from it."""
    rl = opts.rl
    assert rl is not None
    req_id = next(_REQUEST_COUNTER)
    state = _wire_state_25(frame[JOINT_STATE_KEY].astype(np.float32))
    t0 = time.perf_counter()

    # Queued, not written: the dump's writer has its own thread. Keyed by the same step id
    # the learner records.
    if rl.obs_dump is not None:
        rl.obs_dump.submit(req_id, frame, task=prompt)

    obs = openpi_obs(_decode_images(frame), state, prompt)
    prefix_rep, cache = await asyncio.to_thread(_locked, rl.backend.prefix, obs)
    observation = _agent_observation(frame, rl, state=state, prefix_rep=prefix_rep, prompt=prompt)

    # Report the previous step's outcome before asking for a new latent; reward and done
    # ride in on this frame because they describe what the last chunk accomplished.
    if rl_state.last_step_id is not None:
        executed_raw = frame.get(PREV_SKIPPED_ACTION_START_KEY)
        executed_steps = int(executed_raw) if executed_raw else opts.action_horizon
        reward_raw = frame.get(REWARD_KEY)
        await rl.agent.observe(Transition(
            step_id=req_id,
            prev_step_id=rl_state.last_step_id,
            executed_steps=max(1, executed_steps),
            reward=None if reward_raw is None else float(reward_raw),
            done=bool(frame.get(DONE_KEY, False)),
        ))

    agent_action = await rl.agent.act(req_id, observation)
    latent = np.asarray(agent_action.latent, dtype=np.float32)[np.newaxis, ...]
    raw_chunk = await asyncio.to_thread(_locked, rl.backend.denoise, cache, latent)  # (1, T, A)
    rl_state.last_step_id = req_id
    chunk = rl.backend.unnormalize(cache, raw_chunk[0])
    inference_time_s = time.perf_counter() - t0

    # The whole denoised chunk, not the truncated one, goes back to a recording learner:
    # the rows past action_horizon are the part no one else can reconstruct. Sent after
    # the timing so a learner's disk write never reads as inference latency.
    actions_taken = getattr(rl.agent, "actions_taken", None)
    if actions_taken is not None:
        await actions_taken(ProducedActions(
            step_id=req_id,
            actions=chunk,
            raw_actions=raw_chunk[0],
            action_horizon=opts.action_horizon,
        ))

    debug = f"rl=on latent_std={float(latent.std()):.3f} q={agent_action.q_value}"
    return _respond(frame, policy, opts, state=state, chunk=chunk, committed=0,
                    inference_time_s=inference_time_s, req_id=req_id, prompt=prompt,
                    debug=debug, q_value=agent_action.q_value)


async def _rl_episode_end(frame: dict[str, Any], rl_state: _RlState, rl: RlRuntime,
                          *, prompt: str) -> None:
    """Close the episode on the agent, with a real post-action observation when the robot
    sends one (see server.server's ENDPOINT_DONE branch). No latent is requested and no
    chunk is denoised: the prefix pass is only for the critic's view of the last state."""
    reason = frame.get(DONE_REASON_KEY, "")
    executed_raw = frame.get(PREV_SKIPPED_ACTION_START_KEY)
    if executed_raw is None:
        LOGGER.warning("episode ended without terminal observation telemetry; "
                       "the learner will synthesize the last transition")
        await rl.agent.episode_end(str(reason) if reason else "")
        return
    assert rl_state.last_step_id is not None, "terminal telemetry has no preceding RL action"
    assert JOINT_STATE_KEY in frame and all(key in frame for key in CAMERA_SLOTS), (
        "terminal telemetry must include joint state and every policy camera"
    )
    executed_steps = int(executed_raw)
    assert executed_steps >= 1, f"terminal executed steps must be positive, got {executed_steps}"
    step_id = next(_REQUEST_COUNTER)
    state = _wire_state_25(frame[JOINT_STATE_KEY].astype(np.float32))
    if rl.obs_dump is not None:
        rl.obs_dump.submit(step_id, frame, task=prompt, terminal=True)
    obs = openpi_obs(_decode_images(frame), state, prompt)
    prefix_rep, _ = await asyncio.to_thread(_locked, rl.backend.prefix, obs)
    await rl.agent.episode_end(
        str(reason) if reason else "",
        final_step_id=step_id,
        final_observation=_agent_observation(frame, rl, state=state, prefix_rep=prefix_rep, prompt=prompt),
        executed_steps=executed_steps,
    )


def _expo_episode_end(frame: dict[str, Any], expo: Any) -> dict:
    """endpoint=done for EXPO-FT: the terminal observation when the robot sent one."""
    prev = frame.get(PREV_SKIPPED_ACTION_START_KEY)
    has_obs = JOINT_STATE_KEY in frame and all(key in frame for key in CAMERA_SLOTS)
    if not has_obs:
        LOGGER.warning("EXPO-FT: episode ended without a terminal observation")
    return expo.on_done(
        _decode_images(frame) if has_obs else None,
        _wire_state_25(frame[JOINT_STATE_KEY].astype(np.float32)) if has_obs else None,
        None if prev is None else int(prev),
        str(frame.get(DONE_REASON_KEY, "") or ""),
    )


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL, include_image_resolution=True, server_features=server_features,
    )


async def handle_inference_connection(
    connection: ServerConnection,
    policy: ChunkPolicy,
    opts: ServerOptions,
    *,
    server_features: Iterable[str | ServerFeature] = (),
) -> None:
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))

    rtc = _RtcState()
    kv = _KvEmaState()
    rl_state = _RlState()
    replay = opts.replay.new_state() if opts.replay is not None else None
    last_prompt: str | None = None
    last_metadata: str | None = None
    held_prompt: str | None = None

    def resolve_prompt(frame: dict[str, Any]) -> tuple[str, str]:
        # Same rules as server.server: a metadata instruction, once seen, holds until the
        # episode ends rather than falling back to construction names mid-episode.
        nonlocal held_prompt
        if opts.task is not None:
            return opts.task, "--task"
        prompt, origin = _frame_prompt(frame, source=opts.prompt_source)
        if _is_metadata_origin(origin) and prompt:
            held_prompt = prompt
            return prompt, origin
        if held_prompt is not None:
            return held_prompt, "held from an earlier request"
        return prompt, origin

    async for message in connection:
        assert isinstance(message, bytes), type(message)
        frame = deserialize_from_msgpack(message)
        if not isinstance(frame, dict):
            await connection.send(serialize_to_msgpack({"error": "expected dict frame"}))
            continue
        endpoint = frame.get(ENDPOINT_KEY)
        if endpoint == ENDPOINT_DONE:
            if opts.expo is not None:
                # Blocks until the operator labels the episode and the per-episode updates
                # finish: the robot cannot start the next episode before then.
                await asyncio.to_thread(_expo_episode_end, frame, opts.expo)
            if opts.rl is not None:
                # Before held_prompt is cleared: the terminal observation belongs to the
                # episode that just ended and is conditioned on its instruction.
                terminal_prompt, _ = resolve_prompt(frame)
                await _rl_episode_end(frame, rl_state, opts.rl, prompt=terminal_prompt)
                rl_state.reset()
            rtc.reset()
            kv.reset()
            if replay is not None:
                replay.rewind()
            held_prompt = None
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue
        if endpoint == ENDPOINT_INTERVENTION:
            # A human took over: the chunk we last sent was abandoned, so it must not come
            # back as the next request's prefix.
            # rl_state is deliberately kept (as in server.server): its last_step_id links
            # a transition the learner still has to close, across the intervention.
            LOGGER.info("intervention frame: dropping the last chunk; the next request "
                        "starts a fresh, unconditioned plan")
            rtc.reset()
            kv.reset()
            if replay is not None:
                replay.interrupt()
            if opts.expo is not None:
                prev = frame.get(PREV_SKIPPED_ACTION_START_KEY)
                await asyncio.to_thread(
                    opts.expo.on_intervention, _decode_images(frame),
                    _wire_state_25(frame[JOINT_STATE_KEY].astype(np.float32)),
                    np.asarray(frame[ACTION_KEY], np.float32), None if prev is None else int(prev))
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue
        if endpoint is not None:
            if endpoint != ENDPOINT_TELEMETRY:
                LOGGER.info("ignoring unhandled endpoint %r keys=%s", endpoint, sorted(frame))
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue

        validate_wire_inference_request_frame(frame)
        _ = frame[MODEL_ID_KEY]
        prompt, origin = resolve_prompt(frame)
        summary = _metadata_summary(frame)
        if summary != last_metadata:
            LOGGER.info("request metadata %s", summary)
            last_metadata = summary
        if prompt != last_prompt:
            (LOGGER.info if prompt else LOGGER.warning)("prompt=%r from %s", prompt, origin)
            last_prompt = prompt
        # The forward pass blocks for ~100 ms; run it off the event loop so websocket
        # pings keep flowing.
        if replay is not None:
            resp = _replay_response(frame, replay, policy, opts, prompt=prompt)
        elif opts.rl is not None:
            resp = await _inference_response_rl(frame, rl_state, policy, opts, prompt=prompt)
        else:
            resp = await asyncio.to_thread(_inference_response, frame, rtc, policy, opts, prompt=prompt, kv=kv)
        await connection.send(serialize_to_msgpack(resp))
        if opts.one_shot:
            LOGGER.info("one-shot mode: sent one inference response; ignoring further requests")
            await asyncio.Future()


@asynccontextmanager
async def run_server(
    policy: ChunkPolicy,
    opts: ServerOptions,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    server_features: Iterable[str | ServerFeature] = (ServerFeature.REWARDS,),
) -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        LOGGER.info("connection from %s", connection.remote_address)
        try:
            await handle_inference_connection(connection, policy, opts, server_features=server_features)
        finally:
            if opts.expo is not None:
                opts.expo.connection_closed()  # an unlabelled episode is discarded, as in dsrl_pi0

    async with websockets.serve(handler, host, port, max_size=None) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://{host}:{port}/ws"


def _warmup(policy: ChunkPolicy, *, rl: bool = False) -> None:
    """Compile the sampler before the robot connects (the first JAX call takes ~1 min).

    With RL the robot only ever reaches the split prefix / denoise path, so that is what
    gets compiled instead of the whole-chunk sampler.
    """
    h, w = DEFAULT_HARDWARE_MODEL.image_resolution
    images = {slot: np.zeros((h, w, 3), dtype=np.uint8) for slot in CAMERA_SLOTS.values()}
    state = np.zeros(DEFAULT_HARDWARE_MODEL.action_dim, dtype=np.float32)

    # With --kv-ema-alpha one memory across both calls, so the first call compiles the
    # unsmoothed start and the second the blend.
    kv_kwargs = {"kv_memory": KvEmaMemory()} if getattr(policy, "kv_ema", None) is not None else {}

    def once() -> None:
        if not rl:
            policy.predict(images, state, "warmup", prefix=None, **kv_kwargs)
            return
        backend = policy.latent_backend()
        _, cache = backend.prefix(openpi_obs(images, state, "warmup"))
        raw = backend.denoise(cache, np.zeros((1, *backend.latent_shape), dtype=np.float32))
        backend.unnormalize(cache, raw[0])

    t0 = time.perf_counter()
    once()
    t1 = time.perf_counter()
    once()
    LOGGER.info("warmup%s: first call %.1fs (compile), second %.3fs",
                " (rl prefix + denoise)" if rl else "", t1 - t0, time.perf_counter() - t1)


async def init_rl(policy: ChunkPolicy, agent_url: str, *, resize_image: int = DEFAULT_RL_RESIZE_IMAGE,
                  obs_dump_dir: str | None = None) -> RlRuntime:
    """Connect to the learner and check its action space is this policy's noise space."""
    assert resize_image > 0, f"resize_image must be positive, got {resize_image}"
    backend = policy.latent_backend()
    obs_dump = None
    if obs_dump_dir:
        from server.rl_obs_dump import RlObsDump

        # openpi resizes (with padding) to 224 itself and declares no input shape, so the
        # dump records each camera's size rather than checking it.
        obs_dump = RlObsDump(obs_dump_dir, camera_keys=DUMP_CAMERA_NAMES, trained_shapes={},
                             checkpoint=policy.policy_id, policy_type="openpi")
    agent = RemoteAgent(agent_url)
    handshake = await agent.connect()
    assert tuple(handshake.latent_shape) == tuple(backend.latent_shape), (
        f"agent produces latents of shape {handshake.latent_shape} but the policy denoises "
        f"{backend.latent_shape}; the agent's action space must match the policy's noise space "
        "(learner flags --chunk_size / --latent_dim)"
    )
    LOGGER.info("Latent RL enabled: agent=%s latent_shape=%s resize_image=%d",
                agent.url, backend.latent_shape, resize_image)
    return RlRuntime(backend=backend, agent=agent, resize_image=resize_image, obs_dump=obs_dump)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy inference server with an openpi (JAX) backend.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_INFERENCE_SERVER_PORT, help="Bind port")
    parser.add_argument("--checkpoint-path", type=Path, default=None,
                        help="openpi checkpoint step directory (holds params/ and assets/). "
                        "Required unless --replay.")
    parser.add_argument("--replay", type=Path, default=None, metavar="PATH",
                        help="Replay a recorded joint trajectory (.rrd, or an .npz from "
                        "scripts/extract_rrd_actions.py) on the robot instead of running a model. "
                        "Requests are answered with consecutive slices of the recording's "
                        "commanded joints; see server/trajectory_replay.py.")
    parser.add_argument("--replay-hz", type=float, default=DEFAULT_REPLAY_HZ, metavar="HZ",
                        help=f"Robot control rate the recording is resampled to (default: {DEFAULT_REPLAY_HZ:g}).")
    parser.add_argument("--replay-start-s", type=float, default=0.0, metavar="S",
                        help="Skip the recording's first S seconds (default: 0).")
    parser.add_argument("--replay-end-s", type=float, default=None, metavar="S",
                        help="Stop at S seconds into the recording, then hold (default: the end).")
    parser.add_argument("--replay-approach-deg-s", type=float, default=DEFAULT_APPROACH_DEG_S,
                        metavar="DEG_S", help="Peak joint speed of the ramp from the robot's pose to "
                        f"the recording's, at start and after a resume (default: {DEFAULT_APPROACH_DEG_S:g}).")
    parser.add_argument("--replay-extract-python", type=Path, default=DEFAULT_EXTRACT_PYTHON,
                        metavar="PY", help="Python with rerun 0.30 used to read an .rrd "
                        f"(default: {DEFAULT_EXTRACT_PYTHON}).")
    parser.add_argument("--asset-id", default=None,
                        help="Which assets/<asset id>/norm_stats.json to use (default: the only one).")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON,
                        help=f"Action rows returned per request (default: {DEFAULT_ACTION_HORIZON}).")
    parser.add_argument("--model-action-horizon", type=int, default=DEFAULT_MODEL_ACTION_HORIZON,
                        help="Chunk length the checkpoint was trained with (model.action_horizon; "
                             f"not stored in the checkpoint; default: {DEFAULT_MODEL_ACTION_HORIZON}).")
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS,
                        help=f"Flow-matching denoise steps (default: {DEFAULT_NUM_STEPS}).")
    parser.add_argument("--max-token-len", type=int, default=None,
                        help="Prompt token budget (model.max_token_len; default 48 pi0 / 200 pi0.5).")
    parser.add_argument("--task", default=None,
                        help="Task prompt for every request. If omitted, read from each request.")
    parser.add_argument("--prompt-source", type=_check_prompt_source, default="auto", metavar="SOURCE",
                        help="Where to read the instruction when --task is not given; same as "
                             f"server.server: 'auto' (a {'/'.join(_PROMPT_FIELD_NAMES)} field in "
                             "the metadata, else subtask, else task), 'task', 'subtask', or a "
                             "dotted path into the frame.")
    parser.add_argument("--rtc-train", action="store_true",
                        help="Train-time RTC: pin the committed action prefix while denoising the "
                             "rest. Requires a checkpoint trained with model.rtc.enabled.")
    parser.add_argument("--rtc-train-min-delay", type=int, default=DEFAULT_RTC_TRAIN_MIN_DELAY,
                        help="model.rtc.min_delay the checkpoint was trained with (warnings only).")
    parser.add_argument("--rtc-train-max-delay", type=int, default=DEFAULT_RTC_TRAIN_MAX_DELAY,
                        help="model.rtc.max_delay the checkpoint was trained with (warnings only).")
    parser.add_argument("--resume-gap-s", type=float, default=DEFAULT_RESUME_GAP_S, metavar="S",
                        help="A gap this long between requests is a pause: the next request's "
                             f"action_prefix is ignored (default: {DEFAULT_RESUME_GAP_S:g}; 0 = never).")
    parser.add_argument("--no-action-limiter", action="store_true",
                        help="Disable the per-chunk joint limits (on by default; see server.server).")
    parser.add_argument("--max-chunk-start-offset-deg", type=float, default=DEFAULT_MAX_START_OFFSET_DEG,
                        metavar="DEG", help=f"(default: {DEFAULT_MAX_START_OFFSET_DEG:g}; 0 disables)")
    parser.add_argument("--max-step-deg", type=float, default=None, metavar="DEG",
                        help=f"(default: {DEFAULT_MAX_STEP_DEG:g}, or {DEFAULT_REPLAY_MAX_STEP_DEG:g} with "
                        "--replay; 0 disables)")
    parser.add_argument("--limit-dims", nargs="*", default=[], metavar="DIM",
                        help="Action dims the limits apply to. Default: every joint dim, no grippers.")
    parser.add_argument("--no-rewards", action="store_true",
                        help="Do not advertise reward support in the handshake.")
    parser.add_argument("--one-shot", action="store_true",
                        help="Send one chunk per connection, then stop responding.")
    parser.add_argument("--rl-agent-url", default=None, metavar="URL",
                        help="Enable latent-space RL: the denoiser's starting noise comes from an RL "
                             f"agent at this WebSocket URL (e.g. ws://127.0.0.1:{DEFAULT_AGENT_PORT}), "
                             "which also receives the transitions. Same as server.server. Not with "
                             "--rtc-train.")
    parser.add_argument("--rl-obs-dump-dir", default=None, metavar="DIR",
                        help="Write the wire's own camera JPEGs for every RL query here (see "
                             "server.server). Default: a fresh "
                             f"{DEFAULT_RL_OBS_DUMP_ROOT}/<timestamp> per RL run.")
    parser.add_argument("--no-rl-obs-dump", action="store_true",
                        help="Do not write the RL observation dump (no offline latent resampling).")
    parser.add_argument("--kv-ema-alpha", type=float, default=None, metavar="A",
                        help="Experimental: smooth the prefix KV cache over the requests of one "
                             "connection, m = A * kv + (1 - A) * m_prev (A in (0, 1]; 1 = stock "
                             "output). Off by default. See server/openpi_kv_ema.py.")
    parser.add_argument("--kv-ema-target", choices=KV_EMA_TARGETS, default="kv",
                        help="Which half of the cache to smooth (default: kv = both).")
    parser.add_argument("--kv-ema-tokens", choices=KV_EMA_TOKEN_MODES, default="images",
                        help="'images' (default): camera-token positions only. 'all': also prompt "
                             "positions whose token is unchanged since the previous request.")
    parser.add_argument("--kv-ema-layers", type=int, nargs="*", default=None, metavar="L",
                        help="Layers to smooth (default: all).")
    parser.add_argument("--rl-resize-image", type=int, default=DEFAULT_RL_RESIZE_IMAGE,
                        help=f"Edge of the square images sent to the RL agent (default: {DEFAULT_RL_RESIZE_IMAGE}).")
    try:  # EXPO-FT lives in the expo-ft repo; its flags exist only when it is on PYTHONPATH.
        from expo_ft.ultra.server_hook import add_cli_args
    except ImportError:
        pass
    else:
        add_cli_args(parser)
    args = parser.parse_args(argv)
    if args.max_step_deg is None:
        # Human teleop moves the wrists up to ~9 deg per 50 Hz step, three times what the
        # policy limit allows; at 3 the replay lags its recording by up to ~28 deg.
        args.max_step_deg = DEFAULT_REPLAY_MAX_STEP_DEG if args.replay is not None else DEFAULT_MAX_STEP_DEG
    return args


def _obs_dump_dir_from_args(args: argparse.Namespace, *, now: float | None = None) -> str | None:
    """Where the RL observation dump goes, or None when off (server.server's rule)."""
    if args.no_rl_obs_dump or args.rl_obs_dump_dir == "":
        return None
    if args.rl_obs_dump_dir is not None:
        return args.rl_obs_dump_dir
    if args.rl_agent_url is None:
        return None
    stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(now))
    return str(DEFAULT_RL_OBS_DUMP_ROOT.expanduser() / stamp)


def _action_limits_from_args(args: argparse.Namespace) -> ActionLimits | None:
    if args.no_action_limiter or (args.max_chunk_start_offset_deg <= 0 and args.max_step_deg <= 0):
        return None
    return limits_from_degrees(
        max_start_offset_deg=args.max_chunk_start_offset_deg,
        max_step_deg=args.max_step_deg,
        dims=resolve_limit_dims(args.limit_dims),
    )


async def _run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    assert 1 <= args.action_horizon <= args.model_action_horizon, (
        f"--action-horizon must be in [1, {args.model_action_horizon}], got {args.action_horizon}"
    )
    assert not (args.rl_agent_url and args.rtc_train), (
        "--rl-agent-url and --rtc-train are mutually exclusive: the hard action prefix overrides "
        "the first steps of the chunk, so the executed actions are no longer the ones the agent's "
        "latent produced and the critic learns a mapping that does not exist"
    )
    assert not (args.rl_agent_url and args.kv_ema_alpha is not None), (
        "--kv-ema-alpha is not wired into the latent RL path (--rl-agent-url)"
    )
    if args.replay is not None:
        return await _run_replay(args)
    assert args.checkpoint_path is not None, "--checkpoint-path is required (unless --replay)"
    kv_ema = None
    if args.kv_ema_alpha is not None:
        kv_ema = KvEmaConfig(
            alpha=args.kv_ema_alpha, target=args.kv_ema_target, tokens=args.kv_ema_tokens,
            layers=None if not args.kv_ema_layers else tuple(args.kv_ema_layers),
        )
    policy = OpenpiChunkPolicy(
        args.checkpoint_path,
        rtc_train=args.rtc_train,
        model_action_horizon=args.model_action_horizon,
        rtc_train_min_delay=args.rtc_train_min_delay,
        rtc_train_max_delay=args.rtc_train_max_delay,
        num_steps=args.num_steps,
        max_token_len=args.max_token_len,
        asset_id=args.asset_id,
        kv_ema=kv_ema,
    )
    rl = None
    if args.rl_agent_url is not None:
        obs_dump_dir = _obs_dump_dir_from_args(args)
        if obs_dump_dir is None:
            LOGGER.warning("RL OBSERVATION DUMP OFF: this run keeps only the learner's 128px critic "
                           "view, which is too coarse to re-run the policy on")
        rl = await init_rl(policy, args.rl_agent_url, resize_image=args.rl_resize_image,
                           obs_dump_dir=obs_dump_dir)
        _RUNTIME.append(rl)
    expo = None
    if getattr(args, "expo_ft", False):
        assert rl is None, "--expo-ft and --rl-agent-url are mutually exclusive"
        assert kv_ema is None, "--kv-ema-alpha is not wired into --expo-ft"
        from expo_ft.ultra.server_hook import build_runtime, warmup

        expo = build_runtime(args, policy, predict_lock=_PREDICT_LOCK,
                             norm_stats_dir=_norm_stats_dir(args.checkpoint_path, args.asset_id))
        warmup(expo)
    else:
        _warmup(policy, rl=rl is not None)
    opts = ServerOptions(
        action_horizon=args.action_horizon,
        task=args.task,
        prompt_source=args.prompt_source,
        resume_gap_s=args.resume_gap_s if args.resume_gap_s > 0 else None,
        action_limits=_action_limits_from_args(args),
        one_shot=args.one_shot,
        rl=rl,
        expo=expo,
    )
    if opts.action_limits is None:
        LOGGER.warning("ACTION LIMITER OFF: chunks go to the robot exactly as the policy emits them")
    if args.rtc_train and opts.resume_gap_s is None:
        LOGGER.warning("RESUME DETECTION OFF: a prefix sent after a pause will be conditioned on")
    features = () if args.no_rewards else (ServerFeature.REWARDS,)
    if expo is not None:
        features = (*features, ServerFeature.HUMAN_INTERVENTIONS)
    async with run_server(policy, opts, args.host, args.port, server_features=features) as url:
        print(
            f"Server listening on {url} (openpi, action_horizon={args.action_horizon}, "
            f"rtc_train={'on' if args.rtc_train else 'off'}, rl={'on' if rl else 'off'}, "
            f"expo_ft={'on' if expo else 'off'}, "
            f"kv_ema={kv_ema.describe() if kv_ema else 'off'}, "
            f"prompt={'--task ' + repr(args.task) if args.task else 'from requests'})",
            flush=True,
        )
        if args.host == "0.0.0.0":
            # 0.0.0.0 is a bind address, not one a client can dial.
            for ip in _list_lan_ips():
                print(f"  reachable at ws://{ip}:{args.port}/ws", flush=True)
        await asyncio.Future()
    return 0


def _replay_config_from_args(args: argparse.Namespace) -> ReplayConfig:
    assert args.replay_approach_deg_s > 0, "--replay-approach-deg-s must be positive"
    trajectory = load_trajectory(
        args.replay, hz=args.replay_hz, start_s=args.replay_start_s, end_s=args.replay_end_s,
        extract_python=args.replay_extract_python,
    )
    return ReplayConfig(
        trajectory=trajectory,
        name=args.replay.name,
        hz=args.replay_hz,
        approach_speed=float(np.deg2rad(args.replay_approach_deg_s)),
        resume_gap_s=args.resume_gap_s if args.resume_gap_s > 0 else None,
    )


async def _run_replay(args: argparse.Namespace) -> int:
    """--replay: serve a recording instead of the model. Nothing model-side is loaded."""
    conflicts = {
        "--rtc-train": args.rtc_train, "--rl-agent-url": args.rl_agent_url,
        "--kv-ema-alpha": args.kv_ema_alpha is not None,
        "--expo-ft": getattr(args, "expo_ft", False),
    }
    used = [flag for flag, on in conflicts.items() if on]
    assert not used, f"--replay serves a recording, not a model; drop {', '.join(used)}"
    if args.checkpoint_path is not None:
        LOGGER.warning("--replay: ignoring --checkpoint-path %s (no model is loaded)", args.checkpoint_path)
    replay = _replay_config_from_args(args)
    policy = ReplayPolicy(policy_id=f"replay:{replay.name}", action_horizon=args.action_horizon)
    opts = ServerOptions(
        action_horizon=args.action_horizon,
        task=args.task,
        prompt_source=args.prompt_source,
        resume_gap_s=replay.resume_gap_s,
        action_limits=_action_limits_from_args(args),
        one_shot=args.one_shot,
        replay=replay,
    )
    traj = replay.trajectory
    steps = np.rad2deg(np.abs(np.diff(traj, axis=0)))
    if opts.action_limits is None:
        LOGGER.warning("ACTION LIMITER OFF: the recording goes to the robot exactly as logged")
    elif opts.action_limits.max_step is not None:
        dims = list(opts.action_limits.dims)
        over = int((steps[:, dims] > np.rad2deg(opts.action_limits.max_step)).any(axis=1).sum())
        if over:
            LOGGER.warning("%d of the recording's %d steps move a joint faster than --max-step-deg %g "
                           "(its fastest is %.1f deg/step); the limiter will slow those, so the robot "
                           "lags the recording through them. --max-step-deg %.0f replays it exactly",
                           over, len(steps), np.rad2deg(opts.action_limits.max_step),
                           steps[:, dims].max(), np.ceil(steps[:, dims].max()))
    features = () if args.no_rewards else (ServerFeature.REWARDS,)
    async with run_server(policy, opts, args.host, args.port, server_features=features) as url:
        print(
            f"Server listening on {url} (REPLAY {args.replay} -> {len(traj)} steps, "
            f"{len(traj) / replay.hz:.1f}s at {replay.hz:g} Hz, action_horizon={args.action_horizon}, "
            f"approach {args.replay_approach_deg_s:g} deg/s, "
            f"limiter={'on' if opts.action_limits else 'OFF'})",
            flush=True,
        )
        if args.host == "0.0.0.0":
            for ip in _list_lan_ips():
                print(f"  reachable at ws://{ip}:{args.port}/ws", flush=True)
        await asyncio.Future()
    return 0


# The RL runtime _run_cli attached, so main() can flush the observation dump on Ctrl-C.
_RUNTIME: list[RlRuntime] = []


def main(argv: Sequence[str] | None = None) -> None:
    try:
        try:
            raise SystemExit(asyncio.run(_run_cli(argv)))
        except KeyboardInterrupt:
            raise SystemExit(130)
    finally:
        # Runs end with Ctrl-C and the dump's writer is a daemon thread: without this the
        # tail of the last episode, still queued, is lost.
        for rl in _RUNTIME:
            if rl.obs_dump is not None:
                rl.obs_dump.close()


if __name__ == "__main__":
    main()
