# START SERVER (from the repo root, in the DynamicVLA venv -- see requirements-dynamicvla.txt):
#   PYTHONPATH=. .venv-dynamicvla/bin/python -m server.server_dynamicvla \
#       --checkpoint-path models/cubestack-dynamicvla --host 0.0.0.0 --port 18090
#
# The same wire protocol as server.server, with the forward pass replaced by DynamicVLA's
# (https://github.com/hzxie/DynamicVLA; the model code is imported from a local checkout,
# --dynamicvla-root). The robot client is unchanged.
#
# Why a separate process and venv: DynamicVLA is written against lerobot 0.3.3
# (lerobot.constants, lerobot.policies.normalize), which server.server's lerobot no longer
# has, on Python 3.10 / numpy 1.x. This module imports nothing from server.server; the
# backend-free pieces live in server.prompts and server.wire.
#
# Carried over from server.server, with the same flags and semantics: handshake, request
# validation, chunk_id echo, done / intervention / telemetry frames, prompt resolution
# (--task, --prompt-source), and the action limiter. Not carried over: RTC of either kind,
# latent RL, temporal ensembling, state pins.
#
# What is different about DynamicVLA, and what this server does about it:
#   * Observation history. The model fuses n_obs_steps frames (config
#     delta_timestamps.observation, in DATASET FRAMES -- [-5, 0] at the cubestack dataset's
#     50 Hz is "now" plus "0.1 s ago"). The wire carries one frame per request, so the
#     server keeps a short per-connection buffer of every frame it receives (inference
#     requests, plus telemetry frames that carry joint state and every camera) and picks,
#     for each offset, the buffered frame nearest that far back in arrival time. When none
#     is within --history-tolerance-s, the current frame stands in -- which is what
#     training does at the start of an episode, where the window runs off the front and is
#     clamped -- and the log says so. A client that only sends a frame per chunk therefore
#     runs the model on [now, now] most of the time; the per-request log line reports the
#     age of the frame actually used.
#   * Relative actions. use_delta_action subtracts the current state from some action dims
#     at train time; predict_action_chunk does not add it back. WHICH dims is decided here
#     from the checkpoint's own normalization stats (a relative dim's action std is a
#     fraction of its state std, an absolute dim's is not), because the training code that
#     produced the cubestack checkpoint left the grippers absolute, not the "last dim" the
#     local DynamicVLA code assumes. --absolute-action-dims overrides the inference.
#   * Images go in at the resolution each camera was trained at (the dataset's wrist
#     cameras are 300x480, the wire sends 360x640); the model then pads to 384x384 itself.

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

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
    CHUNK_ID_KEY,
    DEFAULT_INFERENCE_SERVER_PORT,
    ENDPOINT_DONE,
    ENDPOINT_INTERVENTION,
    ENDPOINT_KEY,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    Q_VALUE_KEY,
    RL_ENABLED_KEY,
    STATUS_KEY,
    ServerFeature,
    ServerHandshake,
)
from server.prompts import (
    _PROMPT_FIELD_NAMES,
    _check_prompt_source,
    _frame_prompt,
    _is_metadata_origin,
    _metadata_summary,
)
from server.wire import _list_lan_ips, _wire_state_25

LOGGER = logging.getLogger(__name__)

DEFAULT_DYNAMICVLA_ROOT = Path.home() / "embodied_at_sea_ws" / "src" / "DynamicVLA"
DEFAULT_ACTION_HORIZON = 50
# The cubestack LeRobot dataset (ultra-tools/data/cubestack/lerobot) is recorded at 50 Hz.
DEFAULT_DATASET_FPS = 50.0
DEFAULT_HISTORY_TOLERANCE_S = 0.05
# action std / state std above this means the dim was trained absolute (see header).
# Measured on the cubestack checkpoint: relative dims <= 0.51, grippers 1.33-1.36.
_ABSOLUTE_STD_RATIO = 0.8

# Wire camera -> DynamicVLA input feature (the dataset's names for the same cameras).
CAMERA_KEYS: dict[str, str] = {
    "observation/images/main_image": "observation.images.head",
    "observation/images/left_wrist_image": "observation.images.left_wrist",
    "observation/images/right_wrist_image": "observation.images.right_wrist",
}
STATE_KEY = "observation.state"

_REQUEST_COUNTER = count()
# One model on one GPU: serialize forward passes, not connections.
_PREDICT_LOCK = threading.Lock()


# --------------------------------------------------------------------------------------
# DynamicVLA backend
# --------------------------------------------------------------------------------------


def _find_weights(checkpoint_path: Path) -> Path:
    """The one *.safetensors in the checkpoint dir (DynamicVLA names them model.epochNNNN)."""
    candidates = sorted(checkpoint_path.glob("*.safetensors"))
    assert len(candidates) == 1, (
        f"expected exactly one .safetensors under {checkpoint_path}, found "
        f"{[c.name for c in candidates]}; pick one with --weights"
    )
    return candidates[0]


def _infer_absolute_dims(state_std: np.ndarray, action_std: np.ndarray) -> list[int]:
    ratio = action_std / np.maximum(state_std, 1e-8)
    ambiguous = [i for i, r in enumerate(ratio) if 0.6 <= r <= 1.0]
    if ambiguous:
        LOGGER.warning(
            "action/state std ratio is ambiguous for dims %s (%s); the relative/absolute "
            "split may be wrong there -- pass --absolute-action-dims to settle it",
            ambiguous, ", ".join(f"{ratio[i]:.2f}" for i in ambiguous),
        )
    return [i for i, r in enumerate(ratio) if r > _ABSOLUTE_STD_RATIO]


class DynamicVLAChunkPolicy:
    """A DynamicVLA checkpoint plus the wire mapping: n_obs_steps frames in, one
    absolute robot-space chunk out."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        dynamicvla_root: Path = DEFAULT_DYNAMICVLA_ROOT,
        weights: Path | None = None,
        dataset_fps: float = DEFAULT_DATASET_FPS,
        num_steps: int | None = None,
        absolute_action_dims: Sequence[int] | None = None,
        device: str = "cuda",
    ) -> None:
        # Deferred so the connection handler can be imported (and tested) without torch.
        # Appended, not prepended: the DynamicVLA checkout has top-level scripts/ and
        # utils/ that must not shadow anything else on the path.
        sys.path.append(str(dynamicvla_root))
        import safetensors.torch
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy

        self._torch = torch
        checkpoint_path = Path(checkpoint_path)
        raw = json.loads((checkpoint_path / "config.json").read_text())
        assert raw.get("type") == "dynamicvla", f"{checkpoint_path}/config.json is type={raw.get('type')!r}"

        # config.json also carries training-run keys (epoch, delta_timestamps) that are not
        # DynamicVLAConfig fields; keep only the fields, rebuilding the typed ones.
        fields = {f.name for f in dataclasses.fields(DynamicVLAConfig)}
        kwargs = {k: v for k, v in raw.items() if k in fields}

        def features(spec: dict[str, Any]) -> dict[str, Any]:
            return {k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"])) for k, v in spec.items()}

        kwargs["input_features"] = features(raw["input_features"])
        kwargs["output_features"] = features(raw["output_features"])
        kwargs["normalization_mapping"] = {k: NormalizationMode[v] for k, v in raw["normalization_mapping"].items()}
        for key in ("resize_imgs_with_padding", "optimizer_betas"):
            if kwargs.get(key) is not None:
                kwargs[key] = tuple(kwargs[key])
        kwargs["device"] = device
        kwargs["enable_streaming"] = False
        if num_steps is not None:
            kwargs["num_steps"] = num_steps
        config = DynamicVLAConfig(**kwargs)
        assert not getattr(config, "use_delta_tokens", False), (
            "checkpoint uses Delta tokens, which the robot cannot supply over this wire"
        )
        assert not config.adapt_to_pi_aloha, "adapt_to_pi_aloha checkpoints are not supported"

        missing_cams = [k for k in CAMERA_KEYS.values() if k not in config.input_features]
        extra_cams = [
            k for k, f in config.input_features.items()
            if f.type == FeatureType.VISUAL and k not in CAMERA_KEYS.values()
        ]
        assert not extra_cams, f"checkpoint expects cameras the wire does not carry: {extra_cams}"
        self.camera_keys = {w: k for w, k in CAMERA_KEYS.items() if k not in missing_cams}
        self.trained_hw = {k: tuple(config.input_features[k].shape[1:]) for k in self.camera_keys.values()}

        self.robot_action_dim = config.output_features["action"].shape[0]
        assert self.robot_action_dim == DEFAULT_HARDWARE_MODEL.action_dim, (
            f"checkpoint predicts {self.robot_action_dim}-D actions, the wire carries "
            f"{DEFAULT_HARDWARE_MODEL.action_dim}"
        )
        assert config.input_features[STATE_KEY].shape[0] == self.robot_action_dim, (
            "state and action differ in length; the 25-D wire state layout assumes they match"
        )

        # Observation offsets: dataset frames -> seconds (negative = in the past).
        frame_offsets = (raw.get("delta_timestamps") or {}).get("observation") or [0]
        assert len(frame_offsets) == config.n_obs_steps, (
            f"delta_timestamps.observation {frame_offsets} does not match n_obs_steps={config.n_obs_steps}"
        )
        assert frame_offsets[-1] == 0 and all(o <= 0 for o in frame_offsets), frame_offsets
        self.frame_offsets = [int(o) for o in frame_offsets]
        self.offsets_s = [o / dataset_fps for o in self.frame_offsets]

        weights = Path(weights) if weights is not None else _find_weights(checkpoint_path)
        t0 = time.perf_counter()
        policy = DynamicVLAPolicy(config)
        missing, unexpected = safetensors.torch.load_model(policy, str(weights), strict=False)
        assert not missing and not unexpected, (
            f"{weights}: {len(missing)} missing / {len(unexpected)} unexpected keys: "
            f"{sorted(missing)[:5]} {sorted(unexpected)[:5]}"
        )
        self._policy = policy.eval().to(device)
        self._device = device

        state_std = policy.normalize_inputs.buffer_observation_state.std.detach().cpu().numpy()
        action_std = policy.unnormalize_outputs.buffer_action.std.detach().cpu().numpy()
        if not config.use_delta_action:
            self.absolute_dims = list(range(self.robot_action_dim))
        elif absolute_action_dims is not None:
            self.absolute_dims = sorted(int(d) for d in absolute_action_dims)
        else:
            self.absolute_dims = _infer_absolute_dims(state_std, action_std)
        self.relative_mask = np.ones(self.robot_action_dim, dtype=bool)
        self.relative_mask[self.absolute_dims] = False

        self.policy_id = checkpoint_path.name
        self.action_horizon = config.chunk_size
        LOGGER.info(
            "loaded %s (%s) in %.1fs: chunk %d, %d denoise steps, obs offsets %s frames = %s s "
            "at %g Hz, cameras %s, use_delta_action=%s, absolute dims %s",
            checkpoint_path, weights.name, time.perf_counter() - t0, config.chunk_size,
            config.num_steps, self.frame_offsets, [round(s, 3) for s in self.offsets_s],
            dataset_fps, self.trained_hw, config.use_delta_action,
            [f"{d}:{GEN2_DIM_NAMES[d]}" for d in self.absolute_dims],
        )

    def _image_tensor(self, key: str, frames_hwc: list[np.ndarray]) -> Any:
        """(1, n_obs, 3, H, W) float in [0, 1] at the camera's trained resolution."""
        torch = self._torch
        x = torch.from_numpy(np.stack(frames_hwc)).to(self._device)  # (n, h, w, 3) uint8
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        want = self.trained_hw[key]
        if tuple(x.shape[-2:]) != want:
            x = torch.nn.functional.interpolate(x, size=want, mode="bilinear", align_corners=False, antialias=True)
        # prepare_images folds (n, c) with .view, which needs a contiguous tensor.
        return x.contiguous().unsqueeze(0)

    def predict(
        self,
        images: dict[str, list[np.ndarray]],
        states: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """(chunk, robot action dim) absolute robot-space chunk.

        ``images`` maps a DynamicVLA camera key to n_obs_steps uint8 HWC frames, oldest
        first; ``states`` is (n_obs_steps, 25), same order. The last entry is "now".
        """
        torch = self._torch
        batch: dict[str, Any] = {k: self._image_tensor(k, frames) for k, frames in images.items()}
        batch[STATE_KEY] = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self._device).unsqueeze(0)
        batch["task"] = [prompt]
        with torch.no_grad():
            chunk = self._policy.predict_action_chunk(batch)  # (1, T, A), unnormalized
        chunk = chunk[0].float().cpu().numpy()[:, : self.robot_action_dim]
        # Relative dims were trained as action - state(now); put the state back.
        chunk[:, self.relative_mask] += states[-1][self.relative_mask]
        return chunk


# --------------------------------------------------------------------------------------
# Observation history
# --------------------------------------------------------------------------------------


@dataclass
class _Frame:
    t: float
    jpegs: dict[str, bytes]  # wire camera key -> JPEG
    state: np.ndarray  # 25-D
    _decoded: dict[str, np.ndarray] = field(default_factory=dict)

    def image(self, wire_key: str) -> np.ndarray:
        # Decoded lazily: telemetry can arrive at the control rate, and most of it is
        # never picked as a history frame.
        if wire_key not in self._decoded:
            self._decoded[wire_key] = simplejpeg.decode_jpeg(self.jpegs[wire_key], colorspace="RGB")
        return self._decoded[wire_key]


class _History:
    """Per-connection buffer of recent frames, by arrival time."""

    def __init__(self, keep_s: float) -> None:
        self.keep_s = keep_s
        self.frames: deque[_Frame] = deque()
        # Seconds between the newest frame and the one received before it (None if none):
        # what a history lookup had to choose from, logged when it falls back.
        self.last_gap_s: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.last_gap_s = None

    def add(self, frame: _Frame) -> None:
        self.last_gap_s = frame.t - self.frames[-1].t if self.frames else None
        self.frames.append(frame)
        while self.frames and frame.t - self.frames[0].t > self.keep_s:
            self.frames.popleft()

    def pick(self, now: _Frame, offset_s: float, tolerance_s: float) -> tuple[_Frame, float | None]:
        """The buffered frame nearest ``now.t + offset_s``, or ``now`` itself when none is
        within ``tolerance_s``. Returns the frame and its age (None when it fell back)."""
        if offset_s == 0:
            return now, 0.0
        target = now.t + offset_s
        best = min(self.frames, key=lambda f: abs(f.t - target), default=None)
        if best is None or best is now or abs(best.t - target) > tolerance_s:
            return now, None
        return best, now.t - best.t


def _frame_from_wire(frame: dict[str, Any], t: float) -> _Frame | None:
    """A history frame from any wire message that carries joint state and every camera."""
    if JOINT_STATE_KEY not in frame or not all(k in frame for k in CAMERA_KEYS):
        return None
    state = _wire_state_25(np.asarray(frame[JOINT_STATE_KEY], dtype=np.float32))
    return _Frame(t=t, jpegs={k: frame[k] for k in CAMERA_KEYS}, state=state)


# --------------------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------------------


@dataclass
class ServerOptions:
    action_horizon: int = DEFAULT_ACTION_HORIZON
    task: str | None = None
    prompt_source: str = "auto"
    history_tolerance_s: float = DEFAULT_HISTORY_TOLERANCE_S
    action_limits: ActionLimits | None = None
    one_shot: bool = False


_WARNED_IMAGE_KEYS: set[str] = set()


def _check_image_shapes(policy: DynamicVLAChunkPolicy, frame: _Frame, req_id: int) -> None:
    for wire_key, key in policy.camera_keys.items():
        if key in _WARNED_IMAGE_KEYS:
            continue
        h, w = frame.image(wire_key).shape[:2]
        want = policy.trained_hw[key]
        if (h, w) != want:
            _WARNED_IMAGE_KEYS.add(key)
            log = LOGGER.warning if abs(w / h - want[1] / want[0]) > 0.01 else LOGGER.info
            log(
                "predict req=%d %s arrives %dx%d, trained at %dx%d (aspect %.3f vs %.3f); "
                "resizing to the trained resolution", req_id, key, h, w, *want, w / h, want[1] / want[0],
            )


def _inference_response(
    frame: dict[str, Any],
    arrival_s: float,
    history: _History,
    policy: DynamicVLAChunkPolicy,
    opts: ServerOptions,
    *,
    prompt: str,
) -> dict[str, Any]:
    req_id = next(_REQUEST_COUNTER)
    now = _frame_from_wire(frame, arrival_s)
    assert now is not None, "a validated inference frame carries joint state and every camera"
    history.add(now)
    _check_image_shapes(policy, now, req_id)

    picked: list[_Frame] = []
    ages: list[str] = []
    for offset_s in policy.offsets_s:
        f, age = history.pick(now, offset_s, opts.history_tolerance_s)
        picked.append(f)
        ages.append("now" if offset_s == 0 else ("MISSING" if age is None else f"{age * 1000:.0f}ms"))
    fell_back = "MISSING" in ages
    if fell_back:
        gap = history.last_gap_s
        ages.append(
            "(first frame of the episode)" if gap is None
            else f"(previous frame arrived {gap * 1000:.0f}ms earlier)"
        )

    images = {key: [f.image(wire_key) for f in picked] for wire_key, key in policy.camera_keys.items()}
    states = np.stack([f.state for f in picked])

    t0 = time.perf_counter()
    with _PREDICT_LOCK:
        chunk = policy.predict(images, states, prompt)
    inference_time_s = time.perf_counter() - t0

    state = now.state
    action = chunk[: opts.action_horizon]
    if opts.action_limits is not None:
        action, report = limit_chunk(action, state, opts.action_limits, committed_prefix_len=0)
        if report.triggered:
            LOGGER.warning("predict req=%d ACTION LIMITER fired: %s", req_id, report.describe(GEN2_DIM_NAMES))

    (LOGGER.warning if fell_back else LOGGER.info)(
        "predict req=%d prompt=%r history=[%s]%s chunk=%d returned=%d inference_s=%.3f "
        "||action[0]-state||=%.4f",
        req_id, prompt, " ".join(ages),
        " (no frame near the offset: current frame repeated)" if fell_back else "",
        len(chunk), len(action), inference_time_s, float(np.linalg.norm(action[0] - state)),
    )
    resp = {
        ACTION_KEY: np.ascontiguousarray(action, dtype=np.float32),
        INFERENCE_TIME_KEY: inference_time_s,
        POLICY_ID_KEY: policy.policy_id,
        RL_ENABLED_KEY: False,
        Q_VALUE_KEY: 0.0,
    }
    if CHUNK_ID_KEY in frame:
        resp[CHUNK_ID_KEY] = frame[CHUNK_ID_KEY]
    validate_wire_inference_response(resp)
    return resp


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL, include_image_resolution=True, server_features=server_features,
    )


async def handle_inference_connection(
    connection: ServerConnection,
    policy: DynamicVLAChunkPolicy,
    opts: ServerOptions,
    *,
    server_features: Iterable[str | ServerFeature] = (),
) -> None:
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))

    oldest_offset_s = -min(policy.offsets_s)
    history = _History(keep_s=oldest_offset_s + opts.history_tolerance_s + 0.5)
    last_prompt: str | None = None
    last_metadata: str | None = None
    held_prompt: str | None = None
    telemetry_frames = 0

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
        arrival_s = time.perf_counter()
        assert isinstance(message, bytes), type(message)
        frame = deserialize_from_msgpack(message)
        if not isinstance(frame, dict):
            await connection.send(serialize_to_msgpack({"error": "expected dict frame"}))
            continue
        endpoint = frame.get(ENDPOINT_KEY)
        if endpoint == ENDPOINT_DONE:
            # A new episode must not see the last one's frames as its history.
            history.reset()
            held_prompt = None
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue
        if endpoint == ENDPOINT_INTERVENTION:
            # The frames stay: they are real observations whoever was driving.
            await connection.send(serialize_to_msgpack({STATUS_KEY: "ok"}))
            continue
        if endpoint is not None:
            if endpoint == ENDPOINT_TELEMETRY:
                observed = _frame_from_wire(frame, arrival_s)
                if observed is not None:
                    history.add(observed)
                    telemetry_frames += 1
                    if telemetry_frames == 1:
                        LOGGER.info("telemetry frames carry images: using them as observation history")
            else:
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
        resp = await asyncio.to_thread(
            _inference_response, frame, arrival_s, history, policy, opts, prompt=prompt,
        )
        await connection.send(serialize_to_msgpack(resp))
        if opts.one_shot:
            LOGGER.info("one-shot mode: sent one inference response; ignoring further requests")
            await asyncio.Future()


@asynccontextmanager
async def run_server(
    policy: DynamicVLAChunkPolicy,
    opts: ServerOptions,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    server_features: Iterable[str | ServerFeature] = (ServerFeature.REWARDS,),
) -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        LOGGER.info("connection from %s", connection.remote_address)
        await handle_inference_connection(connection, policy, opts, server_features=server_features)

    async with websockets.serve(handler, host, port, max_size=None) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://{host}:{port}/ws"


def _warmup(policy: DynamicVLAChunkPolicy) -> None:
    """The first forward pass pays for CUDA kernel selection; do it before the robot connects."""
    h, w = DEFAULT_HARDWARE_MODEL.image_resolution
    n = len(policy.offsets_s)
    images = {key: [np.zeros((h, w, 3), dtype=np.uint8)] * n for key in policy.camera_keys.values()}
    states = np.zeros((n, policy.robot_action_dim), dtype=np.float32)
    t0 = time.perf_counter()
    policy.predict(images, states, "warmup")
    t1 = time.perf_counter()
    policy.predict(images, states, "warmup")
    LOGGER.info("warmup: first call %.2fs, second %.3fs", t1 - t0, time.perf_counter() - t1)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy inference server with a DynamicVLA backend.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_INFERENCE_SERVER_PORT, help="Bind port")
    parser.add_argument("--checkpoint-path", type=Path, required=True,
                        help="DynamicVLA checkpoint directory (config.json + model.epochNNNN.safetensors).")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Which .safetensors to load (default: the only one in --checkpoint-path).")
    parser.add_argument("--dynamicvla-root", type=Path, default=DEFAULT_DYNAMICVLA_ROOT,
                        help=f"DynamicVLA checkout providing policies.dynamicvla (default: {DEFAULT_DYNAMICVLA_ROOT}).")
    parser.add_argument("--dataset-fps", type=float, default=DEFAULT_DATASET_FPS,
                        help="Frame rate of the training dataset; converts the config's observation "
                             f"offsets from frames to seconds (default: {DEFAULT_DATASET_FPS:g}).")
    parser.add_argument("--history-tolerance-s", type=float, default=DEFAULT_HISTORY_TOLERANCE_S, metavar="S",
                        help="How far a buffered frame's age may be from an observation offset and "
                             f"still be used for it (default: {DEFAULT_HISTORY_TOLERANCE_S:g}).")
    parser.add_argument("--absolute-action-dims", type=int, nargs="*", default=None, metavar="DIM",
                        help="Action dims trained absolute (not relative to state). Default: inferred "
                             "from the checkpoint's normalization stats.")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON,
                        help=f"Action rows returned per request (default: {DEFAULT_ACTION_HORIZON}).")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Flow-matching denoise steps (default: the checkpoint's num_steps).")
    parser.add_argument("--task", default=None,
                        help="Task prompt for every request. If omitted, read from each request.")
    parser.add_argument("--prompt-source", type=_check_prompt_source, default="auto", metavar="SOURCE",
                        help="Where to read the instruction when --task is not given; same as "
                             f"server.server: 'auto' (a {'/'.join(_PROMPT_FIELD_NAMES)} field in "
                             "the metadata, else subtask, else task), 'task', 'subtask', or a "
                             "dotted path into the frame.")
    parser.add_argument("--no-action-limiter", action="store_true",
                        help="Disable the per-chunk joint limits (on by default; see server.server).")
    parser.add_argument("--max-chunk-start-offset-deg", type=float, default=DEFAULT_MAX_START_OFFSET_DEG,
                        metavar="DEG", help=f"(default: {DEFAULT_MAX_START_OFFSET_DEG:g}; 0 disables)")
    parser.add_argument("--max-step-deg", type=float, default=DEFAULT_MAX_STEP_DEG, metavar="DEG",
                        help=f"(default: {DEFAULT_MAX_STEP_DEG:g}; 0 disables)")
    parser.add_argument("--limit-dims", nargs="*", default=[], metavar="DIM",
                        help="Action dims the limits apply to. Default: every joint dim, no grippers.")
    parser.add_argument("--no-rewards", action="store_true",
                        help="Do not advertise reward support in the handshake.")
    parser.add_argument("--one-shot", action="store_true",
                        help="Send one chunk per connection, then stop responding.")
    return parser.parse_args(argv)


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
    policy = DynamicVLAChunkPolicy(
        args.checkpoint_path,
        dynamicvla_root=args.dynamicvla_root,
        weights=args.weights,
        dataset_fps=args.dataset_fps,
        num_steps=args.num_steps,
        absolute_action_dims=args.absolute_action_dims,
    )
    assert 1 <= args.action_horizon <= policy.action_horizon, (
        f"--action-horizon must be in [1, {policy.action_horizon}], got {args.action_horizon}"
    )
    _warmup(policy)
    opts = ServerOptions(
        action_horizon=args.action_horizon,
        task=args.task,
        prompt_source=args.prompt_source,
        history_tolerance_s=args.history_tolerance_s,
        action_limits=_action_limits_from_args(args),
        one_shot=args.one_shot,
    )
    if opts.action_limits is None:
        LOGGER.warning("ACTION LIMITER OFF: chunks go to the robot exactly as the policy emits them")
    features = () if args.no_rewards else (ServerFeature.REWARDS,)
    async with run_server(policy, opts, args.host, args.port, server_features=features) as url:
        print(
            f"Server listening on {url} (dynamicvla, action_horizon={args.action_horizon}, "
            f"obs offsets {policy.frame_offsets} frames, "
            f"prompt={'--task ' + repr(args.task) if args.task else 'from requests'})",
            flush=True,
        )
        if args.host == "0.0.0.0":
            # 0.0.0.0 is a bind address, not one a client can dial.
            for ip in _list_lan_ips():
                print(f"  reachable at ws://{ip}:{args.port}/ws", flush=True)
        await asyncio.Future()
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        raise SystemExit(asyncio.run(_run_cli(argv)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
