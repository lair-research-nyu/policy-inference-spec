# START SERVER:
#  /home/rooholla/miniconda3/envs/lerobot/bin/python -m server.minimal --host 0.0.0.0 --port 18090

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import socket
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable, Sequence, TextIO

import numpy as np
import torch
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.codec import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.debug import ChunkTransform, DeleteAndBackpad, SetConstant
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    server_handshake_for_hardware_model,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)
from policy_inference_spec.protocol import (
    ACTION_KEY,
    CONTEXT_EMBEDDINGS_KEY,
    CONTEXT_EMBEDDING_TOKENS,
    CONTEXT_EMBEDDING_WIDTH,
    DEFAULT_INFERENCE_SERVER_PORT,
    ENDPOINT_KEY,
    ENDPOINT_RESET,
    ENDPOINT_REWARD,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PROMPT_KEY,
    REWARD_DESCRIPTION_KEY,
    REWARDS_H_KEY,
    STATUS_KEY,
    TIMESTAMP_KEY,
    RewardSignal,
    ServerFeature,
    ServerHandshake,
)

import hashlib
import time
from itertools import count
import simplejpeg
from pathlib import Path
from torch.nn.functional import interpolate
from lerobot.policies.factory import make_pre_post_processors

_REQUEST_COUNTER = count()
_CONNECTION_COUNTER = count()

_CSV_FIELDS = (
    "connection_id",
    "req",
    "client_ts",
    "recv_ts",
    "send_ts",
    "offset_recv_minus_client",
    "d_client",
    "d_recv",
    "d_offset",
    "prev_send_to_client_ts",
    "server_processing",
    "d_send",
    "inference_time",
    "idle",
)


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
# CHECKPOINT_PATH = Path("models/pretrained-model-act-cubeinbox-te-065000")  # adjust to your path
# CHECKPOINT_PATH = Path("models/pretrained-model-act-cubeinbox-512imgs-novae-105000")  # adjust to your path
CHECKPOINT_PATH = Path("models/pretrained_model-hri-ah-100000")
# CHECKPOINT_PATH = Path("models/pretrained_model-cube-in-box2-140000")  # adjust to your path

# ── Policy type: change to "act" or "pi0" / "pi0_fast" ──────────────────────
POLICY_TYPE = "pi0"

# Optional image resize: set to (H, W) to square-pad then bilinear-resize all
# camera images before passing to the policy. None = no resize (pass as-is).
# IMAGE_SIZE: tuple[int, int] | None = (512, 512)

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


def _load_policy(checkpoint_path: Path, policy_type: str):
    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy.from_pretrained(str(checkpoint_path))
    elif policy_type in ("pi0", "pi0_fast"):
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy.from_pretrained(str(checkpoint_path))
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


print("Loading policy...", flush=True)
POLICY = _load_policy(CHECKPOINT_PATH, POLICY_TYPE).eval().cuda()
print("Policy loaded.", flush=True)
print("Loading dataset stats...", flush=True)
_DATASET_STATS = _load_dataset_stats(CHECKPOINT_PATH)
print("Dataset stats loaded.", flush=True)
print("Building pre/post processors...", flush=True)
_PREPROCESSOR, _POSTPROCESSOR = make_pre_post_processors(
    policy_cfg=POLICY,
    pretrained_path=str(CHECKPOINT_PATH),
    dataset_stats=_DATASET_STATS,
    preprocessor_overrides={"device_processor": {"device": str(POLICY.config.device)}},
)
print("Processors built.", flush=True)
print("Resetting preprocessor...", flush=True)
_PREPROCESSOR.reset()
print("Resetting postprocessor...", flush=True)
_POSTPROCESSOR.reset()
print("Init complete.", flush=True)

# Map GEN2 wire frame camera keys → lerobot observation keys
# Adjust the values to match the keys your policy was trained with.
_CAMERA_KEY_MAP = {
    "observation/images/main_image":        "observation.images.head",
    "observation/images/left_wrist_image":  "observation.images.left_wrist",
    "observation/images/right_wrist_image": "observation.images.right_wrist",
}

POLICY_ID = str(CHECKPOINT_PATH)
LOGGER = logging.getLogger(__name__)


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=True,
        server_features=server_features
    )


def _decode_jpeg_to_tensor(
    jpeg_bytes: bytes,
    device: str,
    # target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """JPEG bytes → float32 CHW tensor on device, values in [0, 1].

    If target_size=(H, W) is given, square-pads the image first then bilinearly
    resizes to (H, W) before returning.
    """
    img_hwc = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")  # uint8 HWC
    img = torch.from_numpy(img_hwc.astype("float32") / 255.0).permute(2, 0, 1)  # CHW
    # if target_size is not None:
    #     _, H, W = img.shape
    #     max_side = max(H, W)
    #     padded = torch.zeros((3, max_side, max_side), dtype=img.dtype)
    #     r, c = (max_side - H) // 2, (max_side - W) // 2
    #     padded[:, r : r + H, c : c + W] = img
    #     img = interpolate(
    #         padded.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    #     ).squeeze(0)
    return img.to(device)


def _inference_response(
    frame: dict[str, Any],
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    prompt: str = "",
    transforms: Sequence[ChunkTransform] = (),
    idle: bool = False,
    ensemble_buffer: TemporalEnsembleBuffer | None = None,
) -> dict[str, Any]:
    device = str(POLICY.config.device)
    req_id = next(_REQUEST_COUNTER)

    print(f"frame keys: {sorted(frame.keys())}", flush=True)

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
        "predict req=%d raw_state_hash=%s state_25=%s images=[%s]",
        req_id, raw_state_hash, state_25_str, image_fingerprints,
    )

    obs: dict[str, Any] = {
        "observation.state": torch.from_numpy(state_25).unsqueeze(0).to(device),
    }
    for wire_key, lerobot_key in _CAMERA_KEY_MAP.items():
        obs[lerobot_key] = _decode_jpeg_to_tensor(frame[wire_key], device).unsqueeze(0)

    if POLICY_TYPE in ("pi0", "pi0_fast"):
        obs["task"] = [prompt]

    obs_processed = _PREPROCESSOR(obs)

    # Verify normalized state is in-distribution (should be O(1), not O(100+)).
    norm_state = obs_processed["observation.state"].squeeze(0).cpu().float()
    norm_state_abs_max = float(norm_state.abs().max())
    norm_state_str = np.array2string(norm_state.numpy(), precision=2, suppress_small=True, max_line_width=400)
    LOGGER.info(
        "predict req=%d norm_state_abs_max=%.2f norm_state=%s",
        req_id, norm_state_abs_max, norm_state_str,
    )
    if norm_state_abs_max > 10.0:
        LOGGER.warning(
            "predict req=%d POSSIBLE STATE MISMATCH: normalized state has values up to %.1f sigma "
            "(expected O(1) for in-distribution input; raw state may not match training features)",
            req_id, norm_state_abs_max,
        )

    queue_before = _policy_queue_info(POLICY)
    t0 = time.perf_counter()
    with torch.no_grad():
        actions = [POLICY.select_action(obs_processed) for _ in range(action_horizon)]
        action = _POSTPROCESSOR(torch.stack(actions, dim=0)).squeeze(1).cpu().numpy().astype("float32")
    inference_time_s = time.perf_counter() - t0
    queue_after = _policy_queue_info(POLICY)
    action_0_str = np.array2string(action[0], precision=3, suppress_small=True, max_line_width=400)
    action_last_str = np.array2string(action[-1], precision=3, suppress_small=True, max_line_width=400)
    delta_first = float(np.linalg.norm(action[0] - state_25))
    LOGGER.info(
        "predict req=%d queue_before=%s queue_after=%s inference_s=%.3f",
        req_id, queue_before, queue_after, inference_time_s,
    )
    LOGGER.info(
        "predict req=%d action[0]=%s action[-1]=%s ||action[0]-state_25||=%.4f",
        req_id, action_0_str, action_last_str, delta_first,
    )

    if idle:
        action = np.tile(state_25.astype("float32"), (action_horizon, 1))
        LOGGER.info(
            "predict req=%d idle=True policy_action_discarded action=tile(state_25, %d)",
            req_id, action_horizon,
        )

    for i, transform in enumerate(transforms):
        action = transform(action)
        assert action.shape[1] == 25 and action.dtype == np.float32, (
            f"transform[{i}] produced shape={action.shape} dtype={action.dtype}"
        )

    if ensemble_buffer is not None:
        action = ensemble_buffer.add_and_ensemble(action)
        LOGGER.info("predict req=%d temporal_ensemble applied", req_id)

    context_embeddings = np.zeros(
        (CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32
    )
    resp = {
        ACTION_KEY: action,
        CONTEXT_EMBEDDINGS_KEY: context_embeddings,
        INFERENCE_TIME_KEY: inference_time_s,
        POLICY_ID_KEY: POLICY_ID,
    }
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(
    connection: ServerConnection,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    server_features: Iterable[str | ServerFeature] = (),
    one_shot: bool = False,
    transforms: Sequence[ChunkTransform] = (),
    idle: bool = False,
    temporal_ensemble_coeff: float | None = None,
    csv_writer: csv.writer | None = None,
    csv_file: TextIO | None = None,
    save_imgs: Path | None = None,
) -> None:

    assert action_horizon >= 1, f"action_horizon must be positive, got {action_horizon}"
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))

    connection_id = next(_CONNECTION_COUNTER)
    ensemble_buffer = (
        TemporalEnsembleBuffer(action_horizon, DEFAULT_HARDWARE_MODEL.action_dim, temporal_ensemble_coeff)
        if temporal_ensemble_coeff is not None
        else None
    )
    prev_client_timestamp: float | None = None
    prev_recv_time: float | None = None
    prev_send_time: float | None = None
    prev_offset: float | None = None
    request_index = 0

    async for message in connection:
        assert isinstance(message, bytes), type(message)
        frame = deserialize_from_msgpack(message)
        if not isinstance(frame, dict):
            await connection.send(serialize_to_msgpack({"error": "expected dict frame"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_RESET:
            POLICY.reset()
            _PREPROCESSOR.reset()
            _POSTPROCESSOR.reset()
            if ensemble_buffer is not None:
                ensemble_buffer.reset()
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_TELEMETRY:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_REWARD:
            reward_signal = RewardSignal.from_payload(frame)
            await connection.send(
                serialize_to_msgpack(
                    {
                        ENDPOINT_KEY: ENDPOINT_REWARD,
                        STATUS_KEY: "ok",
                        REWARDS_H_KEY: list(reward_signal.rewards_h),
                        **(
                            {REWARD_DESCRIPTION_KEY: reward_signal.description}
                            if reward_signal.description is not None
                            else {}
                        ),
                    }
                )
            )
            continue
        validate_wire_inference_request_frame(frame)
        _ = frame[PROMPT_KEY]
        _ = frame[MODEL_ID_KEY]
        if save_imgs is not None:
            for wire_key in _CAMERA_KEY_MAP:
                camera_short = wire_key.rsplit("/", 1)[-1].removesuffix("_image")
                img_path = save_imgs / f"conn{connection_id:03d}_req{request_index:05d}_{camera_short}.jpg"
                img_data = frame[wire_key]
                assert isinstance(img_data, bytes), (
                    f"expected JPEG bytes for {wire_key}, got {type(img_data)}"
                )
                img_path.write_bytes(img_data)
        recv_time = time.time()
        client_timestamp_raw = frame.get(TIMESTAMP_KEY)
        # Client sends timestamp in nanoseconds since epoch; convert to seconds.
        client_timestamp = float(client_timestamp_raw) * 1e-9 if client_timestamp_raw is not None else None

        def _fmt(value: float | None, spec: str = "+.6f") -> str:
            return format(value, spec) if value is not None else "n/a"

        offset = (recv_time - client_timestamp) if client_timestamp is not None else None
        d_client = (
            client_timestamp - prev_client_timestamp
            if (client_timestamp is not None and prev_client_timestamp is not None)
            else None
        )
        d_recv = recv_time - prev_recv_time if prev_recv_time is not None else None
        d_offset = (
            offset - prev_offset if (offset is not None and prev_offset is not None) else None
        )
        gap_send_to_next_obs = (
            client_timestamp - prev_send_time
            if (client_timestamp is not None and prev_send_time is not None)
            else None
        )
        print(
            f"req={request_index} "
            f"client_ts={_fmt(client_timestamp, '.6f')} "
            f"recv_ts={recv_time:.6f} "
            f"offset(recv-client)={_fmt(offset)} "
            f"d_client={_fmt(d_client)} "
            f"d_recv={_fmt(d_recv)} "
            f"d_offset={_fmt(d_offset)} "
            f"prev_send_to_client_ts={_fmt(gap_send_to_next_obs)}",
            flush=True,
        )

        resp = _inference_response(
            frame,
            action_horizon=action_horizon,
            prompt="place the orange cube in the box",
            transforms=transforms,
            idle=idle,
            ensemble_buffer=ensemble_buffer,
        )
        send_time = time.time()
        d_send = send_time - prev_send_time if prev_send_time is not None else None
        print(
            f"req={request_index} "
            f"send_ts={send_time:.6f} "
            f"server_processing={send_time - recv_time:+.6f}s "
            f"d_send={_fmt(d_send)}",
            flush=True,
        )
        if csv_writer is not None:
            csv_writer.writerow([
                connection_id,
                request_index,
                "" if client_timestamp is None else f"{client_timestamp:.9f}",
                f"{recv_time:.9f}",
                f"{send_time:.9f}",
                "" if offset is None else f"{offset:.9f}",
                "" if d_client is None else f"{d_client:.9f}",
                "" if d_recv is None else f"{d_recv:.9f}",
                "" if d_offset is None else f"{d_offset:.9f}",
                "" if gap_send_to_next_obs is None else f"{gap_send_to_next_obs:.9f}",
                f"{send_time - recv_time:.9f}",
                "" if d_send is None else f"{d_send:.9f}",
                f"{float(resp[INFERENCE_TIME_KEY]):.9f}",
                int(idle),
            ])
            if csv_file is not None:
                csv_file.flush()
        await connection.send(serialize_to_msgpack(resp))
        prev_client_timestamp = client_timestamp
        prev_recv_time = recv_time
        prev_send_time = send_time
        prev_offset = offset
        request_index += 1
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
    transforms: Sequence[ChunkTransform] = (),
    idle: bool = False,
    temporal_ensemble_coeff: float | None = None,
    csv_log: Path | None = None,
    save_imgs: Path | None = None,
) -> AsyncIterator[str]:

    csv_file: TextIO | None = None
    csv_writer: csv.writer | None = None
    if csv_log is not None:
        write_header = not csv_log.exists() or csv_log.stat().st_size == 0
        csv_file = open(csv_log, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(_CSV_FIELDS)
            csv_file.flush()

    if save_imgs is not None:
        save_imgs.mkdir(parents=True, exist_ok=True)

    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(
            connection,
            action_horizon=action_horizon,
            server_features=server_features,
            one_shot=one_shot,
            transforms=transforms,
            idle=idle,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            csv_writer=csv_writer,
            csv_file=csv_file,
            save_imgs=save_imgs,
        )

    try:
        async with websockets.serve(handler, host, port) as server:
            sock = next(iter(server.sockets))
            port = sock.getsockname()[1]
            yield f"ws://{host}:{port}/ws"
    finally:
        if csv_file is not None:
            csv_file.close()


def _parse_set_constant(spec: str) -> tuple[float, list[int] | None]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    assert parts, "--set-constant requires at least a value (e.g. '0.0' or '0.0,0,1')"
    value = float(parts[0])
    dims: list[int] | None = [int(p) for p in parts[1:]] if len(parts) > 1 else None
    return value, dims


def _build_transforms(args: argparse.Namespace) -> tuple[list[ChunkTransform], str]:
    transforms: list[ChunkTransform] = []
    labels: list[str] = []
    if args.delete_first is not None:
        transforms.append(DeleteAndBackpad(args.delete_first))
        labels.append(f"delete{args.delete_first}")
    if args.set_constant is not None:
        value, dims = args.set_constant
        transforms.append(SetConstant(value, dims=dims))
        dims_label = "all" if dims is None else ",".join(str(d) for d in dims)
        labels.append(f"const{value}@{dims_label}")
    return transforms, "+".join(labels) if labels else "none"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal policy inference example server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_INFERENCE_SERVER_PORT, help="Bind port")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_REPLAY_ACTION_HORIZON,
        help="Number of action rows to emit per prediction. Defaults to 50 to match replay_rrd defaults.",
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
        "--delete-first",
        type=int,
        default=None,
        metavar="N",
        help="Drop first N actions and back-pad with action N (DeleteAndBackpad transform).",
    )
    parser.add_argument(
        "--set-constant",
        type=_parse_set_constant,
        default=None,
        metavar="VALUE[,DIM,DIM...]",
        help="Pin action dims to VALUE; dims omitted = all dims (SetConstant transform).",
    )
    parser.add_argument(
        "--idle",
        action="store_true",
        help="Skip the policy and emit the current 25-dim state tiled action_horizon times (robot holds position).",
    )
    parser.add_argument(
        "--csv-log",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append per-request timing fields to this CSV file (header written if file is new/empty).",
    )
    parser.add_argument(
        "--save-imgs",
        type=Path,
        default=None,
        metavar="DIR",
        help="Dump all three JPEG images to DIR on every inference request "
             "(filename: connXXX_reqYYYYY_{main,left_wrist,right_wrist}.jpg).",
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
    return parser.parse_args(argv)


def _cli_server_features(no_rewards: bool) -> tuple[ServerFeature, ...]:
    if no_rewards:
        return ()
    return (ServerFeature.REWARDS,)


async def _run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    assert args.action_horizon >= 1, f"action_horizon must be positive, got {args.action_horizon}"
    server_features = _cli_server_features(args.no_rewards)
    transforms, transforms_label = _build_transforms(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    async with run_example_server(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        server_features=server_features,
        one_shot=args.one_shot,
        transforms=transforms,
        idle=args.idle,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        csv_log=args.csv_log,
        save_imgs=args.save_imgs,
    ) as url:
        mode_suffix = " one_shot=True" if args.one_shot else ""
        if args.idle:
            mode_suffix += " idle=True"
        if args.temporal_ensemble_coeff is not None:
            mode_suffix += f" temporal_ensemble_coeff={args.temporal_ensemble_coeff}"
        if args.csv_log is not None:
            mode_suffix += f" csv_log={args.csv_log}"
        if args.save_imgs is not None:
            mode_suffix += f" save_imgs={args.save_imgs}"
        print(
            f"Server listening on {url} (action_horizon={args.action_horizon} "
            f"transforms={transforms_label}){mode_suffix}",
            flush=True,
        )
        if args.host == "0.0.0.0":
            lan_ips = _list_lan_ips()
            if lan_ips:
                for ip in lan_ips:
                    print(f"  reachable at ws://{ip}:{args.port}/ws", flush=True)
            else:
                print("  (could not detect LAN IP; check `ip addr`)", flush=True)
        LOGGER.info(
            "Server listening on %s action_horizon=%d transforms=%s",
            url, args.action_horizon, transforms_label,
        )
        await asyncio.Future()
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        raise SystemExit(asyncio.run(_run_cli(argv)))
    except KeyboardInterrupt:
        raise SystemExit(130)


__all__ = [
    "EXAMPLE_POLICY_ID",
    "example_policy_actions",
    "handle_inference_connection",
    "main",
    "run_example_server",
    "server_handshake_config",
]


if __name__ == "__main__":
    main()