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
    RewardSignal,
    ServerFeature,
    ServerHandshake,
)

import hashlib
import time
from itertools import count
import simplejpeg
from pathlib import Path
from lerobot.policies.factory import make_pre_post_processors

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
DEFAULT_POLICY_TYPE = "pi0"  # one of: "act", "pi0", "pi0_fast"

# Populated by _init_policy() once the checkpoint path / policy type are known.
POLICY: Any = None
POLICY_TYPE: str | None = None
POLICY_ID: str | None = None
_DATASET_STATS: dict | None = None
_PREPROCESSOR: Any = None
_POSTPROCESSOR: Any = None


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


# Map GEN2 wire frame camera keys → lerobot observation keys
# Adjust the values to match the keys your policy was trained with.
_CAMERA_KEY_MAP = {
    "observation/images/main_image":        "observation.images.head",
    "observation/images/left_wrist_image":  "observation.images.left_wrist",
    "observation/images/right_wrist_image": "observation.images.right_wrist",
}

LOGGER = logging.getLogger(__name__)


def _init_policy(checkpoint_path: Path, policy_type: str) -> None:
    """Load the policy + processors for the given checkpoint and populate module globals."""
    global POLICY, POLICY_TYPE, POLICY_ID, _DATASET_STATS, _PREPROCESSOR, _POSTPROCESSOR
    POLICY_TYPE = policy_type
    POLICY_ID = str(checkpoint_path)
    print("Loading policy...", flush=True)
    POLICY = _load_policy(checkpoint_path, policy_type).eval().cuda()
    print("Policy loaded.", flush=True)
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


def _inference_response(
    frame: dict[str, Any],
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    prompt: str = "",
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
) -> None:
    
    assert action_horizon >= 1, f"action_horizon must be positive, got {action_horizon}"
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))
    
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
        resp = _inference_response(frame, action_horizon=action_horizon, prompt="place the orange cube in the box")
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
) -> AsyncIterator[str]:
    
    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(
            connection,
            action_horizon=action_horizon,
            server_features=server_features,
            one_shot=one_shot,
        )

    async with websockets.serve(handler, host, port) as server:
        sock = next(iter(server.sockets))
        port = sock.getsockname()[1]
        yield f"ws://{host}:{port}/ws"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal policy inference example server.")
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
        choices=("act", "pi0", "pi0_fast"),
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
        "--no-rewards",
        action="store_true",
        help="Do not advertise reward support in the handshake.",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Send one full action chunk per connection, then stop responding to further requests.",
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    _init_policy(args.checkpoint_path, args.policy_type)
    async with run_example_server(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        server_features=server_features,
        one_shot=args.one_shot,
    ) as url:
        mode_suffix = " one_shot=True" if args.one_shot else ""
        print(f"Server listening on {url} (action_horizon={args.action_horizon}){mode_suffix}", flush=True)
        if args.host == "0.0.0.0":
            lan_ips = _list_lan_ips()
            if lan_ips:
                for ip in lan_ips:
                    print(f"  reachable at ws://{ip}:{args.port}/ws", flush=True)
            else:
                print("  (could not detect LAN IP; check `ip addr`)", flush=True)
        LOGGER.info("Server listening on %s action_horizon=%d", url, args.action_horizon)
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
