from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import Sequence
from typing import Any, AsyncIterator, Iterable

import numpy as np
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.codec import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.debug import (
    DeleteAndBackpad,
    Gate,
    GoToPoseSource,
    ImmediateGate,
    KeypressGate,
    RerunObserver,
    RrdActionSource,
    SetConstant,
    ZerosSource,
    run_pipeline,
)
from policy_inference_spec.debug.pipeline import ActionSource, ChunkTransform, ResponseObserver
from policy_inference_spec.feature_engineering import SchemaName
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
    POLICY_ID_KEY,
    REWARDS_H_KEY,
    REWARD_DESCRIPTION_KEY,
    STATUS_KEY,
    RewardSignal,
    ServerFeature,
    ServerHandshake,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_ACTION_HORIZON = 50


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=True,
        server_features=server_features,
    )


def _build_response(chunk: np.ndarray, *, source_name: str) -> dict[str, Any]:
    context_embeddings = np.zeros(
        (CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32
    )
    resp: dict[str, Any] = {
        ACTION_KEY: chunk,
        CONTEXT_EMBEDDINGS_KEY: context_embeddings,
        POLICY_ID_KEY: f"debug:{source_name}",
    }
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(
    connection: ServerConnection,
    *,
    source: ActionSource,
    source_name: str,
    transforms: Sequence[ChunkTransform] = (),
    gate: Gate | None = None,
    observer: ResponseObserver | None = None,
    server_features: Iterable[str | ServerFeature] = (),
) -> None:
    if gate is None:
        gate = ImmediateGate()
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))
    async for message in connection:
        assert isinstance(message, bytes), type(message)
        frame = deserialize_from_msgpack(message)
        if not isinstance(frame, dict):
            await connection.send(serialize_to_msgpack({"error": "expected dict frame"}))
            continue
        endpoint = frame.get(ENDPOINT_KEY)
        if endpoint == ENDPOINT_RESET:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if endpoint == ENDPOINT_TELEMETRY:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if endpoint == ENDPOINT_REWARD:
            reward_signal = RewardSignal.from_payload(frame)
            payload: dict[str, Any] = {
                ENDPOINT_KEY: ENDPOINT_REWARD,
                STATUS_KEY: "ok",
                REWARDS_H_KEY: list(reward_signal.rewards_h),
            }
            if reward_signal.description is not None:
                payload[REWARD_DESCRIPTION_KEY] = reward_signal.description
            await connection.send(serialize_to_msgpack(payload))
            continue
        validate_wire_inference_request_frame(frame)
        await gate.wait_for_release()
        chunk = run_pipeline(frame, source, transforms=transforms)
        if observer is not None:
            observer(frame, chunk)
        resp = _build_response(chunk, source_name=source_name)
        await connection.send(serialize_to_msgpack(resp))


@asynccontextmanager
async def run_debug_server(
    source: ActionSource,
    *,
    source_name: str,
    transforms: Sequence[ChunkTransform] = (),
    gate: Gate | None = None,
    observer: ResponseObserver | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    server_features: Iterable[str | ServerFeature] = (),
) -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(
            connection,
            source=source,
            source_name=source_name,
            transforms=transforms,
            gate=gate,
            observer=observer,
            server_features=server_features,
        )

    async with websockets.serve(handler, host, port) as server:
        sock = next(iter(server.sockets))
        bound_port = sock.getsockname()[1]
        yield f"ws://{host}:{bound_port}/ws"


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


def _build_gate(args: argparse.Namespace) -> tuple[Gate, str]:
    if args.gate == "none":
        return ImmediateGate(), "none"
    if args.gate == "keypress":
        return KeypressGate(), "keypress"
    raise ValueError(f"Unsupported gate: {args.gate!r}")


def _build_observer(
    args: argparse.Namespace,
) -> tuple[ResponseObserver | None, str]:
    if args.observer == "none":
        return None, "none"
    if args.observer == "rerun":
        return RerunObserver(app_id=args.rerun_app_id, hz=args.hz), "rerun"
    raise ValueError(f"Unsupported observer: {args.observer!r}")


def _build_source(args: argparse.Namespace) -> tuple[ActionSource, str]:
    if args.source == "zeros":
        return ZerosSource(horizon=args.action_horizon), "zeros"
    if args.source == "rrd":
        assert args.recording_path is not None, (
            "--recording-path is required when --source=rrd"
        )
        src = RrdActionSource(
            recording_path=args.recording_path,
            horizon=args.action_horizon,
            schema=args.schema,
            hz=args.hz,
        )
        return src, f"rrd:{args.recording_path.name}"
    if args.source == "goto":
        assert args.target_pose_file is not None, (
            "--target-pose-file is required when --source=goto"
        )
        assert args.target_pose_file.is_file(), (
            f"target pose file not found: {args.target_pose_file}"
        )
        target = np.load(args.target_pose_file)
        max_step = (
            args.max_step_per_joint_rad
            if args.max_step_per_joint_rad is not None and args.max_step_per_joint_rad > 0
            else None
        )
        src = GoToPoseSource(
            target_pose=target,
            horizon=args.action_horizon,
            max_step_per_joint_rad=max_step,
        )
        return src, f"goto:{args.target_pose_file.name}"
    raise ValueError(f"Unsupported source: {args.source!r}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug policy inference server with a pluggable action source."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_INFERENCE_SERVER_PORT)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--source", default="zeros", choices=("zeros", "rrd", "goto"))
    parser.add_argument(
        "--recording-path",
        type=Path,
        default=None,
        help="Path to .rrd recording (required with --source=rrd)",
    )
    parser.add_argument(
        "--schema",
        type=SchemaName,
        default=SchemaName.GEN2_32D_STATE,
        help="Feature schema for rrd replay (default: gen2-32d-state)",
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=50,
        help="Native sample rate of the recording (default: 50)",
    )
    parser.add_argument(
        "--target-pose-file",
        type=Path,
        default=None,
        help="Path to .npy file with 25 floats (required with --source=goto)",
    )
    parser.add_argument(
        "--max-step-per-joint-rad",
        type=float,
        default=0.5,
        help="Max |target-state| per joint for goto source; <=0 disables (default: 0.5 rad)",
    )
    parser.add_argument(
        "--delete-first",
        type=int,
        default=None,
        metavar="N",
        help="Drop first N actions and back-pad with action N (DeleteAndBackpad transform)",
    )
    parser.add_argument(
        "--set-constant",
        type=_parse_set_constant,
        default=None,
        metavar="VALUE[,DIM,DIM...]",
        help="Pin action dims to VALUE; dims omitted = all dims (SetConstant transform)",
    )
    parser.add_argument(
        "--gate",
        default="none",
        choices=("none", "keypress"),
        help="Pacing gate: 'none' releases immediately; 'keypress' pauses each "
        "response until Enter is pressed in the server terminal",
    )
    parser.add_argument(
        "--observer",
        default="none",
        choices=("none", "rerun"),
        help="Response observer: 'none' disables; 'rerun' spawns a live Rerun "
        "viewer logging cameras, right-arm state, and predicted actions",
    )
    parser.add_argument(
        "--rerun-app-id",
        default="policy_inference_debug",
        help="Rerun app id when --observer=rerun (default: policy_inference_debug)",
    )
    return parser.parse_args(argv)


async def _run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    assert args.action_horizon >= 1, f"action_horizon must be positive, got {args.action_horizon}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    source, source_name = _build_source(args)
    transforms, transforms_label = _build_transforms(args)
    gate, gate_label = _build_gate(args)
    observer, observer_label = _build_observer(args)
    async with run_debug_server(
        source,
        source_name=source_name,
        transforms=transforms,
        gate=gate,
        observer=observer,
        host=args.host,
        port=args.port,
    ) as url:
        print(
            f"Debug server listening on {url} "
            f"source={source_name} transforms={transforms_label} "
            f"gate={gate_label} observer={observer_label} "
            f"action_horizon={args.action_horizon}",
            flush=True,
        )
        LOGGER.info(
            "Debug server listening on %s source=%s transforms=%s gate=%s observer=%s",
            url,
            source_name,
            transforms_label,
            gate_label,
            observer_label,
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
    "run_debug_server",
    "server_handshake_config",
]


if __name__ == "__main__":
    main()
