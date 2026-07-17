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
    DEFAULT_INFERENCE_SERVER_PORT,
    ENDPOINT_DONE,
    ENDPOINT_KEY,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    Q_VALUE_KEY,
    SUBTASK_KEY,
    TASK_KEY,
    STATUS_KEY,
    ServerFeature,
    ServerHandshake,
)

DEFAULT_ACTION_HORIZON = 4
DEFAULT_REPLAY_ACTION_HORIZON = 50
EXAMPLE_POLICY_ID = "example-linear"
EXAMPLE_IMAGE_RESOLUTION = DEFAULT_HARDWARE_MODEL.image_resolution
EXAMPLE_ACTION_DIM = DEFAULT_HARDWARE_MODEL.action_dim
LOGGER = logging.getLogger(__name__)


class ExampleLinearPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(DEFAULT_HARDWARE_MODEL.state_dim, EXAMPLE_ACTION_DIM)
        with torch.no_grad():
            weight = torch.arange(
                EXAMPLE_ACTION_DIM * DEFAULT_HARDWARE_MODEL.state_dim,
                dtype=torch.float32,
            ).reshape(EXAMPLE_ACTION_DIM, DEFAULT_HARDWARE_MODEL.state_dim)
            self.linear.weight.copy_(weight / 1000.0)
            self.linear.bias.copy_(torch.arange(EXAMPLE_ACTION_DIM, dtype=torch.float32) / 100.0)

    def forward(self, joint_position: torch.Tensor) -> torch.Tensor:
        return self.linear(joint_position)


EXAMPLE_POLICY = ExampleLinearPolicy()


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = ()
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=True,
        server_features=server_features,
    )


def example_policy_actions(
    joint_position: np.ndarray,
    *,
    horizon: int = DEFAULT_ACTION_HORIZON,
) -> np.ndarray:
    assert joint_position.shape == (DEFAULT_HARDWARE_MODEL.state_dim,), (
        f"joint_position must have shape ({DEFAULT_HARDWARE_MODEL.state_dim},), got {joint_position.shape}"
    )
    with torch.no_grad():
        joint_tensor = torch.from_numpy(joint_position.astype(np.float32, copy=True))
        action = EXAMPLE_POLICY(joint_tensor).cpu().numpy().astype(np.float32, copy=False)
    return np.repeat(action[None, :], horizon, axis=0)


def _inference_response(
    frame: dict[str, Any],
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
) -> dict[str, Any]:
    joint_position = frame[JOINT_STATE_KEY]
    assert isinstance(joint_position, np.ndarray), type(joint_position)
    actions = example_policy_actions(joint_position, horizon=action_horizon)
    resp = {
        ACTION_KEY: actions,
        INFERENCE_TIME_KEY: 0.25,
        POLICY_ID_KEY: EXAMPLE_POLICY_ID,
        Q_VALUE_KEY: 0.0,
    }
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(
    connection: ServerConnection,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    server_features: Iterable[str | ServerFeature] = (),
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
        if frame.get(ENDPOINT_KEY) == ENDPOINT_DONE:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_TELEMETRY:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        validate_wire_inference_request_frame(frame)
        _ = frame[TASK_KEY]
        _ = frame[SUBTASK_KEY]
        _ = frame[MODEL_ID_KEY]
        resp = _inference_response(frame, action_horizon=action_horizon)
        await connection.send(serialize_to_msgpack(resp))


@asynccontextmanager
async def run_example_server(
    host: str = "127.0.0.1",
    port: int = 0,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    *, server_features: Iterable[str | ServerFeature] = (ServerFeature.REWARDS,)
) -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(
            connection,
            action_horizon=action_horizon,
            server_features=server_features,
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
    return parser.parse_args(argv)


def _cli_server_features(no_rewards: bool) -> tuple[ServerFeature, ...]:
    if no_rewards:
        return ()
    return (ServerFeature.REWARDS,)


async def _run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    assert args.action_horizon >= 1, f"action_horizon must be positive, got {args.action_horizon}"
    server_features = _cli_server_features(args.no_rewards)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    async with run_example_server(
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
        server_features=server_features,
    ) as url:
        LOGGER.info("Minimal example server listening on %s action_horizon=%d", url, args.action_horizon)
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
