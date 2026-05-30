from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import numpy as np
import simplejpeg  # type: ignore[import-untyped]

from policy_inference_spec.client import RemotePolicyClient
from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL
from policy_inference_spec.protocol import JOINT_STATE_KEY, MODEL_ID_KEY, PROMPT_KEY
from server.minimal import run_example_server


def _random_predict_frame() -> dict[str, object]:
    rng = np.random.default_rng()
    height, width = DEFAULT_HARDWARE_MODEL.image_resolution
    frame: dict[str, object] = {
        JOINT_STATE_KEY: rng.standard_normal(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        PROMPT_KEY: "",
        MODEL_ID_KEY: "",
    }
    for cam in DEFAULT_HARDWARE_MODEL.cameras:
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frame[f"observation/{cam}"] = simplejpeg.encode_jpeg(rgb, quality=75)
    return frame


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict smoke test against a policy inference WebSocket server.")
    p.add_argument(
        "--url",
        default=os.environ.get("POLICY_SERVER_URL"),
        help="WebSocket URL. If omitted, starts the local example server. Also reads $POLICY_SERVER_URL.",
    )
    p.add_argument(
        "--predicts",
        type=int,
        default=2,
        metavar="N",
        help="Number of predict requests to send (default: 2)",
    )
    p.add_argument("--api-key", default=os.environ.get("POLICY_AUTH_TOKEN"), help="Optional x-api-key header")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


@asynccontextmanager
async def _smoke_server_url(url: str | None):
    if url is not None:
        yield url
        return
    async with run_example_server() as local_url:
        yield local_url


async def _run() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if args.predicts < 1:
        print("--predicts must be >= 1", file=sys.stderr)
        return 2

    headers = {"x-api-key": args.api_key} if args.api_key else None
    async with _smoke_server_url(args.url) as predict_url:
        client = RemotePolicyClient(predict_url, policy_auth_headers=headers)

        print(f"URL {predict_url!r} hardware_model={DEFAULT_HARDWARE_MODEL.value}", flush=True)
        async with client:
            for i in range(args.predicts):
                frame = _random_predict_frame()
                pred = await client.predict(frame)
                print(
                    f"predict {i + 1}/{args.predicts}: "
                    f"total_latency_ms={pred.total_latency_ms:.1f} "
                    f"actions_shape={tuple(pred.actions_d.shape)} "
                    f"policy_id={pred.policy_id!r}",
                )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
