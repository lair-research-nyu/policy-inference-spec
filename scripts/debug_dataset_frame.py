#!/usr/bin/env python3
"""Introspect a single lerobot-v3 dataset frame against a running policy server.

What it does:
    1. Loads one frame from a lerobot-v3 dataset (default: data/cube-in-box),
       plus the next (H-1) frames of the same episode for the ground-truth
       action trajectory.
    2. Decodes the three camera streams at that frame's timestamp.
    3. Synthesizes the 97-dim wire state by placing the dataset's 25-dim
       commanded positions into the same slices server.minimal reads from.
    4. Sends the request to a policy server (--predict-url) and receives a
       (H, 25) predicted action chunk.
    5. Plots one matplotlib window with:
        - 3 camera views (main, left_wrist, right_wrist)
        - 4 joint plots (left_arm, right_arm, chest, neck), each overlaying:
            * dots at t=0 for the observation state
            * dotted lines for the dataset action trajectory (up to H steps)
            * solid lines for the predicted action trajectory (H steps)
      One joint-dim per plot gets its own color; both gt and pred share the
      color so their alignment is obvious.

Run the minimal server first (loads the policy into GPU):
    uv run python -m server.minimal --host 127.0.0.1 --port 18090

Then:
    uv run python scripts/debug_dataset_frame.py --episode-index 0 --frame-index 50
"""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

from policy_inference_spec.client import RemotePolicyClient
from policy_inference_spec.protocol import (
    DEFAULT_INFERENCE_SERVER_PORT,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    PROMPT_KEY,
)

DEFAULT_DATASET_PATH = Path("data/cube-in-box")
DEFAULT_HORIZON = 50
DEFAULT_PROMPT = "place the orange cube in the box"
DEFAULT_POLICY_ID = "debug"
DEFAULT_PREDICT_URL = f"ws://127.0.0.1:{DEFAULT_INFERENCE_SERVER_PORT}/ws"
DATASET_FPS = 50
_WIRE_STATE_DIM = 97

# 25-dim action/state layout (same in both). Matches server.minimal's slicing.
_BODY_PARTS: tuple[tuple[str, int, int], ...] = (
    ("left_arm", 0, 8),
    ("right_arm", 8, 16),
    ("chest", 16, 22),
    ("neck", 22, 25),
)
# Mapping from dataset 25-dim into 97-dim wire state (inverse of
# server.minimal._inference_response slicing).
_WIRE_STATE_INJECT: tuple[tuple[slice, slice], ...] = (
    (slice(0, 8), slice(0, 8)),       # left_arm
    (slice(32, 40), slice(8, 16)),    # right_arm
    (slice(64, 70), slice(16, 22)),   # chest
    (slice(88, 91), slice(22, 25)),   # neck
)
_VIDEO_STREAMS: tuple[tuple[str, str, str], ...] = (
    ("observation.images.head", "observation/images/main_image", "head"),
    ("observation.images.left_wrist", "observation/images/left_wrist_image", "left_wrist"),
    ("observation.images.right_wrist", "observation/images/right_wrist_image", "right_wrist"),
)


def _decode_video_frame(video_path: Path, target_ts: float) -> np.ndarray:
    """Decode the video frame at ``target_ts`` seconds, return RGB uint8 HWC."""
    assert video_path.is_file(), f"video not found: {video_path}"
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        seek_pts = int(target_ts / float(stream.time_base))
        container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if frame.time >= target_ts - 0.5 / DATASET_FPS:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"no frame at t>={target_ts} in {video_path}")


def _load_dataset_sample(
    dataset_path: Path,
    episode_index: int,
    frame_index: int,
    horizon: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """Return (wire-keyed images, state_25, gt_actions, tail_horizon).

    ``gt_actions`` has shape ``(tail_horizon, 25)`` where ``tail_horizon`` is
    ``min(horizon, episode_length - frame_index)``.
    """
    assert dataset_path.is_dir(), f"dataset dir not found: {dataset_path}"
    eps = pd.read_parquet(dataset_path / "meta" / "episodes")
    matches = eps[eps["episode_index"] == episode_index]
    assert len(matches) == 1, (
        f"episode_index {episode_index} not found (have {eps['episode_index'].tolist()[:5]}...)"
    )
    ep = matches.iloc[0]
    ep_length = int(ep["length"])
    assert 0 <= frame_index < ep_length, (
        f"frame_index {frame_index} out of range [0, {ep_length})"
    )
    tail_horizon = min(horizon, ep_length - frame_index)

    data_path = dataset_path / (
        f"data/chunk-{int(ep['data/chunk_index']):03d}"
        f"/file-{int(ep['data/file_index']):03d}.parquet"
    )
    data = pd.read_parquet(data_path)
    ep_rows = (
        data[data["episode_index"] == episode_index]
        .sort_values("frame_index")
        .reset_index(drop=True)
    )
    assert len(ep_rows) == ep_length, (
        f"episode_rows={len(ep_rows)} != episode_length={ep_length}"
    )

    state_25 = np.asarray(ep_rows.iloc[frame_index]["observation.state"], dtype=np.float32)
    assert state_25.shape == (25,), f"state shape {state_25.shape} != (25,)"
    gt_actions = np.stack(
        [
            np.asarray(ep_rows.iloc[frame_index + i]["action"], dtype=np.float32)
            for i in range(tail_horizon)
        ]
    )
    assert gt_actions.shape == (tail_horizon, 25), f"gt shape {gt_actions.shape}"

    images: dict[str, np.ndarray] = {}
    for src_key, wire_key, _label in _VIDEO_STREAMS:
        chunk = int(ep[f"videos/{src_key}/chunk_index"])
        file = int(ep[f"videos/{src_key}/file_index"])
        from_ts = float(ep[f"videos/{src_key}/from_timestamp"])
        video_path = (
            dataset_path / f"videos/{src_key}/chunk-{chunk:03d}/file-{file:03d}.mp4"
        )
        target_ts = from_ts + frame_index / DATASET_FPS
        images[wire_key] = _decode_video_frame(video_path, target_ts)

    return images, state_25, gt_actions, tail_horizon


def _build_wire_state(state_25: np.ndarray) -> np.ndarray:
    wire = np.zeros(_WIRE_STATE_DIM, dtype=np.float32)
    for wire_slc, ds_slc in _WIRE_STATE_INJECT:
        wire[wire_slc] = state_25[ds_slc]
    return wire


async def _predict(
    predict_url: str,
    *,
    images: dict[str, np.ndarray],
    state_25: np.ndarray,
    prompt: str,
    policy_id: str,
) -> np.ndarray:
    frame: dict[str, object] = {
        JOINT_STATE_KEY: _build_wire_state(state_25),
        PROMPT_KEY: prompt,
        MODEL_ID_KEY: policy_id,
    }
    for wire_key, image in images.items():
        frame[wire_key] = image
    async with RemotePolicyClient(predict_url) as client:
        prediction = await client.predict(frame)
        return np.asarray(prediction.actions_d, dtype=np.float32)


def _plot(
    *,
    images: dict[str, np.ndarray],
    state_25: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    hz: int,
    title: str,
    save_path: Path | None,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])
    gs_top = gs[0, :].subgridspec(1, 3)
    for col, (_src, wire_key, label) in enumerate(_VIDEO_STREAMS):
        ax = fig.add_subplot(gs_top[0, col])
        ax.imshow(images[wire_key])
        ax.set_title(label)
        ax.set_axis_off()

    joint_axes = {
        "left_arm": fig.add_subplot(gs[1, 0]),
        "right_arm": fig.add_subplot(gs[1, 1]),
        "chest": fig.add_subplot(gs[2, 0]),
        "neck": fig.add_subplot(gs[2, 1]),
    }
    gt_t = np.arange(gt_actions.shape[0]) / hz
    pred_t = np.arange(pred_actions.shape[0]) / hz

    cmap = plt.get_cmap("tab10")
    for part, lo, hi in _BODY_PARTS:
        ax = joint_axes[part]
        for d_local, d_global in enumerate(range(lo, hi)):
            color = cmap(d_local % 10)
            ax.plot(gt_t, gt_actions[:, d_global], linestyle=":", color=color, linewidth=1.2)
            ax.plot(pred_t, pred_actions[:, d_global], linestyle="-", color=color, linewidth=1.4)
            ax.scatter(
                [0.0],
                [state_25[d_global]],
                marker="o",
                s=42,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        ax.set_title(f"{part}  (dims {lo}..{hi - 1})")
        ax.set_xlabel("t (s)")
        ax.grid(True, alpha=0.3)

    fig.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle=":", label="dataset (gt)"),
            Line2D([0], [0], color="black", linestyle="-", label="predicted"),
            Line2D([0], [0], marker="o", color="black", linestyle="", label="state @ t=0"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"saved plot to {save_path}")
    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frame-index", type=int, default=0, help="offset within the episode"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="action-chunk horizon (server must emit at least this many rows)",
    )
    parser.add_argument("--predict-url", default=DEFAULT_PREDICT_URL)
    parser.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="optional .png path; if set the figure is saved before plt.show()",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert args.horizon > 0, f"horizon must be positive, got {args.horizon}"

    images, state_25, gt_actions, tail_horizon = _load_dataset_sample(
        args.dataset_path, args.episode_index, args.frame_index, args.horizon
    )
    print(
        f"loaded ep={args.episode_index} frame={args.frame_index} "
        f"state={state_25.shape} gt_traj={gt_actions.shape} "
        f"(tail_horizon={tail_horizon}/{args.horizon})"
    )
    pred_actions = asyncio.run(
        _predict(
            args.predict_url,
            images=images,
            state_25=state_25,
            prompt=args.prompt,
            policy_id=args.policy_id,
        )
    )
    assert pred_actions.ndim == 2 and pred_actions.shape[1] == 25, (
        f"predicted action shape {pred_actions.shape} != (H, 25)"
    )
    print(f"predicted: {pred_actions.shape}")

    title = (
        f"{args.dataset_path.name}  ep={args.episode_index}  frame={args.frame_index}  "
        f"predict_url={args.predict_url}"
    )
    _plot(
        images=images,
        state_25=state_25,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        hz=DATASET_FPS,
        title=title,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
