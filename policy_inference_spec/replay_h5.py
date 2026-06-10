from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import typer

from policy_inference_spec.client import RemotePolicyPrediction
from policy_inference_spec.feature_engineering import (
    FeatureBundle,
    ScalarFeature,
    SchemaName,
    VideoFeature,
    get_feature_bundle_for_schema,
)
from policy_inference_spec.protocol import DEFAULT_INFERENCE_SERVER_PORT
from policy_inference_spec.replay_rrd import (
    DEFAULT_POLICY_ID,
    log_to_rerun,
    predict_sample,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("output_h5.rrd")
DEFAULT_PREDICT_URL = f"ws://127.0.0.1:{DEFAULT_INFERENCE_SERVER_PORT}/ws"

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class H5ReplaySummary:
    sample_count: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    recording_duration: pd.Timedelta
    wall_time_s: float
    speed_ratio: float
    h5_path: Path
    episode_index: int
    output_path: Path


def _h5_path_for_scalar(feature: ScalarFeature) -> str:
    assert feature.rrd_entity_path.startswith("/"), (
        f"Expected scalar entity path to start with '/', got {feature.rrd_entity_path}"
    )
    return feature.rrd_entity_path[1:]


def _h5_path_for_video(feature: VideoFeature) -> str:
    return f"cameras/{feature.name}"


class H5EpisodeReplayer:
    def __init__(
        self,
        h5_path: Path,
        episode_index: int,
        feature_bundle: FeatureBundle,
        hz: int,
        publish_hz: float,
    ) -> None:
        assert hz > 0, f"hz must be positive, got {hz}"
        assert publish_hz > 0.0, f"publish_hz must be positive, got {publish_hz}"
        samples_per_prediction = hz / publish_hz
        assert float(samples_per_prediction).is_integer(), (
            f"hz must be an integer multiple of publish_hz, got hz={hz} publish_hz={publish_hz}"
        )

        self.h5_path = h5_path
        self.episode_index = episode_index
        self.feature_bundle = feature_bundle
        self.hz = hz
        self.publish_hz = publish_hz
        self.samples_per_prediction = int(samples_per_prediction)

        with h5py.File(h5_path, "r") as f:
            assert "episodes" in f, f"H5 file missing top-level 'episodes' group: {h5_path}"
            episode_key = f"{episode_index:03d}"
            assert episode_key in f["episodes"], (
                f"Episode {episode_key} not in {h5_path}; available: {sorted(f['episodes'].keys())[:5]}..."
            )
            ep = f["episodes"][episode_key]

            timestamps_ns = np.asarray(ep["timestamps"][:], dtype=np.int64)
            n_frames = int(timestamps_ns.shape[0])
            assert n_frames > 0, f"Episode {episode_key} is empty"

            scalars: dict[str, np.ndarray] = {}
            for feature in feature_bundle.all_scalar_features:
                h5_key = _h5_path_for_scalar(feature)
                assert h5_key in ep, (
                    f"Episode {episode_key} missing required scalar dataset '{h5_key}' "
                    f"for feature '{feature.name}'"
                )
                arr = np.asarray(ep[h5_key][:], dtype=np.float32)
                assert arr.shape[0] == n_frames, (
                    f"Scalar '{h5_key}' has {arr.shape[0]} rows, expected {n_frames}"
                )
                assert arr.shape[-1] == feature.shape, (
                    f"Scalar '{h5_key}' has trailing dim {arr.shape[-1]}, expected {feature.shape}"
                )
                scalars[feature.name] = arr

            videos: dict[str, np.ndarray] = {}
            for video in feature_bundle.videos:
                h5_key = _h5_path_for_video(video)
                assert h5_key in ep, (
                    f"Episode {episode_key} missing required video dataset '{h5_key}' "
                    f"for feature '{video.name}'"
                )
                frames = np.asarray(ep[h5_key][:], dtype=np.uint8)
                assert frames.shape[0] == n_frames, (
                    f"Video '{h5_key}' has {frames.shape[0]} frames, expected {n_frames}"
                )
                assert frames.shape[1:] == video.shape, (
                    f"Video '{h5_key}' has frame shape {frames.shape[1:]}, expected {video.shape}. "
                    f"Regenerate the H5 with the matching --image-scale and crop settings."
                )
                videos[video.name] = frames

        self._timestamps_ns = timestamps_ns
        self._scalars = scalars
        self._videos = videos
        self._n_frames = n_frames

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def _build_sample(self, start_index: int, publish_ts: pd.Timestamp) -> dict[str, object]:
        end_index = start_index + self.samples_per_prediction
        sample: dict[str, object] = {"ts": publish_ts}
        for feature in self.feature_bundle.observations:
            sample[feature.name] = self._scalars[feature.name][start_index].astype(np.float32)
        for feature in self.feature_bundle.actions:
            window = self._scalars[feature.name][start_index:end_index]
            assert window.shape[0] == self.samples_per_prediction, (
                f"Not enough action frames for window starting at {start_index}: "
                f"got {window.shape[0]}, expected {self.samples_per_prediction}"
            )
            sample[feature.name] = window.astype(np.float32)
        for video in self.feature_bundle.videos:
            sample[video.name] = self._videos[video.name][start_index]
        return sample

    def __iter__(self) -> Iterator[dict[str, object]]:
        last_window_start = self._n_frames - self.samples_per_prediction
        if last_window_start < 0:
            return
        start_index = 0
        while start_index <= last_window_start:
            publish_ts = pd.Timestamp(int(self._timestamps_ns[start_index]), unit="ns")
            yield self._build_sample(start_index, publish_ts)
            start_index += self.samples_per_prediction


async def replay_h5_episode(
    *,
    h5_path: Path,
    episode_index: int,
    schema: SchemaName = SchemaName.GEN2_32D_STATE,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    predict_url: str = DEFAULT_PREDICT_URL,
    policy_id: str = DEFAULT_POLICY_ID,
    hz: int = 50,
    prediction_hz: float = 1.0,
    max_samples: int = 250,
) -> H5ReplaySummary:
    assert max_samples > 0, f"max_samples must be positive, got {max_samples}"
    h5_path = h5_path.expanduser()
    output_path = output_path.expanduser()
    assert h5_path.is_file(), f"h5_path must be an existing file, got {h5_path}"
    assert output_path.parent.exists(), f"output_path parent does not exist: {output_path.parent}"

    feature_bundle = get_feature_bundle_for_schema(schema)

    LOGGER.info(
        "Replaying %s episode %03d against %s",
        h5_path,
        episode_index,
        predict_url,
    )
    sample_generator = H5EpisodeReplayer(h5_path, episode_index, feature_bundle, hz, prediction_hz)
    samples: list[dict[str, object]] = []
    prediction_tasks: list[asyncio.Task[RemotePolicyPrediction]] = []
    start_time = time.perf_counter()
    for index, sample in enumerate(sample_generator):
        if index >= max_samples:
            break
        samples.append(sample)
        prediction_tasks.append(
            asyncio.create_task(predict_sample(feature_bundle, sample, predict_url, policy_id))
        )
        await asyncio.sleep(0)

    assert samples, (
        f"No replay samples were produced from {h5_path} episode {episode_index} "
        f"(episode has {sample_generator.n_frames} frames, need at least {sample_generator.samples_per_prediction})"
    )
    predictions = await asyncio.gather(*prediction_tasks)
    wall_time_s = time.perf_counter() - start_time
    first_timestamp = min(sample["ts"] for sample in samples)
    last_timestamp = max(sample["ts"] for sample in samples)
    assert isinstance(first_timestamp, pd.Timestamp), f"Expected pandas.Timestamp, got {type(first_timestamp)}"
    assert isinstance(last_timestamp, pd.Timestamp), f"Expected pandas.Timestamp, got {type(last_timestamp)}"
    recording_duration = last_timestamp - first_timestamp
    speed_ratio = recording_duration.total_seconds() / wall_time_s if wall_time_s > 0 else float("inf")

    LOGGER.info(
        "Samples=%d first_ts=%s last_ts=%s duration=%s speed_ratio=%.2f",
        len(samples),
        first_timestamp,
        last_timestamp,
        recording_duration,
        speed_ratio,
    )
    LOGGER.info("Writing replay output to %s", output_path)
    log_to_rerun(output_path, feature_bundle, samples, predictions, hz)
    return H5ReplaySummary(
        sample_count=len(samples),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        recording_duration=recording_duration,
        wall_time_s=wall_time_s,
        speed_ratio=speed_ratio,
        h5_path=h5_path,
        episode_index=episode_index,
        output_path=output_path,
    )


@app.command()
def main(
    h5_path: Path = typer.Option(..., "--h5-path", help="Input multi-episode .h5 file."),
    episode_index: int = typer.Option(..., "--episode-index", min=0, help="Episode index within the file."),
    schema: SchemaName = typer.Option(SchemaName.GEN2_32D_STATE, help="Feature schema to replay."),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Output .rrd path."),
    predict_url: str = typer.Option(
        DEFAULT_PREDICT_URL,
        "--predict-url",
        "--recording-server-url",
        help="Inference server WebSocket URL.",
    ),
    policy_id: str = typer.Option(DEFAULT_POLICY_ID, help="Model id sent in each request."),
    hz: int = typer.Option(50, min=1, help="Input sample rate."),
    prediction_hz: float = typer.Option(1.0, min=0.001, help="Prediction rate."),
    max_samples: int = typer.Option(250, min=1, help="Maximum replay windows."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = asyncio.run(
        replay_h5_episode(
            h5_path=h5_path,
            episode_index=episode_index,
            schema=schema,
            output_path=output_path,
            predict_url=predict_url,
            policy_id=policy_id,
            hz=hz,
            prediction_hz=prediction_hz,
            max_samples=max_samples,
        )
    )
    typer.echo(
        f"episode={summary.episode_index:03d} "
        f"samples={summary.sample_count} "
        f"duration={summary.recording_duration} "
        f"wall_time_s={summary.wall_time_s:.2f} "
        f"speed_ratio={summary.speed_ratio:.2f} "
        f"output={summary.output_path}"
    )


__all__ = [
    "H5EpisodeReplayer",
    "H5ReplaySummary",
    "replay_h5_episode",
]


if __name__ == "__main__":
    app()
