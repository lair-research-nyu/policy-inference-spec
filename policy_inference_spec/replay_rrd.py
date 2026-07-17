from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import typer

from policy_inference_spec.client import RemotePolicyClient, RemotePolicyPrediction
from policy_inference_spec.feature_engineering import (
    FeatureBundle,
    SchemaName,
    VideoFeature,
    get_feature_bundle_for_schema,
    preprocess_image,
)
from policy_inference_spec.protocol import (
    ACTION_PREFIX_KEY,
    DEFAULT_INFERENCE_SERVER_PORT,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    PREFIX_CHANGE_START_KEY,
    SUBTASK_KEY,
    TASK_KEY,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RECORDING_PATH = Path(
    "~/local_data/recordings/GEN2-020-1774982886515653071_e32c92c0-caf1-4955-a9ab-40e9a8db004c.rrd"
).expanduser()
DEFAULT_OUTPUT_PATH = Path("output.rrd")
DEFAULT_POLICY_ID = "ultra-ai/bc-nextgen-v0/just-rings-bigger-3:v24"
DEFAULT_TASK = "tower_stack_unstack"
DEFAULT_SUBTASK = "stack_rings"
DEFAULT_PREDICT_URL = f"ws://127.0.0.1:{DEFAULT_INFERENCE_SERVER_PORT}/ws"
RERUN_APP_ID = "offline_policy_eval_predictions"
IMAGE_STREAMS = ("head", "left_wrist", "right_wrist")
PREDICTED_ROOT = "/predicted"
METRICS_ROOT = "/metrics"
STREAM_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ReplaySummary:
    sample_count: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    recording_duration: pd.Timedelta
    wall_time_s: float
    speed_ratio: float
    recording_path: Path
    output_path: Path


def _is_null(value: Any) -> bool:
    is_value_null = pd.isna(value)
    if isinstance(is_value_null, bool):
        return is_value_null
    return bool(is_value_null.any())


class RerunReplayer:
    def __init__(self, recording_path: Path, feature_bundle: FeatureBundle, hz: int, publish_hz: float) -> None:
        assert hz > 0, f"hz must be positive, got {hz}"
        assert publish_hz > 0.0, f"publish_hz must be positive, got {publish_hz}"
        samples_per_prediction = hz / publish_hz
        assert float(samples_per_prediction).is_integer(), (
            f"hz must be an integer multiple of publish_hz, got hz={hz} publish_hz={publish_hz}"
        )

        self.rerun_server = rr.server.Server(datasets={"recordings": [recording_path]}, host="127.0.0.1")
        self.rerun_dataset = self.rerun_server.client().get_dataset("recordings")
        self.segment_id = self.rerun_dataset.segment_ids()[0]
        all_features = [*feature_bundle.observations, *feature_bundle.actions, *feature_bundle.videos]
        self.rrd_entity_paths = [feature.rrd_entity_path for feature in all_features]
        self.features = {f"{feature.rrd_entity_path}:{feature.rrd_component}": feature for feature in all_features}
        self.column_mapping = {
            f"{feature.rrd_entity_path}:{feature.rrd_component}": feature.name for feature in all_features
        }
        self.column_mapping["ts"] = "ts"
        self.decoders = {
            f"{feature.rrd_entity_path}:{feature.rrd_component}": av.CodecContext.create("h264", "r")
            for feature in feature_bundle.videos
        }
        self.feature_bundle = feature_bundle
        self.hz = hz
        self.publish_hz = publish_hz
        self.samples_per_prediction = int(samples_per_prediction)

    def _decode_frames(self, sample: dict[str, Any]) -> dict[str, Any]:
        for key, value in sample.items():
            if key not in self.decoders:
                continue
            frames = self.decoders[key].decode(av.Packet(value[0].tobytes()))
            assert len(frames) == 1, f"Expected 1 frame per packet for {key}, got {len(frames)}"
            sample[key] = frames[0]
        return sample

    def postprocess_sample(self, samples: list[dict[str, Any]], publish_ts: pd.Timestamp) -> dict[str, Any]:
        assert len(samples) == self.samples_per_prediction, (
            f"Expected {self.samples_per_prediction} samples, got {len(samples)}"
        )
        sample = samples[0].copy()
        for video_feature_key in self.decoders:
            frame = sample[video_feature_key]
            np_frame = frame.to_ndarray(format="rgb24")
            feature = self.features[video_feature_key]
            assert isinstance(feature, VideoFeature), f"Expected VideoFeature, got {type(feature)}"
            sample[video_feature_key] = preprocess_image(
                np_frame,
                crop_to_mono=feature.crop_to_mono,
                downsample_factor=feature.downsample_factor,
            )
        for feature in self.feature_bundle.all_scalar_features:
            sample[feature.name] = sample[feature.name].astype(np.float32)
        for feature in self.feature_bundle.actions:
            sample[feature.name] = np.vstack([s[feature.name] for s in samples], dtype=np.float32)
        processed_sample = {self.column_mapping[key]: value for key, value in sample.items()}
        processed_sample["ts"] = publish_ts
        return processed_sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        stream = (
            self.rerun_dataset.filter_segments(self.segment_id)
            .filter_contents(self.rrd_entity_paths)
            .reader(index="ts")
            .select("ts", *self.features)
            .execute_stream()
        )
        stream_iter = iter(stream)
        most_recent_sample: dict[str, Any] = {}
        last_samples: list[dict[str, Any]] = []
        while set(most_recent_sample) != set(self.column_mapping):
            sample = next(stream_iter)
            dataframe = sample.to_pyarrow().to_pandas()
            for row in dataframe.to_dict(orient="records"):
                try:
                    data = self._decode_frames({key: value for key, value in row.items() if not _is_null(value)})
                    most_recent_sample.update(data)
                except AssertionError as e:
                    print(f"failed one sample: {e}")

        LOGGER.info("All streams populated at %s", most_recent_sample["ts"])
        next_publish_ts = most_recent_sample["ts"] + pd.Timedelta(seconds=1 / self.publish_hz)
        next_sample_ts = most_recent_sample["ts"] + pd.Timedelta(seconds=1 / self.hz)

        for sample in stream:
            dataframe = sample.to_pyarrow().to_pandas()
            for row in dataframe.to_dict(orient="records"):
                try:
                    data = self._decode_frames({key: value for key, value in row.items() if not _is_null(value)})
                except:
                    print(f"Failed to decode a frame. skipping.")
                    continue
                while data["ts"] > next_sample_ts:
                    last_samples.append(most_recent_sample.copy())
                    next_sample_ts = next_sample_ts + pd.Timedelta(seconds=1 / self.hz)
                    if next_sample_ts > next_publish_ts:
                        yield self.postprocess_sample(last_samples, next_publish_ts)
                        next_publish_ts = next_publish_ts + pd.Timedelta(seconds=1 / self.publish_hz)
                        last_samples = []
                most_recent_sample.update(data)


def _action_prefix_payload(
    processed_sample: dict[str, np.ndarray],
    action_prefix_steps: int,
    prefix_change_start: int,
) -> dict[str, object]:
    assert action_prefix_steps >= 0, f"action_prefix_steps must be non-negative, got {action_prefix_steps}"
    if action_prefix_steps == 0:
        return {}

    actions = processed_sample["action"]
    assert actions.ndim == 2, f"Expected action prefix source to be 2-D, got {actions.shape}"
    assert action_prefix_steps <= actions.shape[0], (
        f"action_prefix_steps must be <= action horizon {actions.shape[0]}, got {action_prefix_steps}"
    )
    action_prefix = actions[:action_prefix_steps].astype(np.float32, copy=False)
    return {
        ACTION_PREFIX_KEY: action_prefix,
        PREFIX_CHANGE_START_KEY: prefix_change_start,
    }


async def predict_sample(
    feature_bundle: FeatureBundle,
    sample: dict[str, np.ndarray | pd.Timestamp],
    predict_url: str,
    policy_id: str,
    task: str,
    subtask: str,
    action_prefix_steps: int = 0,
    prefix_change_start: int = 0,
) -> RemotePolicyPrediction:
    assert task, "task cannot be empty"
    assert subtask, "subtask cannot be empty"
    arrays: dict[str, np.ndarray] = {}
    for key, value in sample.items():
        if key == "ts":
            continue
        assert isinstance(value, np.ndarray), f"Expected ndarray for {key}, got {type(value)}"
        arrays[key] = value
    processed_sample = feature_bundle.preprocess(arrays)
    async with RemotePolicyClient(predict_url) as inference_client:
        return await inference_client.predict(
            {
                JOINT_STATE_KEY: processed_sample["observation.state"],
                TASK_KEY: task,
                SUBTASK_KEY: subtask,
                MODEL_ID_KEY: policy_id,
                "observation/images/main_image": processed_sample["head"],
                "observation/images/left_wrist_image": processed_sample["left_wrist"],
                "observation/images/right_wrist_image": processed_sample["right_wrist"],
                **_action_prefix_payload(processed_sample, action_prefix_steps, prefix_change_start),
            }
        )


def _stream_name(entity_path: str) -> str:
    return entity_path.strip("/").split("/")[-1]


def _predicted_entity_path(entity_path: str) -> str:
    return f"{PREDICTED_ROOT}{entity_path}"


def _metric_entity_path(metric_name: str, entity_path: str) -> str:
    return f"{METRICS_ROOT}/{metric_name}{entity_path}"


def _palette_color(index: int) -> tuple[int, int, int]:
    return STREAM_COLORS[index % len(STREAM_COLORS)]


def _dimension_colors(stream_index: int, dimension_count: int) -> list[tuple[int, int, int]]:
    return [_palette_color(stream_index + dimension_idx) for dimension_idx in range(dimension_count)]


def _build_blueprint(feature_bundle: FeatureBundle) -> rrb.Blueprint:
    action_time_range = rrb.VisibleTimeRange(
        "ts",
        start=rr.TimeRangeBoundary.cursor_relative(seconds=-2.0),
        end=rr.TimeRangeBoundary.cursor_relative(seconds=10.0),
    )
    action_tabs = [
        rrb.TimeSeriesView(
            name=_stream_name(feature.rrd_entity_path),
            contents=[feature.rrd_entity_path, _predicted_entity_path(feature.rrd_entity_path)],
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=[action_time_range],
        )
        for feature in feature_bundle.actions
    ]
    image_tabs = [
        rrb.Spatial2DView(
            name=image_stream,
            contents=[f"/{image_stream}"],
        )
        for image_stream in IMAGE_STREAMS
    ]
    loss_tabs = [
        rrb.TimeSeriesView(
            name=metric_name.upper(),
            contents=[_metric_entity_path(metric_name, feature.rrd_entity_path) for feature in feature_bundle.actions],
            plot_legend=rrb.PlotLegend(visible=True),
        )
        for metric_name in ("l1", "l2")
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Tabs(*image_tabs, name="Images"),
                rrb.Tabs(*loss_tabs, name="Loss"),
            ),
            rrb.Tabs(*action_tabs, name="Actions"),
        ),
        collapse_panels=True,
    )


def _log_series_styles(sample: dict[str, np.ndarray | pd.Timestamp], feature_bundle: FeatureBundle) -> None:
    for stream_index, feature in enumerate(feature_bundle.actions):
        stream = sample[feature.name]
        assert isinstance(stream, np.ndarray), f"Expected ndarray for {feature.name}, got {type(stream)}"
        dimension_count = stream.shape[1]
        dimension_colors = _dimension_colors(stream_index * dimension_count, dimension_count)
        stream_name = _stream_name(feature.rrd_entity_path)
        rr.log(
            feature.rrd_entity_path,
            rr.SeriesLines(
                colors=dimension_colors,
                widths=[1.0] * dimension_count,
                names=[f"{stream_name}[{dimension_idx}] gt" for dimension_idx in range(dimension_count)],
            ),
            static=True,
        )
        rr.log(
            _predicted_entity_path(feature.rrd_entity_path),
            rr.SeriesLines(
                colors=dimension_colors,
                widths=[2.0] * dimension_count,
                names=[f"{stream_name}[{dimension_idx}] pred" for dimension_idx in range(dimension_count)],
            ),
            static=True,
        )
        stream_color = _palette_color(stream_index)
        for metric_name in ("l1", "l2"):
            rr.log(
                _metric_entity_path(metric_name, feature.rrd_entity_path),
                rr.SeriesLines(
                    colors=[stream_color],
                    widths=[1.0],
                    names=[stream_name],
                ),
                static=True,
            )


def log_to_rerun(
    output_path: Path,
    feature_bundle: FeatureBundle,
    samples: list[dict[str, np.ndarray | pd.Timestamp]],
    predictions: list[RemotePolicyPrediction],
    hz: int,
) -> None:
    assert samples, "Expected at least one sample to log"
    assert len(samples) == len(predictions), (
        f"Expected one prediction per sample, got {len(samples)} != {len(predictions)}"
    )

    rr.init(RERUN_APP_ID)
    rr.save(str(output_path))
    rr.send_blueprint(_build_blueprint(feature_bundle))
    _log_series_styles(samples[0], feature_bundle)
    for sample, prediction in zip(samples, predictions, strict=True):
        timestamp = sample["ts"]
        assert isinstance(timestamp, pd.Timestamp), f"Expected pandas.Timestamp, got {type(timestamp)}"
        rr.set_time("ts", timestamp=np.datetime64(timestamp.value, "ns"))
        for video_feature in feature_bundle.videos:
            video = sample[video_feature.name]
            assert isinstance(video, np.ndarray), f"Expected ndarray for {video_feature.name}, got {type(video)}"
            rr.log(f"/{video_feature.name}", rr.Image(video).compress(jpeg_quality=50))

        for feature in feature_bundle.observations:
            observation = sample[feature.name]
            assert isinstance(observation, np.ndarray), f"Expected ndarray for {feature.name}, got {type(observation)}"
            rr.log(feature.rrd_entity_path, rr.Scalars(observation))

        parsed_prediction = feature_bundle.parse_actions(prediction.actions_d)
        first_action = sample[feature_bundle.actions[0].name]
        assert isinstance(first_action, np.ndarray), f"Expected ndarray for {feature_bundle.actions[0].name}"
        action_steps = first_action.shape[0]
        for feature in feature_bundle.actions:
            gt_actions = sample[feature.name]
            assert isinstance(gt_actions, np.ndarray), f"Expected ndarray for {feature.name}, got {type(gt_actions)}"
            predicted_actions = parsed_prediction[feature.name]
            l1_loss = float(np.mean(np.abs(gt_actions - predicted_actions)))
            l2_loss = float(np.mean((gt_actions - predicted_actions) ** 2))
            rr.log(_metric_entity_path("l1", feature.rrd_entity_path), rr.Scalars([l1_loss]))
            rr.log(_metric_entity_path("l2", feature.rrd_entity_path), rr.Scalars([l2_loss]))
            for action_index in range(action_steps):
                action_ts_ns = timestamp.value + int((1e9 / hz) * action_index)
                rr.set_time("ts", timestamp=np.datetime64(action_ts_ns, "ns"))
                rr.log(feature.rrd_entity_path, rr.Scalars(gt_actions[action_index]))
                rr.log(_predicted_entity_path(feature.rrd_entity_path), rr.Scalars(predicted_actions[action_index]))
    rr.disconnect()


async def replay_recording(
    *,
    schema: SchemaName = SchemaName.GEN2_32D_STATE,
    recording_path: Path = DEFAULT_RECORDING_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    predict_url: str = DEFAULT_PREDICT_URL,
    policy_id: str = DEFAULT_POLICY_ID,
    task: str = DEFAULT_TASK,
    subtask: str = DEFAULT_SUBTASK,
    hz: int = 50,
    prediction_hz: float = 1.0,
    max_samples: int = 250,
    action_prefix_steps: int = 0,
    prefix_change_start: int = 0,
) -> ReplaySummary:
    assert max_samples > 0, f"max_samples must be positive, got {max_samples}"
    recording_path = recording_path.expanduser()
    output_path = output_path.expanduser()
    assert recording_path.is_file(), f"recording_path must be an existing file, got {recording_path}"
    assert output_path.parent.exists(), f"output_path parent does not exist: {output_path.parent}"

    feature_bundle = get_feature_bundle_for_schema(schema)

    LOGGER.info("Beginning replay against %s", predict_url)
    sample_generator = RerunReplayer(recording_path, feature_bundle, hz, prediction_hz)
    samples: list[dict[str, np.ndarray | pd.Timestamp]] = []
    prediction_tasks: list[asyncio.Task[RemotePolicyPrediction]] = []
    start_time = time.perf_counter()
    for index, sample in enumerate(sample_generator):
        if index >= max_samples:
            break
        samples.append(sample)
        prediction_tasks.append(
            asyncio.create_task(
                predict_sample(
                    feature_bundle,
                    sample,
                    predict_url,
                    policy_id,
                    task,
                    subtask,
                    action_prefix_steps=action_prefix_steps,
                    prefix_change_start=prefix_change_start,
                )
            )
        )
        await asyncio.sleep(0)

    assert samples, f"No replay samples were produced from {recording_path}"
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
    return ReplaySummary(
        sample_count=len(samples),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        recording_duration=recording_duration,
        wall_time_s=wall_time_s,
        speed_ratio=speed_ratio,
        recording_path=recording_path,
        output_path=output_path,
    )


@app.command()
def main(
    schema: SchemaName = typer.Option(SchemaName.GEN2_32D_STATE, help="Feature schema to replay."),
    recording_path: Path = typer.Option(DEFAULT_RECORDING_PATH, help="Input .rrd recording."),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Output .rrd path."),
    predict_url: str = typer.Option(
        DEFAULT_PREDICT_URL,
        "--predict-url",
        "--recording-server-url",
        help="Inference server WebSocket URL.",
    ),
    policy_id: str = typer.Option(DEFAULT_POLICY_ID, help="Model id sent in each request."),
    task: str = typer.Option(DEFAULT_TASK, help="Task sent in each request."),
    subtask: str = typer.Option(DEFAULT_SUBTASK, help="Subtask sent in each request."),
    hz: int = typer.Option(50, min=1, help="Input sample rate."),
    prediction_hz: float = typer.Option(1.0, min=0.001, help="Prediction rate."),
    max_samples: int = typer.Option(250, min=1, help="Maximum replay windows."),
    action_prefix_steps: int = typer.Option(0, min=0, help="Ground-truth action prefix rows sent per request."),
    prefix_change_start: int | None = typer.Option(
        None,
        help="prefix_change_start sent with action_prefix. Defaults to --action-prefix-steps.",
        show_default=False,
    ),
) -> None:
    if prefix_change_start is None:
        prefix_change_start = action_prefix_steps
    assert prefix_change_start >= 0, f"prefix_change_start must be non-negative, got {prefix_change_start}"
    assert prefix_change_start <= action_prefix_steps, (
        f"prefix_change_start must be <= action_prefix_steps, got {prefix_change_start} > {action_prefix_steps}"
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = asyncio.run(
        replay_recording(
            schema=schema,
            recording_path=recording_path,
            output_path=output_path,
            predict_url=predict_url,
            policy_id=policy_id,
            task=task,
            subtask=subtask,
            hz=hz,
            prediction_hz=prediction_hz,
            max_samples=max_samples,
            action_prefix_steps=action_prefix_steps,
            prefix_change_start=prefix_change_start,
        )
    )
    typer.echo(
        "samples="
        f"{summary.sample_count} "
        f"duration={summary.recording_duration} "
        f"wall_time_s={summary.wall_time_s:.2f} "
        f"speed_ratio={summary.speed_ratio:.2f} "
        f"output={summary.output_path}"
    )


__all__ = [
    "ReplaySummary",
    "RerunReplayer",
    "log_to_rerun",
    "predict_sample",
    "replay_recording",
]


if __name__ == "__main__":
    app()
