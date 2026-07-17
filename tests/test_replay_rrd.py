from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import policy_inference_spec.replay_rrd as replay_rrd
from policy_inference_spec.client import RemotePolicyPrediction
from policy_inference_spec.feature_engineering import FeatureBundle, ScalarFeature, SchemaName, VideoFeature
from policy_inference_spec.protocol import ACTION_PREFIX_KEY, PREFIX_CHANGE_START_KEY, SUBTASK_KEY, TASK_KEY


@pytest.mark.asyncio
async def test_replay_recording_orchestrates_predictions_and_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_path = tmp_path / "input.rrd"
    output_path = tmp_path / "output.rrd"
    recording_path.write_bytes(b"rrd")
    samples = [
        {
            "ts": pd.Timestamp("2026-04-01T00:00:01Z"),
            "/commanded_qpos/left_arm:Scalars:scalars": np.zeros((4, 7), dtype=np.float32),
        },
        {
            "ts": pd.Timestamp("2026-04-01T00:00:02Z"),
            "/commanded_qpos/left_arm:Scalars:scalars": np.ones((4, 7), dtype=np.float32),
        },
    ]
    captured: dict[str, object] = {}

    class FakeReplayer:
        def __init__(self, path: Path, feature_bundle: object, hz: int, publish_hz: float) -> None:
            assert path == recording_path
            assert hz == 50
            assert publish_hz == 1.0
            captured["feature_bundle_name"] = getattr(feature_bundle, "name")

        def __iter__(self):
            return iter(samples)

    async def fake_predict_sample(
        feature_bundle: object,
        sample: object,
        predict_url: str,
        policy_id: str,
        task: str,
        subtask: str,
        action_prefix_steps: int,
        prefix_change_start: int,
    ) -> RemotePolicyPrediction:
        assert predict_url == "ws://127.0.0.1:18090/ws"
        assert policy_id == "policy-id"
        assert task == replay_rrd.DEFAULT_TASK
        assert subtask == replay_rrd.DEFAULT_SUBTASK
        assert action_prefix_steps == 6
        assert prefix_change_start == 6
        assert sample in samples
        action_dim = getattr(feature_bundle, "action_dim")
        return RemotePolicyPrediction(
            actions_d=np.zeros((4, action_dim), dtype=np.float32),
            total_latency_ms=1.5,
            policy_id=policy_id,
            chunk_id="test-chunk-id",
        )

    def fake_log_to_rerun(
        output: Path,
        feature_bundle: object,
        replay_samples: list[dict[str, object]],
        predictions: list[RemotePolicyPrediction],
        hz: int,
    ) -> None:
        captured["output_path"] = output
        captured["logged_sample_count"] = len(replay_samples)
        captured["logged_prediction_count"] = len(predictions)
        captured["logged_hz"] = hz
        captured["logged_bundle_name"] = getattr(feature_bundle, "name")

    monkeypatch.setattr(replay_rrd, "RerunReplayer", FakeReplayer)
    monkeypatch.setattr(replay_rrd, "predict_sample", fake_predict_sample)
    monkeypatch.setattr(replay_rrd, "log_to_rerun", fake_log_to_rerun)

    summary = await replay_rrd.replay_recording(
        recording_path=recording_path,
        output_path=output_path,
        predict_url="ws://127.0.0.1:18090/ws",
        policy_id="policy-id",
        max_samples=10,
        action_prefix_steps=6,
        prefix_change_start=6,
    )

    assert summary.sample_count == 2
    assert summary.first_timestamp == pd.Timestamp("2026-04-01T00:00:01Z")
    assert summary.last_timestamp == pd.Timestamp("2026-04-01T00:00:02Z")
    assert summary.recording_duration == pd.Timedelta(seconds=1)
    assert summary.wall_time_s >= 0.0
    assert summary.speed_ratio >= 0.0
    assert summary.recording_path == recording_path
    assert summary.output_path == output_path
    assert captured == {
        "feature_bundle_name": "gen2-training-32d-state",
        "output_path": output_path,
        "logged_sample_count": 2,
        "logged_prediction_count": 2,
        "logged_hz": 50,
        "logged_bundle_name": "gen2-training-32d-state",
    }


@pytest.mark.asyncio
async def test_replay_recording_requires_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recording_path = tmp_path / "input.rrd"
    recording_path.write_bytes(b"rrd")

    class EmptyReplayer:
        def __init__(self, path: Path, feature_bundle: object, hz: int, publish_hz: float) -> None:
            _ = path, feature_bundle, hz, publish_hz

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(replay_rrd, "RerunReplayer", EmptyReplayer)

    with pytest.raises(AssertionError, match="No replay samples were produced"):
        await replay_rrd.replay_recording(recording_path=recording_path, output_path=tmp_path / "output.rrd")


@pytest.mark.asyncio
async def test_main_defaults_prefix_change_start_to_action_prefix_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_path = tmp_path / "input.rrd"
    output_path = tmp_path / "output.rrd"
    recording_path.write_bytes(b"rrd")

    async def fake_replay_recording(
        *,
        schema: SchemaName,
        recording_path: Path,
        output_path: Path,
        predict_url: str,
        policy_id: str,
        task: str,
        subtask: str,
        hz: int,
        prediction_hz: float,
        max_samples: int,
        action_prefix_steps: int,
        prefix_change_start: int,
    ) -> replay_rrd.ReplaySummary:
        assert action_prefix_steps == 7
        assert prefix_change_start == 7
        return replay_rrd.ReplaySummary(
            sample_count=max_samples,
            first_timestamp=pd.Timestamp("2026-04-01T00:00:01Z"),
            last_timestamp=pd.Timestamp("2026-04-01T00:00:01Z"),
            recording_duration=pd.Timedelta(0),
            wall_time_s=0.0,
            speed_ratio=0.0,
            recording_path=recording_path,
            output_path=output_path,
        )

    monkeypatch.setattr(replay_rrd, "replay_recording", fake_replay_recording)

    await asyncio.to_thread(
        replay_rrd.main,
        schema=SchemaName.GEN2_32D_STATE,
        recording_path=recording_path,
        output_path=output_path,
        predict_url="ws://127.0.0.1:18090/ws",
        policy_id="policy-id",
        task="task",
        subtask="subtask",
        hz=50,
        prediction_hz=1.0,
        max_samples=1,
        action_prefix_steps=7,
        prefix_change_start=None,
    )


@pytest.mark.asyncio
async def test_main_rejects_change_start_after_action_prefix_steps(tmp_path: Path) -> None:
    recording_path = tmp_path / "input.rrd"
    recording_path.write_bytes(b"rrd")

    with pytest.raises(AssertionError, match="prefix_change_start must be <= action_prefix_steps"):
        await asyncio.to_thread(
            replay_rrd.main,
            schema=SchemaName.GEN2_32D_STATE,
            recording_path=recording_path,
            output_path=tmp_path / "output.rrd",
            predict_url="ws://127.0.0.1:18090/ws",
            policy_id="policy-id",
            task="task",
            subtask="subtask",
            hz=50,
            prediction_hz=1.0,
            max_samples=1,
            action_prefix_steps=7,
            prefix_change_start=8,
        )


@pytest.mark.asyncio
async def test_predict_sample_adds_unpadded_action_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    feature_bundle = FeatureBundle(
        name="test",
        observations=[
            ScalarFeature(
                name="state",
                dtype="float32",
                shape=2,
                rrd_entity_path="/state",
            )
        ],
        actions=[
            ScalarFeature(
                name="action",
                dtype="float32",
                shape=2,
                rrd_entity_path="/action",
            )
        ],
        videos=[
            VideoFeature(name="head", shape=(1, 1, 3), rrd_entity_path="/head"),
            VideoFeature(name="left_wrist", shape=(1, 1, 3), rrd_entity_path="/left_wrist"),
            VideoFeature(name="right_wrist", shape=(1, 1, 3), rrd_entity_path="/right_wrist"),
        ],
    )
    sample: dict[str, np.ndarray | pd.Timestamp] = {
        "ts": pd.Timestamp("2026-04-01T00:00:01Z"),
        "head": np.zeros((1, 1, 3), dtype=np.uint8),
        "left_wrist": np.zeros((1, 1, 3), dtype=np.uint8),
        "right_wrist": np.zeros((1, 1, 3), dtype=np.uint8),
    }
    for feature in feature_bundle.observations:
        sample[feature.name] = np.zeros((feature.shape,), dtype=np.float32)
    for feature in feature_bundle.actions:
        sample[feature.name] = np.arange(8 * feature.shape, dtype=np.float32).reshape(8, feature.shape)
    captured: dict[str, Any] = {}

    class FakeRemotePolicyClient:
        def __init__(self, predict_url: str) -> None:
            assert predict_url == "ws://127.0.0.1:18090/ws"

        async def __aenter__(self) -> FakeRemotePolicyClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            pass

        async def predict(
            self,
            request: dict[str, object],
            *,
            reward: float | None = None,
            prev_skipped_action_start: int | None = None,
        ) -> RemotePolicyPrediction:
            del reward, prev_skipped_action_start
            captured["request"] = request
            captured[ACTION_PREFIX_KEY] = request.get(ACTION_PREFIX_KEY)
            captured[PREFIX_CHANGE_START_KEY] = request.get(PREFIX_CHANGE_START_KEY)
            return RemotePolicyPrediction(
                actions_d=np.zeros((8, feature_bundle.action_dim), dtype=np.float32),
                total_latency_ms=1.5,
                policy_id="policy-id",
                chunk_id="test-chunk-id",
            )

    monkeypatch.setattr(replay_rrd, "RemotePolicyClient", FakeRemotePolicyClient)

    await replay_rrd.predict_sample(
        feature_bundle,
        sample,
        "ws://127.0.0.1:18090/ws",
        "policy-id",
        "task",
        "sub task",
        action_prefix_steps=5,
        prefix_change_start=3,
    )

    request = captured["request"]
    assert isinstance(request, dict)
    assert request[TASK_KEY] == "task"
    assert request[SUBTASK_KEY] == "sub task"
    prefix = captured[ACTION_PREFIX_KEY]
    assert isinstance(prefix, np.ndarray)
    assert prefix.shape == (5, feature_bundle.action_dim)
    assert captured[PREFIX_CHANGE_START_KEY] == 3
    arrays: dict[str, np.ndarray] = {}
    for key, value in sample.items():
        if key == "ts":
            continue
        assert isinstance(value, np.ndarray)
        arrays[key] = value
    expected_actions = feature_bundle.preprocess(arrays)["action"]
    np.testing.assert_array_equal(prefix, expected_actions[:5])


def test_action_prefix_payload_returns_empty_when_disabled() -> None:
    processed_sample = {"action": np.zeros((50, 25), dtype=np.float32)}
    assert replay_rrd._action_prefix_payload(processed_sample, 0, 0) == {}


def test_action_prefix_payload_uses_requested_prefix_steps() -> None:
    action = np.arange(50 * 25, dtype=np.float32).reshape(50, 25)
    payload = replay_rrd._action_prefix_payload({"action": action}, 7, 3)
    action_prefix = payload[ACTION_PREFIX_KEY]
    assert isinstance(action_prefix, np.ndarray)
    assert action_prefix.shape == (7, 25)
    assert payload[PREFIX_CHANGE_START_KEY] == 3
    np.testing.assert_allclose(action_prefix, action[:7])
