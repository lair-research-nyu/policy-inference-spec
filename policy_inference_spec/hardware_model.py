from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any

import numpy as np
import numpy.typing as npt

from policy_inference_spec.protocol import (
    ACTION_KEY,
    CONTEXT_EMBEDDINGS_KEY,
    CONTEXT_EMBEDDING_TOKENS,
    CONTEXT_EMBEDDING_WIDTH,
    DUMB_REWARD_GOAL_ACTION_CHUNK_KEY,
    DUMB_REWARD_THRESHOLD_KEY,
    ENDPOINT_KEY,
    FAST_MOCK_ACTION_DIM_KEY,
    FAST_MOCK_ACTION_HORIZON_KEY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PROMPT_KEY,
    ACTION_PREFIX_KEY,
    PREFIX_CHANGE_START_KEY,
    TIMESTAMP_KEY,
    ServerFeature,
    ServerHandshake,
    make_server_handshake,
)


@dataclass(frozen=True)
class _HardwareModelSpec:
    state_dim: int
    action_dim: int
    image_resolution: tuple[int, int]
    cameras: tuple[str, ...]


class HardwareModel(StrEnum):
    GEN2 = "gen2"

    @classmethod
    def _missing_(cls, value: object) -> HardwareModel | None:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized == value:
                return None
        else:
            normalized = str(value).strip()
        try:
            return cls(normalized)
        except ValueError:
            return None

    @property
    def state_dim(self) -> int:
        return _HARDWARE_MODEL_SPECS[self].state_dim

    @property
    def action_dim(self) -> int:
        return _HARDWARE_MODEL_SPECS[self].action_dim

    @property
    def image_resolution(self) -> tuple[int, int]:
        return _HARDWARE_MODEL_SPECS[self].image_resolution

    @property
    def cameras(self) -> tuple[str, ...]:
        return _HARDWARE_MODEL_SPECS[self].cameras


_HARDWARE_MODEL_SPECS = {
    HardwareModel.GEN2: _HardwareModelSpec(
        state_dim=97,
        action_dim=25,
        image_resolution=(360, 640),
        cameras=(
            "images/main_image",
            "images/left_wrist_image",
            "images/right_wrist_image",
        ),
    ),
}

DEFAULT_HARDWARE_MODEL = HardwareModel.GEN2


def _observation_keys(hardware_model: HardwareModel) -> list[str]:
    return [f"observation/{cam}" for cam in hardware_model.cameras]


def _summarize_response_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(str(key) for key in value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def _summarize_response_payload(result: dict[str, Any]) -> str:
    parts = [
        f"{key}={_summarize_response_value(value)}"
        for key, value in sorted(result.items(), key=lambda item: str(item[0]))
    ]
    return ", ".join(parts)


def _validate_joint_position_array(
    value: np.ndarray,
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    hm = HardwareModel(hardware_model)
    assert isinstance(value, np.ndarray), f"{JOINT_STATE_KEY} must be ndarray"
    assert value.ndim == 1, f"{JOINT_STATE_KEY} must be 1-D, got shape {value.shape}"
    assert value.shape == (hm.state_dim,), (
        f"{hm.value} {JOINT_STATE_KEY} must be ({hm.state_dim},), got {value.shape}"
    )


def _wire_inference_request_keys(*, hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL) -> frozenset[str]:
    assert isinstance(hardware_model, HardwareModel), (
        f"hardware_model must be HardwareModel, got {type(hardware_model)}"
    )
    return frozenset(
        {
            JOINT_STATE_KEY,
            *_observation_keys(hardware_model),
            PROMPT_KEY,
            MODEL_ID_KEY,
        }
    )


def _optional_wire_inference_request_keys() -> frozenset[str]:
    return frozenset(
        {
            DUMB_REWARD_GOAL_ACTION_CHUNK_KEY,
            DUMB_REWARD_THRESHOLD_KEY,
            FAST_MOCK_ACTION_DIM_KEY,
            FAST_MOCK_ACTION_HORIZON_KEY,
            ACTION_PREFIX_KEY,
            PREFIX_CHANGE_START_KEY,
            TIMESTAMP_KEY,
        }
    )


def server_handshake_for_hardware_model(
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
    *,
    include_image_resolution: bool = True,
    server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    hm = HardwareModel(hardware_model)
    return make_server_handshake(
        camera_names=hm.cameras,
        image_resolution=hm.image_resolution if include_image_resolution else None,
        server_features=server_features,
    )


def validate_ultra_arrays_for_hardware_model(
    arrays: dict[str, npt.NDArray[Any]],
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    hm = HardwareModel(hardware_model)
    image_keys = set(_observation_keys(hm))
    keys = set(arrays.keys())
    expected = {JOINT_STATE_KEY, *image_keys}
    assert keys == expected, f"{hm.value} request keys {keys} != expected {expected}"
    joint_position = arrays[JOINT_STATE_KEY]
    expected_joint_position_shape = (1, hm.state_dim)
    assert joint_position.shape == expected_joint_position_shape, (
        f"{hm.value} {JOINT_STATE_KEY} shape {joint_position.shape} != {expected_joint_position_shape}"
    )
    for key in image_keys:
        image = arrays[key]
        assert image.ndim in (3, 4), f"{hm.value} image {key} must be HWC or BHWC, got {image.shape}"
        if image.ndim == 4:
            assert image.shape[0] == 1, f"{hm.value} image {key} batch size must be 1, got {image.shape}"
            assert image.shape[-1] == 3, f"{hm.value} image {key} channel dim must be 3, got {image.shape}"
            continue
        assert image.shape[-1] == 3, f"{hm.value} image {key} channel dim must be 3, got {image.shape}"


def validate_wire_inference_request_frame(
    frame: dict[str, Any], hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL
) -> HardwareModel:
    assert ENDPOINT_KEY not in frame, "inference frame must not contain endpoint"
    required = _wire_inference_request_keys(hardware_model=hardware_model)
    allowed = required | _optional_wire_inference_request_keys()
    keys = set(frame.keys())
    assert required <= keys <= allowed, f"wire inference keys {keys} must include {required} and stay within {allowed}"
    assert isinstance(frame[PROMPT_KEY], str), f"{PROMPT_KEY} must be str"
    assert isinstance(frame[MODEL_ID_KEY], str), f"{MODEL_ID_KEY} must be str"
    fast_mock_action_dim_raw = frame.get(FAST_MOCK_ACTION_DIM_KEY)
    fast_mock_action_horizon_raw = frame.get(FAST_MOCK_ACTION_HORIZON_KEY)
    has_fast_mock_action_dim = fast_mock_action_dim_raw is not None
    has_fast_mock_action_horizon = fast_mock_action_horizon_raw is not None
    assert has_fast_mock_action_dim == has_fast_mock_action_horizon, (
        f"{FAST_MOCK_ACTION_DIM_KEY} and {FAST_MOCK_ACTION_HORIZON_KEY} must be provided together"
    )
    expected_action_dim = hardware_model.action_dim
    if has_fast_mock_action_dim:
        assert isinstance(fast_mock_action_dim_raw, int), f"{FAST_MOCK_ACTION_DIM_KEY} must be int"
        assert isinstance(fast_mock_action_horizon_raw, int), f"{FAST_MOCK_ACTION_HORIZON_KEY} must be int"
        assert fast_mock_action_dim_raw > 0, f"{FAST_MOCK_ACTION_DIM_KEY} must be positive"
        assert fast_mock_action_horizon_raw > 0, f"{FAST_MOCK_ACTION_HORIZON_KEY} must be positive"
        expected_action_dim = fast_mock_action_dim_raw
    has_goal_chunk = DUMB_REWARD_GOAL_ACTION_CHUNK_KEY in frame
    has_threshold = DUMB_REWARD_THRESHOLD_KEY in frame
    assert has_goal_chunk == has_threshold, (
        f"{DUMB_REWARD_GOAL_ACTION_CHUNK_KEY} and {DUMB_REWARD_THRESHOLD_KEY} must be provided together"
    )
    if has_goal_chunk:
        goal_action_chunk = frame[DUMB_REWARD_GOAL_ACTION_CHUNK_KEY]
        assert isinstance(goal_action_chunk, np.ndarray), f"{DUMB_REWARD_GOAL_ACTION_CHUNK_KEY} must be ndarray"
        assert goal_action_chunk.ndim == 2, f"{DUMB_REWARD_GOAL_ACTION_CHUNK_KEY} must be 2-D, got {goal_action_chunk.shape}"
        assert goal_action_chunk.shape[1] == expected_action_dim, (
            f"{DUMB_REWARD_GOAL_ACTION_CHUNK_KEY} second dim must be {expected_action_dim}, got {goal_action_chunk.shape}"
        )
        assert np.issubdtype(goal_action_chunk.dtype, np.floating), (
            f"{DUMB_REWARD_GOAL_ACTION_CHUNK_KEY} must be floating ndarray, got {goal_action_chunk.dtype}"
        )
        threshold = frame[DUMB_REWARD_THRESHOLD_KEY]
        assert isinstance(threshold, (int, float)), f"{DUMB_REWARD_THRESHOLD_KEY} must be numeric"
        assert float(threshold) > 0.0, f"{DUMB_REWARD_THRESHOLD_KEY} must be positive"
    _validate_joint_position_array(frame[JOINT_STATE_KEY], hardware_model)
    for k in _observation_keys(hardware_model):
        v = frame[k]
        assert isinstance(v, (bytes, np.ndarray)), f"{k} must be jpeg bytes or ndarray, got {type(v)}"
    return hardware_model


def validate_wire_inference_response(
    result: dict[str, Any],
    hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    response_summary = _summarize_response_payload(result)
    assert "error" not in result, f"unexpected error payload: {response_summary}"
    allowed = frozenset({ACTION_KEY, CONTEXT_EMBEDDINGS_KEY, INFERENCE_TIME_KEY, POLICY_ID_KEY})
    assert set(result.keys()) <= allowed, (
        f"response keys {set(result.keys())} not subset of {allowed}; summary={response_summary}"
    )
    assert ACTION_KEY in result, f"response missing action; summary={response_summary}"
    assert CONTEXT_EMBEDDINGS_KEY in result, f"response missing {CONTEXT_EMBEDDINGS_KEY}; summary={response_summary}"
    actions = result[ACTION_KEY]
    assert isinstance(actions, np.ndarray), f"action must be ndarray, got {type(actions)}; summary={response_summary}"
    assert actions.ndim == 2, f"action must be 2-D, got shape {actions.shape}"
    assert actions.shape[1] == hardware_model.action_dim, (
        f"action second dim must be {hardware_model.action_dim}, got {actions.shape}"
    )
    assert np.issubdtype(actions.dtype, np.floating), f"action must be floating ndarray, got {actions.dtype}"
    context_embeddings = result[CONTEXT_EMBEDDINGS_KEY]
    assert isinstance(context_embeddings, np.ndarray), (
        f"{CONTEXT_EMBEDDINGS_KEY} must be ndarray, got {type(context_embeddings)}; summary={response_summary}"
    )
    assert context_embeddings.shape == (CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), (
        f"{CONTEXT_EMBEDDINGS_KEY} must have shape {(CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH)}, "
        f"got {context_embeddings.shape}"
    )
    assert np.issubdtype(context_embeddings.dtype, np.floating), (
        f"{CONTEXT_EMBEDDINGS_KEY} must be floating ndarray, got {context_embeddings.dtype}"
    )
    if INFERENCE_TIME_KEY in result:
        assert isinstance(result[INFERENCE_TIME_KEY], (int, float)), "inference_time must be numeric"
    if POLICY_ID_KEY in result:
        assert isinstance(result[POLICY_ID_KEY], str), f"{POLICY_ID_KEY} must be str"


__all__ = [
    "DEFAULT_HARDWARE_MODEL",
    "HardwareModel",
    "server_handshake_for_hardware_model",
    "validate_ultra_arrays_for_hardware_model",
    "validate_wire_inference_request_frame",
    "validate_wire_inference_response",
]
