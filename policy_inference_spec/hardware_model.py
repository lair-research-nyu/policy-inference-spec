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
    ACTION_PREFIX_KEY,
    CHUNK_ID_KEY,
    CONDITIONING_METADATA_KEY,
    ENDPOINT_INTERVENTION,
    ENDPOINT_KEY,
    DONE_KEY,
    DONE_REASON_KEY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    OBSERVATION_ENV_KEY,
    OBSERVATION_HIDDEN_KEY,
    POLICY_ID_KEY,
    PREV_SKIPPED_ACTION_START_KEY,
    PREFIX_CHANGE_START_KEY,
    Q_VALUE_KEY,
    REWARD_ACTION_INDEX_KEY,
    REWARD_KEY,
    RL_ENABLED_KEY,
    START_METADATA_KEY,
    SUBTASK_KEY,
    TABLE_VIEW_IMAGE_KEY,
    TASK_KEY,
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
    assert value.shape == (hm.state_dim,), f"{hm.value} {JOINT_STATE_KEY} must be ({hm.state_dim},), got {value.shape}"


def _wire_inference_request_keys(*, hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL) -> frozenset[str]:
    assert isinstance(hardware_model, HardwareModel), (
        f"hardware_model must be HardwareModel, got {type(hardware_model)}"
    )
    return frozenset(
        {
            JOINT_STATE_KEY,
            *_observation_keys(hardware_model),
            MODEL_ID_KEY,
        }
    )


def _optional_wire_inference_request_keys() -> frozenset[str]:
    return frozenset(
        {
            ACTION_PREFIX_KEY,
            PREFIX_CHANGE_START_KEY,
            OBSERVATION_ENV_KEY,
            OBSERVATION_HIDDEN_KEY,
            PREV_SKIPPED_ACTION_START_KEY,
            REWARD_KEY,
            REWARD_ACTION_INDEX_KEY,
            DONE_KEY,
            DONE_REASON_KEY,
            START_METADATA_KEY,
            CONDITIONING_METADATA_KEY,
            TASK_KEY,
            SUBTASK_KEY,
            # Optional station hardware; present only when table-view cam is available.
            TABLE_VIEW_IMAGE_KEY,
            CHUNK_ID_KEY,
        }
    )


def _validate_inference_metadata_value(value: Any, *, key: str) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_inference_metadata_value(item, key=f"{key}[{index}]")
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            assert isinstance(child_key, str), f"{key} keys must be str"
            _validate_inference_metadata_value(child_value, key=f"{key}.{child_key}")
        return
    raise AssertionError(f"{key} must contain JSON-like metadata, got {type(value)}")


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

    assert isinstance(frame[TASK_KEY], str), f"{TASK_KEY} must be str"
    assert isinstance(frame[SUBTASK_KEY], str), f"{SUBTASK_KEY} must be str"
    has_task = frame[TASK_KEY] != ""
    has_subtask = frame[SUBTASK_KEY] != ""
    assert has_task == has_subtask, (
        f"{TASK_KEY} and {SUBTASK_KEY} may either both take on values, or neither may have a value"
    )
    expected_action_dim = hardware_model.action_dim
    assert isinstance(frame[MODEL_ID_KEY], str), f"{MODEL_ID_KEY} must be str"
    if REWARD_KEY in frame:
        assert isinstance(frame[REWARD_KEY], (int, float)), f"{REWARD_KEY} must be numeric"
    if REWARD_ACTION_INDEX_KEY in frame:
        reward_action_index = frame[REWARD_ACTION_INDEX_KEY]
        assert isinstance(reward_action_index, int), f"{REWARD_ACTION_INDEX_KEY} must be int"
        assert 0 <= reward_action_index < 50, (
            f"{REWARD_ACTION_INDEX_KEY} must be in [0, 50), got {reward_action_index}"
        )
    if START_METADATA_KEY in frame:
        start_metadata = frame[START_METADATA_KEY]
        assert isinstance(start_metadata, dict), f"{START_METADATA_KEY} must be dict"
        _validate_inference_metadata_value(start_metadata, key=START_METADATA_KEY)
    if CONDITIONING_METADATA_KEY in frame:
        conditioning_metadata = frame[CONDITIONING_METADATA_KEY]
        assert isinstance(conditioning_metadata, dict), f"{CONDITIONING_METADATA_KEY} must be dict"
        _validate_inference_metadata_value(conditioning_metadata, key=CONDITIONING_METADATA_KEY)
    if PREV_SKIPPED_ACTION_START_KEY in frame:
        prev_skipped_action_start = frame[PREV_SKIPPED_ACTION_START_KEY]
        assert isinstance(prev_skipped_action_start, int), f"{PREV_SKIPPED_ACTION_START_KEY} must be int"
        assert 0 <= prev_skipped_action_start < 50, (
            f"{PREV_SKIPPED_ACTION_START_KEY} must be in [0, 50), got {prev_skipped_action_start}"
        )
    has_action_prefix = ACTION_PREFIX_KEY in frame
    has_prefix_change_start = PREFIX_CHANGE_START_KEY in frame
    assert has_action_prefix == has_prefix_change_start, (
        f"{ACTION_PREFIX_KEY} and {PREFIX_CHANGE_START_KEY} must be provided together"
    )
    if has_action_prefix:
        action_prefix = frame[ACTION_PREFIX_KEY]
        assert isinstance(action_prefix, np.ndarray), f"{ACTION_PREFIX_KEY} must be ndarray"
        assert action_prefix.ndim == 2, f"{ACTION_PREFIX_KEY} must be 2-D, got {action_prefix.shape}"
        assert action_prefix.shape[1] == expected_action_dim, (
            f"{ACTION_PREFIX_KEY} second dim must be {expected_action_dim}, got {action_prefix.shape}"
        )
        assert np.issubdtype(action_prefix.dtype, np.floating), (
            f"{ACTION_PREFIX_KEY} must be floating ndarray, got {action_prefix.dtype}"
        )
        prefix_change_start = frame[PREFIX_CHANGE_START_KEY]
        assert isinstance(prefix_change_start, int), f"{PREFIX_CHANGE_START_KEY} must be int"
        assert prefix_change_start >= 0, f"{PREFIX_CHANGE_START_KEY} must be non-negative"
    if OBSERVATION_HIDDEN_KEY in frame:
        hidden = frame[OBSERVATION_HIDDEN_KEY]
        assert isinstance(hidden, np.ndarray), f"{OBSERVATION_HIDDEN_KEY} must be ndarray"
        assert hidden.ndim == 1, f"{OBSERVATION_HIDDEN_KEY} must be 1-D, got {hidden.shape}"
        assert np.issubdtype(hidden.dtype, np.floating), f"{OBSERVATION_HIDDEN_KEY} must be floating"
    if OBSERVATION_ENV_KEY in frame:
        obs_env = frame[OBSERVATION_ENV_KEY]
        assert isinstance(obs_env, np.ndarray), f"{OBSERVATION_ENV_KEY} must be ndarray"
        assert obs_env.ndim == 1, f"{OBSERVATION_ENV_KEY} must be 1-D, got {obs_env.shape}"
        assert np.issubdtype(obs_env.dtype, np.floating), f"{OBSERVATION_ENV_KEY} must be floating"
    has_action_prefix = ACTION_PREFIX_KEY in frame
    has_prefix_change_start = PREFIX_CHANGE_START_KEY in frame
    assert has_action_prefix == has_prefix_change_start, (
        f"{ACTION_PREFIX_KEY} and {PREFIX_CHANGE_START_KEY} must be provided as a pair, or not at all."
    )
    if has_action_prefix:
        action_prefix = frame[ACTION_PREFIX_KEY]
        assert isinstance(action_prefix, np.ndarray), f"{ACTION_PREFIX_KEY} must be ndarray"
        assert action_prefix.ndim == 2, f"{ACTION_PREFIX_KEY} must be 2-D, got {action_prefix.shape}"
        assert action_prefix.shape[-1] == expected_action_dim, (
            f"{ACTION_PREFIX_KEY} must have shape {('*', expected_action_dim)}, got {action_prefix.shape}"
        )
        assert np.issubdtype(action_prefix.dtype, np.floating), (
            f"{ACTION_PREFIX_KEY} must be floating ndarray, got {action_prefix.dtype}"
        )
        prefix_change_start = frame[PREFIX_CHANGE_START_KEY]
        assert isinstance(prefix_change_start, int), f"{PREFIX_CHANGE_START_KEY} must be int"
        assert 0 < prefix_change_start < 50, f"{PREFIX_CHANGE_START_KEY} must be in [1, 49], got {prefix_change_start}"
    _validate_joint_position_array(frame[JOINT_STATE_KEY], hardware_model)
    for k in _observation_keys(hardware_model):
        v = frame[k]
        assert isinstance(v, (bytes, np.ndarray)), f"{k} must be jpeg bytes or ndarray, got {type(v)}"
    if TABLE_VIEW_IMAGE_KEY in frame:
        table_view = frame[TABLE_VIEW_IMAGE_KEY]
        assert isinstance(table_view, (bytes, np.ndarray)), (
            f"{TABLE_VIEW_IMAGE_KEY} must be jpeg bytes or ndarray, got {type(table_view)}"
        )
    return hardware_model


def validate_wire_intervention_request_frame(
    frame: dict[str, Any], hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL
) -> HardwareModel:
    assert frame.get(ENDPOINT_KEY) == ENDPOINT_INTERVENTION, (
        f"intervention frame must contain {ENDPOINT_KEY}={ENDPOINT_INTERVENTION!r}"
    )
    inference_frame = {
        key: value
        for key, value in frame.items()
        if key not in {ENDPOINT_KEY, ACTION_KEY, PREV_SKIPPED_ACTION_START_KEY}
    }
    validate_wire_inference_request_frame(inference_frame, hardware_model)

    action_chunk = frame.get(ACTION_KEY)
    assert isinstance(action_chunk, np.ndarray), f"{ACTION_KEY} must be ndarray"
    assert action_chunk.ndim == 2, f"{ACTION_KEY} must be 2-D, got {action_chunk.shape}"
    assert action_chunk.shape[0] > 0, f"{ACTION_KEY} must contain at least one action, got {action_chunk.shape}"
    assert action_chunk.shape[1] == hardware_model.action_dim, (
        f"{ACTION_KEY} second dim must be {hardware_model.action_dim}, got {action_chunk.shape}"
    )
    assert np.issubdtype(action_chunk.dtype, np.floating), f"{ACTION_KEY} must be floating ndarray"

    if PREV_SKIPPED_ACTION_START_KEY in frame:
        prev_skipped_action_start = frame[PREV_SKIPPED_ACTION_START_KEY]
        assert isinstance(prev_skipped_action_start, int), f"{PREV_SKIPPED_ACTION_START_KEY} must be int"
        assert 0 <= prev_skipped_action_start < action_chunk.shape[0], (
            f"{PREV_SKIPPED_ACTION_START_KEY} must be in [0, {action_chunk.shape[0]}), got {prev_skipped_action_start}"
        )
    return hardware_model


def validate_wire_inference_response(
    result: dict[str, Any],
    hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    response_summary = _summarize_response_payload(result)
    assert "error" not in result, f"unexpected error payload: {response_summary}"
    allowed = frozenset({ACTION_KEY, INFERENCE_TIME_KEY, POLICY_ID_KEY, RL_ENABLED_KEY, Q_VALUE_KEY, CHUNK_ID_KEY})
    assert set(result.keys()) <= allowed, (
        f"response keys {set(result.keys())} not subset of {allowed}; summary={response_summary}"
    )
    assert ACTION_KEY in result, f"response missing action; summary={response_summary}"
    actions = result[ACTION_KEY]
    assert isinstance(actions, np.ndarray), f"action must be ndarray, got {type(actions)}; summary={response_summary}"
    assert actions.ndim == 2, f"action must be 2-D, got shape {actions.shape}"
    assert actions.shape[1] == hardware_model.action_dim, (
        f"action second dim must be {hardware_model.action_dim}, got {actions.shape}"
    )
    assert np.issubdtype(actions.dtype, np.floating), f"action must be floating ndarray, got {actions.dtype}"
    if INFERENCE_TIME_KEY in result:
        assert isinstance(result[INFERENCE_TIME_KEY], (int, float)), "inference_time must be numeric"
    if POLICY_ID_KEY in result:
        assert isinstance(result[POLICY_ID_KEY], str), f"{POLICY_ID_KEY} must be str"
    if RL_ENABLED_KEY in result:
        assert isinstance(result[RL_ENABLED_KEY], bool), f"{RL_ENABLED_KEY} must be bool"
    if Q_VALUE_KEY in result:
        assert isinstance(result[Q_VALUE_KEY], (int, float)), f"{Q_VALUE_KEY} must be numeric"


__all__ = [
    "DEFAULT_HARDWARE_MODEL",
    "HardwareModel",
    "server_handshake_for_hardware_model",
    "validate_ultra_arrays_for_hardware_model",
    "validate_wire_inference_request_frame",
    "validate_wire_intervention_request_frame",
    "validate_wire_inference_response",
]
