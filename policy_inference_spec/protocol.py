from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

DEFAULT_INFERENCE_SERVER_PORT = 18090

# Observation payload keys
JOINT_STATE_KEY = "observation/state"
OBSERVATION_HIDDEN_KEY = "observation/hidden"
OBSERVATION_ENV_KEY = "observation/env"
# Optional hardware camera (stations may omit table-view).
TABLE_VIEW_IMAGE_KEY = "observation/images/table_view_image"

# Action payload keys
ACTION_KEY = "action"
ACTION_PREFIX_KEY = "action_prefix"
PREFIX_CHANGE_START_KEY = "prefix_change_start"
PREV_SKIPPED_ACTION_START_KEY = "prev_skipped_action_start"

# Request metadata keys
MODEL_ID_KEY = "model_id"
POLICY_ID_KEY = "policy_id"
TASK_KEY = "task"
SUBTASK_KEY = "subtask"
START_METADATA_KEY = "start_metadata"
CONDITIONING_METADATA_KEY = "conditioning_metadata"

# Endpoint keys and values
ENDPOINT_KEY = "endpoint"
ENDPOINT_INTERVENTION = "intervention"
ENDPOINT_DONE = "done"
ENDPOINT_TELEMETRY = "telemetry"

# Reward and retrospective episode keys.
REWARD_KEY = "reward"
REWARD_ACTION_INDEX_KEY = "reward_action_index"
REWARD_DESCRIPTION_KEY = "description"
DONE_KEY = "done"
DONE_REASON_KEY = "done_reason"

# Server handshake keys
CAMERA_NAMES_KEY = "camera_names"
IMAGE_RESOLUTION_KEY = "image_resolution"
ACTION_SPACE_KEY = "action_space"
NEEDS_WRIST_CAMERA_KEY = "needs_wrist_camera"
N_EXTERNAL_CAMERAS_KEY = "n_external_cameras"
SERVER_FEATURES_KEY = "server_features"

# Request/response chunk tracking
CHUNK_ID_KEY = "chunk_id"

# Response status keys
INFERENCE_TIME_KEY = "inference_time"
STATUS_KEY = "status"
ERROR_KEY = "error"
RL_ENABLED_KEY = "rl_enabled"
Q_VALUE_KEY = "q_value"

InferenceMetadataValue: TypeAlias = None | str | int | float | bool | list[Any] | dict[str, Any]
ImageArray: TypeAlias = npt.NDArray[np.uint8]
FloatArray: TypeAlias = npt.NDArray[np.float32]
ProtocolValue: TypeAlias = ImageArray | FloatArray | bytes | InferenceMetadataValue
ProtocolPayload: TypeAlias = dict[str, ProtocolValue]


class ServerFeature(StrEnum):
    REWARDS = "rewards"
    HUMAN_INTERVENTIONS = "human_interventions"


def _normalize_server_features(features: Iterable[str | ServerFeature]) -> tuple[str, ...]:
    normalized = tuple(str(feature) for feature in features)
    assert all(feature for feature in normalized), "server_features must not contain empty strings"
    return normalized


def _parse_optional_image_resolution(raw: Any) -> tuple[int, int] | None:
    if raw is None:
        return None
    assert isinstance(raw, (list, tuple)) and len(raw) == 2, f"{IMAGE_RESOLUTION_KEY} must be a 2-item list or tuple"
    assert all(isinstance(dim, (int, float)) for dim in raw), f"{IMAGE_RESOLUTION_KEY} dims must be numeric"
    height, width = int(raw[0]), int(raw[1])
    assert height > 0 and width > 0, f"{IMAGE_RESOLUTION_KEY} dims must be positive, got {(height, width)}"
    return height, width


@dataclass(frozen=True)
class ServerHandshake:
    camera_names: tuple[str, ...]
    image_resolution: tuple[int, int] | None = None
    action_space: str = "joint_position"
    needs_wrist_camera: bool = True
    n_external_cameras: int = 1
    server_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert self.camera_names, "camera_names must not be empty"
        assert all(isinstance(name, str) and name for name in self.camera_names), (
            "camera_names must be non-empty strings"
        )
        assert isinstance(self.action_space, str) and self.action_space, "action_space must be a non-empty string"
        assert isinstance(self.needs_wrist_camera, bool), "needs_wrist_camera must be bool"
        assert isinstance(self.n_external_cameras, int), "n_external_cameras must be int"
        assert self.n_external_cameras >= 0, "n_external_cameras must be non-negative"
        assert all(isinstance(feature, str) and feature for feature in self.server_features), (
            "server_features must be non-empty strings"
        )

    def supports(self, feature: str | ServerFeature) -> bool:
        return str(feature) in self.server_features

    def to_payload(self) -> ProtocolPayload:
        payload: ProtocolPayload = {
            CAMERA_NAMES_KEY: list(self.camera_names),
            ACTION_SPACE_KEY: self.action_space,
            NEEDS_WRIST_CAMERA_KEY: self.needs_wrist_camera,
            N_EXTERNAL_CAMERAS_KEY: self.n_external_cameras,
        }
        if self.image_resolution is not None:
            payload[IMAGE_RESOLUTION_KEY] = list(self.image_resolution)
        if self.server_features:
            payload[SERVER_FEATURES_KEY] = list(self.server_features)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ServerHandshake:
        assert isinstance(payload, Mapping), f"handshake payload must be mapping, got {type(payload)}"
        camera_names_raw = payload.get(CAMERA_NAMES_KEY)
        assert isinstance(camera_names_raw, list), f"{CAMERA_NAMES_KEY} must be list[str]"
        assert all(isinstance(name, str) for name in camera_names_raw), f"{CAMERA_NAMES_KEY} must be list[str]"
        image_resolution = _parse_optional_image_resolution(payload.get(IMAGE_RESOLUTION_KEY))
        action_space = payload.get(ACTION_SPACE_KEY, "joint_position")
        assert isinstance(action_space, str), f"{ACTION_SPACE_KEY} must be str"
        needs_wrist_camera = payload.get(NEEDS_WRIST_CAMERA_KEY, True)
        assert isinstance(needs_wrist_camera, bool), f"{NEEDS_WRIST_CAMERA_KEY} must be bool"
        n_external_cameras = payload.get(N_EXTERNAL_CAMERAS_KEY, 1)
        assert isinstance(n_external_cameras, int), f"{N_EXTERNAL_CAMERAS_KEY} must be int"
        server_features_raw = payload.get(SERVER_FEATURES_KEY, [])
        assert isinstance(server_features_raw, list), f"{SERVER_FEATURES_KEY} must be list[str]"
        assert all(isinstance(feature, str) for feature in server_features_raw), (
            f"{SERVER_FEATURES_KEY} must be list[str]"
        )
        return cls(
            camera_names=tuple(camera_names_raw),
            image_resolution=image_resolution,
            action_space=action_space,
            needs_wrist_camera=needs_wrist_camera,
            n_external_cameras=n_external_cameras,
            server_features=_normalize_server_features(server_features_raw),
        )


def make_server_handshake(
    *,
    camera_names: Iterable[str],
    image_resolution: tuple[int, int] | None = None,
    action_space: str = "joint_position",
    needs_wrist_camera: bool = True,
    n_external_cameras: int = 1,
    server_features: Iterable[str | ServerFeature] = (),
) -> ServerHandshake:
    return ServerHandshake(
        camera_names=tuple(camera_names),
        image_resolution=image_resolution,
        action_space=action_space,
        needs_wrist_camera=needs_wrist_camera,
        n_external_cameras=n_external_cameras,
        server_features=_normalize_server_features(server_features),
    )


__all__ = [
    "ACTION_KEY",
    "ACTION_PREFIX_KEY",
    "ACTION_SPACE_KEY",
    "CAMERA_NAMES_KEY",
    "CHUNK_ID_KEY",
    "CONDITIONING_METADATA_KEY",
    "DEFAULT_INFERENCE_SERVER_PORT",
    "DONE_KEY",
    "DONE_REASON_KEY",
    "ENDPOINT_KEY",
    "ENDPOINT_INTERVENTION",
    "ENDPOINT_DONE",
    "ENDPOINT_TELEMETRY",
    "ERROR_KEY",
    "FloatArray",
    "IMAGE_RESOLUTION_KEY",
    "ImageArray",
    "INFERENCE_TIME_KEY",
    "InferenceMetadataValue",
    "JOINT_STATE_KEY",
    "OBSERVATION_ENV_KEY",
    "OBSERVATION_HIDDEN_KEY",
    "TABLE_VIEW_IMAGE_KEY",
    "MODEL_ID_KEY",
    "N_EXTERNAL_CAMERAS_KEY",
    "NEEDS_WRIST_CAMERA_KEY",
    "POLICY_ID_KEY",
    "PREV_SKIPPED_ACTION_START_KEY",
    "PREFIX_CHANGE_START_KEY",
    "ProtocolPayload",
    "ProtocolValue",
    "Q_VALUE_KEY",
    "REWARD_ACTION_INDEX_KEY",
    "REWARD_DESCRIPTION_KEY",
    "REWARD_KEY",
    "RL_ENABLED_KEY",
    "START_METADATA_KEY",
    "SERVER_FEATURES_KEY",
    "STATUS_KEY",
    "SUBTASK_KEY",
    "TASK_KEY",
    "ServerFeature",
    "ServerHandshake",
    "make_server_handshake",
]
