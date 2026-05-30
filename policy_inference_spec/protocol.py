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

JOINT_STATE_KEY = "observation/state"
PROMPT_KEY = "prompt"
ACTION_KEY = "action"
CONTEXT_EMBEDDINGS_KEY = "context_embeddings"
INFERENCE_TIME_KEY = "inference_time"
ENDPOINT_KEY = "endpoint"
ENDPOINT_RESET = "reset"
ENDPOINT_TELEMETRY = "telemetry"
ENDPOINT_REWARD = "reward"
MODEL_ID_KEY = "model_id"
POLICY_ID_KEY = "policy_id"
DUMB_REWARD_GOAL_ACTION_CHUNK_KEY = "dumb_reward_goal_action_chunk"
DUMB_REWARD_THRESHOLD_KEY = "dumb_reward_threshold"
FAST_MOCK_ACTION_DIM_KEY = "fast_mock_action_dim"
FAST_MOCK_ACTION_HORIZON_KEY = "fast_mock_action_horizon"
REWARDS_H_KEY = "rewards_h"
ACTION_PREFIX_KEY = "action_prefix"
PREFIX_CHANGE_START_KEY = "prefix_change_start"
REWARD_DESCRIPTION_KEY = "description"
TIMESTAMP_KEY = "timestamp"

CAMERA_NAMES_KEY = "camera_names"
IMAGE_RESOLUTION_KEY = "image_resolution"
ACTION_SPACE_KEY = "action_space"
NEEDS_WRIST_CAMERA_KEY = "needs_wrist_camera"
N_EXTERNAL_CAMERAS_KEY = "n_external_cameras"
SERVER_FEATURES_KEY = "server_features"
STATUS_KEY = "status"
ERROR_KEY = "error"
CONTEXT_EMBEDDING_TOKENS = 2
CONTEXT_EMBEDDING_WIDTH = 128

InferenceMetadataValue: TypeAlias = str | int | float | bool | list[str] | list[int] | list[float]
ImageArray: TypeAlias = npt.NDArray[np.uint8]
FloatArray: TypeAlias = npt.NDArray[np.float32]
ProtocolValue: TypeAlias = ImageArray | FloatArray | bytes | InferenceMetadataValue
ProtocolPayload: TypeAlias = dict[str, ProtocolValue]


class ServerFeature(StrEnum):
    REWARDS = "rewards"


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
        assert all(isinstance(name, str) and name for name in self.camera_names), "camera_names must be non-empty strings"
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
        assert all(isinstance(feature, str) for feature in server_features_raw), f"{SERVER_FEATURES_KEY} must be list[str]"
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


@dataclass(frozen=True)
class RewardSignal:
    rewards_h: tuple[float, ...] = (1.0,)
    description: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.rewards_h, tuple), f"{REWARDS_H_KEY} must be tuple[float, ...]"
        assert self.rewards_h, f"{REWARDS_H_KEY} must not be empty"
        assert all(isinstance(reward, (int, float)) for reward in self.rewards_h), f"{REWARDS_H_KEY} must be numeric"
        if self.description is not None:
            assert isinstance(self.description, str), f"{REWARD_DESCRIPTION_KEY} must be str"

    def to_payload(self) -> ProtocolPayload:
        payload: ProtocolPayload = {
            ENDPOINT_KEY: ENDPOINT_REWARD,
            REWARDS_H_KEY: [float(reward) for reward in self.rewards_h],
        }
        if self.description is not None:
            payload[REWARD_DESCRIPTION_KEY] = self.description
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RewardSignal:
        assert isinstance(payload, Mapping), f"reward payload must be mapping, got {type(payload)}"
        assert payload.get(ENDPOINT_KEY) == ENDPOINT_REWARD, (
            f"{ENDPOINT_KEY} must be {ENDPOINT_REWARD!r}, got {payload.get(ENDPOINT_KEY)!r}"
        )
        rewards_h_raw = payload.get(REWARDS_H_KEY, [1.0])
        assert isinstance(rewards_h_raw, list), f"{REWARDS_H_KEY} must be list[float]"
        assert rewards_h_raw, f"{REWARDS_H_KEY} must not be empty"
        assert all(isinstance(reward, (int, float)) for reward in rewards_h_raw), f"{REWARDS_H_KEY} must be numeric"
        description = payload.get(REWARD_DESCRIPTION_KEY)
        assert description is None or isinstance(description, str), f"{REWARD_DESCRIPTION_KEY} must be str"
        return cls(rewards_h=tuple(float(reward) for reward in rewards_h_raw), description=description)


__all__ = [
    "ACTION_KEY",
    "ACTION_SPACE_KEY",
    "CAMERA_NAMES_KEY",
    "CONTEXT_EMBEDDINGS_KEY",
    "CONTEXT_EMBEDDING_TOKENS",
    "CONTEXT_EMBEDDING_WIDTH",
    "DEFAULT_INFERENCE_SERVER_PORT",
    "DUMB_REWARD_GOAL_ACTION_CHUNK_KEY",
    "DUMB_REWARD_THRESHOLD_KEY",
    "ENDPOINT_KEY",
    "ENDPOINT_RESET",
    "ENDPOINT_REWARD",
    "ENDPOINT_TELEMETRY",
    "ERROR_KEY",
    "FAST_MOCK_ACTION_DIM_KEY",
    "FAST_MOCK_ACTION_HORIZON_KEY",
    "FloatArray",
    "IMAGE_RESOLUTION_KEY",
    "ImageArray",
    "INFERENCE_TIME_KEY",
    "InferenceMetadataValue",
    "JOINT_STATE_KEY",
    "MODEL_ID_KEY",
    "N_EXTERNAL_CAMERAS_KEY",
    "NEEDS_WRIST_CAMERA_KEY",
    "POLICY_ID_KEY",
    "PROMPT_KEY",
    "ProtocolPayload",
    "ProtocolValue",
    "REWARD_DESCRIPTION_KEY",
    "REWARDS_H_KEY",
    "RewardSignal",
    "SERVER_FEATURES_KEY",
    "STATUS_KEY",
    "ServerFeature",
    "ServerHandshake",
    "make_server_handshake",
]
