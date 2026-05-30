from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass

import cv2
import numpy as np


@dataclass(frozen=True)
class ScalarFeature:
    name: str
    dtype: str
    shape: int
    rrd_entity_path: str
    rrd_component: str = "Scalars:scalars"


@dataclass(frozen=True)
class VideoFeature:
    name: str
    shape: tuple[int, int, int]
    rrd_entity_path: str
    dtype: str = "uint8"
    gop_size: int = 10
    downsample_factor: float = 2.0
    crop_to_mono: bool = False

    @property
    def rrd_component(self) -> str:
        return "VideoStream:sample"

    @property
    def resolution(self) -> tuple[int, int]:
        height, width, channels = self.shape
        assert channels == 3, f"Expected 3-channel video shape, got {self.shape}"
        return (height, width)


Feature = ScalarFeature | VideoFeature


@dataclass
class FeatureBundle:
    name: str
    observations: list[ScalarFeature]
    actions: list[ScalarFeature]
    videos: list[VideoFeature]
    fps: int = 50

    @property
    def all_scalar_features(self) -> list[ScalarFeature]:
        return self.observations + self.actions

    @property
    def all_features(self) -> list[Feature]:
        return [*self.observations, *self.actions, *self.videos]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.all_features]

    @property
    def state_dim(self) -> int:
        return sum(feature.shape for feature in self.observations)

    @property
    def action_dim(self) -> int:
        return sum(feature.shape for feature in self.actions)

    @property
    def vectors_schema(self) -> dict[str, int]:
        return {feature.name: feature.shape for feature in self.all_scalar_features}

    @property
    def camera_stream_schema(self) -> dict[str, tuple[int, int, int]]:
        return {
            feature.name: (feature.resolution[0], feature.resolution[1], feature.gop_size) for feature in self.videos
        }

    def preprocess(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated_data = self._validate_features(data, self.all_features)
        result = {feature.name: validated_data[feature.name] for feature in self.videos}
        assert self.observations, "observation.state requires at least one scalar feature"
        result["observation.state"] = np.concatenate(
            [validated_data[feature.name] for feature in self.observations],
            axis=-1,
        )
        assert self.actions, "action requires at least one scalar feature"
        result["action"] = np.concatenate([validated_data[feature.name] for feature in self.actions], axis=-1)
        return result

    def preprocess_observations(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated_data = self._validate_features(data, [*self.observations, *self.videos])
        result = {feature.name: validated_data[feature.name] for feature in self.videos}
        assert self.observations, "observation.state requires at least one scalar feature"
        result["observation.state"] = np.concatenate(
            [validated_data[feature.name] for feature in self.observations],
            axis=-1,
        )
        return result

    def parse_actions(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        assert actions.shape[-1] == self.action_dim, (
            f"Expected action shape ending with {self.action_dim}, got {actions.shape}"
        )

        result: dict[str, np.ndarray] = {}
        offset = 0
        for feature in self.actions:
            next_offset = offset + feature.shape
            action_feature = actions[..., offset:next_offset]
            expected_dtype = np.dtype(feature.dtype)
            assert action_feature.dtype == expected_dtype, (
                f"Feature '{feature.name}': expected dtype {expected_dtype}, got {action_feature.dtype}"
            )
            assert tuple(action_feature.shape[-1:]) == (feature.shape,), (
                f"Feature '{feature.name}': expected shape ending with {(feature.shape,)}, got {action_feature.shape}"
            )
            result[feature.name] = action_feature
            offset = next_offset

        assert offset == self.action_dim, f"Expected parsed action dim {self.action_dim}, got {offset}"
        return result

    def _validate_features(
        self,
        data: dict[str, np.ndarray],
        features: Sequence[Feature],
    ) -> dict[str, np.ndarray]:
        expected_keys = {feature.name for feature in features}
        actual_keys = set(data.keys())
        missing_keys = expected_keys - actual_keys
        assert not missing_keys, f"Missing keys in data: {missing_keys}"

        for feature in features:
            processed = data[feature.name]
            expected_dtype = np.dtype(feature.dtype)
            assert processed.dtype == expected_dtype, (
                f"Feature '{feature.name}': expected dtype {expected_dtype}, got {processed.dtype}"
            )
            expected_shape = (feature.shape,) if isinstance(feature, ScalarFeature) else feature.shape
            assert tuple(processed.shape[-len(expected_shape) :]) == expected_shape, (
                f"Feature '{feature.name}': expected shape ending with {expected_shape}, got {processed.shape}"
            )

        return data


def preprocess_image(
    image_hwc: np.ndarray,
    *,
    downsample_factor: float = 1.0,
    crop_to_mono: bool = False,
    output_shape_hwc: tuple[int, int, int] | None = None,
) -> np.ndarray:
    assert image_hwc.ndim == 3, f"Expected HWC image, got shape {image_hwc.shape}"
    assert image_hwc.shape[2] == 3, f"Expected 3-channel image, got shape {image_hwc.shape}"
    assert image_hwc.dtype == np.uint8, f"Expected uint8 image, got {image_hwc.dtype}"
    assert downsample_factor > 0.0, f"downsample_factor must be positive, got {downsample_factor}"

    result = image_hwc
    if crop_to_mono:
        cropped_width = result.shape[1] // 2
        assert cropped_width > 0, f"Cannot crop image with width {result.shape[1]} to mono"
        result = result[:, :cropped_width, :]

    if output_shape_hwc is None:
        output_width = int(round(result.shape[1] / downsample_factor))
        output_height = int(round(result.shape[0] / downsample_factor))
        assert output_width > 0 and output_height > 0, (
            f"Invalid output resolution {(output_width, output_height)} from input {result.shape}"
        )
        output_shape_hwc = (output_height, output_width, 3)
    else:
        output_height, output_width, output_channels = output_shape_hwc
        assert output_height > 0 and output_width > 0, (
            f"Invalid requested output resolution {(output_width, output_height)}"
        )
        assert output_channels == 3, f"Expected requested output shape to have 3 channels, got {output_shape_hwc}"

    if result.shape != output_shape_hwc:
        result = np.asarray(
            cv2.resize(result, dsize=(output_shape_hwc[1], output_shape_hwc[0]), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )

    result = np.ascontiguousarray(result)
    assert result.shape == output_shape_hwc, f"Expected output shape {output_shape_hwc}, got {result.shape}"
    return result


class SchemaName(StrEnum):
    GEN2_28D_STATE = "gen2-28d-state"
    GEN2_32D_STATE = "gen2-32d-state"
    GEN2_28D_STATE_POSE_ACTIONS = "gen2-28d-state-pose-actions"
    GEN2_32D_STATE_POSE_ACTIONS = "gen2-32d-state-pose-actions"


def get_feature_bundle_for_schema(schema: SchemaName) -> FeatureBundle:
    if schema == SchemaName.GEN2_28D_STATE:
        return _build_gen2_training_bundle(arm_state_dim=28, use_pose_actions=False)
    if schema == SchemaName.GEN2_32D_STATE:
        return _build_gen2_training_bundle(arm_state_dim=32, use_pose_actions=False)
    if schema == SchemaName.GEN2_28D_STATE_POSE_ACTIONS:
        return _build_gen2_training_bundle(arm_state_dim=28, use_pose_actions=True)
    if schema == SchemaName.GEN2_32D_STATE_POSE_ACTIONS:
        return _build_gen2_training_bundle(arm_state_dim=32, use_pose_actions=True)
    raise ValueError(f"Invalid SchemaName {schema}")


def _build_gen2_training_bundle(arm_state_dim: int, *, use_pose_actions: bool) -> FeatureBundle:
    assert arm_state_dim in {28, 32}, f"Unsupported arm_state_dim: {arm_state_dim}"
    observations = [
        ScalarFeature(
            name="/state/left_arm:Scalars:scalars",
            dtype="float32",
            shape=arm_state_dim,
            rrd_entity_path="/state/left_arm",
        ),
        ScalarFeature(
            name="/state/right_arm:Scalars:scalars",
            dtype="float32",
            shape=arm_state_dim,
            rrd_entity_path="/state/right_arm",
        ),
        ScalarFeature(
            name="/state/chest:Scalars:scalars",
            dtype="float32",
            shape=24,
            rrd_entity_path="/state/chest",
        ),
        ScalarFeature(
            name="/state/neck:Scalars:scalars",
            dtype="float32",
            shape=9,
            rrd_entity_path="/state/neck",
        ),
    ]
    actions = _build_gen2_actions(use_pose_actions=use_pose_actions)
    videos = [
        VideoFeature(name="head", shape=(360, 640, 3), rrd_entity_path="/cameras/head", crop_to_mono=True),
        VideoFeature(
            name="left_wrist",
            shape=(300, 480, 3),
            rrd_entity_path="/cameras/left_wrist",
        ),
        VideoFeature(
            name="right_wrist",
            shape=(300, 480, 3),
            rrd_entity_path="/cameras/right_wrist",
        ),
    ]
    return FeatureBundle(
        name=(
            f"gen2-training-{arm_state_dim}d-state-pose-actions"
            if use_pose_actions
            else f"gen2-training-{arm_state_dim}d-state"
        ),
        observations=observations,
        actions=actions,
        videos=videos,
        fps=50,
    )


def _build_gen2_actions(*, use_pose_actions: bool) -> list[ScalarFeature]:
    if use_pose_actions:
        return [
            ScalarFeature(
                name="/commanded_pose/left_arm:Transform3D:quaternion",
                dtype="float32",
                shape=4,
                rrd_entity_path="/commanded_pose/left_arm",
                rrd_component="Transform3D:quaternion",
            ),
            ScalarFeature(
                name="/commanded_pose/left_arm:Transform3D:translation",
                dtype="float32",
                shape=3,
                rrd_entity_path="/commanded_pose/left_arm",
                rrd_component="Transform3D:translation",
            ),
            ScalarFeature(
                name="/commanded_qpos/left_arm_ee:Scalars:scalars",
                dtype="float32",
                shape=1,
                rrd_entity_path="/commanded_qpos/left_arm_ee",
            ),
            ScalarFeature(
                name="/commanded_pose/right_arm:Transform3D:quaternion",
                dtype="float32",
                shape=4,
                rrd_entity_path="/commanded_pose/right_arm",
                rrd_component="Transform3D:quaternion",
            ),
            ScalarFeature(
                name="/commanded_pose/right_arm:Transform3D:translation",
                dtype="float32",
                shape=3,
                rrd_entity_path="/commanded_pose/right_arm",
                rrd_component="Transform3D:translation",
            ),
            ScalarFeature(
                name="/commanded_qpos/right_arm_ee:Scalars:scalars",
                dtype="float32",
                shape=1,
                rrd_entity_path="/commanded_qpos/right_arm_ee",
            ),
            ScalarFeature(
                name="/commanded_pose/neck:Transform3D:quaternion",
                dtype="float32",
                shape=4,
                rrd_entity_path="/commanded_pose/neck",
                rrd_component="Transform3D:quaternion",
            ),
            ScalarFeature(
                name="/commanded_pose/neck:Transform3D:translation",
                dtype="float32",
                shape=3,
                rrd_entity_path="/commanded_pose/neck",
                rrd_component="Transform3D:translation",
            ),
        ]

    return [
        ScalarFeature(
            name="/commanded_qpos/left_arm:Scalars:scalars",
            dtype="float32",
            shape=7,
            rrd_entity_path="/commanded_qpos/left_arm",
        ),
        ScalarFeature(
            name="/commanded_qpos/left_arm_ee:Scalars:scalars",
            dtype="float32",
            shape=1,
            rrd_entity_path="/commanded_qpos/left_arm_ee",
        ),
        ScalarFeature(
            name="/commanded_qpos/right_arm:Scalars:scalars",
            dtype="float32",
            shape=7,
            rrd_entity_path="/commanded_qpos/right_arm",
        ),
        ScalarFeature(
            name="/commanded_qpos/right_arm_ee:Scalars:scalars",
            dtype="float32",
            shape=1,
            rrd_entity_path="/commanded_qpos/right_arm_ee",
        ),
        ScalarFeature(
            name="/commanded_qpos/chest:Scalars:scalars",
            dtype="float32",
            shape=6,
            rrd_entity_path="/commanded_qpos/chest",
        ),
        ScalarFeature(
            name="/commanded_qpos/neck:Scalars:scalars",
            dtype="float32",
            shape=3,
            rrd_entity_path="/commanded_qpos/neck",
        ),
    ]
