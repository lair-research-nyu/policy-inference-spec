from __future__ import annotations

from typing import Any, cast

import msgspec
import numpy as np
import pytest
import simplejpeg
from beartype.roar import BeartypeCallHintParamViolation

from policy_inference_spec.codec import NdarrayField, deserialize_from_msgpack, encode_image, serialize_to_msgpack
from policy_inference_spec.protocol import (
    ACTION_KEY,
    CONDITIONING_METADATA_KEY,
    FloatArray,
    ProtocolPayload,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    REWARD_ACTION_INDEX_KEY,
    REWARD_KEY,
    START_METADATA_KEY,
    SUBTASK_KEY,
    TABLE_VIEW_IMAGE_KEY,
    TASK_KEY,
    ServerFeature,
    ServerHandshake,
)
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)


def test_serialize_to_msgpack_uses_flat_ndarray_tags() -> None:
    expected = cast(FloatArray, np.arange(6, dtype=np.float32).reshape(2, 3))
    payload: ProtocolPayload = {"action": expected}
    encoded = serialize_to_msgpack(payload)
    raw = msgspec.msgpack.decode(encoded)

    assert raw["action"]["__ndarray__"] is True
    assert raw["action"]["dtype"] == "float32"
    assert raw["action"]["shape"] == [2, 3]
    assert isinstance(raw["action"]["data"], bytes)


def test_deserialize_from_msgpack_accepts_flat_byte_key_ndarray_tags() -> None:
    expected = cast(FloatArray, np.arange(6, dtype=np.float32).reshape(2, 3))
    payload = {
        "action": {
            b"__ndarray__": True,
            b"data": expected.tobytes(),
            b"dtype": "float32",
            b"shape": [2, 3],
        }
    }

    decoded = deserialize_from_msgpack(msgspec.msgpack.encode(payload))

    assert isinstance(decoded["action"], np.ndarray)
    assert decoded["action"].dtype == np.float32
    assert decoded["action"].shape == (2, 3)
    assert np.array_equal(decoded["action"], expected)


def test_serialize_to_msgpack_rejects_float64_ndarray() -> None:
    payload: Any = {"action": np.arange(6, dtype=np.float64).reshape(2, 3)}

    with pytest.raises(BeartypeCallHintParamViolation):
        serialize_to_msgpack(cast(Any, payload))


def test_deserialize_from_msgpack_rejects_float64_ndarray_tag() -> None:
    payload = {
        "action": {
            b"__ndarray__": True,
            b"data": np.arange(6, dtype=np.float64).reshape(2, 3).tobytes(),
            b"dtype": "float64",
            b"shape": [2, 3],
        }
    }

    with pytest.raises(AssertionError, match="Unsupported ndarray dtype"):
        deserialize_from_msgpack(msgspec.msgpack.encode(payload))


def test_validate_wire_inference_response_summarizes_binary_like_payloads() -> None:
    with pytest.raises(AssertionError) as exc_info:
        validate_wire_inference_response(
            {
                ACTION_KEY: {
                    "__ndarray__": True,
                    "data": b"\x00" * 32,
                    "dtype": "float64",
                    "shape": [2, 2],
                },
            }
        )

    message = str(exc_info.value)
    assert "summary=" in message
    assert "dict(keys=['__ndarray__', 'data', 'dtype', 'shape'])" in message
    assert "\\x00" not in message


def test_encode_image_preserves_original_shape_metadata() -> None:
    expected = np.zeros((1, 12, 16, 3), dtype=np.uint8)

    encoded = encode_image(expected, jpeg_quality=75)
    decoded = simplejpeg.decode_jpeg(encoded.data)

    assert isinstance(encoded, NdarrayField)
    assert encoded.codec == "jpeg"
    assert encoded.shape == expected.shape
    assert encoded.dtype == str(expected.dtype)
    assert decoded.shape == (12, 16, 3)
    assert decoded.dtype == expected.dtype


def test_server_handshake_round_trip_preserves_server_features() -> None:
    handshake = ServerHandshake(
        camera_names=("images/main_image", "images/left_wrist_image"),
        image_resolution=(360, 640),
        server_features=(ServerFeature.REWARDS.value,),
    )

    decoded = ServerHandshake.from_payload(handshake.to_payload())

    assert decoded == handshake
    assert decoded.supports(ServerFeature.REWARDS)


def test_validate_wire_inference_request_frame_accepts_optional_table_view_image() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "",
        SUBTASK_KEY: "",
        MODEL_ID_KEY: "",
        TABLE_VIEW_IMAGE_KEY: np.zeros((300, 480, 3), dtype=np.uint8),
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_accepts_missing_table_view_image() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "",
        SUBTASK_KEY: "",
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    validate_wire_inference_request_frame(payload)
    assert TABLE_VIEW_IMAGE_KEY not in payload


def test_validate_wire_inference_request_frame_accepts_scalar_reward() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "",
        SUBTASK_KEY: "",
        MODEL_ID_KEY: "",
        REWARD_KEY: 1.25,
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_accepts_reward_action_index() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "",
        SUBTASK_KEY: "",
        MODEL_ID_KEY: "",
        REWARD_KEY: 1.25,
        REWARD_ACTION_INDEX_KEY: 12,
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_rejects_out_of_range_reward_action_index() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "",
        SUBTASK_KEY: "",
        MODEL_ID_KEY: "",
        REWARD_ACTION_INDEX_KEY: 50,
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    with pytest.raises(AssertionError, match=REWARD_ACTION_INDEX_KEY):
        validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_accepts_start_metadata() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "bulk_shipping",
        SUBTASK_KEY: "pick_item",
        START_METADATA_KEY: {
            "item_index": 2,
            "pick_location": "table",
            "nested": {"drop_off_location": "Blue Bin"},
        },
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_accepts_conditioning_metadata() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "bulk_shipping",
        SUBTASK_KEY: "pick_item",
        CONDITIONING_METADATA_KEY: {
            "operator_ldap": "operator",
            "post_hoc_score": 4,
            "mistake": "false",
            "episode_length": 1200,
        },
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_rejects_non_numeric_reward() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "",
        SUBTASK_KEY: "",
        MODEL_ID_KEY: "",
        REWARD_KEY: "1.25",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    with pytest.raises(AssertionError, match=REWARD_KEY):
        validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_rejects_non_json_start_metadata() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "bulk_shipping",
        SUBTASK_KEY: "pick_item",
        START_METADATA_KEY: {"bad": object()},
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    with pytest.raises(AssertionError, match=START_METADATA_KEY):
        validate_wire_inference_request_frame(payload)


def test_validate_wire_inference_request_frame_rejects_non_json_conditioning_metadata() -> None:
    payload: ProtocolPayload = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "bulk_shipping",
        SUBTASK_KEY: "pick_item",
        CONDITIONING_METADATA_KEY: {"bad": object()},
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        payload[f"observation/{camera}"] = np.zeros(DEFAULT_HARDWARE_MODEL.image_resolution + (3,), dtype=np.uint8)

    with pytest.raises(AssertionError, match=CONDITIONING_METADATA_KEY):
        validate_wire_inference_request_frame(payload)
