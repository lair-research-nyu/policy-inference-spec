from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import simplejpeg
import websockets

from policy_inference_spec.client_helpers import (
    DEFAULT_PREDICT_URL,
    _emit_server_error_verbatim,
    _log_server_config,
    _summarize_server_payload,
    _wire_camera_names,
    policy_ws_url,
)
from policy_inference_spec.codec import deserialize_from_msgpack, encode_image, serialize_to_msgpack
from policy_inference_spec.hardware_model import (
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
    validate_wire_intervention_request_frame,
)
from policy_inference_spec.protocol import (
    ACTION_KEY,
    CHUNK_ID_KEY,
    DONE_KEY,
    DONE_REASON_KEY,
    ENDPOINT_DONE,
    ENDPOINT_INTERVENTION,
    ENDPOINT_KEY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    PREV_SKIPPED_ACTION_START_KEY,
    Q_VALUE_KEY,
    REWARD_ACTION_INDEX_KEY,
    REWARD_KEY,
    RL_ENABLED_KEY,
    SUBTASK_KEY,
    TASK_KEY,
    ServerFeature,
    ServerHandshake,
)

LOGGER = logging.getLogger(__name__)


def _validate_done_reason(*, done: bool, done_reason: str | None) -> str | None:
    if done:
        assert isinstance(done_reason, str) and done_reason.strip(), "done_reason must be a non-empty str when done=True"
        return done_reason.strip()
    assert done_reason is None, "done_reason must be None when done=False"
    return None


def _wire_image_to_hwc_uint8(value: Any) -> npt.NDArray[np.uint8]:
    if isinstance(value, bytes):
        return np.ascontiguousarray(simplejpeg.decode_jpeg(value)).astype(np.uint8, copy=False)
    assert isinstance(value, np.ndarray), f"image value must be bytes or ndarray, got {type(value)}"
    if value.ndim == 4:
        assert value.shape[0] == 1, f"JPEG transport only supports batch size 1, got shape {value.shape}"
        value = value[0]
    assert value.ndim == 3, f"JPEG transport expects HWC or BHWC arrays, got shape {value.shape}"
    assert value.shape[-1] == 3, f"JPEG transport expects channel-last RGB arrays, got shape {value.shape}"
    assert value.dtype == np.uint8, f"JPEG transport expects uint8 arrays, got {value.dtype}"
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class RemotePolicyPrediction:
    actions_d: npt.NDArray[np.float32]
    total_latency_ms: float
    policy_id: str
    chunk_id: str
    rl_enabled: bool | None = None
    q_value: float | None = None


class InferenceServiceRestartedError(RuntimeError):
    pass


class RemotePolicyClient:
    _LATENCY_LOG_INTERVAL_S = 60.0

    def __init__(
        self,
        predict_url: str,
        *,
        policy_auth_headers: dict[str, str] | None = None,
    ) -> None:
        self.predict_url = policy_ws_url(predict_url)
        self._policy_auth_headers = policy_auth_headers or {}
        self._lock = asyncio.Lock()
        self._ws: Any = None
        self._server_config: ServerHandshake | None = None
        self._connected_url: str | None = None
        self._latency_window_started_at_s: float | None = None
        self._latency_total_ms: list[float] = []
        self._latency_server_ms: list[float] = []

    async def __aenter__(self) -> RemotePolicyClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._close_ws()

    async def update_connection(
        self,
        *,
        predict_url: str,
        policy_auth_headers: dict[str, str] | None = None,
    ) -> None:
        self.predict_url = policy_ws_url(predict_url)
        self._policy_auth_headers = policy_auth_headers or {}
        await self._close_ws()

    def _headers(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in self._policy_auth_headers.items()]

    def _warn_on_camera_name_mismatch(self, wire_frame: dict[str, Any]) -> None:
        if self._server_config is None:
            return
        server_camera_names = self._server_config.camera_names
        sent_camera_names = _wire_camera_names(wire_frame)
        missing_camera_names = sorted(name for name in sent_camera_names if name not in set(server_camera_names))
        if missing_camera_names:
            LOGGER.warning(
                "Sending camera names not present in server config. sent=%s missing=%s server_camera_names=%s",
                sent_camera_names,
                missing_camera_names,
                sorted(server_camera_names),
            )

    def _encode_wire_frame_images(self, wire_frame: dict[str, Any]) -> dict[str, Any]:
        adapted = dict(wire_frame)
        for key, value in wire_frame.items():
            if not key.startswith("observation/") or key == JOINT_STATE_KEY or isinstance(value, bytes):
                continue
            image = _wire_image_to_hwc_uint8(value)
            field = encode_image(image, jpeg_quality=75)
            assert field.codec == "jpeg", f"{key} must use jpeg transport"
            adapted[key] = field.data
        return adapted

    async def _ensure_ws(self) -> None:
        uri = self.predict_url
        if self._ws is not None and self._connected_url == uri:
            return
        if self._ws is not None:
            await self._close_ws()
        async with self._lock:
            self._ws = await websockets.connect(uri, additional_headers=self._headers(), compression=None)
            self._connected_url = uri
            first = await self._ws.recv()
        assert isinstance(first, bytes), type(first)
        server_config_payload = deserialize_from_msgpack(first)
        assert isinstance(server_config_payload, dict), "ServerConfig must be a dict"
        self._server_config = ServerHandshake.from_payload(server_config_payload)
        _log_server_config(self._server_config)

    async def _close_ws(self) -> None:
        self._log_latency_summary(force=True)
        async with self._lock:
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
        self._server_config = None
        self._connected_url = None

    def _log_latency_summary(self, *, force: bool = False, now_s: float | None = None) -> None:
        if not self._latency_total_ms:
            return
        if self._latency_window_started_at_s is None:
            self._latency_window_started_at_s = now_s if now_s is not None else time.monotonic()
        elapsed_s = (now_s if now_s is not None else time.monotonic()) - self._latency_window_started_at_s
        if not force and elapsed_s < self._LATENCY_LOG_INTERVAL_S:
            return
        total = np.array(self._latency_total_ms, dtype=np.float64)
        server = np.array(self._latency_server_ms, dtype=np.float64)
        network = total - server
        LOGGER.info(
            "Inference latency stats: n=%d window=%.0fs mean=%.1fms min=%.1fms p50=%.1fms p95=%.1fms max=%.1fms server_mean=%.1fms network_mean=%.1fms",
            total.size,
            elapsed_s,
            float(total.mean()),
            float(total.min()),
            float(np.percentile(total, 50)),
            float(np.percentile(total, 95)),
            float(total.max()),
            float(server.mean()),
            float(network.mean()),
        )
        self._latency_window_started_at_s = now_s if now_s is not None else time.monotonic()
        self._latency_total_ms.clear()
        self._latency_server_ms.clear()

    def _record_latency(self, *, total_latency_ms: float, server_latency_ms: float, now_s: float | None = None) -> None:
        if self._latency_window_started_at_s is None:
            self._latency_window_started_at_s = now_s if now_s is not None else time.monotonic()
        self._latency_total_ms.append(total_latency_ms)
        self._latency_server_ms.append(server_latency_ms)
        self._log_latency_summary(now_s=now_s)

    @staticmethod
    def _close_code(exc: websockets.ConnectionClosedError) -> int | None:
        if exc.rcvd is not None:
            return exc.rcvd.code
        if exc.sent is not None:
            return exc.sent.code
        return None

    async def _raise_if_service_restarted(self, exc: websockets.ConnectionClosedError) -> None:
        if self._close_code(exc) != 1012:
            raise exc
        await self._close_ws()
        raise InferenceServiceRestartedError("Inference service restarted during prediction") from exc

    async def supports_server_feature(self, feature: str | ServerFeature) -> bool:
        await self._ensure_ws()
        assert self._server_config is not None
        return self._server_config.supports(feature)

    async def predict(
        self,
        wire_frame: dict[str, Any],
        *,
        chunk_id: str | None = None,
        reward: float | None = None,
        reward_action_index: int | None = None,
        done: bool = False,
        done_reason: str | None = None,
        prev_skipped_action_start: int | None = None,
    ) -> RemotePolicyPrediction:
        done_reason = _validate_done_reason(done=done, done_reason=done_reason)
        if chunk_id is None:
            chunk_id = uuid.uuid4().hex
        try:
            await self._ensure_ws()
            assert self._server_config is not None
            wire_frame = self._encode_wire_frame_images(wire_frame)
            wire_frame[CHUNK_ID_KEY] = chunk_id
            if reward is not None:
                if self._server_config.supports(ServerFeature.REWARDS):
                    wire_frame[REWARD_KEY] = float(reward)
                else:
                    LOGGER.warning(
                        "Dropping reward because server does not advertise %s support", ServerFeature.REWARDS
                    )
            if reward_action_index is not None:
                if self._server_config.supports(ServerFeature.REWARDS):
                    wire_frame[REWARD_ACTION_INDEX_KEY] = int(reward_action_index)
                else:
                    LOGGER.warning(
                        "Dropping reward_action_index because server does not advertise %s support",
                        ServerFeature.REWARDS,
                    )
            if prev_skipped_action_start is not None:
                wire_frame[PREV_SKIPPED_ACTION_START_KEY] = int(prev_skipped_action_start)
            if done:
                if self._server_config.supports(ServerFeature.REWARDS):
                    wire_frame[DONE_KEY] = True
                    wire_frame[DONE_REASON_KEY] = done_reason
                else:
                    LOGGER.warning("Dropping done because server does not advertise %s support", ServerFeature.REWARDS)
            validate_wire_inference_request_frame(wire_frame)
            self._warn_on_camera_name_mismatch(wire_frame)
            payload = serialize_to_msgpack(wire_frame)
            start_time_ns = time.time_ns()
            async with self._lock:
                assert self._ws is not None
                await self._ws.send(payload)
                while True:
                    response_raw = await self._ws.recv()
                    if isinstance(response_raw, str):
                        _emit_server_error_verbatim(response_raw)
                        raise AssertionError("unexpected text response from inference server")
                    result = deserialize_from_msgpack(response_raw)
                    _emit_server_error_verbatim(result)
                    assert isinstance(result, dict), f"unexpected response type {type(result)}"
                    response_chunk_id = result.get(CHUNK_ID_KEY, chunk_id)
                    if response_chunk_id == chunk_id:
                        break
                    LOGGER.warning(
                        "Draining stale response: expected chunk_id=%s, got %s",
                        chunk_id,
                        response_chunk_id,
                    )
            end_time_ns = time.time_ns()
        except websockets.ConnectionClosedError as exc:
            await self._raise_if_service_restarted(exc)

        total_latency_ms = (end_time_ns - start_time_ns) / 1e6
        try:
            validate_wire_inference_response(result)
        except AssertionError as exc:
            LOGGER.error("Malformed inference response: %s", _summarize_server_payload(result))
            raise AssertionError(f"{exc}") from exc
        actions = result[ACTION_KEY]
        infer_raw = result.get(INFERENCE_TIME_KEY)
        server_latency_ms = 0.0
        if infer_raw is not None:
            assert isinstance(infer_raw, (int, float)), f"{INFERENCE_TIME_KEY} must be numeric"
            server_latency_ms = float(infer_raw)
        policy_id_used = str(result.get("policy_id", ""))
        rl_enabled_raw = result.get(RL_ENABLED_KEY)
        rl_enabled = bool(rl_enabled_raw) if rl_enabled_raw is not None else None
        q_value_raw = result.get(Q_VALUE_KEY)
        q_value = float(q_value_raw) if q_value_raw is not None else None
        self._record_latency(total_latency_ms=total_latency_ms, server_latency_ms=server_latency_ms)

        actions_d = np.array(actions, dtype=np.float32)
        return RemotePolicyPrediction(
            actions_d=actions_d,
            total_latency_ms=total_latency_ms,
            policy_id=policy_id_used,
            chunk_id=chunk_id,
            rl_enabled=rl_enabled,
            q_value=q_value,
        )

    async def record_human_intervention(
        self,
        wire_frame: dict[str, Any],
        *,
        reward: float | None = None,
        reward_action_index: int | None = None,
        done: bool = False,
        done_reason: str | None = None,
        task: str | None = None,
        subtask: str | None = None,
        action_chunk_hd: npt.NDArray[np.float32],
        prev_skipped_action_start: int | None = None,
    ) -> dict[str, Any]:
        """Record an intervention chunk.

        `prev_skipped_action_start` is retrospective and marks where the
        previous chunk stopped executing before this request.
        """
        done_reason = _validate_done_reason(done=done, done_reason=done_reason)
        try:
            await self._ensure_ws()
            assert self._server_config is not None
            assert self._server_config.supports(ServerFeature.HUMAN_INTERVENTIONS), (
                f"server does not advertise {ServerFeature.HUMAN_INTERVENTIONS} support"
            )
            wire_frame = self._encode_wire_frame_images(wire_frame)
            wire_frame[ENDPOINT_KEY] = ENDPOINT_INTERVENTION
            wire_frame[ACTION_KEY] = np.asarray(action_chunk_hd, dtype=np.float32)
            if task is not None:
                wire_frame[TASK_KEY] = str(task)
            if subtask is not None:
                wire_frame[SUBTASK_KEY] = str(subtask)
            if prev_skipped_action_start is not None:
                wire_frame[PREV_SKIPPED_ACTION_START_KEY] = int(prev_skipped_action_start)
            if reward is not None:
                wire_frame[REWARD_KEY] = float(reward)
            if reward_action_index is not None:
                wire_frame[REWARD_ACTION_INDEX_KEY] = int(reward_action_index)
            if done:
                wire_frame[DONE_KEY] = True
                wire_frame[DONE_REASON_KEY] = done_reason
            validate_wire_intervention_request_frame(wire_frame)
            self._warn_on_camera_name_mismatch(wire_frame)
            payload = serialize_to_msgpack(wire_frame)
            async with self._lock:
                assert self._ws is not None
                await self._ws.send(payload)
                response_raw = await self._ws.recv()
        except websockets.ConnectionClosedError as exc:
            await self._raise_if_service_restarted(exc)

        if isinstance(response_raw, str):
            _emit_server_error_verbatim(response_raw)
            raise AssertionError("unexpected text response from inference server")
        result = deserialize_from_msgpack(response_raw)
        _emit_server_error_verbatim(result)
        assert isinstance(result, dict), f"unexpected response type {type(result)}"
        assert "error" not in result, f"unexpected error payload: {_summarize_server_payload(result)}"
        return result

    async def mark_episode_done(self, done_reason: str) -> None:
        done_reason = _validate_done_reason(done=True, done_reason=done_reason)
        # Flag the most recent inference event on the server as terminal without sending a new
        # observation. Only meaningful over the existing connection: a reconnect would yield a new
        # client_id with no recorded events, so skip when the socket is already closed.
        if self._ws is None:
            return
        try:
            async with self._lock:
                if self._ws is None:
                    return
                await self._ws.send(serialize_to_msgpack({ENDPOINT_KEY: ENDPOINT_DONE, DONE_REASON_KEY: done_reason}))
                await self._ws.recv()
        except websockets.ConnectionClosedError as exc:
            LOGGER.warning("Failed to mark episode done; connection closed: %s", exc)


__all__ = [
    "DEFAULT_PREDICT_URL",
    "InferenceServiceRestartedError",
    "RemotePolicyClient",
    "RemotePolicyPrediction",
    "policy_ws_url",
]
