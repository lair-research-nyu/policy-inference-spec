"""Observers watch (request, response) pairs without altering them.

The debug server calls the observer AFTER ``run_pipeline`` (and any gate
release) and BEFORE sending the response. Observers are side-effect only —
they cannot modify the chunk or the request.

``RerunObserver`` logs each pair to a live ``rr.spawn()`` viewer so you can
watch requests arrive and inspect predicted trajectories in real time.

To add a new observer: implement ``__call__(frame, chunk) -> None``. See
``pipeline.ResponseObserver``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL, HardwareModel
from policy_inference_spec.protocol import JOINT_STATE_KEY

LOGGER = logging.getLogger(__name__)

_RIGHT_ARM_STATE_SLICE = slice(32, 40)
_CAMERA_ENTITIES: tuple[tuple[str, str], ...] = (
    ("observation/images/main_image", "/cameras/main"),
    ("observation/images/left_wrist_image", "/cameras/left_wrist"),
    ("observation/images/right_wrist_image", "/cameras/right_wrist"),
)
_STATE_ENTITY = "/state/right_arm"
_ACTION_ENTITY = "/predicted/action"
_PALETTE = (
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


def _palette_colors(count: int) -> list[tuple[int, int, int]]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(count)]


class RerunObserver:
    """Live Rerun viewer for (request, response) pairs.

    Use when: you want to watch the server in real time while a client drives
    the robot — confirming which observations the policy is seeing and what
    trajectories it's returning.

    Spawns the native Rerun viewer on first request (``rr.spawn()``) and logs:

    - ``/cameras/{main,left_wrist,right_wrist}`` — raw JPEG bytes from the
      request frame (no decode).
    - ``/state/right_arm`` — 8 scalars sliced from ``state[32:40]`` at the
      request timestamp.
    - ``/predicted/action`` — the predicted action chunk (action_dim scalars),
      with each row logged at its projected future timestamp (``1/hz`` apart).
      So scrubbing the ``ts`` timeline forward from "now" traces the predicted
      trajectory; scrubbing the ``request`` timeline steps through successive
      predictions.

    Thread-safe for concurrent connections: state mutation is guarded by a
    lock; ``rr.log`` is already safe.
    """

    def __init__(
        self,
        *,
        app_id: str = "policy_inference_debug",
        hz: int = 50,
        hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
    ) -> None:
        assert hz > 0, f"hz must be positive, got {hz}"
        assert hardware_model == HardwareModel.GEN2, (
            f"RerunObserver only supports GEN2, got {hardware_model}"
        )
        self._app_id = app_id
        self._hz = hz
        self._hardware_model = hardware_model
        self._t0: float | None = None
        self._request_count = 0
        self._lock = threading.Lock()
        self._started = False

    def _ensure_started(self, action_dim: int) -> None:
        if self._started:
            return
        rr.init(self._app_id, spawn=True)
        rr.send_blueprint(_build_blueprint())

        state_dims = _RIGHT_ARM_STATE_SLICE.stop - _RIGHT_ARM_STATE_SLICE.start
        rr.log(
            _STATE_ENTITY,
            rr.SeriesLines(
                colors=_palette_colors(state_dims),
                widths=[1.0] * state_dims,
                names=[f"right_arm[{i}]" for i in range(state_dims)],
            ),
            static=True,
        )
        rr.log(
            _ACTION_ENTITY,
            rr.SeriesLines(
                colors=_palette_colors(action_dim),
                widths=[1.5] * action_dim,
                names=[f"action[{i}]" for i in range(action_dim)],
            ),
            static=True,
        )
        self._t0 = time.monotonic()
        self._started = True
        print(f"[rerun-observer] spawned viewer; app_id={self._app_id!r}", flush=True)

    def __call__(
        self,
        frame: dict[str, Any],
        chunk: npt.NDArray[np.float32],
    ) -> None:
        with self._lock:
            self._ensure_started(action_dim=chunk.shape[1])
            assert self._t0 is not None
            now_s = time.monotonic() - self._t0
            request_index = self._request_count
            self._request_count += 1

        rr.set_time("request", sequence=request_index)
        rr.set_time("ts", duration=now_s)

        for wire_key, entity in _CAMERA_ENTITIES:
            image_bytes = frame[wire_key]
            assert isinstance(image_bytes, bytes), (
                f"{wire_key} expected JPEG bytes, got {type(image_bytes).__name__}"
            )
            rr.log(
                entity,
                rr.EncodedImage(contents=image_bytes, media_type="image/jpeg"),
            )

        state = frame[JOINT_STATE_KEY]
        assert isinstance(state, np.ndarray), (
            f"{JOINT_STATE_KEY} expected ndarray, got {type(state).__name__}"
        )
        assert state.shape == (self._hardware_model.state_dim,), (
            f"{JOINT_STATE_KEY} shape {state.shape} != "
            f"({self._hardware_model.state_dim},)"
        )
        rr.log(_STATE_ENTITY, rr.Scalars(state[_RIGHT_ARM_STATE_SLICE]))

        horizon = chunk.shape[0]
        dt = 1.0 / self._hz
        for i in range(horizon):
            rr.set_time("ts", duration=now_s + i * dt)
            rr.log(_ACTION_ENTITY, rr.Scalars(chunk[i]))


def _build_blueprint() -> rrb.Blueprint:
    action_time_range = rrb.VisibleTimeRange(
        "ts",
        start=rr.TimeRangeBoundary.cursor_relative(seconds=-2.0),
        end=rr.TimeRangeBoundary.cursor_relative(seconds=2.0),
    )
    camera_tabs = [
        rrb.Spatial2DView(name=name, contents=[entity])
        for _, entity in _CAMERA_ENTITIES
        for name in (entity.rsplit("/", 1)[-1],)
    ]
    series_tabs = [
        rrb.TimeSeriesView(
            name="right_arm state",
            contents=[_STATE_ENTITY],
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=[action_time_range],
        ),
        rrb.TimeSeriesView(
            name="predicted action",
            contents=[_ACTION_ENTITY],
            plot_legend=rrb.PlotLegend(visible=True),
            time_ranges=[action_time_range],
        ),
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*camera_tabs, name="Cameras"),
            rrb.Tabs(*series_tabs, name="Signals"),
        ),
        collapse_panels=True,
    )
