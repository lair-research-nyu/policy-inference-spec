"""Slice the 97-dim wire state into the 25-dim commanded-position layout.

The wire state (``observation/state``) is 97 floats; the wire action is 25
floats. A source that wants to produce actions "relative to the current
commanded pose" (e.g. GoToPoseSource) needs to extract the 25 commanded
positions from the 97-dim state vector.

The slicing convention mirrors ``server/minimal.py::_inference_response``:

    left_arm  = state[0:8]     # 7 joints + 1 gripper
    right_arm = state[32:40]   # 7 joints + 1 gripper
    chest     = state[64:70]   # 6 joints
    neck      = state[88:91]   # 3 joints
                               # total = 25

This is GEN2-specific. If another HardwareModel is added with a different
state layout, this helper needs to grow a dispatch on ``hardware_model``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL, HardwareModel

_GEN2_STATE_TO_ACTION_SLICES: tuple[slice, ...] = (
    slice(0, 8),
    slice(32, 40),
    slice(64, 70),
    slice(88, 91),
)


def slice_state_to_commanded_positions(
    state: npt.NDArray[np.float32],
    *,
    hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> npt.NDArray[np.float32]:
    assert hardware_model == HardwareModel.GEN2, (
        f"state slicing is only defined for GEN2, got {hardware_model}"
    )
    assert state.shape == (hardware_model.state_dim,), (
        f"state must be shape ({hardware_model.state_dim},), got {state.shape}"
    )
    sliced = np.concatenate([state[s] for s in _GEN2_STATE_TO_ACTION_SLICES])
    assert sliced.shape == (hardware_model.action_dim,), (
        f"sliced commanded positions have shape {sliced.shape}, "
        f"expected ({hardware_model.action_dim},)"
    )
    return np.ascontiguousarray(sliced.astype(np.float32, copy=False))
