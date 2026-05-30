from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL, HardwareModel


@runtime_checkable
class ActionSource(Protocol):
    def __call__(self, frame: dict[str, Any]) -> npt.NDArray[np.float32]: ...


@runtime_checkable
class ChunkTransform(Protocol):
    def __call__(
        self, chunk: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...


@runtime_checkable
class Gate(Protocol):
    async def wait_for_release(self) -> None: ...


@runtime_checkable
class ResponseObserver(Protocol):
    def __call__(
        self, frame: dict[str, Any], chunk: npt.NDArray[np.float32]
    ) -> None: ...


def run_pipeline(
    frame: dict[str, Any],
    source: ActionSource,
    *,
    transforms: Sequence[ChunkTransform] = (),
    hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> npt.NDArray[np.float32]:
    chunk = source(frame)
    _assert_chunk_contract(chunk, hardware_model, origin="source")
    for i, transform in enumerate(transforms):
        chunk = transform(chunk)
        _assert_chunk_contract(chunk, hardware_model, origin=f"transform[{i}]")
    return chunk


def _assert_chunk_contract(
    chunk: npt.NDArray[np.float32],
    hardware_model: HardwareModel,
    *,
    origin: str,
) -> None:
    assert isinstance(chunk, np.ndarray), (
        f"{origin} must return ndarray, got {type(chunk)}"
    )
    assert chunk.ndim == 2, (
        f"{origin} must return 2-D chunk, got shape {chunk.shape}"
    )
    assert chunk.shape[1] == hardware_model.action_dim, (
        f"{origin} chunk second dim must be {hardware_model.action_dim}, "
        f"got {chunk.shape}"
    )
    assert chunk.dtype == np.float32, (
        f"{origin} chunk must be float32, got {chunk.dtype}"
    )
