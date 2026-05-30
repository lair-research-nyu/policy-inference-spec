"""Action-chunk transforms for the debug pipeline.

A ``ChunkTransform`` is a callable that takes an ``(H, action_dim)`` float32
chunk and returns a modified chunk of the same shape and dtype. The debug
server applies transforms in order between the ``ActionSource`` and the
wire response, letting you compose manipulations like "drop the first 25
actions" and "pin the neck joints to zero" on top of any source.

To add a new transform: implement ``__call__(chunk) -> chunk`` preserving
shape ``(H, action_dim)`` and dtype ``float32``. See ``pipeline.ChunkTransform``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


class DeleteAndBackpad:
    """Drop the first ``n`` rows; back-pad to the original length with ``chunk[n]``.

    Use when: you want the robot to skip the first part of a predicted
    trajectory (e.g. a policy's noisy warm-up actions) while keeping the
    chunk size ``H`` so server pacing is unchanged. The pad value is
    row ``n`` of the original chunk — "the (n+1)th action".

    Example with H=50, n=25::

        output = [chunk[25], chunk[26], ..., chunk[49],  # kept tail
                  chunk[25], chunk[25], ..., chunk[25]]  # 25 pad rows
    """

    def __init__(self, n: int) -> None:
        assert n >= 0, f"n must be non-negative, got {n}"
        self._n = n

    def __call__(
        self, chunk: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        assert chunk.ndim == 2, f"chunk must be 2D, got shape {chunk.shape}"
        horizon = chunk.shape[0]
        n = self._n
        assert n < horizon, f"n={n} must be < horizon={horizon}"
        if n == 0:
            return chunk
        kept = chunk[n:]
        pad = np.broadcast_to(chunk[n][None, :], (n, chunk.shape[1])).copy()
        out = np.concatenate([kept, pad], axis=0)
        assert out.shape == chunk.shape, (
            f"DeleteAndBackpad produced shape {out.shape}, expected {chunk.shape}"
        )
        return np.ascontiguousarray(out.astype(np.float32, copy=False))


class SetConstant:
    """Overwrite the specified action dims of every row with ``value``.

    Use when: you want to pin certain joints (or all joints) to a fixed
    value regardless of what the source produced — e.g. zeroing the grippers,
    freezing the neck, or driving one dim to a test value.

    ``dims=None`` applies to every dim. Otherwise ``dims`` is an iterable of
    column indices (0-indexed into the 25-dim action) to overwrite.
    """

    def __init__(
        self,
        value: float,
        *,
        dims: Sequence[int] | None = None,
    ) -> None:
        self._value = float(value)
        self._dims: list[int] | None = (
            None if dims is None else [int(d) for d in dims]
        )

    def __call__(
        self, chunk: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        assert chunk.ndim == 2, f"chunk must be 2D, got shape {chunk.shape}"
        out = chunk.copy()
        if self._dims is None:
            out[...] = self._value
            return out
        action_dim = chunk.shape[1]
        for d in self._dims:
            assert 0 <= d < action_dim, (
                f"dim index {d} out of range for action_dim={action_dim}"
            )
        out[:, self._dims] = self._value
        return out
