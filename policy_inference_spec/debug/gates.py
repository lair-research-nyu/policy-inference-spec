"""Gates pause the debug pipeline until an external signal arrives.

A ``Gate``'s ``wait_for_release`` coroutine is awaited between the pipeline's
chunk-producing step and the websocket send, so the server can hold each
response until some condition is met. The default (``ImmediateGate``) returns
immediately — the server behaves like a normal predict endpoint.

``KeypressGate`` waits until Enter is pressed in the server terminal, letting
you step through action chunks one at a time.

To add a gate: implement ``async def wait_for_release() -> None`` on a class
that satisfies ``pipeline.Gate``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading

LOGGER = logging.getLogger(__name__)


class ImmediateGate:
    """No-op gate that releases immediately.

    Use when: you don't want any gating (the default). Identical to not
    configuring a gate at all; exists so the pipeline always has something
    to await and call-sites can be uniform.
    """

    async def wait_for_release(self) -> None:
        return


class KeypressGate:
    """Pauses the pipeline until Enter is pressed in the server terminal.

    Use when: you want to step through action chunks one at a time — e.g.
    replaying an rrd recording chunk-by-chunk, or inspecting a policy's
    actions before committing them to the robot.

    Model: each press of Enter releases exactly one waiting response. If
    multiple connections are waiting at once (e.g. client retry), each
    keypress advances one of them in FIFO order. Press Ctrl-C on the
    server to stop.

    Caveat: holding the response blocks the websocket. If the client has a
    predict timeout, long pauses will drop the connection. Keep presses
    prompt, or raise the client-side timeout.

    Requires a TTY stdin — the constructor asserts if stdin is not a
    terminal, since a backgrounded/piped server would hang forever on the
    very first request.
    """

    def __init__(self) -> None:
        assert sys.stdin.isatty(), (
            "KeypressGate requires an interactive TTY stdin; stdin is not a terminal"
        )
        self._queue: asyncio.Queue[str] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._thread = threading.Thread(
            target=self._stdin_loop, daemon=True, name="keypress-gate"
        )
        self._thread.start()
        print(
            "[keypress-gate] press Enter in this terminal to release each chunk",
            flush=True,
        )

    def _stdin_loop(self) -> None:
        assert self._loop is not None and self._queue is not None
        loop = self._loop
        queue = self._queue
        while True:
            line = sys.stdin.readline()
            if not line:
                LOGGER.warning("keypress-gate: stdin closed; subsequent requests will hang")
                return
            loop.call_soon_threadsafe(queue.put_nowait, line)

    async def wait_for_release(self) -> None:
        self._ensure_started()
        assert self._queue is not None
        print("[keypress-gate] waiting for Enter to release chunk...", flush=True)
        await self._queue.get()
