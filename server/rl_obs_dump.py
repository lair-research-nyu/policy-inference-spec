"""Side-channel dump of the camera images the RL path fed to the policy.

The learner (``dsrl_pi0/examples/train_agent_server.py``) records Q values, the actor's
latent and the resulting chunks, but it only ever *sees* the 128x128 images the critic
consumes. Those are far too coarse to re-run pi0 on: measured against the same
observation at native resolution, a 128px round-trip perturbs the action chunk 3.4-5.6x
more than resampling the latent does, which is the entire effect an offline sweep is
trying to measure.

So the server writes the images itself. Two properties make this trustworthy:

* **The bytes are the wire's own JPEGs, copied verbatim.** Nothing is decoded, resized or
  re-encoded, so what lands on disk is bit-for-bit what ``_decode_jpeg_to_tensor`` handed
  the preprocessor. There is no round-trip to argue about.
* **Every frame's dimensions are checked against the checkpoint's ``input_features``**
  (once per camera, then cheaply via the JPEG header). A camera streaming at the wrong
  size is exactly the failure this dump exists to rule out, so it is logged loudly
  rather than silently written.

Writing happens on a background thread. The RL path is in the robot's control loop and
already takes care not to let a learner's disk write show up as inference latency; this
must not undo that. If the queue backs up, frames are dropped and counted -- a gap in an
offline dump is a nuisance, a stalled robot is not.

Join key is ``step_id``, which the server allocates from one global counter and sends to
the learner, so an episode's ``step_ids`` index straight into ``steps.jsonl``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Deep enough to ride out a disk hiccup at ~1-2 queries/s, shallow enough that a truly
# stuck writer is noticed in seconds rather than eating memory.
_QUEUE_DEPTH = 256


class RlObsDump:
    """Writes wire JPEGs plus a ``steps.jsonl`` index under one directory."""

    def __init__(self, out_dir: str | Path, *, camera_keys: dict[str, str],
                 trained_shapes: dict[str, tuple[int, int]], checkpoint: str,
                 policy_type: str) -> None:
        self.root = Path(out_dir).expanduser()
        self.images_dir = self.root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        # Wire key -> short camera name, the name used in filenames and by the viewer.
        self._camera_keys = {k: v.rsplit(".", 1)[-1] for k, v in camera_keys.items()}
        self._trained = trained_shapes
        self._checked: set[str] = set()
        self._dropped = 0
        self._written = 0

        # A restart resets the server's step counter, so a run id keeps a viewer from
        # joining this run's step ids onto a previous run's images.
        self.run_id = f"{int(time.time())}"
        # Which is only a *detector*: this run's step ids start at 0 again, the index is
        # appended to and the frames are named by step id, so a second run in the same
        # directory interleaves its index and overwrites the first run's images. The
        # default directory is per-run for exactly this reason; an explicit one is the
        # caller's to get right, so say so loudly rather than silently corrupt it.
        if (self.root / "steps.jsonl").exists():
            LOGGER.warning(
                "RL observation dump directory %s already holds a dump: this run's images "
                "will overwrite it step id by step id, and the two indexes will interleave. "
                "Point --rl-obs-dump-dir at a new directory, or drop the flag to get a "
                "fresh one per run.", self.root,
            )
        (self.root / "manifest.json").write_text(json.dumps({
            "run_id": self.run_id,
            "checkpoint": checkpoint,
            "policy_type": policy_type,
            "cameras": sorted(self._camera_keys.values()),
            "trained_image_shapes": {k: list(v) for k, v in trained_shapes.items()},
            "started_unix": time.time(),
        }, indent=2))

        self._index = (self.root / "steps.jsonl").open("a", buffering=1)
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._drain, name="rl-obs-dump", daemon=True)
        self._thread.start()
        LOGGER.info("RL observation dump -> %s (run_id=%s)", self.root, self.run_id)

    # -- called from the inference path -----------------------------------------

    def submit(self, step_id: int, frame: dict[str, Any], *, task: str,
               terminal: bool = False) -> None:
        """Queue one observation's JPEGs. Never blocks and never raises."""
        try:
            payload = {wire: frame[wire] for wire in self._camera_keys if wire in frame}
            if not payload:
                return
            self._queue.put_nowait((int(step_id), payload, str(task), bool(terminal), time.time()))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 25 == 1:
                LOGGER.warning("RL observation dump is falling behind; dropped %d frames "
                               "(the robot is never made to wait for this)", self._dropped)
        except Exception:
            LOGGER.warning("RL observation dump: submit failed; continuing", exc_info=True)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        try:
            self._index.close()
        except Exception:
            pass
        LOGGER.info("RL observation dump: wrote %d observations, dropped %d",
                    self._written, self._dropped)

    # -- background writer -------------------------------------------------------

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            step_id, payload, task, terminal, unix = item
            try:
                self._write(step_id, payload, task, terminal, unix)
            except Exception:
                LOGGER.warning("RL observation dump: write failed for step %d", step_id,
                               exc_info=True)

    def _write(self, step_id: int, payload: dict, task: str, terminal: bool,
               unix: float) -> None:
        files = {}
        for wire, data in payload.items():
            cam = self._camera_keys[wire]
            self._check_shape(cam, data)
            name = f"step_{step_id:08d}_{cam}.jpg"
            (self.images_dir / name).write_bytes(data)
            files[cam] = name
        self._index.write(json.dumps({
            "step_id": step_id, "unix": unix, "task": task,
            "terminal": terminal, "files": files,
        }) + "\n")
        self._written += 1

    def _check_shape(self, cam: str, data: bytes) -> None:
        """Compare the JPEG's own dimensions against what the checkpoint was trained on.

        Cheap: reads the JPEG header, not the pixels. Warned once per camera, because a
        resolution mismatch is a property of the stream, not of one frame -- and because
        pi0's ``resize_with_pad`` would otherwise absorb it silently, putting the scene at
        a different scale behind different-sized bars than in training.
        """
        if cam in self._checked:
            return
        self._checked.add(cam)
        want = self._trained.get(cam)
        try:
            import simplejpeg
            height, width = simplejpeg.decode_jpeg_header(data)[:2]
        except Exception:
            LOGGER.warning("RL observation dump: could not read JPEG header for %s", cam)
            return
        if want is None:
            LOGGER.info("RL observation dump: %s is %dx%d (checkpoint declares no shape)",
                        cam, height, width)
        elif (height, width) != tuple(want):
            LOGGER.warning(
                "RL observation dump: %s streams %dx%d but the checkpoint was trained on "
                "%dx%d. The dump is still byte-exact for what the policy received, but an "
                "offline sweep on it inherits this mismatch.",
                cam, height, width, want[0], want[1])
        else:
            LOGGER.info("RL observation dump: %s is %dx%d, matching the checkpoint",
                        cam, height, width)
