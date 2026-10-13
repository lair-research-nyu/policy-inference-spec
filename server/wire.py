"""Wire-frame layout shared by every server backend, free of any policy library."""

from __future__ import annotations

import socket
import subprocess
from typing import Any, Iterable

import numpy as np


def _wire_state_25(raw_state: np.ndarray) -> np.ndarray:
    """The 25 commanded positions out of the 97-dim wire state, unpinned.

    left_arm[0:8], right_arm[32:40], chest[64:70], neck[88:91] -- the same layout as the
    action the policy emits, which is what lets a chunk be compared against it.
    """
    return np.concatenate([
        raw_state[0:8],
        raw_state[32:40],
        raw_state[64:70],
        raw_state[88:91],
    ])


def _agent_images(frame: dict[str, Any], wire_keys: Iterable[str], size: int) -> dict[str, np.ndarray]:
    """Decode the wire frame's JPEGs into small uint8 HWC images for the RL agent.

    Shared so that every backend hands the learner's critic the same pixels for the same
    frame, keyed by the wire key's last component (main_image, left_wrist_image, ...).
    """
    import simplejpeg
    import torch

    images: dict[str, np.ndarray] = {}
    for wire_key in wire_keys:
        decoded = simplejpeg.decode_jpeg(frame[wire_key], colorspace="RGB")  # uint8 HWC
        tensor = torch.from_numpy(decoded).permute(2, 0, 1).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        name = wire_key.rsplit("/", 1)[-1]
        images[name] = resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    return images


def _list_lan_ips() -> list[str]:
    """Best-effort enumeration of non-loopback IPv4 addresses on this machine."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                addrs.add(addr)
    except socket.gaierror:
        pass
    # UDP route-selection trick: no packet is sent, but getsockname returns the
    # IP the OS would use to reach the target. Catches the primary NIC even when
    # gethostname() resolves oddly (e.g. 127.0.1.1 on Debian).
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        if not primary.startswith("127."):
            addrs.add(primary)
    except OSError:
        pass
    finally:
        s.close()
    # Every interface, not just the default route: the robot may be on wifi or tailscale.
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
        addrs.update(a for a in out.split() if "." in a and not a.startswith(("127.", "172.")))
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(addrs)
