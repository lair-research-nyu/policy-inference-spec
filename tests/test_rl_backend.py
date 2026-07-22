"""Equivalence tests for the split prefix/denoise backend.

The backend reimplements the inside of lerobot's ``sample_actions`` so it can hand the
prefix embedding to an RL agent mid-flight. That is only safe if it produces bit-for-bit
the same actions lerobot would have produced from the same latent — otherwise the policy
being trained is not the policy being served. These tests pin that down.

They need a real pi0 checkpoint and a GPU, so they skip when either is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "ultra-towerstack-pi0-15000"
POLICY_TYPE = "pi0"

pytestmark = [
    pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"checkpoint not available: {CHECKPOINT}"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def loaded_policy():
    from server import server as srv

    srv._init_policy(CHECKPOINT, POLICY_TYPE)
    return srv


@pytest.fixture(scope="module")
def backend(loaded_policy):
    from server.pi0_latent import FlowLatentBackend

    return FlowLatentBackend(loaded_policy.POLICY, POLICY_TYPE)


def _make_batch(loaded_policy, seed: int = 0):
    """A processed observation batch shaped exactly like the one the server builds."""
    from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL

    rng = np.random.default_rng(seed)
    device = str(loaded_policy.POLICY.config.device)
    height, width = DEFAULT_HARDWARE_MODEL.image_resolution

    obs = {
        "observation.state": torch.from_numpy(rng.normal(size=25).astype("float32")).unsqueeze(0).to(device),
        "task": ["stack the tower"],
    }
    for lerobot_key in loaded_policy._CAMERA_KEY_MAP.values():
        image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        tensor = torch.from_numpy(image.astype("float32") / 255.0).permute(2, 0, 1).to(device)
        obs[lerobot_key] = tensor.unsqueeze(0)
    return loaded_policy._PREPROCESSOR(obs)


def test_latent_shape_matches_config(backend, loaded_policy):
    config = loaded_policy.POLICY.config
    assert backend.latent_shape == (config.chunk_size, config.max_action_dim)


def test_prefix_rep_shape(backend, loaded_policy):
    prefix_rep, cache = backend.prefix(_make_batch(loaded_policy))
    assert prefix_rep.ndim == 2 and prefix_rep.shape[0] == 1, prefix_rep.shape
    assert prefix_rep.shape[1] == backend.prefix_width
    assert prefix_rep.dtype == np.float32
    assert np.isfinite(prefix_rep).all()
    assert cache.batch_size == 1


def test_same_latent_gives_same_actions(backend, loaded_policy):
    """The mapping z -> actions must be a deterministic function, or the critic is
    learning against noise it cannot see."""
    z = np.random.default_rng(0).standard_normal((1, *backend.latent_shape)).astype(np.float32)

    _, cache_a = backend.prefix(_make_batch(loaded_policy))
    actions_a = backend.denoise(cache_a, z)
    _, cache_b = backend.prefix(_make_batch(loaded_policy))
    actions_b = backend.denoise(cache_b, z)

    np.testing.assert_allclose(actions_a, actions_b, rtol=0, atol=1e-6)


def test_different_latent_gives_different_actions(backend, loaded_policy):
    rng = np.random.default_rng(1)
    z1 = rng.standard_normal((1, *backend.latent_shape)).astype(np.float32)
    z2 = rng.standard_normal((1, *backend.latent_shape)).astype(np.float32)

    _, cache = backend.prefix(_make_batch(loaded_policy))
    actions_1 = backend.denoise(cache, z1)
    actions_2 = backend.denoise(cache, z2)

    assert not np.allclose(actions_1, actions_2, atol=1e-3), (
        "the latent has no effect on the actions — RL would have nothing to optimize"
    )


def test_matches_lerobot_predict_action_chunk(backend, loaded_policy):
    """The whole point: our split path must agree with the path lerobot would take."""
    z = np.random.default_rng(2).standard_normal((1, *backend.latent_shape)).astype(np.float32)
    device = str(loaded_policy.POLICY.config.device)

    _, cache = backend.prefix(_make_batch(loaded_policy))
    ours = backend.denoise(cache, z)

    theirs = (
        loaded_policy.POLICY.predict_action_chunk(
            _make_batch(loaded_policy), noise=torch.from_numpy(z).to(device)
        )
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )

    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-4)


def test_rejects_wrong_latent_shape(backend, loaded_policy):
    _, cache = backend.prefix(_make_batch(loaded_policy))
    with pytest.raises(AssertionError, match="noise must be"):
        backend.denoise(cache, np.zeros((1, 4, 32), dtype=np.float32))


def test_rejects_non_flow_policy(loaded_policy):
    from server.pi0_latent import assert_backend_compatible

    with pytest.raises(AssertionError, match="flow-matching"):
        assert_backend_compatible(loaded_policy.POLICY, "act")
