"""``LatentPolicyBackend`` over openpi's (JAX) pi0 / pi0.5, for server.server_openpi.

The openpi twin of ``server/pi0_latent.py``, and split for the same reason: the RL agent
picks the starting noise from the policy's *conditioning embedding*, and that embedding
comes out of the same prefix pass (SigLIP + all Gemma prefix layers) the denoiser needs.
Running ``Policy.infer(obs, noise=z)`` after a separate embedding pass would pay for that
pass twice per chunk. So ``Pi0.sample_actions`` is cut at its natural seam:

    prefix(obs)      -> prefix_rep + KV cache      (the expensive pass, once)
    ...agent picks z from prefix_rep...
    denoise(cache, z) -> normalized action chunk   (reuses the cache)
    unnormalize(cache, raw) -> robot-space chunk

Both halves mirror ``Pi0.sample_actions`` in lair-openpi's ``models/pi0.py`` with
``action_prefix=None`` (no train-time RTC: a pinned prefix would override the rows the
latent produced). pi0 vs pi0.5 needs no branching here: pi0 embeds the state in the
suffix and pi0.5 tokenizes it into the prompt, and both are handled inside openpi's own
``embed_prefix`` / ``embed_suffix``. ``test_denoise_matches_policy_infer`` pins the split
against ``Policy.infer(noise=z)`` on a real checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from policy_inference_spec.protocol import FloatArray

LOGGER = logging.getLogger(__name__)


@dataclass
class OpenpiPrefixCache:
    """Opaque handle threaded from :meth:`OpenpiLatentBackend.prefix` to ``denoise``."""

    observation: Any  # preprocessed openpi Observation (device arrays); pi0 reads .state
    kv_cache: Any
    prefix_mask: Any
    # The transformed (normalized) inputs' state, for the output transforms: a checkpoint
    # trained on delta actions makes its chunk absolute against it.
    state: np.ndarray


class OpenpiLatentBackend:
    """Splits an openpi ``Policy`` into its prefix and denoise halves.

    ``policy`` is the ``openpi.policies.policy.Policy`` server_openpi builds, so the input
    and output transforms (normalization, tokenization, image resizing) are exactly the
    ones plain inference uses.
    """

    def __init__(self, policy: Any, *, robot_action_dim: int, num_steps: int) -> None:
        import flax.nnx as nnx
        import jax

        model = policy._model  # noqa: SLF001
        assert not getattr(policy, "_is_pytorch_model", False), "only JAX openpi models are supported"
        for attr in ("embed_prefix", "embed_suffix", "PaliGemma", "action_out_proj", "action_horizon", "action_dim"):
            assert hasattr(model, attr), f"{type(model).__name__} has no {attr!r}; openpi internals changed"
        self._policy = policy
        self._robot_action_dim = int(robot_action_dim)
        self._num_steps = int(num_steps)
        self._latent_shape = (int(model.action_horizon), int(model.action_dim))
        self._prefix_width: int | None = None

        # openpi's module_jit, inlined: it only accepts bound methods of the module, and
        # these two halves are not methods of it. Freezing the state the same way keeps
        # memory at what Policy.infer's jitted sampler uses.
        graphdef, state = nnx.split(model)

        def prefix_fn(state, observation):
            return _prefix(nnx.merge(graphdef, state), observation)

        def denoise_fn(state, observation, kv_cache, prefix_mask, noise, num_steps):
            return _denoise(nnx.merge(graphdef, state), observation, kv_cache, prefix_mask, noise, num_steps)

        prefix_jit = jax.jit(prefix_fn)
        denoise_jit = jax.jit(denoise_fn, static_argnums=(5,))
        self._prefix_jit = lambda observation: prefix_jit(state, observation)
        self._denoise_jit = lambda *args: denoise_jit(state, *args)

    @property
    def latent_shape(self) -> tuple[int, int]:
        # The denoiser works in the padded model action space, so the latent is the
        # model's action_dim wide (32) though the robot consumes only the first 25.
        return self._latent_shape

    @property
    def prefix_width(self) -> int:
        assert self._prefix_width is not None, "call prefix() at least once before reading prefix_width"
        return self._prefix_width

    def prefix(self, obs: dict[str, Any]) -> tuple[FloatArray, OpenpiPrefixCache]:
        """Run the conditioning pass on a raw openpi obs dict (state, image, image_mask,
        prompt). Returns ``(prefix_rep (1, W), cache)``."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = self._policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        prefix_rep, observation, kv_cache, prefix_mask = self._prefix_jit(observation)
        prefix_rep = np.asarray(prefix_rep, dtype=np.float32)
        self._prefix_width = int(prefix_rep.shape[-1])
        cache = OpenpiPrefixCache(
            observation=observation,
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            state=np.asarray(inputs["state"][0]),
        )
        return prefix_rep, cache

    def denoise(self, cache: OpenpiPrefixCache, noise: FloatArray) -> FloatArray:
        """Integrate the flow from ``noise`` (1, T, model action dim). Returns the
        normalized chunk trimmed to the robot's dims, (1, T, robot action dim)."""
        import jax.numpy as jnp

        assert noise.shape == (1, *self._latent_shape), (
            f"noise must be (1, {self._latent_shape[0]}, {self._latent_shape[1]}), got {noise.shape}"
        )
        x_0 = self._denoise_jit(
            cache.observation, cache.kv_cache, cache.prefix_mask,
            jnp.asarray(noise, dtype=jnp.float32), self._num_steps,
        )
        return np.asarray(x_0, dtype=np.float32)[:, :, : self._robot_action_dim]

    def unnormalize(self, cache: OpenpiPrefixCache, raw_chunk: FloatArray) -> FloatArray:
        """``denoise``'s (T, robot action dim) output -> robot space, through the same
        output transforms Policy.infer applies."""
        padded = np.zeros((raw_chunk.shape[0], self._latent_shape[1]), dtype=np.float32)
        padded[:, : raw_chunk.shape[1]] = raw_chunk
        out = self._policy._output_transform({"state": cache.state, "actions": padded})  # noqa: SLF001
        return np.asarray(out["actions"][:, : self._robot_action_dim], dtype=np.float32)


def _prefix(model: Any, observation: Any):
    """Pi0.sample_actions up to and including the KV-cache fill, keeping the hidden
    states it discards."""
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return _last_valid_token(prefix_out, prefix_mask), observation, kv_cache, prefix_mask


def _denoise(model: Any, observation: Any, kv_cache: Any, prefix_mask: Any, noise: Any, num_steps: int):
    """Pi0.sample_actions' flow integration with ``action_prefix=None``."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask

    dt = -1.0 / num_steps
    batch_size = noise.shape[0]

    def step(carry):
        x_t, time = carry
        timestep = jnp.broadcast_to(time, batch_size)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, timestep)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def _last_valid_token(hidden: Any, pad_mask: Any) -> Any:
    """The final *non-padding* prefix token per batch element, as float32.

    openpi right-pads the prompt to max_token_len, so ``hidden[:, -1]`` (the reference
    DSRL choice) is usually a pad embedding. Images come first and are always valid, so
    the last valid token is the prompt's last real token (for pi0.5, its state tokens).
    """
    import jax.numpy as jnp

    index = jnp.sum(pad_mask, axis=1).astype(jnp.int32) - 1
    return jnp.take_along_axis(hidden, index[:, None, None], axis=1)[:, 0].astype(jnp.float32)


__all__ = ["OpenpiLatentBackend", "OpenpiPrefixCache"]
