"""``LatentPolicyBackend`` over lerobot's flow-matching policies (pi0 / pi05).

Why this exists rather than just calling ``predict_action_chunk(batch, noise=z)``:
latent-space RL needs the policy's *conditioning embedding* to pick ``z``, and the
embedding comes from the same prefix pass the denoiser needs. Calling lerobot twice —
once for the embedding, once to denoise — would run the SigLIP vision tower and all 18
Gemma prefix layers twice per chunk, which is nearly all of the inference cost. So this
module splits ``sample_actions`` at its natural seam:

    prefix()   -> prefix_rep + KV cache      (the expensive pass, once)
    ...agent picks z from prefix_rep...
    denoise(z) -> action chunk               (reuses the cache)

Everything here mirrors ``modeling_pi0.sample_actions`` (lines 810-891) and
``modeling_pi05.sample_actions`` (lines 785-861); the two differ only in that pi0 passes
proprioceptive state as a suffix token while pi05 folds it into the prefix. If lerobot
changes those internals, ``assert_backend_compatible`` fails loudly at startup rather
than silently producing wrong actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from policy_inference_spec.protocol import FloatArray

LOGGER = logging.getLogger(__name__)

LATENT_CAPABLE_POLICY_TYPES = ("pi0", "pi05")


@dataclass
class PrefixCache:
    """Opaque handle threaded from :meth:`FlowLatentBackend.prefix` to ``denoise``."""

    prefix_pad_masks: torch.Tensor
    past_key_values: Any
    state: torch.Tensor | None  # pi0 only; pi05 encodes state in the prefix
    batch_size: int
    device: torch.device


def assert_backend_compatible(policy: Any, policy_type: str) -> None:
    """Fail at startup if the installed lerobot no longer matches our assumptions."""
    assert policy_type in LATENT_CAPABLE_POLICY_TYPES, (
        f"latent-space RL needs a flow-matching policy whose sampler starts from noise; "
        f"policy_type={policy_type!r} has no such latent. Supported: {LATENT_CAPABLE_POLICY_TYPES}"
    )
    model = policy.model
    for attr in ("embed_prefix", "denoise_step", "paligemma_with_expert", "_prepare_attention_masks_4d"):
        assert hasattr(model, attr), f"{type(model).__name__} has no {attr!r}; lerobot internals changed"
    assert hasattr(policy.config, "chunk_size") and hasattr(policy.config, "max_action_dim"), (
        "policy config must expose chunk_size and max_action_dim"
    )


class FlowLatentBackend:
    """Splits a lerobot pi0/pi05 policy into its prefix and denoise halves."""

    def __init__(self, policy: Any, policy_type: str) -> None:
        assert_backend_compatible(policy, policy_type)
        self._policy = policy
        self._policy_type = policy_type
        self._model = policy.model
        self._prefix_width: int | None = None
        assert not getattr(policy, "_rtc_enabled", lambda: False)(), (
            "RTC and latent-space RL are incompatible: RTC inpaints the previous chunk into the "
            "denoise loop, so the actions no longer are a function of the latent the agent chose "
            "and the critic would be learning against a corrupted action mapping."
        )

    @property
    def latent_shape(self) -> tuple[int, int]:
        # The denoiser works in padded action space, so the latent is max_action_dim wide
        # (32) even though the robot only consumes the first action_dim (25) columns.
        return (int(self._policy.config.chunk_size), int(self._policy.config.max_action_dim))

    @property
    def prefix_width(self) -> int:
        assert self._prefix_width is not None, "call prefix() at least once before reading prefix_width"
        return self._prefix_width

    @property
    def num_inference_steps(self) -> int:
        return int(self._model.config.num_inference_steps)

    @torch.no_grad()
    def prefix(self, batch: dict[str, Any]) -> tuple[FloatArray, PrefixCache]:
        """Run the conditioning pass. Returns ``(prefix_rep (B, W), cache)``."""
        self._policy.eval()
        model = self._model
        images, img_masks = self._policy._preprocess_images(batch)  # noqa: SLF001
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

        # pi0 feeds proprio in as a suffix token at denoise time; pi05 has already
        # tokenized it into the prefix, so there is nothing to carry forward.
        state = self._policy.prepare_state(batch) if self._policy_type == "pi0" else None

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        # sample_actions throws this away; it is the whole reason this module exists.
        prefix_hidden = outputs_embeds[0]
        assert prefix_hidden is not None, (
            "prefix-only forward returned no hidden states; lerobot's PaliGemmaWithExpertModel.forward "
            "no longer returns [prefix_output, suffix_output]"
        )
        assert past_key_values is not None, "prefix forward returned no KV cache despite use_cache=True"

        prefix_rep = _last_valid_token(prefix_hidden, prefix_pad_masks)
        self._prefix_width = int(prefix_rep.shape[-1])

        cache = PrefixCache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            state=state,
            batch_size=int(prefix_pad_masks.shape[0]),
            device=prefix_pad_masks.device,
        )
        return prefix_rep.to(dtype=torch.float32).cpu().numpy(), cache

    @torch.no_grad()
    def denoise(self, cache: PrefixCache, noise: FloatArray, num_steps: int | None = None) -> FloatArray:
        """Integrate the flow from ``noise``. Returns the unpadded chunk (B, T, action_dim)."""
        latent_shape = self.latent_shape
        assert noise.shape[1:] == latent_shape, (
            f"noise must be (B, {latent_shape[0]}, {latent_shape[1]}), got {noise.shape}"
        )
        assert noise.shape[0] == cache.batch_size, (
            f"noise batch {noise.shape[0]} does not match prefix batch {cache.batch_size}"
        )
        if num_steps is None:
            num_steps = self.num_inference_steps

        # copy=True: latents arriving off the wire are backed by read-only buffers.
        x_t = torch.from_numpy(np.array(noise, dtype=np.float32, copy=True)).to(cache.device)
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = 1.0 + step * dt
            timestep = torch.tensor(time, dtype=torch.float32, device=cache.device).expand(cache.batch_size)
            v_t = self._denoise_step(cache, x_t, timestep)
            x_t = x_t + dt * v_t

        # predict_action_chunk trims the padded action space down to what the robot uses.
        original_action_dim = self._policy.config.output_features[ACTION].shape[0]
        return x_t[:, :, :original_action_dim].to(dtype=torch.float32).cpu().numpy()

    def _denoise_step(self, cache: PrefixCache, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # denoise_step deep-copies the KV cache internally, so it is safe to reuse across steps.
        if self._policy_type == "pi0":
            return self._model.denoise_step(
                state=cache.state,
                prefix_pad_masks=cache.prefix_pad_masks,
                past_key_values=cache.past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
        return self._model.denoise_step(
            prefix_pad_masks=cache.prefix_pad_masks,
            past_key_values=cache.past_key_values,
            x_t=x_t,
            timestep=timestep,
        )


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Local copy of lerobot's ``make_att_2d_masks`` (modeling_pi0.py:111).

    Imported by value rather than by name because it is a module-level helper in a file
    whose import surface has churned; a wrong mask here would be silent.
    """
    assert pad_masks.ndim == 2, f"pad_masks must be 2-D, got {pad_masks.shape}"
    assert att_masks.ndim == 2, f"att_masks must be 2-D, got {att_masks.shape}"
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _last_valid_token(hidden: torch.Tensor, pad_masks: torch.Tensor) -> torch.Tensor:
    """Gather the final *non-padding* prefix token per batch element.

    The reference DSRL implementation takes ``hidden[:, -1, :]``, which is only correct
    when nothing is padded. lerobot pads language tokens to a fixed length, so the last
    row is often a pad embedding; this reduces to ``[:, -1, :]`` when there is no padding.
    """
    lengths = pad_masks.sum(dim=1).to(torch.long)
    assert torch.all(lengths >= 1), "every prefix must have at least one valid token"
    index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    return hidden.gather(dim=1, index=index).squeeze(1)


__all__ = [
    "FlowLatentBackend",
    "LATENT_CAPABLE_POLICY_TYPES",
    "PrefixCache",
    "assert_backend_compatible",
]
