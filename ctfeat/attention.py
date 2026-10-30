"""sdpa attention that also returns float32 attention weights for the feature hooks.

Qwen2.5-7B has attention scores up to ~1e4. HF eager attention rounds q·kᵀ to bf16
before the softmax, which moves attention weights by up to ~1 and the final logits by
up to ~17, so eager in bf16 is not the model that generated the traces. Here the
model's own computation stays sdpa (identical logits and hidden states to the pinned
implementation), and the weights handed to the hooks are softmax(q·kᵀ·scale + mask)
computed in float32 from the same q and k. They are never fed back into the model.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

WEIGHTS_IMPL = "sdpa_fp32_weights"

# When a list, each attention call appends max|weights·V − sdpa output| / max|sdpa output|.
_reconstruction: list[float] | None = None


def attention_weights(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                      attention_mask: torch.Tensor | None, scaling: float | None) -> torch.Tensor:
    """float32 softmax(q·kᵀ·scale + mask), (batch, heads, q_len, kv_len), with the mask sdpa uses.

    A missing mask means sdpa's causal path: query i (of q_len, aligned to the end of the
    kv sequence) sees keys ≤ kv_len − q_len + i.
    """
    key = repeat_kv(key, getattr(module, "num_key_value_groups", 1)).float()
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.transpose(2, 3)) * scale
    q_len, kv_len = scores.shape[-2:]
    if attention_mask is None:
        allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).tril(kv_len - q_len)
        scores = scores.masked_fill(~allowed, float("-inf"))
    elif attention_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attention_mask[..., :kv_len], float("-inf"))
    else:
        scores = scores + attention_mask[..., :kv_len].float()
    # Fully masked rows (padding only) give NaN; they carry no attention.
    return torch.nan_to_num(scores.softmax(-1), nan=0.0)


def sdpa_with_weights(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      attention_mask: torch.Tensor | None, scaling: float | None = None, dropout: float = 0.0,
                      **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if kwargs.get("sliding_window") is not None:
        raise NotImplementedError("float32 attention weights are implemented for full attention only")
    output, _ = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    weights = attention_weights(module, query, key, attention_mask, scaling)
    if _reconstruction is not None:
        value = repeat_kv(value, getattr(module, "num_key_value_groups", 1)).float()
        rebuilt = torch.matmul(weights, value).transpose(1, 2)
        reference = output.float()
        _reconstruction.append(float((rebuilt - reference).abs().max() / reference.abs().max().clamp_min(1e-12)))
    return output, weights


AttentionInterface.register(WEIGHTS_IMPL, sdpa_with_weights)
# Masks are built exactly as for sdpa (without this, a custom name gets no mask at all).
AttentionMaskInterface.register(WEIGHTS_IMPL, sdpa_mask)


@contextmanager
def using(model: torch.nn.Module, implementation: str) -> Iterator[None]:
    """Run with ``implementation``, then restore the model's previous one."""
    previous = model.config._attn_implementation
    model.set_attn_implementation(implementation)
    try:
        yield
    finally:
        model.set_attn_implementation(previous)


@contextmanager
def reconstruction_errors() -> Iterator[list[float]]:
    """Collect per-layer weights·V reconstruction errors from every sdpa_with_weights call."""
    global _reconstruction
    _reconstruction = []
    try:
        yield _reconstruction
    finally:
        _reconstruction = None
