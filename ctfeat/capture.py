"""One teacher-forced forward pass per trace, summarized inside per-layer hooks.

Row conventions (P = len(prompt_ids), T = len(prompt_ids + response_ids)):
  - "query rows" j = 0 .. T-P are absolute indices q = P-1+j. Response token r has
    its pre-emission state at row r (index a-1) and post-emission state at row r+1 (index a).
  - residual arrays start one index earlier (q = P-2) so tmp_cos can look back.
Full attention maps never leave the hook; only per-row summaries are kept. The model
must return attention weights (attention.WEIGHTS_IMPL in extraction: sdpa plus float32
weights; eager in float64 tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .attention import WEIGHTS_IMPL
from .features import attention_received, attention_row_stats
from .head import PredictionHead

RECEIVED_DELAYS = (4, 16, 64)


@dataclass
class Capture:
    prompt_len: int
    logits: torch.Tensor          # (R, vocab) float32, query rows
    resid: torch.Tensor           # (n_layers + 1, R + 1, d) bf16; index 0 = embedding output, rows from q = P-2
    mid: torch.Tensor             # (n_layers, R + 1, d) bf16; residual after the attention add
    attn_out_norm: torch.Tensor   # (n_layers, R + 1)
    mlp_out_norm: torch.Tensor    # (n_layers, R + 1)
    attn_stats: torch.Tensor | None   # (R, n_layers, heads, n_stats)
    received: torch.Tensor | None     # (n_response, len(RECEIVED_DELAYS), n_layers, heads)
    prompt_mean_hat: torch.Tensor     # mean of ĥ over the prompt tokens (d,)


def _tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def run_capture(model: torch.nn.Module, head: PredictionHead, prompt_ids: list[int], response_ids: list[int],
                spans: dict[str, list[int]], *, attention: bool = True) -> Capture:
    P, T = len(prompt_ids), len(prompt_ids) + len(response_ids)
    if P < 2:
        raise ValueError("the prompt must have at least two tokens")
    lo = P - 2
    layers = model.model.layers
    n_layers = len(layers)
    resid: list[torch.Tensor | None] = [None] * (n_layers + 1)
    mid: list[torch.Tensor | None] = [None] * n_layers
    attn_norm: list[torch.Tensor | None] = [None] * n_layers
    mlp_norm: list[torch.Tensor | None] = [None] * n_layers
    stats: list[torch.Tensor | None] = [None] * n_layers
    received: list[torch.Tensor | None] = [None] * n_layers
    prompt_mean: list[torch.Tensor] = []
    handles = []

    def keep_embed(_module, _inputs, output):
        resid[0] = _tensor(output)[0, lo:].detach()

    def keep_layer(index: int):
        def hook(_module, _inputs, output):
            hidden = _tensor(output)[0]
            resid[index + 1] = hidden[lo:].detach()
            if index == n_layers - 1:
                prompt_mean.append(head.normed(hidden[:P]).mean(0))
        return hook

    def keep_mid(index: int):
        def hook(_module, inputs):
            mid[index] = inputs[0][0, lo:].detach()
        return hook

    def keep_attention(index: int):
        def hook(_module, _inputs, output):
            attn_norm[index] = output[0][0, lo:].float().norm(dim=-1)
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attention and weights is None:
                raise RuntimeError(f"layer {index + 1} returned no attention weights "
                                   f"(attn_implementation={model.config._attn_implementation!r}); use {WEIGHTS_IMPL!r}")
            if attention:
                stats[index] = attention_row_stats(weights[0], P, spans)
                received[index] = attention_received(weights[0], P, RECEIVED_DELAYS)
        return hook

    def keep_mlp(index: int):
        def hook(_module, _inputs, output):
            mlp_norm[index] = _tensor(output)[0, lo:].float().norm(dim=-1)
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(keep_embed))
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(keep_layer(index)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(keep_mid(index)))
        handles.append(layer.self_attn.register_forward_hook(keep_attention(index)))
        handles.append(layer.mlp.register_forward_hook(keep_mlp(index)))
    try:
        device = next(model.parameters()).device
        input_ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
        logits = model(input_ids=input_ids, logits_to_keep=T - P + 1).logits[0].float()
    finally:
        for handle in handles:
            handle.remove()
    has_attention = attention
    return Capture(
        prompt_len=P, logits=logits,
        resid=torch.stack(resid), mid=torch.stack(mid),
        attn_out_norm=torch.stack(attn_norm), mlp_out_norm=torch.stack(mlp_norm),
        attn_stats=torch.stack(stats, dim=1) if has_attention else None,
        received=torch.stack(received, dim=2) if has_attention else None,
        prompt_mean_hat=prompt_mean[0],
    )
