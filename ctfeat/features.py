"""Feature families. Each function reads a Capture (or the hook's attention map) and
registers online features with their tier and delay; see capture.py for row indexing."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from . import causal
from .registry import Registry

if TYPE_CHECKING:
    from .capture import Capture
    from .head import PredictionHead

ATTN_STATS = ("att_ent", "att_max", "att_argmax_dist", "att_fewshot", "att_question", "att_cue",
              "att_resp", "att_recent16", "att_sink", "lookback_ratio")
TOKEN_TYPES = ("digit", "operator", "latex_other", "connective", "punct", "other")
K_DELAYS = (4, 16, 64)


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


# ---------------------------------------------------------------- attention (inside hooks)

def attention_row_stats(weights: torch.Tensor, prompt_len: int, spans: dict[str, list[int]]) -> torch.Tensor:
    """Summaries of attention rows with query q = P-1 .. T-1 over keys ≤ q.

    ``weights`` is one layer's (heads, T, T) map; returns (R, heads, len(ATTN_STATS)).
    """
    P = prompt_len
    A = weights[:, P - 1:, :].float()                                   # heads x R x T
    heads, rows, T = A.shape
    q = torch.arange(P - 1, T, device=A.device)
    entropy = -(A * torch.log(A.clamp_min(1e-30))).sum(-1)
    top, argmax = A.max(-1)
    C = A.cumsum(-1)
    zero = torch.zeros(heads, rows, 1, device=A.device)
    C0 = torch.cat([zero, C], -1)                                       # C0[..., i] = mass on keys < i

    def mass(start: torch.Tensor | int, end: torch.Tensor | int) -> torch.Tensor:
        """Mass on keys [start, end), per row."""
        start = torch.as_tensor(start, device=A.device).expand(rows)
        end = torch.as_tensor(end, device=A.device).expand(rows)
        return C0.gather(-1, end.view(1, -1, 1).expand(heads, -1, 1)).squeeze(-1) - \
            C0.gather(-1, start.view(1, -1, 1).expand(heads, -1, 1)).squeeze(-1)

    fewshot = mass(*spans["fewshot_span"])
    question = mass(*spans["question_span"])
    cue = mass(*spans["cue_span"])
    resp = mass(P, q + 1)
    recent = mass((q - 15).clamp_min(0), q + 1)
    sink = A[..., 0]
    lookback = question / (question + resp + 1e-8)
    dist = q.view(1, -1).float() - argmax.float()
    out = torch.stack([entropy, top, dist, fewshot, question, cue, resp, recent, sink, lookback], -1)
    return out.transpose(0, 1)                                          # R x heads x stats


def attention_received(weights: torch.Tensor, prompt_len: int, delays: tuple[int, ...]) -> torch.Tensor:
    """Mean attention received by each response key a from queries a+1 .. a+k (existing ones).

    Returns (n_response, len(delays), heads); NaN when no later query exists.
    """
    P = prompt_len
    A = weights[:, :, P:].float()                                       # heads x T x n_resp
    heads, T, n = A.shape
    C = A.cumsum(1)
    a = torch.arange(P, T, device=A.device)
    col = torch.arange(n, device=A.device)
    out = []
    for k in delays:
        end = (a + k).clamp_max(T - 1)
        count = (end - a).float()
        total = C[:, end, col] - C[:, a, col]
        value = total / count.clamp_min(1)
        value[:, count == 0] = float("nan")
        out.append(value)
    return torch.stack(out, 0).permute(2, 0, 1)                         # n x delays x heads


# ---------------------------------------------------------------- tier 1: output

def _distribution(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)
    varent = (p * (logp + ent[:, None]) ** 2).sum(-1)
    top = p.topk(2, -1)
    return {"logp": logp, "ent": ent, "varent": varent, "top_p": top.values, "top_ids": top.indices}


def tier1_features(reg: Registry, cap: Capture, response_ids: list[int], token_types: list[str],
                   step_index: list[int]) -> dict[str, np.ndarray]:
    L = len(response_ids)
    dist = _distribution(cap.logits)
    y = torch.tensor(response_ids, device=cap.logits.device)
    rows = torch.arange(L, device=y.device)
    logp_chosen = dist["logp"][rows, y]
    rank = (cap.logits[rows] > cap.logits[rows, y][:, None]).sum(-1) + 1
    ent_all, pmax_all = _np(dist["ent"]), _np(dist["top_p"][:, 0])
    ent, pmax = ent_all[:L], pmax_all[:L]
    lp = _np(logp_chosen)
    add = lambda name, values, delay: reg.add(name, values, tier=1, delay=delay, family="output", table="tier1")
    add("ent", ent, "pre")
    add("varent", _np(dist["varent"])[:L], "pre")
    add("pmax", pmax, "pre")
    add("p_chosen", np.exp(lp), "pre")
    add("logp_chosen", lp, "pre")
    add("rank_chosen", _np(rank), "pre")
    add("margin12", _np(dist["top_p"][:L, 0] - dist["top_p"][:L, 1]), "pre")
    add("surprisal_gap", -lp - ent, "pre")
    add("ent_z", causal.window_zscore(ent), "pre")
    add("pmax_z", causal.window_zscore(pmax), "pre")
    add("ent_ema_delta", causal.ema_delta(ent), "pre")
    add("ent_prev_delta", causal.prev_delta(ent), "pre")
    add("warmup", causal.warmup(L), "pre")
    for kind in TOKEN_TYPES:
        add(f"tt_{kind}", [token == kind for token in token_types], "pre")
    add("r", np.arange(L), "pre")
    add("step_index", step_index, "pre")
    add("ent_next", ent_all[1: L + 1], "post")
    add("pmax_next", pmax_all[1: L + 1], "post")
    for k in K_DELAYS:
        add(f"ent_mean_k{k}", causal.future(ent, k, "mean"), f"k{k}")
        add(f"ent_max_k{k}", causal.future(ent, k, "max"), f"k{k}")
        add(f"logp_chosen_mean_k{k}", causal.future(lp, k, "mean"), f"k{k}")
    return {"ent": ent_all}


# ---------------------------------------------------------------- tier 2: attention

def tier2_features(reg: Registry, cap: Capture, n_response: int) -> None:
    """Head-averaged per-layer row statistics (pre/post) and attention received (k)."""
    if cap.attn_stats is None:
        return
    stats = _np(cap.attn_stats)                                         # R x layers x heads x stats
    L = n_response
    for layer in range(stats.shape[1]):
        mean = stats[:, layer].mean(1)                                  # R x stats
        for index, name in enumerate(ATTN_STATS):
            reg.add(f"{name}_L{layer + 1}", mean[:L, index], tier=2, delay="pre", family="attention", table="tier2", layer=layer + 1)
            reg.add(f"{name}_L{layer + 1}", mean[1: L + 1, index], tier=2, delay="post", family="attention", table="tier2", layer=layer + 1)
    received = _np(cap.received)                                        # n x delays x layers x heads
    for d_index, k in enumerate(K_DELAYS):
        for layer in range(received.shape[2]):
            values = received[:L, d_index, layer]
            reg.add(f"att_recv_mean_L{layer + 1}", values.mean(-1), tier=2, delay=f"k{k}", family="attention_received", table="tier2", layer=layer + 1)
            reg.add(f"att_recv_max_L{layer + 1}", values.max(-1), tier=2, delay=f"k{k}", family="attention_received", table="tier2", layer=layer + 1)


# ---------------------------------------------------------------- tier 3: hidden states

def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def _row_tokens(cap: Capture, response_ids: list[int]) -> dict[str, dict[str, torch.Tensor]]:
    """Chosen / alternative / top-5 tokens per delay. pre: y_r and the best other token of
    z[a-1]; post: argmax and second-best of z[a]."""
    L = len(response_ids)
    logits = cap.logits
    y = torch.tensor(response_ids, device=logits.device)
    rows = torch.arange(L, device=logits.device)
    masked = logits[:L].clone()
    masked[rows, y] = -float("inf")
    top5 = logits.topk(5, -1).indices
    top2 = logits.topk(2, -1).indices
    return {
        "pre": {"rows": rows, "chosen": y, "alt": masked.argmax(-1), "top5": top5[:L]},
        "post": {"rows": rows + 1, "chosen": top2[1: L + 1, 0], "alt": top2[1: L + 1, 1], "top5": top5[1: L + 1]},
    }


def hidden_norm_features(reg: Registry, cap: Capture, n_response: int) -> None:
    L = n_response
    n_layers = cap.mid.shape[0]
    for delay, offset in (("pre", 1), ("post", 2)):
        idx = torch.arange(L, device=cap.resid.device) + offset        # residual index of q
        for layer in range(1, n_layers + 1):
            h, prev = cap.resid[layer, idx].float(), cap.resid[layer - 1, idx].float()
            update = h - prev
            add = lambda name, values: reg.add(f"{name}_L{layer}", _np(values), tier=3, delay=delay,
                                               family="hidden_norms", table="tier3_derived", layer=layer)
            add("h_norm", h.norm(dim=-1))
            add("upd_norm", update.norm(dim=-1))
            add("attn_out_norm", cap.attn_out_norm[layer - 1, idx])
            add("mlp_out_norm", cap.mlp_out_norm[layer - 1, idx])
            add("upd_cos", _cos(update, prev))
            add("tmp_cos", _cos(h, cap.resid[layer, idx - 1]))


def logit_lens_features(reg: Registry, cap: Capture, head: PredictionHead, tokens: dict[str, dict[str, torch.Tensor]]) -> None:
    n_layers = cap.mid.shape[0]
    for delay, info in tokens.items():
        rows, chosen = info["rows"], info["chosen"]
        # "Final" is the lens at the last layer, in the same float32 as every other layer. The
        # model's own logits are bf16-rounded (|Δ| up to ~0.25), which would flip near-tied
        # top-1s at the last layer and make commit_layer spuriously "never".
        final = head.logits(head.normed(cap.resid[n_layers, rows + 1]))
        f_logp, f_top1 = torch.log_softmax(final, -1), final.argmax(-1)
        del final
        top1s, chosen_top = [], []
        for layer in range(1, n_layers + 1):
            logits = head.logits(head.normed(cap.resid[layer, rows + 1]))
            logp = torch.log_softmax(logits, -1)
            p = logp.exp()
            chosen_logit = logits.gather(-1, chosen[:, None])
            top1 = logits.argmax(-1)
            add = lambda name, values: reg.add(f"{name}_L{layer}", _np(values), tier=3, delay=delay,
                                               family="logit_lens", table="tier3_derived", layer=layer)
            add("ll_ent", -(p * logp).sum(-1))
            add("ll_p_chosen", p.gather(-1, chosen[:, None])[:, 0])
            add("ll_rank_chosen", (logits > chosen_logit).sum(-1) + 1)
            # KL from the layer's distribution to the final one: KL(p_layer ‖ p_final).
            add("ll_kl_final", (p * (logp - f_logp)).sum(-1))
            add("ll_top1_matches_final", top1 == f_top1)
            top1s.append(top1 == f_top1)
            chosen_top.append(top1 == chosen)
            flips = torch.zeros_like(top1) if layer == 1 else flips + (top1 != previous_top1)
            previous_top1 = top1
        add_agg = lambda name, values: reg.add(name, values, tier=3, delay=delay, family="logit_lens", table="tier3_derived")
        add_agg("commit_layer", _commit_layer(torch.stack(top1s)))
        add_agg("n_top1_flips", _np(flips))
        add_agg("chosen_commit_layer", _commit_layer(torch.stack(chosen_top)))


def _commit_layer(matches: torch.Tensor) -> np.ndarray:
    """First layer (1-based) from which ``matches`` holds at every later layer; inf if never."""
    m = _np(matches).astype(bool)                                       # layers x rows
    suffix = np.flip(np.cumprod(np.flip(m, 0), 0), 0).astype(bool)      # holds from here to the end
    out = np.full(m.shape[1], np.inf)
    any_hold = suffix.any(0)
    out[any_hold] = suffix[:, any_hold].argmax(0) + 1
    return out


def head_alignment_features(reg: Registry, cap: Capture, head: PredictionHead, tokens: dict[str, dict[str, torch.Tensor]]) -> dict[str, np.ndarray]:
    """Groups 1-3 (final layer; pre and post) and Group 2 direct logit attribution."""
    n_layers = cap.mid.shape[0]
    checks = {}
    for delay, info in tokens.items():
        rows, chosen, alt = info["rows"], info["chosen"], info["alt"]
        h_final = cap.resid[n_layers, rows + 1]
        h_hat = head.normed(h_final)
        w_c, w_a = head.weight[chosen], head.weight[alt]
        add = lambda name, values: reg.add(name, _np(values) if torch.is_tensor(values) else values, tier=3, delay=delay,
                                           family="head_align", table="tier3_derived")
        cos_c, cos_a = _cos(h_hat, w_c), _cos(h_hat, w_a)
        add("head_scale", h_hat.norm(dim=-1))
        add("head_cos_chosen", cos_c)
        add("head_cos_alt", cos_a)
        add("head_cos_gap", cos_c - cos_a)
        add("head_wnorm_chosen", w_c.norm(dim=-1))
        add("head_wnorm_alt", w_a.norm(dim=-1))
        add("readout_frac_64", head.readout_frac(h_hat, 64))
        add("readout_frac_512", head.readout_frac(h_hat, 512))
        add("cand_span_resid", head.cand_span_resid(h_hat, info["top5"]))
        # Direct logit attribution with the final RMSNorm's scale frozen. Block outputs are
        # the residual increments actually added (bf16 differences are exact in float32),
        # so the contributions telescope to the final residual.
        direction = (w_c - w_a) * head.norm_weight
        scale = head.rms(h_final)[:, 0]
        idx = rows + 1
        blocks = [cap.resid[0, idx]]
        for layer in range(n_layers):
            blocks.append(cap.mid[layer, idx].float() - cap.resid[layer, idx].float())
            blocks.append(cap.resid[layer + 1, idx].float() - cap.mid[layer, idx].float())
        contrib = torch.stack([(block.float() * direction).sum(-1) / scale for block in blocks])  # blocks x rows
        dla = lambda name, values, layer=None: reg.add(name, _np(values) if torch.is_tensor(values) else values, tier=3, delay=delay,
                                                       family="dla", table="tier3_derived", layer=layer)
        dla("dla_embed", contrib[0])
        for layer in range(1, n_layers + 1):
            dla(f"dla_attn_L{layer}", contrib[2 * layer - 1], layer)
            dla(f"dla_mlp_L{layer}", contrib[2 * layer], layer)
        total = contrib.sum(0)
        conflict = (-contrib.clamp_max(0)).sum(0)
        dla("dla_total", total)
        dla("dla_conflict", conflict)
        dla("dla_conflict_frac", conflict / contrib.abs().sum(0))
        dla("dla_decision_layer", _decision_layer(_np(contrib)))
        dla("dla_top_block", _np(contrib.argmax(0)))
        checks[f"logit_gap_{delay}"] = _np((h_hat * (w_c - w_a)).sum(-1))
        checks[f"dla_total_{delay}"] = _np(total)
    return checks


def _decision_layer(contrib: np.ndarray) -> np.ndarray:
    """Last layer at which the running block sum changes sign (blocks: embed, attn1, mlp1, ...); 0 if never."""
    running = np.cumsum(contrib, 0)
    sign = np.sign(running)
    changes = (sign[1:] != sign[:-1]) & (sign[1:] != 0)                 # change at block index b+1
    layer_of_block = (np.arange(1, contrib.shape[0]) + 1) // 2          # block b>=1 belongs to layer ceil(b/2)
    out = np.zeros(contrib.shape[1])
    for row in range(contrib.shape[1]):
        where = np.nonzero(changes[:, row])[0]
        if len(where):
            out[row] = layer_of_block[where[-1]]
    return out


def depth_alignment_features(reg: Registry, cap: Capture, head: PredictionHead, tokens: dict[str, dict[str, torch.Tensor]]) -> None:
    """Group 4: per-layer alignment of ĥ^ℓ = final RMSNorm(h^ℓ[a-1]) with the head (pre)."""
    info = tokens["pre"]
    rows, chosen, alt = info["rows"], info["chosen"], info["alt"]
    w_c, w_a = head.weight[chosen], head.weight[alt]
    for layer in range(1, cap.mid.shape[0] + 1):
        h_hat = head.normed(cap.resid[layer, rows + 1])
        add = lambda name, values: reg.add(f"{name}_L{layer}", _np(values), tier=3, delay="pre",
                                           family="head_align_depth", table="tier3_derived", layer=layer)
        cos_c = _cos(h_hat, w_c)
        add("head_cos_chosen", cos_c)
        add("head_cos_gap", cos_c - _cos(h_hat, w_a))
        add("readout_frac_64", head.readout_frac(h_hat, 64))
        add("cand_span_resid", head.cand_span_resid(h_hat, info["top5"]))


def phi_features(reg: Registry, cap: Capture, head: PredictionHead, n_response: int,
                 mu_ref: dict[int, torch.Tensor], phi_layers: list[int]) -> None:
    """Group 5: Φ-Align per token and over causal windows (pre: rows ≤ r; post: rows ≤ r+1)."""
    L = n_response
    n_layers = cap.mid.shape[0]
    d = head.d
    rows = torch.arange(L + 1, device=cap.resid.device)
    frob = head.phi_frobenius
    strengths = head.phi_strengths.double().cpu().numpy()
    h_hat = head.normed(cap.resid[n_layers, rows + 1])                  # all query rows
    tok = {
        "phi_tok_ref": _np(head.phi_tok(h_hat - mu_ref[n_layers])),
        "phi_tok_prompt": _np(head.phi_tok(h_hat - cap.prompt_mean_hat)),
        "phi_tok_raw": _np(head.phi_tok(h_hat)),
    }
    u = head.phi_coords(h_hat).double().cpu().numpy()
    win = {f"phi_win_{w}": causal.window_phi(u, strengths, frob, w) for w in (32, 64)}
    win["phi_win_64_delta16"] = causal.lagged_delta(win["phi_win_64"], 16)
    zscore = causal.window_zscore(win["phi_win_64"])
    window_size = np.minimum(np.arange(L + 1) + 1, 64)
    per_layer = {}
    for layer in phi_layers:
        hl = head.normed(cap.resid[layer, rows + 1])
        per_layer[layer] = {"phi_tok_ref": _np(head.phi_tok(hl - mu_ref[layer])),
                            "phi_win_64": causal.window_phi(head.phi_coords(hl).double().cpu().numpy(), strengths, frob, 64)}
    # Every Φ feature also has a width-normalized `_dnorm` version (× d; isotropic baseline 1).
    for delay, sl in (("pre", slice(0, L)), ("post", slice(1, L + 1))):
        def add(name: str, values: np.ndarray, layer: int | None = None) -> None:
            reg.add(name, values[sl], tier=3, delay=delay, family="phi", table="tier3_derived", layer=layer)
            reg.add(f"{name}_dnorm" if layer is None else name.replace(f"_L{layer}", f"_dnorm_L{layer}"),
                    values[sl] * d, tier=3, delay=delay, family="phi", table="tier3_derived", layer=layer)
        for name, values in {**tok, **win, "phi_win_64_z": zscore}.items():
            add(name, values)
        reg.add("phi_warmup", (window_size < causal.WARMUP)[sl], tier=3, delay=delay, family="phi", table="tier3_derived")
        for layer, values in per_layer.items():
            for name, series in values.items():
                add(f"{name}_L{layer}", series, layer)


# ---------------------------------------------------------------- one trace

def step_indices(token_texts: list[str]) -> list[int]:
    out, newlines = [], 0
    for text in token_texts:
        out.append(newlines)
        newlines += text.count("\n")
    return out


def trace_features(cap: Capture, head: PredictionHead, response_ids: list[int], token_texts: list[str],
                   token_types: list[str], mu_ref: dict[int, torch.Tensor], phi_layers: list[int]) -> tuple[Registry, dict[str, Any]]:
    """Every feature family for one trace's response tokens r = 0 .. len(response_ids)-1."""
    reg = Registry()
    L = len(response_ids)
    tier1_features(reg, cap, response_ids, token_types, step_indices(token_texts))
    tier2_features(reg, cap, L)
    tokens = _row_tokens(cap, response_ids)
    hidden_norm_features(reg, cap, L)
    logit_lens_features(reg, cap, head, tokens)
    checks = head_alignment_features(reg, cap, head, tokens)
    depth_alignment_features(reg, cap, head, tokens)
    phi_features(reg, cap, head, L, mu_ref, phi_layers)
    return reg, checks
