"""Per-family online cost (Section 4.3.6): extra latency per token in a decoder loop.

    python -m ctfeat.cost --dataset dataset/ --features features/ --out results/

HF ``generate`` with a KV cache, batch size 1, 200 new tokens; each family's hooks do
that family's per-token work on the newest position. Latency is the median over
repeats minus the plain-decoding baseline (the pinned implementation, sdpa). The
attention families decode with the same sdpa computation plus float32 attention
weights (attention.WEIGHTS_IMPL, as in extraction); their cost includes computing
those weights, and the weights-only overhead is reported separately.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from .attention import WEIGHTS_IMPL
from .data import iter_traces, load_meta
from .extract import load_model
from .head import PredictionHead

VLLM = {
    "output": "partial: top-k logprobs are returned; full-vocabulary entropy/varentropy needs a logits processor",
    "attention": "no: attention weights are not exposed (flash/paged kernels); needs a custom attention backend",
    "attention_received": "no: as attention, and needs k future steps",
    "hidden_norms": "no: needs model-runner hooks",
    "logit_lens": "no: needs hidden states per layer",
    "head_align": "no: needs the final hidden state (not exposed without hooks)",
    "head_align_depth": "no: needs hidden states from every decoder layer",
    "dla": "no: needs per-block outputs",
    "phi": "no: needs the final hidden state",
}


def _last(output: Any) -> torch.Tensor:
    return (output[0] if isinstance(output, tuple) else output)[0, -1]


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def _distribution(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    return logp, p, p.topk(5)


def family_hooks(model: Any, head: PredictionHead, family: str, sink: list,
                 *, prompt_len: int = 0, spans: dict[str, list[int]] | None = None) -> list[Any]:
    """Install hooks that execute the registered family's complete newest-token workload.

    The benchmark keeps only tiny summaries in ``sink``. Intermediate states live in this
    invocation's closures and are replaced at each decoder step, matching an online loop.
    """
    layers = model.model.layers
    handles = []
    if family == "output":
        history: list[torch.Tensor] = []
        pmax_history: list[torch.Tensor] = []
        ema: torch.Tensor | None = None

        def output_hook(_m, _i, out):
            nonlocal ema
            logits = out[0, -1].float()
            logp, p, top = _distribution(logits)
            ent = -(p * logp).sum()
            varent = (p * (logp + ent) ** 2).sum()
            chosen = top.indices[0]
            chosen_logp = logp[chosen]
            earlier = torch.stack(history[-64:]) if history else None
            z = torch.zeros_like(ent) if earlier is None or len(earlier) < 2 else (ent - earlier.mean()) / earlier.std(unbiased=False).clamp_min(1e-8)
            earlier_pmax = torch.stack(pmax_history[-64:]) if pmax_history else None
            pmax_z = (torch.zeros_like(ent) if earlier_pmax is None or len(earlier_pmax) < 2 else
                      (top.values[0] - earlier_pmax.mean()) / earlier_pmax.std(unbiased=False).clamp_min(1e-8))
            delta_ema = torch.zeros_like(ent) if ema is None else ent - ema
            prev = torch.zeros_like(ent) if not history else ent - history[-1]
            ema = ent if ema is None else (1 - (1 - 2 ** (-1 / 16))) * ema + (1 - 2 ** (-1 / 16)) * ent
            history.append(ent.detach())
            pmax_history.append(top.values[0].detach())
            sink.append(torch.stack([ent, varent, top.values[0], p[chosen], chosen_logp,
                                     (logits > logits[chosen]).sum().float() + 1,
                                     top.values[0] - top.values[1], -chosen_logp - ent, z, pmax_z, delta_ema, prev]))
        handles.append(model.lm_head.register_forward_hook(output_hook))
    elif family == "attention":
        spans = spans or {"fewshot_span": [0, 0], "question_span": [0, 0], "cue_span": [0, 0]}

        def attn(_m, _i, out):
            w = out[1][0, :, -1].float()
            top, argmax = w.max(-1)
            q = w.shape[-1] - 1
            mass = lambda span: w[:, span[0]:span[1]].sum(-1)
            question = mass(spans["question_span"])
            response = w[:, prompt_len:].sum(-1)
            sink.append(torch.stack([
                -(w * w.clamp_min(1e-30).log()).sum(-1), top, q - argmax.float(),
                mass(spans["fewshot_span"]), question, mass(spans["cue_span"]), response,
                w[:, -16:].sum(-1), w[:, 0], question / (question + response + 1e-8),
            ]))
        handles += [layer.self_attn.register_forward_hook(attn) for layer in layers]
    elif family == "attention_received":
        receiver_sums: dict[int, torch.Tensor] = {}

        def received(index: int):
            def hook(_m, inputs, kwargs, out):
                # The prefill row has no emitted response key yet. Decoder rows update the
                # accumulators of the previous 64 response keys, then append the new key.
                hidden = inputs[0] if inputs else kwargs["hidden_states"]
                if hidden.shape[1] != 1:
                    receiver_sums[index] = torch.empty(out[1].shape[1], 0, device=out[1].device, dtype=torch.float32)
                    return
                w = out[1][0, :, -1].float()
                sums = receiver_sums[index]
                prior = min(sums.shape[1], w.shape[1] - 1, 64)
                if prior:
                    sums[:, -prior:] += w[:, -prior - 1:-1]
                sums = torch.cat([sums, torch.zeros(sums.shape[0], 1, device=sums.device)], -1)[:, -65:]
                receiver_sums[index] = sums
                ready = [sums[:, -k - 1] / k if sums.shape[1] > k else torch.zeros(sums.shape[0], device=sums.device)
                         for k in (4, 16, 64)]
                sink.append(torch.stack(ready))
            return hook
        handles += [layer.self_attn.register_forward_hook(received(index), with_kwargs=True) for index, layer in enumerate(layers)]
    elif family == "hidden_norms":
        previous: dict[int, torch.Tensor] = {}

        def layer_hook(index: int):
            def hook(_m, inp, out):
                incoming, hidden = inp[0][0, -1].float(), _last(out).float()
                update = hidden - incoming
                temporal = torch.zeros((), device=hidden.device) if index not in previous else _cos(hidden[None], previous[index][None])[0]
                previous[index] = hidden.detach()
                sink.append(torch.stack([hidden.norm(), update.norm(), _cos(update[None], incoming[None])[0], temporal]))
            return hook

        handles += [layer.register_forward_hook(layer_hook(i)) for i, layer in enumerate(layers)]
        handles += [layer.self_attn.register_forward_hook(lambda _m, _i, out: sink.append(_last(out).float().norm()[None])) for layer in layers]
        handles += [layer.mlp.register_forward_hook(lambda _m, _i, out: sink.append(_last(out).float().norm()[None])) for layer in layers]
    elif family == "logit_lens":
        layer_logits: dict[int, torch.Tensor] = {}
        handles += [layer.register_forward_hook(
            lambda _m, _i, out, index=index: layer_logits.__setitem__(index, head.logits(head.normed(_last(out)[None]))[0]))
            for index, layer in enumerate(layers)]

        def lens_final(_m, _i, out):
            chosen = out[0, -1].argmax()
            final_logp = torch.log_softmax(layer_logits[len(layers) - 1], -1)     # last-layer lens, as in features.py
            values, previous = [], None
            for index in range(len(layers)):
                logits = layer_logits[index]
                logp, p, _ = _distribution(logits)
                top1 = logits.argmax()
                values += [-(p * logp).sum(), p[chosen], (logits > logits[chosen]).sum().float() + 1,
                           (p * (logp - final_logp)).sum(), (top1 == final_logp.argmax()).float()]
                if previous is not None:
                    values.append((top1 != previous).float())
                previous = top1
            sink.append(torch.stack(values))
        handles.append(model.lm_head.register_forward_hook(lens_final))
    elif family in ("head_align", "head_align_depth"):
        states: dict[int, torch.Tensor] = {}
        selected = range(len(layers)) if family == "head_align_depth" else (len(layers) - 1,)
        handles += [layers[index].register_forward_hook(
            lambda _m, _i, out, index=index: states.__setitem__(index, head.normed(_last(out)[None]))) for index in selected]

        def align_final(_m, _i, out):
            logits = out[0, -1].float()
            top = logits.topk(5).indices
            chosen, alt = top[0], top[1]
            wc, wa = head.weight[chosen][None], head.weight[alt][None]
            values = []
            for index in selected:
                h_hat = states[index]
                cos_c, cos_a = _cos(h_hat, wc), _cos(h_hat, wa)
                if family == "head_align":
                    values += [h_hat.norm(dim=-1), cos_c, cos_a, cos_c - cos_a, wc.norm(dim=-1), wa.norm(dim=-1),
                               head.readout_frac(h_hat, 64), head.readout_frac(h_hat, 512),
                               head.cand_span_resid(h_hat, top[None])]
                else:
                    values += [cos_c, cos_c - cos_a, head.readout_frac(h_hat, 64),
                               head.cand_span_resid(h_hat, top[None])]
            sink.append(torch.cat(values))
        handles.append(model.lm_head.register_forward_hook(align_final))
    elif family == "dla":
        block: dict[str, Any] = {"attn": {}, "mlp": {}}
        handles.append(layers[0].register_forward_pre_hook(
            lambda _m, inp: block.__setitem__("embed", inp[0][0, -1].float())))
        handles += [layer.self_attn.register_forward_hook(
            lambda _m, _i, out, index=index: block["attn"].__setitem__(index, _last(out).float()))
            for index, layer in enumerate(layers)]
        handles += [layer.mlp.register_forward_hook(
            lambda _m, _i, out, index=index: block["mlp"].__setitem__(index, _last(out).float()))
            for index, layer in enumerate(layers)]
        handles.append(layers[-1].register_forward_hook(
            lambda _m, _i, out: block.__setitem__("final", _last(out).float())))

        def dla_final(_m, _i, out):
            top = out[0, -1].float().topk(2).indices
            direction = (head.weight[top[0]] - head.weight[top[1]]) * head.norm_weight
            scale = head.rms(block["final"][None])[0, 0]
            contributions = [(block["embed"] * direction).sum() / scale]
            for index in range(len(layers)):
                contributions += [(block["attn"][index] * direction).sum() / scale,
                                  (block["mlp"][index] * direction).sum() / scale]
            c = torch.stack(contributions)
            running = c.cumsum(0)
            changes = ((running[1:].sign() != running[:-1].sign()) & (running[1:] != 0)).nonzero()
            decision = torch.zeros((), device=c.device) if not len(changes) else ((changes[-1, 0] + 2) // 2).float()
            sink.append(torch.stack([c.sum(), (-c.clamp_max(0)).sum(), (-c.clamp_max(0)).sum() / c.abs().sum().clamp_min(1e-8),
                                     decision, c.argmax().float()]))
        handles.append(model.lm_head.register_forward_hook(dla_final))
    elif family == "phi":
        phi_layers = sorted(set(range(3, len(layers), 4)) | {len(layers) - 1})
        states: dict[int, torch.Tensor] = {}
        history: dict[int, list[torch.Tensor]] = {index: [] for index in phi_layers}
        final_windows: list[torch.Tensor] = []
        handles += [layers[index].register_forward_hook(
            lambda _m, _i, out, index=index: states.__setitem__(index, head.normed(_last(out)[None]))) for index in phi_layers]

        def phi_final(_m, _i, _out):
            values = []
            for index in phi_layers:
                h_hat = states[index]
                coords = head.phi_coords(h_hat)[0]
                history[index].append(coords.detach())
                window = torch.stack(history[index][-64:])
                var = window.var(0, unbiased=False) if len(window) > 1 else torch.zeros_like(coords)
                denom = var.sum() * head.phi_frobenius
                win = (head.phi_strengths * var).sum() / denom.clamp_min(1e-12)
                if index == len(layers) - 1:
                    window32 = torch.stack(history[index][-32:])
                    var32 = window32.var(0, unbiased=False) if len(window32) > 1 else torch.zeros_like(coords)
                    win32 = ((head.phi_strengths * var32).sum() /
                             (var32.sum() * head.phi_frobenius).clamp_min(1e-12))
                    delta = torch.zeros_like(win) if len(final_windows) < 16 else win - final_windows[-16]
                    earlier = torch.stack(final_windows[-64:]) if final_windows else None
                    z = (torch.zeros_like(win) if earlier is None or len(earlier) < 2 else
                         (win - earlier.mean()) / earlier.std(unbiased=False).clamp_min(1e-8))
                    final_windows.append(win.detach())
                    raw = [head.phi_tok(h_hat)[0], head.phi_tok(h_hat)[0], head.phi_tok(h_hat)[0], win32, win, delta, z]
                    values += raw + [value * head.d for value in raw]
                else:
                    raw = [head.phi_tok(h_hat)[0], win]
                    values += raw + [value * head.d for value in raw]
            sink.append(torch.stack(values))
        handles.append(model.lm_head.register_forward_hook(phi_final))
    else:
        raise ValueError(f"unknown cost family {family!r}")
    return handles


def _entropy_stats(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    ent = -(p * logp).sum()
    return torch.stack([ent, (p * (logp + ent) ** 2).sum(), p.max()])


@torch.inference_mode()
def time_generate(model: Any, ids: list[int], tokens: int, repeats: int, before: Callable[[], list[Any]]) -> float:
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device)
    samples = []
    for _ in range(repeats + 1):
        handles = before()
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate(inputs, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) / tokens * 1000)
        for handle in handles:
            handle.remove()
    return float(sorted(samples[1:])[len(samples[1:]) // 2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    meta = load_meta(args.dataset)
    model, _ = load_model(meta, "cuda", attn=meta.get("attn_implementation", "sdpa"))
    head = PredictionHead.from_model(model)
    trace = next(iter_traces(args.dataset))
    ids = trace["prompt_ids"]
    spans = {key: trace[key] for key in ("fewshot_span", "question_span", "cue_span")}
    rows = []
    sink: list = []
    base_sdpa = time_generate(model, ids, args.tokens, args.repeats, lambda: [])
    rows.append({"family": "baseline", "attn_implementation": model.config._attn_implementation, "ms_per_token": base_sdpa,
                 "extra_ms_per_token": 0.0, "vllm": "yes"})
    for family in ("output", "hidden_norms", "logit_lens", "head_align", "head_align_depth", "dla", "phi"):
        ms = time_generate(model, ids, args.tokens, args.repeats,
                           lambda f=family: family_hooks(model, head, f, sink, prompt_len=len(ids), spans=spans))
        sink.clear()
        rows.append({"family": family, "attn_implementation": model.config._attn_implementation, "ms_per_token": ms,
                     "extra_ms_per_token": ms - base_sdpa, "vllm": VLLM[family]})
    model.set_attn_implementation(WEIGHTS_IMPL)
    base_weights = time_generate(model, ids, args.tokens, args.repeats, lambda: [])
    for family in ("attention", "attention_received"):
        ms = time_generate(model, ids, args.tokens, args.repeats,
                           lambda f=family: family_hooks(model, head, f, sink, prompt_len=len(ids), spans=spans))
        sink.clear()
        rows.append({"family": family, "attn_implementation": WEIGHTS_IMPL, "ms_per_token": ms,
                     "extra_ms_per_token": ms - base_sdpa, "weights_only_extra_ms_per_token": base_weights - base_sdpa,
                     "vllm": VLLM[family]})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "cost.csv", index=False)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
