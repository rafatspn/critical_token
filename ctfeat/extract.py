"""Causal feature extraction: one teacher-forced pass per trace.

    python -m ctfeat.extract --dataset dataset/ --out features/ [--traces N] [--layers every4]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .attention import WEIGHTS_IMPL, reconstruction_errors, using
from .capture import RECEIVED_DELAYS, run_capture
from .data import iter_traces, load_meta, load_token_types
from .features import ATTN_STATS, trace_features
from .head import PredictionHead
from ctlabel.select import token_type as step1_token_type

TABLES = ("tier1", "tier2", "tier3_derived")
LOGIT_TOL = 1e-2            # max |Δlogit| (test 5, test 6)
TOP1_TOL = 0.999            # top-1 agreement (test 5)
RECONSTRUCTION_TOL = 1e-2   # max |weights·V − sdpa output| / max |sdpa output|, per layer


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "every4":
        return [layer for layer in range(4, n_layers + 1, 4)] or [n_layers]
    if spec == "all":
        return list(range(1, n_layers + 1))
    layers = sorted({int(item) for item in spec.split(",")})
    if not all(1 <= layer <= n_layers for layer in layers):
        raise ValueError(f"layers must be in 1..{n_layers}: {spec}")
    return layers


def load_model(meta: dict[str, Any], device: str, attn: str = WEIGHTS_IMPL) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = meta["config"]
    revision = meta.get("model_revision")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision, trust_remote_code=config.get("trust_remote_code", True))
    dtype = getattr(torch, meta.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=revision, dtype=dtype, attn_implementation=attn,
        trust_remote_code=config.get("trust_remote_code", True),
    ).to(device).eval()
    return model, tokenizer


def token_texts(tokenizer: Any, response_ids: list[int]) -> list[str]:
    return [tokenizer.decode([token], skip_special_tokens=False) for token in response_ids]


def trace_inputs(trace: dict[str, Any], tokenizer: Any, types: dict[str, list[str]]) -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Token texts, token types and prompt spans for one trace.

    token_type comes from data.load_token_types; trailing stop tokens have no label row, so
    they get step 1's own token_type() of the token text.
    """
    texts = token_texts(tokenizer, trace["response_ids"])
    trace_types = types.get(trace["trace_id"], [])
    trace_types = trace_types + [step1_token_type(text) for text in texts[len(trace_types):]]
    spans = {key: trace[key] for key in ("fewshot_span", "question_span", "cue_span")}
    return texts, trace_types, spans


@torch.inference_mode()
def reference_means(model: Any, tokenizer: Any, head: PredictionHead, layers: list[int], dataset: str,
                    n_prompts: int, seed: int) -> dict[int, torch.Tensor]:
    """μ_ref per layer: mean of ĥ^ℓ over all tokens of ``n_prompts`` MATH train prompts (few-shot + question)."""
    from ctlabel.datasets import load_problems
    from ctlabel.prompts import prompt_parts

    problems = load_problems(dataset.removesuffix("_test"))
    problems = random.Random(seed).sample(problems, min(n_prompts, len(problems)))
    sums = {layer: torch.zeros(head.d, dtype=torch.float64, device=head.weight.device) for layer in layers}
    count = 0
    store: dict[int, torch.Tensor] = {}
    hooks = [model.model.layers[layer - 1].register_forward_hook(
        lambda _m, _i, out, layer=layer: store.__setitem__(layer, (out[0] if isinstance(out, tuple) else out)[0]))
        for layer in layers]
    try:
        for problem in problems:
            fewshot, question, _ = prompt_parts(dataset, problem["question"])
            ids = tokenizer.encode(fewshot + question, add_special_tokens=False)
            model.model(input_ids=torch.tensor([ids], device=head.weight.device))
            for layer in layers:
                sums[layer] += head.normed(store[layer]).double().sum(0)
            count += len(ids)
    finally:
        for hook in hooks:
            hook.remove()
    return {layer: (total / count).float() for layer, total in sums.items()}


@torch.inference_mode()
def attention_consistency(model: Any, traces: list[dict[str, Any]], pinned: str) -> dict[str, Any]:
    """Test 5 for the attention used by the features (see attention.py).

    On each trace: logits with float32 weights returned (WEIGHTS_IMPL) must match the pinned
    implementation, and every layer's float32 weights times V must reproduce the sdpa output.
    Eager attention is measured as well, for the record only: it is not used.
    """
    device = next(model.parameters()).device
    results = []
    for trace in traces:
        ids = torch.tensor([trace["prompt_ids"] + trace["response_ids"]], device=device)
        keep = len(trace["response_ids"]) + 1
        logits = {}
        for impl in (pinned, WEIGHTS_IMPL, "eager"):
            with using(model, impl), reconstruction_errors() as errors:
                logits[impl] = model(input_ids=ids, logits_to_keep=keep).logits[0].float()
            if impl == WEIGHTS_IMPL:
                reconstruction = errors
        compare = lambda impl: ((logits[pinned] - logits[impl]).abs().max().item(),
                                (logits[pinned].argmax(-1) == logits[impl].argmax(-1)).float().mean().item())
        diff, top1 = compare(WEIGHTS_IMPL)
        eager_diff, eager_top1 = compare("eager")
        results.append({"trace_id": trace["trace_id"], "max_abs_logit_diff": diff, "top1_agreement": top1,
                        "max_rel_reconstruction_err": max(reconstruction), "reconstruction_err_by_layer": reconstruction,
                        "eager_max_abs_logit_diff": eager_diff, "eager_top1_agreement": eager_top1})
        del logits
    worst = max(item["max_abs_logit_diff"] for item in results)
    agreement = float(np.mean([item["top1_agreement"] for item in results]))
    reconstruction = max(item["max_rel_reconstruction_err"] for item in results)
    return {"pinned": pinned, "features_attention": WEIGHTS_IMPL, "traces": results,
            "max_abs_logit_diff": worst, "top1_agreement": agreement, "max_rel_reconstruction_err": reconstruction,
            "eager_max_abs_logit_diff": max(item["eager_max_abs_logit_diff"] for item in results),
            "eager_top1_agreement": float(np.mean([item["eager_top1_agreement"] for item in results])),
            "passed": worst <= LOGIT_TOL and agreement >= TOP1_TOL and reconstruction <= RECONSTRUCTION_TOL}


def disk_estimate(traces: list[dict[str, Any]], n_layers: int, heads: int, d: int, hidden_layers: int) -> dict[str, float]:
    rows = sum(len(trace["response_ids"]) + 1 for trace in traces)
    return {
        "hidden_GB": rows * hidden_layers * d * 2 / 1e9,
        "attn_GB": rows * n_layers * heads * len(ATTN_STATS) * 2 / 1e9,
        "attn_recv_GB": rows * len(RECEIVED_DELAYS) * n_layers * heads * 2 / 1e9,
    }


@torch.inference_mode()
def head_checks(model: Any, cap: Any, head: PredictionHead, checks: dict[str, np.ndarray],
                rng: np.random.Generator) -> dict[str, float]:
    """Test 6: ĥ·w reproduces the model's logits, and dla_total matches the logit gap.

    The model's logits are computed in bf16 (|Δ| up to ~0.25 at |logit| ~ 40 is bf16
    rounding), so the check uses the float32 option: the model's own final norm and
    lm_head applied in float32 to its final residual. The bf16 difference is recorded.
    """
    n_layers = cap.mid.shape[0]
    rows = torch.arange(cap.logits.shape[0], device=cap.logits.device)
    final = cap.resid[n_layers, rows + 1]
    ours = head.logits(head.normed(final))
    reference = torch.nn.functional.linear(model.model.norm(final.float()).float(), model.lm_head.weight.float())
    diff = (ours - reference).abs()
    bf16 = (ours - cap.logits).abs()
    gap, total = checks["logit_gap_pre"], checks["dla_total_pre"]
    pick = rng.choice(len(gap), size=min(100, len(gap)), replace=False)
    rel = np.abs(total[pick] - gap[pick]) / np.maximum(np.abs(gap[pick]), 1e-6)
    return {"max_abs_logit_diff": float(diff.max()), "logit_within_tol": bool(diff.max() <= LOGIT_TOL),
            "bf16_max_abs_logit_diff": float(bf16.max()),
            "bf16_top1_agreement": float((ours.argmax(-1) == cap.logits.argmax(-1)).float().mean()),
            "dla_max_rel_err": float(rel.max()), "dla_median_rel_err": float(np.median(rel))}


def write_trace(out: Path, trace_id: str, reg: Any, n_rows: int) -> None:
    safe = trace_id.replace(":", "_").replace("/", "_")
    for table in TABLES:
        columns = {"trace_id": pa.array([trace_id] * n_rows), "pos": pa.array(np.arange(n_rows), pa.int32())}
        for column, values in reg.table(table).items():
            columns[column] = pa.array(values[:n_rows], pa.float32())
        if len(columns) > 2:
            (out / "parts" / table).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(columns), out / "parts" / table / f"{safe}.parquet")


def save_arrays(out: Path, trace_id: str, cap: Any, hidden_layers: list[int], n_response: int) -> None:
    safe = trace_id.replace(":", "_").replace("/", "_")
    if cap.attn_stats is not None:
        (out / "attn").mkdir(parents=True, exist_ok=True)
        (out / "attn_recv").mkdir(parents=True, exist_ok=True)
        np.save(out / "attn" / f"{safe}.npy", cap.attn_stats.half().cpu().numpy())
        np.save(out / "attn_recv" / f"{safe}.npy", cap.received[:n_response].half().cpu().numpy())
    (out / "hidden").mkdir(parents=True, exist_ok=True)
    rows = torch.arange(n_response + 1, device=cap.resid.device) + 1
    np.save(out / "hidden" / f"{safe}.npy", cap.resid[hidden_layers][:, rows].half().cpu().numpy())


def trace_done(out: Path, trace_id: str) -> bool:
    """True when every per-trace artifact (all three tables, attention, received attention, hidden states) exists."""
    safe = trace_id.replace(":", "_").replace("/", "_")
    paths = [out / "parts" / table / f"{safe}.parquet" for table in TABLES]
    paths += [out / directory / f"{safe}.npy" for directory in ("attn", "attn_recv", "hidden")]
    return all(path.exists() for path in paths)


def merge_tables(out: Path) -> None:
    for table in TABLES:
        parts = sorted((out / "parts" / table).glob("*.parquet"))
        if parts:
            pq.write_table(pa.concat_tables([pq.read_table(part) for part in parts], promote_options="default"),
                           out / f"{table}.parquet", compression="zstd")


def extract(dataset: Path, out: Path, *, n_traces: int | None, layers_spec: str, mu_prompts: int,
            device: str, seed: int = 0, check_consistency: bool = True, strict: bool = True,
            model: Any = None, tokenizer: Any = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or load_meta(dataset)
    out.mkdir(parents=True, exist_ok=True)
    traces = list(iter_traces(dataset))
    if n_traces is not None:
        traces = traces[:n_traces]
    types = load_token_types(dataset)
    if model is None:
        model, tokenizer = load_model(meta, device)
    pinned = meta.get("attn_implementation", "sdpa")
    head = PredictionHead.from_model(model)
    n_layers, heads = len(model.model.layers), model.config.num_attention_heads
    hidden_layers = parse_layers(layers_spec, n_layers)
    estimate = disk_estimate(traces, n_layers, heads, head.d, len(hidden_layers))
    print("expected disk: " + ", ".join(f"{name} {value:.2f}" for name, value in estimate.items()), flush=True)
    run: dict[str, Any] = {"hidden_layers": hidden_layers, "disk_estimate": estimate,
                           "attn_stats": list(ATTN_STATS), "received_delays": list(RECEIVED_DELAYS)}
    check_resume(out, {"features_attention": WEIGHTS_IMPL, "pinned": pinned, "dtype": meta.get("dtype", "bfloat16"),
                       "model_revision": meta.get("model_revision"), "hidden_layers": hidden_layers})
    orthonormal = float((head.basis.T @ head.basis - torch.eye(head.d, device=head.basis.device)).abs().max())
    run["wu_basis_orthonormality_error"] = orthonormal
    if check_consistency:
        consistency = run["attention_consistency"] = attention_consistency(model, traces[:5], pinned)
        print(f"{WEIGHTS_IMPL} vs {pinned}: max |Δlogit| {consistency['max_abs_logit_diff']:.4f}, "
              f"top-1 agreement {consistency['top1_agreement']:.4%}, weights·V reconstruction error "
              f"{consistency['max_rel_reconstruction_err']:.2e}; (unused) eager vs {pinned}: max |Δlogit| "
              f"{consistency['eager_max_abs_logit_diff']:.4f}", flush=True)
    _enforce(run, out, strict, head_checks_done=False)
    mu_path = out / "mu_ref.pt"
    mu_layers = sorted(set(hidden_layers) | {n_layers})
    if mu_path.exists():
        mu_ref = {int(k): v.to(head.weight.device) for k, v in torch.load(mu_path).items()}
    else:
        with using(model, pinned):                  # no attention weights needed
            mu_ref = reference_means(model, tokenizer, head, mu_layers, meta["config"]["dataset"], mu_prompts, seed)
        torch.save({k: v.cpu() for k, v in mu_ref.items()}, mu_path)
    model.set_attn_implementation(WEIGHTS_IMPL)    # the model's sdpa computation, plus float32 weights for the hooks
    rng = np.random.default_rng(seed)
    manifest: dict[str, Any] = {}
    start = time.monotonic()

    def process(index: int, trace: dict[str, Any], *, write: bool) -> tuple[Any, Any, dict[str, np.ndarray]]:
        texts, trace_types, spans = trace_inputs(trace, tokenizer, types)
        cap = run_capture(model, head, trace["prompt_ids"], trace["response_ids"], spans)
        reg, checks = trace_features(cap, head, trace["response_ids"], texts, trace_types, mu_ref, hidden_layers)
        if write:
            write_trace(out, trace["trace_id"], reg, len(trace["response_ids"]))       # every response position
            save_arrays(out, trace["trace_id"], cap, hidden_layers, len(trace["response_ids"]))
            print(f"[{index + 1}/{len(traces)}] {trace['trace_id']} ({len(trace['response_ids'])} positions) "
                  f"{time.monotonic() - start:.0f}s", flush=True)
        return cap, reg, checks

    # Prediction-head checks always run on the first five traces, before the main loop, so a
    # resumed run enforces them too; traces not yet on disk are written in the same pass.
    run["head_checks"] = []
    for index, trace in enumerate(traces[:5]):
        cap, reg, checks = process(index, trace, write=not trace_done(out, trace["trace_id"]))
        run["head_checks"].append({"trace_id": trace["trace_id"], **head_checks(model, cap, head, checks, rng)})
        manifest = manifest or {item["column"]: item for item in reg.manifest()}
        del cap, reg
    _enforce(run, out, strict, head_checks_done=True)
    for index, trace in enumerate(traces):
        if trace_done(out, trace["trace_id"]):
            continue
        cap, reg, _ = process(index, trace, write=True)
        del cap, reg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    merge_tables(out)
    manifest.update(head_manifest(n_layers, heads))
    (out / "manifest.json").write_text(json.dumps(list(manifest.values()), indent=1))
    run["versions"] = {name: _version(name) for name in ("torch", "transformers", "numpy")}
    run["model"], run["model_revision"] = meta["config"]["model"], meta.get("model_revision")
    run["attn_implementation"], run["dtype"] = f"{pinned} (features: {WEIGHTS_IMPL})", meta.get("dtype", "bfloat16")
    run["n_traces"], run["seconds"] = len(traces), time.monotonic() - start
    (out / "extract_meta.json").write_text(json.dumps(run, indent=1, default=str))
    return run


def check_resume(out: Path, signature: dict[str, Any]) -> None:
    """Resume only into an output directory written with the same settings.

    Finished traces and mu_ref.pt are reused on resume, so files from another setup (for
    example an earlier eager-attention run, which predates the signature) must not mix in.
    """
    marker = out / "extract_signature.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous != signature:
            raise RuntimeError(f"{out} was written with {previous}, not {signature}; use a fresh --out")
        return
    leftovers = [name for name in ("parts", "attn", "attn_recv", "hidden", "mu_ref.pt") if (out / name).exists()]
    if leftovers:
        raise RuntimeError(f"{out} holds {leftovers} from an extraction without a signature (e.g. eager attention); use a fresh --out")
    marker.write_text(json.dumps(signature, indent=1))


def _enforce(run: dict[str, Any], out: Path, strict: bool, *, head_checks_done: bool) -> None:
    """Stop before (or, for the head checks, right after the first five traces of) extraction if a check fails."""
    failures = []
    if run["wu_basis_orthonormality_error"] > 1e-4:
        failures.append(f"W_U singular directions not orthonormal (max |VᵀV − I| = {run['wu_basis_orthonormality_error']:.2e})")
    consistency = run.get("attention_consistency")
    if consistency is not None and not consistency["passed"]:
        failures.append(f"{consistency.get('features_attention', WEIGHTS_IMPL)} vs {consistency['pinned']} logits: "
                        f"max |Δ| {consistency['max_abs_logit_diff']:.4f} (≤ {LOGIT_TOL}), "
                        f"top-1 agreement {consistency['top1_agreement']:.4%} (≥ {TOP1_TOL:.1%}), weights·V reconstruction "
                        f"error {consistency.get('max_rel_reconstruction_err', float('nan')):.2e} (≤ {RECONSTRUCTION_TOL})")
    if head_checks_done:
        for check in run["head_checks"]:
            if not check["logit_within_tol"]:
                failures.append(f"{check['trace_id']}: ĥ·w_y does not reproduce the float32 logits (max |Δ| {check['max_abs_logit_diff']:.4f})")
            if check["dla_max_rel_err"] > 0.01:
                failures.append(f"{check['trace_id']}: dla_total vs logit gap relative error {check['dla_max_rel_err']:.2e} > 1%")
    run["check_failures"] = failures
    if failures:
        (out / "extract_meta.json").write_text(json.dumps(run, indent=1, default=str))
        message = "extraction check failed:\n  " + "\n  ".join(failures)
        if strict:
            raise RuntimeError(message + "\n(results recorded in extract_meta.json; --no-strict-checks records and continues)")
        print("WARNING " + message, flush=True)


def head_manifest(n_layers: int, heads: int) -> dict[str, dict[str, Any]]:
    """Manifest entries for the per-head arrays in attn/ and attn_recv/ (tier 2)."""
    entries = {}
    for layer in range(1, n_layers + 1):
        for head in range(heads):
            for delay in ("pre", "post"):
                for stat in ATTN_STATS:
                    name = f"{stat}_L{layer}_H{head}"
                    entries[f"{name}__{delay}"] = {"name": name, "tier": 2, "delay": delay, "family": "attention_head",
                                                    "table": "attn", "layer": layer, "head": head, "column": f"{name}__{delay}"}
            for k in RECEIVED_DELAYS:
                name = f"att_recv_L{layer}_H{head}"
                entries[f"{name}__k{k}"] = {"name": name, "tier": 2, "delay": f"k{k}", "family": "attention_received_head",
                                            "table": "attn_recv", "layer": layer, "head": head, "column": f"{name}__k{k}"}
    return entries


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--traces", type=int, default=None, help="only the first N traces")
    parser.add_argument("--layers", default="every4", help="hidden-state / per-layer Φ layers: every4, all, or e.g. 4,28")
    parser.add_argument("--mu-ref-prompts", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument("--no-strict-checks", action="store_true", help="record failed checks and continue")
    args = parser.parse_args()
    from .preflight import check_labels
    print(f"label preflight: {check_labels(args.dataset)}", flush=True)
    extract(args.dataset, args.out, n_traces=args.traces, layers_spec=args.layers, mu_prompts=args.mu_ref_prompts,
            device=args.device, check_consistency=not args.skip_consistency, strict=not args.no_strict_checks)


if __name__ == "__main__":
    main()
