"""Optional (test 7): replicate the paper's model-level Φ_W for Qwen2.5-7B base.

    python -m ctfeat.phi_replication --dataset dataset/ --out features/ [--m 2000]

Final-layer, last-token states of M FineWeb sequences truncated to 512 tokens; the
paper reports Φ_W × 10³ = 0.109 with M = 15,000. Logged only; it never blocks a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from .data import load_meta
from .extract import load_model
from .head import PredictionHead

PAPER_PHI_X1000 = 0.109
SHARD = ("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet")


def phi_model_level(states: torch.Tensor, head: PredictionHead) -> float:
    """Paper Eq. 4: tr(W_c Σ W_cᵀ) / (tr(Σ) ‖W_c‖²_F) for the covariance Σ of ``states``."""
    u = head.phi_coords(states.float()).double()
    var = u.var(0, unbiased=False)
    return float((head.phi_strengths.double() * var).sum() / (var.sum() * head.phi_frobenius))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--m", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    result = {"paper_phi_x1000": PAPER_PHI_X1000, "paper_m": 15000, "m": args.m}
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(SHARD[0], SHARD[1], repo_type="dataset")
    except Exception as error:                      # FineWeb not downloadable here: log and stop
        result["error"] = f"FineWeb download failed: {error}"
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "phi_replication.json").write_text(json.dumps(result, indent=1))
        print(result["error"])
        return
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()[: args.m]
    meta = load_meta(args.dataset)
    model, tokenizer = load_model(meta, "cuda", attn=meta.get("attn_implementation", "sdpa"))
    head = PredictionHead.from_model(model)
    raw, normed = [], []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[: args.max_tokens]
        if not ids:
            continue
        hidden = model.model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0, -1]   # after the final norm
        normed.append(hidden.float())
    states = torch.stack(normed)
    value = phi_model_level(states, head)
    result.update({"phi_x1000_final_norm_states": value * 1000, "gap_x1000": value * 1000 - PAPER_PHI_X1000,
                   "n_sequences": len(states), "shard": "/".join(SHARD),
                   "note": "states are the final-layer last-token outputs after the model's final RMSNorm (the lm_head input)"})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phi_replication.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
