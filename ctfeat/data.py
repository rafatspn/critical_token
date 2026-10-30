"""Dataset access for feature extraction. Labels are never read here: from
positions.parquet only ``token_type`` is loaded, and negatives.parquet is never opened."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq


def load_meta(dataset: Path) -> dict[str, Any]:
    meta = json.loads((dataset / "meta.json").read_text())
    if meta.get("config", {}).get("mode") != "exhaustive":
        raise ValueError(f"labels in {dataset} are not from exhaustive values (mode={meta.get('config', {}).get('mode')!r})")
    return meta


def iter_traces(dataset: Path) -> Iterator[dict[str, Any]]:
    with (dataset / "traces.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                trace = json.loads(line)
                trace.pop("vllm_top_logprobs", None)
                yield trace


def load_token_types(dataset: Path) -> dict[str, list[str]]:
    """token_type per response position, in pos order, for every labelled position."""
    table = pq.read_table(dataset / "positions.parquet", columns=["trace_id", "pos", "token_type"]).to_pylist()
    by_trace: dict[str, dict[int, str]] = defaultdict(dict)
    for row in table:
        by_trace[row["trace_id"]][int(row["pos"])] = row["token_type"]
    return {trace_id: [types[pos] for pos in range(len(types))] for trace_id, types in by_trace.items()}
