"""Label-side preflight (Section 2), run before any expensive work.

It reads labels only to validate the dataset and returns nothing that reaches a
feature; the extraction code itself never imports it (only the CLI does).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def check_labels(dataset: Path) -> dict[str, Any]:
    """Labels come from exhaustive values, and every wrong trace has a row with V_before and
    V_with at each labellable position 0 .. n_labellable_tokens - 1."""
    meta = json.loads((dataset / "meta.json").read_text())
    mode = meta.get("config", {}).get("mode")
    if mode != "exhaustive":
        raise ValueError(f"labels in {dataset} are not from exhaustive values (mode={mode!r})")
    expected: dict[str, int] = {}
    with (dataset / "traces.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not row["correct"]:
                    expected[row["trace_id"]] = int(row.get("n_labellable_tokens", len(row["response_ids"])))
    table = pq.read_table(dataset / "positions.parquet", columns=["trace_id", "pos", "v_before", "v_with"]).to_pydict()
    complete: dict[str, set[int]] = defaultdict(set)
    for trace_id, pos, before, with_ in zip(table["trace_id"], table["pos"], table["v_before"], table["v_with"]):
        if trace_id in expected and before is not None and with_ is not None:
            complete[trace_id].add(int(pos))
    problems = {trace_id: n - len(complete[trace_id] & set(range(n))) for trace_id, n in expected.items()
                if complete[trace_id] != set(range(n))}
    if problems:
        trace_id, missing = next(iter(problems.items()))
        raise ValueError(f"{len(problems)} wrong traces lack a position row with V_before / V_with "
                         f"(e.g. trace {trace_id}: {missing} of {expected[trace_id]} positions)")
    return {"mode": mode, "wrong_traces": len(expected), "wrong_positions": sum(expected.values())}
