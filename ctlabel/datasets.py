from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import dataset_base
from .io import read_jsonl


def bundled_dataset_path(dataset: str) -> Path:
    name = dataset_base(dataset) + ("_test" if dataset.lower().endswith("_test") else "")
    return Path(__file__).parent / "vendor" / "input" / f"{name}.jsonl"


def load_problems(dataset: str, path: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path or bundled_dataset_path(dataset)))
    return rows if limit is None else rows[:limit]

