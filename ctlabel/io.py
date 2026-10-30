from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A job killed mid-append leaves an unterminated final line; skip it
                # (its work is redone on resume). Corruption anywhere else is an error.
                if line.endswith("\n"):
                    raise


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    path = Path(path)
    return pq.read_table(path).to_pylist() if path.exists() else []


def write_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write schema-less empty parquet file: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    # from_pylist takes its columns from the first row only; rows with different
    # keys (e.g. values-only vs swap rows) would silently lose columns.
    names = list(dict.fromkeys(key for row in materialized for key in row))
    table = pa.Table.from_pylist([{name: row.get(name) for name in names} for row in materialized])
    for name in table.column_names:
        if name.endswith("token_ids") and pa.types.is_list(table[name].type):
            index = table.schema.get_field_index(name)
            table = table.set_column(index, name, table[name].cast(pa.list_(pa.int32())))
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)

