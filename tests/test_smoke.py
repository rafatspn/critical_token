import os
import subprocess
import sys
from pathlib import Path

import pytest
import pyarrow.parquet as pq


def gpu_stack_available():
    if os.environ.get("CTLABEL_SKIP_GPU_SMOKE") == "1":
        return False
    try:
        import torch
        import vllm  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_stack_available(), reason="requires CUDA, torch, and vLLM")
def test_gpu_smoke():
    subprocess.run([sys.executable, "-m", "ctlabel.run", "--config", "configs/smoke.yaml", "--stage", "all", "--yes"], check=True)
    root = Path("outputs/smoke")
    expected = [
        "traces_all.jsonl", "traces.jsonl", "token_stats.parquet", "values.parquet", "swap.parquet",
        "dataset/positions.parquet", "dataset/traces.jsonl", "dataset/negatives.parquet", "dataset/meta.json", "dataset/summary.md",
    ]
    assert all((root / name).exists() for name in expected)
    for name in ("traces_all.jsonl", "traces.jsonl", "dataset/traces.jsonl"):
        assert [__import__("json").loads(line) for line in (root / name).read_text().splitlines() if line]
    for name in ("token_stats.parquet", "values.parquet", "swap.parquet", "dataset/positions.parquet", "dataset/negatives.parquet"):
        pq.read_table(root / name)
    __import__("json").loads((root / "dataset/meta.json").read_text())
    assert (root / "dataset/summary.md").read_text().strip()

