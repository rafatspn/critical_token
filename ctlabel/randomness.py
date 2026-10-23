from __future__ import annotations

import hashlib


def request_seed(global_seed: int, trace_id: str, position: int, variant: str) -> int:
    raw = f"{global_seed}\0{trace_id}\0{position}\0{variant}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") % (2**31 - 1)

