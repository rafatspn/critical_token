from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

DELAYS = ("pre", "post", "k4", "k16", "k64")
DELAY_ORDER = {delay: index for index, delay in enumerate(DELAYS)}


@dataclass(frozen=True)
class FeatureSpec:
    name: str              # base name, including any _L{layer} suffix
    tier: int              # 1 output, 2 attention, 3 hidden states
    delay: str             # one of DELAYS
    family: str            # cost / reporting family
    table: str             # tier1, tier2, tier3_derived
    layer: int | None = None
    head: int | None = None

    @property
    def column(self) -> str:
        return f"{self.name}__{self.delay}"


@dataclass
class Registry:
    """Feature specs plus per-trace values, filled by the feature-family functions."""

    specs: dict[str, FeatureSpec] = field(default_factory=dict)
    values: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, name: str, values: Any, *, tier: int, delay: str, family: str, table: str,
            layer: int | None = None, head: int | None = None) -> None:
        if delay not in DELAY_ORDER:
            raise ValueError(f"unknown delay {delay!r}")
        spec = FeatureSpec(name, tier, delay, family, table, layer, head)
        if spec.column in self.specs:
            raise ValueError(f"duplicate feature {spec.column}")
        self.specs[spec.column] = spec
        self.values[spec.column] = np.asarray(values, dtype=np.float32)

    def table(self, table: str) -> dict[str, np.ndarray]:
        return {column: self.values[column] for column, spec in self.specs.items() if spec.table == table}

    def manifest(self) -> list[dict[str, Any]]:
        return [dict(asdict(spec), column=spec.column) for spec in self.specs.values()]


def allowed(column_delay: str, limit: str) -> bool:
    """True if a feature with ``column_delay`` may be used by a detector scored at ``limit``."""
    return DELAY_ORDER[column_delay] <= DELAY_ORDER[limit]


def delay_of(column: str) -> str:
    return column.rsplit("__", 1)[1]
