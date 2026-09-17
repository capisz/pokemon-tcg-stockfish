from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    data: Path
    workers: int = 2
    max_jobs: int = 8
    max_decisions: int = 3000
    max_disk_bytes: int = 2 * 1024**3
    engine_timeout: float = 300.0
    analysis_model: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.environ.get("PTCG_LAB_ROOT", Path(__file__).resolve().parents[2])).resolve()
        selected = os.environ.get("PTCG_ANALYSIS_MODEL")
        return cls(root=root, data=Path(os.environ.get("PTCG_LAB_DATA", root / "data")).resolve(),
                   analysis_model=Path(selected).resolve() if selected else None)

    @property
    def worker(self) -> Path:
        return self.root / "packages" / "engine" / "dist" / "worker.cjs"
