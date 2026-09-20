from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    data: Path
    workers: int = 2
    max_jobs: int = 8
    max_decisions: int = 3000
    max_disk_bytes: int = 25 * 1024**3
    engine_timeout: float = 300.0
    analysis_model: Path | None = None
    max_memory_bytes: int = 8 * 1024**3
    min_free_bytes: int = 20 * 1024**3
    profile: str = "mac"

    def __post_init__(self):
        if self.profile not in {"mac", "windows"} or not 1 <= self.workers <= 8:
            raise ValueError("Use mac/windows profile and one to eight workers")
        if self.max_disk_bytes <= 0 or self.max_memory_bytes <= 0 or self.min_free_bytes < 0:
            raise ValueError("Resource limits must be positive; free-space reserve cannot be negative")

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.environ.get("PTCG_LAB_ROOT", Path(__file__).resolve().parents[2])).resolve()
        selected = os.environ.get("PTCG_ANALYSIS_MODEL")
        profile = os.environ.get("PTCG_PROFILE", "windows" if platform.system() == "Windows" else "mac")
        if profile not in {"mac", "windows"}:
            raise ValueError("PTCG_PROFILE must be mac or windows")
        return cls(root=root, data=Path(os.environ.get("PTCG_LAB_DATA", root / "data")).resolve(),
                   analysis_model=Path(selected).resolve() if selected else None, profile=profile,
                   max_disk_bytes=int(os.environ.get("PTCG_MAX_DATA_GIB", 200 if profile == "windows" else 25)) * 1024**3,
                   max_memory_bytes=int(os.environ.get("PTCG_MAX_MEMORY_GIB", 40 if profile == "windows" else 8)) * 1024**3,
                   workers=int(os.environ.get("PTCG_WORKERS", "2")))

    @property
    def worker_tuning_limit(self) -> int:
        return max(1, min(8, (os.cpu_count() or 2) - 2)) if self.profile == "windows" else 2

    @property
    def worker(self) -> Path:
        return self.root / "packages" / "engine" / "dist" / "worker.cjs"
