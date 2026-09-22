from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


PHASES = ("collection", "training", "evaluation", "retention")


@dataclass
class MindState:
    status: str = "PAUSED"
    phase: str = "collection"
    cursor: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    pause_reason: str | None = "reboot-safe default"


class MindSupervisor:
    def __init__(self, root: Path, *, reserve_bytes: int, data_cap_bytes: int,
                 notifier=None, clock=time.time):
        self.root = Path(root); self.reserve_bytes = reserve_bytes; self.data_cap_bytes = data_cap_bytes
        self.notifier = notifier or (lambda event: None); self.clock = clock
        self.state_path = self.root / "state.json"
        self.state = self._load()

    def _load(self) -> MindState:
        if not self.state_path.exists(): return MindState()
        value = json.loads(self.state_path.read_text())
        # A process start is always paused, even if the previous process died while running.
        return MindState(status="PAUSED", phase=value.get("phase", "collection"), cursor=value.get("cursor", {}),
                         failures=value.get("failures", []), pause_reason="reboot-safe default")

    def persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        value = {"schemaVersion": 1, **self.state.__dict__}
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        with temporary.open("rb") as source: os.fsync(source.fileno())
        temporary.replace(self.state_path)

    def start(self, *, human_enabled: bool) -> None:
        if not human_enabled: raise PermissionError("a human must explicitly enable each run")
        self.check_disk(); self.state.status = "RUNNING"; self.state.pause_reason = None; self.persist()

    def pause(self, reason: str, *, notify: bool = True) -> None:
        self.state.status = "PAUSED"; self.state.pause_reason = reason; self.persist()
        if notify: self.notifier({"kind": "pause", "reason": reason})

    def record_failure(self, kind: str) -> None:
        now = self.clock()
        self.state.failures = [item for item in self.state.failures if now - item["time"] <= 3600]
        self.state.failures.append({"time": now, "kind": kind})
        strikes = sum(item["kind"] in {"rejected-update", "worker-restart"} for item in self.state.failures)
        if kind in {"non-finite", "identity-drift", "replay-corruption", "legal-action-omission", "private-view-leakage"}:
            self.pause(kind)
        elif strikes >= 3:
            self.pause("three rejected updates or worker restarts within one hour")
        else:
            self.persist()

    def check_disk(self) -> None:
        usage = shutil.disk_usage(self.root.parent if self.root.parent.exists() else Path.cwd())
        artifact_bytes = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()) if self.root.exists() else 0
        if usage.free < self.reserve_bytes: raise RuntimeError("free-space reserve reached")
        if artifact_bytes >= self.data_cap_bytes: raise RuntimeError("learning data cap reached")


def cpu_worker_count(physical_cores: int | None) -> int:
    physical = physical_cores or (os.cpu_count() or 2)
    return max(1, min(physical - 2, 12))
