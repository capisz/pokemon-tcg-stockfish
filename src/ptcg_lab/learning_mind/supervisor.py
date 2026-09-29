from __future__ import annotations

import json
import fcntl
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


PHASES = ("collection", "training", "evaluation", "retention")
FAILURE_KINDS = frozenset({"non-finite", "identity-drift", "replay-corruption",
    "legal-action-omission", "private-view-leakage", "rejected-update", "worker-restart"})


class VerifiedContinuousOperationRecord:
    """Capability issued after rechecking promotion, specialist, and safety evidence."""
    __slots__ = ("_values",)

    def __init__(self, values: dict, *, _verification_token: object):
        if _verification_token is not _VERIFIED_CONTINUOUS_TOKEN:
            raise TypeError("continuous-operation records must come from the evidence verifier")
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, _name, _value):
        raise AttributeError("verified continuous-operation evidence is immutable")


_VERIFIED_CONTINUOUS_TOKEN = object()


def continuous_operation_enablement(stage_record: VerifiedContinuousOperationRecord | None) -> dict:
    """Keep the always-on supervisor closed until PPO and specialization are accepted."""
    if not isinstance(stage_record, VerifiedContinuousOperationRecord):
        return {"enabled": False,
                "reason": "continuous operation requires a verified evidence capability, not editable gate booleans"}
    stage_record = stage_record._values
    prerequisites = (stage_record.get("ppoEnabled") is True
        and stage_record.get("specialistCurriculumPassed") is True
        and stage_record.get("continuousOperationEnabled") is True
        and stage_record.get("humanEnableContinuousOperation") is True)
    return {"enabled": bool(prerequisites),
            "reason": None if prerequisites else "PPO, specialization, continuous-operation gate, and human approval are all required"}


@dataclass
class MindState:
    status: str = "PAUSED"
    phase: str = "collection"
    cursor: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    pause_reason: str | None = "reboot-safe default"


class MindSupervisor:
    def __init__(self, root: Path, *, reserve_bytes: int, data_cap_bytes: int,
                 artifact_roots: list[Path] | None = None, notifier=None, clock=time.time):
        if (type(reserve_bytes) is not int or reserve_bytes < 0
                or type(data_cap_bytes) is not int or data_cap_bytes < 1):
            raise ValueError("supervisor disk reserve and data cap must be nonnegative/positive byte counts")
        self.root = Path(root).resolve()
        supplied_roots = artifact_roots or []
        if not isinstance(supplied_roots, list) or any(not isinstance(path, Path) for path in supplied_roots):
            raise ValueError("supervisor artifact roots must be a list of paths")
        self.artifact_roots = tuple(dict.fromkeys((self.root, *(path.resolve() for path in supplied_roots))))
        self.reserve_bytes = reserve_bytes; self.data_cap_bytes = data_cap_bytes
        self.notifier = notifier or (lambda event: None); self.clock = clock
        self.state_path = self.root / "state.json"
        self.configuration_path = self.root / "configuration.json"
        self.lock_path = self.root / ".supervisor.lock"
        self._lock_fd: int | None = None
        self._freeze_configuration()
        self.state = self._load()

    def _freeze_configuration(self) -> None:
        expected = {"schemaVersion": 1, "reserveBytes": self.reserve_bytes,
            "dataCapBytes": self.data_cap_bytes,
            "artifactRoots": [str(path) for path in self.artifact_roots]}
        if self.configuration_path.is_file():
            configured = json.loads(self.configuration_path.read_text())
            if configured != expected:
                raise ValueError("supervisor configuration identity drift; use a new state root")
            return
        if self.state_path.exists() or (self.root.exists() and any(self.root.iterdir())):
            raise ValueError("supervisor state root has data but no frozen configuration")
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                prefix=".configuration.", suffix=".tmp", delete=False) as temporary:
            temporary.write(json.dumps(expected, sort_keys=True, indent=2) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            try:
                os.link(temporary_path, self.configuration_path)
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
        if json.loads(self.configuration_path.read_text()) != expected:
            raise ValueError("supervisor configuration changed during initialization")

    def _load(self) -> MindState:
        if not self.state_path.exists(): return MindState()
        value = json.loads(self.state_path.read_text())
        expected_keys = {"schemaVersion", "status", "phase", "cursor", "failures", "pause_reason"}
        if (not isinstance(value, dict) or set(value) != expected_keys
                or value.get("schemaVersion") != 1
                or not isinstance(value.get("status"), str)
                or value.get("status") not in {"PAUSED", "RUNNING"}
                or value.get("phase") not in PHASES
                or not isinstance(value.get("cursor"), dict)
                or not isinstance(value.get("failures"), list)
                or (value.get("pause_reason") is not None
                    and not isinstance(value.get("pause_reason"), str))):
            raise ValueError("supervisor state is malformed; refusing to resume")
        try:
            json.dumps(value["cursor"], allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("supervisor cursor is not finite JSON; refusing to resume") from error
        for failure in value["failures"]:
            if (not isinstance(failure, dict) or set(failure) != {"time", "kind"}
                    or type(failure.get("time")) not in (int, float)
                    or not isinstance(failure.get("time"), (int, float))
                    or not math.isfinite(failure["time"])
                    or not isinstance(failure.get("kind"), str)
                    or failure.get("kind") not in FAILURE_KINDS):
                raise ValueError("supervisor failure history is malformed; refusing to resume")
        # A process start is always paused, even if the previous process died while running.
        return MindState(status="PAUSED", phase=value.get("phase", "collection"), cursor=value.get("cursor", {}),
                         failures=value.get("failures", []), pause_reason="reboot-safe default")

    def persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        value = {"schemaVersion": 1, **self.state.__dict__}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                prefix=".state.", suffix=".tmp", delete=False) as temporary:
            temporary.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(self.state_path)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)

    def start(self, *, human_enabled: bool, stage_record: dict | None = None) -> None:
        if self.state.status != "PAUSED":
            raise PermissionError("supervisor is already running")
        if not human_enabled: raise PermissionError("a human must explicitly enable each run")
        if not continuous_operation_enablement(stage_record)["enabled"]:
            raise PermissionError("continuous operation is not enabled by the accepted stage record")
        self._acquire_process_lock()
        try:
            self._check_capacity_or_pause(status="RUNNING", clear_pause_reason=True)
            self.state.status = "RUNNING"; self.state.pause_reason = None; self.persist()
        except Exception:
            self.close()
            raise

    def _acquire_process_lock(self) -> None:
        if self._lock_fd is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise PermissionError("another supervisor process owns this state root") from error
        except Exception:
            os.close(descriptor)
            raise
        self._lock_fd = descriptor

    def close(self) -> None:
        """Release the process lock when this supervisor is permanently exiting."""
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    def record_progress(self, cursor: dict) -> None:
        if self.state.status != "RUNNING":
            raise PermissionError("progress can be checkpointed only while the supervisor is running")
        if not isinstance(cursor, dict):
            raise ValueError("phase cursor must be a JSON object")
        try:
            frozen = json.loads(json.dumps(cursor, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("phase cursor must contain finite JSON values only") from error
        self._check_capacity_or_pause(cursor=frozen)
        self.state.cursor = frozen
        self.persist()

    def advance_phase(self, next_phase: str, *, next_cursor: dict | None = None) -> None:
        if self.state.status != "RUNNING":
            raise PermissionError("phase transitions require a running supervisor")
        if self.state.phase not in PHASES or next_phase not in PHASES:
            raise ValueError("unknown learning phase")
        expected = PHASES[(PHASES.index(self.state.phase) + 1) % len(PHASES)]
        if next_phase != expected:
            raise ValueError(f"phase transition must advance from {self.state.phase} to {expected}")
        if next_cursor is not None and not isinstance(next_cursor, dict):
            raise ValueError("next phase cursor must be a JSON object")
        try:
            frozen = json.loads(json.dumps(next_cursor or {}, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("next phase cursor must contain finite JSON values only") from error
        self._check_capacity_or_pause(phase=next_phase, cursor=frozen)
        self.state.phase = next_phase
        self.state.cursor = frozen
        self.persist()

    def pause(self, reason: str, *, notify: bool = True) -> None:
        self.state.status = "PAUSED"; self.state.pause_reason = reason; self.persist()
        if notify: self.notifier({"kind": "pause", "reason": reason})

    def record_failure(self, kind: str) -> None:
        if not isinstance(kind, str) or kind not in FAILURE_KINDS:
            raise ValueError("unknown supervisor failure kind")
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("supervisor clock must return a finite timestamp")
        self.state.failures = [item for item in self.state.failures if now - item["time"] <= 3600]
        self.state.failures.append({"time": now, "kind": kind})
        strikes = sum(item["kind"] in {"rejected-update", "worker-restart"} for item in self.state.failures)
        if kind in {"non-finite", "identity-drift", "replay-corruption", "legal-action-omission", "private-view-leakage"}:
            self.pause(kind)
        elif strikes >= 3:
            self.pause("three rejected updates or worker restarts within one hour")
        else:
            self.persist()

    def check_disk(self, *, projected_state_bytes: int | None = None) -> None:
        for directory in self.artifact_roots:
            probe = directory
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            try:
                usage = shutil.disk_usage(probe)
            except OSError as error:
                raise RuntimeError("unable to verify free-space reserve for an artifact root") from error
            if usage.free < self.reserve_bytes:
                raise RuntimeError("free-space reserve reached")
        files = {path.resolve(): path for directory in self.artifact_roots if directory.exists()
                 for path in directory.rglob("*") if path.is_file()}
        artifact_bytes = sum(path.stat().st_size for path in files.values())
        if projected_state_bytes is not None:
            current_state_bytes = self.state_path.stat().st_size if self.state_path.is_file() else 0
            artifact_bytes = artifact_bytes - current_state_bytes + projected_state_bytes
        if artifact_bytes >= self.data_cap_bytes: raise RuntimeError("learning data cap reached")

    def _check_capacity_or_pause(self, *, phase: str | None = None, cursor: dict | None = None,
                                 status: str | None = None,
                                 clear_pause_reason: bool = False) -> None:
        try:
            projected = {"schemaVersion": 1, **self.state.__dict__}
            if phase is not None:
                projected["phase"] = phase
            if cursor is not None:
                projected["cursor"] = cursor
            if status is not None:
                projected["status"] = status
            if clear_pause_reason:
                projected["pause_reason"] = None
            serialized = json.dumps(projected, sort_keys=True, indent=2, allow_nan=False) + "\n"
            self.check_disk(projected_state_bytes=len(serialized.encode("utf-8")))
        except RuntimeError as error:
            self.pause(str(error))
            raise


def cpu_worker_count(physical_cores: int | None) -> int:
    physical = physical_cores or (os.cpu_count() or 2)
    return max(1, min(physical - 2, 12))
