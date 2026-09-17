from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def terminal_score(replay: dict, player_id: int) -> float:
    """Only known rules terminals produce learning/evaluation labels."""
    if replay.get("status") != "finished" or not replay.get("outcome"):
        raise ValueError("Only a completed rules outcome has a score")
    outcome = replay["outcome"]
    if outcome.get("winner") is None:
        if outcome.get("reason") != "rules-draw":
            raise ValueError("Unknown terminal winner is not a genuine rules draw")
        return .5
    if type(outcome["winner"]) is not int or outcome["winner"] not in (0, 1) or not outcome.get("reason"):
        raise ValueError("Invalid terminal outcome")
    return float(outcome["winner"] == player_id)


class Store:
    def __init__(self, path: Path, max_bytes: int = 2 * 1024**3):
        self.path, self.max_bytes = path, max_bytes
        self.lock = threading.Lock()

    def location(self, category: str, identifier: str) -> Path:
        for item in (category, identifier):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", item):
                raise ValueError("Invalid storage identifier")
        return self.path / category / f"{identifier}.json"

    def put(self, category: str, identifier: str, value: Any) -> None:
        target = self.location(category, identifier)
        content = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        with self.lock:
            current = sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file()) if self.path.exists() else 0
            existing = target.stat().st_size if target.exists() else 0
            if current - existing + len(content) > self.max_bytes:
                raise RuntimeError("Local data disk cap reached; archive data before continuing")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, category: str, identifier: str) -> Any:
        return json.loads(self.location(category, identifier).read_text())

    def list(self, category: str) -> list[dict]:
        self.location(category, "validation")
        directory = self.path / category
        if not directory.exists():
            return []
        records = []
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            records.append(json.loads(path.read_text()))
        return records

    def save_replay(self, replay: dict) -> str:
        if replay.get("status") not in {"finished", "truncated", "error"}:
            raise ValueError("Replay must explicitly distinguish a terminal result from truncation")
        if replay["status"] != "finished" and replay.get("outcome") is not None:
            raise ValueError("An unfinished replay cannot carry a game outcome")
        if replay["status"] == "finished":
            terminal_score(replay, 0)
        identifier = replay.get("id") or uuid.uuid4().hex
        original = self.location("replays", identifier)
        if original.exists():
            previous = json.loads(original.read_text())
            if previous != replay:
                # A deterministic engine trajectory id can recur with different
                # build/evaluation provenance. Never overwrite an earlier record.
                replay.setdefault("sourceEngineId", identifier)
                identifier = f"{identifier[:110]}-{digest({key: value for key, value in replay.items() if key != 'id'})[:24]}"
        replay["id"] = identifier
        self.put("replays", identifier, replay)
        self.put("replay-index", identifier, {"id": identifier, "decks": replay.get("decks", []),
                                             "status": replay["status"], "frames": len(replay.get("frames", []))})
        return identifier
