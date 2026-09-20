from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import threading
import uuid
import zlib
from pathlib import Path
from typing import Any


MAX_REPLAY_FRAME_BYTES = 128 * 1024**2
_FRAME_ENVELOPE = "compressedFrames"


def _encode_record(category: str, value: Any) -> Any:
    if category != "replays" or not isinstance(value, dict):
        return value
    if _FRAME_ENVELOPE in value:
        raise ValueError("Replay frame compression is reserved for the storage layer")
    if value.get("dataTier") != "experimental" or "frames" not in value:
        return value
    if not isinstance(value["frames"], list):
        raise ValueError("Experimental replay frames must be a list")
    # Serialize incrementally so oversized input fails before building a second
    # unbounded JSON string. The source replay itself remains unchanged.
    decoded = bytearray()
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value["frames"]):
        data = chunk.encode()
        if len(decoded) + len(data) > MAX_REPLAY_FRAME_BYTES:
            raise ValueError("Experimental replay frames exceed the 128 MiB decoded bound")
        decoded.extend(data)
    encoded = gzip.compress(decoded, compresslevel=3, mtime=0)
    result = {key: item for key, item in value.items() if key != "frames"}
    result[_FRAME_ENVELOPE] = {"schemaVersion": 1, "codec": "gzip+base64", "decodedBytes": len(decoded),
                              "frameCount": len(value["frames"]), "sha256": hashlib.sha256(decoded).hexdigest(),
                              "data": base64.b64encode(encoded).decode("ascii")}
    return result


def _decode_record(category: str, value: Any) -> Any:
    if category != "replays" or not isinstance(value, dict) or _FRAME_ENVELOPE not in value:
        return value
    envelope = value[_FRAME_ENVELOPE]
    if (value.get("dataTier") != "experimental" or "frames" in value or not isinstance(envelope, dict)
            or envelope.get("schemaVersion") != 1 or envelope.get("codec") != "gzip+base64"
            or type(envelope.get("decodedBytes")) is not int
            or not 0 <= envelope["decodedBytes"] <= MAX_REPLAY_FRAME_BYTES
            or type(envelope.get("frameCount")) is not int or envelope["frameCount"] < 0
            or not isinstance(envelope.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", envelope["sha256"])
            or not isinstance(envelope.get("data"), str)
            or len(envelope["data"]) > 4 * ((MAX_REPLAY_FRAME_BYTES + 1024**2 + 2) // 3)):
        raise ValueError("Invalid or unsupported experimental replay frame encoding")
    try:
        compressed = base64.b64decode(envelope["data"], validate=True)
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        # The hard output limit defeats small compressed payloads with huge
        # expansion. Reject tails/concatenated gzip members instead of ignoring.
        decoded = decoder.decompress(compressed, envelope["decodedBytes"] + 1)
    except (ValueError, zlib.error) as exc:
        raise ValueError("Invalid experimental replay frame compression") from exc
    if (len(decoded) != envelope["decodedBytes"] or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data
            or hashlib.sha256(decoded).hexdigest() != envelope["sha256"]):
        raise ValueError("Experimental replay frame checksum, length, or stream boundary differs")
    frames = json.loads(decoded)
    if not isinstance(frames, list) or len(frames) != envelope["frameCount"]:
        raise ValueError("Decoded experimental replay frames do not match the envelope")
    return {key: item for key, item in value.items() if key != _FRAME_ENVELOPE} | {"frames": frames}


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
    def __init__(self, path: Path, max_bytes: int = 25 * 1024**3, min_free_bytes: int = 0):
        self.path, self.max_bytes = path, max_bytes
        self.min_free_bytes = min_free_bytes
        self.lock = threading.Lock()

    def location(self, category: str, identifier: str) -> Path:
        for item in (category, identifier):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", item):
                raise ValueError("Invalid storage identifier")
        return self.path / category / f"{identifier}.json"

    def put(self, category: str, identifier: str, value: Any) -> None:
        target = self.location(category, identifier)
        content = json.dumps(_encode_record(category, value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        with self.lock:
            from .resources import directory_bytes
            current = directory_bytes(self.path)
            existing = target.stat().st_size if target.exists() else 0
            if current - existing + len(content) > self.max_bytes:
                raise RuntimeError("Local data disk cap reached; archive data before continuing")
            if self.min_free_bytes:
                from .resources import volume_free, ResourceLimit
                if volume_free(target) - len(content) < self.min_free_bytes:
                    raise ResourceLimit("Destination free-space reserve reached")
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

    def emergency_control_pause(self, identifier: str, record: dict) -> dict:
        """Persist only an existing runner's pause when ordinary writes hit a limit.

        This does not bypass storage gates for artifacts or computation. The
        temporary atomic replacement needs real disk space; a physically full or
        disconnected volume still fails without replacing the acknowledged file.
        """
        if (not isinstance(record, dict) or record.get("id") != identifier
                or record.get("status") != "paused" or record.get("desired") != "paused"):
            raise ValueError("Emergency control writes may only pause an existing learning run")
        target = self.location("learning-runs", identifier)
        with self.lock:
            if self.path.is_symlink() or target.parent.is_symlink() or target.is_symlink():
                raise ValueError("Emergency control record cannot use symlink storage")
            if not target.is_file():
                raise ValueError("Emergency control cannot create a learning run")
            size = target.stat().st_size
            if size > 2 * 1024**2:
                raise ValueError("Emergency control record exceeds the 2 MiB safety bound")
            current = json.loads(target.read_bytes())
            if (not isinstance(current, dict) or current.get("id") != identifier
                    or type(current.get("revision")) is not int):
                raise ValueError("Emergency control requires an intact acknowledged learning run")
            if current.get("desired") == "stopped" or current.get("status") == "stopped":
                raise ValueError("Emergency pause cannot replace an acknowledged stop")

            def bounded_text(value, limit):
                if value is None:
                    return None
                text = str(value)[:limit]
                while len(json.dumps(text, ensure_ascii=False).encode()) > limit:
                    text = text[:int(len(text) * .8)]
                return text

            paused = {**current, "status": "paused", "desired": "paused", "revision": current["revision"] + 1}
            for key, limit in (("pauseReason", 500), ("error", 1000), ("updatedAt", 80)):
                if key in record:
                    paused[key] = bounded_text(record[key], limit)
            content = json.dumps(paused, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            if len(content) > size + 2048 or len(content) > 2 * 1024**2:
                raise ValueError("Emergency pause exceeds its 2 KiB safety growth allowance")
            from .resources import volume_free, ResourceLimit
            if volume_free(target) < len(content) + 4096:
                raise ResourceLimit("Insufficient physical space to save the emergency pause safely")
            temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(target)
                if os.name != "nt":
                    directory = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
            return paused

    def get(self, category: str, identifier: str) -> Any:
        return _decode_record(category, json.loads(self.location(category, identifier).read_text()))

    def list(self, category: str) -> list[dict]:
        return list(self.iter_records(category))

    def iter_records(self, category: str):
        """Stream records; callers can bound resident replay memory."""
        self.location(category, "validation")
        directory = self.path / category
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            yield self.get(category, path.stem)

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
            previous = self.get("replays", identifier)
            if previous != replay:
                # A deterministic engine trajectory id can recur with different
                # build/evaluation provenance. Never overwrite an earlier record.
                replay.setdefault("sourceEngineId", identifier)
                identifier = f"{identifier[:110]}-{digest({key: value for key, value in replay.items() if key != 'id'})[:24]}"
        replay["id"] = identifier
        from .dataset import family_key, training_eligible
        key, eligible = family_key(replay), training_eligible(replay)
        if not eligible:
            self.put("partitions", f"excluded-{key}", {"familyId": key, "excludedFromTraining": True})
        self.put("replays", identifier, replay)
        self.put("replay-index", identifier, {"id": identifier, "decks": replay.get("decks", []),
                                             "status": replay["status"], "frames": len(replay.get("frames", [])),
                                             "familyKey": key, "trainingEligible": eligible})
        return identifier
