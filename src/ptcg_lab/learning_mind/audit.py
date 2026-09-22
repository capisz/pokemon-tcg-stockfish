from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .encoding import encode_decision
from .schema import UnsupportedPosition
from .tracker import ObservableHistoryTracker


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_replay(path: Path, expected: dict | None = None) -> dict:
    if expected and _hash_file(path) != expected["sha256"]:
        raise ValueError(f"replay artifact hash mismatch: {path}")
    decoded_hash = hashlib.sha256()
    chunks = []
    with gzip.open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            decoded_hash.update(chunk); chunks.append(chunk)
    if expected and decoded_hash.hexdigest() != expected["decodedSha256"]:
        raise ValueError(f"decoded replay hash mismatch: {path}")
    replay = json.loads(b"".join(chunks))
    trackers = {0: ObservableHistoryTracker(0), 1: ObservableHistoryTracker(1)}
    decisions = unsupported = 0
    identities = []
    for frame in replay.get("frames", []):
        actor = frame.get("actor")
        observations = frame.get("observations") or []
        if actor not in (0, 1) or len(observations) != 2:
            raise ValueError("replay frame has no actor-private view")
        observation = observations[actor]
        if observation.get("playerId") != actor:
            raise ValueError("actor observation mismatch")
        snapshot = trackers[actor].update(observation)
        try:
            encoded = encode_decision(observation, snapshot)
        except UnsupportedPosition:
            unsupported += 1
            continue
        action = frame.get("action")
        if action is not None:
            matches = [group for group in encoded.action_classes if any(item.get("id") == action.get("id") for item in group.actions)]
            if len(matches) != 1:
                raise ValueError("selected legal action is omitted or represented more than once")
        identities.append(encoded.identity); decisions += 1
    return {"replayId": replay.get("id"), "status": replay.get("status"), "decisions": decisions,
            "unsupportedPositions": unsupported, "decisionIdentityHash": hashlib.sha256("".join(identities).encode()).hexdigest()}


def audit_manifest(manifest_path: Path, *, limit: int | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("replays") or []
    if limit is not None: rows = rows[:limit]
    results = [audit_replay(Path(row["path"]), row) for row in rows]
    return {"schemaVersion": 1, "manifest": str(manifest_path), "scheduled": len(manifest.get("replays") or []),
            "audited": len(results), "decisions": sum(row["decisions"] for row in results),
            "unsupportedPositions": sum(row["unsupportedPositions"] for row in results),
            "replays": results,
            "representationParity": len(results) == len(manifest.get("replays") or [])
                                    and not any(row["unsupportedPositions"] for row in results)}
