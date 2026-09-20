"""Durable cursor streams and explicit player projections for browser presentation.

Private stream lines contain research observations. Only projection functions may
cross the HTTP boundary. The owning job/match record commits the visible cursor.
"""
from __future__ import annotations

import copy
import json
import os
import threading


def player_observation(observation: dict) -> dict:
    result = copy.deepcopy(observation)
    # Belief reconstruction belongs to research workers, not the presentation API.
    result.pop("searchPosition", None)
    return result


def player_action(action: dict | None, actor: int, player_id: int) -> dict | None:
    if action is None:
        return None
    if actor != player_id and action.get("type") in {"prompt", "choice"}:
        return {"type": action["type"], "label": "Opponent choice · private details hidden"}
    result = copy.deepcopy(action)
    if actor != player_id:
        # Hand positions and prompt bindings are not public card identities.
        for key in ("sourceRef", "targetRef"):
            if result.get(key, {}).get("zone") in {"hand", "prompt"}:
                result.pop(key, None)
    return result


def project_frame(frame: dict, player_id: int) -> dict:
    observations = frame.get("observations")
    observation = observations[player_id] if observations is not None else frame["observation"]
    if observation["playerId"] != player_id:
        raise ValueError("Frame has no observation for this perspective")
    result = {key: frame[key] for key in ("cursor", "decisionIndex", "actor", "revision", "gameNumber") if key in frame}
    result["observation"] = player_observation(observation)
    result["priorAction"] = player_action(frame.get("priorAction"), frame.get("priorActor", frame["actor"]), player_id)
    return result


def project_replay(replay: dict, player_id: int) -> dict:
    if player_id not in (0, 1):
        raise ValueError("Invalid player perspective")
    # Whitelist metadata: private artifacts may gain new research fields later.
    result = {key: copy.deepcopy(replay[key]) for key in
              ("id", "engineVersion", "status", "outcome", "warnings", "policyContext") if key in replay}
    result["decks"] = [deck if index == player_id else "opponent" for index, deck in enumerate(replay.get("decks", []))]
    result.update(schemaVersion=2, playerId=player_id, visibility="player-projected", frames=[])
    prior = None
    for index, frame in enumerate(replay.get("frames", [])):
        item = project_frame({**frame, "cursor": index, "priorAction": prior.get("action") if prior else None,
                              "priorActor": prior["actor"] if prior else frame["actor"]}, player_id)
        item["action"] = player_action(frame.get("action"), frame["actor"], player_id)
        result["frames"].append(item)
        prior = frame
    return result


class FrameStream:
    """Append/fsync without rewriting the full game at every decision.

    Uncommitted crash tails are ignored by readers and removed on the next append.
    Offsets are cached, so polling a long game reads only the requested window.
    """
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.offsets: dict[str, tuple[int, list[int]]] = {}

    def _path(self, identifier):
        return self.store.location("private-streams", identifier).with_suffix(".jsonl")

    def _index(self, identifier):
        path = self._path(identifier)
        size = path.stat().st_size if path.exists() else 0
        indexed, offsets = self.offsets.get(identifier, (0, []))
        if indexed > size:
            indexed, offsets = 0, []
        offsets = offsets.copy()
        if indexed < size:
            with path.open("rb") as stream:
                stream.seek(indexed)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    offsets.append(start)
                    indexed = stream.tell()
        self.offsets[identifier] = (indexed, offsets)
        return indexed, offsets

    def append(self, identifier: str, frame: dict, committed: int) -> int:
        cursor = committed + 1
        content = json.dumps({**frame, "cursor": cursor}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        with self.lock, self.store.lock:
            from .resources import check_storage
            check_storage(self.store.path, self.store.max_bytes, self.store.min_free_bytes, additional=len(content))
            path = self._path(identifier)
            path.parent.mkdir(parents=True, exist_ok=True)
            indexed, offsets = self._index(identifier)
            if cursor > len(offsets):
                raise ValueError("Durable frame stream is missing an acknowledged decision")
            end = offsets[cursor] if cursor < len(offsets) else indexed
            with path.open("r+b" if path.exists() else "w+b") as output:
                output.truncate(end)
                output.seek(end)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            self.offsets[identifier] = (end + len(content), offsets[:cursor] + [end])
        return cursor

    def read(self, identifier: str, *, after: int, limit: int, committed: int, player_id: int, status: str) -> dict:
        if after < -1 or not 1 <= limit <= 100:
            raise ValueError("Use after >= -1 and a frame limit from 1 to 100")
        with self.lock:
            _, offsets = self._index(identifier)
            end = min(committed + 1, after + 1 + limit)
            if committed >= len(offsets):
                raise ValueError("Durable frame stream is missing an acknowledged decision")
            frames = []
            if after + 1 < end:
                with self._path(identifier).open("rb") as source:
                    source.seek(offsets[after + 1])
                    for _ in range(after + 1, end):
                        frames.append(project_frame(json.loads(source.readline()), player_id))
            next_cursor = frames[-1]["cursor"] if frames else after
            return {"schemaVersion": 1, "frames": frames, "nextCursor": next_cursor,
                    "hasMore": next_cursor < committed, "status": status}
