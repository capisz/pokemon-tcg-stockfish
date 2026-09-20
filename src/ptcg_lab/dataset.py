from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from .features import action_features, card_tokens, resource_features
from .storage import Store, digest, terminal_score


def family_key(game: dict) -> str:
    return digest({"seed": game["seed"], "decks": sorted(game["decks"])})


def family_partition(game: dict) -> str:
    if game.get("evaluationExperiment") or game.get("benchmarkMatch") or "heldout" in game.get("deckRoles", []):
        return "benchmark"
    bucket = int(family_key(game)[:8], 16) % 10
    return "test" if bucket == 0 else "calibration" if bucket == 1 else "train"


def training_eligible(game: dict) -> bool:
    if "evaluationExperiment" in game or game.get("benchmarkMatch") or game.get("trainingEligible") is False:
        return False
    return not any(role in {"heldout", "historical"} for role in game.get("deckRoles", []))


def complete_games(store: Store, max_games: int = 2000, max_source_bytes: int = 256 * 1024**2) -> list[dict]:
    """Bounded recent replay buffer. Permanent source files/partitions stay intact."""
    unique, source_bytes = {}, 0
    if max_games <= 0 or max_source_bytes <= 0:
        return []
    for entry in store.iter_records("replay-index"):
        if entry["status"] != "finished":
            continue
        size = store.location("replays", entry["id"]).stat().st_size
        if source_bytes + size > max_source_bytes:
            continue
        game = store.get("replays", entry["id"])
        if not training_eligible(game) or game.get("status") != "finished" or not game.get("outcome"):
            continue
        terminal_score(game, 0)
        content_key = digest({key: value for key, value in game.items() if key not in {"id", "createdAt"}})
        if content_key not in unique:
            source_bytes += size
        unique[content_key] = game
        if len(unique) >= max_games:
            break
    return [unique[key] for key in sorted(unique)]


def game_split(games: list[dict]) -> dict[str, list[dict]]:
    """Immutable whole-family hash split, unchanged when the buffer grows."""
    assignment = {"train": [], "calibration": [], "test": []}
    for game in games:
        if not training_eligible(game):
            continue
        assignment[family_partition(game)].append(game)
    if any(not selected for selected in assignment.values()):
        raise ValueError("Stable whole-game hash split has an empty partition; collect more complete game families")
    return assignment


def iter_examples(games, limit: int = 10000):
    if limit <= 0:
        return
    emitted = 0
    for game in games:
        if game["status"] != "finished" or not game.get("outcome") or not training_eligible(game):
            continue
        winner = game["outcome"]["winner"]
        targets = {target["decisionIndex"]: target for target in game.get("searchTargets", [])}
        for frame in game["frames"]:
            if frame.get("action") is None:
                continue
            actor = frame["actor"]
            observation = frame["observations"][actor]
            if observation["playerId"] != actor:
                raise ValueError("Training observation belongs to the wrong player")
            legal = observation.get("legalActions", [])
            selected = next((index for index, action in enumerate(legal) if action["id"] == frame["action"]["id"]), None)
            if selected is None or not legal:
                continue
            row = {"gameId": game["id"], "decisionIndex": frame["decisionIndex"], "actor": actor,
                   "resources": resource_features(observation).tolist(), "cards": card_tokens(observation).tolist(),
                   "actions": [action_features(action).tolist() for action in legal], "selected": selected,
                   "expectedResult": terminal_score(game, actor),
                   "outcomeClass": 1 if winner is None else (2 if winner == actor else 0),
                   "policyDistribution": None, "policyTargetSource": "behavior-cloning"}
            target = targets.get(frame["decisionIndex"])
            if target and target.get("observationHash") == digest(observation) and target.get("actor") == actor:
                probabilities = target.get("probabilities", {})
                weights = [float(probabilities.get(action["id"], 0)) for action in legal]
                if set(probabilities) == {action["id"] for action in legal} and all(math.isfinite(value) and value >= 0 for value in weights) and abs(sum(weights) - 1) < 1e-5:
                    row["policyDistribution"] = weights
                    row["policyTargetSource"] = "search-distillation"
            yield row
            emitted += 1
            if emitted >= limit:
                return


def examples(games: list[dict], limit: int = 10000) -> list[dict]:
    return list(iter_examples(games, limit))


def export_parquet(store: Store, destination: Path, limit: int = 100000) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet export needs pip install -e '.[training]'") from exc
    from .resources import check_storage, volume_free, ResourceLimit
    if limit <= 0:
        raise ValueError("Parquet position limit must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_storage(store.path, store.max_bytes, store.min_free_bytes)
    if volume_free(destination) < store.min_free_bytes:
        raise ResourceLimit("Parquet destination free-space reserve reached")
    if destination.exists() or destination.with_suffix(".manifest.json").exists():
        raise ValueError("Parquet exports are immutable; choose a new destination")
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.partial")
    writer, count, selected = None, 0, {"train": [], "calibration": [], "test": []}
    # Fixed schema permits a chunk containing no search targets before a chunk
    # that does, without Arrow's inferred null type changing the file schema.
    schema = pa.schema([("gameId", pa.string()), ("decisionIndex", pa.int64()), ("actor", pa.int64()),
                        ("resources", pa.list_(pa.float64())), ("cards", pa.list_(pa.int64())),
                        ("actions", pa.list_(pa.list_(pa.float64()))), ("selected", pa.int64()),
                        ("expectedResult", pa.float64()), ("outcomeClass", pa.int64()),
                        ("policyDistribution", pa.list_(pa.float64())), ("policyTargetSource", pa.string()), ("split", pa.string())])
    try:
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        for entry in store.iter_records("replay-index"):
            if entry["status"] != "finished":
                continue
            game = store.get("replays", entry["id"])
            if not training_eligible(game):
                continue
            split = family_partition(game)
            selected[split].append(game["id"])
            buffer = []
            for row in iter_examples([game], max(1, limit - count)):
                buffer.append({**row, "split": split})
                count += 1
                if len(buffer) == 256:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
                    check_storage(store.path, store.max_bytes, store.min_free_bytes)
                    if volume_free(destination) < store.min_free_bytes:
                        raise ResourceLimit("Parquet destination free-space reserve reached")
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            if count >= limit:
                break
        writer.close()
        writer = None
        check_storage(store.path, store.max_bytes, store.min_free_bytes)
        if volume_free(destination) < store.min_free_bytes:
            raise ResourceLimit("Parquet destination free-space reserve reached")
        manifest = {"schemaVersion": 2, "rows": count, "games": selected,
                    "splitPolicy": "stable-family-hash-v1; completed eligible non-evaluation games only"}
        # A manifest is the commit marker; data without this marker is incomplete.
        temporary.rename(destination)
        marker = destination.with_suffix(".manifest.json")
        marker_tmp = marker.with_suffix(".tmp")
        marker_tmp.write_text(json.dumps(manifest, indent=2))
        marker_tmp.replace(marker)
        return manifest
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
