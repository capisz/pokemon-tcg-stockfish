from __future__ import annotations

import json
from pathlib import Path

from .features import action_features, card_tokens, resource_features
from .storage import Store, digest, terminal_score


def complete_games(store: Store) -> list[dict]:
    games = []
    for entry in store.list("replay-index"):
        if entry["status"] != "finished":
            continue
        game = store.get("replays", entry["id"])
        if "evaluationExperiment" in game:
            continue
        if game.get("status") != "finished" or not game.get("outcome"):
            continue
        terminal_score(game, 0)
        games.append(game)
    # Same trajectory with a new id must never enter multiple data splits.
    unique = {}
    for game in games:
        content_key = digest({key: value for key, value in game.items() if key not in {"id", "createdAt"}})
        unique[content_key] = game
    return [unique[key] for key in sorted(unique)]


def game_split(games: list[dict]) -> dict[str, list[dict]]:
    """Split game families, keeping reseeded duplicates and seat pairs together.

    No frame-level random split is allowed. All games sharing a simulation seed
    and deck pairing form one group, including paired first-player reversals.
    """
    groups: dict[str, list[dict]] = {}
    for game in games:
        key = digest({"seed": game["seed"], "decks": sorted(game["decks"])})
        groups.setdefault(key, []).append(game)
    # Fixed hash buckets never move an existing family when more games arrive.
    # Approximate 80/10/10 proportions; small corpora can have empty holdouts and
    # must collect additional games rather than quietly reshuffle held-out data.
    assignment = {"train": [], "calibration": [], "test": []}
    for key in sorted(groups):
        bucket = int(key[:8], 16) % 10
        split = "test" if bucket == 0 else "calibration" if bucket == 1 else "train"
        assignment[split].extend(groups[key])
    if any(not selected for selected in assignment.values()):
        raise ValueError("Stable whole-game hash split has an empty partition; collect more complete game families")
    return assignment


def examples(games: list[dict], limit: int = 10000) -> list[dict]:
    result = []
    for game in games:
        if game["status"] != "finished" or not game.get("outcome") or "evaluationExperiment" in game:
            continue
        winner = game["outcome"]["winner"]
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
            result.append({"gameId": game["id"], "decisionIndex": frame["decisionIndex"], "actor": actor,
                           "resources": resource_features(observation).tolist(), "cards": card_tokens(observation).tolist(),
                           "actions": [action_features(action).tolist() for action in legal], "selected": selected,
                           "expectedResult": terminal_score(game, actor),
                           "outcomeClass": 1 if winner is None else (2 if winner == actor else 0)})
            if len(result) >= limit:
                return result
    return result


def export_parquet(store: Store, destination: Path, limit: int = 100000) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet export needs pip install -e '.[training]'") from exc
    splits = game_split(complete_games(store))
    rows = []
    for split, games in splits.items():
        rows.extend({**row, "split": split} for row in examples(games, limit))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), destination, compression="zstd")
    manifest = {"schemaVersion": 1, "rows": len(rows), "games": {name: [game["id"] for game in games] for name, games in splits.items()},
                "splitPolicy": "immutable SHA256(seed, sorted decks) buckets: 0=test,1=calibration,2..9=train; complete non-evaluation games only"}
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
