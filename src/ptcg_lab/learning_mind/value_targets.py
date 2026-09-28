from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ptcg_lab.storage import Store, terminal_score, digest as legacy_digest

from .dataset_v1 import file_sha256, stable_split
from .encoding import encode_decision
from .schema import identity_hash
from .tracker import ObservableHistoryTracker


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


def build_value_target_dataset(*, output: Path, experimental_root: Path,
                               source_manifest_path: Path, identity: dict) -> dict:
    """Freeze actor-view terminal outcomes as value-only targets from eligible games."""
    output = output.resolve()
    if output.exists():
        raise ValueError("value-target datasets are immutable; choose a new output directory")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_hash = source_manifest.get("manifestHash")
    if source_hash != identity_hash({key: value for key, value in source_manifest.items()
                                     if key != "manifestHash"}):
        raise ValueError("value-target source manifest checksum mismatch")
    settings = source_manifest.get("settings")
    if not isinstance(settings, dict) or settings.get("identity") != identity:
        raise ValueError("value-target source runtime identity mismatch")
    replay_items = source_manifest.get("replays")
    if not isinstance(replay_items, list):
        raise ValueError("value-target source manifest has no replay list")
    replay_ids = [item.get("id") if isinstance(item, dict) else None for item in replay_items]
    if (any(not isinstance(value, str) or not value for value in replay_ids)
            or len(set(replay_ids)) != len(replay_ids)):
        raise ValueError("value-target source replay IDs are invalid or duplicated")

    store = Store(experimental_root)
    rows, sources, excluded = [], [], []
    game_splits: dict[str, str] = {}
    seen_decisions: set[tuple[str, int]] = set()
    for item in sorted(replay_items, key=lambda record: record["id"]):
        replay_id = item["id"]
        replay_path = store.location("replays", replay_id)
        replay = store.get("replays", replay_id)
        if not isinstance(replay, dict) or replay.get("id") != replay_id:
            raise ValueError(f"value-target replay identity mismatch: {replay_id}")
        if replay.get("status") != item.get("status"):
            raise ValueError(f"value-target replay status differs from source manifest: {replay_id}")
        if replay.get("status") != "finished":
            excluded.append({"replayId": replay_id, "reason": "unfinished-game"})
            continue
        if (replay.get("dataTier") != "experimental" or replay.get("experimentalLearning") is not True
                or replay.get("trainingEligible") is not True):
            excluded.append({"replayId": replay_id, "reason": "not-explicitly-training-eligible"})
            continue
        seat_zero_score = terminal_score(replay, 0)
        family = item.get("familyId") or identity_hash({"replay": replay_id})
        if not isinstance(family, str) or not family:
            raise ValueError(f"value-target replay family is invalid: {replay_id}")
        split = stable_split(family)
        prior_split = game_splits.setdefault(replay_id, split)
        if prior_split != split:
            raise ValueError("value-target source game crosses splits")
        outcome = replay.get("outcome")
        if not isinstance(outcome, dict):
            raise ValueError(f"finished value-target replay lacks a terminal outcome: {replay_id}")
        winner = outcome.get("winner")
        if winner is not None and (type(winner) is not int or winner not in (0, 1)):
            raise ValueError(f"finished value-target replay has an invalid winner: {replay_id}")

        trackers = {0: ObservableHistoryTracker(0), 1: ObservableHistoryTracker(1)}
        game_rows = 0
        last_index = -1
        for frame in replay.get("frames", []):
            if not isinstance(frame, dict):
                raise ValueError(f"value-target replay contains an invalid frame: {replay_id}")
            if frame.get("action") is None:
                continue
            actor = frame.get("actor")
            decision_index = frame.get("decisionIndex")
            views = frame.get("observations")
            if (type(actor) is not int or actor not in (0, 1)
                    or type(decision_index) is not int or decision_index != last_index + 1
                    or not isinstance(views, list) or len(views) != 2):
                raise ValueError(f"value-target decision frame identity is invalid: {replay_id}")
            last_index = decision_index
            # Deliberately access only the acting seat's redacted view.
            observation = views[actor]
            if (not isinstance(observation, dict) or observation.get("playerId") != actor
                    or observation.get("decisionPlayer", actor) != actor):
                raise ValueError(f"value-target frame is not the actor-visible observation: {replay_id}")
            snapshot = trackers[actor].update(observation)
            encoded = encode_decision(observation, snapshot)
            decision_key = (replay_id, decision_index)
            if decision_key in seen_decisions:
                raise ValueError("value-target source repeats a game decision index")
            seen_decisions.add(decision_key)
            actor_score = seat_zero_score if actor == 0 else 1.0 - seat_zero_score
            target = 2.0 * actor_score - 1.0
            rows.append({"positionHash": legacy_digest(observation), "sourceGameId": replay_id,
                "sourceDecisionIndex": decision_index, "actor": actor, "familyId": family,
                "split": split, "featureIdentityHash": encoded.identity,
                "valueLabelSource": "completed-self-play-outcome", "valueTarget": target,
                "observation": observation, "tracker": snapshot})
            game_rows += 1
        if game_rows == 0:
            raise ValueError(f"eligible finished value-target replay has no actor decisions: {replay_id}")
        sources.append({"replayId": replay_id, "familyId": family, "split": split,
                        "status": "finished", "sha256": file_sha256(replay_path),
                        "decisionRows": game_rows})

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        rows_path = temporary_root / "rows.jsonl"
        _atomic_text(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                          for row in rows))
        manifest = {"schemaVersion": 1, "kind": "learning-mind-value-targets-v1",
            "identity": identity, "sourceManifestHash": source_hash,
            "sourceManifestSha256": file_sha256(source_manifest_path),
            "builderSha256": file_sha256(Path(__file__).resolve()),
            "rows": len(rows), "rowsSha256": file_sha256(rows_path),
            "games": sources, "excludedGames": excluded,
            "status": "eligible" if rows else "no-eligible-completed-games",
            "splitCounts": {split: sum(row["split"] == split for row in rows)
                            for split in ("train", "development", "heldout")},
            "outcomeTargets": {str(target): sum(row["valueTarget"] == target for row in rows)
                               for target in (-1.0, 0.0, 1.0)},
            "policyLabels": 0, "targetContract": "terminal-win-draw-loss-minus-one-zero-plus-one"}
        manifest["manifestHash"] = identity_hash(manifest)
        _atomic_text(temporary_root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        temporary_root.replace(output)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return manifest


def load_value_target_dataset(path: Path, *, identity: dict | None = None) -> tuple[dict, list[dict]]:
    path = path.resolve()
    manifest_path = path / "manifest.json"
    rows_path = path / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("kind") != "learning-mind-value-targets-v1"
            or manifest.get("manifestHash") != identity_hash({key: value for key, value in manifest.items()
                                                               if key != "manifestHash"})
            or manifest.get("builderSha256") != file_sha256(Path(__file__).resolve())
            or manifest.get("rowsSha256") != file_sha256(rows_path)
            or manifest.get("targetContract") != "terminal-win-draw-loss-minus-one-zero-plus-one"
            or manifest.get("policyLabels") != 0
            or (identity is not None and manifest.get("identity") != identity)):
        raise ValueError("value-target dataset manifest, rows, or runtime identity mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if len(rows) != manifest.get("rows"):
        raise ValueError("value-target dataset row count mismatch")
    seen = set()
    game_splits: dict[str, str] = {}
    game_families: dict[str, str] = {}
    game_row_counts: dict[str, int] = {}
    for row in rows:
        required = {"positionHash", "sourceGameId", "sourceDecisionIndex", "actor", "familyId",
                    "split", "featureIdentityHash", "valueLabelSource", "valueTarget",
                    "observation", "tracker"}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("value-target row schema mismatch")
        if (not isinstance(row["sourceGameId"], str) or not row["sourceGameId"]
                or type(row["sourceDecisionIndex"]) is not int or row["sourceDecisionIndex"] < 0):
            raise ValueError("value-target source decision identity is invalid")
        key = (row["sourceGameId"], row["sourceDecisionIndex"])
        if (key in seen or not isinstance(row["familyId"], str) or not row["familyId"]
                or not isinstance(row["split"], str) or row["split"] not in {"train", "development", "heldout"}
                or type(row["actor"]) is not int or row["actor"] not in (0, 1)
                or row["valueLabelSource"] != "completed-self-play-outcome"
                or type(row["valueTarget"]) not in {int, float} or row["valueTarget"] not in {-1, 0, 1}
                or row["split"] != stable_split(row["familyId"])
                or not isinstance(row["positionHash"], str) or not row["positionHash"]
                or not isinstance(row["featureIdentityHash"], str) or not row["featureIdentityHash"]
                or not isinstance(row["observation"], dict)
                or not isinstance(row["tracker"], dict)
                or row["observation"].get("playerId") != row["actor"]
                or row["observation"].get("decisionPlayer", row["actor"]) != row["actor"]
                or legacy_digest(row["observation"]) != row["positionHash"]):
            raise ValueError("value-target row actor, target, or decision identity is invalid")
        encoded = encode_decision(row["observation"], row["tracker"])
        if encoded.identity != row["featureIdentityHash"]:
            raise ValueError("value-target row feature identity mismatch")
        prior = game_splits.setdefault(row["sourceGameId"], row["split"])
        if prior != row["split"]:
            raise ValueError("value-target source game crosses splits")
        prior_family = game_families.setdefault(row["sourceGameId"], row["familyId"])
        if prior_family != row["familyId"]:
            raise ValueError("value-target source game crosses families")
        game_row_counts[row["sourceGameId"]] = game_row_counts.get(row["sourceGameId"], 0) + 1
        seen.add(key)
    game_records = manifest.get("games")
    if not isinstance(game_records, list):
        raise ValueError("value-target game manifest is malformed")
    recorded_games = {}
    for game in game_records:
        if (not isinstance(game, dict) or set(game) !=
                {"replayId", "familyId", "split", "status", "sha256", "decisionRows"}
                or not isinstance(game.get("replayId"), str) or not game["replayId"]
                or game["status"] != "finished"
                or not isinstance(game.get("sha256"), str) or len(game["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in game["sha256"])
                or type(game.get("decisionRows")) is not int or game["decisionRows"] < 1
                or game["replayId"] in recorded_games):
            raise ValueError("value-target game manifest record is invalid")
        recorded_games[game["replayId"]] = game
    if set(recorded_games) != set(game_row_counts):
        raise ValueError("value-target games do not exactly cover the value rows")
    for replay_id, count in game_row_counts.items():
        game = recorded_games[replay_id]
        if (game["familyId"] != game_families[replay_id]
                or game["split"] != game_splits[replay_id]
                or game["decisionRows"] != count):
            raise ValueError("value-target game summary differs from its rows")
    expected_splits = {split: sum(row["split"] == split for row in rows)
                       for split in ("train", "development", "heldout")}
    expected_outcomes = {str(target): sum(row["valueTarget"] == target for row in rows)
                         for target in (-1.0, 0.0, 1.0)}
    if (manifest.get("splitCounts") != expected_splits
            or manifest.get("outcomeTargets") != expected_outcomes
            or manifest.get("status") != ("eligible" if rows else "no-eligible-completed-games")):
        raise ValueError("value-target dataset summary counts do not match its rows")
    return manifest, rows


def training_records(rows: list[dict], split: str = "train") -> list[dict]:
    if split not in {"train", "development", "heldout"}:
        raise ValueError("unknown value-target split")
    records = []
    for row in rows:
        if row["split"] != split:
            continue
        encoded = encode_decision(row["observation"], row["tracker"])
        if encoded.identity != row["featureIdentityHash"]:
            raise ValueError("value-target encoded feature identity drift")
        records.append({"encoded": encoded, "positionHash": row["positionHash"],
                        "valueLabelSource": row["valueLabelSource"], "valueTarget": row["valueTarget"]})
    return records
