"""Small, resumable current-engine games for actor-visible position coverage.

These are research artifacts only. The collector never creates policy labels,
trains a model, reads the opposite observation for a decision, or writes to the
canonical competitive replay store.
"""
from __future__ import annotations

import hashlib
from itertools import combinations_with_replacement
import json
import os
from pathlib import Path

from ptcg_lab.engine import EngineClient
from ptcg_lab.selfplay import Agent
from ptcg_lab.storage import Store

from .dataset_v1 import file_sha256
from .experiment import runtime_identity
from .schema import identity_hash
from .curriculum import ARCHETYPES

POLICY_NAMES = {"typescript-heuristic", "python-heuristic"}
DEFAULT_MATCHUPS = tuple(combinations_with_replacement(ARCHETYPES, 2))
COLLECTOR_VERSION = "fresh-actor-position-collector-v4-five-archetype"


def game_schedule(*, games_per_matchup: int = 2, policy: str = "typescript-heuristic",
                  collection_namespace: str = "main") -> list[dict]:
    if type(games_per_matchup) is not int or not 1 <= games_per_matchup <= 20:
        raise ValueError("games per matchup must be an integer from 1 to 20")
    if policy not in POLICY_NAMES:
        raise ValueError("unsupported frozen collection policy")
    if (not isinstance(collection_namespace, str) or not collection_namespace
            or len(collection_namespace) > 32
            or any(not (char.isascii() and (char.isalnum() or char in "-_")) for char in collection_namespace)):
        raise ValueError("collection namespace must be a 1-32 character ASCII slug")
    result = []
    used_seeds = set()
    for deck_a, deck_b in DEFAULT_MATCHUPS:
        cell_id = f"{deck_a}-vs-{deck_b}"
        for game_index in range(games_per_matchup):
            epoch = "" if collection_namespace == "main" else f"|{collection_namespace}"
            token = f"learning-mind-v1-fresh|{COLLECTOR_VERSION}|{policy}{epoch}|{cell_id}|{game_index}".encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
            while seed in used_seeds:
                seed = (seed + 1) % 2**32
            used_seeds.add(seed)
            decks = [deck_a, deck_b] if game_index % 2 == 0 or deck_a == deck_b else [deck_b, deck_a]
            # Cross-matchups cycle every four games to balance both deck seat
            # and which archetype starts. Mirrors alternate the starting seat.
            first_player = game_index % 2 if deck_a == deck_b else (game_index // 2) % 2
            epoch_label = "" if collection_namespace == "main" else f"-{collection_namespace}"
            result.append({"index": len(result), "cellId": cell_id, "gameIndex": game_index,
                           "seed": seed, "firstPlayer": first_player,
                           "decks": decks, "policy": policy,
                           "collectionNamespace": collection_namespace,
                           "replayId": f"fresh-v2-{policy.removesuffix('-heuristic')}-{cell_id}{epoch_label}-{game_index}"})
    return result


def actor_only_replay(replay: dict, *, schedule: dict, run_id: str,
                      engine_identity: dict) -> dict:
    """Retain each decision-maker's redacted view, never the other private view."""
    cleaned = dict(replay)
    frames = []
    for frame in replay.get("frames", []):
        actor = frame.get("actor")
        observations = frame.get("observations")
        if type(actor) is not int or actor not in (0, 1) or not isinstance(observations, list) or len(observations) != 2:
            raise ValueError("engine replay frame has an invalid actor/view pair")
        view = observations[actor]
        if not isinstance(view, dict) or view.get("playerId") != actor or view.get("decisionPlayer") != actor:
            raise ValueError("engine replay actor view does not match the decision-maker")
        frames.append({**frame, "observations": [view if seat == actor else None for seat in (0, 1)]})
    cleaned.update({"id": schedule["replayId"], "frames": frames,
                    "dataTier": "experimental", "experimentalLearning": True,
                    "trainingEligible": False, "learningRun": run_id,
                    "learningPurpose": "fresh-actor-position-coverage",
                    "seed": schedule["seed"], "decks": list(schedule["decks"]),
                    "policies": [schedule["policy"], schedule["policy"]],
                    "firstPlayer": schedule["firstPlayer"],
                    "engineVersion": engine_identity["engineVersion"],
                    "engineBuildHash": engine_identity["engineBuildHash"],
                    "chance": []})
    return cleaned


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)


def _play_python_heuristic(engine: EngineClient, schedule: dict, max_decisions: int) -> dict:
    engine.request("reset", {"seed": schedule["seed"], "decks": schedule["decks"],
                              "firstPlayer": schedule["firstPlayer"]})
    agents = [Agent("heuristic", schedule["seed"] + seat) for seat in (0, 1)]
    for decision_index in range(max_decisions):
        views = [engine.request("observe", {"playerId": seat}) for seat in (0, 1)]
        if views[0].get("status") != "running":
            break
        actor = views[0].get("decisionPlayer")
        if type(actor) is not int or actor not in (0, 1):
            raise ValueError("current engine did not provide a valid decision actor")
        # Counter-based choice randomness makes retries deterministic without
        # coupling policy randomness to the engine's chance stream.
        agents[actor].rng.seed(f"{schedule['seed']}:{decision_index}:{actor}")
        action_id = agents[actor].choose(views[actor])
        engine.request("step", {"actionId": action_id})
    return engine.request("replay")


def collect_fresh_positions(*, root: Path, output: Path, games_per_matchup: int = 2,
                            policy: str = "typescript-heuristic", max_decisions: int = 1200,
                            collection_namespace: str = "main") -> dict:
    if type(max_decisions) is not int or not 1 <= max_decisions <= 5000:
        raise ValueError("max decisions must be an integer from 1 to 5000")
    root, output = root.resolve(), output.resolve()
    identity = runtime_identity(root).record()
    with EngineClient(root, timeout=300) as engine:
        engine_identity = engine.request("health")
        if engine_identity.get("engineBuildHash") != identity.get("engine_build_hash"):
            raise ValueError("runtime identity and active engine bundle disagree")
        schedule = game_schedule(games_per_matchup=games_per_matchup, policy=policy,
                                 collection_namespace=collection_namespace)
        settings = {"collectorVersion": COLLECTOR_VERSION, "identity": identity,
                    "engineIdentity": engine_identity, "policy": policy,
                    "collectionNamespace": collection_namespace,
                    "gamesPerMatchup": games_per_matchup, "maxDecisions": max_decisions,
                    "schedule": schedule, "collectorCodeHash": file_sha256(Path(__file__))}
        run_id = identity_hash(settings)[:32]
        manifest_path = output / "run-manifest.json"
        store = Store(output / "store")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("manifestHash") != identity_hash({key: value for key, value in manifest.items()
                                                               if key != "manifestHash"}):
                raise ValueError("fresh-position collection manifest checksum mismatch")
            if manifest.get("settings") != settings or manifest.get("runId") != run_id:
                raise ValueError("fresh-position collection identity/configuration drift")
            completed = {record["scheduleIndex"]: record for record in manifest.get("games", [])}
            expected_files = {record["replayId"] for record in completed.values()}
            actual_files = {path.stem for path in (output / "store" / "replays").glob("*.json")}
            if actual_files != expected_files:
                raise ValueError("fresh-position replay files do not exactly match the checkpoint")
            for record in completed.values():
                if file_sha256(store.location("replays", record["replayId"])) != record["sha256"]:
                    raise ValueError(f"fresh-position replay checksum mismatch: {record['replayId']}")
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("fresh-position output is nonempty without a collection manifest")
            output.mkdir(parents=True, exist_ok=True)
            manifest = {"schemaVersion": 1, "runId": run_id, "settings": settings,
                        "games": [], "replays": []}
            completed = {}
            manifest["manifestHash"] = identity_hash(manifest)
            _atomic_json(manifest_path, manifest)

        for item in schedule:
            if item["index"] in completed:
                continue
            expected_path = store.location("replays", item["replayId"])
            if expected_path.exists():
                raise ValueError("uncheckpointed replay exists; preserve it for manual reconciliation")
            try:
                if policy == "typescript-heuristic":
                    replay = engine.request("run", {"seed": item["seed"], "decks": item["decks"],
                        "firstPlayer": item["firstPlayer"], "policy": "heuristic",
                        "maxDecisions": max_decisions})
                else:
                    replay = _play_python_heuristic(engine, item, max_decisions)
            except Exception as error:
                replay = {"schemaVersion": 1, "status": "error", "outcome": None,
                          "warnings": [f"collector error: {type(error).__name__}: {str(error)[:500]}"],
                          "frames": [], "seed": item["seed"], "decks": item["decks"],
                          "engineVersion": engine_identity["engineVersion"]}
            replay = actor_only_replay(replay, schedule=item, run_id=run_id,
                                       engine_identity=engine_identity)
            replay_id = store.save_replay(replay)
            if replay_id != item["replayId"]:
                raise ValueError("store changed a deterministic fresh replay ID")
            path = store.location("replays", replay_id)
            actor_positions = sum(1 for frame in replay["frames"]
                if isinstance(frame.get("observations"), list)
                and isinstance(frame["observations"][frame["actor"]], dict)
                and frame["observations"][frame["actor"]].get("searchPosition"))
            game_record = {"scheduleIndex": item["index"], "replayId": replay_id,
                "cellId": item["cellId"], "seed": item["seed"], "firstPlayer": item["firstPlayer"],
                "decks": item["decks"], "policy": policy, "status": replay["status"],
                "decisions": max(0, len(replay["frames"]) - 1),
                "actorSearchPositions": actor_positions,
                "sha256": file_sha256(path), "bytes": path.stat().st_size}
            completed[item["index"]] = game_record
            manifest["games"] = [completed[index] for index in sorted(completed)]
            manifest["replays"] = [{"id": row["replayId"], "familyId": row["cellId"],
                "decks": row["decks"], "engineVersion": engine_identity["engineVersion"],
                "policies": [policy, policy], "status": row["status"]}
                for row in manifest["games"] if row["status"] == "finished"]
            manifest["completedGames"] = len(completed)
            manifest["scheduledGames"] = len(schedule)
            manifest["statusCounts"] = {status: sum(row["status"] == status for row in completed.values())
                                         for status in ("finished", "truncated", "error")}
            manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                                       if key != "manifestHash"})
            _atomic_json(manifest_path, manifest)
    return manifest
