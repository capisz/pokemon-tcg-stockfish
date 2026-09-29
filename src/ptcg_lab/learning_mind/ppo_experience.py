from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterator

from .dataset_v1 import file_sha256
from .curriculum import assignment, promotion_seed_namespace_disjoint
from .schema import identity_hash


EXPERIENCE_STORE_VERSION = "ppo-actor-experience-v2"
PPO_SCHEDULER_VERSION = "ppo-training-scheduler-v1"
GAME_STATUSES = frozenset({"finished", "truncated", "error"})
FORBIDDEN_VIEW_KEYS = frozenset({"observations", "oppositeObservation", "otherObservation",
    "opponentPrivateObservation", "chance", "hiddenState", "engineStore", "rawReplay",
    "replay", "frames", "opponentHand", "opponentDeck", "opponentCards"})


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def ppo_training_schedule(*, game_count: int, historical_policy_hashes: list[str],
                          seed_base: int = 7543298) -> list[dict]:
    """Freeze the 5-archetype 80/20 policy mix, 5% mirrors, seats, and seeds."""
    if type(game_count) is not int or game_count < 1:
        raise ValueError("PPO schedule game_count must be positive")
    if (not isinstance(historical_policy_hashes, list) or not historical_policy_hashes
            or any(not _is_sha256(value) for value in historical_policy_hashes)
            or len(set(historical_policy_hashes)) != len(historical_policy_hashes)):
        raise ValueError("PPO schedule requires unique frozen historical-policy hashes")
    if type(seed_base) is not int or not 0 <= seed_base < 2**32:
        raise ValueError("PPO training seed base must be a uint32")
    result, used_seeds = [], set()
    for index in range(game_count):
        record = _ppo_schedule_row(index, historical_policy_hashes, seed_base)
        if record["seed"] in used_seeds:
            raise ValueError("deterministic PPO scheduler produced a seed collision")
        used_seeds.add(record["seed"])
        result.append(record)
    return result


def _ppo_schedule_row(index: int, historical_policy_hashes: list[str], seed_base: int) -> dict:
    assignment_record = assignment(index, historical_policy_hashes)
    seed = promotion_seed_namespace_disjoint(seed_base, "training", index)
    first_player = (index // 2) % 2
    if assignment_record["policyFamily"] == "historical":
        learner_seat = index % 2
        seats = [learner_seat]
        decks = ([assignment_record["ownArchetype"], assignment_record["opponentArchetype"]]
                 if learner_seat == 0 else
                 [assignment_record["opponentArchetype"], assignment_record["ownArchetype"]])
    else:
        seats = [0, 1]
        decks = [assignment_record["ownArchetype"], assignment_record["opponentArchetype"]]
    record = {"scheduleIndex": index, "seed": seed, "firstPlayer": first_player,
        "decks": decks, "learnerSeats": seats, **assignment_record}
    record["gameId"] = identity_hash({"schedulerVersion": PPO_SCHEDULER_VERSION, **record})
    return record


def _has_forbidden_view_key(value: object) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_VIEW_KEYS.intersection(value)) or any(
            _has_forbidden_view_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_forbidden_view_key(item) for item in value)
    return False


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_decision(decision: dict, *, behavior_policy_hash: str,
                       feature_schema_hash: str) -> None:
    if not isinstance(decision, dict):
        raise ValueError("PPO experience decisions must be JSON objects")
    if _has_forbidden_view_key(decision):
        raise ValueError("PPO experience must contain actor-visible decisions, not private replay views")
    actor = decision.get("actor")
    observation = decision.get("observation")
    tracker = decision.get("tracker")
    if (type(actor) is not int or actor not in (0, 1) or not isinstance(observation, dict)
            or observation.get("playerId") != actor or observation.get("decisionPlayer") != actor
            or not isinstance(tracker, dict)):
        raise ValueError("PPO experience observation/tracker must belong to its decision actor")
    actions = observation.get("legalActions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("PPO experience decision must preserve all actor-visible legal actions")
    action_ids = [action.get("id") for action in actions if isinstance(action, dict)]
    if (len(action_ids) != len(actions) or any(not isinstance(value, str) or not value for value in action_ids)
            or len(set(action_ids)) != len(action_ids)):
        raise ValueError("PPO experience legal action IDs must be present and unique")
    if decision.get("behaviorPolicyHash") != behavior_policy_hash:
        raise ValueError("PPO experience mixes behavior-policy identities")
    if type(decision.get("decisionIndex")) is not int or decision["decisionIndex"] < 0:
        raise ValueError("PPO experience decisions require a nonnegative engine decision index")
    if (type(decision.get("selectedAction")) is not int
            or type(decision.get("actionClassCount")) is not int
            or not 0 < decision["actionClassCount"] <= 128
            or not 0 <= decision["selectedAction"] < decision["actionClassCount"]):
        raise ValueError("PPO selected action class must be represented within the frozen 128-class cap")
    if (decision.get("featureSchemaHash") != feature_schema_hash
            or not _is_sha256(decision.get("featureIdentityHash"))):
        raise ValueError("PPO decision feature schema/encoding identity differs from the frozen run")
    for key in ("oldLogProb", "oldValue"):
        number = decision.get(key)
        if type(number) not in (int, float):
            raise ValueError(f"PPO experience {key} must be numeric")
        if not math.isfinite(number):
            raise ValueError(f"PPO experience {key} must be finite")


def _validate_game(game: dict, *, behavior_policy_hash: str,
                   feature_schema_hash: str, schedule_settings: dict) -> None:
    if (not isinstance(game, dict) or set(game) != {
            "schemaVersion", "gameId", "status", "outcome", "actorDecisions", "schedule"}
            or game.get("schemaVersion") != 1):
        raise ValueError("invalid PPO experience game schema")
    if _has_forbidden_view_key(game):
        raise ValueError("PPO experience game contains a full replay or hidden/private view")
    schedule = game.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("PPO experience game requires its frozen public schedule assignment")
    game_id = game.get("gameId")
    if not isinstance(game_id, str) or len(game_id) != 64 or any(c not in "0123456789abcdef" for c in game_id):
        raise ValueError("PPO experience gameId must be a SHA-256 identity")
    required_schedule_keys = {"scheduleIndex", "seed", "firstPlayer", "decks", "learnerSeats",
        "ownArchetype", "opponentArchetype", "mirror", "opponentPolicy", "policyFamily", "gameId"}
    if (set(schedule) != required_schedule_keys or type(schedule.get("scheduleIndex")) is not int
            or schedule["scheduleIndex"] < 0
            or schedule.get("gameId") != game_id
            or identity_hash({"schedulerVersion": PPO_SCHEDULER_VERSION,
                              **{key: value for key, value in schedule.items() if key != "gameId"}}) != game_id):
        raise ValueError("PPO game ID does not bind its frozen scheduler assignment")
    learner_seats = schedule.get("learnerSeats")
    if (not isinstance(learner_seats, list) or not learner_seats
            or any(type(seat) is not int or seat not in (0, 1) for seat in learner_seats)
            or len(set(learner_seats)) != len(learner_seats)):
        raise ValueError("PPO scheduler assignment has invalid learner seats")
    expected = _ppo_schedule_row(schedule["scheduleIndex"],
        schedule_settings["historicalPolicyHashes"], schedule_settings["trainingSeedBase"])
    if (schedule_settings.get("schedulerVersion") != PPO_SCHEDULER_VERSION
            or schedule != expected):
        raise ValueError("PPO game assignment differs from the frozen scheduler identity")
    if game.get("status") not in GAME_STATUSES:
        raise ValueError("PPO experience game status must preserve finished/truncated/error")
    outcome = game.get("outcome")
    if game["status"] == "finished":
        winner = outcome.get("winner") if isinstance(outcome, dict) else object()
        if (not isinstance(outcome, dict) or (winner is not None and (type(winner) is not int or winner not in (0, 1)))
                or not isinstance(outcome.get("reason"), str) or not outcome["reason"]):
            raise ValueError("finished PPO games require the public terminal W/D/L outcome")
    elif outcome is not None:
        raise ValueError("truncated/error PPO games cannot be converted into terminal outcomes")
    decisions = game.get("actorDecisions")
    if not isinstance(decisions, list):
        raise ValueError("PPO experience game is missing its actor decision list")
    for decision in decisions:
        _validate_decision(decision, behavior_policy_hash=behavior_policy_hash,
                           feature_schema_hash=feature_schema_hash)
        if decision["actor"] not in learner_seats:
            raise ValueError("PPO experience includes a decision from a non-learner seat")
    previous: dict[int, int] = {}
    for decision in decisions:
        actor, index = decision["actor"], decision["decisionIndex"]
        if index <= previous.get(actor, -1):
            raise ValueError("PPO actor decisions must preserve strictly increasing engine order")
        previous[actor] = index


def ppo_records_from_game(game: dict, *, behavior_policy_hash: str,
                          feature_schema_hash: str, scheduler_settings: dict) -> list[dict]:
    """Re-encode actor-view decisions and attach terminal perspective rewards/GAE."""
    _validate_game(game, behavior_policy_hash=behavior_policy_hash,
                   feature_schema_hash=feature_schema_hash,
                   schedule_settings=scheduler_settings)
    from .encoding import encode_decision
    from .training import generalized_advantages

    by_actor: dict[int, list[dict]] = {}
    for decision in game["actorDecisions"]:
        actor = decision["actor"]
        encoded = encode_decision(decision["observation"], decision["tracker"])
        if (encoded.identity != decision["featureIdentityHash"]
                or len(encoded.action_classes) != decision["actionClassCount"]
                or not 0 <= decision["selectedAction"] < len(encoded.action_classes)):
            raise ValueError("PPO actor decision encoding/action differs from its frozen feature identity")
        by_actor.setdefault(actor, []).append({**decision, "encoded": encoded})

    result = []
    for actor in sorted(by_actor):
        decisions = sorted(by_actor[actor], key=lambda row: row["decisionIndex"])
        rows = []
        for offset, decision in enumerate(decisions):
            terminal_reward = 0
            if offset == len(decisions) - 1 and game["status"] == "finished":
                winner = game["outcome"]["winner"]
                terminal_reward = 0 if winner is None else 1 if winner == actor else -1
            rows.append({"episodeId": f"{game['gameId']}:seat{actor}",
                "episodeStatus": game["status"], "episodeEnd": offset == len(decisions) - 1,
                "reward": terminal_reward, "gameId": game["gameId"], "actor": actor,
                "decisionIndex": decision["decisionIndex"], "encoded": decision["encoded"],
                "selectedAction": decision["selectedAction"], "oldLogProb": decision["oldLogProb"],
                "oldValue": decision["oldValue"], "behaviorPolicyHash": behavior_policy_hash,
                "featureIdentityHash": decision["featureIdentityHash"],
                "featureSchemaHash": feature_schema_hash})
        advantages = generalized_advantages(rows, [row["oldValue"] for row in rows])
        for row, advantage in zip(rows, advantages):
            row["advantage"] = advantage
            row["return"] = row["oldValue"] + advantage
            result.append(row)
    return result


class PPOExperienceStore:
    """Resumable, checksummed actor-view game store for future PPO collection.

    This storage API does not start an engine or trainer. Each completed game is
    immutable; an interrupted unpublished game is retried with the same schedule
    seed by its caller, while mismatched identities or orphan artifacts fail closed.
    """

    def __init__(self, root: Path, *, settings: dict):
        if not isinstance(settings, dict) or not settings:
            raise ValueError("PPO experience store requires frozen settings")
        behavior_hash = settings.get("behaviorPolicyHash")
        if not _is_sha256(behavior_hash):
            raise ValueError("PPO experience settings require a behaviorPolicyHash")
        if not _is_sha256(settings.get("featureSchemaHash")):
            raise ValueError("PPO experience settings require a featureSchemaHash")
        historical = settings.get("historicalPolicyHashes")
        if (settings.get("schedulerVersion") != PPO_SCHEDULER_VERSION
                or not isinstance(historical, list) or not historical
                or any(not _is_sha256(value) for value in historical)
                or len(set(historical)) != len(historical)
                or type(settings.get("trainingSeedBase")) is not int
                or not 0 <= settings["trainingSeedBase"] < 2**32):
            raise ValueError("PPO experience settings require the exact scheduler, historical policies, and seed base")
        self.root = Path(root).resolve()
        self.games_dir = self.root / "games"
        self.manifest_path = self.root / "manifest.json"
        self.settings = json.loads(json.dumps(settings, sort_keys=True, allow_nan=False))
        self.run_id = identity_hash({"version": EXPERIENCE_STORE_VERSION, "settings": self.settings})
        if self.manifest_path.exists():
            self.manifest = self._load_and_verify()
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("PPO experience output is nonempty without a verified manifest")
            self.root.mkdir(parents=True, exist_ok=True)
            self.games_dir.mkdir(parents=True, exist_ok=True)
            self.manifest = self._new_manifest()
            self._write_manifest()

    def _new_manifest(self) -> dict:
        manifest = {"schemaVersion": 1, "kind": EXPERIENCE_STORE_VERSION,
            "runId": self.run_id, "settings": self.settings, "games": [],
            "gameCounts": {"finished": 0, "truncated": 0, "error": 0},
            "actorDecisions": 0}
        manifest["manifestHash"] = identity_hash(manifest)
        return manifest

    def _load_and_verify(self) -> dict:
        if not self.games_dir.is_dir():
            raise ValueError("PPO experience game-artifact directory is missing")
        manifest = json.loads(self.manifest_path.read_text())
        if (not isinstance(manifest, dict)
                or set(manifest) != {"schemaVersion", "kind", "runId", "settings", "games",
                                     "gameCounts", "actorDecisions", "manifestHash"}
                or manifest.get("schemaVersion") != 1
                or manifest.get("manifestHash") != identity_hash(
                    {key: value for key, value in manifest.items() if key != "manifestHash"})
                or manifest.get("kind") != EXPERIENCE_STORE_VERSION
                or manifest.get("runId") != self.run_id
                or manifest.get("settings") != self.settings):
            raise ValueError("PPO experience manifest identity/checksum mismatch")
        games = manifest.get("games")
        if not isinstance(games, list):
            raise ValueError("PPO experience manifest has no game list")
        game_counts = manifest.get("gameCounts")
        if (not isinstance(game_counts, dict) or set(game_counts) != GAME_STATUSES
                or any(type(count) is not int or count < 0 for count in game_counts.values())
                or type(manifest.get("actorDecisions")) is not int
                or manifest["actorDecisions"] < 0):
            raise ValueError("PPO experience manifest aggregate schema is invalid")
        expected_names = set()
        counts = {status: 0 for status in GAME_STATUSES}
        actor_decisions = 0
        seen_ids = set()
        seen_schedule_indices = set()
        for item in games:
            if (not isinstance(item, dict)
                    or set(item) != {"gameId", "scheduleIndex", "status", "actorDecisions", "sha256", "bytes"}
                    or type(item.get("scheduleIndex")) is not int or item["scheduleIndex"] < 0
                    or item.get("status") not in GAME_STATUSES
                    or type(item.get("actorDecisions")) is not int or item["actorDecisions"] < 0
                    or type(item.get("bytes")) is not int or item["bytes"] < 1):
                raise ValueError("PPO experience manifest contains an invalid game entry")
            game_id = item.get("gameId")
            if game_id in seen_ids or not isinstance(game_id, str):
                raise ValueError("PPO experience manifest repeats or omits a game ID")
            seen_ids.add(game_id)
            name = f"{game_id}.json.gz"
            expected_names.add(name)
            path = self.games_dir / name
            if (not path.is_file() or file_sha256(path) != item.get("sha256")
                    or path.stat().st_size != item.get("bytes")):
                raise ValueError(f"PPO experience game artifact checksum mismatch: {game_id}")
            game = self._read_game(path)
            _validate_game(game, behavior_policy_hash=self.settings["behaviorPolicyHash"],
                           feature_schema_hash=self.settings["featureSchemaHash"],
                           schedule_settings=self.settings)
            if game.get("gameId") != game_id or game.get("status") != item.get("status"):
                raise ValueError("PPO experience game artifact differs from its manifest entry")
            if item["actorDecisions"] != len(game["actorDecisions"]):
                raise ValueError("PPO experience per-game decision count differs from its artifact")
            schedule_index = game["schedule"]["scheduleIndex"]
            if (schedule_index in seen_schedule_indices or item.get("scheduleIndex") != schedule_index):
                raise ValueError("PPO experience manifest repeats or misstates a scheduler index")
            seen_schedule_indices.add(schedule_index)
            counts[game["status"]] += 1
            actor_decisions += len(game["actorDecisions"])
        actual_names = {path.name for path in self.games_dir.glob("*.json.gz")}
        if actual_names != expected_names:
            raise ValueError("PPO experience files do not exactly match the manifest")
        if game_counts != counts or manifest["actorDecisions"] != actor_decisions:
            raise ValueError("PPO experience manifest aggregate counts do not match verified games")
        return manifest

    def _write_manifest(self) -> None:
        value = {key: item for key, item in self.manifest.items() if key != "manifestHash"}
        self.manifest["manifestHash"] = identity_hash(value)
        _atomic_json(self.manifest_path, self.manifest)

    @staticmethod
    def _read_game(path: Path) -> dict:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("PPO experience artifact must contain a JSON object")
        return value

    def save_game(self, game: dict) -> dict:
        _validate_game(game, behavior_policy_hash=self.settings["behaviorPolicyHash"],
                       feature_schema_hash=self.settings["featureSchemaHash"],
                       schedule_settings=self.settings)
        if any(item["gameId"] == game["gameId"] for item in self.manifest["games"]):
            raise ValueError("PPO experience games are immutable and cannot be replaced")
        if any(item.get("scheduleIndex") == game["schedule"]["scheduleIndex"]
               for item in self.manifest["games"]):
            raise ValueError("PPO experience scheduler indices are immutable and cannot be reused")
        path = self.games_dir / f"{game['gameId']}.json.gz"
        if path.exists():
            raise ValueError("unpublished PPO game artifact exists; preserve it for manual reconciliation")
        payload = json.dumps(game, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=self.games_dir, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with temporary_path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    compressed.write(payload)
                raw.flush()
                os.fsync(raw.fileno())
            os.link(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        item = {"gameId": game["gameId"], "scheduleIndex": game["schedule"]["scheduleIndex"],
            "status": game["status"],
            "actorDecisions": len(game["actorDecisions"]), "sha256": file_sha256(path),
            "bytes": path.stat().st_size}
        self.manifest["games"].append(item)
        self.manifest["games"].sort(key=lambda entry: entry["gameId"])
        self.manifest["gameCounts"][game["status"]] += 1
        self.manifest["actorDecisions"] += item["actorDecisions"]
        self._write_manifest()
        return item

    def completed_game_ids(self) -> frozenset[str]:
        return frozenset(item["gameId"] for item in self.manifest["games"])

    def iter_games(self) -> Iterator[dict]:
        for item in sorted(self.manifest["games"], key=lambda entry: entry["gameId"]):
            path = self.games_dir / f"{item['gameId']}.json.gz"
            game = self._read_game(path)
            _validate_game(game, behavior_policy_hash=self.settings["behaviorPolicyHash"],
                           feature_schema_hash=self.settings["featureSchemaHash"],
                           schedule_settings=self.settings)
            yield game
