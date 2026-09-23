from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from ptcg_lab.storage import Store, digest as legacy_digest

from .encoding import encode_decision
from .schema import IdentityManifest, identity_hash
from .tracker import ObservableHistoryTracker

EXACT_REVIEWS = (
    "530af43192745703ac2acf2dc7ea4a4ac1520f226b4cba3a4636eac617f954e4",
    "832239559647d57880eba66a7f28718c1bebb0ebd82f5f66040bcd9f7a4aab32",
    "b6a6c6526c4fb448e856f471edb25d94c048cb9c34c0d3ab9456502657ac0f0c",
    "da82cf9119099fbff943b254a499f579c8c8512020f5cbdcaf767512ab7ee23b",
)
STALE_REVIEWS = (
    "1e803b1a30c55cf8f61bbe7ee1cc2ec6aca2b5dbaa7f20f416560aeaf84a0510",
    "49f4021acf629bce7b69352b8eee611f490665f771b222c1aa0b042fae6fa36d",
    "c0d1cfe46ceda16810fc05b4e2290083685000e9186962350cc33c5e0875ea82",
    "f34b1c1bc4182ba857136585dcb3c683b195e59487830308444a1eab3be51260",
)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_split(family: str) -> str:
    bucket = int(hashlib.sha256(f"learning-mind-v1|{family}".encode()).hexdigest()[:8], 16) % 10
    return "heldout" if bucket == 0 else "development" if bucket == 1 else "train"


def _map_acceptable(encoded, action_ids: set[str]) -> list[int]:
    indices = [index for index, group in enumerate(encoded.action_classes)
               if any(action.get("id") in action_ids for action in group.actions)]
    represented = {action.get("id") for index in indices for action in encoded.action_classes[index].actions
                   if action.get("id") in action_ids}
    if represented != action_ids:
        raise ValueError("review action is not represented exactly once")
    return indices


def _map_distribution(encoded, probabilities: dict[str, float]) -> list[float]:
    result = []
    represented = set()
    for group in encoded.action_classes:
        ids = {str(action.get("id")) for action in group.actions}
        represented.update(ids)
        result.append(sum(float(probabilities.get(identifier, 0)) for identifier in ids))
    if represented != set(probabilities) or abs(sum(result) - 1) > 1e-5:
        raise ValueError("search distribution does not cover the semantic legal-action classes")
    return result + [0.0]  # STOP is not a root search action.


def _review_rows(review_root: Path) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    sources = []
    for review_hash in EXACT_REVIEWS:
        path = review_root / f"{review_hash}.json"
        record = json.loads(path.read_text())
        if record.get("reviewHash") != review_hash or record.get("reviewStatus") != "reviewed" or record.get("trainingEligible") is not True:
            raise ValueError(f"exact review is no longer compatible: {review_hash}")
        if legacy_digest(record.get("observation")) != record.get("positionHash"):
            raise ValueError(f"review observation changed: {review_hash}")
        grouped[record["positionHash"]].append(record)
        sources.append({"kind": "teaching-review", "path": str(path), "sha256": file_sha256(path),
                        "reviewHash": review_hash, "familyId": record["familyId"]})
    rows = []
    for position_hash, records in sorted(grouped.items()):
        observation = records[0]["observation"]
        if any(record["observation"] != observation for record in records):
            raise ValueError("reviews with one position hash contain different observations")
        accepted = set().union(*(set(record["acceptableActionIds"]) for record in records))
        tracker = ObservableHistoryTracker(observation["playerId"]); snapshot = tracker.update(observation)
        encoded = encode_decision(observation, snapshot)
        rows.append({"positionHash": position_hash, "familyId": records[0]["familyId"],
                     "sourceGameId": None, "sourceDecisionIndex": None, "actor": observation["playerId"],
                     "deckHash": None, "opponentArchetype": "unknown-reviewed-position",
                     "opponentPolicyFamily": "human-review", "featureIdentityHash": encoded.identity,
                     "policyLabelSource": "compatible-reviewed-acceptable-set",
                     "acceptableActionIndices": _map_acceptable(encoded, accepted), "policyDistribution": None,
                     "split": records[0].get("partition", "train"), "observation": observation,
                     "tracker": snapshot, "reviewHashes": sorted(record["reviewHash"] for record in records)})
    return rows, sources


def _search_rows(experimental_root: Path, dataset_manifest: Path) -> tuple[list[dict], list[dict]]:
    store = Store(experimental_root)
    source_manifest = json.loads(dataset_manifest.read_text())
    rows, sources = [], [{"kind": "experimental-dataset-manifest", "path": str(dataset_manifest),
                          "sha256": file_sha256(dataset_manifest)}]
    for item in source_manifest.get("replays", []):
        replay_path = experimental_root / "replays" / f"{item['id']}.json"
        replay = store.get("replays", item["id"])
        if not replay.get("searchTargets"):
            continue
        sources.append({"kind": "experimental-replay", "path": str(replay_path),
                        "sha256": file_sha256(replay_path), "replayId": item["id"]})
        trackers = {0: ObservableHistoryTracker(0), 1: ObservableHistoryTracker(1)}
        targets = {target["decisionIndex"]: target for target in replay["searchTargets"]}
        for frame in replay.get("frames", []):
            actor = frame.get("actor")
            if actor not in (0, 1): continue
            observation = frame["observations"][actor]
            snapshot = trackers[actor].update(observation)
            target = targets.get(frame["decisionIndex"])
            if not target: continue
            if target.get("actor") != actor or legacy_digest(observation) != target.get("observationHash"):
                raise ValueError("search target identity differs from its actor-visible frame")
            encoded = encode_decision(observation, snapshot)
            family = item.get("familyId", identity_hash({"replay": item["id"]}))
            opponent = replay.get("decks", ["unknown", "unknown"])[1 - actor]
            rows.append({"positionHash": target["observationHash"], "familyId": family,
                         "sourceGameId": replay["id"], "sourceDecisionIndex": frame["decisionIndex"], "actor": actor,
                         "deckHash": (replay.get("deckHashes") or [None, None])[actor],
                         "opponentArchetype": opponent.replace("-training", ""),
                         "opponentPolicyFamily": "frozen-search-source", "featureIdentityHash": encoded.identity,
                         "policyLabelSource": "exact-search-distribution", "acceptableActionIndices": None,
                         "policyDistribution": _map_distribution(encoded, target["probabilities"]),
                         "split": stable_split(family), "observation": observation, "tracker": snapshot,
                         "searchTarget": {key: target.get(key) for key in ("method", "budgetMs", "iterations", "rootPolicy")}})
    return rows, sources


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    with temporary.open("rb") as source: os.fsync(source.fileno())
    temporary.replace(path)


def build_dataset(*, root: Path, output: Path, review_root: Path,
                  experimental_root: Path, source_dataset_manifest: Path,
                  identity: IdentityManifest) -> dict:
    if output.exists(): raise ValueError("dataset outputs are immutable; choose a new directory")
    output.mkdir(parents=True)
    reviews, review_sources = _review_rows(review_root)
    searches, search_sources = _search_rows(experimental_root, source_dataset_manifest)
    unique = {}
    exclusions = []
    for row in reviews + searches:
        key = row["positionHash"]
        if key in unique:
            exclusions.append({"positionHash": row["positionHash"], "reason": "duplicate-actor-visible-position",
                               "excludedSource": row["policyLabelSource"],
                               "retainedSource": unique[key]["policyLabelSource"]})
        else: unique[key] = row
    rows = [unique[key] for key in sorted(unique)]
    rows_path = output / "rows.jsonl"
    _atomic_text(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    counts = Counter((row["split"], row["policyLabelSource"]) for row in rows)
    dimensions = {}
    for field in ("split", "policyLabelSource", "familyId", "opponentArchetype", "opponentPolicyFamily"):
        dimensions[field] = dict(sorted(Counter(str(row.get(field)) for row in rows).items()))
    search_configurations = sorted({identity_hash(row["searchTarget"]) for row in rows if row.get("searchTarget")})
    manifest = {"schemaVersion": 1, "id": "learning-mind-supervised-v1", "identity": identity.record(),
                "rows": len(rows), "rowsSha256": file_sha256(rows_path),
                "counts": [{"split": split, "source": source, "count": count}
                           for (split, source), count in sorted(counts.items())],
                "countsByDimension": dimensions,
                "families": sorted({row["familyId"] for row in rows}),
                "searchConfigurationHashes": search_configurations,
                "sources": review_sources + search_sources, "exclusions": exclusions,
                "staleReviewsExcluded": list(STALE_REVIEWS),
                "policyLabelSources": sorted({row["policyLabelSource"] for row in rows}),
                "ordinarySelfPlayPolicyLabels": 0,
                "splitPolicy": "review partition retained; search families use sha256 stable 80/10/10 split"}
    manifest["manifestHash"] = identity_hash(manifest)
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_dataset(path: Path, *, identity: dict | None = None) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    if file_sha256(path / "rows.jsonl") != manifest["rowsSha256"]: raise ValueError("dataset rows hash mismatch")
    if identity is not None and manifest["identity"] != identity: raise ValueError("dataset identity mismatch")
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    if len(rows) != manifest["rows"]: raise ValueError("dataset row count mismatch")
    return manifest, rows


def training_records(rows: list[dict], split: str = "train") -> list[dict]:
    result = []
    for row in rows:
        if row["split"] != split: continue
        encoded = encode_decision(row["observation"], row["tracker"])
        if encoded.identity != row["featureIdentityHash"]: raise ValueError("encoded row identity drift")
        result.append({"encoded": encoded, "policyLabelSource": row["policyLabelSource"],
                       "acceptableActionIndices": row.get("acceptableActionIndices"),
                       "policyDistribution": row.get("policyDistribution")})
    return result


def build_macro_position_pool(*, output: Path, experimental_root: Path,
                              source_dataset_manifest: Path, identity: IdentityManifest,
                              target_deck: str = "raging-bolt", limit: int = 18) -> dict:
    """Freeze actor-visible, unlabeled positions for rollout ranking only."""
    if output.exists():
        raise ValueError("macro position pools are immutable; choose a new directory")
    source = json.loads(source_dataset_manifest.read_text())
    store = Store(experimental_root)
    candidates = []
    sources = []
    source_engine_versions = set()

    def position_stage(observation: dict) -> str:
        turn = observation.get("turn")
        prizes = [player.get("prizesRemaining") for player in observation.get("players", [])]
        visible_prizes = [value for value in prizes if isinstance(value, int) and not isinstance(value, bool)]
        if isinstance(turn, int) and turn <= 2 and len(visible_prizes) == len(prizes) and all(value >= 5 for value in visible_prizes):
            return "opening"
        if ((isinstance(turn, int) and turn >= 8)
                or (visible_prizes and min(visible_prizes) <= 3)):
            return "late"
        return "midgame"
    for item in sorted(source.get("replays", []), key=lambda row: row["id"]):
        if not any(str(deck).replace("-training", "") == target_deck for deck in item.get("decks", [])):
            continue
        replay = store.get("replays", item["id"])
        source_engine_version = replay.get("engineVersion") or item.get("engineVersion")
        if isinstance(source_engine_version, str) and source_engine_version:
            source_engine_versions.add(source_engine_version)
        sources.append({"replayId": item["id"], "sha256": file_sha256(store.location("replays", item["id"])),
                        "engineVersion": source_engine_version})
        trackers = {0: ObservableHistoryTracker(0), 1: ObservableHistoryTracker(1)}
        for frame in replay.get("frames", []):
            actor = frame.get("actor")
            if actor not in (0, 1):
                continue
            observation = frame["observations"][actor]
            snapshot = trackers[actor].update(observation)
            own_deck = str(replay.get("decks", ["", ""])[actor]).replace("-training", "")
            if (own_deck != target_deck or observation.get("prompt") or observation.get("searchUnavailableReason")
                    or str(observation.get("phase", "")).lower().replace("_", "-") != "player-turn"
                    or len(observation.get("legalActions", [])) < 2):
                continue
            encoded = encode_decision(observation, snapshot)
            if len(encoded.action_classes) < 2:
                continue
            position_hash = legacy_digest(observation)
            opponent = str(replay.get("decks", ["unknown", "unknown"])[1 - actor]).replace("-training", "")
            policies = replay.get("policies") or ["unknown", "unknown"]
            candidates.append({"positionHash": position_hash,
                "familyId": f"{target_deck}-macro-plan:{item['familyId']}",
                "sourceGameId": replay["id"], "sourceDecisionIndex": frame["decisionIndex"], "actor": actor,
                "deckHash": (replay.get("deckHashes") or [None, None])[actor],
                "opponentArchetype": opponent, "opponentPolicyFamily": str(policies[1 - actor]),
                "positionStage": position_stage(observation),
                "featureIdentityHash": encoded.identity, "policyLabelSource": None,
                "acceptableActionIndices": None, "policyDistribution": None,
                "observation": observation, "tracker": snapshot})
    # A repeated visible state is one position even if several replay records contain it.
    unique_candidates = {}
    for row in candidates:
        unique_candidates.setdefault(row["positionHash"], row)
    candidates = list(unique_candidates.values())

    # Round-robin across matchup, policy family, and observable game stage.
    buckets = defaultdict(list)
    for row in candidates:
        buckets[(row["opponentArchetype"], row["opponentPolicyFamily"], row["positionStage"])].append(row)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    game_rows = defaultdict(list)
    for row in selected:
        game_rows[row["sourceGameId"]].append(row)
    ordered_games = sorted(game_rows, key=lambda game_id: (
        game_rows[game_id][0]["opponentArchetype"], game_rows[game_id][0]["opponentPolicyFamily"], game_id))
    game_splits = {}
    if len(ordered_games) < 3:
        game_splits = {game_id: "development" for game_id in ordered_games}
    else:
        development_index = max(1, int(len(ordered_games) * .6))
        heldout_index = max(development_index + 1, int(len(ordered_games) * .8))
        for index, game_id in enumerate(ordered_games):
            game_splits[game_id] = ("development" if index == development_index else
                                    "heldout" if index == heldout_index else "train")
    for row in selected:
        row["split"] = game_splits[row["sourceGameId"]]
    output.mkdir(parents=True)
    rows_path = output / "rows.jsonl"
    _atomic_text(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected))
    manifest = {"schemaVersion": 1, "id": "learning-mind-macro-position-pool-v1",
                "identity": identity.record(), "targetDeck": target_deck, "rows": len(selected),
                "rowsSha256": file_sha256(rows_path), "sourceManifestSha256": file_sha256(source_dataset_manifest),
                "sources": sources, "sourceEngineVersions": sorted(source_engine_versions),
                "ordinarySelfPlayPolicyLabels": 0,
                "splitCounts": dict(sorted(Counter(row["split"] for row in selected).items())),
                "opponentArchetypes": sorted({row["opponentArchetype"] for row in selected}),
                "opponentPolicyFamilies": sorted({row["opponentPolicyFamily"] for row in selected}),
                "positionStages": sorted({row["positionStage"] for row in selected}),
                "positionStageCounts": dict(sorted(Counter(row["positionStage"] for row in selected).items())),
                "sourceGameSplitCounts": dict(sorted(Counter(game_splits.values()).items())),
                "selection": "deterministic round-robin by opponent archetype, policy family, and position stage; disjoint source-game splits"}
    manifest["manifestHash"] = identity_hash(manifest)
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest
