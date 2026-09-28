from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
import shutil
import tempfile
from pathlib import Path

from ptcg_lab.storage import Store, digest as legacy_digest
from research.strategy_baseline.probes import registry_hash as strategy_probe_registry_hash

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


def _source_game_split_quotas(game_count: int) -> dict[str, int]:
    if game_count < 3:
        return {"train": 0, "development": game_count, "heldout": 0}
    if game_count < 8:
        return {"train": game_count - 2, "development": 1, "heldout": 1}
    development = max(2, int(game_count * 0.15 + 0.5))
    heldout = max(2, int(game_count * 0.15 + 0.5))
    while game_count - development - heldout < 3:
        if development >= heldout and development > 2:
            development -= 1
        elif heldout > 2:
            heldout -= 1
        else:
            break
    return {"train": game_count - development - heldout,
            "development": development, "heldout": heldout}


def _assign_source_game_splits(game_rows: dict[str, list[dict]]) -> dict[str, str]:
    """Assign whole games to row-balanced splits, stratified by matchup/policy."""
    game_ids = sorted(game_rows, key=lambda game_id: hashlib.sha256(
        f"learning-mind-v1|macro-pool-split-v2|{game_id}".encode()).hexdigest())
    quotas = _source_game_split_quotas(len(game_ids))
    if len(game_ids) < 3:
        return {game_id: "development" for game_id in game_ids}
    contexts = {game_id: (game_rows[game_id][0]["opponentArchetype"],
                          game_rows[game_id][0]["opponentPolicyFamily"])
                for game_id in game_ids}
    total_rows = sum(len(rows) for rows in game_rows.values())
    fractions = {name: count / len(game_ids) for name, count in quotas.items()}
    total_by_context = Counter(contexts.values())

    def score(assignment: dict[str, str]) -> tuple[float, tuple[str, ...]]:
        row_counts = Counter()
        context_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        game_counts = Counter(assignment.values())
        for game_id, split in assignment.items():
            row_counts[split] += len(game_rows[game_id])
            context_counts[contexts[game_id]][split] += 1
        row_loss = sum(((row_counts[name] / total_rows - fractions[name]) ** 2)
                       / max(fractions[name], 1e-9) for name in quotas if fractions[name])
        game_loss = sum(((game_counts[name] - quotas[name]) / max(quotas[name], 1)) ** 2
                        for name in quotas)
        context_loss = 0.0
        for context, context_total in total_by_context.items():
            for name in quotas:
                expected = context_total * fractions[name]
                actual = context_counts[context][name]
                context_loss += ((actual - expected) ** 2) / max(expected, 1.0)
        signature = tuple(assignment[game_id] for game_id in game_ids)
        return row_loss + 0.05 * game_loss + 0.10 * context_loss, signature

    best: tuple[float, tuple[str, ...]] | None = None
    best_assignment: dict[str, str] | None = None
    if len(game_ids) <= 18:
        for development_ids in itertools.combinations(game_ids, quotas["development"]):
            remaining = [game_id for game_id in game_ids if game_id not in development_ids]
            for heldout_ids in itertools.combinations(remaining, quotas["heldout"]):
                dev, held = set(development_ids), set(heldout_ids)
                assignment = {game_id: ("development" if game_id in dev else
                                         "heldout" if game_id in held else "train")
                              for game_id in game_ids}
                candidate = score(assignment)
                if best is None or candidate < best:
                    best, best_assignment = candidate, assignment
    else:
        # Stable, bounded local search for larger collections; every swap keeps
        # the exact game-count quotas and the source games remain disjoint.
        labels = [name for name in ("train", "development", "heldout")
                  for _ in range(quotas[name])]
        assignment = dict(zip(game_ids, labels))
        current = score(assignment)
        improved = True
        while improved:
            improved = False
            next_assignment = assignment
            next_score = current
            for left_index, left in enumerate(game_ids):
                for right in game_ids[left_index + 1:]:
                    if assignment[left] == assignment[right]:
                        continue
                    swapped = dict(assignment)
                    swapped[left], swapped[right] = swapped[right], swapped[left]
                    candidate = score(swapped)
                    if candidate < next_score:
                        next_assignment, next_score = swapped, candidate
            if next_score < current:
                assignment, current = next_assignment, next_score
                improved = True
        best_assignment = assignment
    assert best_assignment is not None
    return best_assignment


def _select_macro_positions(candidates: list[dict], limit: int) -> list[dict]:
    """Round-robin target decks, context buckets, and source games."""
    if type(limit) is not int or limit < 1:
        raise ValueError("macro position limit must be a positive integer")

    target_groups: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        target_groups[str(row.get("targetDeck", "unknown"))].append(row)
    if len(target_groups) > 1:
        per_target = {target: _select_macro_positions_for_target(rows, limit)
                      for target, rows in target_groups.items()}
        selected = []
        offsets = {target: 0 for target in target_groups}
        while len(selected) < limit:
            added = False
            for target in sorted(per_target):
                offset = offsets[target]
                if offset < len(per_target[target]) and len(selected) < limit:
                    selected.append(per_target[target][offset])
                    offsets[target] += 1
                    added = True
            if not added:
                break
        return selected
    return _select_macro_positions_for_target(candidates, limit)


def _select_macro_positions_for_target(candidates: list[dict], limit: int) -> list[dict]:
    """Balance matchup, policy family, stage, and source games within one deck."""
    buckets: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        context = (row["opponentArchetype"], row["opponentPolicyFamily"], row["positionStage"])
        buckets[context][row["sourceGameId"]].append(row)
    for games in buckets.values():
        for rows in games.values():
            rows.sort(key=lambda row: (row["sourceDecisionIndex"], row["positionHash"]))

    selected = []
    while len(selected) < limit:
        added = False
        for context in sorted(buckets):
            games = buckets[context]
            game_ids = sorted(games, key=lambda game_id: hashlib.sha256(
                f"learning-mind-v1|macro-pool-source-order-v1|{game_id}".encode()).hexdigest())
            for game_id in game_ids:
                if games[game_id] and len(selected) < limit:
                    selected.append(games[game_id].pop(0))
                    added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


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


def build_strategy_probe_dataset(*, output: Path, experimental_root: Path,
                                 source_dataset_manifest: Path,
                                 identity: IdentityManifest) -> dict:
    """Freeze every actor-visible decision from source games in the held-out family split."""
    if output.exists():
        raise ValueError("strategy-probe datasets are immutable; choose a new directory")
    source_manifest = json.loads(source_dataset_manifest.read_text())
    if source_manifest.get("manifestHash") != identity_hash({key: value for key, value in source_manifest.items()
                                                              if key != "manifestHash"}):
        raise ValueError("strategy-probe source replay manifest hash mismatch")
    source_identity = source_manifest.get("identity")
    source_settings = source_manifest.get("settings")
    if source_identity is None and isinstance(source_settings, dict):
        source_identity = source_settings.get("identity")
    if source_identity != identity.record():
        raise ValueError("strategy-probe source replay identity mismatch")
    replay_items = source_manifest.get("replays")
    if not isinstance(replay_items, list) or not replay_items:
        raise ValueError("strategy-probe source manifest must list frozen replays")
    if any(not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]
           for item in replay_items):
        raise ValueError("strategy-probe source manifest has an invalid replay record")
    replay_ids_in_manifest = [item["id"] for item in replay_items]
    if len(set(replay_ids_in_manifest)) != len(replay_items):
        raise ValueError("strategy-probe source manifest has missing or duplicate replay IDs")
    store = Store(experimental_root)
    selected_sources = []
    rows = []
    replay_ids = set()
    for item in sorted(replay_items, key=lambda value: value["id"]):
        replay_id = item["id"]
        family = item.get("familyId", identity_hash({"replay": replay_id}))
        if not isinstance(family, str) or not family:
            raise ValueError(f"strategy-probe source replay has invalid family ID: {replay_id}")
        split = stable_split(family)
        if split != "heldout":
            continue
        if replay_id in replay_ids:
            raise ValueError("strategy-probe source repeats a replay ID")
        replay_ids.add(replay_id)
        path = store.location("replays", replay_id)
        replay = store.get("replays", replay_id)
        if (replay.get("id") != replay_id or replay.get("dataTier") != "experimental"
                or replay.get("status") != "finished"):
            raise ValueError(f"strategy-probe source replay identity/tier mismatch: {replay_id}")
        trackers = {0: ObservableHistoryTracker(0), 1: ObservableHistoryTracker(1)}
        game_rows = []
        last_index = -1
        for frame in replay.get("frames", []):
            if not isinstance(frame, dict):
                raise ValueError(f"strategy-probe replay contains an invalid frame: {replay_id}")
            # Engine replays end with a terminal observation frame whose action
            # is null. It is an outcome boundary, not a policy decision.
            if frame.get("action") is None:
                continue
            actor = frame.get("actor")
            if type(actor) is not int or actor not in (0, 1):
                raise ValueError(f"strategy-probe decision frame has an invalid actor: {replay_id}")
            decision_index = frame.get("decisionIndex")
            if type(decision_index) is not int or decision_index != last_index + 1:
                raise ValueError(f"strategy-probe replay decisions are not strictly ordered: {replay_id}")
            last_index = decision_index
            observations = frame.get("observations")
            if not isinstance(observations, list) or len(observations) != 2:
                raise ValueError(f"strategy-probe frame lacks two private-view slots: {replay_id}")
            observation = observations[actor]
            if not isinstance(observation, dict) or observation.get("playerId") != actor:
                raise ValueError(f"strategy-probe frame is not the actor's observation: {replay_id}")
            if observation.get("decisionPlayer", actor) != actor:
                raise ValueError(f"strategy-probe frame exposes another decision player: {replay_id}")
            snapshot = trackers[actor].update(observation)
            encoded = encode_decision(observation, snapshot)
            position_hash = legacy_digest(observation)
            game_rows.append({"positionHash": position_hash, "sourceGameId": replay_id,
                "sourceDecisionIndex": decision_index, "actor": actor, "gameSideKey": f"{replay_id}:{actor}",
                "familyId": family, "split": "heldout", "featureIdentityHash": encoded.identity,
                "observation": observation, "tracker": snapshot})
        if not game_rows:
            raise ValueError(f"held-out source replay has no actor decision frames: {replay_id}")
        rows.extend(game_rows)
        selected_sources.append({"replayId": replay_id, "familyId": family, "split": split,
            "status": replay.get("status", "unknown"), "actorDecisionRows": len(game_rows),
            "sha256": file_sha256(path)})
    if not selected_sources:
        raise ValueError("frozen source manifest contains no held-out source games")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        rows_path = temporary_root / "rows.jsonl"
        _atomic_text(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                          for row in rows))
        manifest = {"schemaVersion": 1, "kind": "learning-mind-strategy-probes-v1",
            "identity": identity.record(), "sourceDatasetManifestSha256": file_sha256(source_dataset_manifest),
            "sourcePolicy": "stable_split(familyId)==heldout; every actor decision frame from each selected game",
            "probeRegistryHash": strategy_probe_registry_hash(),
            "actorViewOnly": True, "actualActionsExcluded": True,
            "sourceGames": selected_sources, "sourceGameCount": len(selected_sources),
            "actorDecisionRows": len(rows), "rowsSha256": file_sha256(rows_path),
            "rows": len(rows), "split": "heldout"}
        manifest["manifestHash"] = identity_hash(manifest)
        _atomic_text(temporary_root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        temporary_root.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root)
        raise


def load_strategy_probe_dataset(path: Path, *, identity: dict) -> tuple[dict, list[dict]]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "learning-mind-strategy-probes-v1":
        raise ValueError("strategy-probe evaluation requires its dedicated full-decision corpus")
    if manifest.get("identity") != identity:
        raise ValueError("strategy-probe corpus identity mismatch")
    if manifest.get("actorViewOnly") is not True or manifest.get("actualActionsExcluded") is not True:
        raise ValueError("strategy-probe corpus violates its actor-visible, unlabeled contract")
    if manifest.get("probeRegistryHash") != strategy_probe_registry_hash():
        raise ValueError("strategy-probe evaluator registry hash mismatch")
    if manifest.get("split") != "heldout":
        raise ValueError("strategy-probe evaluation requires held-out source games")
    if manifest.get("manifestHash") != identity_hash({key: value for key, value in manifest.items()
                                                       if key != "manifestHash"}):
        raise ValueError("strategy-probe corpus manifest hash mismatch")
    rows_path = path / "rows.jsonl"
    if file_sha256(rows_path) != manifest.get("rowsSha256"):
        raise ValueError("strategy-probe corpus rows hash mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows) != manifest.get("rows") or len(rows) != manifest.get("actorDecisionRows"):
        raise ValueError("strategy-probe corpus row count mismatch")
    source_games = manifest.get("sourceGames")
    if not isinstance(source_games, list) or len(source_games) != manifest.get("sourceGameCount"):
        raise ValueError("strategy-probe source game list mismatch")
    if any(not isinstance(item, dict) or not isinstance(item.get("replayId"), str)
           or type(item.get("actorDecisionRows")) is not int or item["actorDecisionRows"] <= 0
           or item.get("split") != "heldout" or item.get("status") != "finished"
           for item in source_games):
        raise ValueError("strategy-probe source game metadata is invalid")
    expected_games = {item["replayId"]: item for item in source_games}
    if len(expected_games) != len(source_games):
        raise ValueError("strategy-probe source game list contains duplicate IDs")
    counts = Counter(row.get("sourceGameId") for row in rows)
    if set(counts) != set(expected_games) or any(counts[key] != expected_games[key]["actorDecisionRows"]
                                                for key in expected_games):
        raise ValueError("strategy-probe corpus does not contain every frozen actor decision")
    allowed_row_fields = {"positionHash", "sourceGameId", "sourceDecisionIndex", "actor",
                          "gameSideKey", "familyId", "split", "featureIdentityHash",
                          "observation", "tracker"}
    last_index_by_game: dict[str, int] = {}
    for row in rows:
        source_game = expected_games.get(row.get("sourceGameId"))
        decision_index = row.get("sourceDecisionIndex")
        actor = row.get("actor")
        if (set(row) != allowed_row_fields or source_game is None or row.get("split") != "heldout"
                or type(decision_index) is not int or decision_index <= last_index_by_game.get(row["sourceGameId"], -1)
                or type(actor) is not int or actor not in (0, 1)
                or row.get("gameSideKey") != f"{row['sourceGameId']}:{actor}"
                or row.get("familyId") != source_game.get("familyId")
                or not isinstance(row.get("observation"), dict)
                or row["observation"].get("playerId") != actor
                or legacy_digest(row["observation"]) != row.get("positionHash")):
            raise ValueError("strategy-probe row violates heldout actor-only unlabeled contract")
        last_index_by_game[row["sourceGameId"]] = decision_index
    return manifest, rows


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
        if target_deck != "all" and not any(
                str(deck).replace("-training", "") == target_deck for deck in item.get("decks", [])):
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
            if ((target_deck != "all" and own_deck != target_deck)
                    or observation.get("prompt") or observation.get("searchUnavailableReason")
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
                "familyId": f"{own_deck}-macro-plan:{item['familyId']}",
                "sourceGameId": replay["id"], "sourceDecisionIndex": frame["decisionIndex"], "actor": actor,
                "targetDeck": own_deck,
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

    # Balance matchup, policy family, stage, and source-game representation.
    selected = _select_macro_positions(candidates, limit)
    game_rows = defaultdict(list)
    for row in selected:
        game_rows[row["sourceGameId"]].append(row)
    game_splits = _assign_source_game_splits(game_rows)
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
                "poolBuilderVersion": "game-balanced-selection-v5-target-deck-round-robin",
                "selection": "deterministic round-robin by target deck, then matchup/policy/stage/source-game; stable row-balanced source-game assignments with matchup stratification and disjoint splits",
                "splitQuotasBySourceGame": dict(sorted(Counter(game_splits.values()).items()))}
    manifest["manifestHash"] = identity_hash(manifest)
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest
