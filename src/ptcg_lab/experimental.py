"""Quarantined learning from explicitly authorized, unverified simulations.

Nothing here changes trusted replay admission or publishes a champion. Inputs are
pinned by content hash before training, and every descendant retains its tier.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import demonstration_examples, demonstration_manifest, demonstration_records, family_key, family_partition
from .features import ACTION_DIM, FEATURE_NAMES, FEATURE_VERSION, action_features, card_tokens, resource_features
from .storage import Store, digest, file_digest, terminal_score

TEACHING_CATEGORIES = ("teaching", "teaching-families", "teaching-reviews", "fixture-receipts", "teaching-audits")
MAX_ROWS = 20000
MAX_ROWS_BYTES = 64 * 1024**2


class InsufficientExperimentalData(ValueError):
    """Collect another bounded batch; no completed train-partition inputs yet."""


def _immutable_put(store, category, identifier, value):
    path = store.location(category, identifier)
    if path.exists():
        if store.get(category, identifier) != value:
            raise ValueError("Immutable experimental artifact already exists with different content")
        return
    store.put(category, identifier, value)


def _checked_manifest(store, category, identifier):
    result = store.get(category, identifier)
    if digest({key: value for key, value in result.items() if key != "hash"}) != result.get("hash"):
        raise ValueError("Pinned experimental manifest checksum differs")
    return result


class _BoundedTeachingSource:
    """Bound all source reads before JSON allocation, including admission scans."""
    def __init__(self, store):
        self.store, self.seen, self.total = store, set(), 0

    def get(self, category, identifier):
        path = self.store.location(category, identifier)
        if path not in self.seen:
            self.total += path.stat().st_size
            if self.total > MAX_ROWS_BYTES:
                raise ValueError("Teaching source history exceeds 64 MiB; archive old teaching evidence before snapshotting")
            self.seen.add(path)
        return self.store.get(category, identifier)

    def iter_records(self, category):
        directory = self.store.path / category
        for path in sorted(directory.glob("*.json"), key=lambda value: value.stat().st_mtime, reverse=True):
            yield self.get(category, path.stem)

    def put(self, *args):
        return self.store.put(*args)


def snapshot_teaching(source_store, experimental_store, snapshot_id: str) -> dict:
    """Freeze effective decisions and their referenced original review/audit evidence."""
    if experimental_store.location("teaching-snapshots", snapshot_id).exists():
        return _checked_manifest(experimental_store, "teaching-snapshots", snapshot_id)
    bounded = _BoundedTeachingSource(source_store)
    records = demonstration_records(bounded, limit=1000)
    required = {category: set() for category in TEACHING_CATEGORIES}
    for record in records:
        required["teaching"].add(record["id"])
        required["teaching-families"].add(digest({"teachingFamily": record["familyId"]}))
        required["teaching-reviews"].add(record["reviewHash"])
        if record.get("fixtureReceiptHash"):
            required["fixture-receipts"].add(record["id"])
        if record.get("mechanicsAttestationHash"):
            required["teaching-audits"].add(record["mechanicsAttestationHash"])
    categories = {category: {identifier: bounded.get(category, identifier) for identifier in sorted(identifiers)}
                  for category, identifiers in required.items()}
    result = {"schemaVersion": 1, "id": snapshot_id, "dataTier": "experimental",
              "records": copy.deepcopy(records), "manifest": demonstration_manifest(records), "categories": categories}
    if len(json.dumps(result)) > MAX_ROWS_BYTES:
        raise ValueError("Teaching snapshot exceeds the bounded local input size")
    result["hash"] = digest(result)
    if demonstration_manifest(demonstration_records(_PinnedTeachingStore(experimental_store, result))) != result["manifest"]:
        raise ValueError("Teaching evidence changed while pinning its reviewed snapshot")
    _immutable_put(experimental_store, "teaching-snapshots", snapshot_id, result)
    return result


class _PinnedTeachingStore:
    """Read teaching only from the snapshot; forward model/experiment writes."""
    def __init__(self, store, snapshot):
        self.store, self.categories = store, snapshot["categories"]
        self.path, self.max_bytes, self.min_free_bytes = store.path, store.max_bytes, store.min_free_bytes

    def location(self, *args):
        return self.store.location(*args)

    def get(self, category, identifier):
        if category in self.categories:
            try:
                return copy.deepcopy(self.categories[category][identifier])
            except KeyError as exc:
                raise FileNotFoundError(identifier) from exc
        return self.store.get(category, identifier)

    def iter_records(self, category):
        if category in self.categories:
            return iter(copy.deepcopy(list(self.categories[category].values())))
        return self.store.iter_records(category)

    def put(self, category, identifier, value):
        if category in self.categories:
            raise ValueError("Pinned teaching evidence is immutable")
        return self.store.put(category, identifier, value)


def bootstrap_policy(store, *, snapshot_id: str, run_id: str, guard=None) -> dict:
    from .training import train_policy
    snapshot = _checked_manifest(store, "teaching-snapshots", snapshot_id)
    store.location("experiments", run_id)
    if not snapshot["records"]:
        raise ValueError("No admitted reviewed examples in the pinned teaching snapshot")
    pinned = _PinnedTeachingStore(store, snapshot)
    if demonstration_manifest(demonstration_records(pinned)) != snapshot["manifest"]:
        raise ValueError("Pinned teaching admission differs from the captured audit/review evidence")
    path = store.path / "models" / f"{run_id}.pt"
    report = train_policy(pinned, epochs=20, seed=42, resume=path if path.exists() else None,
                          run_id=run_id, data_tier="experimental", guard=guard)
    report.update(teachingSnapshotId=snapshot_id, teachingSnapshotHash=snapshot["hash"])
    store.put("experiments", run_id, report)
    return report


def position_group(observation: dict) -> str:
    """A recurrence key for sampling, never authority to transfer action labels."""
    players = []
    for player in observation.get("players", []):
        board = [player.get("active"), *player.get("bench", [])]
        players.append({"id": player["id"], "prizes": player.get("prizesRemaining"),
                        "board": [{"card": item["card"]["id"], "damageBand": item.get("damage", 0) // 30,
                                   "energy": sorted(item.get("energy", [])), "conditions": item.get("conditions", [])}
                                  for item in board if item],
                        "handCountBand": player.get("handCount", 0) // 3})
    return digest({"perspective": observation["playerId"], "players": players,
                   "turnBand": observation.get("turn", 0) // 4, "phase": observation.get("phase"),
                   "prompt": (observation.get("prompt") or {}).get("type"), "knowledge": observation.get("knowledge", [])})


def experimental_partition(game: dict) -> str:
    if game.get("experimentalComparison") or game.get("humanMatch"):
        return "benchmark"
    return family_partition(game)


def _admit_replay(game):
    roles = game.get("deckRoles", [])
    if (game.get("dataTier") != "experimental" or game.get("experimentalLearning") is not True
            or game.get("trainingEligible") is not False or len(roles) != 2
            or any(role not in {"main", "training-variant"} for role in roles)
            or any(key in game for key in ("evaluationExperiment", "benchmarkMatch", "humanMatch", "experimentalComparison"))):
        raise ValueError("Replay is not explicitly admitted experimental training data")
    if game.get("status") != "finished":
        if game.get("outcome") is not None:
            raise ValueError("Incomplete experimental game carries an invalid outcome")
        return False
    terminal_score(game, 0)
    if game["outcome"].get("reason") not in {"rules-terminal", "rules-draw"}:
        raise ValueError("Experimental targets require a genuine simulator terminal")
    if not game.get("engineVersion") or not game.get("deckHashes") or not game.get("learningRun"):
        raise ValueError("Experimental replay lacks engine/deck/run provenance")
    return True


def _frame_row(game, frame, additional_targets=()):
    actor = frame["actor"]
    if type(actor) is not int or actor not in (0, 1):
        raise ValueError("Experimental decision actor is invalid")
    observation = frame["observations"][actor]
    if observation.get("playerId") != actor or observation.get("decisionPlayer") != actor:
        raise ValueError("Experimental observation belongs to another player")
    legal = observation.get("legalActions", [])
    if not legal or len(legal) > 2048 or not frame.get("action") or frame["action"]["id"] not in {a["id"] for a in legal}:
        return None
    winner = game["outcome"]["winner"]
    row = {"gameId": game["id"], "decisionIndex": frame["decisionIndex"], "actor": actor,
           "observationHash": digest(observation), "positionGroup": position_group(observation),
           "resources": resource_features(observation).tolist(), "cards": card_tokens(observation).tolist(),
           "actions": [action_features(action).tolist() for action in legal], "expectedResult": terminal_score(game, actor),
           "outcomeClass": 1 if winner is None else (2 if winner == actor else 0),
           "policyDistribution": None, "policyTargetSource": "none"}
    preferred = [item for item in additional_targets if item.get("dataTier") == "experimental"
                 and item.get("status") == "supported" and item.get("source") == "reanalysis"
                 and item.get("independentSeeds") == 2 and item.get("replayId") == game["id"]]
    target = next((item for item in [*preferred, *game.get("searchTargets", [])] if item.get("decisionIndex") == frame["decisionIndex"]
                   and item.get("actor") == actor and item.get("observationHash") == row["observationHash"]), None)
    if target:
        probabilities = target.get("probabilities", {})
        weights = [float(probabilities.get(action["id"], 0)) for action in legal]
        if (set(probabilities) == {action["id"] for action in legal}
                and all(math.isfinite(value) and value >= 0 for value in weights) and abs(sum(weights) - 1) < 1e-5):
            row.update(policyDistribution=weights, policyTargetSource="search-distillation")
            if target in preferred:
                row.update(searchTargetId=target["id"], searchTargetHash=digest(target))
    return row


def prepare_dataset(store, replay_ids: list[str], *, teaching_snapshot_id: str | None = None,
                    seed: int = 42, max_rows: int = MAX_ROWS) -> dict:
    """Pin specified inputs; select bounded matchup/prompt/group-stratified rows."""
    if (not 1 <= max_rows <= MAX_ROWS or not 1 <= len(replay_ids) <= 2000 or len(set(replay_ids)) != len(replay_ids)
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Provide 1..2000 unique replay IDs and 1..20000 training rows")
    snapshot = _checked_manifest(store, "teaching-snapshots", teaching_snapshot_id) if teaching_snapshot_id else None
    buckets, inputs, skipped, excluded = defaultdict(list), [], [], []
    source_bytes = 0
    for identifier in sorted(replay_ids):
        source_bytes += store.location("replays", identifier).stat().st_size
        if source_bytes > 256 * 1024**2:
            raise ValueError("Pinned replay inputs exceed 256 MiB; select a smaller bounded buffer")
        game = store.get("replays", identifier)
        if not _admit_replay(game):
            skipped.append(identifier)
            continue
        partition = experimental_partition(game)
        key = family_key(game)
        _immutable_put(store, "experimental-partitions", key, {"id": key, "partition": partition, "dataTier": "experimental"})
        if partition != "train":
            excluded.append({"id": identifier, "partition": partition})
            continue
        if game.get("partition", "train") != "train":
            raise ValueError("Explicit experimental replay partition conflicts with immutable family assignment")
        inputs.append({"id": identifier, "hash": digest(game), "familyId": family_key(game), "seed": game["seed"],
                       "decks": game["decks"], "deckHashes": game["deckHashes"], "engineVersion": game["engineVersion"]})
        seen_groups = defaultdict(int)
        indices = list(range(len(game.get("frames", []))))
        random.Random(digest({"seed": seed, "game": identifier})).shuffle(indices)
        for index in indices[:256]:
            frame = game["frames"][index]
            if frame.get("action") is None:
                continue
            actor = frame["actor"]
            observation = frame["observations"][actor]
            group = position_group(observation)
            if seen_groups[group] >= 4:
                continue
            seen_groups[group] += 1
            context = (game["decks"][actor], game["decks"][1 - actor], (observation.get("prompt") or {}).get("type", observation.get("phase", "unknown")))
            bucket = buckets[context]
            if len(bucket) < 1024:
                bucket.append({"gameId": identifier, "frameIndex": index, "positionGroup": group})
    if not inputs:
        raise InsufficientExperimentalData("No completed experimental games; collect another bounded batch")
    rng = random.Random(seed)
    contexts = sorted(buckets)
    rng.shuffle(contexts)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    demo_records = snapshot["records"] if snapshot else []
    reserved = max_rows // 10 if demo_records else 0
    selected = []
    while len(selected) < max_rows - reserved and contexts:
        remaining = []
        for context in contexts:
            if buckets[context] and len(selected) < max_rows - reserved:
                selected.append(buckets[context].pop())
            if buckets[context]:
                remaining.append(context)
        contexts = remaining
    by_game = defaultdict(list)
    for item in selected:
        by_game[item["gameId"]].append(item)
    rows, selected_refs, rows_bytes = [], [], 2
    extra_by_game = defaultdict(list)
    for target in store.iter_records("search-targets"):
        if target.get("replayId") in by_game:
            extra_by_game[target["replayId"]].append(target)
    for targets in extra_by_game.values():
        targets.sort(key=lambda target: target["id"])
    for identifier in sorted(by_game):
        game = store.get("replays", identifier)
        if digest(game) != next(item["hash"] for item in inputs if item["id"] == identifier):
            raise ValueError("Replay changed while pinning experimental data")
        for item in by_game[identifier]:
            row = _frame_row(game, game["frames"][item["frameIndex"]], extra_by_game[identifier])
            if row is not None:
                rows_bytes += len(json.dumps(row)) + 1
                if rows_bytes > MAX_ROWS_BYTES:
                    raise ValueError("Feature rows exceed 64 MiB; reduce the row budget")
                rows.append(row)
                selected_refs.append(item)
    if not rows:
        raise ValueError("No valid legal decisions in completed experimental inputs")
    demonstration_count = min(reserved, len(rows) // 9)
    anchors = demonstration_examples(demo_records)
    for index in range(demonstration_count):
        row = copy.deepcopy(anchors[index % len(anchors)])
        rows_bytes += len(json.dumps(row)) + 1
        if rows_bytes > MAX_ROWS_BYTES:
            raise ValueError("Feature rows exceed 64 MiB; reduce the row budget")
        rows.append(row)
    manifest = {"schemaVersion": 1, "dataTier": "experimental", "featureVersion": FEATURE_VERSION,
                "seed": seed, "maxRows": max_rows, "replays": inputs, "skippedIncomplete": skipped, "excludedPartitions": excluded,
                "teachingSnapshotId": teaching_snapshot_id, "teachingSnapshotHash": snapshot["hash"] if snapshot else None,
                "teachingManifest": snapshot["manifest"] if snapshot else [], "selected": selected_refs,
                "rowsHash": digest(rows), "rows": len(rows), "outcomeRows": len(rows) - demonstration_count,
                "demonstrationRows": demonstration_count,
                "searchRows": sum(row["policyTargetSource"] == "search-distillation" for row in rows),
                "searchTargets": [{"id": identifier, "hash": fingerprint} for identifier, fingerprint in sorted({
                    (row["searchTargetId"], row["searchTargetHash"]) for row in rows if row.get("searchTargetId")})],
                "sampling": "matchup/prompt round-robin; <=256 candidate frames/game; <=4 repeated groups/game; demos <=10%"}
    identifier = digest(manifest)
    manifest.update(id=identifier)
    manifest["hash"] = digest(manifest)
    _immutable_put(store, "dataset-rows", identifier, rows)
    _immutable_put(store, "datasets", identifier, manifest)
    return manifest


def train_experimental(store, *, dataset_id: str, run_id: str, seed: int = 42,
                       checkpoint: Path | None = None, resume: bool = False, guard=None) -> dict:
    from .training import _torch, _batch, _save_checkpoint, policy_loss, load_model, checkpoint_lineage
    from .model import PolicyResourceModel
    torch = _torch()
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Experimental seed must be uint32")
    store.location("experiments", run_id)
    manifest = _checked_manifest(store, "datasets", dataset_id)
    rows = store.get("dataset-rows", dataset_id)
    if (manifest.get("dataTier") != "experimental" or manifest.get("featureVersion") != FEATURE_VERSION
            or len(rows) != manifest["rows"] or digest(rows) != manifest["rowsHash"]):
        raise ValueError("Pinned experimental dataset is incompatible or changed")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = PolicyResourceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    config = {"seed": seed, "device": "cpu", "epochs": 1, "batchSize": 32, "learningRate": .001,
              "features": list(FEATURE_NAMES), "featureVersion": FEATURE_VERSION, "actionDimensions": ACTION_DIM,
              "datasetHash": digest(manifest), "eligibilityPolicy": "experimental-simulator-v1", "dataTier": "experimental"}
    start, parent, previous_lineage, saved_history = 0, None, {}, []
    if resume and checkpoint is None:
        raise ValueError("Resume requires the saved experimental checkpoint")
    if checkpoint:
        previous_model, previous = load_model(checkpoint, allow_experimental=True)
        model.load_state_dict(previous_model.state_dict())
        previous_lineage = checkpoint_lineage(previous)
        parent = {"experimentId": previous["experimentId"], "checkpointHash": file_digest(checkpoint),
                  "dataTier": previous.get("dataTier", "verified"), "mode": "warm-start"}
        if resume:
            if previous.get("experimentId") != run_id or previous["config"] != config:
                raise ValueError("Experimental resume requires the exact run, pinned data, and configuration")
            optimizer.load_state_dict(previous["optimizer"])
            start, parent = previous.get("nextBatch", 0), previous.get("parent")
            saved_history = previous.get("history", [])
    elif (store.path / "models" / f"{run_id}.pt").exists():
        raise ValueError("Experimental run already has a checkpoint; explicitly resume it")
    if type(start) is not int or start < 0 or start > len(rows) or (start != len(rows) and start % 32):
        raise ValueError("Saved experimental batch cursor is invalid")
    seeds = {item["seed"] for item in manifest["replays"]} | set(previous_lineage.get("seenSeeds", []))
    families = {item["familyId"] for item in manifest["replays"]} | set(previous_lineage.get("familyIds", []))
    teaching_families = {item["familyId"] for item in manifest["teachingManifest"]} | set(previous_lineage.get("teachingFamilyIds", []))
    teaching_reviews = {item["reviewHash"] for item in manifest["teachingManifest"]} | set(previous_lineage.get("teachingReviewHashes", []))
    families.update(digest({"teachingFamily": family}) for family in teaching_families)
    lineage = {"schemaVersion": 2, "seenSeeds": sorted(seeds), "familyIds": sorted(families),
               "teachingFamilyIds": sorted(teaching_families), "teachingReviewHashes": sorted(teaching_reviews),
               "experimentalAncestry": True}
    lineage["hash"] = digest(lineage)
    path = store.path / "models" / f"{run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 2_000_000:
        raise ValueError("Experimental model exceeds the local parameter bound")
    cursor, history = start, list(saved_history)

    def payload():
        return {"schemaVersion": 4, "experimentId": run_id, "dataTier": "experimental", "modelKind": "policy-value",
                "valueTrained": True, "config": config, "manifest": manifest, "dataLineage": lineage,
                "nextBatch": cursor, "epoch": int(cursor >= len(rows)), "parent": parent, "parameters": parameters,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "calibration": None,
                "history": history, "policyTarget": "Reviewed anchors and explicit search distributions only; unlabeled moves have no policy loss"}

    _save_checkpoint(store, path, payload())
    store.put("experiments", run_id, {"id": run_id, "status": "training", "type": "experimental-training",
                                      "dataTier": "experimental", "config": config, "checkpoint": str(path), "nextBatch": cursor})
    try:
        order = np.random.default_rng(seed).permutation(len(rows))
        last_saved = time.monotonic()
        model.train()
        for offset in range(start, len(order), 32):
            if guard:
                guard.check(storage=offset % 1024 == 0)
            batch = [rows[int(index)] for index in order[offset:offset + 32]]
            resources, cards, actions, mask, targets, classes, _ = _batch(batch, "cpu")
            result = model(resources, cards, actions)
            outcome = torch.tensor(["expectedResult" in row for row in batch])
            supervised = torch.tensor([row["policyTargetSource"] in {"reviewed-demonstration", "search-distillation"} for row in batch])
            loss = result["logit"].sum() * 0
            if outcome.any():
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(result["logit"][outcome], targets[outcome])
                loss = loss + .5 * torch.nn.functional.cross_entropy(result["outcome_logits"][outcome], classes[outcome])
            if supervised.any():
                loss = loss + .25 * policy_loss(result["policy"][supervised], mask[supervised],
                                               [row for row, selected in zip(batch, supervised.tolist()) if selected])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            cursor = min(len(rows), offset + 32)
            history.append({"throughRow": cursor, "loss": float(loss.detach())})
            if cursor % 1024 == 0 or time.monotonic() - last_saved >= 30:
                _save_checkpoint(store, path, payload())
                last_saved = time.monotonic()
        _save_checkpoint(store, path, payload())
        report = {"id": run_id, "status": "completed", "type": "experimental-training", "dataTier": "experimental",
                  "modelKind": "policy-value", "valueTrained": True, "config": config, "checkpoint": str(path),
                  "modelHash": file_digest(path), "dataLineageHash": lineage["hash"], "parameters": parameters,
                  "datasetId": dataset_id, "rows": len(rows), "outcomeRows": manifest["outcomeRows"],
                  "demonstrationExamples": manifest["demonstrationRows"], "searchExamples": manifest["searchRows"],
                  "history": history, "parent": parent, "promotion": "Experimental candidate only; trusted champion promotion is prohibited"}
        store.put("experiments", run_id, report)
        return report
    except (KeyboardInterrupt, RuntimeError, OSError) as exc:
        try:
            _save_checkpoint(store, path, payload())
            store.put("experiments", run_id, {"id": run_id, "status": "paused", "type": "experimental-training",
                                              "dataTier": "experimental", "config": config, "checkpoint": str(path),
                                              "nextBatch": cursor, "pauseReason": str(exc) or "Interrupted"})
        except (RuntimeError, OSError):
            pass
        raise


def tactical_regressions(store, *, snapshot_id: str, candidate: Path, incumbent: Path | str) -> dict:
    from .training import load_model, _teaching_policy_snapshot
    from .features import heuristic_action_score
    snapshot = _checked_manifest(store, "teaching-snapshots", snapshot_id)
    records = snapshot["records"][:100]
    if str(incumbent) == "heuristic":
        before = []
        for record in records:
            observation = record["observation"]
            legal = observation["legalActions"]
            scores = [heuristic_action_score(action, observation) for action in legal]
            # A tie is acceptable only when every possible heuristic choice is acceptable.
            best = max(scores)
            choices = [action["id"] for action, score in zip(legal, scores) if score == best]
            acceptable = set(record["acceptableActionIds"])
            before.append({"teachingId": record["id"], "observationHash": record["positionHash"],
                           "reviewHash": record["reviewHash"],
                           "selectedIsAcceptable": all(choice in acceptable for choice in choices)})
    else:
        before = _teaching_policy_snapshot(load_model(Path(incumbent), allow_experimental=True)[0], records, "cpu")
    after = _teaching_policy_snapshot(load_model(Path(candidate), allow_experimental=True)[0], records, "cpu")
    regressions = [{"teachingId": old["teachingId"], "observationHash": old["observationHash"], "reviewHash": old["reviewHash"]}
                   for old, new in zip(before, after) if old["selectedIsAcceptable"] and not new["selectedIsAcceptable"]]
    return {"status": "measured" if records else "unavailable", "positions": len(records), "regressions": regressions,
            "candidateAccuracy": sum(item["selectedIsAcceptable"] for item in after) / len(after) if after else None,
            "passes": bool(records) and not regressions,
            "description": "Reviewed training-position retention diagnostic, not held-out playing strength or verified strategy discovery"}


# Backward-compatible singular spelling for callers written before the runner contract.
tactical_regression = tactical_regressions
