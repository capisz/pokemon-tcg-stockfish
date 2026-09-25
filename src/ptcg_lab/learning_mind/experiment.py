from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from contextlib import closing
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ptcg_lab.engine import EngineClient, EnginePool
from ptcg_lab.features import heuristic_action_score

from .dataset_v1 import file_sha256, load_dataset, training_records
from .encoding import encode_decision
from .macro import (CANDIDATE_GENERATOR_VERSION, candidates_from_transition_plans,
                    label_candidates, rollout_seed)
from .model import StrategyTransformerV1
from .ranker import FrozenIteration, XGBoostMacroRanker, holdout_splits
from .sampling import select_stratified_rows
from .schema import IdentityManifest, UnsupportedPosition, identity_hash
from .training import require_checkpoint_identity, train_supervised


def transition_generator_identity(root: Path) -> dict:
    planner = root / "research/learning_mind/transition_macro_planner.ts"
    planner_worker = root / "research/learning_mind/planner_worker.ts"
    planner_bundle = root / "packages/engine/dist/learning-mind-planner.cjs"
    action_key = root / "packages/engine/src/action-key.ts"
    adapter = Path(__file__).with_name("macro.py")
    if not all(path.is_file() for path in (planner, planner_worker, planner_bundle, action_key)):
        raise FileNotFoundError("transition planner bundle missing; run npm run engine:build before macro collection")
    return {"version": CANDIDATE_GENERATOR_VERSION,
            "plannerSha256": file_sha256(planner), "actionKeySha256": file_sha256(action_key),
            "plannerWorkerSha256": file_sha256(planner_worker),
            "plannerBundleSha256": file_sha256(planner_bundle),
            "adapterSha256": file_sha256(adapter)}


def generate_transition_candidates(root: Path, observation: dict, seed: int):
    planner = root / "packages/engine/dist/learning-mind-planner.cjs"
    if not planner.is_file():
        raise FileNotFoundError("transition planner bundle missing; run npm run engine:build before macro collection")
    result = subprocess.run(["node", str(planner)], cwd=root,
        input=json.dumps({"observation": observation, "seed": seed}, separators=(",", ":")),
        text=True, capture_output=True, timeout=180, check=False)
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[-2000:]
        if "unsupported position:" in message.lower():
            raise UnsupportedPosition(message)
        if "belief pool does not match public zone counts" in message.lower():
            raise UnsupportedPosition(
                "unsupported position: actor-visible deck hypothesis cannot reconcile public zone counts")
        raise RuntimeError(f"transition macro planner failed ({result.returncode}): {message}")
    response = json.loads(result.stdout)
    if response.get("version") != CANDIDATE_GENERATOR_VERSION:
        raise ValueError("transition macro planner version mismatch")
    return candidates_from_transition_plans(response.get("candidates", [])), response


def runtime_identity(root: Path) -> IdentityManifest:
    deck_paths = sorted((root / "decks").glob("*.json"))
    decks = [{"path": str(path.relative_to(root)), "sha256": file_sha256(path)} for path in deck_paths]
    cards = []
    for path in deck_paths:
        data = json.loads(path.read_text())
        cards.extend({key: card.get(key) for key in ("cardId", "name", "count", "engineName")}
                     for card in data.get("cards", []))
    with EngineClient(root) as engine:
        health = engine.request("health")
    return IdentityManifest.create(engine_build_hash=health["engineBuildHash"], deck_manifests=decks,
                                   card_metadata=cards, extra_schema={"model": "StrategyTransformerV1"})


def candidate_features(observation: dict, candidate: dict) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    own = next(player for player in observation["players"] if player["id"] == observation["playerId"])
    other = next(player for player in observation["players"] if player["id"] != observation["playerId"])
    vector[:12] = [min(observation.get("turn", 0) / 50, 1), own.get("handCount", 0) / 20,
                   other.get("handCount", 0) / 20, own.get("prizesRemaining", 6) / 6,
                   other.get("prizesRemaining", 6) / 6, len(own.get("bench", [])) / 5,
                   len(other.get("bench", [])) / 5, len(candidate.get("action_ids") or candidate.get("actionIds") or []) / 5,
                   1 if candidate.get("intended_attack") else 0, 1 if candidate.get("supporter") else 0,
                   1 if candidate.get("disruption_intent") else 0, 1 if candidate.get("energy_destination") else 0]
    text = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    for token in text.lower().replace('"', ' ').replace(':', ' ').replace(',', ' ').split():
        digest = hashlib.sha256(token.encode()).digest()
        vector[12 + digest[0] % 116] += 1 if digest[1] & 1 else -1
    norm = np.linalg.norm(vector[12:])
    if norm: vector[12:] /= norm
    return vector


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    with temporary.open("rb") as source: os.fsync(source.fileno())
    temporary.replace(path)


def collect_macro_labels(*, root: Path, dataset_dir: Path, output: Path, identity: dict,
                         limit: int = 20, initial: int = 16, maximum: int = 64,
                         extension_batch_size: int = 8,
                         horizon: int = 16, rollout_budget_ms: int = 1000,
                         rollout_workers: int = 1,
                         position_hash: str | None = None,
                         position_hashes: list[str] | None = None,
                         split: str | None = None) -> dict:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 500:
        raise ValueError("macro rollout horizon must be an integer from 1 to 500")
    if split is not None and split not in {"train", "development", "heldout"}:
        raise ValueError("macro label split must be train, development, or heldout")
    if not isinstance(rollout_budget_ms, int) or isinstance(rollout_budget_ms, bool) or not 1 <= rollout_budget_ms <= 120000:
        raise ValueError("rollout budget must be an integer from 1 to 120000 milliseconds")
    if not isinstance(rollout_workers, int) or isinstance(rollout_workers, bool) or not 1 <= rollout_workers <= 8:
        raise ValueError("rollout workers must be an integer from 1 to 8")
    manifest, rows = load_dataset(dataset_dir, identity=identity)
    if position_hash is not None and position_hashes:
        raise ValueError("use either position_hash or position_hashes, not both")
    requested_hashes = position_hashes or ([position_hash] if position_hash is not None else None)
    if requested_hashes is not None:
        if not requested_hashes or any(not isinstance(value, str) or not value for value in requested_hashes):
            raise ValueError("position_hashes must contain one or more nonempty position hashes")
        if len(requested_hashes) != len(set(requested_hashes)):
            raise ValueError("position_hashes must not contain duplicates")
        by_hash = {row["positionHash"]: row for row in rows}
        missing = [key for key in requested_hashes if key not in by_hash]
        if missing:
            raise ValueError(f"requested macro positions are not present in the frozen dataset: {missing}")
        selected = [by_hash[key] for key in requested_hashes]
        if split is not None:
            outside = [row["positionHash"] for row in selected if row.get("split") != split]
            if outside:
                raise ValueError(f"requested macro positions are not in the requested {split} split: {outside}")
    else:
        candidates = [row for row in rows if split is None or row.get("split") == split]
        if not candidates:
            raise ValueError(f"frozen dataset has no positions in the requested {split} split")
        selected = (select_stratified_rows(candidates, limit)
                    if split is not None else candidates[:limit])
    output.mkdir(parents=True, exist_ok=True)
    generator_identity = transition_generator_identity(root)
    settings = {"identity": identity, "datasetManifestHash": manifest["manifestHash"],
                "requestedPositions": len(selected), "initialRollouts": initial,
                "maximumRollouts": maximum, "extensionBatchSize": extension_batch_size,
                "horizon": horizon,
                "rolloutBudgetMs": rollout_budget_ms,
                "rolloutWorkers": rollout_workers,
                "splitFilter": split,
                "selectionMethod": "position-hash-list" if position_hashes else
                    "position-hash" if position_hash is not None else
                    "deterministic-stratified-v2-target-deck" if split is not None else "dataset-order",
                "selectedPositionHashes": [row["positionHash"] for row in selected],
                "candidateGeneratorVersion": CANDIDATE_GENERATOR_VERSION,
                "candidateGeneratorIdentity": generator_identity,
                "rolloutSeedVersion": "configuration-bound-v1",
                "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
                "labelCollectorVersion": "macro-rollout-labeler-v6",
                "labelCollectorSha256": file_sha256(Path(__file__))}
    rollout_identity = identity_hash(settings)
    settings["rolloutIdentity"] = rollout_identity
    manifest_path = output / "manifest.json"
    completed = set()
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        changed = [key for key, value in settings.items() if prior.get(key) != value]
        if changed:
            raise ValueError(f"macro-label resume identity/configuration drift: {', '.join(changed)}")
        expected_files = {item["path"] for item in prior.get("files", [])}
        actual_files = {path.name for path in output.glob("*.json") if path.name != "manifest.json"}
        if actual_files != expected_files:
            raise ValueError("macro-label checkpoint contains unexpected or missing position files")
        for item in prior.get("files", []):
            path = output / item["path"]
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                raise ValueError(f"macro-label checkpoint hash mismatch: {item['path']}")
            completed.add(path.stem)
    else:
        for path in output.glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("identity") != identity or record.get("datasetManifestHash") != manifest["manifestHash"]:
                raise ValueError(f"unpublished macro-label checkpoint identity drift: {path.name}")
            completed.add(path.stem)
    with closing(EnginePool(root, size=rollout_workers, timeout=300)) as engine_pool:
        for row in selected:
            key = row["positionHash"]
            if key in completed: continue
            observation = row["observation"]
            legal = {str(action["id"]): action for action in observation.get("legalActions", [])}
            generator_seed = int.from_bytes(hashlib.sha256(
                f"learning-mind-v1|macro-generator|{key}".encode()).digest()[:4], "big")
            try:
                candidates, generation = generate_transition_candidates(root, observation, generator_seed)
            except UnsupportedPosition as error:
                namespace = "training" if row["split"] == "train" else "development"
                record = {"schemaVersion": 1, "positionHash": key, "split": row["split"],
                    "identity": identity, "datasetManifestHash": manifest["manifestHash"],
                    "familyId": row["familyId"], "opponentArchetype": row["opponentArchetype"],
                    "opponentPolicyFamily": row["opponentPolicyFamily"], "observation": observation,
                    "labels": [], "status": "unsupported", "unsupportedReason": str(error),
                    "seedNamespace": namespace, "generatorSeed": generator_seed,
                    "semantics": "transition-aware plan generation exceeded the declared supported bounds",
                    "highConfidencePolicyEligible": False}
                _atomic_json(output / f"{key}.json", record)
                continue
            executable = [candidate for candidate in candidates
                          if candidate.action_sequence and candidate.turn_intent in {"attack", "no-attack"}]
            if not executable:
                namespace = "training" if row["split"] == "train" else "development"
                record = {"schemaVersion": 1, "positionHash": key, "split": row["split"],
                    "identity": identity, "datasetManifestHash": manifest["manifestHash"],
                    "familyId": row["familyId"], "opponentArchetype": row["opponentArchetype"],
                    "opponentPolicyFamily": row["opponentPolicyFamily"], "observation": observation,
                    "labels": [], "status": "unsupported",
                    "unsupportedReason": "no complete attack or deliberate no-attack plan within the frozen action-depth bound",
                    "seedNamespace": namespace, "generatorSeed": generator_seed,
                    "semantics": "transition-aware candidate generation produced only incomplete prefixes",
                    "highConfidencePolicyEligible": False}
                _atomic_json(output / f"{key}.json", record)
                continue

            progress_path = output / f".{key}.progress"
            candidate_hashes = [candidate.key() for candidate in executable]
            progress_identity = {"positionHash": key, "identity": identity,
                "datasetManifestHash": manifest["manifestHash"], "generatorSeed": generator_seed,
                "generatorHypothesisId": generation.get("hypothesisId"),
                "candidateHashes": candidate_hashes, "initialRollouts": initial,
                "maximumRollouts": maximum, "horizon": horizon,
                "rolloutBudgetMs": rollout_budget_ms, "rolloutWorkers": rollout_workers,
                "rolloutIdentity": rollout_identity,
                "seedNamespace": "training" if row["split"] == "train" else "development",
                "candidateGeneratorIdentity": generator_identity}
            progress_identity_hash = identity_hash(progress_identity)
            resume_state = None
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                progress_hash = progress.pop("progressHash", None)
                if progress_hash != identity_hash(progress):
                    raise ValueError(f"macro rollout checkpoint hash mismatch: {progress_path.name}")
                if progress.get("identityHash") != progress_identity_hash:
                    raise ValueError(f"macro rollout checkpoint identity hash mismatch: {progress_path.name}")
                if progress.get("identity") != progress_identity:
                    raise ValueError(f"macro rollout checkpoint identity/configuration drift: {progress_path.name}")
                resume_state = progress.get("state")

            def save_progress(state):
                progress = {"schemaVersion": 1, "identityHash": progress_identity_hash,
                            "identity": progress_identity, "state": state}
                progress["progressHash"] = identity_hash(progress)
                _atomic_json(progress_path, progress)

            def rollout(candidate, seed):
                root_action = legal.get(candidate.action_ids[0])
                if root_action is None: return {"status": "error", "reason": "macro-root-no-longer-legal"}
                narrowed = copy.deepcopy(observation); narrowed["legalActions"] = [root_action]
                plan_actions = list(candidate.action_sequence)
                if not plan_actions or plan_actions[0].get("id") != root_action["id"]:
                    return {"status": "error", "reason": "macro-plan-root-mismatch"}
                with engine_pool.lease() as engine:
                    result = engine.request("search", {"observation": narrowed, "seed": int(seed) & 0xffffffff,
                        "budgetMs": rollout_budget_ms, "method": "rollout", "iterations": 1,
                        "maxRolloutDecisions": horizon, "macroPlanActions": plan_actions,
                        "researchHypothesisId": generation.get("hypothesisId"),
                        "researchDeterminizationSeed": generator_seed,
                        "researchMaxRolloutDecisions": horizon})
                alternative = next((item for item in result.get("alternatives", [])
                                    if item.get("actionId") == root_action["id"] and item.get("visits", 0) > 0), None)
                execution = result.get("macroPlanExecution") or {}
                if execution.get("requested") and not execution.get("completed"):
                    reason = (execution.get("failures") or [{"reason": "macro-plan-unexecuted"}])[0]["reason"]
                    return {"status": "error", "reason": reason}
                if result.get("status") != "complete":
                    warnings = result.get("warnings") or []
                    detail = warnings[0] if warnings and isinstance(warnings[0], str) else "no search warning supplied"
                    return {"status": "error",
                            "reason": f"search-result-{result.get('status', 'missing')}: {detail}"}
                if alternative is None:
                    return {"status": "error", "reason": "search-action-unvisited"}
                if alternative.get("score") is None:
                    return {"status": "error", "reason": "search-action-score-missing"}
                continuation = alternative.get("continuation") or {}
                if continuation.get("end") != "terminal":
                    cutoff = continuation.get("cutoffReason")
                    reason = "search-rollout-budget-cutoff" if cutoff == "budget" else "search-rollout-horizon-cutoff"
                    return {"status": "truncated", "reason": reason,
                            "decisionCount": continuation.get("decisionCount")}
                outcome = continuation.get("outcome")
                winner = outcome.get("winner") if isinstance(outcome, dict) else "invalid"
                if winner is not None and (type(winner) is not int or winner not in (0, 1)):
                    return {"status": "error", "reason": "terminal-rollout-missing-valid-outcome"}
                score = .5 if winner is None else 1. if winner == observation["playerId"] else 0.
                return {"status": "finished", "score": score,
                        "decisionCount": continuation.get("decisionCount")}

            namespace = "training" if row["split"] == "train" else "development"
            labels = label_candidates(executable, key, rollout, namespace=namespace,
                                      initial=initial, maximum=maximum,
                                      extension_batch_size=extension_batch_size,
                                      rollout_workers=rollout_workers,
                                      resume_state=resume_state, checkpoint=save_progress,
                                      rollout_identity=rollout_identity)
            record = {"schemaVersion": 1, "positionHash": key, "split": row["split"],
                      "identity": identity, "datasetManifestHash": manifest["manifestHash"],
                      "familyId": row["familyId"], "opponentArchetype": row["opponentArchetype"],
                      "opponentPolicyFamily": row["opponentPolicyFamily"],
                      "observation": observation, "labels": labels,
                      "status": "collected", "generatorSeed": generator_seed,
                      "generatorHypothesisId": generation.get("hypothesisId"),
                      "candidateCount": len(executable),
                      "rolloutBudgetMs": rollout_budget_ms,
                      "exploredPrefixCount": generation.get("exploredPrefixCount"),
                      "seedNamespace": namespace,
                      "rolloutIdentity": rollout_identity,
                      "rolloutSeeds": [rollout_seed(namespace, key, index, rollout_identity)
                                       for index in range(maximum)],
                      "semantics": "complete transition-aware attack or deliberate no-attack candidates only; incomplete traversal prefixes do not consume candidate cap or receive labels; later steps are revalidated at rollout",
                      "highConfidencePolicyEligible": False}
            _atomic_json(output / f"{key}.json", record)
            progress_path.unlink(missing_ok=True)
    files = sorted(path for path in output.glob("*.json") if path.name != "manifest.json")
    records = [json.loads(path.read_text()) for path in files]
    result = {"schemaVersion": 1, **settings, "positions": len(files),
              "supportedPositions": sum(record.get("status") == "collected" for record in records),
              "unsupportedPositions": sum(record.get("status") == "unsupported" for record in records),
              "files": [{"path": path.name, "sha256": file_sha256(path)} for path in files],
              "highConfidencePolicyLabels": 0, "resumePolicy": "verified position files are immutable"}
    result["manifestHash"] = identity_hash(result)
    _atomic_json(output / "manifest.json", result)
    return result


def _load_ranker_input(labels_dir: Path) -> tuple[dict, list[dict]]:
    """Verify immutable combined train/development labels before any model fit."""
    labels_dir = labels_dir.resolve()
    manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "combined-macro-label-runs-v1":
        raise ValueError("ranker input must be a verified combined macro-label manifest")
    recorded_hash = manifest.get("manifestHash")
    if recorded_hash != identity_hash({key: value for key, value in manifest.items()
                                        if key != "manifestHash"}):
        raise ValueError("combined ranker-input manifest hash mismatch")
    expected_families = ["python-heuristic", "typescript-heuristic"]
    if manifest.get("policyFamilies") != expected_families:
        raise ValueError("ranker input must contain both approved policy families")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != manifest.get("positions"):
        raise ValueError("combined ranker-input file list mismatch")
    identity = manifest.get("identity")
    seen_positions: set[str] = set()
    records = []
    observed_families = set()
    observed_splits = Counter()
    for item in files:
        name = item.get("path") if isinstance(item, dict) else None
        relative = Path(name) if isinstance(name, str) else None
        if (relative is None or relative.is_absolute() or ".." in relative.parts
                or not name.endswith(".json") or name == "manifest.json"):
            raise ValueError("unsafe macro-label record path in combined ranker input")
        source = (labels_dir / relative).resolve()
        if not source.is_relative_to(labels_dir) or not source.is_file():
            raise ValueError("macro-label record path escapes or is missing from ranker input")
        if file_sha256(source) != item.get("sha256"):
            raise ValueError(f"macro-label record hash mismatch: {relative}")
        record = json.loads(source.read_text())
        position_hash = record.get("positionHash")
        if (not isinstance(position_hash, str) or source.name != f"{position_hash}.json"
                or position_hash in seen_positions):
            raise ValueError("macro-label record position identity mismatch or duplicate")
        if record.get("identity") != identity:
            raise ValueError(f"macro-label record frozen identity mismatch: {relative}")
        if record.get("status") != "collected":
            raise ValueError(f"ranker input contains an unsupported macro-label record: {relative}")
        if record.get("split") not in {"train", "development"}:
            raise ValueError("ranker input may contain only train/development records")
        family = record.get("opponentPolicyFamily")
        if family not in expected_families:
            raise ValueError(f"ranker input has invalid policy-family provenance: {relative}")
        seen_positions.add(position_hash)
        observed_families.add(family)
        observed_splits[record["split"]] += 1
        records.append(record)
    if observed_families != set(expected_families):
        raise ValueError("ranker input records do not cover both approved policy families")
    if dict(sorted(observed_splits.items())) != manifest.get("positionsBySplit"):
        raise ValueError("ranker-input split counts do not match its records")
    return manifest, records


def fit_ranker(labels_dir: Path, output: Path, *, teacher_hash: str,
               opponent_policy_hash: str, iteration: int = 1) -> dict:
    output = output.resolve()
    manifest_output = output.with_suffix(".manifest.json")
    if output.exists() or manifest_output.exists():
        raise ValueError("macro ranker outputs are immutable; choose a new output path")
    manifest, records = _load_ranker_input(labels_dir)
    train = [record for record in records if record["split"] == "train"]
    if not train: raise ValueError("macro ranker has no training positions")
    def flatten(selected):
        features, labels, weights, groups, positions, metadata = [], [], [], [], [], []
        for record in selected:
            usable = [item for item in record["labels"] if item["expectedResult"] is not None]
            if len(usable) < 2:
                continue
            groups.append(len(usable)); positions.append(record["positionHash"])
            metadata.append({key: record[key] for key in
                             ("positionHash", "split", "opponentArchetype", "opponentPolicyFamily")})
            for item in usable:
                features.append(candidate_features(record["observation"], item["candidate"]))
                labels.append(item["relativeResult"]); weights.append(item["weight"])
        return features, labels, weights, groups, positions, metadata

    features, labels, weights, groups, positions, _ = flatten(train)
    if not groups: raise ValueError("macro ranker needs at least one position with two completed candidates")

    def metrics(model, selected):
        x, y, _, sizes, _, metadata = flatten(selected)
        if not sizes:
            return {"status": "insufficient", "positions": 0}
        predicted = model.predict(x); offset = 0; details = []
        for size, meta in zip(sizes, metadata):
            truth = np.asarray(y[offset:offset + size]); scores = predicted[offset:offset + size]
            chosen = int(np.argmax(scores)); best = int(np.argmax(truth))
            comparisons = correct = 0
            for left in range(size):
                for right in range(left + 1, size):
                    if truth[left] == truth[right]: continue
                    comparisons += 1
                    correct += int((scores[left] - scores[right]) * (truth[left] - truth[right]) > 0)
            details.append({**meta, "candidates": size,
                            "top1RelativeRegret": float(truth[best] - truth[chosen]),
                            "top3Recall": bool(best in np.argsort(scores)[-min(3, size):]),
                            "pairwiseCorrect": correct, "pairwiseComparisons": comparisons})
            offset += size
        by_archetype, by_family = {}, {}
        for field, destination in (("opponentArchetype", by_archetype),
                                   ("opponentPolicyFamily", by_family)):
            for value in sorted({item[field] for item in details}):
                subset = [item for item in details if item[field] == value]
                destination[value] = {"positions": len(subset),
                    "meanTop1RelativeRegret": float(np.mean([item["top1RelativeRegret"] for item in subset]))}
        pair_total = sum(item["pairwiseComparisons"] for item in details)
        return {"status": "measured", "positions": len(details),
                "meanTop1RelativeRegret": float(np.mean([item["top1RelativeRegret"] for item in details])),
                "top3Recall": sum(item["top3Recall"] for item in details) / len(details),
                "pairwiseOrderingAccuracy": (sum(item["pairwiseCorrect"] for item in details) / pair_total
                                               if pair_total else None),
                "byOpponentArchetype": by_archetype, "byOpponentPolicyFamily": by_family,
                "details": details}

    frozen = FrozenIteration(iteration, teacher_hash, opponent_policy_hash, manifest["manifestHash"], tuple(positions))
    ranker = XGBoostMacroRanker().fit(features, labels, groups, weights)
    train_metrics = metrics(ranker, train)
    development_metrics = metrics(ranker, [record for record in records if record["split"] == "development"])
    heldout_metrics = metrics(ranker, [record for record in records if record["split"] == "heldout"])
    holdouts = []
    for split in holdout_splits(records):
        train_records = [records[index] for index in split["train"] if records[index]["split"] == "train"]
        test_records = [records[index] for index in split["test"]]
        hx, hy, hw, hg, _, _ = flatten(train_records)
        if not hg or not test_records:
            holdouts.append({**split, "status": "insufficient"})
            continue
        held_model = XGBoostMacroRanker().fit(hx, hy, hg, hw)
        holdouts.append({**split, "status": "measured", "metrics": metrics(held_model, test_records)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=output.suffix,
                                     dir=output.parent, delete=False) as temporary_file:
        temporary_model = Path(temporary_file.name)
    try:
        ranker.model.save_model(temporary_model)
        with temporary_model.open("rb") as source:
            os.fsync(source.fileno())
        model_sha256 = file_sha256(temporary_model)
        # Hard-link publication is atomic and fails instead of replacing a
        # model created concurrently by another iteration.
        os.link(temporary_model, output)
    finally:
        temporary_model.unlink(missing_ok=True)
    result = {**ranker.manifest(frozen), "modelPath": str(output), "modelSha256": model_sha256,
              "trainingPositions": len(groups), "trainingCandidates": len(labels),
              "training": train_metrics, "development": development_metrics,
              "heldout": heldout_metrics,
              "holdouts": holdouts,
              "acceptance": "insufficient" if development_metrics["status"] != "measured"
                            or heldout_metrics["status"] != "measured"
                            or any(item["status"] != "measured" for item in holdouts) else "review-required"}
    if manifest_output.exists():
        raise ValueError("macro ranker manifest output is immutable; choose a new output path")
    _atomic_json(manifest_output, result)
    return result


def train_candidate(dataset_dir: Path, output: Path, *, epochs: int = 1) -> dict:
    manifest, rows = load_dataset(dataset_dir)
    records = training_records(rows, "train")
    result = train_supervised(records, output, manifest["identity"], epochs=epochs)
    return {**result, "datasetManifestHash": manifest["manifestHash"]}


def evaluate_candidate(dataset_dir: Path, checkpoint: Path) -> dict:
    import torch
    manifest, rows = load_dataset(dataset_dir)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_checkpoint_identity(saved, manifest["identity"])
    model = StrategyTransformerV1(); model.load_state_dict(saved["model"]); model.eval()
    evaluated = []
    for row in rows:
        if row["split"] == "train": continue
        encoded = encode_decision(row["observation"], row["tracker"])
        from .encoding import collate
        batch = {key: torch.as_tensor(value) for key, value in collate([encoded]).items()}
        with torch.no_grad(): logits = model.policy_forward(**batch)[0]
        probabilities = torch.softmax(logits, dim=-1)
        candidate = int(torch.argmax(logits).item())
        legal_actions = row["observation"]["legalActions"]
        heuristic_id = max(legal_actions, key=lambda action: heuristic_action_score(action, row["observation"]))["id"]
        heuristic_class = next(index for index, group in enumerate(encoded.action_classes)
                               if any(action["id"] == heuristic_id for action in group.actions))
        if row.get("acceptableActionIndices"):
            acceptable = set(row["acceptableActionIndices"])
            model_hit, heuristic_hit = candidate in acceptable, heuristic_class in acceptable
            cross_entropy = None
            label_kind = "acceptable-set"
        else:
            target = row["policyDistribution"]; best = int(np.argmax(target))
            model_hit, heuristic_hit = candidate == best, heuristic_class == best
            target_tensor = torch.as_tensor(target, dtype=probabilities.dtype)
            positive = target_tensor > 0
            cross_entropy = float(-(target_tensor[positive] * torch.log(probabilities[positive])).sum())
            label_kind = "distribution"
        evaluated.append({"positionHash": row["positionHash"], "split": row["split"],
                          "modelHit": model_hit, "heuristicHit": heuristic_hit,
                          "modelClass": candidate, "heuristicClass": heuristic_class,
                          "labelKind": label_kind, "modelCrossEntropy": cross_entropy})

    def summarize(selected):
        unique = {row["positionHash"]: row for row in selected}
        distribution = [row["modelCrossEntropy"] for row in selected if row["modelCrossEntropy"] is not None]
        return {"records": len(selected), "uniquePositions": len(unique),
                "modelAccuracyPerRecord": (sum(row["modelHit"] for row in selected) / len(selected)
                                            if selected else None),
                "heuristicAccuracyPerRecord": (sum(row["heuristicHit"] for row in selected) / len(selected)
                                                if selected else None),
                "modelAccuracyPerUniquePosition": (sum(row["modelHit"] for row in unique.values()) / len(unique)
                                                   if unique else None),
                "heuristicAccuracyPerUniquePosition": (sum(row["heuristicHit"] for row in unique.values()) / len(unique)
                                                       if unique else None),
                "meanDistributionCrossEntropy": float(np.mean(distribution)) if distribution else None}
    heldout = [row for row in evaluated if row["split"] == "heldout"]
    development = [row for row in evaluated if row["split"] == "development"]
    result = {"schemaVersion": 1, "checkpoint": str(checkpoint), "checkpointSha256": file_sha256(checkpoint),
              "datasetManifestHash": manifest["manifestHash"], "positions": evaluated,
              "development": summarize(development), "heldout": summarize(heldout),
              "heldOutLabelWin": bool(heldout and sum(r["modelHit"] for r in heldout) > sum(r["heuristicHit"] for r in heldout)),
              "targetProbeWin": False, "severityThreeRegression": False,
              "acceptance": "passed" if heldout and sum(r["modelHit"] for r in heldout) > sum(r["heuristicHit"] for r in heldout)
                            else "insufficient-or-not-improved",
              "note": "Probe gate remains closed until candidate actions are evaluated on the frozen v1.2 probe positions."}
    return result
