from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ptcg_lab.engine import EngineClient
from ptcg_lab.features import heuristic_action_score

from .dataset_v1 import file_sha256, load_dataset, training_records
from .encoding import encode_decision
from .macro import (CANDIDATE_GENERATOR_VERSION, candidates_from_transition_plans,
                    label_candidates, rollout_seed)
from .model import StrategyTransformerV1
from .ranker import FrozenIteration, XGBoostMacroRanker, holdout_splits
from .schema import IdentityManifest, UnsupportedPosition, identity_hash
from .training import require_checkpoint_identity, train_supervised


def transition_generator_identity(root: Path) -> dict:
    planner = root / "research/learning_mind/transition_macro_planner.ts"
    action_key = root / "packages/engine/src/action-key.ts"
    adapter = Path(__file__).with_name("macro.py")
    if not planner.is_file() or not action_key.is_file():
        raise FileNotFoundError(f"transition planner/action identity helper missing: {planner} / {action_key}")
    return {"version": CANDIDATE_GENERATOR_VERSION,
            "plannerSha256": file_sha256(planner), "actionKeySha256": file_sha256(action_key),
            "adapterSha256": file_sha256(adapter)}


def generate_transition_candidates(root: Path, observation: dict, seed: int):
    planner = root / "research/learning_mind/transition_macro_planner.ts"
    runner = root / "node_modules/.bin/tsx"
    result = subprocess.run([str(runner), str(planner)], cwd=root,
        input=json.dumps({"observation": observation, "seed": seed}, separators=(",", ":")),
        text=True, capture_output=True, timeout=180, check=False)
    if result.returncode:
        message = (result.stderr or result.stdout).strip()[-2000:]
        if "unsupported position:" in message.lower():
            raise UnsupportedPosition(message)
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
                         horizon: int = 16) -> dict:
    manifest, rows = load_dataset(dataset_dir, identity=identity)
    output.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit]
    generator_identity = transition_generator_identity(root)
    settings = {"identity": identity, "datasetManifestHash": manifest["manifestHash"],
                "requestedPositions": len(selected), "initialRollouts": initial,
                "maximumRollouts": maximum, "horizon": horizon,
                "candidateGeneratorVersion": CANDIDATE_GENERATOR_VERSION,
                "candidateGeneratorIdentity": generator_identity}
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
    with EngineClient(root, timeout=300) as engine:
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

            def rollout(candidate, seed):
                root_action = legal.get(candidate.action_ids[0])
                if root_action is None: return {"status": "error", "reason": "macro-root-no-longer-legal"}
                narrowed = copy.deepcopy(observation); narrowed["legalActions"] = [root_action]
                plan_actions = list(candidate.action_sequence)
                if not plan_actions or plan_actions[0].get("id") != root_action["id"]:
                    return {"status": "error", "reason": "macro-plan-root-mismatch"}
                result = engine.request("search", {"observation": narrowed, "seed": seed, "budgetMs": 120000,
                                                    "method": "rollout", "iterations": 1,
                                                    "maxRolloutDecisions": horizon,
                                                    "macroPlanActions": plan_actions})
                alternative = next((item for item in result.get("alternatives", [])
                                    if item.get("actionId") == root_action["id"] and item.get("visits", 0) > 0), None)
                execution = result.get("macroPlanExecution") or {}
                if execution.get("requested") and not execution.get("completed"):
                    reason = (execution.get("failures") or [{"reason": "macro-plan-unexecuted"}])[0]["reason"]
                    return {"status": "error", "reason": reason}
                if result.get("status") != "complete" or alternative is None or alternative.get("score") is None:
                    return {"status": "error", "reason": "search-rollout-unavailable"}
                return {"status": "finished", "score": float(alternative["score"])}

            namespace = "training" if row["split"] == "train" else "development"
            labels = label_candidates(executable, key, rollout, namespace=namespace,
                                      initial=initial, maximum=maximum)
            record = {"schemaVersion": 1, "positionHash": key, "split": row["split"],
                      "identity": identity, "datasetManifestHash": manifest["manifestHash"],
                      "familyId": row["familyId"], "opponentArchetype": row["opponentArchetype"],
                      "opponentPolicyFamily": row["opponentPolicyFamily"],
                      "observation": observation, "labels": labels,
                      "status": "collected", "generatorSeed": generator_seed,
                      "generatorHypothesisId": generation.get("hypothesisId"),
                      "candidateCount": len(executable),
                      "incompletePrefixCount": len(candidates) - len(executable),
                      "seedNamespace": namespace,
                      "rolloutSeeds": [rollout_seed(namespace, key, index) for index in range(maximum)],
                      "semantics": "complete transition-aware attack or deliberate no-attack plan from one actor-visible public determinization; later steps are revalidated at rollout",
                      "highConfidencePolicyEligible": False}
            _atomic_json(output / f"{key}.json", record)
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


def fit_ranker(labels_dir: Path, output: Path, *, teacher_hash: str,
               opponent_policy_hash: str, iteration: int = 1) -> dict:
    manifest = json.loads((labels_dir / "manifest.json").read_text())
    records = [json.loads((labels_dir / item["path"]).read_text()) for item in manifest["files"]]
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
    output.parent.mkdir(parents=True, exist_ok=True); ranker.model.save_model(output)
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
    result = {**ranker.manifest(frozen), "modelPath": str(output), "modelSha256": file_sha256(output),
              "trainingPositions": len(groups), "trainingCandidates": len(labels),
              "training": train_metrics, "development": development_metrics,
              "heldout": heldout_metrics,
              "holdouts": holdouts,
              "acceptance": "insufficient" if development_metrics["status"] != "measured"
                            or heldout_metrics["status"] != "measured"
                            or any(item["status"] != "measured" for item in holdouts) else "review-required"}
    _atomic_json(output.with_suffix(".manifest.json"), result)
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
