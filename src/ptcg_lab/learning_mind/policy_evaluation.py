"""Decoder-faithful supervised and strategy-probe evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ptcg_lab.features import heuristic_action_score
from research.strategy_baseline.probes import (PROBES,
    registry_hash as strategy_probe_registry_hash)

from .dataset_v1 import (file_sha256, load_dataset, load_strategy_probe_dataset)
from .encoding import collate, encode_decision
from .experiment import summarize_strategy_probe_decisions
from .model import StrategyTransformerV1, greedy_single_action_class
from .training import require_checkpoint_identity, require_checkpoint_implementation


def _single_action(model, encoded):
    tensors = {key: torch.as_tensor(value) for key, value in collate([encoded]).items()}

    def step_logits(chosen):
        selected = torch.zeros_like(tensors["option_mask"])
        for index in chosen:
            selected[0, index] = True
        with torch.no_grad():
            return model.policy_forward(**tensors, selected_mask=selected)[0]

    action_class = greedy_single_action_class(step_logits,
        action_count=len(encoded.action_classes),
        legality=lambda _chosen, index: 0 <= index < len(encoded.action_classes))
    return action_class, tensors


def _evaluate_strategy_probes(probe_dataset_dir: Path, model, identity: dict,
                               source_manifest_sha256: str) -> dict:
    manifest, rows = load_strategy_probe_dataset(probe_dataset_dir, identity=identity)
    if manifest.get("sourceDatasetManifestSha256") != source_manifest_sha256:
        raise ValueError("strategy-probe corpus and supervised dataset use different frozen source games")
    decisions = []
    for row in sorted(rows, key=lambda item: (item["sourceGameId"], item["actor"],
                                               item["sourceDecisionIndex"], item["positionHash"])):
        encoded = encode_decision(row["observation"], row["tracker"])
        if encoded.identity != row.get("featureIdentityHash"):
            raise ValueError("strategy-probe feature identity differs from frozen actor-view row")
        model_class, _tensors = _single_action(model, encoded)
        model_action = encoded.action_classes[model_class].actions[0]
        observation = row["observation"]
        legal_actions = observation["legalActions"]
        heuristic_id = max(legal_actions,
            key=lambda action: heuristic_action_score(action, observation))["id"]
        heuristic_action = next(action for action in legal_actions if action["id"] == heuristic_id)
        for probe in PROBES:
            model_result = probe.evaluate(observation, model_action)
            heuristic_result = probe.evaluate(observation, heuristic_action)
            if (model_result is None) != (heuristic_result is None):
                raise ValueError(f"probe eligibility changed with chosen action: {probe.id}")
            if model_result is not None:
                decisions.append({"probeId": probe.id, "gameSideKey": row["gameSideKey"],
                    "positionHash": row["positionHash"], "decisionIndex": row["sourceDecisionIndex"],
                    "modelAction": model_action, "heuristicAction": heuristic_action,
                    "modelAdherent": bool(model_result), "heuristicAdherent": bool(heuristic_result)})
    result = summarize_strategy_probe_decisions(decisions)
    return {**result, "probeRegistryHash": strategy_probe_registry_hash(),
            "probeDatasetManifestSha256": file_sha256(probe_dataset_dir / "manifest.json"),
            "sourceGameCount": manifest["sourceGameCount"],
            "actorDecisionRows": manifest["actorDecisionRows"], "actorViewOnly": True}


def evaluate_candidate(dataset_dir: Path, checkpoint: Path, probe_dataset_dir: Path) -> dict:
    """Evaluate the executable one-action policy, never raw STOP/padding argmax."""
    manifest, rows = load_dataset(dataset_dir)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_checkpoint_identity(saved, manifest["identity"])
    require_checkpoint_implementation(saved)
    if saved.get("datasetManifestHash") != manifest.get("manifestHash"):
        raise ValueError("supervised checkpoint was not trained from this frozen dataset manifest")
    model = StrategyTransformerV1()
    model.load_state_dict(saved["model"])
    model.eval()

    evaluated = []
    for row in rows:
        if row["split"] == "train":
            continue
        encoded = encode_decision(row["observation"], row["tracker"])
        if encoded.identity != row.get("featureIdentityHash"):
            raise ValueError("evaluation feature identity differs from frozen actor-view row")
        candidate, tensors = _single_action(model, encoded)
        with torch.no_grad():
            logits = model.policy_forward(**tensors)[0]
            probabilities = torch.softmax(logits, dim=-1)
        legal_actions = row["observation"]["legalActions"]
        heuristic_id = max(legal_actions,
            key=lambda action: heuristic_action_score(action, row["observation"]))["id"]
        heuristic_class = next(index for index, group in enumerate(encoded.action_classes)
            if any(action["id"] == heuristic_id for action in group.actions))
        if row.get("acceptableActionIndices"):
            acceptable = set(row["acceptableActionIndices"])
            model_hit, heuristic_hit = candidate in acceptable, heuristic_class in acceptable
            cross_entropy = None
            label_kind = "acceptable-set"
        else:
            target = row["policyDistribution"]
            best = int(np.argmax(target))
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
        distribution = [row["modelCrossEntropy"] for row in selected
                        if row["modelCrossEntropy"] is not None]
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
    source_manifests = [source["sha256"] for source in manifest.get("sources", [])
                        if source.get("kind") == "experimental-dataset-manifest"]
    if len(source_manifests) != 1:
        raise ValueError("supervised dataset must bind exactly one frozen replay manifest for probe evaluation")
    probe_results = _evaluate_strategy_probes(probe_dataset_dir, model, manifest["identity"],
                                               source_manifests[0])
    label_win = bool(heldout and sum(row["modelHit"] for row in heldout)
                     > sum(row["heuristicHit"] for row in heldout))
    return {"schemaVersion": 1, "checkpoint": str(checkpoint),
        "checkpointSha256": file_sha256(checkpoint), "datasetManifestHash": manifest["manifestHash"],
        "positions": evaluated, "development": summarize(development), "heldout": summarize(heldout),
        "heldOutLabelWin": label_win, "strategyProbes": probe_results,
        "targetProbeWin": probe_results["targetProbeWin"],
        "severityThreeRegression": probe_results["severityThreeRegression"],
        "severityThreeProbeCoverage": probe_results["severityThreeCoverage"],
        "acceptance": "passed" if label_win and probe_results["targetProbeWin"]
            and not probe_results["severityThreeRegression"]
            and probe_results["severityThreeCoverage"] == "sufficient"
            else "insufficient-or-not-improved",
        "policyDecoder": "greedy-autoregressive-one-legal-action-v1",
        "automaticPromotion": False}
