from __future__ import annotations

import copy
import json

import pytest
import torch

from ptcg_lab.features import heuristic_action_score
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.encoding import collate, encode_decision
from ptcg_lab.learning_mind.model import StrategyTransformerV1
from ptcg_lab.learning_mind.schema import IdentityManifest, identity_hash
from ptcg_lab.learning_mind.supervised_evidence import (audit_supervised_evaluation,
    summarize_blind_policy_families, summarize_paired_game_sides)
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
from ptcg_lab.storage import digest as legacy_digest
from test_learning_mind_representation import observation


def _dataset_and_evaluation(tmp_path, *, game_sides=2, positions_per_side=2,
                            heldout_policy_family="blind-policy",
                            training_policy_family="used-in-training"):
    identity = IdentityManifest.create(engine_build_hash="engine", deck_manifests={}, card_metadata={}).record()
    torch.manual_seed(123)
    model = StrategyTransformerV1().eval()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows, evaluations = [], []
    for game in range(game_sides):
        for index in range(positions_per_side):
            obs = copy.deepcopy(observation())
            obs["turn"] += index + game * positions_per_side
            tracker = ObservableHistoryTracker(0).update(obs)
            encoded = encode_decision(obs, tracker)
            tensor_batch = {key: torch.as_tensor(value) for key, value in collate([encoded]).items()}
            with torch.no_grad():
                model_class = int(torch.argmax(model.policy_forward(**tensor_batch)[0]).item())
            heuristic_id = max(obs["legalActions"],
                key=lambda action: heuristic_action_score(action, obs))["id"]
            heuristic_class = next(i for i, group in enumerate(encoded.action_classes)
                                   if any(action["id"] == heuristic_id for action in group.actions))
            acceptable = [model_class] if model_class < len(encoded.action_classes) else [0]
            position_hash = legacy_digest(obs)
            rows.append({"positionHash": position_hash, "familyId": f"family-{game}",
                "sourceGameId": f"game-{game}", "sourceDecisionIndex": index, "actor": 0,
                "opponentPolicyFamily": heldout_policy_family,
                "split": "heldout", "featureIdentityHash": encoded.identity,
                "policyLabelSource": "compatible-reviewed-acceptable-set",
                "acceptableActionIndices": acceptable, "policyDistribution": None,
                "observation": obs, "tracker": tracker})
            evaluations.append({"positionHash": position_hash, "split": "heldout",
                "modelHit": model_class in acceptable, "heuristicHit": heuristic_class in acceptable,
                "modelClass": model_class, "heuristicClass": heuristic_class,
                "labelKind": "acceptable-set", "modelCrossEntropy": None})
    training_observation = copy.deepcopy(observation())
    training_observation["turn"] += 10000
    training_tracker = ObservableHistoryTracker(0).update(training_observation)
    training_encoded = encode_decision(training_observation, training_tracker)
    rows.append({"positionHash": legacy_digest(training_observation), "familyId": "train-family",
        "sourceGameId": "train-game", "sourceDecisionIndex": 0, "actor": 0,
        "opponentPolicyFamily": training_policy_family, "split": "train",
        "featureIdentityHash": training_encoded.identity,
        "policyLabelSource": "compatible-reviewed-acceptable-set",
        "acceptableActionIndices": [0], "policyDistribution": None,
        "observation": training_observation, "tracker": training_tracker})
    rows_path = dataset / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                for row in rows))
    manifest = {"schemaVersion": 1, "identity": identity, "rows": len(rows),
        "rowsSha256": file_sha256(rows_path)}
    manifest["manifestHash"] = identity_hash(manifest)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"kind": "StrategyTransformerV1-supervised", "identity": identity,
        "datasetManifestHash": manifest["manifestHash"], "model": model.state_dict()}, checkpoint)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps({"checkpointSha256": file_sha256(checkpoint),
        "datasetManifestHash": manifest["manifestHash"], "positions": evaluations,
        "heldOutLabelWin": True}))
    return dataset, checkpoint, evaluation_path


def test_audit_does_not_accept_small_correlated_heldout_samples(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=2, positions_per_side=10)
    result = audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, output=tmp_path / "audit")
    assert result["independentGameSides"] == 2
    assert result["heldoutPositions"] == 20
    assert result["status"] == "insufficient"
    assert result["blindOpponentPolicyFamilyStatus"] == "insufficient"
    assert result["blindOpponentPolicyFamilies"]["blind-policy"]["independentGameSides"] == 2
    assert result["heldOutLabelWin"] is False


def test_audit_recomputes_model_logits_and_rejects_forged_hits(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=1)
    value = json.loads(evaluation.read_text())
    value["positions"][0]["modelClass"] = (value["positions"][0]["modelClass"] + 1) % 4
    value["positions"][0]["modelHit"] = True
    evaluation.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen checkpoint logits"):
        audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
            evaluation_path=evaluation, output=tmp_path / "audit")


def test_audit_does_not_call_a_seen_policy_family_blind(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1,
        positions_per_side=1, heldout_policy_family="used-in-training")
    result = audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, output=tmp_path / "audit")
    assert result["blindOpponentPolicyFamilyStatus"] == "missing"
    assert result["blindOpponentPolicyFamilies"] == {}


def test_audit_refuses_a_threshold_below_the_frozen_minimum(tmp_path):
    with pytest.raises(ValueError, match="cannot be lower than the frozen 20"):
        audit_supervised_evaluation(dataset_dir=tmp_path / "missing", checkpoint=tmp_path / "missing.pt",
            evaluation_path=tmp_path / "missing.json", output=tmp_path / "audit", minimum_game_sides=1)


def test_paired_game_side_wilson_gate_requires_enough_supported_wins():
    winning = [{"modelAccuracy": 1.0, "heuristicAccuracy": 0.0} for _ in range(20)]
    result = summarize_paired_game_sides(winning)
    assert result["status"] == "supported-improvement"
    assert result["heldOutLabelWin"] is True
    too_few = summarize_paired_game_sides(winning[:19])
    assert too_few["status"] == "insufficient" and not too_few["heldOutLabelWin"]
    inconclusive = summarize_paired_game_sides(
        [{"modelAccuracy": .6, "heuristicAccuracy": .4} for _ in range(10)]
        + [{"modelAccuracy": .4, "heuristicAccuracy": .6} for _ in range(10)])
    assert inconclusive["status"] == "inconclusive"


def test_blind_family_gate_requires_a_supported_individual_family():
    rows = [{"opponentPolicyFamily": family, "modelAccuracy": 0.0,
             "heuristicAccuracy": 1.0, "blindFamilyEligible": True}
            for family in ("blind-a", "blind-b") for _ in range(10)]
    result = summarize_blind_policy_families(rows, {"training-family"})
    assert result["status"] == "insufficient"
    assert {item["independentGameSides"] for item in result["families"].values()} == {10}


def test_blind_family_gate_ignores_review_and_replay_source_labels():
    rows = [{"opponentPolicyFamily": family, "modelAccuracy": 1.0,
             "heuristicAccuracy": 0.0, "blindFamilyEligible": False}
            for family in ("human-review", "frozen-search-source") for _ in range(20)]
    result = summarize_blind_policy_families(rows, set())
    assert result["status"] == "missing"
