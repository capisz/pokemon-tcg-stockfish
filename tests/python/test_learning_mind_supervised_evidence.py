from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from ptcg_lab.features import heuristic_action_score
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.disagreement_review import (audit_disagreement_review,
    build_disagreement_review_packet, write_disagreement_review_template)
from ptcg_lab.learning_mind.candidate_safety import (audit_candidate_safety,
    _verify_legal_action_coverage)
from ptcg_lab.learning_mind.encoding import collate, encode_decision
from ptcg_lab.learning_mind.model import StrategyTransformerV1, greedy_single_action_class
from ptcg_lab.learning_mind import policy_evaluation
from ptcg_lab.learning_mind.schema import IdentityManifest, identity_hash
from ptcg_lab.learning_mind.supervised_evidence import (audit_supervised_evaluation,
    summarize_blind_policy_families, summarize_paired_game_sides)
from ptcg_lab.learning_mind.training import supervised_implementation_identity
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
            def step_logits(chosen):
                selected = torch.zeros_like(tensor_batch["option_mask"])
                for selected_index in chosen:
                    selected[0, selected_index] = True
                with torch.no_grad():
                    return model.policy_forward(**tensor_batch, selected_mask=selected)[0]
            model_class = greedy_single_action_class(step_logits,
                action_count=len(encoded.action_classes),
                legality=lambda _chosen, action_index: 0 <= action_index < len(encoded.action_classes))
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
    implementation = supervised_implementation_identity()
    config = {"implementationIdentity": implementation, "teacherHashes": [],
        "policyLabelSources": [], "valueLabelSources": [], "valueDatasetManifestHash": None,
        "valueLossCoefficient": .5, "lossContract": "approved-policy-plus-terminal-outcome-mse-v1"}
    torch.save({"schemaVersion": 2, "kind": "StrategyTransformerV1-supervised", "identity": identity,
        "datasetManifestHash": manifest["manifestHash"], "model": model.state_dict(),
        "implementationIdentity": implementation,
        "parentCheckpointSha256": None, "teacherHashes": [],
        "policyLabelSources": [], "valueLabelSources": [],
        "trainingConfig": config}, checkpoint)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps({"checkpointSha256": file_sha256(checkpoint),
        "datasetManifestHash": manifest["manifestHash"], "positions": evaluations,
        "policyDecoder": "greedy-autoregressive-one-legal-action-v1",
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
    value["positions"][0]["modelClass"] = 0 if value["positions"][0]["modelClass"] != 0 else 1
    value["positions"][0]["modelHit"] = True
    evaluation.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen checkpoint logits"):
        audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
            evaluation_path=evaluation, output=tmp_path / "audit")


def test_audit_rejects_legacy_raw_argmax_evaluations(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=1)
    value = json.loads(evaluation.read_text())
    value.pop("policyDecoder")
    evaluation.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not identify the frozen executable policy decoder"):
        audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
            evaluation_path=evaluation, output=tmp_path / "audit")


def test_audit_rejects_checkpoint_implementation_drift(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=1)
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value["implementationIdentity"]["sources"]["modelSha256"] = "0" * 64
    torch.save(value, checkpoint)
    with pytest.raises(ValueError, match="implementation identity mismatch"):
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


def test_human_disagreement_review_is_actor_view_only_and_hash_bound(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=2)
    audit_path = tmp_path / "audit.json"
    audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, output=audit_path)
    packet_path = tmp_path / "packet.json"
    packet = build_disagreement_review_packet(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, audit_path=audit_path, output=packet_path)
    expected = [item for item in json.loads(evaluation.read_text())["positions"]
                if item["split"] == "heldout" and item["modelClass"] != item["heuristicClass"]]
    assert packet["heldoutDisagreementCount"] == len(expected) > 0
    assert all(item["actorObservation"]["playerId"] == item["actor"]
               for item in packet["positions"])
    assert all("oppositeObservation" not in item for item in packet["positions"])

    review_path = tmp_path / "review.json"
    review = write_disagreement_review_template(packet_path=packet_path, output=review_path)
    review["reviewer"] = "human reviewer"
    for item in review["reviews"]:
        item["finding"] = "acceptable"
        item["rationale"] = "Reviewed against the frozen actor-visible context."
    review_path.write_text(json.dumps(review))
    receipt = audit_disagreement_review(packet_path=packet_path, review_path=review_path,
        output=tmp_path / "receipt.json")
    assert receipt["representativeDisagreementsReviewed"] is True
    assert receipt["reviewedPositions"] == len(expected)
    assert receipt["automaticPromotion"] is False


def test_downstream_review_rejects_self_rehashed_forged_audit_conclusion(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=2)
    audit_path = tmp_path / "audit.json"
    audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, output=audit_path)
    claimed = json.loads(audit_path.read_text())
    claimed["heldOutLabelWin"] = True
    claimed["status"] = "supported-improvement"
    claimed["reportHash"] = identity_hash({key: value for key, value in claimed.items()
                                           if key != "reportHash"})
    audit_path.write_text(json.dumps(claimed))
    with pytest.raises(ValueError, match="fresh recomputation"):
        build_disagreement_review_packet(dataset_dir=dataset, checkpoint=checkpoint,
            evaluation_path=evaluation, audit_path=audit_path, output=tmp_path / "packet.json")


def test_human_disagreement_review_requires_every_action_to_be_acceptable(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=1)
    audit_path = tmp_path / "audit.json"
    audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, output=audit_path)
    packet_path = tmp_path / "packet.json"
    packet = build_disagreement_review_packet(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, audit_path=audit_path, output=packet_path)
    if not packet["positions"]:
        pytest.skip("deterministic fixture produced no model/heuristic disagreement")
    review_path = tmp_path / "review.json"
    review = write_disagreement_review_template(packet_path=packet_path, output=review_path)
    review["reviewer"] = "human reviewer"
    review["reviews"][0].update(finding="concern", rationale="The move violates the reviewed plan.")
    review_path.write_text(json.dumps(review))
    receipt = audit_disagreement_review(packet_path=packet_path, review_path=review_path,
        output=tmp_path / "receipt.json")
    assert receipt["representativeDisagreementsReviewed"] is False
    assert len(receipt["unresolvedPositions"]) == 1


def test_candidate_safety_audits_all_actor_view_options_and_one_step_decoder(tmp_path):
    dataset, checkpoint, evaluation = _dataset_and_evaluation(tmp_path, game_sides=2, positions_per_side=2)
    audit_path = tmp_path / "audit.json"
    audit_supervised_evaluation(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, output=audit_path)
    report = audit_candidate_safety(dataset_dir=dataset, checkpoint=checkpoint,
        evaluation_path=evaluation, audit_path=audit_path, output=tmp_path / "safety.json")
    assert report["positionsChecked"] == 4
    assert report["legalActionsChecked"] > report["positionsChecked"]
    assert report["legalActionOmission"] is False
    assert report["illegalAutoregressiveSelection"] is False
    assert report["capOverflow"] is False
    assert report["maximumActionClasses"] <= 128


def test_official_policy_evaluation_reports_legal_decoder_choice_when_stop_has_highest_logit(tmp_path, monkeypatch):
    dataset, checkpoint, _evaluation = _dataset_and_evaluation(tmp_path, game_sides=1, positions_per_side=1)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = [{"kind": "experimental-dataset-manifest", "sha256": "e" * 64}]
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                               if key != "manifestHash"})
    manifest_path.write_text(json.dumps(manifest))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["datasetManifestHash"] = manifest["manifestHash"]
    torch.save(saved, checkpoint)

    class StopBiasedModel:
        def __init__(self):
            pass
        def load_state_dict(self, _state):
            return None
        def eval(self):
            return self
        def policy_forward(self, *, option_mask, **_kwargs):
            logits = torch.zeros(option_mask.shape, dtype=torch.float32)
            logits[:, -1] = 100.0
            return logits

    monkeypatch.setattr(policy_evaluation, "StrategyTransformerV1", StopBiasedModel)
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "manifest.json").write_text("{}\n")
    monkeypatch.setattr(policy_evaluation, "load_strategy_probe_dataset", lambda *_args, **_kwargs: (
        {"sourceDatasetManifestSha256": "e" * 64, "sourceGameCount": 0, "actorDecisionRows": 0}, []))
    report = policy_evaluation.evaluate_candidate(dataset, checkpoint, probe_dir)
    assert report["policyDecoder"] == "greedy-autoregressive-one-legal-action-v1"
    assert report["positions"]
    for item in report["positions"]:
        assert 0 <= item["modelClass"] < 3

    stale = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stale["implementationIdentity"]["sources"]["policyEvaluatorSha256"] = "0" * 64
    torch.save(stale, checkpoint)
    with pytest.raises(ValueError, match="implementation identity mismatch"):
        policy_evaluation.evaluate_candidate(dataset, checkpoint, probe_dir)


def test_action_coverage_audit_detects_omission_and_duplicate_representation():
    legal = [{"id": "a"}, {"id": "b"}]
    grouped = [SimpleNamespace(actions=({"id": "a"}, {"id": "b"}))]
    assert _verify_legal_action_coverage(legal, grouped) == 2
    with pytest.raises(ValueError, match="omit or duplicate"):
        _verify_legal_action_coverage(legal, [SimpleNamespace(actions=({"id": "a"},))])
    duplicated = [SimpleNamespace(actions=({"id": "a"}, {"id": "a"}, {"id": "b"}))]
    with pytest.raises(ValueError, match="omit or duplicate"):
        _verify_legal_action_coverage(legal, duplicated)


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
