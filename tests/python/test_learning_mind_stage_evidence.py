from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptcg_lab.learning_mind import stage_evidence
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.learning_mind.training import ppo_enablement


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "root": tmp_path / "root",
        "baseline_manifest": tmp_path / "baseline.json",
        "dataset_dir": tmp_path / "dataset",
        "probe_dataset_dir": tmp_path / "probe-dataset",
        "checkpoint": tmp_path / "checkpoint.pt",
        "evaluation_path": tmp_path / "evaluation.json",
        "supervised_audit_path": tmp_path / "supervised-audit.json",
        "safety_report_path": tmp_path / "safety.json",
        "macro_selection_path": tmp_path / "selection.json",
        "python_dataset": tmp_path / "python-dataset",
        "typescript_dataset": tmp_path / "typescript-dataset",
        "python_labels": tmp_path / "python-labels",
        "typescript_labels": tmp_path / "typescript-labels",
        "macro_fidelity_path": tmp_path / "fidelity.json",
        "ranker_labels_dir": tmp_path / "ranker-labels",
        "confidence_audit_path": tmp_path / "confidence-audit.json",
        "ranker_model_path": tmp_path / "ranker-model.json",
        "ranker_report_path": tmp_path / "ranker-report.json",
        "disagreement_packet_path": tmp_path / "packet.json",
        "disagreement_review_path": tmp_path / "review.json",
        "disagreement_receipt_path": tmp_path / "review-receipt.json",
        "output": tmp_path / "stage-evidence.json",
    }
    for key, path in paths.items():
        if key == "output":
            continue
        if key == "baseline_manifest":
            path.write_text(json.dumps({"replays": [{"path": f"replay-{index}.json"}
                                                      for index in range(192)]}))
            continue
        if key.endswith("_dir") or key in {"root", "python_dataset", "typescript_dataset",
                                          "python_labels", "typescript_labels"}:
            path.mkdir(parents=True)
            if key.endswith("_dir") or key in {"python_dataset", "typescript_dataset",
                                                "python_labels", "typescript_labels"}:
                (path / "manifest.json").write_text("{}\n")
        else:
            path.write_text("{}\n")
    paths["macro_selection_path"].write_text(json.dumps({"selectionHash": "selection-hash"}))
    return paths


def _patch_verifiers(monkeypatch, paths, *, safety=None):
    evaluation = {"strategyProbes": {"targetProbeWin": True,
        "severityThreeRegression": False, "severityThreeCoverage": "sufficient"}}
    audit = {"status": "supported-improvement",
        "blindOpponentPolicyFamilyStatus": "supported-improvement",
        "evaluationIdentityStatus": "matched"}
    safety = safety or {"legalActionOmission": False,
        "illegalAutoregressiveSelection": False, "capOverflow": False,
        "reportHash": "safety-hash"}
    fidelity = {"ragingBoltMacroPlanFidelity": "passed", "reportHash": "fidelity-hash",
                "identity": {"frozen": "same"}}
    review = {"representativeDisagreementsReviewed": True, "receiptHash": "review-hash"}
    label_manifest = {"manifestHash": "ranker-label-manifest", "selectionHash": "selection-hash",
        "identity": {"frozen": "same"}}
    confidence_report = {"reportHash": "confidence-report-hash",
        "confidenceAuditImplementationSha256": "c" * 64}
    ranker_report = {"identity": {"frozen": "same"}, "acceptance": "review-required",
        "modelSha256": "a" * 64, "reportHash": "b" * 64,
        "confidenceAuditReportHash": confidence_report["reportHash"],
        "inputManifestHash": label_manifest["manifestHash"],
        "inputManifestSha256": file_sha256(paths["ranker_labels_dir"] / "manifest.json"),
        "selectionHash": label_manifest["selectionHash"],
        "selectionManifestSha256": file_sha256(paths["macro_selection_path"]),
        "confidenceAuditSha256": file_sha256(paths["confidence_audit_path"]),
        "confidenceAuditImplementationSha256": confidence_report["confidenceAuditImplementationSha256"],
        "development": {"status": "measured"}, "holdouts": [
            {"kind": "leave-one-opponent-archetype-out", "status": "measured"},
            {"kind": "frozen-policy-family", "status": "measured"}]}
    monkeypatch.setattr(stage_evidence, "audit_manifest", lambda _path: {
        "scheduled": 192, "audited": 192, "unsupportedPositions": 0,
        "representationParity": True})
    monkeypatch.setattr(stage_evidence, "load_dataset", lambda _path: (
        {"manifestHash": "dataset", "identity": {"frozen": "same"},
         "teacherHashes": sorted({ranker_report["modelSha256"], ranker_report["reportHash"],
                                  confidence_report["reportHash"]})}, []))
    monkeypatch.setattr(stage_evidence, "verify_supervised_audit_report", lambda **_kwargs: audit)
    monkeypatch.setattr(stage_evidence, "audit_candidate_safety", lambda **_kwargs: safety)
    monkeypatch.setattr(stage_evidence, "evaluate_candidate", lambda *_args: evaluation)
    monkeypatch.setattr(stage_evidence, "audit_raging_bolt_macro_fidelity", lambda **_kwargs: fidelity)
    monkeypatch.setattr(stage_evidence, "audit_disagreement_review", lambda **_kwargs: review)
    monkeypatch.setattr(stage_evidence, "verify_macro_ranker_v2_artifact", lambda *_args: {
        "report": ranker_report, "artifact": {}})
    monkeypatch.setattr(stage_evidence, "_load_ranker_input", lambda *_args: (label_manifest, []))
    monkeypatch.setattr(stage_evidence, "verify_macro_label_confidence_audit",
        lambda **_kwargs: confidence_report)
    (paths["evaluation_path"]).write_text(json.dumps(evaluation))
    (paths["safety_report_path"]).write_text(json.dumps(safety))
    (paths["macro_fidelity_path"]).write_text(json.dumps(fidelity))
    (paths["disagreement_receipt_path"]).write_text(json.dumps(review))
    return ranker_report


def test_stage_evidence_requires_recomputed_reports_and_explicit_human_authorization(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    _patch_verifiers(monkeypatch, paths)
    report, capability = stage_evidence.verify_ppo_stage_evidence(**paths)
    assert report["prerequisitesPassed"] is True
    assert report["ppoEnabled"] is False
    assert ppo_enablement(capability)["enabled"] is False
    with pytest.raises(TypeError):
        capability._values["humanEnablePPO"] = True
    with pytest.raises(AttributeError, match="immutable"):
        capability._values = {**capability._values, "humanEnablePPO": True}
    assert report["reportHash"] == identity_hash({key: value for key, value in report.items()
                                                   if key != "reportHash"})
    assert report["macroRankerAcceptance"] == "review-required"
    assert set(report["macroRankerMeasuredHoldoutKinds"]) == {
        "leave-one-opponent-archetype-out", "frozen-policy-family"}
    assert report["macroRankerDistillationBound"] is True

    authorized_paths = {**paths, "output": tmp_path / "authorized-stage.json"}
    report, capability = stage_evidence.verify_ppo_stage_evidence(
        **authorized_paths, human_enable_ppo=True)
    assert report["ppoEnabled"] is True
    assert ppo_enablement(capability)["enabled"] is True
    report["macroRankerMeasuredHoldoutKinds"].clear()
    next(iter(report["sourceArtifacts"].values()))["sha256"] = "tampered"
    assert ppo_enablement(capability)["enabled"] is True
    with pytest.raises(TypeError):
        capability._values["macroRankerMeasuredHoldoutKinds"][0] = "tampered"
    source_artifact = next(iter(capability._values["sourceArtifacts"].values()))
    with pytest.raises(TypeError):
        source_artifact["sha256"] = "tampered"

    torch = pytest.importorskip("torch")
    from ptcg_lab.learning_mind.encoding import collate
    from ptcg_lab.learning_mind.model import StrategyTransformerV1
    from ptcg_lab.learning_mind.training import (load_ppo_checkpoint, ppo_policy_fingerprint,
        ppo_update, save_ppo_checkpoint, supervised_ppo_update)
    from ptcg_lab.learning_mind.supervisor import MindSupervisor
    from test_learning_mind_representation import encoded

    torch.manual_seed(4)
    model = StrategyTransformerV1()
    behavior_weights = {key: value.clone() for key, value in model.state_dict().items()}
    decision = encoded()
    batch = {key: torch.as_tensor(value) for key, value in collate([decision]).items()}
    with torch.no_grad():
        logits = model.policy_forward(**batch)
        legal_count = len(decision.action_classes)
        old_log_prob = torch.log_softmax(logits[0, :legal_count], -1)[0].item()
        value = model.evaluation_forward(**{key: batch[key] for key in
            ("state_card_ids", "state_features", "state_type_ids", "state_mask")})[0].item()
    row = {"episodeId": "completed-0", "episodeStatus": "finished", "episodeEnd": True,
        "reward": 1, "encoded": decision, "selectedAction": 0, "oldLogProb": old_log_prob,
        "return": value, "advantage": 1., "behaviorPolicyHash": ppo_policy_fingerprint(model)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    checkpoint_path = tmp_path / "ppo-update-000000.pt"
    experience_hash = "a" * 64
    checkpoint_hash = save_ppo_checkpoint(checkpoint_path, model, optimizer,
        update_index=0, experiment_identity={"engine": "frozen"},
        experience_manifest_sha256=experience_hash, stage_record=capability)
    restored = StrategyTransformerV1()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=1e-4)
    restored_state = load_ppo_checkpoint(checkpoint_path, restored, restored_optimizer,
        expected_sha256=checkpoint_hash, experiment_identity={"engine": "frozen"},
        experience_manifest_sha256=experience_hash, stage_record=capability)
    assert restored_state["updateIndex"] == 0 and restored_state["nextUpdateIndex"] == 1
    assert all(torch.equal(model.state_dict()[key], restored.state_dict()[key])
               for key in model.state_dict())
    uninterrupted_update = ppo_update(model, optimizer, [row], stage_record=capability)
    resumed_update = ppo_update(restored, restored_optimizer, [row], stage_record=capability)
    assert uninterrupted_update == resumed_update
    assert uninterrupted_update["acceptedMinibatches"] == 1
    assert uninterrupted_update["optimizationEpochs"] == 1
    assert all(torch.equal(model.state_dict()[key], restored.state_dict()[key])
               for key in model.state_dict())
    for key, value in optimizer.state_dict()["state"].items():
        for field, moment in value.items():
            assert torch.equal(moment, restored_optimizer.state_dict()["state"][key][field])
    with pytest.raises(ValueError, match="identity/configuration"):
        load_ppo_checkpoint(checkpoint_path, restored, restored_optimizer,
            expected_sha256=checkpoint_hash, experiment_identity={"engine": "changed"},
            experience_manifest_sha256=experience_hash, stage_record=capability)
    with pytest.raises(ValueError, match="every ten updates"):
        save_ppo_checkpoint(tmp_path / "bad-index.pt", model, optimizer,
            update_index=1, experiment_identity={"engine": "frozen"},
            experience_manifest_sha256=experience_hash, stage_record=capability)

    stale_model = StrategyTransformerV1()
    stale_model.load_state_dict(behavior_weights)
    stale_row = {**row, "oldLogProb": row["oldLogProb"] + 1e-3}
    stale_optimizer = torch.optim.AdamW(stale_model.parameters(), lr=1e-4, weight_decay=1e-4)
    before_stale = {key: value.clone() for key, value in stale_model.state_dict().items()}
    with pytest.raises(ValueError, match="oldLogProb does not match"):
        ppo_update(stale_model, stale_optimizer, [stale_row], stage_record=capability)
    assert all(torch.equal(before_stale[key], stale_model.state_dict()[key]) for key in before_stale)

    wrong_optimizer = torch.optim.AdamW(stale_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="learning rate and weight decay"):
        ppo_update(stale_model, wrong_optimizer, [row], stage_record=capability)

    nonfinite_model = StrategyTransformerV1()
    nonfinite_model.load_state_dict(behavior_weights)
    nonfinite_optimizer = torch.optim.AdamW(nonfinite_model.parameters(), lr=1e-4, weight_decay=1e-4)
    before_nonfinite = {key: value.clone() for key, value in nonfinite_model.state_dict().items()}
    supervisor = MindSupervisor(tmp_path / "nonfinite-supervisor", reserve_bytes=0,
                                data_cap_bytes=10_000_000)
    supervisor.state.status = "RUNNING"
    gradient_hook = nonfinite_model.policy_query[-1].weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan")))
    rejected = supervised_ppo_update(nonfinite_model, nonfinite_optimizer, [row],
        supervisor=supervisor, stage_record=capability)
    gradient_hook.remove()
    assert rejected["pauseRequired"] is True
    assert rejected["rejectionReasons"] == {"non-finite-gradient": 1}
    assert rejected["acceptedMinibatches"] == 0
    assert supervisor.state.status == "PAUSED"
    assert supervisor.state.pause_reason == "non-finite"
    assert all(torch.equal(before_nonfinite[key], nonfinite_model.state_dict()[key])
               for key in before_nonfinite)

    from ptcg_lab.learning_mind import training as training_module
    monkeypatch.setattr(training_module, "ppo_update", lambda *_args, **_kwargs: {
        "acceptedMinibatches": 0, "rejectedMinibatches": 1, "rejectionReasons": {"approximate-kl-exceeded": 1},
        "pauseRequired": False})
    rejected_supervisor = MindSupervisor(tmp_path / "rejected-supervisor", reserve_bytes=0,
                                         data_cap_bytes=10_000_000)
    rejected_supervisor.state.status = "RUNNING"
    for _ in range(2):
        training_module.supervised_ppo_update(None, None, [], supervisor=rejected_supervisor,
                                              stage_record=capability)
        assert rejected_supervisor.state.status == "RUNNING"
    training_module.supervised_ppo_update(None, None, [], supervisor=rejected_supervisor,
                                          stage_record=capability)
    assert rejected_supervisor.state.status == "PAUSED"
    assert rejected_supervisor.state.pause_reason == \
        "three rejected updates or worker restarts within one hour"


def test_refresh_stage_gates_copies_verified_evidence_but_never_enables_learning(tmp_path):
    gates_path = tmp_path / "stage-gates.json"
    report_path = tmp_path / "ppo-stage.json"
    source_path = tmp_path / "checkpoint.pt"
    source_path.write_bytes(b"frozen-checkpoint")
    gates_path.write_text(json.dumps({"schemaVersion": 1, "nextGate": "keep this context",
        "ppoEnabled": False, "continuousOperationEnabled": False, "trustedPromotion": False}))
    report = {
        "kind": "verified-ppo-stage-evidence-v1",
        "sourceArtifacts": {"checkpoint": {"path": str(source_path),
            "sha256": file_sha256(source_path)}},
        "representationParity": True, "supervisedManifestFrozen": True,
        "supervisedCheckpointTrained": True, "heldOutLabelWin": False,
        "heldOutLabelEvidenceStatus": "insufficient",
        "blindOpponentPolicyFamilyStatus": "insufficient",
        "legalActionOmission": False, "illegalAutoregressiveSelection": False,
        "capOverflow": False, "evaluationIdentityStatus": "matched",
        "representativeDisagreementsReviewed": False, "targetProbeWin": False,
        "severityThreeRegression": False, "severityThreeProbeCoverage": "insufficient",
        "ragingBoltMacroPlanFidelity": "passed", "humanEnablePPO": True,
        "prerequisitesPassed": False, "ppoEnabled": False,
    }
    report["reportHash"] = identity_hash(report)
    report_path.write_text(json.dumps(report))

    refreshed = stage_evidence.refresh_stage_gate_evidence(stage_gates_path=gates_path,
        verified_report=report, report_path=report_path)

    assert refreshed["heldOutLabelEvidenceStatus"] == "insufficient"
    assert refreshed["evaluationIdentityStatus"] == "matched"
    assert refreshed["ragingBoltMacroPlanFidelity"] == "passed"
    assert refreshed["nextGate"] == "keep this context"
    assert refreshed["stageEvidenceReport"]["sha256"] == file_sha256(report_path)
    assert refreshed["humanEnablePPO"] is False
    assert refreshed["ppoEnabled"] is False
    assert refreshed["continuousOperationEnabled"] is False
    assert refreshed["trustedPromotion"] is False

    forged = {**report, "heldOutLabelWin": True}
    with pytest.raises(ValueError, match="report hash"):
        stage_evidence.refresh_stage_gate_evidence(stage_gates_path=gates_path,
            verified_report=forged, report_path=report_path)


@pytest.mark.parametrize("ranker_report", [
    {"identity": {"frozen": "same"}, "acceptance": "insufficient",
     "development": {"status": "insufficient"}, "holdouts": []},
    {"identity": {"frozen": "different"}, "acceptance": "review-required",
     "development": {"status": "measured"}, "holdouts": [
        {"kind": "leave-one-opponent-archetype-out", "status": "measured"},
        {"kind": "frozen-policy-family", "status": "measured"}]},
])
def test_stage_evidence_rejects_missing_or_identity_mismatched_ranker(tmp_path, monkeypatch, ranker_report):
    paths = _inputs(tmp_path)
    _patch_verifiers(monkeypatch, paths)
    base = {"identity": {"frozen": "same"}, "acceptance": "review-required",
        "modelSha256": "a" * 64, "reportHash": "b" * 64,
        "confidenceAuditReportHash": "confidence-report-hash",
        "development": {"status": "measured"}, "holdouts": [
            {"kind": "leave-one-opponent-archetype-out", "status": "measured"},
            {"kind": "frozen-policy-family", "status": "measured"}]}
    label_manifest = {"manifestHash": "ranker-label-manifest", "selectionHash": "selection-hash"}
    confidence_report = {"reportHash": "confidence-report-hash",
        "confidenceAuditImplementationSha256": "c" * 64}
    ranker_report = {**base, **ranker_report,
        "inputManifestHash": label_manifest["manifestHash"],
        "inputManifestSha256": file_sha256(paths["ranker_labels_dir"] / "manifest.json"),
        "selectionHash": label_manifest["selectionHash"],
        "selectionManifestSha256": file_sha256(paths["macro_selection_path"]),
        "confidenceAuditReportHash": confidence_report["reportHash"],
        "confidenceAuditSha256": file_sha256(paths["confidence_audit_path"]),
        "confidenceAuditImplementationSha256": confidence_report["confidenceAuditImplementationSha256"]}
    monkeypatch.setattr(stage_evidence, "verify_macro_ranker_v2_artifact", lambda *_args: {
        "report": ranker_report, "artifact": {}})
    with pytest.raises(ValueError, match="macro-ranker"):
        stage_evidence.verify_ppo_stage_evidence(**paths)


def test_stage_evidence_rejects_ranker_from_stale_confidence_audit(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    ranker_report = _patch_verifiers(monkeypatch, paths)
    stale_report = {**ranker_report, "confidenceAuditSha256": "d" * 64}
    monkeypatch.setattr(stage_evidence, "verify_macro_ranker_v2_artifact", lambda *_args: {
        "report": stale_report, "artifact": {}})
    with pytest.raises(ValueError, match="exact labels, selection, and confidence audit"):
        stage_evidence.verify_ppo_stage_evidence(**paths)


@pytest.mark.parametrize("overrides", [
    {"macroRankerDistillationBound": False},
    {"macroRankerMeasuredHoldoutKinds": ["leave-one-opponent-archetype-out"]},
    {"prerequisitesPassed": False},
])
def test_ppo_enablement_rechecks_macro_ranker_gates(overrides):
    from ptcg_lab.learning_mind.training import (VerifiedPPOStageRecord,
        _VERIFIED_PPO_STAGE_TOKEN)
    values = {"prerequisitesPassed": True, "representationParity": True,
        "heldOutLabelWin": True, "heldOutLabelEvidenceStatus": "supported-improvement",
        "blindOpponentPolicyFamilyStatus": "supported-improvement", "targetProbeWin": True,
        "ragingBoltMacroPlanFidelity": "passed", "severityThreeProbeCoverage": "sufficient",
        "severityThreeRegression": False, "legalActionOmission": False,
        "illegalAutoregressiveSelection": False, "capOverflow": False,
        "evaluationIdentityStatus": "matched", "representativeDisagreementsReviewed": True,
        "macroRankerAcceptance": "review-required", "macroRankerDevelopmentStatus": "measured",
        "macroRankerMeasuredHoldoutKinds": ["leave-one-opponent-archetype-out", "frozen-policy-family"],
        "macroRankerDistillationBound": True, "humanEnablePPO": True}
    capability = VerifiedPPOStageRecord({**values, **overrides},
        _verification_token=_VERIFIED_PPO_STAGE_TOKEN)
    assert ppo_enablement(capability)["enabled"] is False


def test_stage_evidence_rejects_checkpoint_without_exact_ranker_distillation(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    _patch_verifiers(monkeypatch, paths)
    monkeypatch.setattr(stage_evidence, "load_dataset", lambda _path: (
        {"manifestHash": "dataset", "identity": {"frozen": "same"}, "teacherHashes": []}, []))
    with pytest.raises(ValueError, match="not trained from this exact macro-ranker distribution"):
        stage_evidence.verify_ppo_stage_evidence(**paths)


def test_stage_evidence_rejects_tampered_recomputed_safety_receipt(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    generated = {"legalActionOmission": False,
        "illegalAutoregressiveSelection": False, "capOverflow": False,
        "reportHash": "fresh"}
    _patch_verifiers(monkeypatch, paths, safety=generated)
    (paths["safety_report_path"]).write_text(json.dumps({**generated, "capOverflow": True}))
    with pytest.raises(ValueError, match="candidate safety report differs"):
        stage_evidence.verify_ppo_stage_evidence(**paths)
