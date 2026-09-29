from __future__ import annotations

import json
from pathlib import Path

import pytest

from ptcg_lab.learning_mind import stage_evidence
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
    monkeypatch.setattr(stage_evidence, "audit_manifest", lambda _path: {
        "scheduled": 192, "audited": 192, "unsupportedPositions": 0,
        "representationParity": True})
    monkeypatch.setattr(stage_evidence, "load_dataset", lambda _path: (
        {"manifestHash": "dataset", "identity": {"frozen": "same"}}, []))
    monkeypatch.setattr(stage_evidence, "verify_supervised_audit_report", lambda **_kwargs: audit)
    monkeypatch.setattr(stage_evidence, "audit_candidate_safety", lambda **_kwargs: safety)
    monkeypatch.setattr(stage_evidence, "evaluate_candidate", lambda *_args: evaluation)
    monkeypatch.setattr(stage_evidence, "audit_raging_bolt_macro_fidelity", lambda **_kwargs: fidelity)
    monkeypatch.setattr(stage_evidence, "audit_disagreement_review", lambda **_kwargs: review)
    (paths["evaluation_path"]).write_text(json.dumps(evaluation))
    (paths["safety_report_path"]).write_text(json.dumps(safety))
    (paths["macro_fidelity_path"]).write_text(json.dumps(fidelity))
    (paths["disagreement_receipt_path"]).write_text(json.dumps(review))


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

    authorized_paths = {**paths, "output": tmp_path / "authorized-stage.json"}
    report, capability = stage_evidence.verify_ppo_stage_evidence(
        **authorized_paths, human_enable_ppo=True)
    assert report["ppoEnabled"] is True
    assert ppo_enablement(capability)["enabled"] is True

    torch = pytest.importorskip("torch")
    from ptcg_lab.learning_mind.encoding import collate
    from ptcg_lab.learning_mind.model import StrategyTransformerV1
    from ptcg_lab.learning_mind.training import (load_ppo_checkpoint, ppo_policy_fingerprint,
        ppo_update, save_ppo_checkpoint)
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
    gradient_hook = nonfinite_model.policy_query[-1].weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan")))
    rejected = ppo_update(nonfinite_model, nonfinite_optimizer, [row], stage_record=capability)
    gradient_hook.remove()
    assert rejected["pauseRequired"] is True
    assert rejected["rejectionReasons"] == {"non-finite-gradient": 1}
    assert rejected["acceptedMinibatches"] == 0
    assert all(torch.equal(before_nonfinite[key], nonfinite_model.state_dict()[key])
               for key in before_nonfinite)


def test_stage_evidence_rejects_tampered_recomputed_safety_receipt(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    generated = {"legalActionOmission": False,
        "illegalAutoregressiveSelection": False, "capOverflow": False,
        "reportHash": "fresh"}
    _patch_verifiers(monkeypatch, paths, safety=generated)
    (paths["safety_report_path"]).write_text(json.dumps({**generated, "capOverflow": True}))
    with pytest.raises(ValueError, match="candidate safety report differs"):
        stage_evidence.verify_ppo_stage_evidence(**paths)
