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
    from ptcg_lab.learning_mind.training import ppo_update
    from test_learning_mind_representation import encoded

    torch.manual_seed(4)
    model = StrategyTransformerV1()
    decision = encoded()
    batch = {key: torch.as_tensor(value) for key, value in collate([decision]).items()}
    with torch.no_grad():
        logits = model.policy_forward(**batch)
        old_log_prob = torch.log_softmax(logits, -1)[0, 0].item()
        value = model.evaluation_forward(**{key: batch[key] for key in
            ("state_card_ids", "state_features", "state_type_ids", "state_mask")})[0].item()
    row = {"episodeId": "completed-0", "episodeStatus": "finished", "episodeEnd": True,
        "reward": 1, "encoded": decision, "selectedAction": 0, "oldLogProb": old_log_prob,
        "return": value, "advantage": 1.}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    update = ppo_update(model, optimizer, [row], stage_record=capability)
    assert update["acceptedMinibatches"] == 1 and update["optimizationEpochs"] == 1


def test_stage_evidence_rejects_tampered_recomputed_safety_receipt(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    generated = {"legalActionOmission": False,
        "illegalAutoregressiveSelection": False, "capOverflow": False,
        "reportHash": "fresh"}
    _patch_verifiers(monkeypatch, paths, safety=generated)
    (paths["safety_report_path"]).write_text(json.dumps({**generated, "capOverflow": True}))
    with pytest.raises(ValueError, match="candidate safety report differs"):
        stage_evidence.verify_ppo_stage_evidence(**paths)
