from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import time

import pytest

from ptcg_lab.learning_mind.continuous_evidence import (
    _REQUIRED_DRILLS, audit_supervised_soak, issue_continuous_operation_evidence)
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.notifications import VerifiedPromotionEvidence, _VERIFIED_PROMOTION_TOKEN
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.learning_mind.specialist_evidence import (
    VerifiedSpecialistCurriculumEvidence, _VERIFIED_SPECIALIST_TOKEN)
from ptcg_lab.learning_mind.supervisor import continuous_operation_enablement
from ptcg_lab.learning_mind.supervisor import MindSupervisor
from ptcg_lab.learning_mind.training import (
    VerifiedPPOStageRecord, _VERIFIED_PPO_STAGE_TOKEN, _thaw_evidence)


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identities():
    candidate_hash = "d" * 64
    stage = VerifiedPPOStageRecord({"prerequisitesPassed": True,
        "representationParity": True, "heldOutLabelWin": True,
        "heldOutLabelEvidenceStatus": "supported-improvement",
        "blindOpponentPolicyFamilyStatus": "supported-improvement", "targetProbeWin": True,
        "ragingBoltMacroPlanFidelity": "passed", "macroRankerAcceptance": "review-required",
        "macroRankerDevelopmentStatus": "measured",
        "macroRankerMeasuredHoldoutKinds": ["leave-one-opponent-archetype-out", "frozen-policy-family"],
        "macroRankerDistillationBound": True, "severityThreeProbeCoverage": "sufficient",
        "severityThreeRegression": False, "legalActionOmission": False,
        "illegalAutoregressiveSelection": False, "capOverflow": False,
        "evaluationIdentityStatus": "matched", "representativeDisagreementsReviewed": True,
        "humanEnablePPO": True}, _verification_token=_VERIFIED_PPO_STAGE_TOKEN)
    promotion = VerifiedPromotionEvidence({"promotionCriteriaPassed": True,
        "humanApproved": True, "automaticPromotion": False,
        "candidateCheckpointSha256": candidate_hash, "sourceReportHash": "a" * 64},
        _verification_token=_VERIFIED_PROMOTION_TOKEN)
    specialists = VerifiedSpecialistCurriculumEvidence({"specialistCurriculumPassed": True,
        "automaticPromotion": False, "generalistCheckpointSha256": candidate_hash,
        "sourceReportHash": "b" * 64}, _verification_token=_VERIFIED_SPECIALIST_TOKEN)
    bindings = {"ppoStageEvidenceHash": identity_hash(_thaw_evidence(stage._values)),
        "promotionReportHash": promotion._values["sourceReportHash"],
        "specialistReportHash": specialists._values["sourceReportHash"],
        "candidateCheckpointSha256": candidate_hash}
    return stage, promotion, specialists, bindings


def _write_soak(root, bindings, *, start=None):
    start = time.time() - 24 * 60 * 60 - 10 if start is None else start
    root.mkdir(parents=True, exist_ok=True)
    end = start + 24 * 60 * 60
    samples = []
    for index in range(25):
        state_path = root / "snapshots" / f"state-{index:02}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"schemaVersion": 1, "status": "RUNNING",
            "phase": "collection", "cursor": {"sample": index}, "failures": [],
            "pause_reason": None}, sort_keys=True))
        samples.append({"at": start + index * 3600,
            "statePath": str(state_path.relative_to(root)), "stateSha256": _sha(state_path)})
    drills = []
    for kind in sorted(_REQUIRED_DRILLS):
        receipt = root / "drills" / f"{kind}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"kind": kind, "status": "passed"}, sort_keys=True))
        drills.append({"kind": kind, "status": "passed",
            "artifactPath": str(receipt.relative_to(root)), "artifactSha256": _sha(receipt)})
    manifest = {"schemaVersion": 1, "kind": "learning-mind-supervised-soak-v1",
        "bindings": bindings, "startedAt": _iso(start), "endedAt": _iso(end),
        "samples": samples, "drills": drills}
    manifest_path = root / "soak-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path, end


def test_supervised_soak_audit_binds_24_hour_trace_and_issues_continuous_capability(tmp_path):
    stage, promotion, specialists, bindings = _identities()
    root = tmp_path / "soak"
    manifest_path, ended = _write_soak(root, bindings)
    soak_report, soak = audit_supervised_soak(root=root, manifest_path=manifest_path,
        output=tmp_path / "soak-report.json", human_reviewed=True)
    assert soak_report["status"] == "passed"
    assert soak_report["durationSeconds"] == 24 * 60 * 60
    assert soak_report["sampleCount"] == 25
    assert len(soak_report["drills"]) == 7

    report, capability = issue_continuous_operation_evidence(stage_record=stage,
        promotion=promotion, specialists=specialists, soak=soak, human_enable=True,
        output=tmp_path / "continuous-report.json")
    assert report["automaticPromotion"] is False
    assert report["reportHash"] == identity_hash({key: value for key, value in report.items()
                                                     if key != "reportHash"})
    assert continuous_operation_enablement(capability)["enabled"] is True
    supervisor = MindSupervisor(tmp_path / "supervisor", reserve_bytes=0, data_cap_bytes=100_000)
    supervisor.start(human_enabled=True, stage_record=capability)
    assert supervisor.state.status == "RUNNING"
    supervisor.pause("test complete", notify=False)
    supervisor.close()


def test_soak_audit_requires_human_review_and_rejects_tampered_snapshot(tmp_path):
    _stage, _promotion, _specialists, bindings = _identities()
    root = tmp_path / "soak"
    manifest_path, ended = _write_soak(root, bindings)
    with pytest.raises(PermissionError, match="human must explicitly review"):
        audit_supervised_soak(root=root, manifest_path=manifest_path,
            output=tmp_path / "unreviewed.json", human_reviewed=False)

    snapshot = root / "snapshots" / "state-00.json"
    snapshot.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit_supervised_soak(root=root, manifest_path=manifest_path,
            output=tmp_path / "tampered.json", human_reviewed=True)
    linked_manifest = root / "manifest-link.json"
    linked_manifest.symlink_to(manifest_path)
    with pytest.raises(ValueError, match="regular file under the evidence root"):
        audit_supervised_soak(root=root, manifest_path=linked_manifest,
            output=tmp_path / "symlink-manifest.json", human_reviewed=True)


def test_soak_audit_rejects_gaps_missing_drills_and_path_escape(tmp_path):
    _stage, _promotion, _specialists, bindings = _identities()
    root = tmp_path / "soak"
    manifest_path, ended = _write_soak(root, bindings)
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"][12]["at"] += 3601
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="span the full run"):
        audit_supervised_soak(root=root, manifest_path=manifest_path,
            output=tmp_path / "gap.json", human_reviewed=True)

    manifest_path, _ended = _write_soak(root / "duplicate-time", bindings)
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"][1]["at"] = manifest["samples"][0]["at"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="span the full run"):
        audit_supervised_soak(root=root / "duplicate-time", manifest_path=manifest_path,
            output=tmp_path / "duplicate-time.json", human_reviewed=True)

    manifest_path, ended = _write_soak(root / "new-run", bindings)
    manifest = json.loads(manifest_path.read_text())
    manifest["drills"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly the seven"):
        audit_supervised_soak(root=root / "new-run", manifest_path=manifest_path,
            output=tmp_path / "missing-drill.json", human_reviewed=True)

    manifest_path, ended = _write_soak(root / "escaped", bindings)
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"][0]["statePath"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path or hash is invalid"):
        audit_supervised_soak(root=root / "escaped", manifest_path=manifest_path,
            output=tmp_path / "escape.json", human_reviewed=True)


def test_continuous_capability_rejects_stale_bindings_and_human_gate(tmp_path):
    stage, promotion, specialists, bindings = _identities()
    root = tmp_path / "soak"
    bad_bindings = {**bindings, "specialistReportHash": "e" * 64}
    manifest_path, ended = _write_soak(root, bad_bindings)
    _report, soak = audit_supervised_soak(root=root, manifest_path=manifest_path,
        output=tmp_path / "soak-report.json", human_reviewed=True)
    with pytest.raises(ValueError, match="identities differ"):
        issue_continuous_operation_evidence(stage_record=stage, promotion=promotion,
            specialists=specialists, soak=soak, human_enable=True,
            output=tmp_path / "stale.json")
    with pytest.raises(PermissionError, match="human must explicitly enable"):
        issue_continuous_operation_evidence(stage_record=stage, promotion=promotion,
            specialists=specialists, soak=soak, human_enable=False,
            output=tmp_path / "not-enabled.json")
