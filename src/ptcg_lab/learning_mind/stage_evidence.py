from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .audit import audit_manifest
from .candidate_safety import audit_candidate_safety
from .confidence_audit import verify_macro_label_confidence_audit
from .dataset_v1 import file_sha256, load_dataset
from .disagreement_review import audit_disagreement_review
from .experiment import _load_ranker_input
from .macro_fidelity import audit_raging_bolt_macro_fidelity
from .policy_evaluation import evaluate_candidate
from .ranker_v2 import verify_macro_ranker_v2_artifact
from .schema import identity_hash
from .supervised_evidence import verify_supervised_audit_report
from .training import VerifiedPPOStageRecord, _VERIFIED_PPO_STAGE_TOKEN


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence artifact must contain a JSON object: {path}")
    return value


def _require_same_report(path: Path, generated: dict, label: str) -> dict:
    supplied = _read_json(path)
    if supplied != generated:
        raise ValueError(f"{label} differs from a fresh evidence recomputation")
    return supplied


def _write_immutable(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError("stage evidence reports are immutable; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


_STAGE_GATE_EVIDENCE_FIELDS = (
    "representationParity", "supervisedManifestFrozen", "supervisedCheckpointTrained",
    "heldOutLabelWin", "heldOutLabelEvidenceStatus", "blindOpponentPolicyFamilyStatus",
    "legalActionOmission", "illegalAutoregressiveSelection", "capOverflow",
    "evaluationIdentityStatus", "representativeDisagreementsReviewed", "targetProbeWin",
    "severityThreeRegression", "severityThreeProbeCoverage", "ragingBoltMacroPlanFidelity",
)


def refresh_stage_gate_evidence(*, stage_gates_path: Path, verified_report: dict,
        report_path: Path) -> dict:
    """Refresh evidence-derived gate fields without enabling any learning stage.

    Call only with the report returned by ``verify_ppo_stage_evidence`` in the same
    process. Authorization fields are always forced off; this is not an enablement
    operation or a substitute for the verifier's source-artifact recomputation.
    """
    gates_path = Path(stage_gates_path).resolve()
    report_file = Path(report_path).resolve()
    report = dict(verified_report)
    report_hash = report.pop("reportHash", None)
    if not isinstance(report_hash, str) or report_hash != identity_hash(report):
        raise ValueError("stage report hash is missing or invalid")
    if report.get("kind") != "verified-ppo-stage-evidence-v1":
        raise ValueError("stage report is not verified PPO-stage evidence")
    if not all(field in report for field in _STAGE_GATE_EVIDENCE_FIELDS):
        raise ValueError("verified stage report is missing gate evidence fields")
    if not isinstance(report.get("sourceArtifacts"), dict) or not report["sourceArtifacts"]:
        raise ValueError("verified stage report lacks source-artifact receipts")
    if not report_file.is_file() or json.loads(report_file.read_text()) != verified_report:
        raise ValueError("stage report file does not match the verified report object")
    for name, receipt in report["sourceArtifacts"].items():
        if (not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
                or not isinstance(receipt.get("sha256"), str)):
            raise ValueError(f"stage report has an invalid source-artifact receipt: {name}")
        source = Path(receipt["path"])
        if not source.is_file() or file_sha256(source) != receipt["sha256"]:
            raise ValueError(f"stage report source artifact changed since verification: {name}")

    gates = _read_json(gates_path)
    for field in _STAGE_GATE_EVIDENCE_FIELDS:
        gates[field] = report[field]
    gates["stageEvidenceReport"] = {
        "path": os.path.relpath(report_file, gates_path.parent),
        "sha256": file_sha256(report_file),
        "reportHash": report_hash,
    }
    # Evidence refresh must never authorize a run, even if the verifier was invoked
    # with its separate human-authorization flag.
    gates["humanEnablePPO"] = False
    gates["ppoEnabled"] = False
    gates["continuousOperationEnabled"] = False
    gates["trustedPromotion"] = False

    gates_path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = gates_path.stat().st_mode & 0o777 if gates_path.exists() else 0o644
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=gates_path.parent,
                                     prefix=f".{gates_path.name}.", delete=False) as temporary:
        temporary.write(json.dumps(gates, sort_keys=True, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        temporary_path.chmod(existing_mode)
        os.replace(temporary_path, gates_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return gates


def _audit_frozen_baseline(root: Path, manifest_path: Path) -> dict:
    manifest = _read_json(manifest_path)
    replays = manifest.get("replays")
    if not isinstance(replays, list) or len(replays) != 192:
        raise ValueError("PPO stage requires the complete frozen 192-replay baseline manifest")
    resolved = []
    for replay in replays:
        if not isinstance(replay, dict) or not isinstance(replay.get("path"), str):
            raise ValueError("frozen baseline manifest contains an invalid replay entry")
        item = dict(replay)
        replay_path = Path(item["path"])
        item["path"] = str((replay_path if replay_path.is_absolute() else root / replay_path).resolve())
        resolved.append(item)
    with tempfile.TemporaryDirectory(prefix="learning-mind-baseline-verify-") as temporary:
        normalized = Path(temporary) / "baseline-manifest.json"
        normalized.write_text(json.dumps({**manifest, "replays": resolved}))
        return audit_manifest(normalized)


def verify_ppo_stage_evidence(*, root: Path, baseline_manifest: Path,
        dataset_dir: Path, probe_dataset_dir: Path, checkpoint: Path,
        evaluation_path: Path, supervised_audit_path: Path, safety_report_path: Path,
        macro_selection_path: Path, python_dataset: Path, typescript_dataset: Path,
        python_labels: Path, typescript_labels: Path, macro_fidelity_path: Path,
        ranker_labels_dir: Path, confidence_audit_path: Path,
        ranker_model_path: Path, ranker_report_path: Path,
        disagreement_packet_path: Path, disagreement_review_path: Path,
        disagreement_receipt_path: Path, output: Path,
        human_enable_ppo: bool = False) -> tuple[dict, VerifiedPPOStageRecord]:
    """Recompute every supervised PPO prerequisite from its immutable source artifacts.

    The generated record is a report, not a bearer credential. The returned in-process
    capability is issued only after every source report has been regenerated and matched.
    Human enablement is a separate explicit input and is never inferred from the reports.
    """
    paths = [Path(value).resolve() for value in (
        root, baseline_manifest, dataset_dir, probe_dataset_dir, checkpoint,
        evaluation_path, supervised_audit_path, safety_report_path,
        macro_selection_path, python_dataset, typescript_dataset, python_labels,
        typescript_labels, macro_fidelity_path, ranker_labels_dir, confidence_audit_path,
        ranker_model_path, ranker_report_path,
        disagreement_packet_path,
        disagreement_review_path, disagreement_receipt_path)]
    (root, baseline_manifest, dataset_dir, probe_dataset_dir, checkpoint,
     evaluation_path, supervised_audit_path, safety_report_path,
     macro_selection_path, python_dataset, typescript_dataset, python_labels,
     typescript_labels, macro_fidelity_path, ranker_labels_dir, confidence_audit_path,
     ranker_model_path, ranker_report_path,
     disagreement_packet_path,
     disagreement_review_path, disagreement_receipt_path) = paths
    if type(human_enable_ppo) is not bool:
        raise ValueError("human PPO authorization must be an explicit boolean")

    baseline = _audit_frozen_baseline(root, baseline_manifest)
    if (baseline.get("scheduled") != 192 or baseline.get("audited") != 192
            or baseline.get("unsupportedPositions") != 0
            or baseline.get("representationParity") is not True):
        raise ValueError("frozen baseline replay representation audit did not pass")

    manifest, _rows = load_dataset(dataset_dir)
    audit = verify_supervised_audit_report(dataset_dir=dataset_dir,
        checkpoint=checkpoint, evaluation_path=evaluation_path, audit_path=supervised_audit_path)
    with tempfile.TemporaryDirectory(prefix="learning-mind-safety-verify-") as temporary:
        generated_safety = audit_candidate_safety(dataset_dir=dataset_dir, checkpoint=checkpoint,
            evaluation_path=evaluation_path, audit_path=supervised_audit_path,
            output=Path(temporary) / "safety.json")
    safety = _require_same_report(safety_report_path, generated_safety, "candidate safety report")

    generated_evaluation = evaluate_candidate(dataset_dir, checkpoint, probe_dataset_dir)
    evaluation = _read_json(evaluation_path)
    if evaluation != generated_evaluation:
        raise ValueError("frozen supervised evaluation differs from a fresh model/probe evaluation")

    with tempfile.TemporaryDirectory(prefix="learning-mind-macro-fidelity-verify-") as temporary:
        generated_fidelity = audit_raging_bolt_macro_fidelity(root=root,
            selection_path=macro_selection_path, python_dataset=python_dataset,
            typescript_dataset=typescript_dataset, python_labels=python_labels,
            typescript_labels=typescript_labels, output=Path(temporary) / "fidelity.json")
    fidelity = _require_same_report(macro_fidelity_path, generated_fidelity,
                                    "Raging Bolt macro-fidelity report")
    if fidelity.get("identity") != manifest.get("identity"):
        raise ValueError("macro-fidelity and supervised evidence have different frozen identities")

    ranker_manifest, _ranker_records = _load_ranker_input(ranker_labels_dir, macro_selection_path)
    confidence = verify_macro_label_confidence_audit(labels_dir=ranker_labels_dir,
        selection_path=macro_selection_path, report_path=confidence_audit_path)
    ranker = verify_macro_ranker_v2_artifact(ranker_model_path, ranker_report_path)["report"]
    if ranker.get("identity") != manifest.get("identity"):
        raise ValueError("macro-ranker and supervised evidence have different frozen identities")
    if (ranker.get("inputManifestHash") != ranker_manifest.get("manifestHash")
            or ranker.get("inputManifestSha256") != file_sha256(ranker_labels_dir / "manifest.json")
            or ranker.get("selectionHash") != ranker_manifest.get("selectionHash")
            or ranker.get("selectionManifestSha256") != file_sha256(macro_selection_path)
            or ranker.get("confidenceAuditReportHash") != confidence.get("reportHash")
            or ranker.get("confidenceAuditSha256") != file_sha256(confidence_audit_path)
            or ranker.get("confidenceAuditImplementationSha256") !=
                confidence.get("confidenceAuditImplementationSha256")):
        raise ValueError("macro-ranker is not bound to the exact labels, selection, and confidence audit")
    expected_teacher_hashes = sorted({ranker["modelSha256"], ranker["reportHash"],
                                      confidence["reportHash"]})
    if manifest.get("teacherHashes") != expected_teacher_hashes:
        raise ValueError("supervised checkpoint was not trained from this exact macro-ranker distribution")
    holdouts = ranker.get("holdouts")
    required_holdouts = {"leave-one-opponent-archetype-out", "frozen-policy-family"}
    measured_holdouts = {row.get("kind") for row in holdouts if row.get("status") == "measured"}
    if (ranker.get("acceptance") != "review-required"
            or ranker.get("development", {}).get("status") != "measured"
            or not required_holdouts.issubset(measured_holdouts)):
        raise ValueError("macro-ranker development and both frozen holdout axes must be measured")

    with tempfile.TemporaryDirectory(prefix="learning-mind-review-verify-") as temporary:
        generated_review = audit_disagreement_review(packet_path=disagreement_packet_path,
            review_path=disagreement_review_path, output=Path(temporary) / "review.json")
    review = _require_same_report(disagreement_receipt_path, generated_review,
                                  "human disagreement-review receipt")

    probe_results = evaluation.get("strategyProbes")
    if not isinstance(probe_results, dict):
        raise ValueError("frozen evaluation lacks recomputed strategy-probe results")
    heldout_win = audit.get("status") == "supported-improvement"
    blind_status = audit.get("blindOpponentPolicyFamilyStatus")
    probe_win = probe_results.get("targetProbeWin") is True
    severity_regression = probe_results.get("severityThreeRegression")
    severity_coverage = probe_results.get("severityThreeCoverage")
    sources = {
        "baselineManifest": baseline_manifest, "datasetManifest": dataset_dir / "manifest.json",
        "probeDatasetManifest": probe_dataset_dir / "manifest.json", "checkpoint": checkpoint,
        "evaluation": evaluation_path, "supervisedAudit": supervised_audit_path,
        "candidateSafety": safety_report_path, "macroSelection": macro_selection_path,
        "pythonDatasetManifest": python_dataset / "manifest.json",
        "typescriptDatasetManifest": typescript_dataset / "manifest.json",
        "pythonLabelsManifest": python_labels / "manifest.json",
        "typescriptLabelsManifest": typescript_labels / "manifest.json",
        "macroFidelity": macro_fidelity_path, "disagreementPacket": disagreement_packet_path,
        "rankerLabelsManifest": ranker_labels_dir / "manifest.json",
        "confidenceAudit": confidence_audit_path,
        "macroRankerModel": ranker_model_path, "macroRankerReport": ranker_report_path,
        "disagreementReview": disagreement_review_path, "disagreementReceipt": disagreement_receipt_path,
    }
    source_hashes = {name: {"path": str(path), "sha256": file_sha256(path)}
                     for name, path in sorted(sources.items())}
    values = {
        "schemaVersion": 1, "kind": "verified-ppo-stage-evidence-v1",
        "sourceArtifacts": source_hashes,
        "representationParity": True,
        "supervisedManifestFrozen": True,
        "supervisedCheckpointTrained": True,
        "heldOutLabelWin": heldout_win,
        "heldOutLabelEvidenceStatus": audit.get("status"),
        "blindOpponentPolicyFamilyStatus": blind_status,
        "legalActionOmission": safety.get("legalActionOmission"),
        "illegalAutoregressiveSelection": safety.get("illegalAutoregressiveSelection"),
        "capOverflow": safety.get("capOverflow"),
        "evaluationIdentityStatus": audit.get("evaluationIdentityStatus"),
        "representativeDisagreementsReviewed": review.get("representativeDisagreementsReviewed"),
        "targetProbeWin": probe_win,
        "severityThreeRegression": severity_regression,
        "severityThreeProbeCoverage": severity_coverage,
        "ragingBoltMacroPlanFidelity": fidelity.get("ragingBoltMacroPlanFidelity"),
        "macroRankerAcceptance": ranker.get("acceptance"),
        "macroRankerDevelopmentStatus": ranker["development"].get("status"),
        "macroRankerMeasuredHoldoutKinds": sorted(measured_holdouts),
        "macroRankerDistillationBound": manifest.get("teacherHashes") == expected_teacher_hashes,
        "humanEnablePPO": human_enable_ppo,
        "automaticPromotion": False,
    }
    prerequisites = (values["representationParity"] is True
        and values["heldOutLabelWin"] is True
        and values["heldOutLabelEvidenceStatus"] == "supported-improvement"
        and values["blindOpponentPolicyFamilyStatus"] == "supported-improvement"
        and values["targetProbeWin"] is True
        and values["ragingBoltMacroPlanFidelity"] == "passed"
        and values["macroRankerAcceptance"] == "review-required"
        and values["macroRankerDevelopmentStatus"] == "measured"
        and values["macroRankerDistillationBound"] is True
        and required_holdouts.issubset(values["macroRankerMeasuredHoldoutKinds"])
        and values["severityThreeProbeCoverage"] == "sufficient"
        and values["severityThreeRegression"] is False
        and values["legalActionOmission"] is False
        and values["illegalAutoregressiveSelection"] is False
        and values["capOverflow"] is False
        and values["evaluationIdentityStatus"] == "matched"
        and values["representativeDisagreementsReviewed"] is True)
    values["prerequisitesPassed"] = prerequisites
    values["ppoEnabled"] = prerequisites and human_enable_ppo
    report = dict(values)
    report["reportHash"] = identity_hash(values)
    _write_immutable(Path(output), report)

    record = dict(values)
    capability = VerifiedPPOStageRecord(record, _verification_token=_VERIFIED_PPO_STAGE_TOKEN)
    return report, capability
