from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType

from .notifications import VerifiedPromotionEvidence
from .promotion_evidence import reissue_promotion_evidence_report
from .schema import identity_hash
from .specialist_evidence import (VerifiedSpecialistCurriculumEvidence,
    verify_specialist_curriculum)
from .stage_evidence import verify_ppo_stage_evidence
from .supervisor import (FAILURE_KINDS, VerifiedContinuousOperationRecord,
    _VERIFIED_CONTINUOUS_TOKEN)
from .training import (VerifiedPPOStageRecord, _thaw_evidence, ppo_enablement)


_VERIFIED_SOAK_TOKEN = object()
_PHASES = frozenset({"collection", "training", "evaluation", "retention"})
_REQUIRED_DRILLS = frozenset({"pause", "disk-exhaustion", "corrupted-replay",
    "worker-restart", "rejected-update", "notification", "rollback"})
_BINDING_FIELDS = frozenset({"ppoStageEvidenceHash", "promotionReportHash",
    "specialistReportHash", "candidateCheckpointSha256"})


class VerifiedSupervisedSoakEvidence:
    """In-process receipt for a hash-checked and human-reviewed 24-hour soak."""
    __slots__ = ("_values",)

    def __init__(self, values: dict, *, _verification_token: object):
        if _verification_token is not _VERIFIED_SOAK_TOKEN:
            raise TypeError("supervised soak evidence must come from its artifact verifier")
        frozen = dict(values)
        frozen["bindings"] = MappingProxyType(dict(values["bindings"]))
        object.__setattr__(self, "_values", MappingProxyType(frozen))

    def __setattr__(self, _name, _value):
        raise AttributeError("verified supervised soak evidence is immutable")


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read_timestamp(value: object, label: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"soak {label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"soak {label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"soak {label} must include a timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def _artifact_bytes(root: Path, relative: object, expected_sha256: object,
                    label: str) -> tuple[Path, bytes]:
    relative_path = Path(relative) if isinstance(relative, str) else Path("/")
    if (not isinstance(relative, str) or not relative or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not _is_sha256(expected_sha256)):
        raise ValueError(f"soak {label} artifact path or hash is invalid")
    path = root / relative_path
    resolved = path.resolve()
    if (any((root / Path(*relative_path.parts[:index])).is_symlink()
            for index in range(1, len(relative_path.parts) + 1))
            or not path.is_file() or root not in resolved.parents):
        raise ValueError(f"soak {label} artifact is missing or outside the evidence root")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"soak {label} artifact checksum mismatch")
    return path, content


def _publish(path: Path, report: dict) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("continuous-operation evidence is immutable; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
        json.dump(report, temporary, sort_keys=True, indent=2, allow_nan=False)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def audit_supervised_soak(*, root: Path, manifest_path: Path, output: Path,
                          human_reviewed: bool):
    """Verify a 24-hour supervisor trace, state snapshots, and seven drill receipts.

    This audits evidence only; it never starts the supervisor or performs a drill.
    """
    root = Path(root).resolve()
    manifest_source_path = Path(manifest_path)
    manifest_is_symlink = manifest_source_path.is_symlink()
    manifest_path = manifest_source_path.resolve()
    output = Path(output).resolve()
    if type(human_reviewed) is not bool or not human_reviewed:
        raise PermissionError("a human must explicitly review the completed supervised soak")
    if output.exists():
        raise ValueError("continuous-operation evidence is immutable; choose a new output path")
    if manifest_is_symlink or not manifest_path.is_file() or root not in manifest_path.parents:
        raise ValueError("soak manifest must be a regular file under the evidence root")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("soak manifest is unreadable") from error
    expected_keys = {"schemaVersion", "kind", "bindings", "startedAt", "endedAt", "samples", "drills"}
    if (not isinstance(manifest, dict) or set(manifest) != expected_keys
            or type(manifest.get("schemaVersion")) is not int or manifest["schemaVersion"] != 1
            or manifest.get("kind") != "learning-mind-supervised-soak-v1"):
        raise ValueError("unsupported supervised-soak manifest schema")
    bindings = manifest.get("bindings")
    if (not isinstance(bindings, dict) or set(bindings) != _BINDING_FIELDS
            or any(not _is_sha256(value) for value in bindings.values())):
        raise ValueError("soak manifest must bind the exact PPO, promotion, specialist, and candidate identities")
    started, ended = _read_timestamp(manifest.get("startedAt"), "start time"), _read_timestamp(
        manifest.get("endedAt"), "end time")
    now = time.time()
    if (isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now)
            or ended <= started or ended - started < 24 * 60 * 60 or ended > now + 60):
        raise ValueError("soak must cover at least 24 elapsed hours and have ended no later than now")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) < 25:
        raise ValueError("soak requires at least 25 hourly running-state snapshots")
    sample_times = []
    sample_paths = set()
    sample_receipts = []
    for sample in samples:
        if (not isinstance(sample, dict) or set(sample) != {"at", "statePath", "stateSha256"}
                or type(sample.get("at")) not in (int, float) or not math.isfinite(sample["at"])):
            raise ValueError("soak snapshot entry is malformed")
        _snapshot_path, snapshot_bytes = _artifact_bytes(
            root, sample["statePath"], sample["stateSha256"], "state snapshot")
        if sample["statePath"] in sample_paths:
            raise ValueError("soak snapshot paths must be unique for every observation")
        sample_paths.add(sample["statePath"])
        try:
            state = json.loads(snapshot_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("soak state snapshot is unreadable") from error
        if (not isinstance(state, dict)
                or set(state) != {"schemaVersion", "status", "phase", "cursor", "failures", "pause_reason"}
                or type(state.get("schemaVersion")) is not int or state["schemaVersion"] != 1
                or state.get("status") != "RUNNING" or state.get("phase") not in _PHASES
                or not isinstance(state.get("cursor"), dict)
                or not isinstance(state.get("failures"), list) or state.get("pause_reason") is not None):
            raise ValueError("soak snapshot does not prove a running supervisor with a valid cursor")
        try:
            json.dumps(state["cursor"], allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("soak snapshot cursor contains non-finite or non-JSON state") from error
        for failure in state["failures"]:
            if (not isinstance(failure, dict) or set(failure) != {"time", "kind"}
                    or type(failure.get("time")) not in (int, float)
                    or not math.isfinite(failure["time"])
                    or failure.get("kind") not in FAILURE_KINDS):
                raise ValueError("soak snapshot failure history is malformed")
        sample_times.append(float(sample["at"]))
        sample_receipts.append({"at": sample["at"], "statePath": sample["statePath"],
                                "stateSha256": sample["stateSha256"]})
    if (sample_times != sorted(sample_times) or len(set(sample_times)) != len(sample_times)
            or sample_times[0] > started + 60
            or sample_times[-1] < ended - 60
            or sample_times[0] < started - 60 or sample_times[-1] > ended + 60
            or any(later - earlier > 60 * 60 for earlier, later in zip(sample_times, sample_times[1:]))):
        raise ValueError("soak state snapshots must span the full run with no gap over one hour")
    drills = manifest.get("drills")
    if not isinstance(drills, list) or len(drills) != len(_REQUIRED_DRILLS):
        raise ValueError("soak must include exactly the seven required failure drills")
    verified_drills = []
    seen_drills = set()
    for drill in drills:
        if (not isinstance(drill, dict) or set(drill) != {"kind", "status", "artifactPath", "artifactSha256"}
                or drill.get("kind") not in _REQUIRED_DRILLS or drill.get("status") != "passed"
                or drill["kind"] in seen_drills):
            raise ValueError("soak failure-drill receipt is missing, duplicated, or not passed")
        _receipt, receipt_bytes = _artifact_bytes(
            root, drill["artifactPath"], drill["artifactSha256"], drill["kind"])
        try:
            receipt_value = json.loads(receipt_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"soak {drill['kind']} drill receipt is unreadable") from error
        if (not isinstance(receipt_value, dict) or receipt_value.get("kind") != drill["kind"]
                or receipt_value.get("status") != "passed"):
            raise ValueError(f"soak {drill['kind']} drill receipt does not confirm a pass")
        seen_drills.add(drill["kind"])
        verified_drills.append({"kind": drill["kind"], "status": "passed",
            "artifactPath": drill["artifactPath"], "artifactSha256": drill["artifactSha256"]})
    if seen_drills != _REQUIRED_DRILLS:
        raise ValueError("soak does not cover every required failure drill")
    report = {"schemaVersion": 1, "kind": "verified-learning-mind-supervised-soak-v1",
        "status": "passed", "durationSeconds": ended - started,
        "startedAt": manifest["startedAt"], "endedAt": manifest["endedAt"],
        "humanReviewedSoak": True, "failureDrillsPassed": True,
        "sampleCount": len(sample_receipts), "maximumSnapshotGapSeconds": max(
            later - earlier for earlier, later in zip(sample_times, sample_times[1:])),
        "bindings": bindings, "samples": sample_receipts,
        "drills": sorted(verified_drills, key=lambda item: item["kind"]),
        "sourceManifestPath": str(manifest_path),
        "sourceManifestSha256": hashlib.sha256(manifest_bytes).hexdigest()}
    report["reportHash"] = identity_hash(report)
    _publish(output, report)
    values = {"reportHash": report["reportHash"], "durationSeconds": report["durationSeconds"],
        "humanReviewedSoak": True, "failureDrillsPassed": True, "bindings": bindings}
    return report, VerifiedSupervisedSoakEvidence(values, _verification_token=_VERIFIED_SOAK_TOKEN)


def issue_continuous_operation_evidence(*, stage_record: VerifiedPPOStageRecord,
        promotion: VerifiedPromotionEvidence, specialists: VerifiedSpecialistCurriculumEvidence,
        soak: VerifiedSupervisedSoakEvidence, human_enable: bool, output: Path):
    """Issue the final continuous-operation capability only from matching verified evidence."""
    if type(human_enable) is not bool or not human_enable:
        raise PermissionError("a human must explicitly enable continuous operation for this run")
    if not ppo_enablement(stage_record)["enabled"]:
        raise PermissionError("continuous operation requires a passed, human-enabled PPO stage")
    if (not isinstance(promotion, VerifiedPromotionEvidence)
            or promotion._values.get("promotionCriteriaPassed") is not True
            or promotion._values.get("humanApproved") is not True
            or promotion._values.get("automaticPromotion") is not False):
        raise PermissionError("continuous operation requires human-approved promotion evidence")
    if (not isinstance(specialists, VerifiedSpecialistCurriculumEvidence)
            or specialists._values.get("specialistCurriculumPassed") is not True
            or specialists._values.get("automaticPromotion") is not False
            or specialists._values.get("generalistCheckpointSha256")
                != promotion._values.get("candidateCheckpointSha256")):
        raise PermissionError("continuous operation requires the verified exact-deck specialist curriculum")
    if (not isinstance(soak, VerifiedSupervisedSoakEvidence)
            or soak._values.get("humanReviewedSoak") is not True
            or soak._values.get("failureDrillsPassed") is not True
            or type(soak._values.get("durationSeconds")) not in (int, float)
            or soak._values["durationSeconds"] < 24 * 60 * 60):
        raise PermissionError("continuous operation requires a verified, human-reviewed 24-hour soak and drills")
    expected_bindings = {"ppoStageEvidenceHash": identity_hash(_thaw_evidence(stage_record._values)),
        "promotionReportHash": promotion._values.get("sourceReportHash"),
        "specialistReportHash": specialists._values.get("sourceReportHash"),
        "candidateCheckpointSha256": promotion._values.get("candidateCheckpointSha256")}
    if soak._values.get("bindings") != expected_bindings:
        raise ValueError("supervised soak identities differ from PPO, promotion, specialist, or candidate evidence")
    values = {"ppoEnabled": True, "promotionCriteriaPassed": True,
        "specialistCurriculumPassed": True, "supervised24HourSoakPassed": True,
        "failureDrillsPassed": True, "humanReviewedSoak": True,
        "continuousOperationEnabled": True, "humanEnableContinuousOperation": True,
        "automaticPromotion": False, "ppoStageEvidenceHash": expected_bindings["ppoStageEvidenceHash"],
        "promotionReportHash": expected_bindings["promotionReportHash"],
        "specialistReportHash": expected_bindings["specialistReportHash"],
        "soakReportHash": soak._values["reportHash"]}
    report = {"schemaVersion": 1, "kind": "verified-learning-mind-continuous-operation-v1",
        **values}
    report["reportHash"] = identity_hash(report)
    _publish(Path(output), report)
    capability = VerifiedContinuousOperationRecord(values,
        _verification_token=_VERIFIED_CONTINUOUS_TOKEN)
    return report, capability


def verify_continuous_operation_sources(*, ppo_stage_sources: dict,
        promotion_report_path: Path, specialist_root: Path, specialist_registry_path: Path,
        soak_root: Path, soak_manifest_path: Path, output_dir: Path,
        human_enable_ppo: bool, human_reviewed_soak: bool,
        human_enable_continuous_operation: bool):
    """Reverify every source in one process and return the only usable start capability.

    This command path does not start the supervisor or perform the soak. Each
    attempt writes to a fresh output directory because its reports are immutable.
    """
    if (type(human_enable_ppo) is not bool or not human_enable_ppo
            or type(human_reviewed_soak) is not bool or not human_reviewed_soak
            or type(human_enable_continuous_operation) is not bool
            or not human_enable_continuous_operation):
        raise PermissionError("continuous operation requires separate explicit human approvals for PPO, soak, and this run")
    if not isinstance(ppo_stage_sources, dict) or not ppo_stage_sources:
        raise ValueError("continuous verification requires all frozen PPO-stage source paths")
    if {"output", "human_enable_ppo"} & set(ppo_stage_sources):
        raise ValueError("PPO-stage output and human authorization are controlled by this verifier")
    output_dir = Path(output_dir).resolve()
    output_paths = {"ppoStage": output_dir / "ppo-stage-evidence.json",
        "specialists": output_dir / "specialist-curriculum-evidence.json",
        "soak": output_dir / "supervised-soak-evidence.json",
        "continuous": output_dir / "continuous-operation-evidence.json"}
    if any(path.exists() or path.is_symlink() for path in output_paths.values()):
        raise ValueError("continuous verification outputs are immutable; choose a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_report, stage_record = verify_ppo_stage_evidence(**ppo_stage_sources,
        output=output_paths["ppoStage"], human_enable_ppo=True)
    if not ppo_enablement(stage_record)["enabled"]:
        raise PermissionError("verified PPO-stage evidence is not eligible for continuous operation")
    promotion_report, promotion = reissue_promotion_evidence_report(Path(promotion_report_path))
    if promotion is None:
        raise PermissionError("promotion source did not reissue human-approved passing evidence")
    specialist_report, specialists = verify_specialist_curriculum(root=Path(specialist_root),
        registry_path=Path(specialist_registry_path), output=output_paths["specialists"])
    soak_report, soak = audit_supervised_soak(root=Path(soak_root),
        manifest_path=Path(soak_manifest_path), output=output_paths["soak"],
        human_reviewed=True)
    continuous_report, capability = issue_continuous_operation_evidence(
        stage_record=stage_record, promotion=promotion, specialists=specialists,
        soak=soak, human_enable=True, output=output_paths["continuous"])
    return {"ppoStage": stage_report, "promotion": promotion_report,
        "specialists": specialist_report, "soak": soak_report,
        "continuousOperation": continuous_report}, capability
