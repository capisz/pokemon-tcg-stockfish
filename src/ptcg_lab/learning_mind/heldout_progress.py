"""Read-only progress inspection for heldout macro-label collection."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset_v1 import file_sha256
from .heldout_evaluation import FAMILIES, _read_frozen_heldout_selection
from .schema import identity_hash


def _count_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("heldout checkpoint candidate record is malformed")
    outcomes = {key: record.get(key) for key in ("finished", "truncated", "error")}
    if any(type(value) is not int or value < 0 for value in outcomes.values()):
        raise ValueError("heldout checkpoint outcome counts are malformed")
    by_index = record.get("outcomesByIndex")
    if not isinstance(by_index, dict):
        raise ValueError("heldout checkpoint lacks per-seed outcome receipts")
    receipt_counts = {key: 0 for key in outcomes}
    for index, receipt in by_index.items():
        if (not isinstance(index, str) or not index.isdecimal() or str(int(index)) != index
                or not isinstance(receipt, dict) or receipt.get("status") not in receipt_counts):
            raise ValueError("heldout checkpoint seed receipt is malformed")
        receipt_counts[receipt["status"]] += 1
    if receipt_counts != outcomes:
        raise ValueError("heldout checkpoint receipts do not reconcile with outcome totals")
    samples = sum(outcomes.values())
    if set(by_index) != {str(index) for index in range(samples)}:
        raise ValueError("heldout candidate seed receipts are not a complete sampled prefix")
    return {"samples": samples, "outcomes": outcomes, "sampleIndices": set(by_index)}


def _inspect_checkpoint(path: Path, *, family: str, position_hash: str,
                        selected_positions: dict[str, dict]) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"heldout progress checkpoint is not a regular file: {path}")
    checkpoint = json.loads(path.read_text())
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("schemaVersion") != 1
            or checkpoint.get("progressHash") != identity_hash(
                {key: value for key, value in checkpoint.items() if key != "progressHash"})):
        raise ValueError(f"heldout progress checkpoint hash/schema mismatch: {path}")
    progress_identity = checkpoint.get("progressIdentity")
    if (not isinstance(progress_identity, dict)
            or checkpoint.get("progressIdentityHash") != identity_hash(progress_identity)
            or progress_identity.get("positionHash") != position_hash
            or position_hash not in selected_positions):
        raise ValueError(f"heldout progress checkpoint is outside frozen {family} selection: {path}")
    candidate_hashes = progress_identity.get("candidateHashes")
    allocation = progress_identity.get("allocation")
    rollout_identity = progress_identity.get("rolloutIdentity")
    if (not isinstance(candidate_hashes, list) or not candidate_hashes
            or any(not isinstance(value, str) or not value for value in candidate_hashes)
            or len(candidate_hashes) != len(set(candidate_hashes))
            or not isinstance(rollout_identity, str) or not rollout_identity
            or not isinstance(allocation, dict)):
        raise ValueError(f"heldout progress candidate/allocation identity is malformed: {path}")
    initial, maximum = allocation.get("initial"), allocation.get("maximum")
    extension_batch_size = allocation.get("extensionBatchSize")
    if (type(initial) is not int or type(maximum) is not int or not 1 <= initial <= maximum <= 64
            or type(extension_batch_size) is not int or not 1 <= extension_batch_size <= 64
            or allocation.get("closeMargin") != .10):
        raise ValueError(f"heldout progress allocation is malformed: {path}")
    state = checkpoint.get("state")
    if (not isinstance(state, dict) or state.get("allocation") != allocation
            or state.get("candidateHashes") != candidate_hashes
            or state.get("rolloutIdentity") != rollout_identity):
        raise ValueError(f"heldout progress state differs from its frozen identity: {path}")
    completed_initial = state.get("completedInitialIndices")
    completed_extension = state.get("completedExtensionIndices")
    if (not isinstance(completed_initial, list) or not isinstance(completed_extension, list)
            or any(type(index) is not int or not 0 <= index < initial for index in completed_initial)
            or any(type(index) is not int or not initial <= index < maximum for index in completed_extension)
            or len(completed_initial) != len(set(completed_initial))
            or len(completed_extension) != len(set(completed_extension))):
        raise ValueError(f"heldout progress seed indices are malformed: {path}")
    if (completed_initial != list(range(len(completed_initial)))
            or completed_extension != list(range(initial, initial + len(completed_extension)))):
        raise ValueError(f"heldout progress completed seeds are not a monotone prefix: {path}")
    records = state.get("records")
    if not isinstance(records, dict) or set(records) != set(candidate_hashes):
        raise ValueError(f"heldout progress candidate records do not match identity: {path}")
    active = state.get("closeCandidateHashes")
    if active is None:
        active = list(candidate_hashes)
        active_status = "initial-confidence-check-pending"
    else:
        if (not isinstance(active, list) or len(active) != len(set(active))
                or any(candidate not in records for candidate in active)):
            raise ValueError(f"heldout active-candidate set is malformed: {path}")
        active_status = "adaptive-extension"
    candidate_stats = {key: _count_record(records[key]) for key in candidate_hashes}
    completed_indices = set(completed_initial) | set(completed_extension)
    confidence_watermark = state.get("confidenceEvaluatedThroughIndex", -1)
    if (type(confidence_watermark) is not int or not -1 <= confidence_watermark < maximum
            or (confidence_watermark >= 0 and confidence_watermark not in completed_indices)
            or (confidence_watermark >= 0 and not set(range(confidence_watermark + 1)).issubset(completed_indices))):
        raise ValueError(f"heldout confidence reevaluation watermark is malformed: {path}")
    if any(not stat["sampleIndices"].issubset({str(index) for index in completed_indices})
           for stat in candidate_stats.values()):
        raise ValueError(f"heldout candidate outcome exceeds completed checkpoint indices: {path}")
    active_stats = [candidate_stats[key] for key in active]
    sample_range = ([min(stat["samples"] for stat in active_stats),
                     max(stat["samples"] for stat in active_stats)] if active_stats else [0, 0])
    outcome_totals = {key: sum(candidate_stats[item]["outcomes"][key] for item in candidate_hashes)
                      for key in ("finished", "truncated", "error")}
    active_outcomes = {key: sum(candidate_stats[item]["outcomes"][key] for item in active)
                       for key in ("finished", "truncated", "error")}
    highest_completed = max(completed_indices) if completed_indices else -1
    if not active:
        next_gate = "no active candidates remain; finalize as unsupported/unscored for evaluator review"
    elif active_status == "initial-confidence-check-pending":
        next_gate = f"finish initial common-seed batch through seed index {initial - 1}, then evaluate confidence"
    elif confidence_watermark < highest_completed:
        stage_start = initial + ((highest_completed - initial) // extension_batch_size) * extension_batch_size
        stage_end = min(maximum, stage_start + extension_batch_size)
        stage_indices = set(range(stage_start, stage_end))
        if stage_indices.issubset(completed_extension):
            next_gate = f"confidence reevaluation is due through completed seed index {highest_completed}"
        else:
            next_gate = (f"complete extension batch through seed index {stage_end - 1} "
                         "before confidence reevaluation")
    elif sample_range[1] >= maximum:
        next_gate = "maximum sampling and final confidence reevaluation complete; position result is next"
    else:
        next_gate = "confidence reevaluation after the next completed extension batch"
    return {
        "positionHash": position_hash,
        "checkpointSha256": file_sha256(path),
        "checkpointHashValid": True,
        "initialSeedsCompleted": len(completed_initial),
        "initialSeedsPlanned": initial,
        "completedExtensionSeeds": len(completed_extension),
        "highestExtensionIndex": max(completed_extension) if completed_extension else None,
        "completedSeedIndices": sorted(completed_indices),
        "confidenceEvaluatedThroughIndex": confidence_watermark,
        "activeCandidates": len(active),
        "activeCandidateHashes": list(active),
        "samplesPerActiveCandidateMinMax": sample_range,
        "outcomesAcrossSampledCandidates": outcome_totals,
        "outcomesAcrossActiveCandidates": active_outcomes,
        "nextStoppingGate": next_gate,
    }


def heldout_macro_label_progress(*, family: str, selection_path: Path, run_dir: Path) -> dict:
    """Inspect one heldout family run without modifying or polling a process."""
    if family not in FAMILIES:
        raise ValueError("heldout progress requires one approved policy family")
    selection, selected_by_family = _read_frozen_heldout_selection(Path(selection_path))
    selected = selected_by_family[family]
    run_dir = Path(run_dir)
    if run_dir.is_symlink() or (run_dir.exists() and not run_dir.is_dir()):
        raise ValueError("heldout progress run path must be a directory and cannot be a symlink")
    directory_exists = run_dir.exists()

    manifest_path = run_dir / "manifest.json"
    manifest = None
    if directory_exists and (manifest_path.exists() or manifest_path.is_symlink()):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("heldout progress manifest must be a regular file")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("manifestHash") != identity_hash(
                {key: value for key, value in manifest.items() if key != "manifestHash"})
                or manifest.get("identity") != selection["identity"]
                or manifest.get("selectionHash") != selection["selectionHash"]
                or manifest.get("selectedPositionHashes") != sorted(selected)
                or manifest.get("positions") != len(selected)):
            raise ValueError("heldout progress manifest differs from the frozen selection")

    result_files = {}
    for path in (run_dir.glob("*.json") if directory_exists else ()):
        if path.name == "manifest.json":
            continue
        if path.is_symlink() or not path.is_file() or path.stem not in selected or path.name != f"{path.stem}.json":
            raise ValueError(f"unexpected heldout progress result file: {path.name}")
        record = json.loads(path.read_text())
        if (record.get("positionHash") != path.stem
                or record.get("identity") != selection["identity"]
                or record.get("selectionHash") != selection["selectionHash"]
                or record.get("split") != "heldout"
                or record.get("opponentPolicyFamily") != family
                or record.get("status") not in {"collected", "unsupported"}):
            raise ValueError(f"heldout progress result differs from frozen selection: {path.name}")
        result_files[path.stem] = {"status": record.get("status"), "sha256": file_sha256(path)}

    if manifest is not None:
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != len(selected):
            raise ValueError("published heldout progress manifest file list is malformed")
        listed = {item.get("path"): item.get("sha256") for item in files if isinstance(item, dict)}
        if (len(listed) != len(files) or set(listed) != {f"{key}.json" for key in selected}
                or set(result_files) != set(selected)
                or any(listed[f"{key}.json"] != result_files[key]["sha256"] for key in selected)):
            raise ValueError("published heldout result files do not reconcile with the manifest")
        if (manifest.get("supportedPositions") != sum(
                result["status"] == "collected" for result in result_files.values())
                or manifest.get("unsupportedPositions") != sum(
                    result["status"] == "unsupported" for result in result_files.values())
                or manifest.get("highConfidencePolicyLabels") != 0):
            raise ValueError("published heldout position counts differ from verified result records")

    checkpoints = []
    for path in (run_dir.glob(".*.progress") if directory_exists else ()):
        if path.name == ".manifest.progress":
            raise ValueError("unexpected heldout progress checkpoint name")
        position_hash = path.name[1:-len(".progress")]
        if position_hash not in selected or position_hash in result_files:
            raise ValueError(f"heldout progress checkpoint is outside selection or already finalized: {path.name}")
        checkpoints.append(_inspect_checkpoint(path, family=family, position_hash=position_hash,
                                               selected_positions=selected))

    allowed_names = {"manifest.json", *(f"{key}.json" for key in result_files),
                     *(f".{item['positionHash']}.progress" for item in checkpoints)}
    if directory_exists and {path.name for path in run_dir.iterdir()} - allowed_names:
        raise ValueError("heldout progress run directory contains unrecognized files")
    unsupported = sum(result_files[key]["status"] == "unsupported" for key in result_files)
    return {
        "schemaVersion": 1,
        "kind": "heldout-macro-label-progress-v1",
        "family": family,
        "selectionHash": selection["selectionHash"],
        "selectedPositions": len(selected),
        "finalizedResultFiles": len(result_files),
        "finalizedPositionPercent": round(100 * len(result_files) / len(selected), 1),
        "unsupportedPositions": unsupported,
        "positionsInProgress": len(checkpoints),
        "positionsNotStarted": len(selected) - len(result_files) - len(checkpoints),
        "checkpoints": sorted(checkpoints, key=lambda item: item["positionHash"]),
        "manifestStatus": "verified-complete" if manifest is not None else "not-yet-published",
        "processStatus": "unknown; filesystem checkpoints do not prove a collector is running",
        "writesArtifacts": False,
        "trainingEligible": False,
        "ppoEnabled": False,
        "continuousOperationEnabled": False,
        "trustedPromotion": False,
    }
