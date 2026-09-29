from __future__ import annotations

import json
from pathlib import Path

from .aggregation import POLICY_FAMILIES, load_frozen_selection
from .dataset_v1 import file_sha256
from .schema import identity_hash


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def _verify_progress_checkpoint(path: Path, *, family: str, expected: set[str],
                                identity: dict, dataset_hash: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"macro-label checkpoint is not a regular file: {path}")
    checkpoint = json.loads(path.read_text())
    progress_hash = checkpoint.get("progressHash")
    if progress_hash != identity_hash({key: value for key, value in checkpoint.items()
                                      if key != "progressHash"}):
        raise ValueError(f"macro-label checkpoint hash mismatch: {path}")
    frozen = checkpoint.get("identity")
    if not isinstance(frozen, dict):
        raise ValueError(f"macro-label checkpoint identity is missing: {path}")
    if checkpoint.get("identityHash") != identity_hash(frozen):
        raise ValueError(f"macro-label checkpoint identity hash mismatch: {path}")
    position_hash = frozen.get("positionHash")
    if position_hash not in expected or frozen.get("identity") != identity:
        raise ValueError(f"macro-label checkpoint is outside the frozen {family} selection: {path}")
    if frozen.get("datasetManifestHash") != dataset_hash:
        raise ValueError(f"macro-label checkpoint dataset identity mismatch: {path}")
    state = checkpoint.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"macro-label checkpoint state is missing: {path}")
    candidates = frozen.get("candidateHashes")
    if (not isinstance(candidates, list) or not candidates
            or any(not isinstance(value, str) or not value for value in candidates)
            or len(candidates) != len(set(candidates))
            or state.get("candidateHashes") != candidates):
        raise ValueError(f"macro-label checkpoint candidate identity mismatch: {path}")
    records = state.get("records")
    if not isinstance(records, dict) or set(records) != set(candidates):
        raise ValueError(f"macro-label checkpoint candidate records mismatch: {path}")
    sample_counts = []
    outcome_totals = {key: 0 for key in ("finished", "truncated", "error")}
    for candidate_hash in candidates:
        record = records[candidate_hash]
        if not isinstance(record, dict):
            raise ValueError(f"macro-label checkpoint candidate record is malformed: {path}")
        counts = {key: record.get(key) for key in outcome_totals}
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError(f"macro-label checkpoint has invalid outcome counts: {path}")
        sample_counts.append(sum(counts.values()))
        for key, value in counts.items():
            outcome_totals[key] += value
    initial = frozen.get("initialRollouts")
    maximum = frozen.get("maximumRollouts")
    completed_initial = state.get("completedInitialIndices")
    extensions = state.get("completedExtensionIndices")
    allocation = state.get("allocation")
    if (type(initial) is not int or initial < 1 or type(maximum) is not int or maximum < initial
            or not isinstance(allocation, dict) or allocation.get("initial") != initial
            or allocation.get("maximum") != maximum or not isinstance(completed_initial, list)
            or any(type(index) is not int or not 0 <= index < initial for index in completed_initial)
            or len(completed_initial) != len(set(completed_initial))
            or not isinstance(extensions, list)
            or any(type(index) is not int or not initial <= index < maximum for index in extensions)):
        raise ValueError(f"macro-label checkpoint seed progress is malformed: {path}")
    if len(extensions) != len(set(extensions)):
        raise ValueError(f"macro-label checkpoint seed progress is malformed: {path}")
    close_candidates = state.get("closeCandidateHashes")
    if close_candidates is not None and (not isinstance(close_candidates, list)
            or any(not isinstance(value, str) for value in close_candidates)
            or len(close_candidates) != len(set(close_candidates))
            or any(value not in records for value in close_candidates)):
        raise ValueError(f"macro-label checkpoint active candidate set is malformed: {path}")
    return {
        "positionHash": position_hash,
        "candidateCount": len(candidates),
        "completedInitialSeeds": len(completed_initial),
        "initialSeeds": initial,
        "initialSeedProgressPercent": _percent(len(completed_initial), initial),
        "completedExtensionSeeds": len(extensions),
        "highestExtensionIndex": max(extensions) if extensions else None,
        "activeCandidateCount": len(close_candidates) if close_candidates is not None else None,
        "sampleCountRangePerCandidate": [min(sample_counts), max(sample_counts)],
        "outcomes": outcome_totals,
        "checkpointHashValid": True,
    }


def macro_label_progress(*, selection_path: Path, runs: dict[str, Path]) -> dict:
    """Read-only, selection-bound progress report for frozen macro-label runs."""
    if set(runs) != POLICY_FAMILIES:
        raise ValueError("progress requires one run directory per approved policy family")
    selection_value = json.loads(selection_path.read_text())
    identity = selection_value.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("frozen selection has no experiment identity")
    selection, expected = load_frozen_selection(selection_path, identity=identity)
    source_pools = selection.get("sourcePools")
    if not isinstance(source_pools, list) or any(not isinstance(item, dict) for item in source_pools):
        raise ValueError("frozen selection source pools are malformed")
    source_hashes = {}
    for item in source_pools:
        family = item.get("policyFamily")
        dataset_hash = item.get("datasetManifestHash")
        if (not isinstance(family, str) or family not in POLICY_FAMILIES or family in source_hashes
                or not isinstance(dataset_hash, str) or not dataset_hash):
            raise ValueError("frozen selection lacks both policy-family dataset identities")
        source_hashes[family] = dataset_hash
    if set(source_hashes) != POLICY_FAMILIES:
        raise ValueError("frozen selection lacks both policy-family dataset identities")

    family_reports = {}
    total_selected = total_completed = total_policy_eligible = 0
    for family in sorted(POLICY_FAMILIES):
        run_dir = runs[family]
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError(f"macro-label run directory is missing or symlinked: {run_dir}")
        selected = set().union(*expected[family].values())
        total_selected += len(selected)
        manifest_path = run_dir / "manifest.json"
        manifest = None
        if manifest_path.exists() or manifest_path.is_symlink():
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"macro-label manifest is not a regular file: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("manifestHash") != identity_hash(
                    {key: value for key, value in manifest.items() if key != "manifestHash"}):
                raise ValueError(f"macro-label manifest hash mismatch: {manifest_path}")
            if (manifest.get("identity") != identity
                    or manifest.get("datasetManifestHash") != source_hashes[family]):
                raise ValueError(f"macro-label manifest identity mismatch: {manifest_path}")
            listed = manifest.get("files")
            if not isinstance(listed, list) or len(listed) != manifest.get("positions"):
                raise ValueError(f"macro-label manifest file list is malformed: {manifest_path}")
            listed_names = [item.get("path") if isinstance(item, dict) else None for item in listed]
            if (any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".json")
                    or name == "manifest.json" for name in listed_names)
                    or len(listed_names) != len(set(listed_names))):
                raise ValueError(f"macro-label manifest contains an unsafe or duplicate path: {manifest_path}")
            actual_names = {path.name for path in run_dir.glob("*.json") if path.name != "manifest.json"}
            if actual_names != set(listed_names):
                raise ValueError(f"macro-label files differ from manifest: {manifest_path}")
            for item in listed:
                source = run_dir / item["path"]
                if source.is_symlink() or not source.is_file() or file_sha256(source) != item.get("sha256"):
                    raise ValueError(f"macro-label record hash mismatch: {source}")

        records_by_position = {}
        for record_path in run_dir.glob("*.json"):
            if record_path.name == "manifest.json":
                continue
            if record_path.is_symlink() or not record_path.is_file():
                raise ValueError(f"macro-label record is not a regular file: {record_path}")
            record = json.loads(record_path.read_text())
            position_hash = record.get("positionHash")
            if (not isinstance(position_hash, str) or record_path.name != f"{position_hash}.json"
                    or position_hash not in selected or position_hash in records_by_position):
                raise ValueError(f"macro-label record is unexpected or duplicated: {record_path}")
            if (record.get("identity") != identity
                    or record.get("datasetManifestHash") != source_hashes[family]
                    or record.get("opponentPolicyFamily") != family):
                raise ValueError(f"macro-label record identity mismatch: {record_path}")
            split = record.get("split")
            if split not in expected[family] or position_hash not in expected[family][split]:
                raise ValueError(f"macro-label record split differs from frozen selection: {record_path}")
            records_by_position[position_hash] = record
        if manifest is not None:
            manifest_positions = manifest.get("selectedPositionHashes")
            if (not isinstance(manifest_positions, list) or len(manifest_positions) != len(selected)
                    or any(not isinstance(value, str) or not value for value in manifest_positions)
                    or set(manifest_positions) != selected
                    or set(records_by_position) != selected):
                raise ValueError(f"completed macro-label manifest does not exactly cover frozen {family} selection")

        checkpoints = []
        checkpoint_positions = set()
        for checkpoint_path in run_dir.glob(".*.progress"):
            checkpoint = _verify_progress_checkpoint(checkpoint_path, family=family,
                expected=selected, identity=identity, dataset_hash=source_hashes[family])
            if (checkpoint["positionHash"] in records_by_position
                    or checkpoint["positionHash"] in checkpoint_positions):
                raise ValueError(f"duplicate final result or active checkpoint for one position: {checkpoint_path}")
            if checkpoint_path.name != f".{checkpoint['positionHash']}.progress":
                raise ValueError(f"macro-label checkpoint filename differs from its position: {checkpoint_path}")
            checkpoint_positions.add(checkpoint["positionHash"])
            checkpoints.append(checkpoint)
        if len(checkpoints) > len(selected) - len(records_by_position):
            raise ValueError(f"more active macro-label checkpoints than uncollected positions: {run_dir}")

        completed = len(records_by_position)
        total_completed += completed
        policy_eligible = sum(record.get("highConfidencePolicyEligible") is True
                              for record in records_by_position.values())
        total_policy_eligible += policy_eligible
        split_counts = {
            split: {"selected": len(expected[family][split]),
                    "completed": sum(position_hash in records_by_position
                                     for position_hash in expected[family][split]),
                    "percent": _percent(sum(position_hash in records_by_position
                                             for position_hash in expected[family][split]),
                                        len(expected[family][split]))}
            for split in ("train", "development")
        }
        family_reports[family] = {
            "selectedPositions": len(selected),
            "completedResultFiles": completed,
            "completedPositionPercent": _percent(completed, len(selected)),
            "positionsInProgress": len(checkpoints),
            "positionsNotStarted": len(selected) - completed - len(checkpoints),
            "splits": split_counts,
            "policyEligiblePositions": policy_eligible,
            "manifestStatus": "verified-complete" if manifest is not None else "not-yet-published",
            "checkpoints": sorted(checkpoints, key=lambda item: item["positionHash"]),
            "processStatus": "unknown",
        }
    return {
        "schemaVersion": 1,
        "kind": "macro-label-progress-v1",
        "selectionHash": selection["selectionHash"],
        "positionCoverage": {
            "selected": total_selected,
            "completedResultFiles": total_completed,
            "percent": _percent(total_completed, total_selected),
            "positionsInProgress": sum(item["positionsInProgress"] for item in family_reports.values()),
            "policyEligiblePositions": total_policy_eligible,
        },
        "families": family_reports,
        "interpretation": "position-file coverage only; not total project readiness or policy strength",
        "processStatus": "unknown; file checkpoints do not prove a collector is running",
        "writesArtifacts": False,
        "ppoEnabled": False,
        "continuousOperationEnabled": False,
        "trustedPromotion": False,
    }
