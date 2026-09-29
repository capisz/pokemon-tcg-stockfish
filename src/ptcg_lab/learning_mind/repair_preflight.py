"""No-write preflight for the frozen v19 training-root repair labels."""

from __future__ import annotations

import json
from pathlib import Path

from .aggregation import load_frozen_selection
from .dataset_v1 import file_sha256, load_dataset
from .experiment import CANDIDATE_GENERATOR_VERSION, transition_generator_identity
from .schema import identity_hash


FAMILIES = ("python-heuristic", "typescript-heuristic")


def preflight_training_repair_collection(*, root: Path, family: str,
        dataset_dir: Path, support_path: Path, selection_path: Path,
        parent_selection_path: Path,
        output: Path, initial: int = 16, maximum: int = 64,
        extension_batch_size: int = 8, horizon: int = 500,
        rollout_budget_ms: int = 60_000, rollout_workers: int = 8) -> dict:
    """Validate exactly one family's v19 replacement shard without writes or engine use."""
    if family not in FAMILIES:
        raise ValueError("training-repair preflight requires an approved policy family")
    if (type(initial) is not int or type(maximum) is not int or not 1 <= initial <= maximum <= 64
            or type(extension_batch_size) is not int or not 1 <= extension_batch_size <= 64
            or type(horizon) is not int or not 1 <= horizon <= 500
            or type(rollout_budget_ms) is not int or not 1 <= rollout_budget_ms <= 120_000
            or type(rollout_workers) is not int or not 1 <= rollout_workers <= 8):
        raise ValueError("invalid training-repair rollout allocation or runtime limits")

    raw_root, raw_dataset, raw_support, raw_selection, raw_parent = map(Path,
        (root, dataset_dir, support_path, selection_path, parent_selection_path))
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("training-repair project root must be a real directory")
    if raw_dataset.is_symlink() or not raw_dataset.is_dir():
        raise ValueError("training-repair dataset must be a real directory")
    for source_path in (raw_support, raw_selection, raw_parent):
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("training-repair evidence inputs must be regular non-symlink files")
    root, dataset_dir, support_path, selection_path, parent_selection_path = (
        path.resolve() for path in
        (raw_root, raw_dataset, raw_support, raw_selection, raw_parent))
    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("training-repair output must be a fresh directory, not a symlink or file")
    if output.exists() and any(output.iterdir()):
        raise ValueError("training-repair preflight requires an empty output directory")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (not isinstance(selection, dict) or selection.get("schemaVersion") != 1
            or selection.get("selection") != "candidate-supported-source-game-preserving-train-repair-v1"
            or selection.get("status") != "draft-unlabeled-lineage-merge-required"
            or selection.get("selectionHash") != identity_hash(
                {key: value for key, value in selection.items() if key != "selectionHash"})):
        raise ValueError("training-repair selection is not a valid frozen draft")
    identity = selection.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("training-repair selection has no frozen identity")
    from .experiment import runtime_identity
    if runtime_identity(root).record() != identity:
        raise ValueError("training-repair runtime identity differs from selection")
    parent, parent_expected = load_frozen_selection(parent_selection_path, identity=identity)
    lineage = selection.get("lineage")
    if (not isinstance(lineage, dict)
            or lineage.get("kind") != "same-source-game-train-root-repair-v1"
            or lineage.get("parentSelectionHash") != parent.get("selectionHash")
            or lineage.get("parentSelectionManifestSha256") != file_sha256(parent_selection_path)
            or lineage.get("developmentPositionHashesUnchanged") is not True
            or lineage.get("heldoutPositionsSelected") is not False
            or lineage.get("rawParentLabelsMustRemainImmutable") is not True):
        raise ValueError("training-repair lineage differs from the immutable parent selection")
    parent_positions = {(split["policyFamily"], split["split"]): split["positions"]
                        for split in parent.get("splits", [])}
    proposed_splits = {(split.get("policyFamily"), split.get("split")): split
                       for split in selection.get("splits", []) if isinstance(split, dict)}
    expected_split_keys = {(policy_family, split_name) for policy_family in FAMILIES
                           for split_name in ("train", "development")}
    if set(proposed_splits) != expected_split_keys or len(proposed_splits) != len(selection.get("splits", [])):
        raise ValueError("training-repair draft must contain exactly the four parent train/development splits")
    for (policy_family, split_name), split_record in proposed_splits.items():
        positions = split_record.get("positions")
        hashes = split_record.get("positionHashes")
        if (not isinstance(positions, list) or not positions or not isinstance(hashes, list)
                or [row.get("positionHash") for row in positions if isinstance(row, dict)] != hashes
                or len(positions) != len(hashes) or len(hashes) != len(set(hashes))
                or split_record.get("sourceGames") != len(positions)
                or len({row.get("sourceGameId") for row in positions if isinstance(row, dict)}) != len(positions)
                or any(not isinstance(row, dict) or not isinstance(row.get("sourceGameId"), str)
                       or not row.get("sourceGameId") for row in positions)):
            raise ValueError(f"training-repair {policy_family}/{split_name} split metadata is malformed")

    replacements_all = lineage.get("replacements")
    if not isinstance(replacements_all, list) or not replacements_all:
        raise ValueError("training-repair lineage has no exact replacement list")
    if any(not isinstance(item, dict) or item.get("policyFamily") not in FAMILIES
           for item in replacements_all):
        raise ValueError("training-repair lineage contains an unknown policy family")
    for policy_family in FAMILIES:
        parent_train = parent_positions[(policy_family, "train")]
        parent_dev_hashes = parent_expected[policy_family]["development"]
        parent_train_hashes = parent_expected[policy_family]["train"]
        proposal_train = proposed_splits[(policy_family, "train")]
        proposal_dev = proposed_splits[(policy_family, "development")]
        if ({row.get("positionHash") for row in proposal_dev.get("positions", [])}
                != parent_dev_hashes
                or proposal_dev.get("positionHashes") != next(
                    split["positionHashes"] for split in parent["splits"]
                    if split["policyFamily"] == policy_family and split["split"] == "development")):
            raise ValueError("training-repair draft changed immutable development positions")
        declared = [item for item in replacements_all
                    if isinstance(item, dict) and item.get("policyFamily") == policy_family]
        if any(item.get("split") != "train" for item in declared):
            raise ValueError("training-repair lineage contains a non-training replacement")
        superseded = [item.get("supersededPositionHash") for item in declared]
        replacement_hashes = [item.get("replacementPositionHash") for item in declared]
        if (len(superseded) != len(set(superseded))
                or len(replacement_hashes) != len(set(replacement_hashes))
                or not set(superseded).issubset(parent_train_hashes)
                or set(replacement_hashes).intersection(parent_train_hashes)
                or set(proposal_train.get("positionHashes", []))
                   != (parent_train_hashes - set(superseded)) | set(replacement_hashes)):
            raise ValueError("training-repair draft changes train roots beyond declared replacements")
        parent_by_hash = {row["positionHash"]: row for row in parent_train}
        proposed_by_hash = {row.get("positionHash"): row for row in proposal_train.get("positions", [])}
        for replacement in declared:
            old_hash, new_hash = replacement["supersededPositionHash"], replacement["replacementPositionHash"]
            if (old_hash not in parent_by_hash or new_hash not in proposed_by_hash
                    or replacement.get("sourceGameId") != parent_by_hash[old_hash].get("sourceGameId")
                    or replacement.get("sourceGameId") != proposed_by_hash[new_hash].get("sourceGameId")):
                raise ValueError("training-repair replacement does not preserve its parent source game")

    pool = next((item for item in selection.get("sourcePools", [])
                 if isinstance(item, dict) and item.get("policyFamily") == family), None)
    if pool is None:
        raise ValueError("training-repair selection has no source pool for family")
    manifest, rows = load_dataset(dataset_dir, identity=identity)
    if (manifest.get("manifestHash") != pool.get("datasetManifestHash")
            or file_sha256(dataset_dir / "manifest.json") != pool.get("datasetManifestSha256")
            or file_sha256(dataset_dir / "rows.jsonl") != pool.get("datasetRowsSha256")):
        raise ValueError("training-repair dataset differs from frozen source-pool receipt")

    support = json.loads(support_path.read_text(encoding="utf-8"))
    if (support.get("identity") != identity
            or support.get("datasetManifestHash") != manifest.get("manifestHash")
            or support.get("datasetRowsSha256") != pool.get("datasetRowsSha256")
            or support.get("reportHash") != pool.get("supportReportHash")
            or file_sha256(support_path) != pool.get("supportReportSha256")
            or support.get("reportHash") != identity_hash(
                {key: value for key, value in support.items() if key != "reportHash"})):
        raise ValueError("training-repair support report differs from frozen receipt")
    generator_identity = support.get("candidateGeneratorIdentity")
    if (not isinstance(generator_identity, dict)
            or transition_generator_identity(root) != generator_identity):
        raise ValueError("training-repair candidate generator differs from support audit")

    replacements = [item for item in replacements_all
                    if isinstance(item, dict) and item.get("policyFamily") == family
                    and item.get("split") == "train"]
    if not replacements:
        raise ValueError("training-repair selection has no replacement positions for family")
    hashes = [item.get("replacementPositionHash") for item in replacements]
    if any(not isinstance(value, str) or not value for value in hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("training-repair selection contains malformed or duplicate replacements")
    rows_by_hash = {row.get("positionHash"): row for row in rows}
    support_by_hash = {row.get("positionHash"): row for row in support.get("positions", [])
                       if isinstance(row, dict)}
    if len(support_by_hash) != len(support.get("positions", [])):
        raise ValueError("training-repair support report repeats or malforms position hashes")
    candidate_plans = 0
    for item in replacements:
        position_hash = item["replacementPositionHash"]
        row, support_row = rows_by_hash.get(position_hash), support_by_hash.get(position_hash)
        count = support_row.get("completeCandidateCount") if support_row else None
        if (row is None or row.get("split") != "train"
                or row.get("sourceGameId") != item.get("sourceGameId")
                or support_row is None or support_row.get("status") != "supported"
                or type(count) is not int or count < 2
                or count != item.get("replacementCompleteCandidateCount")):
            raise ValueError("training-repair replacement differs from supported frozen training row")
        candidate_plans += count

    settings = {"identity": identity, "datasetManifestHash": manifest["manifestHash"],
        "requestedPositions": len(hashes), "initialRollouts": initial,
        "maximumRollouts": maximum, "extensionBatchSize": extension_batch_size,
        "horizon": horizon, "rolloutBudgetMs": rollout_budget_ms,
        "rolloutWorkers": rollout_workers, "splitFilter": None,
        "selectionMethod": "position-hash-list", "selectedPositionHashes": hashes,
        "candidateGeneratorVersion": CANDIDATE_GENERATOR_VERSION,
        "candidateGeneratorIdentity": generator_identity,
        "rolloutSeedVersion": "configuration-bound-v1",
        "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
        "labelCollectorVersion": "macro-rollout-labeler-v6",
        "labelCollectorSha256": file_sha256(Path(__file__).with_name("experiment.py"))}
    rollout_identity = identity_hash(settings)
    settings["rolloutIdentity"] = rollout_identity
    return {"schemaVersion": 1, "kind": "training-repair-macro-label-preflight-v1",
        "family": family, "selectionHash": selection["selectionHash"],
        "parentSelectionHash": parent["selectionHash"],
        "parentSelectionManifestSha256": file_sha256(parent_selection_path),
        "sourcePool": {key: pool[key] for key in (
            "datasetManifestHash", "datasetManifestSha256", "datasetRowsSha256",
            "supportReportHash", "supportReportSha256")},
        "rolloutIdentity": rollout_identity, "settings": settings,
        "outputState": "fresh" if not output.exists() else "empty-directory",
        "selectedPositions": len(hashes), "verifiedSupportedPositions": len(hashes),
        "candidatePlans": candidate_plans,
        "initialCandidateSeedRollouts": candidate_plans * initial,
        "maximumCandidateSeedRollouts": candidate_plans * maximum,
        "adaptivePruningMayReduceAttempts": True,
        "writesArtifacts": False, "startsEngine": False,
        "elapsedTimeEstimateSeconds": None,
        "elapsedTimeEstimateStatus": "unknown; no comparable training-repair runtime benchmark"}
