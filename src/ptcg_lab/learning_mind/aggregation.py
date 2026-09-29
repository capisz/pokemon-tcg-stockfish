from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from .dataset_v1 import file_sha256, load_dataset
from .collector_compatibility import collector_compatibility
from .schema import identity_hash

POLICY_FAMILIES = {"python-heuristic", "typescript-heuristic"}
RANKER_SPLITS = {"train", "development"}
SHARED_ROLLOUT_SETTINGS = (
    "initialRollouts", "maximumRollouts", "extensionBatchSize", "horizon",
    "rolloutBudgetMs", "rolloutWorkers", "rolloutSeedVersion",
    "adaptiveAllocationVersion", "selectionMethod", "splitFilter",
)


def load_frozen_selection(path: Path, *, identity: dict) -> tuple[dict, dict[str, dict[str, set[str]]]]:
    """Load and validate the exact train/development position universe for a ranker run."""
    selection = json.loads(path.read_text())
    if selection.get("schemaVersion") != 1:
        raise ValueError("unsupported frozen macro-label selection schema")
    if selection.get("identity") != identity:
        raise ValueError("frozen macro-label selection identity mismatch")
    recorded_hash = selection.get("selectionHash")
    if recorded_hash != identity_hash({key: value for key, value in selection.items()
                                       if key != "selectionHash"}):
        raise ValueError("frozen macro-label selection hash mismatch")
    splits = selection.get("splits")
    if not isinstance(splits, list):
        raise ValueError("frozen macro-label selection has no split records")
    expected: dict[str, dict[str, set[str]]] = {
        family: {split: set() for split in RANKER_SPLITS} for family in POLICY_FAMILIES
    }
    seen: set[tuple[str, str]] = set()
    for item in splits:
        if not isinstance(item, dict):
            raise ValueError("invalid frozen macro-label selection split")
        family, split = item.get("policyFamily"), item.get("split")
        hashes = item.get("positionHashes")
        if family not in POLICY_FAMILIES or split not in RANKER_SPLITS:
            raise ValueError("frozen macro-label selection may contain only approved families and train/development")
        if not isinstance(hashes, list) or not hashes or any(not isinstance(value, str) or not value for value in hashes):
            raise ValueError("frozen macro-label selection has an empty or invalid position list")
        if len(hashes) != len(set(hashes)):
            raise ValueError("frozen macro-label selection contains duplicate positions")
        key = (family, split)
        if key in seen:
            raise ValueError("frozen macro-label selection repeats a family/split")
        seen.add(key)
        if item.get("sourceGames") != len(hashes):
            raise ValueError("frozen selection source-game count differs from its position list")
        expected[family][split] = set(hashes)
    required = {(family, split) for family in POLICY_FAMILIES for split in RANKER_SPLITS}
    if seen != required:
        raise ValueError("frozen selection must cover both families in train and development")
    all_positions = [position for family in POLICY_FAMILIES for split in RANKER_SPLITS
                     for position in expected[family][split]]
    if len(all_positions) != len(set(all_positions)):
        raise ValueError("frozen selection reuses a position across families or splits")
    return selection, expected


def combine_macro_label_runs(*, inputs: list[Path], output: Path, identity: dict,
                             selection_path: Path) -> dict:
    """Verify separate policy-family label runs and combine them for ranker evaluation."""
    if not inputs:
        raise ValueError("at least one macro-label run is required")
    output = output.resolve()
    if output.exists():
        raise ValueError("combined macro-label outputs are immutable; choose a new directory")
    selection, expected_positions = load_frozen_selection(selection_path, identity=identity)
    loaded = []
    position_hashes = set()
    families = set()
    generator_identity = None
    collector_versions: list[str] = []
    collector_hashes: list[str] = []
    shared_rollout_settings = None
    for input_dir in inputs:
        input_dir = input_dir.resolve()
        manifest_path = input_dir / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"macro-label run manifest is missing or is a symlink: {input_dir}")
        manifest = json.loads(manifest_path.read_text())
        recorded_hash = manifest.get("manifestHash")
        if recorded_hash != identity_hash({key: value for key, value in manifest.items()
                                             if key != "manifestHash"}):
            raise ValueError(f"input manifest hash mismatch: {input_dir}")
        if manifest.get("identity") != identity:
            raise ValueError(f"input experiment identity mismatch: {input_dir}")
        current_rollout_settings = {key: manifest.get(key) for key in SHARED_ROLLOUT_SETTINGS}
        if shared_rollout_settings is None:
            shared_rollout_settings = current_rollout_settings
        elif current_rollout_settings != shared_rollout_settings:
            raise ValueError("macro-label runs use different frozen rollout settings")
        if not isinstance(manifest.get("files"), list) or len(manifest["files"]) != manifest.get("positions"):
            raise ValueError(f"input manifest file list mismatch: {input_dir}")
        listed_names = [item.get("path") if isinstance(item, dict) else None
                        for item in manifest["files"]]
        if (any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".json")
                or name == "manifest.json" for name in listed_names)
                or len(listed_names) != len(set(listed_names))):
            raise ValueError(f"input manifest contains missing, unsafe, or duplicate paths: {input_dir}")
        actual_names = {path.name for path in input_dir.glob("*.json")
                        if path.name != "manifest.json"}
        if actual_names != set(listed_names):
            raise ValueError(f"input run has missing or unlisted JSON records: {input_dir}")
        if generator_identity is None:
            generator_identity = manifest.get("candidateGeneratorIdentity")
        elif manifest.get("candidateGeneratorIdentity") != generator_identity:
            raise ValueError("macro-label runs use different candidate generators")
        collector_versions.append(manifest.get("labelCollectorVersion"))
        collector_hashes.append(manifest.get("labelCollectorSha256"))
        records = []
        run_positions: set[str] = set()
        run_family = None
        for item in manifest["files"]:
            name = item.get("path")
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") or name == "manifest.json":
                raise ValueError(f"unsafe macro-label record path in {input_dir}")
            source = input_dir / name
            if source.is_symlink():
                raise ValueError(f"macro-label record must not be a symlink: {source}")
            if file_sha256(source) != item.get("sha256"):
                raise ValueError(f"macro-label record hash mismatch: {source}")
            record = json.loads(source.read_text())
            key = record.get("positionHash")
            if not isinstance(key, str) or name != f"{key}.json":
                raise ValueError(f"macro-label record filename/position mismatch: {source}")
            if key in position_hashes:
                raise ValueError(f"duplicate macro-label position across runs: {key}")
            if record.get("identity") != identity or record.get("datasetManifestHash") != manifest.get("datasetManifestHash"):
                raise ValueError(f"macro-label record identity mismatch: {source}")
            if record.get("rolloutIdentity") != manifest.get("rolloutIdentity"):
                raise ValueError(f"macro-label rollout identity mismatch: {source}")
            family = record.get("opponentPolicyFamily")
            if family not in POLICY_FAMILIES:
                raise ValueError(f"macro-label record lacks policy-family provenance: {source}")
            if record.get("status") != "collected":
                raise ValueError(f"macro-label record is not collected: {source}")
            if record.get("split") not in {"train", "development"}:
                raise ValueError(
                    f"ranker input may contain only train/development records; "
                    f"held-out or unknown split is not publishable: {source}"
                )
            if run_family is None:
                run_family = family
            elif family != run_family:
                raise ValueError(f"each input run must contain exactly one policy family: {input_dir}")
            position_hashes.add(key)
            run_positions.add(key)
            families.add(family)
            records.append((source, item, record, family))
        if len({record[3] for record in records}) != 1:
            raise ValueError(f"each input run must contain exactly one policy family: {input_dir}")
        if set(manifest.get("selectedPositionHashes", [])) != {record[2]["positionHash"] for record in records}:
            raise ValueError(f"selected positions differ from input records: {input_dir}")
        expected_for_family = expected_positions[run_family]
        observed_by_split = {split: {record[2]["positionHash"] for record in records
                                     if record[2]["split"] == split} for split in RANKER_SPLITS}
        if observed_by_split != expected_for_family or run_positions != set().union(*expected_for_family.values()):
            missing = sorted(set().union(*expected_for_family.values()) - run_positions)
            unexpected = sorted(run_positions - set().union(*expected_for_family.values()))
            raise ValueError(
                f"macro-label run does not exactly cover frozen {run_family} selection "
                f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
            )
        loaded.append({"directory": input_dir, "manifestPath": manifest_path,
                       "manifest": manifest, "records": records})

    if families != POLICY_FAMILIES:
        raise ValueError("combined ranker input must contain both Python and TypeScript policy families")

    collector_equivalence = collector_compatibility(
        versions=collector_versions, source_hashes=collector_hashes)
    collector_version = collector_equivalence["labelCollectorVersion"]
    unique_collector_hashes = collector_equivalence["sourceModuleSha256s"]
    collector_sha = unique_collector_hashes[0] if len(unique_collector_hashes) == 1 else None

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        files = []
        splits = Counter()
        policy_labels = 0
        for run in loaded:
            for source, item, record, family in run["records"]:
                relative = Path(family) / source.name
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                files.append({"path": relative.as_posix(), "sha256": file_sha256(target)})
                splits[record["split"]] += 1
                policy_labels += int(record.get("highConfidencePolicyEligible") is True)
        files.sort(key=lambda item: item["path"])
        combined = {
            "schemaVersion": 1,
            "kind": "combined-macro-label-runs-v1",
            "identity": identity,
            "selectionHash": selection["selectionHash"],
            "selectionManifestSha256": file_sha256(selection_path),
            "selectedPositionHashes": sorted(position_hashes),
            "positionsByFamilySplit": {
                family: {split: len(expected_positions[family][split])
                         for split in sorted(RANKER_SPLITS)}
                for family in sorted(POLICY_FAMILIES)
            },
            "candidateGeneratorIdentity": generator_identity,
            "labelCollectorVersion": collector_version,
            "labelCollectorSha256": collector_sha,
            "labelCollectorSourceSha256s": unique_collector_hashes,
            "labelCollectorCompatibility": collector_equivalence,
            "sharedRolloutSettings": shared_rollout_settings,
            "sourceRuns": [{
                "policyFamilies": sorted({record[3] for record in run["records"]}),
                "datasetManifestHash": run["manifest"]["datasetManifestHash"],
                "rolloutIdentity": run["manifest"]["rolloutIdentity"],
                "labelCollectorVersion": run["manifest"]["labelCollectorVersion"],
                "labelCollectorSha256": run["manifest"]["labelCollectorSha256"],
                "candidateGeneratorVersion": run["manifest"].get("candidateGeneratorVersion"),
                "candidateGeneratorIdentity": run["manifest"]["candidateGeneratorIdentity"],
                "sharedRolloutSettings": {key: run["manifest"].get(key)
                    for key in SHARED_ROLLOUT_SETTINGS},
                "manifestHash": run["manifest"]["manifestHash"],
                "manifestSha256": file_sha256(run["manifestPath"]),
                "positions": len(run["records"]),
            } for run in loaded],
            "positions": len(files),
            "positionsBySplit": dict(sorted(splits.items())),
            "policyFamilies": sorted(families),
            "highConfidencePolicyLabels": policy_labels,
            "files": files,
            "rawInputsRemainImmutable": True,
        }
        combined["manifestHash"] = identity_hash(combined)
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary_root.replace(output)
        return combined
    except Exception:
        shutil.rmtree(temporary_root)
        raise


def combine_macro_label_training_repair_runs(*, base_runs: dict[str, Path],
        repair_runs: dict[str, Path], datasets: dict[str, Path],
        support_reports: dict[str, Path], output: Path, identity: dict,
        selection_path: Path, parent_selection_path: Path) -> dict:
    """Combine immutable v17 runs with only the explicitly frozen train repairs.

    The parent runs remain unchanged. Superseded single-candidate records are
    excluded from the new ranker input, while every retained/repair record is
    bound to the exact source-run manifest that produced it.
    """
    expected_families = {"python-heuristic", "typescript-heuristic"}
    if (set(base_runs) != expected_families or set(repair_runs) != expected_families
            or set(datasets) != expected_families or set(support_reports) != expected_families):
        raise ValueError("training repair merge requires exact inputs for both policy families")
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("training repair outputs are immutable; choose a new directory")
    selection_path, parent_selection_path = Path(selection_path).resolve(), Path(parent_selection_path).resolve()
    selection, expected = load_frozen_selection(selection_path, identity=identity)
    parent, parent_expected = load_frozen_selection(parent_selection_path, identity=identity)
    lineage = selection.get("lineage")
    if (not isinstance(lineage, dict)
            or lineage.get("kind") != "same-source-game-train-root-repair-v1"
            or lineage.get("parentSelectionHash") != parent.get("selectionHash")
            or lineage.get("parentSelectionManifestSha256") != file_sha256(parent_selection_path)
            or lineage.get("developmentPositionHashesUnchanged") is not True
            or lineage.get("heldoutPositionsSelected") is not False
            or lineage.get("rawParentLabelsMustRemainImmutable") is not True):
        raise ValueError("training repair selection has missing or mismatched parent lineage")
    replacement_rows = lineage.get("replacements")
    if not isinstance(replacement_rows, list) or not replacement_rows:
        raise ValueError("training repair selection has no explicit replacements")
    parent_rows = {row["positionHash"]: row for split in parent["splits"] for row in split.get("positions", [])}
    current_rows = {row["positionHash"]: row for split in selection["splits"] for row in split.get("positions", [])}
    superseded, replacement_hashes = set(), set()
    replacements_by_family = {family: set() for family in expected_families}
    for item in replacement_rows:
        if (not isinstance(item, dict) or item.get("policyFamily") not in expected_families
                or item.get("split") != "train"
                or not isinstance(item.get("sourceGameId"), str)
                or not isinstance(item.get("supersededPositionHash"), str)
                or not isinstance(item.get("replacementPositionHash"), str)
                or type(item.get("replacementCompleteCandidateCount")) is not int
                or item["replacementCompleteCandidateCount"] < 2):
            raise ValueError("training repair replacement record is malformed or unsupported")
        old_hash, new_hash = item["supersededPositionHash"], item["replacementPositionHash"]
        old, new = parent_rows.get(old_hash), current_rows.get(new_hash)
        if (old_hash in superseded or new_hash in replacement_hashes or old is None or new is None
                or old_hash not in parent_expected[item["policyFamily"]]["train"]
                or new_hash not in expected[item["policyFamily"]]["train"]
                or old.get("sourceGameId") != item["sourceGameId"]
                or new.get("sourceGameId") != item["sourceGameId"]):
            raise ValueError("training repair replacement does not preserve its exact parent source game")
        superseded.add(old_hash)
        replacement_hashes.add(new_hash)
        replacements_by_family[item["policyFamily"]].add(new_hash)
    for family in expected_families:
        if (expected[family]["development"] != parent_expected[family]["development"]
                or expected[family]["train"] != (parent_expected[family]["train"] -
                    {item["supersededPositionHash"] for item in replacement_rows
                     if item["policyFamily"] == family} | replacements_by_family[family])):
            raise ValueError("training repair selection changes positions beyond its declared train replacements")

    run_rows, records_by_family = [], {family: {} for family in expected_families}
    all_versions, all_hashes, shared_settings = [], [], None
    generator_identity = None
    support_generators = {}
    source_manifests = []

    def read_run(path: Path, family: str, role: str) -> dict:
        nonlocal shared_settings, generator_identity
        directory = Path(path).resolve()
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"training repair source run manifest is missing or a symlink: {directory}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("manifestHash") != identity_hash(
                {key: value for key, value in manifest.items() if key != "manifestHash"})
                or manifest.get("identity") != identity
                or manifest.get("candidateGeneratorIdentity") is None
                or not isinstance(manifest.get("files"), list)
                or len(manifest["files"]) != manifest.get("positions")):
            raise ValueError(f"training repair source run manifest is invalid: {directory}")
        settings = {key: manifest.get(key) for key in SHARED_ROLLOUT_SETTINGS}
        if shared_settings is None:
            shared_settings = settings
        elif settings != shared_settings:
            raise ValueError("training repair source runs have different shared rollout settings")
        current_generator = manifest.get("candidateGeneratorIdentity")
        if generator_identity is None:
            generator_identity = current_generator
        elif current_generator != generator_identity:
            raise ValueError("training repair source runs use different candidate generators")
        listed, records = [], {}
        for file_record in manifest["files"]:
            name = file_record.get("path") if isinstance(file_record, dict) else None
            if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".json")
                    or name == "manifest.json"):
                raise ValueError("training repair source run contains an unsafe record path")
            path = directory / name
            if path.is_symlink() or not path.is_file() or file_sha256(path) != file_record.get("sha256"):
                raise ValueError(f"training repair source record hash/path mismatch: {name}")
            record = json.loads(path.read_text())
            position_hash = record.get("positionHash")
            if (name != f"{position_hash}.json" or position_hash in records
                    or record.get("identity") != identity
                    or record.get("datasetManifestHash") != manifest.get("datasetManifestHash")
                    or record.get("rolloutIdentity") != manifest.get("rolloutIdentity")
                    or record.get("opponentPolicyFamily") != family
                    or record.get("status") != "collected"
                    or record.get("split") not in RANKER_SPLITS):
                raise ValueError(f"training repair source record identity/status mismatch: {name}")
            records[position_hash] = (path, file_record, record)
            listed.append(position_hash)
        if set(listed) != set(manifest.get("selectedPositionHashes", [])):
            raise ValueError("training repair source manifest position list differs from its records")
        collector_version = manifest.get("labelCollectorVersion")
        collector_hash = manifest.get("labelCollectorSha256")
        rollout_identity = manifest.get("rolloutIdentity")
        dataset_hash = manifest.get("datasetManifestHash")
        if (not isinstance(collector_version, str) or not collector_version
                or not isinstance(collector_hash, str) or len(collector_hash) != 64
                or not isinstance(rollout_identity, str) or not rollout_identity
                or not isinstance(dataset_hash, str) or not dataset_hash):
            raise ValueError("training repair source collector/rollout provenance is invalid")
        all_versions.append(collector_version)
        all_hashes.append(collector_hash)
        row = {"directory": directory, "manifestPath": manifest_path, "manifest": manifest,
            "records": records, "family": family, "role": role, "settings": settings}
        source_manifests.append(row)
        return row

    for family in sorted(expected_families):
        dataset_dir = Path(datasets[family]).resolve()
        dataset_manifest, dataset_rows = load_dataset(dataset_dir, identity=identity)
        support_path = Path(support_reports[family]).resolve()
        support = json.loads(support_path.read_text())
        if (support.get("identity") != identity
                or support.get("datasetManifestHash") != dataset_manifest.get("manifestHash")
                or support.get("datasetRowsSha256") != file_sha256(dataset_dir / "rows.jsonl")
                or support.get("reportHash") != identity_hash(
                    {key: value for key, value in support.items() if key != "reportHash"})):
            raise ValueError(f"training repair dataset/support audit mismatch for {family}")
        support_generators[family] = support.get("candidateGeneratorIdentity")
        pool_record = next((item for item in selection.get("sourcePools", [])
                            if item.get("policyFamily") == family), None)
        if (not isinstance(pool_record, dict)
                or pool_record.get("datasetManifestHash") != dataset_manifest.get("manifestHash")
                or pool_record.get("datasetManifestSha256") != file_sha256(dataset_dir / "manifest.json")
                or pool_record.get("datasetRowsSha256") != file_sha256(dataset_dir / "rows.jsonl")
                or pool_record.get("supportReportHash") != support.get("reportHash")
                or pool_record.get("supportReportSha256") != file_sha256(support_path)):
            raise ValueError(f"training repair selection source-pool provenance mismatch for {family}")
        dataset_by_hash = {row["positionHash"]: row for row in dataset_rows}
        support_by_hash = {row["positionHash"]: row for row in support.get("positions", [])
                           if isinstance(row, dict)}
        for split in RANKER_SPLITS:
            for position_hash in expected[family][split]:
                selected_row = current_rows.get(position_hash)
                source_row = dataset_by_hash.get(position_hash)
                support_row = support_by_hash.get(position_hash)
                if (selected_row is None or source_row is None or support_row is None
                        or source_row.get("split") != split
                        or source_row.get("opponentPolicyFamily") != family
                        or any(source_row.get(key) != selected_row.get(key) for key in (
                            "sourceGameId", "targetDeck", "opponentArchetype", "positionStage"))
                        or support_row.get("status") != "supported"
                        or type(support_row.get("completeCandidateCount")) is not int
                        or support_row["completeCandidateCount"] < 2):
                    raise ValueError("training repair selection contains an unsupported or mismatched root")

        base = read_run(base_runs[family], family, "parent-retained")
        supplement = read_run(repair_runs[family], family, "train-repair")
        if (base["manifest"].get("candidateGeneratorIdentity") != support_generators[family]
                or supplement["manifest"].get("candidateGeneratorIdentity") != support_generators[family]):
            raise ValueError(f"training repair candidate-generator identity differs from support for {family}")
        if (base["manifest"].get("datasetManifestHash") != dataset_manifest.get("manifestHash")
                or supplement["manifest"].get("datasetManifestHash") != dataset_manifest.get("manifestHash")):
            raise ValueError(f"training repair run uses another dataset for {family}")
        if set(base["records"]) != set().union(*parent_expected[family].values()):
            raise ValueError(f"parent run does not exactly cover frozen {family} selection")
        expected_repairs = replacements_by_family[family]
        if set(supplement["records"]) != expected_repairs:
            raise ValueError(f"supplement run does not exactly cover frozen {family} replacements")
        superseded_for_family = {item["supersededPositionHash"] for item in replacement_rows
                                 if item["policyFamily"] == family}
        for position_hash, source_tuple in base["records"].items():
            record = source_tuple[2]
            if position_hash in superseded_for_family:
                continue
            if position_hash not in expected[family][record["split"]]:
                raise ValueError("parent run contributes a position excluded by the repair selection")
            records_by_family[family][position_hash] = (base, source_tuple)
        for position_hash, source_tuple in supplement["records"].items():
            record = source_tuple[2]
            lineage_row = next(item for item in replacement_rows
                               if item["replacementPositionHash"] == position_hash)
            if (record["split"] != "train"
                    or current_rows[position_hash].get("sourceGameId") != lineage_row["sourceGameId"]):
                raise ValueError("supplement label does not match replacement family/split/source game")
            records_by_family[family][position_hash] = (supplement, source_tuple)
        if set(records_by_family[family]) != set().union(*expected[family].values()):
            raise ValueError(f"merged repair records do not exactly cover frozen {family} selection")

    compatibility = collector_compatibility(versions=all_versions, source_hashes=all_hashes)
    source_hashes = compatibility["sourceModuleSha256s"]
    collector_sha = source_hashes[0] if len(source_hashes) == 1 else None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        files, splits, source_run_rows = [], Counter(), []
        policy_labels = 0
        for source_run in source_manifests:
            family, role = source_run["family"], source_run["role"]
            included = sorted(position_hash for position_hash, (owner, _source) in records_by_family[family].items()
                              if owner is source_run)
            included_splits = {split: sorted(position_hash for position_hash in included
                if records_by_family[family][position_hash][1][2]["split"] == split)
                for split in sorted(RANKER_SPLITS)}
            source = source_run["manifest"]
            source_run_rows.append({
                "policyFamilies": [family], "sourceRole": role,
                "datasetManifestHash": source["datasetManifestHash"],
                "rolloutIdentity": source["rolloutIdentity"],
                "labelCollectorVersion": source["labelCollectorVersion"],
                "labelCollectorSha256": source["labelCollectorSha256"],
                "candidateGeneratorVersion": source.get("candidateGeneratorVersion"),
                "candidateGeneratorIdentity": source["candidateGeneratorIdentity"],
                "sharedRolloutSettings": {key: source.get(key) for key in SHARED_ROLLOUT_SETTINGS},
                "manifestHash": source["manifestHash"],
                "manifestSha256": file_sha256(source_run["manifestPath"]),
                "sourceRunPositions": source["positions"],
                "positions": len(included),
                "includedPositionHashes": included,
                "includedPositionSplits": included_splits,
            })
            for position_hash in included:
                source_path, _file_record, record = source_run["records"][position_hash]
                relative = Path(family) / source_path.name
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                files.append({"path": relative.as_posix(), "sha256": file_sha256(target)})
                splits[record["split"]] += 1
                policy_labels += int(record.get("highConfidencePolicyEligible") is True)
        files.sort(key=lambda item: item["path"])
        family_split_counts = {family: {split: len(expected[family][split])
            for split in sorted(RANKER_SPLITS)} for family in sorted(expected_families)}
        combined = {
            "schemaVersion": 1,
            "kind": "combined-macro-label-runs-v1",
            "identity": identity,
            "selectionHash": selection["selectionHash"],
            "selectionManifestSha256": file_sha256(selection_path),
            "selectionLineage": lineage,
            "parentSelectionHash": parent["selectionHash"],
            "selectedPositionHashes": sorted(position for family in expected_families
                for split in RANKER_SPLITS for position in expected[family][split]),
            "positionsByFamilySplit": family_split_counts,
            "candidateGeneratorIdentity": generator_identity,
            "labelCollectorVersion": compatibility["labelCollectorVersion"],
            "labelCollectorSha256": collector_sha,
            "labelCollectorSourceSha256s": source_hashes,
            "labelCollectorCompatibility": compatibility,
            "sharedRolloutSettings": shared_settings,
            "sourceRuns": source_run_rows,
            "positions": len(files),
            "positionsBySplit": dict(sorted(splits.items())),
            "policyFamilies": sorted(expected_families),
            "highConfidencePolicyLabels": policy_labels,
            "files": files,
            "rawInputsRemainImmutable": True,
        }
        combined["manifestHash"] = identity_hash(combined)
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary_root.replace(output)
        return combined
    except Exception:
        shutil.rmtree(temporary_root)
        raise


def finalize_macro_label_runs(*, inputs: list[Path], combined_output: Path,
                              confidence_output: Path, identity: dict,
                              selection_path: Path) -> dict:
    """Combine finalized family runs and independently verify their confidence report.

    This is an offline evidence operation: it never launches rollouts, mutates
    source runs, changes gates, or fits a model. If confidence generation fails,
    the already-verified combined labels remain available for diagnosis/retry.
    """
    combined_output = combined_output.resolve()
    confidence_output = confidence_output.resolve()
    if combined_output == confidence_output or confidence_output.is_relative_to(combined_output):
        raise ValueError("confidence report must be outside the combined-label directory")
    if combined_output.exists() or combined_output.is_symlink():
        raise ValueError("combined macro-label outputs are immutable; choose a new directory")
    if confidence_output.exists() or confidence_output.is_symlink():
        raise ValueError("confidence audit reports are immutable; choose a new output path")

    combined = combine_macro_label_runs(inputs=inputs, output=combined_output,
        identity=identity, selection_path=selection_path)
    from .confidence_audit import (audit_macro_label_confidence,
        verify_macro_label_confidence_audit)
    confidence = audit_macro_label_confidence(labels_dir=combined_output,
        selection_path=selection_path, output=confidence_output)
    verified = verify_macro_label_confidence_audit(labels_dir=combined_output,
        selection_path=selection_path, report_path=confidence_output)
    if verified != confidence:
        raise ValueError("confidence audit changed during final verification")
    return {"status": "verified-analysis-only", "combinedManifestHash": combined["manifestHash"],
        "combinedPositions": combined["positions"], "confidenceReportHash": verified["reportHash"],
        "confidenceReportPath": str(confidence_output), "policyLabelEligibilityChanged": False,
        "ppoEnablement": False, "promotionAuthority": "none"}
