from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile

from ptcg_lab.storage import digest as observation_digest

from .aggregation import load_frozen_selection
from .dataset_v1 import file_sha256, load_dataset
from .macro import CANDIDATE_GENERATOR_VERSION, MacroCandidateV1, rollout_seed
from .schema import identity_hash


POLICY_FAMILIES = ("python-heuristic", "typescript-heuristic")
TYPED_PLAN_FAILURE_PREFIXES = ("MACRO_PLAN_UNEXECUTABLE:", "macro-plan-unexecuted")


def _new_json(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError("macro-plan fidelity reports are immutable; choose a new output path")
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


def _macro_candidate(raw: dict) -> MacroCandidateV1:
    fields = {"turn_intent", "intended_attack", "attack_target", "supporter", "pivot_destination",
        "energy_source", "energy_destination", "disruption_intent", "protected_pokemon", "setup_target",
        "action_ids", "action_sequence"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("macro candidate has an unknown or incomplete schema")
    if (not isinstance(raw["action_ids"], list) or not isinstance(raw["action_sequence"], list)
            or not raw["action_ids"] or len(raw["action_ids"]) != len(raw["action_sequence"])
            or any(not isinstance(value, str) or not value for value in raw["action_ids"])
            or any(not isinstance(action, dict) for action in raw["action_sequence"])):
        raise ValueError("macro candidate action sequence is malformed")
    candidate = MacroCandidateV1(**{**raw, "action_ids": tuple(raw["action_ids"]),
                                    "action_sequence": tuple(raw["action_sequence"])})
    if [str(action.get("id")) for action in candidate.action_sequence] != list(candidate.action_ids):
        raise ValueError("macro candidate action IDs differ from its declared sequence")
    if not candidate.action_sequence:
        raise ValueError("macro candidate has no declared action sequence")
    attacks = [action for action in candidate.action_sequence if action.get("type") == "attack"]
    final = candidate.action_sequence[-1]
    if candidate.turn_intent == "attack":
        if len(attacks) != 1 or final.get("type") != "attack":
            raise ValueError("attack plan does not end in exactly one declared attack")
        attack_name = str(attacks[0].get("cardId") or attacks[0].get("label") or attacks[0].get("id"))
        if candidate.intended_attack != attack_name:
            raise ValueError("declared attack differs from the action sequence")
        attack_target = str(attacks[0].get("target")) if attacks[0].get("target") is not None else None
        if candidate.attack_target != attack_target:
            raise ValueError("declared attack target differs from the action sequence")
    elif candidate.turn_intent == "no-attack":
        if attacks or final.get("type") != "pass" or candidate.intended_attack is not None:
            raise ValueError("deliberate no-attack plan does not end in a pass")
    else:
        raise ValueError("macro candidate is not a complete attack/no-attack plan")
    if identity_hash(raw) != candidate.key():
        raise ValueError("macro candidate identity hash mismatch")
    return candidate


def _validate_rollout_label(label: dict) -> dict:
    if not isinstance(label, dict) or not isinstance(label.get("candidate"), dict):
        raise ValueError("macro label lacks its complete candidate record")
    candidate = _macro_candidate(label["candidate"])
    if label.get("candidateHash") != candidate.key():
        raise ValueError("macro label candidate hash mismatch")
    outcomes = label.get("outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != {"finished", "truncated", "error"}:
        raise ValueError("macro label outcome counts are malformed")
    counts = {}
    for name, value in outcomes.items():
        if type(value) is not int or value < 0:
            raise ValueError("macro label has a negative or non-integer outcome count")
        counts[name] = value
    attempted = label.get("attemptedRollouts")
    if type(attempted) is not int or attempted < 1 or sum(counts.values()) != attempted:
        raise ValueError("macro label outcome counts do not reconcile with attempted rollouts")
    if label.get("completedRollouts") != counts["finished"]:
        raise ValueError("macro label completed count differs from finished outcomes")
    reasons = label.get("outcomeReasons")
    if not isinstance(reasons, dict) or any(type(value) is not int or value < 0 for value in reasons.values()):
        raise ValueError("macro label outcome reasons are malformed")
    unfinished_samples = counts["truncated"] + counts["error"]
    if sum(reasons.values()) != unfinished_samples:
        raise ValueError("macro label truncations and errors are not fully explained by typed reasons")
    decision_counts = label.get("decisionCountDistribution")
    if (not isinstance(decision_counts, dict)
            or any(not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                   or type(value) is not int or value < 0
                   for key, value in decision_counts.items())):
        raise ValueError("macro label decision-count distribution is malformed")
    observed_decisions = sum(decision_counts.values())
    if not counts["finished"] + counts["truncated"] <= observed_decisions <= attempted:
        raise ValueError("macro label decision-count distribution does not reconcile with outcomes")
    typed_errors = sum(value for reason, value in reasons.items()
                       if reason.startswith(TYPED_PLAN_FAILURE_PREFIXES))
    successful_plan_samples = counts["finished"] + counts["truncated"]
    return {"candidate": candidate, "successfulPlanSamples": successful_plan_samples,
            "typedPlanFailures": typed_errors,
            "untypedErrors": counts["error"] - typed_errors,
            "attemptedRollouts": attempted}


def _load_label_run(*, root: Path, family: str, dataset_path: Path, labels_path: Path,
                    identity: dict, expected_by_split: dict[str, set[str]]) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    dataset_manifest, dataset_rows = load_dataset(dataset_path, identity=identity)
    rows_by_hash = {row.get("positionHash"): row for row in dataset_rows}
    if len(rows_by_hash) != len(dataset_rows):
        raise ValueError("macro-position dataset has duplicate position hashes")
    manifest_path = labels_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifestHash") != identity_hash({key: value for key, value in manifest.items()
                                                       if key != "manifestHash"}):
        raise ValueError("macro-label run manifest hash mismatch")
    if manifest.get("identity") != identity or manifest.get("datasetManifestHash") != dataset_manifest.get("manifestHash"):
        raise ValueError("macro-label run does not match the frozen dataset identity")
    if (manifest.get("labelCollectorVersion") != "macro-rollout-labeler-v6"
            or manifest.get("labelCollectorSha256") != file_sha256(root / "src/ptcg_lab/learning_mind/experiment.py")):
        raise ValueError("macro-label run collector code identity differs from the audited source")
    planner_paths = {
        "plannerSha256": root / "research/learning_mind/transition_macro_planner.ts",
        "actionKeySha256": root / "packages/engine/src/action-key.ts",
        "plannerWorkerSha256": root / "research/learning_mind/planner_worker.ts",
        "plannerBundleSha256": root / "packages/engine/dist/learning-mind-planner.cjs",
        "adapterSha256": root / "src/ptcg_lab/learning_mind/macro.py",
    }
    if any(not path.is_file() for path in planner_paths.values()):
        raise ValueError("current transition-planner identity inputs are incomplete")
    current_generator = {"version": CANDIDATE_GENERATOR_VERSION,
        **{key: file_sha256(path) for key, path in planner_paths.items()}}
    if (manifest.get("candidateGeneratorVersion") != CANDIDATE_GENERATOR_VERSION
            or manifest.get("candidateGeneratorIdentity") != current_generator):
        raise ValueError("macro-label candidate generator differs from the audited source")
    if family not in POLICY_FAMILIES:
        raise ValueError("macro-label run has ambiguous policy-family provenance")
    expected = set().union(*expected_by_split.values())
    if set(manifest.get("selectedPositionHashes", [])) != expected:
        raise ValueError(f"macro-label run does not exactly cover frozen {family} positions")
    if (manifest.get("requestedPositions") != len(expected)
            or manifest.get("splitFilter") is not None
            or manifest.get("selectionMethod") != "position-hash-list"):
        raise ValueError("macro-label run did not use the exact frozen train/development position list")
    required_settings = {"identity", "datasetManifestHash", "requestedPositions", "initialRollouts",
        "maximumRollouts", "extensionBatchSize", "horizon", "rolloutBudgetMs", "rolloutWorkers",
        "splitFilter", "selectionMethod", "selectedPositionHashes", "candidateGeneratorVersion",
        "candidateGeneratorIdentity", "rolloutSeedVersion", "adaptiveAllocationVersion",
        "labelCollectorVersion", "labelCollectorSha256"}
    if not required_settings.issubset(manifest):
        raise ValueError("macro-label run manifest omits rollout identity settings")
    maximum_rollouts = manifest.get("maximumRollouts")
    initial_rollouts = manifest.get("initialRollouts")
    if (type(maximum_rollouts) is not int or not 1 <= maximum_rollouts <= 64
            or type(initial_rollouts) is not int or not 1 <= initial_rollouts <= maximum_rollouts
            or manifest.get("rolloutSeedVersion") != "configuration-bound-v1"):
        raise ValueError("macro-label run rollout seed configuration is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(expected) or manifest.get("positions") != len(expected):
        raise ValueError("macro-label run file list differs from frozen position selection")
    expected_paths = {f"{position}.json" for position in expected}
    listed_paths = [item.get("path") if isinstance(item, dict) else None for item in files]
    if (any(not isinstance(name, str) or Path(name).name != name for name in listed_paths)
            or len(set(listed_paths)) != len(expected) or set(listed_paths) != expected_paths):
        raise ValueError("macro-label manifest does not list every frozen position exactly once")
    actual_paths = {path.name for path in labels_path.glob("*.json") if path.name != "manifest.json"}
    if actual_paths != expected_paths:
        raise ValueError("macro-label run has missing or unexpected position files")
    settings_keys = ("schemaVersion", "identity", "datasetManifestHash", "requestedPositions",
        "initialRollouts", "maximumRollouts", "extensionBatchSize", "horizon", "rolloutBudgetMs",
        "rolloutWorkers", "splitFilter", "selectionMethod", "selectedPositionHashes",
        "candidateGeneratorVersion", "candidateGeneratorIdentity", "rolloutSeedVersion",
        "adaptiveAllocationVersion", "labelCollectorVersion", "labelCollectorSha256")
    rollout_settings = {key: manifest[key] for key in settings_keys if key in manifest and key != "schemaVersion"}
    if (manifest.get("schemaVersion") != 1
            or manifest.get("rolloutIdentity") != identity_hash(rollout_settings)):
        raise ValueError("macro-label rollout identity does not match its frozen settings")
    if (manifest.get("positions") != len(expected)
            or manifest.get("supportedPositions") != len(expected)
            or manifest.get("unsupportedPositions") != 0
            or manifest.get("highConfidencePolicyLabels") != 0):
        raise ValueError("macro-label run completion or conservative label-status totals are inconsistent")
    records = {}
    for item in files:
        name = item.get("path") if isinstance(item, dict) else None
        if not isinstance(name, str) or Path(name).name != name or name not in expected_paths:
            raise ValueError("macro-label manifest contains an unsafe or unexpected record path")
        path = labels_path / name
        if file_sha256(path) != item.get("sha256"):
            raise ValueError(f"macro-label record checksum mismatch: {name}")
        record = json.loads(path.read_text())
        position_hash = path.stem
        row = rows_by_hash.get(position_hash)
        if (row is None or record.get("positionHash") != position_hash or record.get("identity") != identity
                or record.get("datasetManifestHash") != dataset_manifest.get("manifestHash")
                or record.get("rolloutIdentity") != manifest.get("rolloutIdentity")
                or record.get("opponentPolicyFamily") != family or row.get("opponentPolicyFamily") != family
                or record.get("split") != row.get("split")
                or record.get("status") != "collected" or record.get("highConfidencePolicyEligible") is not False):
            raise ValueError(f"macro-label record provenance/status mismatch: {name}")
        if record.get("split") not in expected_by_split or position_hash not in expected_by_split[record["split"]]:
            raise ValueError(f"macro-label record is outside frozen family/split selection: {name}")
        namespace = "training" if record["split"] == "train" else "development"
        expected_seeds = [rollout_seed(namespace, position_hash, index, manifest["rolloutIdentity"])
                          for index in range(maximum_rollouts)]
        generator_seed = int.from_bytes(hashlib.sha256(
            f"learning-mind-v1|macro-generator|{position_hash}".encode()).digest()[:4], "big")
        if (record.get("seedNamespace") != namespace
                or record.get("rolloutSeeds") != expected_seeds
                or record.get("generatorSeed") != generator_seed):
            raise ValueError(f"macro-label record seeds differ from frozen position/split identity: {name}")
        if observation_digest(record.get("observation")) != position_hash:
            raise ValueError(f"macro-label record actor observation hash mismatch: {name}")
        if record.get("observation") != row.get("observation"):
            raise ValueError(f"macro-label record actor observation differs from frozen dataset: {name}")
        labels = record.get("labels")
        if not isinstance(labels, list) or len(labels) != record.get("candidateCount") or not labels:
            raise ValueError(f"macro-label record has no complete candidate labels: {name}")
        candidate_hashes = set()
        for label in labels:
            evidence = _validate_rollout_label(label)
            if evidence["attemptedRollouts"] > maximum_rollouts:
                raise ValueError(f"macro candidate exceeds frozen rollout seed allocation: {name}")
            candidate = evidence["candidate"]
            if candidate.key() in candidate_hashes:
                raise ValueError(f"macro-label position repeats a candidate: {name}")
            candidate_hashes.add(candidate.key())
            root_actions = {str(action.get("id")): action for action in row["observation"].get("legalActions", [])}
            first = candidate.action_sequence[0]
            if root_actions.get(str(first.get("id"))) != first:
                raise ValueError(f"macro candidate root action is not bound to the frozen actor observation: {name}")
        records[position_hash] = record
    return manifest, records, rows_by_hash


def audit_raging_bolt_macro_fidelity(*, root: Path, selection_path: Path,
                                     python_dataset: Path, typescript_dataset: Path,
                                     python_labels: Path, typescript_labels: Path,
                                     output: Path) -> dict:
    """Audit real collected Raging Bolt plans without running games or touching collector state."""
    root = root.resolve()
    selection = json.loads(selection_path.read_text())
    identity = selection.get("identity")
    selection, expected = load_frozen_selection(selection_path, identity=identity)
    run_inputs = {
        "python-heuristic": (python_dataset, python_labels),
        "typescript-heuristic": (typescript_dataset, typescript_labels),
    }
    all_records: dict[str, dict] = {}
    all_source_rows: dict[str, dict] = {}
    run_hashes = {}
    dataset_hashes = {}
    for family, (dataset_path, labels_path) in run_inputs.items():
        manifest, records, source_rows = _load_label_run(root=root, family=family,
            dataset_path=dataset_path.resolve(),
            labels_path=labels_path.resolve(), identity=identity, expected_by_split=expected[family])
        run_hashes[family] = file_sha256(labels_path.resolve() / "manifest.json")
        dataset_hashes[family] = file_sha256(dataset_path.resolve() / "manifest.json")
        if set(all_records).intersection(records):
            raise ValueError("Raging Bolt audit inputs reuse frozen position hashes across policy families")
        for position_hash, record in records.items():
            all_records[position_hash] = {**record, "_family": family}
            all_source_rows[position_hash] = source_rows[position_hash]

    position_results = []
    candidate_count = successful_candidates = typed_failure_candidates = 0
    untyped_error_count = 0
    for position_hash, record in sorted(all_records.items()):
        source = all_source_rows[position_hash]
        if source.get("targetDeck") != "raging-bolt":
            continue
        candidate_results = []
        for label in record["labels"]:
            evidence = _validate_rollout_label(label)
            candidate_count += 1
            successful = evidence["successfulPlanSamples"] > 0
            typed_only = evidence["attemptedRollouts"] == evidence["typedPlanFailures"]
            untyped_error_count += evidence["untypedErrors"]
            if successful:
                successful_candidates += 1
            elif typed_only:
                typed_failure_candidates += 1
            candidate_results.append({"candidateHash": label["candidateHash"],
                "turnIntent": evidence["candidate"].turn_intent,
                "successfulPlanSamples": evidence["successfulPlanSamples"],
                "typedPlanFailures": evidence["typedPlanFailures"],
                "untypedErrors": evidence["untypedErrors"],
                "fidelityDisposition": "executed" if successful else
                    "typed-fail-closed" if typed_only else "insufficient"})
        position_results.append({"positionHash": position_hash, "split": record["split"],
            "policyFamily": record["_family"], "candidateCount": len(candidate_results),
            "successfulCandidates": sum(item["fidelityDisposition"] == "executed" for item in candidate_results),
            "candidates": candidate_results})

    every_position_exercised = bool(position_results) and all(row["successfulCandidates"] > 0
                                                              for row in position_results)
    all_candidates_accounted = all(candidate["fidelityDisposition"] in {"executed", "typed-fail-closed"}
                                   for position in position_results for candidate in position["candidates"])
    status = ("passed" if every_position_exercised and all_candidates_accounted and untyped_error_count == 0
              else "insufficient" if not position_results or not successful_candidates
              else "failed")
    report = {"schemaVersion": 1, "kind": "raging-bolt-macro-plan-fidelity-v1",
        "status": status, "ragingBoltMacroPlanFidelity": status,
        "selectionHash": selection["selectionHash"],
        "selectionManifestSha256": file_sha256(selection_path.resolve()),
        "identity": identity, "collectorSha256": file_sha256(root / "src/ptcg_lab/learning_mind/experiment.py"),
        "runManifestSha256": run_hashes, "datasetManifestSha256": dataset_hashes,
        "ragingBoltPositions": len(position_results), "candidates": candidate_count,
        "successfulCandidates": successful_candidates,
        "typedFailureCandidates": typed_failure_candidates,
        "untypedErrors": untyped_error_count,
        "positions": position_results,
        "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    _new_json(output, report)
    return report
