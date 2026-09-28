from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .dataset_v1 import file_sha256
from .experiment import _load_ranker_input, _ranker_evidence_status, _ranker_holdout_rows
from .ranker import FrozenIteration, XGBoostMacroRanker
from .ranker_features import (MACRO_FEATURE_SCHEMA, MACRO_FEATURE_SCHEMA_HASH,
                              candidate_features_v2)
from .ranker_portable import inference_implementation_sha256, predict_macro_ranker_v2
from .schema import identity_hash
from .macro_fidelity import _validate_rollout_label
from .macro import rollout_seed
from ptcg_lab.storage import digest as observation_digest


def _bootstrap_mean(values: list[float], *, seed_material: str, group_ids: list[str] | None = None,
                    replicates: int = 2000) -> dict:
    if not values:
        return {"mean": None, "interval95": {"low": None, "high": None},
                "method": "source-game-cluster-bootstrap-percentile-v1",
                "replicates": replicates, "seed": None, "independentUnits": 0}
    values_array = np.asarray(values, dtype=np.float64)
    if group_ids is None:
        group_ids = [f"position-{index}" for index in range(len(values))]
    if len(group_ids) != len(values) or any(not isinstance(value, str) or not value for value in group_ids):
        raise ValueError("bootstrap cluster IDs must be nonempty and match the values")
    clusters: dict[str, list[float]] = {}
    for group_id, value in zip(group_ids, values_array):
        clusters.setdefault(group_id, []).append(float(value))
    cluster_keys = sorted(clusters)
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for sample_index in range(replicates):
        drawn = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
        cluster_values = [value for index in drawn for value in clusters[cluster_keys[int(index)]]]
        samples[sample_index] = float(np.mean(cluster_values))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(values_array.mean()),
            "interval95": {"low": float(low), "high": float(high)},
            "method": "source-game-cluster-bootstrap-percentile-v1", "replicates": replicates,
            "seed": seed, "independentUnits": len(cluster_keys)}


def _validate_source_game_units(selection_path: Path, records: list[dict]) -> None:
    selection = json.loads(selection_path.read_text())
    selected_source_games = {}
    split_rows = selection.get("splits")
    if not isinstance(split_rows, list):
        raise ValueError("ranker v2 selection has no split records")
    for item in split_rows:
        family, split = item.get("policyFamily"), item.get("split")
        positions = item.get("positions")
        hashes = item.get("positionHashes")
        if (family not in {"python-heuristic", "typescript-heuristic"}
                or split not in {"train", "development"}
                or not isinstance(positions, list) or not isinstance(hashes, list)
                or len(positions) != len(hashes) or item.get("sourceGames") != len(hashes)
                or (family, split) in selected_source_games):
            raise ValueError("ranker v2 selection omits source-game position metadata")
        position_games = {}
        for position in positions:
            if (not isinstance(position, dict) or position.get("positionHash") not in hashes
                    or not isinstance(position.get("sourceGameId"), str)
                    or not position["sourceGameId"]
                    or position["positionHash"] in position_games):
                raise ValueError("ranker v2 frozen selection has invalid source-game metadata")
            position_games[position["positionHash"]] = position["sourceGameId"]
        if set(position_games) != set(hashes) or len(set(position_games.values())) != len(position_games):
            raise ValueError("ranker v2 selection reuses or omits a source game within a split")
        selected_source_games[(family, split)] = position_games
    required = {(family, split) for family in ("python-heuristic", "typescript-heuristic")
                for split in ("train", "development")}
    if set(selected_source_games) != required:
        raise ValueError("ranker v2 selection source-game metadata has incomplete family/split coverage")
    for family in ("python-heuristic", "typescript-heuristic"):
        if (set(selected_source_games[(family, "train")].values())
                & set(selected_source_games[(family, "development")].values())):
            raise ValueError("ranker v2 train and development selections reuse a source game")
    for record in records:
        key = (record.get("opponentPolicyFamily"), record.get("split"))
        expected_game = selected_source_games.get(key, {}).get(record.get("positionHash"))
        if (expected_game is None or record.get("sourceGameId") != expected_game):
            raise ValueError("ranker v2 record source game differs from frozen selection")


def validate_ranker_v2_report(report: dict) -> None:
    if not isinstance(report, dict) or report.get("kind") != "xgboost-macro-ranker-v2":
        raise ValueError("not a macro ranker v2 report")
    if (report.get("featureSchema") != MACRO_FEATURE_SCHEMA
            or report.get("featureSchemaHash") != MACRO_FEATURE_SCHEMA_HASH):
        raise ValueError("macro ranker v2 feature schema mismatch")
    current_source_hash = file_sha256(Path(__file__).with_name("ranker_features.py"))
    if report.get("featureImplementationSha256") != current_source_hash:
        raise ValueError("macro ranker v2 feature implementation mismatch")
    if report.get("inferenceImplementationSha256") != inference_implementation_sha256():
        raise ValueError("macro ranker v2 inference implementation mismatch")
    if report.get("reportHash") != identity_hash({key: value for key, value in report.items()
                                                   if key != "reportHash"}):
        raise ValueError("macro ranker v2 report hash mismatch")


def verify_macro_ranker_v2_artifact(model_path: Path, report_path: Path) -> dict:
    """Verify a portable ranker artifact without invoking native XGBoost loading."""
    report = json.loads(report_path.read_text())
    validate_ranker_v2_report(report)
    model_path = model_path.resolve()
    if not model_path.is_file() or file_sha256(model_path) != report.get("modelSha256"):
        raise ValueError("macro ranker v2 model checksum mismatch")
    artifact = json.loads(model_path.read_text())
    if (artifact.get("kind") != "xgboost-macro-ranker-v2-portable"
            or artifact.get("featureSchemaHash") != MACRO_FEATURE_SCHEMA_HASH
            or artifact.get("featureImplementationSha256") != report.get("featureImplementationSha256")
            or artifact.get("inferenceImplementationSha256") != report.get("inferenceImplementationSha256")
            or artifact.get("featureCount") != MACRO_FEATURE_SCHEMA["dimension"]
            or report.get("modelFeatureCount") != MACRO_FEATURE_SCHEMA["dimension"]
            or not isinstance(artifact.get("trees"), list) or not artifact["trees"]):
        raise ValueError("macro ranker v2 model feature dimension mismatch")
    return {"report": report, "artifact": artifact}


def _validated_macro_labels(record: dict, *, expected_rollout_identity: str | None = None) -> list[dict]:
    labels = record.get("labels")
    if (not isinstance(labels, list) or not labels
            or type(record.get("candidateCount")) is not int
            or record["candidateCount"] != len(labels)):
        raise ValueError("ranker v2 record candidate labels are incomplete")
    observation = record.get("observation")
    if observation_digest(observation) != record.get("positionHash"):
        raise ValueError("ranker v2 record actor observation hash mismatch")
    split = record.get("split")
    expected_namespace = {"train": "training", "development": "development"}.get(split)
    namespace = record.get("seedNamespace")
    rollout_identity = record.get("rolloutIdentity")
    if (namespace != expected_namespace or not isinstance(rollout_identity, str) or not rollout_identity
            or (expected_rollout_identity is not None and rollout_identity != expected_rollout_identity)):
        raise ValueError("ranker v2 record seed namespace or rollout identity mismatch")
    seeds = record.get("rolloutSeeds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or seeds != [rollout_seed(namespace, record["positionHash"], index, rollout_identity)
                         for index in range(len(seeds))]):
        raise ValueError("ranker v2 record seed list does not match its frozen namespace and identity")
    legal_actions = {str(action.get("id")): action
                     for action in observation.get("legalActions", []) if isinstance(action, dict)}
    candidates = set()
    for label in labels:
        evidence = _validate_rollout_label(label)
        if evidence["attemptedRollouts"] > len(seeds):
            raise ValueError("ranker v2 candidate attempted more rollouts than its frozen seed list")
        candidate = evidence["candidate"]
        if candidate.key() in candidates:
            raise ValueError("ranker v2 record repeats a macro candidate")
        candidates.add(candidate.key())
        first = candidate.action_sequence[0]
        if legal_actions.get(str(first.get("id"))) != first:
            raise ValueError("ranker v2 candidate root action is not actor-visible and legal")
    completed = []
    for label in labels:
        expected = label.get("expectedResult")
        relative = label.get("relativeResult")
        uncertainty = label.get("uncertainty")
        weight = label.get("weight")
        finished = label["outcomes"]["finished"]
        if expected is None:
            if finished != 0 or relative is not None or uncertainty is not None or weight != 0:
                raise ValueError("ranker v2 missing-score label contains inconsistent targets")
            continue
        if (isinstance(expected, bool) or not isinstance(expected, (int, float))
                or not math.isfinite(expected) or not 0 <= expected <= 1
                or isinstance(relative, bool) or not isinstance(relative, (int, float))
                or not math.isfinite(relative)
                or isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float))
                or not math.isfinite(uncertainty) or uncertainty < 0
                or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight <= 0 or finished <= 0):
            raise ValueError("ranker v2 label has invalid expected result, relative target, or weight")
        expected_uncertainty = math.sqrt(max(float(expected) * (1 - float(expected)), .25) / finished)
        if not math.isclose(float(uncertainty), expected_uncertainty, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("ranker v2 uncertainty does not match completed-rollout count")
        expected_weight = finished / (1 + expected_uncertainty)
        if not math.isclose(float(weight), expected_weight, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("ranker v2 rollout weight does not match completed evidence")
        completed.append(label)
    center = max(float(label["expectedResult"]) for label in completed) if completed else 0.0
    for label in completed:
        if not math.isclose(float(label["relativeResult"]),
                            float(label["expectedResult"]) - center, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("ranker v2 relative target does not match the position's best completed result")
    return completed


def _publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as temporary:
        json.dump(value, temporary, sort_keys=True, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temp_path = Path(temporary.name)
    try:
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def fit_macro_ranker_v2(labels_dir: Path, output: Path, *, selection_path: Path,
                        teacher_hash: str, opponent_policy_hash: str,
                        iteration: int = 1) -> dict:
    """Train feature-v2 ranker without changing collector-bound experiment.py."""
    output = output.resolve()
    report_path = output.with_suffix(".manifest.json")
    if output.exists() or report_path.exists():
        raise ValueError("macro ranker v2 outputs are immutable; choose new output paths")
    labels_manifest, records = _load_ranker_input(labels_dir.resolve(), selection_path.resolve())
    _validate_source_game_units(selection_path.resolve(), records)
    source_runs = labels_manifest.get("sourceRuns")
    expected_rollout_identities = {}
    if not isinstance(source_runs, list):
        raise ValueError("ranker v2 combined labels omit source-run rollout identities")
    for source_run in source_runs:
        families = source_run.get("policyFamilies") if isinstance(source_run, dict) else None
        rollout_identity = source_run.get("rolloutIdentity") if isinstance(source_run, dict) else None
        if (not isinstance(families, list) or len(families) != 1
                or families[0] not in {"python-heuristic", "typescript-heuristic"}
                or not isinstance(rollout_identity, str) or not rollout_identity
                or families[0] in expected_rollout_identities):
            raise ValueError("ranker v2 source-run family/rollout identity is ambiguous")
        expected_rollout_identities[families[0]] = rollout_identity
    if set(expected_rollout_identities) != {"python-heuristic", "typescript-heuristic"}:
        raise ValueError("ranker v2 requires one frozen rollout identity per policy family")
    train_records = [record for record in records if record["split"] == "train"]
    development_records = [record for record in records if record["split"] == "development"]
    if not train_records:
        raise ValueError("macro ranker v2 has no training positions")

    def flatten(selected):
        features, labels, weights, groups, positions, metadata = [], [], [], [], [], []
        for record in selected:
            usable = _validated_macro_labels(record,
                expected_rollout_identity=expected_rollout_identities[record["opponentPolicyFamily"]])
            if len(usable) < 2:
                continue
            groups.append(len(usable))
            positions.append(record["positionHash"])
            metadata.append({key: record[key] for key in
                ("positionHash", "sourceGameId", "split", "opponentArchetype", "opponentPolicyFamily")
                if key in record})
            for item in usable:
                features.append(candidate_features_v2(record["observation"], item["candidate"]))
                labels.append(item["relativeResult"])
                weights.append(item["weight"])
        return features, labels, weights, groups, positions, metadata

    features, labels, weights, groups, positions, _ = flatten(train_records)
    if not groups:
        raise ValueError("macro ranker v2 needs positions with at least two completed candidates")

    def metrics(model, selected):
        x, truth_values, _, sizes, _, metadata = flatten(selected)
        scored_positions = {item["positionHash"] for item in metadata}
        insufficient_positions = sorted(record["positionHash"] for record in selected
            if record["positionHash"] not in scored_positions)
        if not sizes:
            return {"status": "insufficient", "requestedPositions": len(selected),
                    "positions": 0, "insufficientPositionHashes": insufficient_positions}
        predicted = model.predict(x)
        offset = 0
        details = []
        for size, meta in zip(sizes, metadata):
            truth = np.asarray(truth_values[offset:offset + size])
            scores = predicted[offset:offset + size]
            chosen, best = int(np.argmax(scores)), int(np.argmax(truth))
            comparisons = correct = 0
            for left in range(size):
                for right in range(left + 1, size):
                    if truth[left] == truth[right]:
                        continue
                    comparisons += 1
                    correct += int((scores[left] - scores[right]) * (truth[left] - truth[right]) > 0)
            details.append({**meta, "candidates": size,
                "top1RelativeRegret": float(truth[best] - truth[chosen]),
                "top3Recall": bool(best in np.argsort(scores)[-min(3, size):]),
                "pairwiseCorrect": correct, "pairwiseComparisons": comparisons})
            offset += size
        pair_count = sum(row["pairwiseComparisons"] for row in details)
        overall = _bootstrap_mean([row["top1RelativeRegret"] for row in details],
            group_ids=[row["sourceGameId"] for row in details],
            seed_material="macro-ranker-v2|all|" + "|".join(sorted(row["positionHash"] for row in details)))
        by_archetype, by_family = {}, {}
        for field, destination in (("opponentArchetype", by_archetype),
                                   ("opponentPolicyFamily", by_family)):
            for value in sorted({row[field] for row in details}):
                subset = [row for row in details if row[field] == value]
                summary = _bootstrap_mean([row["top1RelativeRegret"] for row in subset],
                    group_ids=[row["sourceGameId"] for row in subset],
                    seed_material=f"macro-ranker-v2|{field}|{value}|" +
                        "|".join(sorted(row["positionHash"] for row in subset)))
                destination[value] = {"positions": len(subset),
                    "meanTop1RelativeRegret": summary["mean"],
                    "meanTop1RelativeRegretCI95": summary["interval95"],
                    "bootstrapSeed": summary["seed"],
                    "independentSourceGames": summary["independentUnits"]}
        return {"status": "measured" if not insufficient_positions else "insufficient",
            "requestedPositions": len(selected), "positions": len(details),
            "insufficientPositionHashes": insufficient_positions,
            "meanTop1RelativeRegret": overall["mean"],
            "meanTop1RelativeRegretCI95": overall["interval95"],
            "bootstrap": {"method": overall["method"], "replicates": overall["replicates"],
                          "seed": overall["seed"], "independentSourceGames": overall["independentUnits"]},
            "top3Recall": sum(row["top3Recall"] for row in details) / len(details),
            "pairwiseOrderingAccuracy": (sum(row["pairwiseCorrect"] for row in details) / pair_count
                                         if pair_count else None),
            "byOpponentArchetype": by_archetype,
            "byOpponentPolicyFamily": by_family,
            "details": details}

    input_hash = labels_manifest["manifestHash"]
    source_hash = file_sha256(Path(__file__).with_name("ranker_features.py"))
    inference_hash = inference_implementation_sha256()
    frozen = FrozenIteration(iteration, teacher_hash, opponent_policy_hash, input_hash, tuple(positions))
    ranker = XGBoostMacroRanker().fit(features, labels, groups, weights)
    training_metrics = metrics(ranker, train_records)
    development_metrics = metrics(ranker, development_records)
    holdouts = []
    for split in _ranker_holdout_rows(records):
        held_train = [records[index] for index in split["train"]]
        held_test = [records[index] for index in split["test"]]
        hx, hy, hw, hg, _, _ = flatten(held_train)
        if not hg or not held_test:
            holdouts.append({**split, "status": "insufficient"})
            continue
        held_model = XGBoostMacroRanker().fit(hx, hy, hg, hw)
        held_metrics = metrics(held_model, held_test)
        holdouts.append({**split, "status": held_metrics["status"], "metrics": held_metrics})

    model_artifact = {
        "schemaVersion": 1,
        "kind": "xgboost-macro-ranker-v2-portable",
        "objective": "rank:pairwise",
        "featureCount": int(ranker.model.num_features()),
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": source_hash,
        "inferenceImplementationSha256": inference_hash,
        "trees": [json.loads(tree) for tree in ranker.model.get_dump(dump_format="json")],
    }
    if model_artifact["featureCount"] != MACRO_FEATURE_SCHEMA["dimension"]:
        raise ValueError("trained XGBoost model has a feature count inconsistent with ranker v2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=output.suffix,
                                     dir=output.parent, delete=False) as temporary:
        temporary_model = Path(temporary.name)
    try:
        temporary_model.write_text(json.dumps(model_artifact, sort_keys=True, separators=(",", ":")) + "\n")
        with temporary_model.open("rb") as stream:
            os.fsync(stream.fileno())
        model_hash = file_sha256(temporary_model)
        os.link(temporary_model, output)
    finally:
        temporary_model.unlink(missing_ok=True)

    report = {
        "schemaVersion": 1,
        "kind": "xgboost-macro-ranker-v2",
        "parameters": ranker.parameters,
        "iteration": frozen.__dict__,
        "featureSchema": MACRO_FEATURE_SCHEMA,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": source_hash,
        "inferenceImplementationSha256": inference_hash,
        "identity": labels_manifest["identity"],
        "selectionHash": labels_manifest["selectionHash"],
        "inputManifestSha256": file_sha256(labels_dir.resolve() / "manifest.json"),
        "selectionManifestSha256": file_sha256(selection_path.resolve()),
        "modelPath": str(output),
        "modelSha256": model_hash,
        "modelFeatureCount": int(ranker.model.num_features()),
        "trainingPositions": len(groups),
        "trainingCandidates": len(labels),
        "training": training_metrics,
        "development": development_metrics,
        "heldout": {"status": "not-included", "reason": "frozen selection excludes heldout positions"},
        "holdouts": holdouts,
        "acceptance": _ranker_evidence_status(development_metrics, holdouts),
        "automaticPromotion": False,
    }
    report["reportHash"] = identity_hash(report)
    _publish_json(report_path, report)
    return report
