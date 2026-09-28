from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .dataset_v1 import file_sha256
from .experiment import _load_ranker_input, _ranker_evidence_status, _ranker_holdout_rows
from .ranker import FrozenIteration, XGBoostMacroRanker
from .ranker_features import (MACRO_FEATURE_SCHEMA, MACRO_FEATURE_SCHEMA_HASH,
                              candidate_features_v2)
from .schema import identity_hash


def validate_ranker_v2_report(report: dict) -> None:
    if not isinstance(report, dict) or report.get("kind") != "xgboost-macro-ranker-v2":
        raise ValueError("not a macro ranker v2 report")
    if (report.get("featureSchema") != MACRO_FEATURE_SCHEMA
            or report.get("featureSchemaHash") != MACRO_FEATURE_SCHEMA_HASH):
        raise ValueError("macro ranker v2 feature schema mismatch")
    current_source_hash = file_sha256(Path(__file__).with_name("ranker_features.py"))
    if report.get("featureImplementationSha256") != current_source_hash:
        raise ValueError("macro ranker v2 feature implementation mismatch")
    if report.get("reportHash") != identity_hash({key: value for key, value in report.items()
                                                   if key != "reportHash"}):
        raise ValueError("macro ranker v2 report hash mismatch")


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
    train_records = [record for record in records if record["split"] == "train"]
    development_records = [record for record in records if record["split"] == "development"]
    if not train_records:
        raise ValueError("macro ranker v2 has no training positions")

    def flatten(selected):
        features, labels, weights, groups, positions, metadata = [], [], [], [], [], []
        for record in selected:
            usable = [item for item in record["labels"] if item.get("expectedResult") is not None]
            if len(usable) < 2:
                continue
            groups.append(len(usable))
            positions.append(record["positionHash"])
            metadata.append({key: record[key] for key in
                ("positionHash", "split", "opponentArchetype", "opponentPolicyFamily")})
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
        if not sizes:
            return {"status": "insufficient", "positions": 0}
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
        by_archetype, by_family = {}, {}
        for field, destination in (("opponentArchetype", by_archetype),
                                   ("opponentPolicyFamily", by_family)):
            for value in sorted({row[field] for row in details}):
                subset = [row for row in details if row[field] == value]
                destination[value] = {"positions": len(subset),
                    "meanTop1RelativeRegret": float(np.mean([row["top1RelativeRegret"] for row in subset]))}
        return {"status": "measured", "positions": len(details),
            "meanTop1RelativeRegret": float(np.mean([row["top1RelativeRegret"] for row in details])),
            "top3Recall": sum(row["top3Recall"] for row in details) / len(details),
            "pairwiseOrderingAccuracy": (sum(row["pairwiseCorrect"] for row in details) / pair_count
                                         if pair_count else None),
            "byOpponentArchetype": by_archetype,
            "byOpponentPolicyFamily": by_family,
            "details": details}

    input_hash = labels_manifest["manifestHash"]
    source_hash = file_sha256(Path(__file__).with_name("ranker_features.py"))
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
        holdouts.append({**split, "status": "measured", "metrics": metrics(held_model, held_test)})

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=output.suffix,
                                     dir=output.parent, delete=False) as temporary:
        temporary_model = Path(temporary.name)
    try:
        ranker.model.save_model(temporary_model)
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
        "identity": labels_manifest["identity"],
        "selectionHash": labels_manifest["selectionHash"],
        "inputManifestSha256": file_sha256(labels_dir.resolve() / "manifest.json"),
        "selectionManifestSha256": file_sha256(selection_path.resolve()),
        "modelPath": str(output),
        "modelSha256": model_hash,
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
