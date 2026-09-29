"""Create actor-visible policy targets from a frozen, verified macro ranker."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
from collections import Counter

import numpy as np

from .dataset_v1 import (file_sha256, load_dataset, _atomic_text,
                         _validate_supervised_rows)
from .encoding import encode_decision
from .ranker_features import candidate_features_v2
from .ranker_portable import predict_macro_ranker_v2
from .ranker_v2 import (_validate_source_game_units, _validated_macro_labels,
                        verify_macro_ranker_v2_artifact)
from .schema import identity_hash


def _validate_metric_position_coverage(metrics: dict, expected: dict[str, tuple[dict, int]], *,
                                      name: str) -> None:
    details = metrics.get("details") if isinstance(metrics, dict) else None
    detail_hashes = ([detail.get("positionHash") for detail in details]
                     if isinstance(details, list) and all(isinstance(detail, dict)
                                                          for detail in details) else [])
    if len(detail_hashes) != len(expected) or set(detail_hashes) != set(expected):
        raise ValueError(f"ranker {name} metrics do not cover the frozen positions")
    for detail in details:
        source, candidate_count = expected[detail["positionHash"]]
        if (type(detail.get("candidates")) is not int or detail["candidates"] != candidate_count
                or any(detail.get(field) != source.get(field) for field in (
                    "sourceGameId", "split", "opponentArchetype", "opponentPolicyFamily"))):
            raise ValueError(f"ranker {name} metric provenance or candidate count differs from its frozen position")


def _validate_ranker_holdout_coverage(report: dict, records: list[dict],
                                      rollout_ids: dict[str, str]) -> None:
    from .experiment import _ranker_holdout_rows

    expected = _ranker_holdout_rows(records)
    supplied = report.get("holdouts")
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        raise ValueError("ranker report does not exactly cover the frozen holdout set")
    expected_by_key = {(row["kind"], row["heldOut"]): row for row in expected}
    seen = set()
    for row in supplied:
        if not isinstance(row, dict):
            raise ValueError("ranker report contains a malformed holdout row")
        key = (row.get("kind"), row.get("heldOut"))
        baseline = expected_by_key.get(key)
        if (baseline is None or key in seen or row.get("train") != baseline["train"]
                or row.get("test") != baseline["test"] or row.get("status") != "measured"
                or not isinstance(row.get("metrics"), dict)
                or row["metrics"].get("status") != "measured"):
            raise ValueError("ranker report holdout coverage or partition differs from frozen records")
        expected = {}
        for index in baseline["test"]:
            source = records[index]
            labels = _validated_macro_labels(source,
                expected_rollout_identity=rollout_ids[source["opponentPolicyFamily"]])
            if len(labels) >= 2:
                expected[source["positionHash"]] = (source, len(labels))
        _validate_metric_position_coverage(row["metrics"], expected,
                                           name="holdout evaluation")
        seen.add(key)
    if seen != set(expected_by_key):
        raise ValueError("ranker report does not exactly cover the frozen holdout set")


def _validate_ranker_metric_coverage(report: dict, records: list[dict],
                                     rollout_ids: dict[str, str]) -> None:
    expected_by_split = {split: {} for split in ("train", "development")}
    for record in records:
        labels = _validated_macro_labels(record,
            expected_rollout_identity=rollout_ids[record["opponentPolicyFamily"]])
        if len(labels) >= 2:
            expected_by_split[record["split"]][record["positionHash"]] = (record, len(labels))
    _validate_metric_position_coverage(report.get("training", {}), expected_by_split["train"],
                                       name="training")
    _validate_metric_position_coverage(report.get("development", {}), expected_by_split["development"],
                                       name="development")
    _validate_ranker_holdout_coverage(report, records, rollout_ids)


def _distribution(scores: np.ndarray, candidates: list[dict], observation: dict,
                   encoded, *, temperature: float) -> list[float]:
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or len(scores) != len(candidates) or not len(scores)
            or not np.isfinite(scores).all() or type(temperature) not in {int, float}
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("ranker distillation scores or temperature are invalid")
    actions = {str(action.get("id")): action for action in observation.get("legalActions", [])
               if isinstance(action, dict)}
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = (scores - float(scores.max())) / temperature
    probabilities = np.exp(np.clip(scaled, -745.0, 0.0))
    probabilities /= probabilities.sum()
    result = [0.0] * (len(encoded.action_classes) + 1)
    for candidate, probability in zip(candidates, probabilities):
        sequence = candidate.get("action_sequence")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("ranker candidate has no executable root action")
        root = sequence[0]
        action_id = str(root.get("id"))
        if actions.get(action_id) != root:
            raise ValueError("ranker candidate root action is not actor-visible and legal")
        matches = [index for index, group in enumerate(encoded.action_classes)
                   if any(str(action.get("id")) == action_id for action in group.actions)]
        if len(matches) != 1:
            raise ValueError("ranker candidate root action does not map to exactly one semantic option")
        result[matches[0]] += float(probability)
    total = sum(result)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("ranker distillation produced an empty action distribution")
    result = [value / total for value in result]
    # The final slot is STOP; a macro plan's first executable action is never STOP.
    result[-1] = 0.0
    return result


def build_macro_ranker_distillation(*, output: Path, macro_position_pool: Path,
        labels_dir: Path, selection_path: Path, model_path: Path, report_path: Path,
        identity: dict, temperature: float = 1.0) -> dict:
    """Freeze train-only ranker-to-legal-action distributions as an immutable dataset."""
    output = output.resolve()
    if output.exists():
        raise ValueError("ranker distillation outputs are immutable; choose a new directory")
    if (type(temperature) not in {int, float} or not math.isfinite(temperature) or temperature <= 0
            or not isinstance(identity, dict) or not identity):
        raise ValueError("ranker distillation identity or temperature is invalid")

    from .experiment import _load_ranker_input

    verified = verify_macro_ranker_v2_artifact(model_path, report_path)
    report, artifact = verified["report"], verified["artifact"]
    if (report.get("identity") != identity or report.get("acceptance") != "review-required"
            or report.get("automaticPromotion") is not False
            or report.get("development", {}).get("status") != "measured"):
        raise ValueError("ranker report is not a measured, non-promoting research teacher")
    holdouts = report.get("holdouts")
    required_holdouts = {"leave-one-opponent-archetype-out", "frozen-policy-family"}
    if (not isinstance(holdouts, list)
            or not required_holdouts.issubset({item.get("kind") for item in holdouts
                                               if isinstance(item, dict) and item.get("status") == "measured"})):
        raise ValueError("ranker distillation requires measured archetype and policy-family holdouts")

    pool_manifest, pool_rows = load_dataset(macro_position_pool, identity=identity)
    if pool_manifest.get("id") != "learning-mind-macro-position-pool-v1":
        raise ValueError("ranker distillation requires the frozen macro position pool")
    labels_dir = labels_dir.resolve()
    selection_path = selection_path.resolve()
    labels_manifest, records = _load_ranker_input(labels_dir, selection_path)
    if (report.get("inputManifestSha256") != file_sha256(labels_dir / "manifest.json")
            or report.get("selectionManifestSha256") != file_sha256(selection_path)
            or report.get("selectionHash") != labels_manifest.get("selectionHash")
            or labels_manifest.get("identity") != identity):
        raise ValueError("ranker distillation source manifests differ from the verified teacher")
    _validate_source_game_units(selection_path, records)
    source_runs = labels_manifest.get("sourceRuns")
    rollout_ids = {}
    if isinstance(source_runs, list):
        for run in source_runs:
            families, rollout_id = run.get("policyFamilies"), run.get("rolloutIdentity")
            if (not isinstance(families, list) or len(families) != 1
                    or families[0] in rollout_ids or not isinstance(rollout_id, str) or not rollout_id):
                raise ValueError("ranker distillation source rollout identity is ambiguous")
            rollout_ids[families[0]] = rollout_id
    if set(rollout_ids) != {"python-heuristic", "typescript-heuristic"}:
        raise ValueError("ranker distillation source families are incomplete")

    _validate_ranker_metric_coverage(report, records, rollout_ids)

    expected_training_positions = {record["positionHash"] for record in records
        if record.get("split") == "train" and len(_validated_macro_labels(record,
            expected_rollout_identity=rollout_ids[record["opponentPolicyFamily"]])) >= 2}
    iteration = report.get("iteration")
    if (report.get("inputManifestHash") != labels_manifest.get("manifestHash")
            or not isinstance(iteration, dict)
            or iteration.get("input_hash") != labels_manifest.get("manifestHash")
            or set(iteration.get("position_hashes", [])) != expected_training_positions):
        raise ValueError("ranker teacher iteration does not match the frozen label manifest and train positions")

    incomplete = {split: [] for split in ("train", "development")}
    for record in records:
        split = record.get("split")
        if split not in incomplete:
            raise ValueError("ranker distillation input contains a non-train/development position")
        usable = _validated_macro_labels(record,
            expected_rollout_identity=rollout_ids[record["opponentPolicyFamily"]])
        if len(usable) < 2:
            incomplete[split].append(record["positionHash"])
    for split in ("train", "development"):
        metrics = report.get("training" if split == "train" else "development")
        if (incomplete[split] or not isinstance(metrics, dict)
                or metrics.get("status") != "measured"
                or metrics.get("requestedPositions") != sum(row.get("split") == split for row in records)
                or metrics.get("positions") != metrics.get("requestedPositions")
                or metrics.get("insufficientPositionHashes") != []):
            raise ValueError(f"ranker distillation requires complete, measured {split} label coverage")

    pool_by_hash = {}
    for row in pool_rows:
        if row.get("positionHash") in pool_by_hash:
            raise ValueError("macro position pool repeats an actor-visible position")
        pool_by_hash[row.get("positionHash")] = row
    output_rows, skipped = [], []
    for record in records:
        if record.get("split") != "train":
            continue
        position_hash = record.get("positionHash")
        source = pool_by_hash.get(position_hash)
        if (source is None or source.get("split") != "train"
                or source.get("observation") != record.get("observation")
                or source.get("familyId") != record.get("familyId")):
            raise ValueError("ranker training position differs from its frozen actor-visible pool row")
        labels = _validated_macro_labels(record,
            expected_rollout_identity=rollout_ids[record["opponentPolicyFamily"]])
        if len(labels) < 2:
            skipped.append({"positionHash": position_hash, "reason": "fewer-than-two-completed-candidates"})
            continue
        tracker = source.get("tracker")
        observation = source.get("observation")
        if (not isinstance(observation, dict) or observation.get("playerId") != source.get("actor")
                or observation.get("decisionPlayer", source.get("actor")) != source.get("actor")):
            raise ValueError("ranker pool row is not an actor-visible observation")
        encoded = encode_decision(observation, tracker)
        if encoded.identity != source.get("featureIdentityHash"):
            raise ValueError("ranker pool feature identity mismatch")
        candidates = [label["candidate"] for label in labels]
        features = np.stack([candidate_features_v2(observation, candidate) for candidate in candidates])
        scores = predict_macro_ranker_v2(artifact, features)
        target = _distribution(scores, candidates, observation, encoded, temperature=temperature)
        if record.get("sourceGameId") != source.get("sourceGameId"):
            raise ValueError("ranker position disagrees with its frozen source game")
        provenance = {"modelSha256": report["modelSha256"],
            "reportHash": report["reportHash"],
            "inputManifestSha256": report["inputManifestSha256"],
            "selectionManifestSha256": report["selectionManifestSha256"],
            "candidateSetHash": identity_hash(sorted(label["candidateHash"] for label in labels)),
            "candidateCount": len(labels), "temperature": temperature}
        output_rows.append({key: source.get(key) for key in (
            "positionHash", "familyId", "sourceGameId", "sourceDecisionIndex", "actor", "deckHash",
            "opponentArchetype", "opponentPolicyFamily", "featureIdentityHash", "split", "observation", "tracker")}
            | {"policyLabelSource": "macro-ranker-distillation", "acceptableActionIndices": None,
               "policyDistribution": target, "rankerDistillation": provenance})

    if not output_rows:
        raise ValueError("ranker distillation has no train positions with two completed candidates")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        rows_path = temporary_root / "rows.jsonl"
        _atomic_text(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                          for row in output_rows))
        source_files = [("macro-position-pool-manifest", macro_position_pool / "manifest.json"),
            ("combined-label-manifest", labels_dir / "manifest.json"),
            ("frozen-selection", selection_path), ("ranker-model", model_path.resolve()),
            ("ranker-report", report_path.resolve())]
        sources = [{"kind": kind, "path": str(path.resolve()), "sha256": file_sha256(path)}
                   for kind, path in source_files]
        manifest = {"schemaVersion": 1, "kind": "macro-ranker-distillation-v1",
            "identity": identity, "teacherAcceptance": report["acceptance"],
            "rankerReportHash": report["reportHash"], "rankerModelSha256": report["modelSha256"],
            "temperature": temperature, "rows": len(output_rows), "rowsSha256": file_sha256(rows_path),
            "skipped": skipped, "sources": sources, "policySplit": "train-only"}
        manifest["manifestHash"] = identity_hash(manifest)
        _atomic_text(temporary_root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        temporary_root.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def load_macro_ranker_distillation(path: Path, *, identity: dict) -> tuple[dict, list[dict]]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("kind") != "macro-ranker-distillation-v1"
            or manifest.get("identity") != identity
            or manifest.get("policySplit") != "train-only"
            or manifest.get("manifestHash") != identity_hash({key: value for key, value in manifest.items()
                                                                if key != "manifestHash"})):
        raise ValueError("ranker distillation manifest identity or checksum mismatch")
    if manifest.get("teacherAcceptance") != "review-required":
        raise ValueError("ranker distillation teacher is not marked review-required")
    required_source_kinds = {"macro-position-pool-manifest", "combined-label-manifest",
        "frozen-selection", "ranker-model", "ranker-report"}
    sources = manifest.get("sources")
    if (not isinstance(sources, list) or len(sources) != len(required_source_kinds)
            or any(not isinstance(source, dict) or set(source) != {"kind", "path", "sha256"}
                   or not isinstance(source.get("kind"), str)
                   or not isinstance(source.get("path"), str)
                   or not Path(source["path"]).is_absolute()
                   or not isinstance(source.get("sha256"), str)
                   for source in sources)
            or {source["kind"] for source in sources} != required_source_kinds
            or len({source["kind"] for source in sources}) != len(sources)):
        raise ValueError("ranker distillation source artifact list is incomplete or ambiguous")
    temperature = manifest.get("temperature")
    if (type(temperature) not in {int, float} or not math.isfinite(temperature) or temperature <= 0
            or type(manifest.get("rows")) is not int or manifest["rows"] < 1
            or manifest.get("skipped") != []):
        raise ValueError("ranker distillation temperature, row count, or coverage record is invalid")
    rows_path = path / "rows.jsonl"
    if file_sha256(rows_path) != manifest.get("rowsSha256"):
        raise ValueError("ranker distillation rows checksum mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows) != manifest.get("rows"):
        raise ValueError("ranker distillation row count mismatch")
    for source in sources:
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or file_sha256(source_path) != source.get("sha256"):
            raise ValueError("ranker distillation source artifact changed or is unavailable")
    sources_by_kind = {source.get("kind"): Path(source["path"])
                       for source in manifest.get("sources", []) if isinstance(source, dict)}
    verified = verify_macro_ranker_v2_artifact(sources_by_kind["ranker-model"],
                                               sources_by_kind["ranker-report"])
    from .experiment import _load_ranker_input
    labels_manifest, records = _load_ranker_input(
        sources_by_kind["combined-label-manifest"].parent,
        sources_by_kind["frozen-selection"])
    _validate_source_game_units(sources_by_kind["frozen-selection"], records)
    if (verified["report"].get("reportHash") != manifest.get("rankerReportHash")
            or verified["report"].get("modelSha256") != manifest.get("rankerModelSha256")
            or verified["report"].get("acceptance") != manifest.get("teacherAcceptance")
            or verified["report"].get("automaticPromotion") is not False
            or verified["report"].get("identity") != identity
            or verified["report"].get("inputManifestSha256") != file_sha256(
                sources_by_kind["combined-label-manifest"])
            or verified["report"].get("selectionManifestSha256") != file_sha256(
                sources_by_kind["frozen-selection"])
            or verified["report"].get("selectionHash") != labels_manifest.get("selectionHash")
            or labels_manifest.get("identity") != identity):
        raise ValueError("ranker distillation teacher no longer matches its frozen manifest")
    pool_manifest_path = sources_by_kind["macro-position-pool-manifest"]
    pool_manifest, pool_rows = load_dataset(pool_manifest_path.parent, identity=identity)
    if pool_manifest.get("id") != "learning-mind-macro-position-pool-v1":
        raise ValueError("ranker distillation source is not the frozen macro position pool")
    pool_by_hash = {row.get("positionHash"): row for row in pool_rows}
    if len(pool_by_hash) != len(pool_rows):
        raise ValueError("frozen macro position pool repeats a position")

    rollout_ids = {}
    for source_run in labels_manifest.get("sourceRuns", []):
        families = source_run.get("policyFamilies") if isinstance(source_run, dict) else None
        rollout_id = source_run.get("rolloutIdentity") if isinstance(source_run, dict) else None
        if (not isinstance(families, list) or len(families) != 1
                or families[0] not in {"python-heuristic", "typescript-heuristic"}
                or families[0] in rollout_ids or not isinstance(rollout_id, str) or not rollout_id):
            raise ValueError("ranker distillation source-run identity is ambiguous")
        rollout_ids[families[0]] = rollout_id
    if set(rollout_ids) != {"python-heuristic", "typescript-heuristic"}:
        raise ValueError("ranker distillation source-run families are incomplete")
    _validate_ranker_metric_coverage(verified["report"], records, rollout_ids)

    expected_training_positions = {record["positionHash"] for record in records
        if record.get("split") == "train" and len(_validated_macro_labels(record,
            expected_rollout_identity=rollout_ids[record["opponentPolicyFamily"]])) >= 2}
    iteration = verified["report"].get("iteration")
    if (verified["report"].get("inputManifestHash") != labels_manifest.get("manifestHash")
            or not isinstance(iteration, dict)
            or iteration.get("input_hash") != labels_manifest.get("manifestHash")
            or set(iteration.get("position_hashes", [])) != expected_training_positions):
        raise ValueError("ranker teacher iteration does not match the frozen label manifest and train positions")

    rows_by_hash = {row.get("positionHash"): row for row in rows}
    if len(rows_by_hash) != len(rows):
        raise ValueError("ranker distillation repeats a position")
    train_records = [record for record in records if record.get("split") == "train"]
    if set(rows_by_hash) != {record.get("positionHash") for record in train_records}:
        raise ValueError("ranker distillation rows do not exactly cover frozen training positions")
    source_fields = ("positionHash", "familyId", "sourceGameId", "sourceDecisionIndex", "actor",
        "deckHash", "opponentArchetype", "opponentPolicyFamily", "featureIdentityHash",
        "split", "observation", "tracker")
    temperature = manifest.get("temperature")
    for record in train_records:
        position_hash = record["positionHash"]
        row = rows_by_hash[position_hash]
        source = pool_by_hash.get(position_hash)
        if (source is None or source.get("split") != "train"
                or source.get("observation") != record.get("observation")
                or source.get("familyId") != record.get("familyId")
                or source.get("sourceGameId") != record.get("sourceGameId")
                or any(row.get(field) != source.get(field) for field in source_fields)):
            raise ValueError("ranker distillation row differs from its frozen actor-visible pool position")
        labels = _validated_macro_labels(record,
            expected_rollout_identity=rollout_ids.get(record.get("opponentPolicyFamily")))
        if len(labels) < 2:
            raise ValueError("ranker distillation training position lacks two completed candidates")
        encoded = encode_decision(source["observation"], source["tracker"])
        if encoded.identity != source.get("featureIdentityHash"):
            raise ValueError("ranker distillation source feature identity mismatch")
        candidates = [label["candidate"] for label in labels]
        features = np.stack([candidate_features_v2(source["observation"], candidate)
                             for candidate in candidates])
        scores = predict_macro_ranker_v2(verified["artifact"], features)
        expected_distribution = _distribution(scores, candidates, source["observation"], encoded,
                                              temperature=temperature)
        actual_distribution = row.get("policyDistribution")
        if (not isinstance(actual_distribution, list)
                or len(actual_distribution) != len(expected_distribution)
                or not np.allclose(actual_distribution, expected_distribution, rtol=1e-6, atol=1e-8)):
            raise ValueError("ranker distillation action distribution does not reproduce from its frozen teacher")
        expected_provenance = {"modelSha256": verified["report"]["modelSha256"],
            "reportHash": verified["report"]["reportHash"],
            "inputManifestSha256": verified["report"]["inputManifestSha256"],
            "selectionManifestSha256": verified["report"]["selectionManifestSha256"],
            "candidateSetHash": identity_hash(sorted(label["candidateHash"] for label in labels)),
            "candidateCount": len(labels), "temperature": temperature}
        if (row.get("policyLabelSource") != "macro-ranker-distillation"
                or row.get("acceptableActionIndices") is not None
                or row.get("rankerDistillation") != expected_provenance):
            raise ValueError("ranker distillation row provenance does not reproduce from its teacher inputs")
    if any(row.get("rankerDistillation", {}).get("modelSha256") != verified["report"].get("modelSha256")
           or row.get("rankerDistillation", {}).get("reportHash") != verified["report"].get("reportHash")
           or row.get("rankerDistillation", {}).get("inputManifestSha256") !=
              verified["report"].get("inputManifestSha256")
           or row.get("rankerDistillation", {}).get("selectionManifestSha256") !=
              verified["report"].get("selectionManifestSha256")
           or row.get("rankerDistillation", {}).get("temperature") != manifest.get("temperature")
           or row.get("split") != "train" for row in rows):
        raise ValueError("ranker distillation row provenance differs from its teacher manifest")
    counts = Counter((row.get("split"), row.get("policyLabelSource")) for row in rows)
    validation_manifest = {"counts": [{"split": split, "source": source, "count": count}
        for (split, source), count in sorted(counts.items())],
        "families": sorted({row.get("familyId") for row in rows}),
        "teacherHashes": sorted({value for row in rows
            for value in (row["rankerDistillation"]["modelSha256"],
                          row["rankerDistillation"]["reportHash"])}),
        "ordinarySelfPlayPolicyLabels": 0}
    _validate_supervised_rows(validation_manifest, rows)
    return manifest, rows
