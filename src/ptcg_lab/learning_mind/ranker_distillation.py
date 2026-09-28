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
    rows_path = path / "rows.jsonl"
    if file_sha256(rows_path) != manifest.get("rowsSha256"):
        raise ValueError("ranker distillation rows checksum mismatch")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows) != manifest.get("rows"):
        raise ValueError("ranker distillation row count mismatch")
    for source in manifest.get("sources", []):
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or file_sha256(source_path) != source.get("sha256"):
            raise ValueError("ranker distillation source artifact changed or is unavailable")
    sources_by_kind = {source.get("kind"): Path(source["path"])
                       for source in manifest.get("sources", []) if isinstance(source, dict)}
    if set(sources_by_kind) != {"macro-position-pool-manifest", "combined-label-manifest",
                                "frozen-selection", "ranker-model", "ranker-report"}:
        raise ValueError("ranker distillation source manifest is incomplete")
    verified = verify_macro_ranker_v2_artifact(sources_by_kind["ranker-model"],
                                               sources_by_kind["ranker-report"])
    from .experiment import _load_ranker_input
    labels_manifest, records = _load_ranker_input(
        sources_by_kind["combined-label-manifest"].parent,
        sources_by_kind["frozen-selection"])
    _validate_source_game_units(sources_by_kind["frozen-selection"], records)
    if (verified["report"].get("reportHash") != manifest.get("rankerReportHash")
            or verified["report"].get("modelSha256") != manifest.get("rankerModelSha256")
            or verified["report"].get("identity") != identity
            or verified["report"].get("inputManifestSha256") != file_sha256(
                sources_by_kind["combined-label-manifest"])
            or verified["report"].get("selectionManifestSha256") != file_sha256(
                sources_by_kind["frozen-selection"])
            or verified["report"].get("selectionHash") != labels_manifest.get("selectionHash")
            or labels_manifest.get("identity") != identity):
        raise ValueError("ranker distillation teacher no longer matches its frozen manifest")
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
