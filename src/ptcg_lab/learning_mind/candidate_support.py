from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from .dataset_v1 import file_sha256, load_dataset
from .encoding import encode_decision
from .experiment import generate_transition_candidates, transition_generator_identity
from .schema import IdentityError, UnsupportedPosition, identity_hash


def _select_stratified_sample(rows: list[dict], limit: int) -> list[dict]:
    """Deterministically sample broadly across archetype, stage, and split."""
    remaining = list(rows)
    selected: list[dict] = []
    counts: dict[str, Counter] = {
        name: Counter() for name in (
            "opponentArchetype", "positionStage", "split", "sourceGameId",
        )
    }
    while remaining and len(selected) < limit:
        row = min(remaining, key=lambda item: (
            counts["opponentArchetype"][str(item.get("opponentArchetype", "unknown"))],
            counts["positionStage"][str(item.get("positionStage", "unknown"))],
            counts["split"][str(item.get("split", "unknown"))],
            counts["sourceGameId"][str(item.get("sourceGameId", "unknown"))],
            str(item.get("positionHash", "")),
        ))
        selected.append(row)
        remaining.remove(row)
        for name, counter in counts.items():
            counter[str(row.get(name, "unknown"))] += 1
    return selected


def audit_macro_candidate_support(*, root: Path, dataset_dir: Path, output: Path,
                                  identity: dict, workers: int = 1,
                                  limit: int | None = None) -> dict:
    """Audit executable candidate support only; never runs rollout search."""
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("support audit workers must be an integer from 1 to 8")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("support audit limit must be a positive integer")
    if output.exists():
        raise ValueError("candidate support audit outputs are immutable; choose a new directory")
    root, dataset_dir, output = root.resolve(), dataset_dir.resolve(), output.resolve()
    manifest, rows = load_dataset(dataset_dir, identity=identity)
    selected = _select_stratified_sample(rows, limit) if limit is not None else rows
    generator_identity = transition_generator_identity(root)

    def inspect(row: dict) -> dict:
        observation = row.get("observation")
        tracker = row.get("tracker")
        if not isinstance(observation, dict) or not isinstance(tracker, dict):
            raise ValueError("support audit row lacks actor-visible observation/tracker")
        if observation.get("playerId") != row.get("actor"):
            raise ValueError("support audit row actor differs from observation playerId")
        encoded = encode_decision(observation, tracker)
        if encoded.identity != row.get("featureIdentityHash"):
            raise IdentityError(f"feature identity drift for position {row.get('positionHash')}")
        item = {key: row.get(key) for key in (
            "positionHash", "sourceGameId", "split", "opponentArchetype",
            "opponentPolicyFamily", "positionStage")}
        item["featureIdentityHash"] = encoded.identity
        item["legalActionClassCount"] = len(encoded.action_classes)
        try:
            candidates, generation = generate_transition_candidates(
                root, observation, int.from_bytes(
                    hashlib.sha256(
                        f"learning-mind-v1|macro-generator|{row['positionHash']}".encode()
                    ).digest()[:4], "big"))
        except UnsupportedPosition as error:
            item.update({"status": "unsupported", "reason": str(error)})
            return item
        complete = [candidate for candidate in candidates
                    if candidate.action_sequence and candidate.turn_intent in {"attack", "no-attack"}]
        item.update({
            "status": "supported" if complete else "no-complete-candidate",
            "candidateCount": len(candidates),
            "completeCandidateCount": len(complete),
            "completeDepthCounts": dict(sorted(Counter(
                str(len(candidate.action_sequence)) for candidate in complete).items())),
            "intentCounts": dict(sorted(Counter(candidate.turn_intent for candidate in complete).items())),
            "hypothesisId": generation.get("hypothesisId"),
        })
        return item

    if workers == 1:
        positions = [inspect(row) for row in selected]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            positions = list(executor.map(inspect, selected))
    statuses = Counter(item["status"] for item in positions)
    by_dimension: dict[str, dict[str, dict[str, int]]] = {}
    for dimension in ("opponentArchetype", "opponentPolicyFamily", "positionStage", "split"):
        values: dict[str, Counter] = defaultdict(Counter)
        for item in positions:
            values[str(item.get(dimension, "unknown"))][item["status"]] += 1
        by_dimension[dimension] = {key: dict(sorted(counter.items()))
                                  for key, counter in sorted(values.items())}
    report = {
        "schemaVersion": 1,
        "audit": "actor-visible-macro-candidate-support-only",
        "status": "no-rollouts-no-labels",
        "identity": identity,
        "datasetManifestHash": manifest["manifestHash"],
        "datasetManifestFileSha256": file_sha256(dataset_dir / "manifest.json"),
        "datasetRowsSha256": file_sha256(dataset_dir / "rows.jsonl"),
        "candidateGeneratorIdentity": generator_identity,
        "workers": workers,
        "requestedLimit": limit,
        "sampleSelection": "deterministic-stratified-v1" if limit is not None else "all-rows",
        "rows": len(selected),
        "statusCounts": dict(sorted(statuses.items())),
        "supportedPositions": statuses["supported"],
        "unsupportedPositions": statuses["unsupported"],
        "noCompleteCandidatePositions": statuses["no-complete-candidate"],
        "completeCandidates": sum(item.get("completeCandidateCount", 0) for item in positions),
        "byDimension": by_dimension,
        "positions": positions,
    }
    report["reportHash"] = identity_hash(report)
    output.mkdir(parents=True)
    report_path = output / "report.json"
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(report_path)
    return report
