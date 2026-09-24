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
from .sampling import select_stratified_rows
from .schema import IdentityError, UnsupportedPosition, identity_hash


def audit_macro_candidate_support(*, root: Path, dataset_dir: Path, output: Path,
                                  identity: dict, workers: int = 1,
                                  limit: int | None = None,
                                  split: str | None = None,
                                  position_hashes: list[str] | None = None) -> dict:
    """Audit executable candidate support only; never runs rollout search."""
    if position_hashes == []:
        position_hashes = None
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("support audit workers must be an integer from 1 to 8")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("support audit limit must be a positive integer")
    if split is not None and split not in {"train", "development", "heldout"}:
        raise ValueError("support audit split must be train, development, or heldout")
    if position_hashes is not None and limit is not None:
        raise ValueError("support audit accepts either position_hashes or limit, not both")
    if output.exists():
        raise ValueError("candidate support audit outputs are immutable; choose a new directory")
    root, dataset_dir, output = root.resolve(), dataset_dir.resolve(), output.resolve()
    manifest, rows = load_dataset(dataset_dir, identity=identity)
    candidates = [row for row in rows if split is None or row.get("split") == split]
    if not candidates:
        raise ValueError(f"frozen dataset has no positions in the requested {split} split")
    if position_hashes is not None:
        if not position_hashes or any(not isinstance(value, str) or not value for value in position_hashes):
            raise ValueError("support audit position_hashes must contain nonempty position hashes")
        if len(position_hashes) != len(set(position_hashes)):
            raise ValueError("support audit position_hashes must not contain duplicates")
        by_hash = {row["positionHash"]: row for row in rows}
        missing = [key for key in position_hashes if key not in by_hash]
        if missing:
            raise ValueError(f"support audit positions are not present in the frozen dataset: {missing}")
        selected = [by_hash[key] for key in position_hashes]
        outside = [row["positionHash"] for row in selected if split is not None and row.get("split") != split]
        if outside:
            raise ValueError(f"support audit positions are not in the requested {split} split: {outside}")
    else:
        selected = select_stratified_rows(candidates, limit) if limit is not None else candidates
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
            "positionHash", "sourceGameId", "split", "targetDeck", "opponentArchetype",
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
    for dimension in ("targetDeck", "opponentArchetype", "opponentPolicyFamily", "positionStage", "split"):
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
        "splitFilter": split,
        "selectedPositionHashes": [row["positionHash"] for row in selected],
        "sampleSelection": "position-hash-list" if position_hashes is not None else
            "deterministic-stratified-v2-target-deck" if limit is not None else "all-rows",
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
