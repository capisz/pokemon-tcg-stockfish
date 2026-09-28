from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .dataset_v1 import file_sha256, load_dataset
from .encoding import encode_decision
from .schema import identity_hash
from .supervised_evidence import verify_supervised_audit_report


def _read_verified_audit(*, dataset_dir: Path, checkpoint: Path,
                         evaluation_path: Path, audit_path: Path) -> tuple[dict, dict, list[dict]]:
    audit = verify_supervised_audit_report(dataset_dir=dataset_dir, checkpoint=checkpoint,
        evaluation_path=evaluation_path, audit_path=audit_path)
    manifest, rows = load_dataset(dataset_dir)
    if (audit.get("datasetManifestHash") != manifest.get("manifestHash")
            or audit.get("datasetManifestSha256") != file_sha256(dataset_dir / "manifest.json")):
        raise ValueError("review source dataset differs from the audited dataset")
    if audit.get("checkpointSha256") != file_sha256(checkpoint):
        raise ValueError("review checkpoint differs from the audited checkpoint")
    if audit.get("evaluationSha256") != file_sha256(evaluation_path):
        raise ValueError("review evaluation differs from the audited evaluation")
    if audit.get("evaluationIdentityStatus") != "matched":
        raise ValueError("supervised evidence audit did not verify exact evaluation identity")
    evaluation = json.loads(evaluation_path.read_text())
    expected_positions = {row["positionHash"] for row in rows if row.get("split") != "train"}
    evaluated_positions = [item.get("positionHash") for item in evaluation.get("positions", [])]
    if len(evaluated_positions) != len(set(evaluated_positions)) or set(evaluated_positions) != expected_positions:
        raise ValueError("audited evaluation no longer exactly covers non-training positions")
    source_by_hash = {row["positionHash"]: row for row in rows}
    return audit, evaluation, [source_by_hash[position] for position in evaluated_positions]


def _action_class(observation: dict, tracker: dict, index: int) -> dict:
    encoded = encode_decision(observation, tracker)
    if type(index) is not int or not 0 <= index < len(encoded.action_classes):
        raise ValueError("review action class is outside the frozen encoded options")
    return {"classIndex": index, "kind": "legal-action-class",
            "actions": encoded.action_classes[index].actions}


def _write_immutable_json(output: Path, value: dict) -> None:
    output = output.resolve()
    if output.exists():
        raise ValueError("disagreement review artifacts are immutable; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", delete=False) as temporary:
        temporary.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_disagreement_review_packet(*, dataset_dir: Path, checkpoint: Path,
                                     evaluation_path: Path, audit_path: Path,
                                     output: Path) -> dict:
    """Create a hash-bound, actor-view-only packet for every held-out model/heuristic disagreement."""
    audit, evaluation, source_rows = _read_verified_audit(dataset_dir=dataset_dir,
        checkpoint=checkpoint, evaluation_path=evaluation_path, audit_path=audit_path)
    sources = {row["positionHash"]: row for row in source_rows}
    positions = []
    for item in evaluation["positions"]:
        source = sources[item["positionHash"]]
        if source.get("split") != "heldout" or item.get("modelClass") == item.get("heuristicClass"):
            continue
        observation, tracker = source.get("observation"), source.get("tracker")
        if (not isinstance(observation, dict) or observation.get("playerId") != source.get("actor")
                or not isinstance(tracker, dict)):
            raise ValueError("review packet source is not an actor-visible held-out decision")
        positions.append({
            "positionHash": item["positionHash"],
            "sourceGameId": source.get("sourceGameId"),
            "sourceDecisionIndex": source.get("sourceDecisionIndex"),
            "actor": source["actor"],
            "opponentArchetype": source.get("opponentArchetype"),
            "opponentPolicyFamily": source.get("opponentPolicyFamily"),
            "positionStage": source.get("positionStage"),
            "modelAction": _action_class(observation, tracker, item["modelClass"]),
            "heuristicAction": _action_class(observation, tracker, item["heuristicClass"]),
            "acceptableActionIndices": source.get("acceptableActionIndices"),
            "policyDistribution": source.get("policyDistribution"),
            "modelHit": item.get("modelHit"),
            "heuristicHit": item.get("heuristicHit"),
            "actorObservation": observation,
        })
    packet = {"schemaVersion": 1, "kind": "supervised-disagreement-review-packet-v1",
        "datasetManifestHash": audit["datasetManifestHash"],
        "checkpointSha256": audit["checkpointSha256"],
        "evaluationSha256": audit["evaluationSha256"],
        "supervisedAuditHash": audit["reportHash"],
        "heldoutDisagreementCount": len(positions),
        "reviewStatus": "pending" if positions else "none-to-review",
        "positions": positions,
        "note": "Actor-visible decisions only. Review does not modify labels or enable training."}
    packet["packetHash"] = identity_hash(packet)
    _write_immutable_json(output, packet)
    return packet


def write_disagreement_review_template(*, packet_path: Path, output: Path) -> dict:
    packet = json.loads(packet_path.read_text())
    recorded_hash = packet.get("packetHash")
    if recorded_hash != identity_hash({key: value for key, value in packet.items() if key != "packetHash"}):
        raise ValueError("disagreement review packet hash mismatch")
    template = {"schemaVersion": 1, "packetSha256": file_sha256(packet_path), "reviewer": "",
        "reviews": [{"positionHash": item["positionHash"], "finding": None, "rationale": ""}
                    for item in packet["positions"]]}
    _write_immutable_json(output, template)
    return template


def audit_disagreement_review(*, packet_path: Path, review_path: Path, output: Path) -> dict:
    packet = json.loads(packet_path.read_text())
    if packet.get("packetHash") != identity_hash({key: value for key, value in packet.items()
                                                    if key != "packetHash"}):
        raise ValueError("disagreement review packet hash mismatch")
    review = json.loads(review_path.read_text())
    if review.get("schemaVersion") != 1 or review.get("packetSha256") != file_sha256(packet_path):
        raise ValueError("human review is not bound to this exact packet")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("human review must identify its reviewer")
    expected = {item["positionHash"] for item in packet["positions"]}
    rows = review.get("reviews")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("human review must contain exactly one decision per disagreement")
    observed: dict[str, dict] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("human review contains an invalid decision")
        position = item.get("positionHash")
        finding = item.get("finding")
        rationale = item.get("rationale")
        if (not isinstance(position, str) or position not in expected or position in observed
                or not isinstance(finding, str)
                or finding not in {"acceptable", "concern", "needs-follow-up"}
                or not isinstance(rationale, str) or not rationale.strip()):
            raise ValueError("human review is incomplete, duplicated, or has an invalid finding")
        observed[position] = item
    unresolved = [position for position, item in observed.items()
                  if item["finding"] != "acceptable"]
    receipt = {"schemaVersion": 1, "kind": "supervised-disagreement-review-receipt-v1",
        "packetSha256": file_sha256(packet_path), "reviewSha256": file_sha256(review_path),
        "reviewer": reviewer.strip(), "reviewedPositions": len(observed),
        "unresolvedPositions": sorted(unresolved),
        "representativeDisagreementsReviewed": bool(observed) and not unresolved,
        "automaticPromotion": False}
    receipt["receiptHash"] = identity_hash(receipt)
    _write_immutable_json(output, receipt)
    return receipt
