"""Human-reviewed teaching records; prose alone can never become a demonstration."""
from __future__ import annotations

import json
import copy
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .storage import digest


def review_hash(record: dict) -> str:
    """Bind the human annotation to its exact information set and audit receipt."""
    fields = ("familyId", "partition", "positionHash", "engineVersion", "source", "rulesAuditStatus",
              "mechanicsAudit", "fixtureReceiptHash", "acceptableActionIds", "rejectedActionIds",
              "conditionalReasoning", "reviewStatus", "reviewedAt")
    return digest({key: record.get(key) for key in fields})


def valid_review(record: dict, *, partition: str = "train") -> bool:
    """Validate the unchanged human review, independently of root audit coverage."""
    legal = {action["id"] for action in record.get("observation", {}).get("legalActions", [])}
    accepted = record.get("acceptableActionIds", [])
    return bool(record.get("reviewStatus") == "reviewed" and record.get("rulesValidation") == "legal-position"
                and not record.get("superseded")
                and record.get("rulesAuditStatus") == "verified"
                and record.get("partition") == partition and record.get("positionHash") == digest(record.get("observation"))
                and accepted and set(accepted) <= legal and record.get("conditionalReasoning", "").strip()
                and record.get("reviewHash") == review_hash(record))


def eligible(record: dict, *, partition: str = "train") -> bool:
    return bool(valid_review(record, partition=partition)
                and (not record.get("fixtureReceiptHash")
                     or set(record["acceptableActionIds"]) <= set(record.get("mechanicsAudit", {}).get("validatedActionIds", []))))


def bind_family(store, family_id: str, partition: str) -> None:
    """Once a family is assigned, later imports cannot turn a test into training."""
    if partition not in {"train", "validation", "test"}:
        raise ValueError("The curriculum family has no valid fixed partition.")
    identifier = digest({"teachingFamily": family_id})
    path = store.location("teaching-families", identifier)
    if path.exists():
        if store.get("teaching-families", identifier)["partition"] != partition:
            raise ValueError("Teaching family partition is immutable; use the original partition.")
    else:
        store.put("teaching-families", identifier, {"id": identifier, "familyId": family_id, "partition": partition})


def from_position(store, root: Path, position_id: str, family_id: str) -> dict:
    """Bind a reviewed source family to a real saved pre-decision position.

    The family fixes the split. A client cannot relabel a test family as train,
    and a human benchmark position remains test data even in a train family.
    """
    family = next((item for item in curriculum(root) if item.get("id", item.get("familyId")) == family_id), None)
    if family is None:
        raise ValueError("Choose an existing curriculum family.")
    position = store.get("positions", position_id)
    replay = store.get("replays", position["sourceReplayId"])
    partition = family.get("partition", "test")
    if replay.get("evaluationExperiment") or replay.get("humanMatch"):
        partition = "test"
    bind_family(store, family_id, family.get("partition", "test"))
    observation = copy.deepcopy(position["observation"])
    record = {"id": uuid.uuid4().hex, "schemaVersion": 1, "familyId": family_id,
              "title": position["title"], "partition": partition, "observation": observation,
              "positionId": position_id, "playerId": position["playerId"], "positionHash": digest(observation),
              "engineVersion": position["engineVersion"], "source": copy.deepcopy(family.get("source", {})),
              "deckVersion": family.get("deckVersion"), "formatDate": family.get("formatDate", "2026-09-17"),
              "reviewStatus": "draft", "trainingEligible": False, "rulesValidation": "unverified",
              "rulesAuditStatus": "verified" if replay.get("trainingEligible") is True else "unverified",
              "reason": "guide-position", "createdAt": datetime.now(timezone.utc).isoformat()}
    # Validate against the original simulator frame, not editable client labels.
    from .analysis import select_frame
    frame = select_frame(replay, position["decisionIndex"], position["playerId"])
    if digest(frame) != record["positionHash"]:
        raise ValueError("Saved position differs from its original engine frame.")
    record["rulesValidation"] = "legal-position"
    store.put("teaching", record["id"], record)
    return record


def curriculum(root: Path) -> list[dict]:
    path = root / "research/curriculum.json"
    if not path.exists(): return []
    content = json.loads(path.read_text())
    return content if isinstance(content, list) else content.get("families", content.get("fixtures", []))


def queue(store, limit: int = 10) -> list[dict]:
    # Recent user bookmarks take priority; stable ordering makes weekly review reproducible.
    records = [r for r in store.list("teaching") if r.get("reviewStatus") == "draft" and not r.get("superseded")]
    priority = {"rules-issue": 0, "disagreement": 1, "discovery": 2, "human-bookmark": 3}
    records.sort(key=lambda r: (priority.get(r.get("reason"), 4), r.get("createdAt", ""), r["id"]))
    return [{k: v for k, v in record.items() if k != "observation"} for record in records[:min(10, max(1, limit))]]


def review(store, identifier: str, *, review_status: str, acceptable_action_ids: list[str], reasoning: str,
           rejected_action_ids: list[str] | None = None, critical_resources: str = "", confidence: str = "uncertain") -> dict:
    record = store.get("teaching", identifier)
    if review_status not in {"draft", "reviewed", "rejected"} or confidence not in {"uncertain", "likely", "confident"}:
        raise ValueError("Invalid review status or confidence.")
    legal = {a["id"] for a in record.get("observation", {}).get("legalActions", [])}
    rejected = rejected_action_ids or []
    if not set(acceptable_action_ids + rejected) <= legal:
        raise ValueError("Annotations must refer to legal actions in the saved position.")
    if set(acceptable_action_ids) & set(rejected): raise ValueError("An action cannot be both acceptable and rejected.")
    if review_status == "reviewed" and (not acceptable_action_ids or not reasoning.strip()):
        raise ValueError("A reviewed example needs acceptable legal actions and conditional reasoning.")
    record.update(reviewStatus=review_status, acceptableActionIds=acceptable_action_ids,
                  rejectedActionIds=rejected, conditionalReasoning=reasoning.strip(),
                  criticalResources=critical_resources.strip(), confidence=confidence)
    record["reviewedAt"] = datetime.now(timezone.utc).isoformat()
    record["reviewHash"] = review_hash(record)
    record["trainingEligible"] = eligible(record)
    # A later correction updates the queue entry, but cannot erase the exact
    # annotation that supplied a prior checkpoint's demonstration provenance.
    if not store.location("teaching-reviews", record["reviewHash"]).exists():
        store.put("teaching-reviews", record["reviewHash"], copy.deepcopy(record))
    store.put("teaching", identifier, record)
    return record
