"""Admit simulator-produced tactical positions without certifying an entire deck.

Only the server calls this module with the simulator's fixture response. Strategy
labels are deliberately absent: a mechanics audit cannot stand in for review.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

from .storage import digest
from .teaching import bind_family, curriculum


def from_fixture(store, root: Path, receipt: dict) -> dict:
    family_id, variation_id = receipt.get("fixtureId"), receipt.get("variationId")
    family = next((item for item in curriculum(root) if item.get("id") == family_id), None)
    if family is None:
        raise ValueError("Simulator fixture is not an existing curriculum family.")
    variations = family.get("variations", family.get("variants", []))
    variation = next((item for item in variations if item.get("id") == variation_id), None)
    if variation is None:
        raise ValueError("Simulator variation is not registered in its curriculum family.")
    observation = receipt.get("observation", {})
    if observation.get("playerId") != observation.get("decisionPlayer") or not observation.get("legalActions"):
        raise ValueError("A tactical fixture needs a legal decision for its information perspective.")
    if not receipt.get("engineVersion") or not receipt.get("fixtureHash"):
        raise ValueError("Simulator fixture requires engine and recipe provenance.")
    audit = receipt.get("mechanicsAudit", {})
    legal = {action["id"] for action in observation["legalActions"]}
    verified = bool(audit.get("status") == "verified" and audit.get("tests") and audit.get("scope")
                    and audit.get("validatedActionIds") and set(audit["validatedActionIds"]) <= legal)
    partition = family.get("partition", "test")
    bind_family(store, family_id, partition)
    # Deterministic identity prevents repeated hydration from filling the review
    # queue or overwriting a human annotation. Changed recipes get new identities.
    receipt_hash = digest(receipt)
    identifier = digest({"family": family_id, "variation": variation_id, "receipt": receipt_hash})[:32]
    def supersede_drafts():
        for previous in store.iter_records("teaching"):
            if (previous["id"] != identifier and previous.get("familyId") == family_id
                    and previous.get("variationId") == variation_id and previous.get("reviewStatus") == "draft"
                    and not previous.get("superseded")):
                previous.update(superseded=True, supersededBy=identifier, trainingEligible=False)
                store.put("teaching", previous["id"], previous)
    if store.location("teaching", identifier).exists():
        supersede_drafts()
        return store.get("teaching", identifier)
    record = {"id": identifier, "schemaVersion": 2, "familyId": family_id, "variationId": variation_id,
              "title": variation.get("title", variation.get("description", variation.get("condition", variation_id))),
              "partition": partition, "observation": copy.deepcopy(observation),
              "playerId": observation["playerId"], "positionHash": digest(observation),
              "engineVersion": receipt["engineVersion"], "fixtureHash": receipt["fixtureHash"],
              "fixtureReceiptHash": receipt_hash, "mechanicsAudit": copy.deepcopy(audit),
              "decisionBindings": copy.deepcopy(receipt.get("decisionBindings", {})),
              "source": copy.deepcopy(family.get("source", {})), "deckVersion": family.get("deckVersion"),
              "formatDate": family.get("formatDate", "2026-09-17"), "reviewStatus": "draft",
              "trainingEligible": False, "rulesValidation": "legal-position",
              "rulesAuditStatus": "verified" if verified else "unverified", "reason": "guide-position",
              "createdAt": datetime.now(timezone.utc).isoformat(),
              "limitations": ["Supported-deck teaching analogue; strategic preference awaits human review.",
                              "Position-specific transition audit does not certify complete-game outcomes.",
                              *audit.get("limitations", [])]}
    store.put("fixture-receipts", identifier, copy.deepcopy(receipt))
    store.put("teaching", identifier, record)
    supersede_drafts()
    return record
