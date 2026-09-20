"""Append-only mechanical evidence for an unchanged immutable human review.

The original review and receipt are never rewritten. A later audit can extend
transition coverage, but cannot alter an annotation, split, or information set.
"""
from __future__ import annotations

from pathlib import Path

from .engine import EngineClient
from .storage import Store, digest
from .teaching import eligible, valid_review


def action_bindings(record: dict) -> dict:
    actions = {action["id"]: action for action in record["observation"]["legalActions"]}
    return {identifier: {key: actions[identifier][key] for key in ("type", "cardId", "sourceRef", "targetRef")
                         if key in actions[identifier]} for identifier in record["acceptableActionIds"]}


def source_receipt(store: Store, record: dict) -> dict:
    """Require the exact immutable review snapshot and original simulator receipt."""
    if not valid_review(record, partition=record.get("partition")) or not record.get("fixtureReceiptHash"):
        raise ValueError("A valid reviewed fixture with original receipt is required")
    snapshot = store.get("teaching-reviews", record["reviewHash"])
    if digest(snapshot) != digest(record):
        raise ValueError("Current record differs from its immutable human review snapshot")
    receipt = store.get("fixture-receipts", record["id"])
    if (digest(receipt) != record["fixtureReceiptHash"] or digest(receipt.get("observation")) != record["positionHash"]
            or receipt.get("engineVersion") != record["engineVersion"] or receipt.get("fixtureHash") != record.get("fixtureHash")
            or receipt.get("fixtureId") != record["familyId"] or receipt.get("variationId") != record.get("variationId")
            or receipt.get("mechanicsAudit") != record.get("mechanicsAudit")):
        raise ValueError("Original fixture receipt or its position/provenance has changed")
    return receipt


def attestation_valid(record: dict, attestation: dict) -> bool:
    """Pure binding validation; caller also validates stored source artifacts."""
    payload = {key: value for key, value in attestation.items() if key not in {"id", "attestationHash"}}
    if attestation.get("id") != digest(payload) or attestation.get("attestationHash") != digest(payload):
        return False
    for key, expected in {"teachingId": record["id"], "sourceRecordHash": digest(record),
                          "reviewHash": record.get("reviewHash"), "fixtureReceiptHash": record.get("fixtureReceiptHash"),
                          "positionHash": record.get("positionHash"), "familyId": record.get("familyId"),
                          "variationId": record.get("variationId"), "partition": record.get("partition"),
                          "sourceEngineVersion": record.get("engineVersion"), "sourceFixtureHash": record.get("fixtureHash"),
                          "acceptedActionBindings": action_bindings(record)}.items():
        if attestation.get(key) != expected:
            return False
    audit = attestation.get("mechanicsAudit", {})
    compatibility = audit.get("compatibility", {})
    accepted = set(record["acceptableActionIds"])
    transitions = audit.get("transitions", [])
    if (attestation.get("schemaVersion") != 1 or attestation.get("kind") != "mechanical-review-extension"
            or not attestation.get("engineBuildHash") or not attestation.get("engineVersion")
            or audit.get("status") != "verified" or not audit.get("tests") or not audit.get("scope")
            or compatibility != {"mode": "exact-observation-and-action-bindings-v1", "observationEqual": True, "actionBindingsEqual": True}
            or audit.get("sourceEngineVersion") != record["engineVersion"] or audit.get("sourceFixtureHash") != record["fixtureHash"]
            or audit.get("engineVersion") != attestation["engineVersion"] or not audit.get("fixtureHash")
            or audit.get("fixtureId") != record["familyId"] or audit.get("variationId") != record["variationId"]
            or set(audit.get("validatedActionIds", [])) != accepted or len(transitions) != len(accepted)):
        return False
    bindings = action_bindings(record)
    return ({item.get("actionId") for item in transitions} == accepted
            and all(item.get("binding") == bindings[item["actionId"]] and item.get("transitionHash")
                    and all(item.get(flag) is True for flag in ("deterministic", "conserved", "oncePerTurn", "noImmediateDamage"))
                    for item in transitions))


def composed_admission(store: Store, record: dict, *, partition: str = "train") -> dict | None:
    """Return an in-memory view; never mutate or re-review the source record."""
    if not valid_review(record, partition=partition):
        return None
    try:
        if record.get("fixtureReceiptHash"):
            source_receipt(store, record)
        if eligible(record, partition=partition):
            return record
        matches = sorted((item for item in store.iter_records("teaching-audits") if attestation_valid(record, item)), key=lambda item: item["id"])
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None
    if not matches:
        return None
    return {**record, "effectiveTrainingEligible": partition == "train", "sourceRecordHash": digest(record),
            "mechanicsAttestationHash": matches[0]["attestationHash"]}


def audit_review(store: Store, root: Path, identifier: str) -> dict:
    """Execute the scoped simulator audit and append its checksum-addressed receipt."""
    record = store.get("teaching", identifier)
    receipt = source_receipt(store, record)
    with EngineClient(root) as engine:
        health = engine.request("health")
        audit = engine.request("auditFixture", {"fixtureId": record["familyId"], "variationId": record["variationId"],
            "observation": record["observation"], "acceptedActionIds": record["acceptableActionIds"],
            "sourceEngineVersion": record["engineVersion"], "sourceFixtureHash": receipt["fixtureHash"]})
    payload = {"schemaVersion": 1, "kind": "mechanical-review-extension", "teachingId": identifier,
               "sourceRecordHash": digest(record), "reviewHash": record["reviewHash"], "fixtureReceiptHash": record["fixtureReceiptHash"],
               "positionHash": record["positionHash"], "familyId": record["familyId"], "variationId": record["variationId"],
               "partition": record["partition"], "sourceEngineVersion": record["engineVersion"], "sourceFixtureHash": record["fixtureHash"],
               "acceptedActionBindings": action_bindings(record), "engineVersion": health["engineVersion"],
               "engineBuildHash": health.get("engineBuildHash"), "mechanicsAudit": audit}
    identifier_hash = digest(payload)
    attestation = {**payload, "id": identifier_hash, "attestationHash": identifier_hash}
    if not attestation_valid(record, attestation):
        raise ValueError("Simulator audit did not validate the complete unchanged review")
    # Detect concurrent annotation/source changes before publishing evidence.
    if digest(store.get("teaching", identifier)) != digest(record):
        raise ValueError("Review changed during mechanical audit; no admission was written")
    source_receipt(store, record)
    path = store.location("teaching-audits", identifier_hash)
    if path.exists():
        if store.get("teaching-audits", identifier_hash) != attestation:
            raise ValueError("Existing immutable teaching audit differs from its checksum")
    else:
        store.put("teaching-audits", identifier_hash, attestation)
    return attestation
