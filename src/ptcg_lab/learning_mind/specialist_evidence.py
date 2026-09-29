"""Source-bound, exact-deck specialist curriculum evidence.

This module verifies previously completed specialist promotion receipts. It does
not train, evaluate, or promote a model; promotion re-verification reproduces
the already-frozen source evidence before a routing capability is issued.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from types import MappingProxyType

import torch

from .curriculum import ARCHETYPES
from .dataset_v1 import file_sha256
from .notifications import VerifiedPromotionEvidence
from .promotion_evidence import reissue_promotion_evidence_report
from .schema import identity_hash


_VERIFIED_SPECIALIST_TOKEN = object()


class VerifiedSpecialistCurriculumEvidence:
    """Immutable routing capability issued only from reverified evidence."""
    __slots__ = ("_values",)

    def __init__(self, values: dict, *, _verification_token: object):
        if _verification_token is not _VERIFIED_SPECIALIST_TOKEN:
            raise TypeError("specialist evidence must come from the source-artifact verifier")
        frozen = dict(values)
        routes = values.get("specialistCheckpointsByDeckHash")
        if isinstance(routes, dict):
            frozen["specialistCheckpointsByDeckHash"] = MappingProxyType({
                key: MappingProxyType(dict(record)) for key, record in routes.items()})
        object.__setattr__(self, "_values", MappingProxyType(frozen))

    def __setattr__(self, _name, _value):
        raise AttributeError("verified specialist evidence is immutable")


def _read_registry(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("specialist registry is unreadable") from error
    if (not isinstance(value, dict)
            or set(value) != {"schemaVersion", "kind", "generalist", "specialists"}
            or value.get("schemaVersion") != 1
            or value.get("kind") != "learning-mind-specialist-registry-v1"):
        raise ValueError("unsupported specialist registry")
    return value


def _checkpoint_identity(path: Path, *, label: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"{label} checkpoint is unreadable") from error
    if not isinstance(payload, dict) or payload.get("kind") != "learning-mind-ppo-checkpoint-v1":
        raise ValueError(f"{label} checkpoint is not a frozen PPO checkpoint")
    metadata = {key: value for key, value in payload.items()
                if key not in {"manifestHash", "model", "optimizer", "torchRngState"}}
    experiment_identity = payload.get("experimentIdentity")
    if (payload.get("manifestHash") != identity_hash(metadata)
            or not isinstance(experiment_identity, dict)
            or payload.get("experimentIdentityHash") != identity_hash(experiment_identity)):
        raise ValueError(f"{label} checkpoint manifest/experiment identity hash mismatch")
    experiment = experiment_identity
    if not isinstance(experiment, dict) or not isinstance(experiment.get("learningMindIdentity"), dict):
        raise ValueError(f"{label} checkpoint has no learning-mind identity")
    return experiment


def _verified_promotion(path: Path, *, expected_candidate_hash: str,
                        expected_control_hash: str | None = None) -> tuple[dict, VerifiedPromotionEvidence]:
    report, evidence = reissue_promotion_evidence_report(path)
    if (not isinstance(evidence, VerifiedPromotionEvidence)
            or evidence._values.get("promotionCriteriaPassed") is not True
            or evidence._values.get("humanApproved") is not True
            or evidence._values.get("automaticPromotion") is not False
            or evidence._values.get("candidateCheckpointSha256") != expected_candidate_hash
            or report.get("candidateCheckpointSha256") != expected_candidate_hash
            or report.get("humanApproved") is not True
            or report.get("promotionCriteriaPassed") is not True
            or report.get("automaticPromotion") is not False):
        raise ValueError("specialist curriculum requires human-approved passing promotion evidence")
    if (expected_control_hash is not None
            and (evidence._values.get("controlCheckpointSha256") != expected_control_hash
                 or report.get("controlCheckpointSha256") != expected_control_hash)):
        raise ValueError("specialist promotion was not evaluated against the frozen generalist")
    return report, evidence


def _publish_immutable(path: Path, report: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError("specialist evidence reports are immutable; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
        json.dump(report, temporary, sort_keys=True, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_specialist_curriculum(*, root: Path, registry_path: Path,
                                 output: Path) -> tuple[dict, VerifiedSpecialistCurriculumEvidence]:
    """Recheck all five exact-deck specialists and their human-approved receipts."""
    root = Path(root).resolve()
    registry_path = Path(registry_path).resolve()
    registry = _read_registry(registry_path)
    generalist_entry = registry.get("generalist")
    specialists = registry.get("specialists")
    if (not isinstance(generalist_entry, dict)
            or set(generalist_entry) != {"checkpoint", "checkpointSha256", "promotionEvidence"}
            or not isinstance(specialists, list) or len(specialists) != len(ARCHETYPES)):
        raise ValueError("specialist registry must include one frozen generalist and all five specialists")

    deck_hashes = {}
    for archetype in ARCHETYPES:
        deck_path = root / "decks" / f"{archetype}.json"
        if not deck_path.is_file():
            raise ValueError(f"approved specialist deck is missing: {archetype}")
        deck_hashes[archetype] = file_sha256(deck_path)
    if len(set(deck_hashes.values())) != len(deck_hashes):
        raise ValueError("approved specialist deck hashes are not unique")

    generalist_path = Path(generalist_entry["checkpoint"]).resolve()
    generalist_hash = file_sha256(generalist_path)
    if generalist_hash != generalist_entry["checkpointSha256"]:
        raise ValueError("frozen generalist checkpoint hash mismatch")
    generalist_experiment = _checkpoint_identity(generalist_path, label="generalist")
    if generalist_experiment.get("specializedForDeckHash") is not None:
        raise ValueError("generalist checkpoint is marked as a deck-specific specialist")
    generalist_promotion_path = Path(generalist_entry["promotionEvidence"]).resolve()
    generalist_promotion, _generalist_evidence = _verified_promotion(
        generalist_promotion_path, expected_candidate_hash=generalist_hash)

    by_deck_hash = {}
    specialist_sources = []
    for entry in specialists:
        if (not isinstance(entry, dict)
                or set(entry) != {"archetype", "deckHash", "checkpoint", "checkpointSha256",
                                  "promotionEvidence"}):
            raise ValueError("specialist registry entry has an invalid schema")
        archetype = entry["archetype"]
        if archetype not in deck_hashes or entry["deckHash"] != deck_hashes[archetype]:
            raise ValueError("specialist registry deck hash differs from the exact approved deck")
        if entry["deckHash"] in by_deck_hash:
            raise ValueError("specialist registry repeats an exact deck hash")
        checkpoint = Path(entry["checkpoint"]).resolve()
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash != entry["checkpointSha256"]:
            raise ValueError(f"specialist checkpoint hash mismatch: {archetype}")
        experiment = _checkpoint_identity(checkpoint, label=f"specialist {archetype}")
        if (experiment.get("specializedForDeckHash") != entry["deckHash"]
                or experiment.get("parentCheckpointSha256") != generalist_hash
                or experiment.get("learningMindIdentity") != generalist_experiment.get("learningMindIdentity")):
            raise ValueError(f"specialist checkpoint lineage/deck binding mismatch: {archetype}")
        promotion_path = Path(entry["promotionEvidence"]).resolve()
        promotion, _evidence = _verified_promotion(promotion_path,
            expected_candidate_hash=checkpoint_hash, expected_control_hash=generalist_hash)
        by_deck_hash[entry["deckHash"]] = {
            "archetype": archetype, "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_hash,
            "promotionEvidenceReportHash": promotion["reportHash"]}
        specialist_sources.append({
            "deckHash": entry["deckHash"], "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_hash,
            "promotionEvidence": str(promotion_path),
            "promotionEvidenceSha256": file_sha256(promotion_path),
            "promotionEvidenceReportHash": promotion["reportHash"]})

    if set(by_deck_hash) != set(deck_hashes.values()):
        raise ValueError("specialist registry does not exactly cover all approved deck hashes")
    sources = {"registry": {"path": str(registry_path), "sha256": file_sha256(registry_path)},
        "generalistCheckpoint": {"path": str(generalist_path), "sha256": generalist_hash},
        "generalistPromotionEvidence": {"path": str(generalist_promotion_path),
                                        "sha256": file_sha256(generalist_promotion_path)},
        "specialists": specialist_sources,
        "decks": [{"archetype": name, "path": str(root / "decks" / f"{name}.json"),
                   "sha256": deck_hashes[name]} for name in ARCHETYPES]}
    report = {"schemaVersion": 1, "kind": "verified-learning-mind-specialist-curriculum-v1",
        "status": "passed", "specialistCount": len(by_deck_hash),
        "generalistCheckpointSha256": generalist_hash,
        "specialistCheckpointsByDeckHash": by_deck_hash,
        "sources": sources, "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    _publish_immutable(Path(output), report)
    capability_values = {"specialistCurriculumPassed": True,
        "generalistCheckpointSha256": generalist_hash,
        "specialistCheckpointsByDeckHash": by_deck_hash,
        "sourceReportHash": report["reportHash"], "automaticPromotion": False}
    capability = VerifiedSpecialistCurriculumEvidence(capability_values,
        _verification_token=_VERIFIED_SPECIALIST_TOKEN)
    return report, capability
