from __future__ import annotations

import hashlib
import json

import pytest
import torch

from ptcg_lab.learning_mind import specialist_evidence
from ptcg_lab.learning_mind.curriculum import ARCHETYPES, specialist_for_deck
from ptcg_lab.learning_mind.notifications import _VERIFIED_PROMOTION_TOKEN
from ptcg_lab.learning_mind.schema import identity_hash


def _checkpoint(path, experiment_identity):
    metadata = {"schemaVersion": 1, "kind": "learning-mind-ppo-checkpoint-v1",
        "updateIndex": 0, "experimentIdentity": experiment_identity,
        "experimentIdentityHash": identity_hash(experiment_identity),
        "implementationIdentity": {"test": "fixture"}, "ppoConfig": {"test": True},
        "experienceManifestSha256": "a" * 64, "stageEvidenceHash": "b" * 64,
        "behaviorPolicyHash": "c" * 64, "device": "cpu"}
    payload = {**metadata, "manifestHash": identity_hash(metadata),
        "model": {}, "optimizer": {}, "torchRngState": torch.get_rng_state()}
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    decks = root / "decks"
    decks.mkdir(parents=True)
    deck_hashes = {}
    for archetype in ARCHETYPES:
        path = decks / f"{archetype}.json"
        path.write_text(json.dumps({"archetype": archetype}))
        deck_hashes[archetype] = hashlib.sha256(path.read_bytes()).hexdigest()

    generalist_path = tmp_path / "generalist.pt"
    generalist_hash = _checkpoint(generalist_path, {"learningMindIdentity": {"frozen": "same"}})
    generalist_promotion = tmp_path / "generalist-promotion.json"
    generalist_promotion.write_text("{}\n")
    specialists = []
    checkpoint_hashes = {}
    for archetype in ARCHETYPES:
        checkpoint = tmp_path / f"{archetype}-specialist.pt"
        checkpoint_hash = _checkpoint(checkpoint, {
            "learningMindIdentity": {"frozen": "same"},
            "specializedForDeckHash": deck_hashes[archetype],
            "parentCheckpointSha256": generalist_hash})
        promotion = tmp_path / f"{archetype}-promotion.json"
        promotion.write_text("{}\n")
        checkpoint_hashes[archetype] = checkpoint_hash
        specialists.append({"archetype": archetype, "deckHash": deck_hashes[archetype],
            "checkpoint": str(checkpoint), "checkpointSha256": checkpoint_hash,
            "promotionEvidence": str(promotion)})

    registry = {"schemaVersion": 1, "kind": "learning-mind-specialist-registry-v1",
        "generalist": {"checkpoint": str(generalist_path),
            "checkpointSha256": generalist_hash,
            "promotionEvidence": str(generalist_promotion)},
        "specialists": specialists}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))

    def verify(path, *, expected_candidate_hash, expected_control_hash=None):
        name = path.name
        archetype = name.removesuffix("-promotion.json")
        expected = (generalist_hash if archetype == "generalist"
                    else checkpoint_hashes[archetype])
        assert expected_candidate_hash == expected
        if expected_control_hash is not None:
            assert expected_control_hash == generalist_hash
        report = {"reportHash": identity_hash({"receipt": name}),
            "candidateCheckpointSha256": expected_candidate_hash,
            "controlCheckpointSha256": expected_control_hash or "d" * 64,
            "promotionCriteriaPassed": True, "humanApproved": True,
            "automaticPromotion": False}
        evidence = specialist_evidence.VerifiedPromotionEvidence({
            "promotionCriteriaPassed": True, "humanApproved": True,
            "automaticPromotion": False, "candidateCheckpointSha256": expected_candidate_hash,
            "controlCheckpointSha256": expected_control_hash or "d" * 64},
            _verification_token=_VERIFIED_PROMOTION_TOKEN)
        return report, evidence

    monkeypatch.setattr(specialist_evidence, "_verified_promotion", verify)
    return root, registry_path, deck_hashes


def test_specialist_verifier_binds_exact_decks_and_routes_only_from_capability(tmp_path, monkeypatch):
    root, registry_path, deck_hashes = _fixture(tmp_path, monkeypatch)
    report, capability = specialist_evidence.verify_specialist_curriculum(
        root=root, registry_path=registry_path, output=tmp_path / "verified.json")
    assert report["status"] == "passed"
    assert report["specialistCount"] == 5
    assert specialist_for_deck(deck_hashes["dragapult"], {}, "generalist.pt") == "generalist.pt"
    assert specialist_for_deck(deck_hashes["dragapult"], capability, "generalist.pt").endswith(
        "dragapult-specialist.pt")
    assert specialist_for_deck("0" * 64, capability, "generalist.pt") == "generalist.pt"
    with pytest.raises(TypeError):
        capability._values["specialistCheckpointsByDeckHash"][deck_hashes["dragapult"]]["checkpoint"] = "tampered"


def test_specialist_verifier_rejects_registry_deck_hash_drift(tmp_path, monkeypatch):
    root, registry_path, _deck_hashes = _fixture(tmp_path, monkeypatch)
    registry = json.loads(registry_path.read_text())
    registry["specialists"][0]["deckHash"] = "0" * 64
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="exact approved deck"):
        specialist_evidence.verify_specialist_curriculum(
            root=root, registry_path=registry_path, output=tmp_path / "verified.json")


def test_specialist_verifier_rejects_modified_checkpoint_manifest(tmp_path, monkeypatch):
    root, registry_path, _deck_hashes = _fixture(tmp_path, monkeypatch)
    registry = json.loads(registry_path.read_text())
    checkpoint = registry["specialists"][0]["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["manifestHash"] = "0" * 64
    torch.save(payload, checkpoint)
    with open(checkpoint, "rb") as stream:
        registry["specialists"][0]["checkpointSha256"] = hashlib.sha256(stream.read()).hexdigest()
    registry_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="manifest/experiment identity hash mismatch"):
        specialist_evidence.verify_specialist_curriculum(
            root=root, registry_path=registry_path, output=tmp_path / "verified.json")
