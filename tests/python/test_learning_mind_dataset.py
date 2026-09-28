from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind import dataset_v1
from ptcg_lab.learning_mind.dataset_v1 import build_dataset, load_dataset
from ptcg_lab.learning_mind.schema import IdentityManifest, identity_hash
from ptcg_lab.storage import Store, digest as legacy_digest
from test_learning_mind_representation import observation


def _source_fixture(tmp_path, probabilities=None):
    identity = IdentityManifest.create(engine_build_hash="engine-v1", deck_manifests={"deck": "hash"},
                                       card_metadata={"card": "hash"})
    experimental = tmp_path / "experimental"
    obs = observation()
    replay = {
        "id": "game-1",
        "decks": ["own-deck", "opponent-deck"],
        "deckHashes": ["own-hash", "opponent-hash"],
        "searchTargets": [{"decisionIndex": 0, "actor": 0,
                           "observationHash": legacy_digest(obs),
                           "probabilities": probabilities or {"1": .6, "2": .4, "3": 0., "4": 0.}}],
        "frames": [{"decisionIndex": 0, "actor": 0, "observations": [obs, {"private": "must-not-copy"}]}],
    }
    Store(experimental).put("replays", replay["id"], replay)
    source = {"settings": {"identity": identity.record()},
              "replays": [{"id": "game-1", "familyId": "family-1"}]}
    source["manifestHash"] = identity_hash(source)
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps(source))
    return identity, experimental, manifest_path


def test_supervised_dataset_freeze_binds_source_and_keeps_actor_view(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_v1, "EXACT_REVIEWS", ())
    identity, experimental, source = _source_fixture(tmp_path)
    output = tmp_path / "frozen-dataset"
    manifest = build_dataset(root=tmp_path, output=output, review_root=tmp_path / "reviews",
        experimental_root=experimental, source_dataset_manifest=source, identity=identity)

    loaded, rows = load_dataset(output, identity=identity.record())
    assert manifest == loaded and len(rows) == 1
    row = rows[0]
    assert row["policyLabelSource"] == "exact-search-distribution"
    assert row["policyDistribution"][-1] == 0.0  # search targets never label STOP
    assert row["observation"]["playerId"] == row["actor"] == 0
    assert "private" not in json.dumps(row)
    assert manifest["sources"][0]["manifestHash"] == json.loads(source.read_text())["manifestHash"]
    assert manifest["ordinarySelfPlayPolicyLabels"] == 0


def test_supervised_dataset_rejects_source_manifest_hash_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_v1, "EXACT_REVIEWS", ())
    identity, experimental, source_path = _source_fixture(tmp_path)
    source = json.loads(source_path.read_text())
    source["replays"][0]["familyId"] = "tampered-family"
    source_path.write_text(json.dumps(source))
    output = tmp_path / "frozen-dataset"
    with pytest.raises(ValueError, match="manifest hash"):
        build_dataset(root=tmp_path, output=output, review_root=tmp_path / "reviews",
            experimental_root=experimental, source_dataset_manifest=source_path, identity=identity)
    assert not output.exists()
    assert list(tmp_path.glob(".frozen-dataset.*")) == []


def test_supervised_dataset_rejects_source_identity_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_v1, "EXACT_REVIEWS", ())
    identity, experimental, source = _source_fixture(tmp_path)
    incompatible = IdentityManifest.create(engine_build_hash="different-engine", deck_manifests={"deck": "hash"},
                                           card_metadata={"card": "hash"})
    output = tmp_path / "frozen-dataset"
    with pytest.raises(ValueError, match="identity mismatch"):
        build_dataset(root=tmp_path, output=output, review_root=tmp_path / "reviews",
            experimental_root=experimental, source_dataset_manifest=source, identity=incompatible)
    assert not output.exists()
    assert list(tmp_path.glob(".frozen-dataset.*")) == []


def test_supervised_dataset_failed_conversion_leaves_no_partial_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_v1, "EXACT_REVIEWS", ())
    identity, experimental, source = _source_fixture(tmp_path, {"1": .6, "NOT-LEGAL": .4})
    output = tmp_path / "frozen-dataset"
    with pytest.raises(ValueError, match="semantic legal-action classes"):
        build_dataset(root=tmp_path, output=output, review_root=tmp_path / "reviews",
            experimental_root=experimental, source_dataset_manifest=source, identity=identity)
    assert not output.exists()
    assert list(tmp_path.glob(".frozen-dataset.*")) == []
