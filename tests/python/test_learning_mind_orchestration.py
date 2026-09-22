from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind import experiment
from ptcg_lab.learning_mind.dataset_v1 import file_sha256, load_dataset, training_records
from ptcg_lab.learning_mind.encoding import encode_decision
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
from test_learning_mind_representation import observation


def frozen_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    obs = observation()
    tracker = ObservableHistoryTracker(0).update(obs)
    encoded = encode_decision(obs, tracker)
    row = {"positionHash": "position", "familyId": "family", "sourceGameId": None,
           "sourceDecisionIndex": None, "actor": 0, "deckHash": None,
           "opponentArchetype": "fixture", "opponentPolicyFamily": "fixture-policy",
           "featureIdentityHash": encoded.identity, "policyLabelSource": "compatible-reviewed-acceptable-set",
           "acceptableActionIndices": [0], "policyDistribution": None, "split": "train",
           "observation": obs, "tracker": tracker}
    rows = dataset / "rows.jsonl"
    rows.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    identity = {"identityHash": "fixture"}
    manifest = {"schemaVersion": 1, "identity": identity, "rows": 1,
                "rowsSha256": file_sha256(rows), "manifestHash": "dataset-manifest"}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    return dataset, identity


def test_dataset_hash_identity_and_feature_identity_are_enforced(tmp_path):
    dataset, identity = frozen_dataset(tmp_path)
    _, rows = load_dataset(dataset, identity=identity)
    assert len(training_records(rows)) == 1
    with pytest.raises(ValueError, match="identity mismatch"):
        load_dataset(dataset, identity={"identityHash": "drift"})
    with (dataset / "rows.jsonl").open("a") as target:
        target.write("{}\n")
    with pytest.raises(ValueError, match="rows hash mismatch"):
        load_dataset(dataset)


def test_macro_collection_is_checkpointed_and_resume_does_not_replace_positions(tmp_path, monkeypatch):
    dataset, identity = frozen_dataset(tmp_path)
    calls = []

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, payload=None):
            calls.append((method, payload))
            action = payload["observation"]["legalActions"][0]
            return {"status": "complete", "alternatives": [
                {"actionId": action["id"], "visits": 1, "score": .4}]}

    monkeypatch.setattr(experiment, "EngineClient", FakeEngine)
    output = tmp_path / "labels"
    first = experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                            identity=identity, limit=1, initial=1, maximum=1)
    call_count = len(calls)
    assert first["positions"] == 1 and first["highConfidencePolicyLabels"] == 0
    second = experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                             identity=identity, limit=1, initial=1, maximum=1)
    assert second["manifestHash"] == first["manifestHash"]
    assert len(calls) == call_count
    record = json.loads(next(path for path in output.glob("*.json") if path.name != "manifest.json").read_text())
    assert record["semantics"].startswith("root-action-proxy")
    assert record["highConfidencePolicyEligible"] is False
    assert len(record["rolloutSeeds"]) == 1
    assert all(label["outcomes"] == {"finished": 1, "truncated": 0, "error": 0}
               for label in record["labels"])
    with pytest.raises(ValueError, match="configuration drift"):
        experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                        identity=identity, limit=1, initial=1, maximum=2)
