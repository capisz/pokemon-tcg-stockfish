from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind import experiment
from ptcg_lab.learning_mind.dataset_v1 import (build_macro_position_pool, file_sha256,
                                               load_dataset, training_records)
from ptcg_lab.learning_mind.encoding import encode_decision
from ptcg_lab.learning_mind.macro import CANDIDATE_GENERATOR_VERSION
from ptcg_lab.learning_mind.macro import candidates_from_transition_plans
from ptcg_lab.learning_mind.schema import IdentityManifest
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
from ptcg_lab.storage import Store
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
    observation_row = json.loads((dataset / "rows.jsonl").read_text())
    root_action = observation_row["observation"]["legalActions"][0]
    next_action = {**observation_row["observation"]["legalActions"][-1], "id": "next:0", "type": "attack"}

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
                {"actionId": action["id"], "visits": 1, "score": .4,
                 "continuation": {"end": "terminal", "outcome": {"winner": 0, "reason": "fixture"}}}],
                "macroPlanExecution": {"requested": True, "completed": 1, "failures": []}}

    monkeypatch.setattr(experiment, "EngineClient", FakeEngine)
    monkeypatch.setattr(experiment, "transition_generator_identity", lambda root: {"version": CANDIDATE_GENERATOR_VERSION,
        "plannerSha256": "planner", "actionKeySha256": "action-key-v1", "adapterSha256": "adapter"})
    monkeypatch.setattr(experiment, "generate_transition_candidates", lambda root, observation, seed: (
        candidates_from_transition_plans([{"actions": [root_action, next_action], "completion": "attack"}]),
        {"hypothesisId": "public-test-hypothesis"}))
    output = tmp_path / "labels"
    first = experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                            identity=identity, limit=1, initial=1, maximum=1,
                                            position_hash="position")
    call_count = len(calls)
    assert first["positions"] == 1 and first["highConfidencePolicyLabels"] == 0
    assert first["candidateGeneratorVersion"] == CANDIDATE_GENERATOR_VERSION
    assert first["selectedPositionHashes"] == ["position"]
    second = experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                             identity=identity, limit=1, initial=1, maximum=1)
    assert second["manifestHash"] == first["manifestHash"]
    assert len(calls) == call_count
    record = json.loads(next(path for path in output.glob("*.json") if path.name != "manifest.json").read_text())
    assert record["semantics"].startswith("complete transition-aware attack or deliberate no-attack candidates only")
    assert record["highConfidencePolicyEligible"] is False
    assert record["generatorHypothesisId"] == "public-test-hypothesis"
    assert len(record["rolloutSeeds"]) == 1
    assert all(len(call[1].get("macroPlanActions", [])) == 2 for call in calls)
    assert all(label["outcomes"] == {"finished": 1, "truncated": 0, "error": 0}
               for label in record["labels"])
    with pytest.raises(ValueError, match="configuration drift"):
        experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                        identity=identity, limit=1, initial=1, maximum=2)
    monkeypatch.setattr(experiment, "transition_generator_identity", lambda root: {"version": CANDIDATE_GENERATOR_VERSION,
        "plannerSha256": "planner", "actionKeySha256": "action-key-v2", "adapterSha256": "adapter"})
    with pytest.raises(ValueError, match="configuration drift"):
        experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                        identity=identity, limit=1, initial=1, maximum=1)


def test_macro_collector_rejects_position_hash_outside_frozen_pool(tmp_path):
    dataset, identity = frozen_dataset(tmp_path)
    with pytest.raises(ValueError, match="not present in the frozen dataset"):
        experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=tmp_path / "labels",
                                        identity=identity, position_hash="not-in-pool", initial=1, maximum=1)


def test_macro_collector_treats_search_horizon_cutoff_as_truncated_not_a_label(tmp_path, monkeypatch):
    dataset, identity = frozen_dataset(tmp_path)
    row = json.loads((dataset / "rows.jsonl").read_text())
    root_action = row["observation"]["legalActions"][0]
    attack = {**row["observation"]["legalActions"][-1], "id": "attack:0", "type": "attack"}

    class CutoffEngine:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def request(self, method, payload=None):
            action = payload["observation"]["legalActions"][0]
            return {"status": "complete", "alternatives": [{"actionId": action["id"], "visits": 1,
                    "score": .8, "continuation": {"end": "cutoff"}}],
                    "macroPlanExecution": {"requested": True, "completed": 1, "failures": []}}

    monkeypatch.setattr(experiment, "EngineClient", CutoffEngine)
    monkeypatch.setattr(experiment, "transition_generator_identity", lambda root: {"version": CANDIDATE_GENERATOR_VERSION,
        "plannerSha256": "planner", "actionKeySha256": "action-key", "adapterSha256": "adapter"})
    monkeypatch.setattr(experiment, "generate_transition_candidates", lambda root, observation, seed: (
        candidates_from_transition_plans([{"actions": [root_action, attack], "completion": "attack"}]),
        {"hypothesisId": "cutoff-test"}))
    output = tmp_path / "cutoff-labels"
    experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output, identity=identity,
                                    initial=1, maximum=1, position_hash="position")
    record = json.loads(next(path for path in output.glob("*.json") if path.name != "manifest.json").read_text())
    label, = record["labels"]
    assert label["expectedResult"] is None
    assert label["completedRollouts"] == 0
    assert label["outcomes"] == {"finished": 0, "truncated": 1, "error": 0}


def test_macro_position_pool_is_unlabeled_actor_visible_and_balanced(tmp_path):
    experimental = tmp_path / "experimental"
    store = Store(experimental)
    replay_items = []
    for index, opponent in enumerate(("crustle", "dragapult", "grimmsnarl")):
        obs = observation()
        replay_id = f"replay{index}"
        replay = {"id": replay_id, "dataTier": "experimental", "status": "finished",
                  "decks": ["raging-bolt", opponent], "deckHashes": [f"own{index}", f"opp{index}"],
                  "policies": ["current", "historical" if index % 2 else "heuristic"],
                  "frames": [{"decisionIndex": index, "actor": 0, "observations": [obs, {}]}]}
        store.put("replays", replay_id, replay)
        replay_items.append({"id": replay_id, "familyId": f"family{index}",
                             "decks": replay["decks"]})
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"replays": replay_items}))
    identity = IdentityManifest.create(engine_build_hash="engine", deck_manifests={}, card_metadata={})
    output = tmp_path / "pool"
    manifest = build_macro_position_pool(output=output, experimental_root=experimental,
                                         source_dataset_manifest=source, identity=identity, limit=3)
    _, rows = load_dataset(output, identity=identity.record())
    assert manifest["ordinarySelfPlayPolicyLabels"] == 0
    assert {row["split"] for row in rows} == {"train", "development", "heldout"}
    assert all(row["policyLabelSource"] is None and row["observation"]["playerId"] == row["actor"]
               for row in rows)
