from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from threading import Lock
from time import sleep

import pytest

from ptcg_lab.learning_mind import experiment
from ptcg_lab.learning_mind.dataset_v1 import (build_macro_position_pool, file_sha256,
                                               load_dataset, training_records)
from ptcg_lab.learning_mind.encoding import encode_decision
from ptcg_lab.learning_mind.macro import (CANDIDATE_GENERATOR_VERSION, MacroCandidateV1,
                                          candidates_from_transition_plans, label_candidates, rollout_seed)
from ptcg_lab.learning_mind.schema import IdentityManifest
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
from ptcg_lab.storage import Store
from test_learning_mind_representation import observation


def install_fake_engine_pool(monkeypatch, engine_type):
    class FakePool:
        def __init__(self, root, size=1, timeout=300):
            self.engine = engine_type()
        def close(self): pass
        @contextmanager
        def lease(self):
            yield self.engine

    monkeypatch.setattr(experiment, "EnginePool", FakePool)


def test_macro_rollout_workers_preserve_matched_seeds_and_candidate_order():
    candidates = [MacroCandidateV1(turn_intent="attack", intended_attack=f"attack-{i}") for i in range(3)]
    lock = Lock()
    active = 0
    peak = 0
    seen = {candidate.key(): [] for candidate in candidates}

    def rollout(candidate, seed):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen[candidate.key()].append(seed)
        sleep(.02)
        with lock:
            active -= 1
        return {"status": "finished", "score": .5, "decisionCount": 4}

    labels = label_candidates(candidates, "matched-position", rollout, initial=2, maximum=2,
                              rollout_workers=2)
    expected_seeds = [rollout_seed("training", "matched-position", index) for index in range(2)]
    assert peak == 2
    assert all(seeds == expected_seeds for seeds in seen.values())
    assert [label["candidateHash"] for label in labels] == [candidate.key() for candidate in candidates]
    assert all(label["decisionCountDistribution"] == {"4": 2} for label in labels)


def test_macro_candidate_rollouts_resume_at_last_complete_matched_seed():
    candidate = MacroCandidateV1(turn_intent="attack", intended_attack="resume-fixture")
    checkpoints = []
    seen_seeds = []

    def rollout(_candidate, seed):
        seen_seeds.append(seed)
        return {"status": "finished", "score": .5, "decisionCount": 12}

    def interrupt_after_first_seed(state):
        checkpoints.append(json.loads(json.dumps(state)))
        if len(checkpoints) == 1:
            raise InterruptedError("simulated interruption after complete seed batch")

    with pytest.raises(InterruptedError):
        label_candidates([candidate], "resume-position", rollout, initial=2, maximum=2,
                         checkpoint=interrupt_after_first_seed)
    assert checkpoints[0]["completedInitialIndices"] == [0]
    seen_seeds.clear()
    resumed = label_candidates([candidate], "resume-position", rollout, initial=2, maximum=2,
                               resume_state=checkpoints[0])
    assert seen_seeds == [rollout_seed("training", "resume-position", 1)]
    clean = label_candidates([candidate], "resume-position", rollout, initial=2, maximum=2)
    assert resumed == clean


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


def test_transition_macro_generation_uses_built_cjs_worker(tmp_path, monkeypatch):
    bundle = tmp_path / "packages/engine/dist/learning-mind-planner.cjs"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("generated fixture bundle")
    response = {"version": CANDIDATE_GENERATOR_VERSION, "hypothesisId": "public-fixture",
                "exploredPrefixCount": 2,
                "candidates": [{"actions": [{"id": "root:0", "type": "pass", "label": "End turn"}],
                                "completion": "no-attack"}]}
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(experiment.subprocess, "run", run)
    candidates, metadata = experiment.generate_transition_candidates(
        tmp_path, observation(), seed=123)
    assert calls[0][0] == ["node", str(bundle)]
    assert candidates[0].action_sequence[0]["id"] == "root:0"
    assert metadata["hypothesisId"] == "public-fixture"


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
                 "continuation": {"end": "terminal", "decisionCount": 77,
                                  "outcome": {"winner": 0, "reason": "fixture"}}}],
                "macroPlanExecution": {"requested": True, "completed": 1, "failures": []}}

    install_fake_engine_pool(monkeypatch, FakeEngine)
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
    assert first["rolloutWorkers"] == 1
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
    assert record["rolloutBudgetMs"] == 1000
    assert all(call[1]["budgetMs"] == 1000 for call in calls)
    assert all(len(call[1].get("macroPlanActions", [])) == 2 for call in calls)
    assert all(label["outcomes"] == {"finished": 1, "truncated": 0, "error": 0}
               for label in record["labels"])
    assert all(label["decisionCountDistribution"] == {"77": 1} for label in record["labels"])
    with pytest.raises(ValueError, match="configuration drift"):
        experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                        identity=identity, limit=1, initial=1, maximum=2)
    with pytest.raises(ValueError, match="configuration drift"):
        experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output,
                                        identity=identity, limit=1, initial=1, maximum=1,
                                        rollout_workers=2)
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

    install_fake_engine_pool(monkeypatch, CutoffEngine)
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
    assert label["outcomeReasons"] == {"search-rollout-horizon-cutoff": 1}


def test_macro_collector_treats_search_budget_cutoff_as_truncated_and_freezes_budget(tmp_path, monkeypatch):
    dataset, identity = frozen_dataset(tmp_path)
    row = json.loads((dataset / "rows.jsonl").read_text())
    root_action = row["observation"]["legalActions"][0]
    attack = {**row["observation"]["legalActions"][-1], "id": "attack:budget", "type": "attack"}

    class BudgetCutoffEngine:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def request(self, method, payload=None):
            assert payload["budgetMs"] == 250
            action = payload["observation"]["legalActions"][0]
            return {"status": "complete", "alternatives": [{"actionId": action["id"], "visits": 1,
                    "score": .7, "continuation": {"end": "cutoff", "cutoffReason": "budget",
                                                   "decisionCount": 9}}],
                    "macroPlanExecution": {"requested": True, "completed": 1, "failures": []}}

    install_fake_engine_pool(monkeypatch, BudgetCutoffEngine)
    monkeypatch.setattr(experiment, "transition_generator_identity", lambda root: {"version": CANDIDATE_GENERATOR_VERSION,
        "plannerSha256": "planner", "actionKeySha256": "action-key", "adapterSha256": "adapter"})
    monkeypatch.setattr(experiment, "generate_transition_candidates", lambda root, observation, seed: (
        candidates_from_transition_plans([{"actions": [root_action, attack], "completion": "attack"}]),
        {"hypothesisId": "budget-cutoff-test"}))
    output = tmp_path / "budget-cutoff-labels"
    manifest = experiment.collect_macro_labels(root=tmp_path, dataset_dir=dataset, output=output, identity=identity,
        initial=1, maximum=1, horizon=300, rollout_budget_ms=250, position_hash="position")
    record = json.loads(next(path for path in output.glob("*.json") if path.name != "manifest.json").read_text())
    label, = record["labels"]
    assert manifest["rolloutBudgetMs"] == 250
    assert record["rolloutBudgetMs"] == 250
    assert label["expectedResult"] is None
    assert label["outcomes"] == {"finished": 0, "truncated": 1, "error": 0}
    assert label["outcomeReasons"] == {"search-rollout-budget-cutoff": 1}
    assert label["decisionCountDistribution"] == {"9": 1}


def test_macro_position_pool_is_unlabeled_actor_visible_and_balanced(tmp_path):
    experimental = tmp_path / "experimental"
    store = Store(experimental)
    replay_items = []
    for index, opponent in enumerate(("crustle", "dragapult", "grimmsnarl")):
        replay_id = f"replay{index}"
        frames = []
        for decision_index, (turn, prizes) in enumerate(((2, 6), (6, 4), (12, 2))):
            obs = observation()
            obs["turn"] = turn
            obs["history"] = [f"P1: fixture game {index}"]
            for player in obs["players"]:
                player["prizesRemaining"] = prizes
            frames.append({"decisionIndex": decision_index, "actor": 0, "observations": [obs, {}]})
        replay = {"id": replay_id, "dataTier": "experimental", "status": "finished",
                  "engineVersion": "test-engine-v1",
                  "decks": ["raging-bolt", opponent], "deckHashes": [f"own{index}", f"opp{index}"],
                  "policies": ["current", "historical" if index % 2 else "heuristic"],
                  "frames": frames}
        store.put("replays", replay_id, replay)
        replay_items.append({"id": replay_id, "familyId": f"family{index}",
                             "decks": replay["decks"]})
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"replays": replay_items}))
    identity = IdentityManifest.create(engine_build_hash="engine", deck_manifests={}, card_metadata={})
    output = tmp_path / "pool"
    manifest = build_macro_position_pool(output=output, experimental_root=experimental,
                                         source_dataset_manifest=source, identity=identity, limit=9)
    _, rows = load_dataset(output, identity=identity.record())
    assert manifest["ordinarySelfPlayPolicyLabels"] == 0
    assert manifest["sourceEngineVersions"] == ["test-engine-v1"]
    assert {row["split"] for row in rows} == {"train", "development", "heldout"}
    assert {row["positionStage"] for row in rows} == {"opening", "midgame", "late"}
    assert manifest["positionStageCounts"] == {"late": 3, "midgame": 3, "opening": 3}
    splits_by_game = {}
    for row in rows:
        splits_by_game.setdefault(row["sourceGameId"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_game.values())
    assert len({row["positionHash"] for row in rows}) == len(rows)
    assert all(row["policyLabelSource"] is None and row["observation"]["playerId"] == row["actor"]
               for row in rows)
