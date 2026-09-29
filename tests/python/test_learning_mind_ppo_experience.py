from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind.ppo_experience import (PPOExperienceStore,
    ppo_records_from_game, ppo_training_schedule)


BEHAVIOR_HASH = "a" * 64
FEATURE_HASH = "b" * 64
SCHEMA_HASH = "d" * 64
HISTORY = ["e" * 64, "f" * 64]


def settings():
    return {"behaviorPolicyHash": BEHAVIOR_HASH, "featureSchemaHash": SCHEMA_HASH,
        "engineBuildHash": "c" * 64, "schedulerVersion": "ppo-training-scheduler-v1",
        "historicalPolicyHashes": HISTORY, "trainingSeedBase": 7543298}


def decision(actor=0):
    return {"actor": actor,
        "observation": {"schemaVersion": 1, "playerId": actor, "decisionPlayer": actor,
            "legalActions": [{"id": "action-0", "type": "attack"}]},
        "tracker": {"version": "observable-history-v1"},
        "decisionIndex": 0, "selectedAction": 0, "actionClassCount": 1, "oldLogProb": -0.2,
        "oldValue": 0.1, "behaviorPolicyHash": BEHAVIOR_HASH,
        "featureIdentityHash": FEATURE_HASH, "featureSchemaHash": SCHEMA_HASH}


def game(game_id, status="finished", *, outcome=None, decisions=None):
    if status == "finished" and outcome is None:
        outcome = {"winner": 0, "reason": "rules-terminal"}
    index = int(game_id, 16)
    schedule = ppo_training_schedule(game_count=index + 1, historical_policy_hashes=HISTORY)[index]
    return {"schemaVersion": 1, "gameId": schedule["gameId"], "status": status,
        "outcome": outcome, "schedule": schedule,
        "actorDecisions": decisions if decisions is not None else [decision(schedule["learnerSeats"][0])]}


def test_experience_store_resumes_with_checksums_and_separate_outcomes(tmp_path):
    store = PPOExperienceStore(tmp_path / "run", settings=settings())
    store.save_game(game("1"))
    store.save_game(game("2", "truncated", outcome=None))
    store.save_game(game("3", "error", outcome=None, decisions=[]))
    assert store.manifest["gameCounts"] == {"finished": 1, "truncated": 1, "error": 1}
    assert store.manifest["actorDecisions"] == 2
    assert [item["status"] for item in sorted(store.iter_games(),
            key=lambda value: value["schedule"]["scheduleIndex"])] == ["finished", "truncated", "error"]
    with pytest.raises(ValueError, match="cannot be replaced"):
        store.save_game(game("1"))

    resumed = PPOExperienceStore(tmp_path / "run", settings=settings())
    assert resumed.completed_game_ids() == frozenset(game(str(index))["gameId"] for index in (1, 2, 3))
    assert resumed.manifest == store.manifest


def test_experience_store_rejects_identity_drift_and_artifact_corruption(tmp_path):
    root = tmp_path / "run"
    store = PPOExperienceStore(root, settings=settings())
    item = store.save_game(game("4"))
    with pytest.raises(ValueError, match="identity/checksum mismatch"):
        PPOExperienceStore(root, settings={**settings(), "engineBuildHash": "d" * 64})
    path = store.games_dir / f"{item['gameId']}.json.gz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        PPOExperienceStore(root, settings=settings())


def test_experience_store_rejects_self_consistent_false_per_game_decision_count(tmp_path):
    from ptcg_lab.learning_mind.schema import identity_hash

    root = tmp_path / "run"
    store = PPOExperienceStore(root, settings=settings())
    store.save_game(game("4"))
    manifest_path = store.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["games"][0]["actorDecisions"] += 1
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                                if key != "manifestHash"})
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="per-game decision count differs"):
        PPOExperienceStore(root, settings=settings())


def test_experience_store_rejects_boolean_aggregate_counts(tmp_path):
    from ptcg_lab.learning_mind.schema import identity_hash

    root = tmp_path / "run"
    store = PPOExperienceStore(root, settings=settings())
    store.save_game(game("4"))
    manifest_path = store.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["gameCounts"]["finished"] = True
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                                if key != "manifestHash"})
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="aggregate schema is invalid"):
        PPOExperienceStore(root, settings=settings())


def test_actor_game_artifact_compression_is_deterministic(tmp_path):
    artifact_hashes = []
    for name in ("first", "second"):
        store = PPOExperienceStore(tmp_path / name, settings=settings())
        artifact_hashes.append(store.save_game(game("8"))["sha256"])
    assert artifact_hashes[0] == artifact_hashes[1]


def test_ppo_schedule_freezes_uniform_archetypes_policy_mix_mirrors_and_seeds():
    history = [f"{index:064x}" for index in range(3)]
    schedule = ppo_training_schedule(game_count=100, historical_policy_hashes=history)
    assert len(schedule) == 100
    assert sum(row["policyFamily"] == "historical" for row in schedule) == 20
    assert sum(row["mirror"] for row in schedule) == 5
    assert {archetype: sum(row["ownArchetype"] == archetype for row in schedule)
            for archetype in {row["ownArchetype"] for row in schedule}} == {
                "crustle": 20, "dragapult": 20, "raging-bolt": 20,
                "grimmsnarl": 20, "mega-lucario": 20}
    assert len({row["seed"] for row in schedule}) == 100
    assert schedule == ppo_training_schedule(game_count=100, historical_policy_hashes=history)
    for row in schedule:
        assert row["learnerSeats"] == ([row["scheduleIndex"] % 2]
            if row["policyFamily"] == "historical" else [0, 1])


@pytest.mark.parametrize("mutate,match", [
    (lambda row: row.update(actor=1), "observation/tracker must belong"),
    (lambda row: row["observation"].update(oppositeObservation={"playerId": 1}), "hidden/private view"),
    (lambda row: row.update(behaviorPolicyHash="f" * 64), "mixes behavior-policy"),
    (lambda row: row.update(selectedAction=2), "within the frozen 128-class cap"),
    (lambda row: row["observation"].update(legalActions=[]), "preserve all actor-visible legal actions"),
])
def test_experience_store_rejects_leakage_or_invalid_decision(mutate, match, tmp_path):
    store = PPOExperienceStore(tmp_path / "run", settings=settings())
    row = decision()
    mutate(row)
    with pytest.raises(ValueError, match=match):
        store.save_game(game("5", decisions=[row]))


def test_experience_store_rejects_full_replay_and_unknown_game_fields(tmp_path):
    store = PPOExperienceStore(tmp_path / "run", settings=settings())
    full_replay = game("6")
    full_replay["frames"] = []
    with pytest.raises(ValueError, match="invalid PPO experience game schema"):
        store.save_game(full_replay)
    with pytest.raises(ValueError, match="finished PPO games"):
        store.save_game(game("7", outcome={"winner": True, "reason": "not a winner seat"}))


def test_experience_store_rejects_self_consistent_but_wrong_scheduler_assignment(tmp_path):
    from ptcg_lab.learning_mind.schema import identity_hash

    store = PPOExperienceStore(tmp_path / "run", settings=settings())
    altered = game("5")
    altered["schedule"]["ownArchetype"] = "mega-lucario"
    altered["gameId"] = identity_hash({"schedulerVersion": "ppo-training-scheduler-v1",
        **{key: value for key, value in altered["schedule"].items() if key != "gameId"}})
    altered["schedule"]["gameId"] = altered["gameId"]
    with pytest.raises(ValueError, match="differs from the frozen scheduler identity"):
        store.save_game(altered)


def test_experience_store_rejects_decisions_from_historical_opponent_seat(tmp_path):
    store = PPOExperienceStore(tmp_path / "run", settings=settings())
    historical_game = game("5")
    assert historical_game["schedule"]["policyFamily"] == "historical"
    assert historical_game["schedule"]["learnerSeats"] == [1]
    opponent_decision = decision(0)
    with pytest.raises(ValueError, match="non-learner seat"):
        store.save_game(game("5", decisions=[opponent_decision]))


def test_ppo_trace_conversion_requires_the_exact_frozen_scheduler_assignment():
    from ptcg_lab.learning_mind.schema import identity_hash

    altered = game("5")
    altered["schedule"]["opponentPolicy"] = "current"
    altered["gameId"] = identity_hash({"schedulerVersion": "ppo-training-scheduler-v1",
        **{key: value for key, value in altered["schedule"].items() if key != "gameId"}})
    altered["schedule"]["gameId"] = altered["gameId"]
    with pytest.raises(ValueError, match="differs from the frozen scheduler identity"):
        ppo_records_from_game(altered, behavior_policy_hash=BEHAVIOR_HASH,
            feature_schema_hash=SCHEMA_HASH, scheduler_settings=settings())


def test_finished_and_unfinished_games_map_to_seat_relative_terminal_ppo_traces():
    from ptcg_lab.learning_mind.encoding import encode_decision
    from ptcg_lab.learning_mind.training import eligible_ppo_records
    from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
    from test_learning_mind_representation import observation

    decisions = []
    for actor, decision_index, old_value in ((0, 0, .1), (1, 1, .3), (0, 2, .2)):
        view = observation()
        view["playerId"] = view["decisionPlayer"] = actor
        snapshot = ObservableHistoryTracker(actor).update(view)
        encoded = encode_decision(view, snapshot)
        decisions.append({"actor": actor, "decisionIndex": decision_index,
            "observation": view, "tracker": snapshot, "selectedAction": 0,
            "actionClassCount": len(encoded.action_classes), "oldLogProb": -.2,
            "oldValue": old_value, "behaviorPolicyHash": BEHAVIOR_HASH,
            "featureIdentityHash": encoded.identity})
        decisions[-1]["featureSchemaHash"] = SCHEMA_HASH
    feature_hash = SCHEMA_HASH
    finished = game("9", decisions=decisions)
    rows = ppo_records_from_game(finished, behavior_policy_hash=BEHAVIOR_HASH,
                                 feature_schema_hash=feature_hash, scheduler_settings=settings())
    assert [row["actor"] for row in rows] == [0, 0, 1]
    assert [row["reward"] for row in rows] == [0, 1, -1]
    assert [rows[1]["return"], rows[2]["return"]] == [1., -1.]
    assert all(row["episodeStatus"] == "finished" for row in rows)

    unfinished = {**finished, "status": "truncated", "outcome": None}
    discarded = ppo_records_from_game(unfinished, behavior_policy_hash=BEHAVIOR_HASH,
                                      feature_schema_hash=feature_hash, scheduler_settings=settings())
    assert all(row["reward"] == 0 for row in discarded)
    assert eligible_ppo_records(discarded) == []
