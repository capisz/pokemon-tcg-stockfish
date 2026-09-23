from __future__ import annotations

import pytest

from ptcg_lab.learning_mind import fresh_collection
from ptcg_lab.learning_mind.fresh_collection import actor_only_replay, game_schedule


def test_fresh_game_schedule_is_seeded_collision_free_and_balanced():
    first = game_schedule(games_per_matchup=4)
    assert first == game_schedule(games_per_matchup=4)
    assert len({row["seed"] for row in first}) == 12
    assert len({row["replayId"] for row in first}) == 12
    for opponent in ("crustle", "dragapult"):
        cell = [row for row in first if row["cellId"] == f"raging-bolt-vs-{opponent}"]
        assert {row["decks"].index("raging-bolt") for row in cell} == {0, 1}
        assert {row["firstPlayer"] for row in cell} == {0, 1}
        assert {(row["decks"].index("raging-bolt"), row["firstPlayer"]) for row in cell} == {
            (0, 0), (0, 1), (1, 0), (1, 1)}
    assert {row["firstPlayer"] for row in first if row["cellId"].endswith("raging-bolt")} == {0, 1}
    mirror = [row for row in game_schedule(games_per_matchup=2)
              if row["cellId"] == "raging-bolt-vs-raging-bolt"]
    assert {row["firstPlayer"] for row in mirror} == {0, 1}


def test_fresh_seed_namespace_changes_with_collector_version(monkeypatch):
    current = game_schedule(games_per_matchup=4)
    monkeypatch.setattr(fresh_collection, "COLLECTOR_VERSION", "fresh-actor-position-collector-next")
    next_version = game_schedule(games_per_matchup=4)
    assert {row["seed"] for row in current}.isdisjoint({row["seed"] for row in next_version})


def test_collection_epoch_is_deterministic_collision_free_and_legacy_default_stable():
    legacy = game_schedule(games_per_matchup=4)
    assert legacy == game_schedule(games_per_matchup=4, collection_namespace="main")
    later = game_schedule(games_per_matchup=4, collection_namespace="coverage-2026-09b")
    assert later == game_schedule(games_per_matchup=4, collection_namespace="coverage-2026-09b")
    assert {row["seed"] for row in legacy}.isdisjoint({row["seed"] for row in later})
    assert {row["replayId"] for row in legacy}.isdisjoint({row["replayId"] for row in later})
    assert all(row["collectionNamespace"] == "coverage-2026-09b" for row in later)


@pytest.mark.parametrize("namespace", ["", "has spaces", "x/../../tmp", "x" * 33, "époque"])
def test_collection_epoch_requires_safe_ascii_slug(namespace):
    with pytest.raises(ValueError, match="collection namespace"):
        game_schedule(games_per_matchup=1, collection_namespace=namespace)


@pytest.mark.parametrize("policy", ["typescript-heuristic", "python-heuristic"])
def test_schedule_freezes_policy_and_game_boundary(policy):
    games = game_schedule(games_per_matchup=2, policy=policy)
    assert len(games) == 6
    assert all(row["policy"] == policy for row in games)
    assert all(row["firstPlayer"] in (0, 1) and 0 <= row["seed"] < 2**32 for row in games)


def test_actor_only_replay_drops_other_seat_private_view_and_chance():
    replay = {"schemaVersion": 1, "chance": [{"secret": "hidden"}], "frames": [
        {"decisionIndex": 0, "actor": 0, "observations": [
            {"playerId": 0, "decisionPlayer": 0, "private": "actor"},
            {"playerId": 1, "decisionPlayer": 0, "private": "opponent"}]},
        {"decisionIndex": 1, "actor": 1, "observations": [
            {"playerId": 0, "decisionPlayer": 1, "private": "opponent"},
            {"playerId": 1, "decisionPlayer": 1, "private": "actor"}]},
    ]}
    identity = {"engineVersion": "test", "engineBuildHash": "build"}
    schedule = {"replayId": "fresh-v1-test", "seed": 7, "decks": ["raging-bolt", "crustle"],
                "policy": "typescript-heuristic", "firstPlayer": 0}
    cleaned = actor_only_replay(replay, schedule=schedule, run_id="run", engine_identity=identity)
    assert cleaned["chance"] == []
    assert cleaned["frames"][0]["observations"] == [{"playerId": 0, "decisionPlayer": 0, "private": "actor"}, None]
    assert cleaned["frames"][1]["observations"] == [None, {"playerId": 1, "decisionPlayer": 1, "private": "actor"}]
    assert cleaned["trainingEligible"] is False
    assert replay["frames"][0]["observations"][1]["private"] == "opponent"


def test_actor_only_replay_rejects_a_mismatched_actor_view():
    schedule = {"replayId": "fresh-v1-test", "seed": 7, "decks": ["raging-bolt", "crustle"],
                "policy": "typescript-heuristic", "firstPlayer": 0}
    replay = {"frames": [{"actor": 1, "observations": [
        {"playerId": 0, "decisionPlayer": 1}, {"playerId": 0, "decisionPlayer": 1}]}]}
    with pytest.raises(ValueError, match="does not match"):
        actor_only_replay(replay, schedule=schedule, run_id="run",
                          engine_identity={"engineVersion": "test", "engineBuildHash": "build"})


def test_fresh_collector_checkpoints_games_and_resume_rejects_identity_drift(tmp_path, monkeypatch):
    identity = {"engine_build_hash": "build", "identityHash": "identity"}
    monkeypatch.setattr(fresh_collection, "runtime_identity",
                        lambda _root: type("Identity", (), {"record": lambda self: identity})())

    class FakeEngine:
        def __init__(self, _root, timeout=300):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def request(self, method, _params=None):
            self.calls.append(method)
            if method == "health":
                return {"engineVersion": "test-engine", "engineBuildHash": "build"}
            if method == "run":
                return {"status": "error", "outcome": None, "frames": [], "warnings": []}
            raise AssertionError(f"unexpected engine request: {method}")

    instances = []

    def engine_factory(*args, **kwargs):
        instance = FakeEngine(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(fresh_collection, "EngineClient", engine_factory)
    output = tmp_path / "fresh-run"
    first = fresh_collection.collect_fresh_positions(root=tmp_path, output=output, games_per_matchup=1)
    assert first["completedGames"] == first["scheduledGames"] == 3
    assert first["statusCounts"] == {"finished": 0, "truncated": 0, "error": 3}
    assert len(first["games"]) == 3
    requests_after_first_run = sum(instance.calls.count("run") for instance in instances)
    second = fresh_collection.collect_fresh_positions(root=tmp_path, output=output, games_per_matchup=1)
    assert second["manifestHash"] == first["manifestHash"]
    assert sum(instance.calls.count("run") for instance in instances) == requests_after_first_run

    changed_identity = {**identity, "identityHash": "changed"}
    monkeypatch.setattr(fresh_collection, "runtime_identity",
                        lambda _root: type("Identity", (), {"record": lambda self: changed_identity})())
    with pytest.raises(ValueError, match="identity/configuration drift"):
        fresh_collection.collect_fresh_positions(root=tmp_path, output=output, games_per_matchup=1)
