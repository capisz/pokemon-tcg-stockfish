from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ptcg_lab.config import Settings
from ptcg_lab.matches import MatchConflict, MatchService
from ptcg_lab.storage import Store, digest, file_digest
from ptcg_lab import teaching


REGISTRY = [
    {"id": "human", "archetype": "Crustle", "role": "main", "cards": [{"cardId": "public-0", "count": 60}]},
    {"id": "engine-private-list", "archetype": "Dragapult", "role": "main", "cards": [{"cardId": "secret-list", "count": 60}]},
    {"id": "heldout", "archetype": "Dragapult", "role": "heldout", "cards": []},
]


class ScenarioEngine:
    """Deterministic transport double, with two decisions sharing an engine turn."""
    def __init__(self, base):
        self.base = base
        self.index = 0
        self.seed = 0
        self.calls = []
        self.identity = {"engineVersion": "test-engine", "engineBuildHash": "test-build"}
        self.fail_step = False

    def request(self, method, params=None):
        params = params or {}
        self.calls.append((method, copy.deepcopy(params)))
        if method == "health": return copy.deepcopy(self.identity)
        if method == "decks": return copy.deepcopy(REGISTRY)
        if method == "reset":
            self.index = 0
            self.seed = params["seed"]
            return {"observation": self.request("observe", {"playerId": 0})}
        if method == "observe":
            observation = copy.deepcopy(self.base)
            actor = [0, 1, 1, 0, 0][self.index]
            player = params.get("playerId", actor)
            observation.update(playerId=player, decisionPlayer=actor, turn=2 if self.index in (1, 2) else 3,
                               status="finished" if self.index == 4 else "running")
            observation["legalActions"] = [] if player != actor or self.index == 4 else [
                {"id": f"move-{self.index}", "type": "attack", "label": "Legal move"},
                {"id": f"other-{self.index}", "type": "end", "label": "Other legal move"}]
            for side in observation["players"]:
                side["active"]["card"]["id"] = f"public-{side['id']}"
                side["hand"] = [{"id": f"private-hand-{side['id']}", "name": "Private card"}] if side["id"] == player else []
            observation["searchPosition"] = {"publicTestPosition": True}
            return observation
        if method == "step":
            if self.fail_step: raise RuntimeError("worker unavailable")
            if params["actionId"] not in {f"move-{self.index}", f"other-{self.index}"}:
                raise ValueError("stale engine choice")
            self.index += 1
            return {"observation": self.request("observe", {"playerId": 0})}
        if method == "search":
            return {"status": "complete", "alternatives": [{"actionId": f"other-{self.index}"}]}
        if method == "replay":
            return {"id": f"fake-replay-{len(self.calls)}", "seed": self.seed, "status": "finished" if self.index == 4 else "truncated",
                    "outcome": {"winner": 0, "reason": "rules-terminal"} if self.index == 4 else None,
                    "frames": [], "decks": ["human", "engine-private-list"]}
        raise AssertionError(method)


class ScenarioPool:
    def __init__(self, engine): self.engine = engine
    @contextmanager
    def lease(self, **kwargs): yield self.engine
    def close(self): pass


@pytest.fixture
def service(tmp_path, observation, monkeypatch):
    # The simulator is a transport double with no child processes. Control the
    # OS measurement while retaining the real ResourceGuard checks and pausing.
    import ptcg_lab.resources as resources
    monkeypatch.setattr(resources, "process_memory", lambda: 1024**2)
    engine = ScenarioEngine(observation)
    settings = Settings(root=tmp_path, data=tmp_path / "data", min_free_bytes=0)
    result = MatchService(settings, Store(settings.data), ScenarioPool(engine), lambda: copy.deepcopy(REGISTRY))
    yield result
    result.close()


def create(service, **kwargs):
    return service.create("human", "Dragapult", **kwargs)


def act(service, match, request="first"):
    return service.command(match["id"], "action", match["revision"], request, actionId=match["observation"]["legalActions"][0]["id"])


def engine_turn(service, identifier):
    service.pending.add(identifier)
    service._engine_turn(identifier)
    return service.get(identifier)


def test_match_ack_retry_stale_choice_and_restart(service):
    initial = create(service)
    accepted = act(service, initial)
    assert act(service, initial) == accepted
    with pytest.raises(MatchConflict, match="different operation"):
        service.command(initial["id"], "pause", 0, "first")
    with pytest.raises(MatchConflict, match="position changed"):
        act(service, initial, "another")
    assert len(service.get_private(initial["id"])["actions"]) == 1
    # A new service object reads only the durable journal, not the old worker state.
    restarted = MatchService(service.settings, service.store, ScenarioPool(ScenarioEngine(service.pool.engine.base)), service.registry)
    try:
        restarted.recover()
        paused = restarted.get(initial["id"])
        assert paused["status"] == "paused"
        restored = restarted.command(paused["id"], "resume", paused["revision"], "resume")
        assert restored["observation"] == accepted["observation"]
        assert restored["status"] == "active"
    finally:
        restarted.close()


def test_public_match_does_not_reveal_seed_private_journal_or_opponent(service):
    match = create(service)
    encoded = json.dumps(match)
    for secret in ("engine-private-list", "secret-list", "private-hand-1", '"seed"', '"actions"', '"engineIdentity"'):
        assert secret not in encoded
    assert service.store.list("replays") == []
    with pytest.raises(MatchConflict, match="whole match"):
        service.publish(match["id"])
    assert "observation" not in service.list()[0]


def test_lab_explicitly_discloses_opponent_list(service):
    match = create(service, known_list=True)
    assert match["opponentList"] == REGISTRY[1]["cards"]
    act(service, match)
    engine_turn(service, match["id"])
    searches = [params for method, params in service.pool.engine.calls if method == "search"]
    assert searches and all(p["knownOpponentDeckId"] == "human" for p in searches)


def test_budget_is_shared_across_engine_prompts_and_public_knowledge_is_separated(service, monkeypatch):
    import ptcg_lab.matches as module
    ticks = iter([0, .01, .20, .20, .21, .40])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    match = act(service, create(service, budget_ms=250))
    state = engine_turn(service, match["id"])
    searches = [params for method, params in service.pool.engine.calls if method == "search"]
    assert len(searches) == 1
    assert searches[0]["budgetMs"] == 240
    assert searches[0]["priorRevealedCards"] == ["public-0"]
    assert "knownOpponentDeckId" not in searches[0]
    assert state["engineTurnRemainingMs"] == 0
    assert state["observation"]["decisionPlayer"] == 0
    assert state["status"] == "active"


def test_frozen_model_supplies_live_policy_and_learned_search_priors(service, monkeypatch):
    import ptcg_lab.matches as module
    model = service.settings.data / "models/champion.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"frozen-model")
    calls = []
    class Policy:
        def __init__(self, policy, seed):
            calls.append(Path(policy).read_bytes())
            self.loaded = "frozen-policy"
        def choose(self, observation): return observation["legalActions"][0]["id"]
    monkeypatch.setattr(module, "Agent", Policy)
    import ptcg_lab.training as training
    monkeypatch.setattr(training, "predict", lambda path, observation, loaded: (0, [3, 1]))
    match = act(service, create(service))
    model.write_bytes(b"new-champion")
    state = engine_turn(service, match["id"])
    record = service.get_private(match["id"])
    assert state["modelVersion"] == file_digest(Path(record["policy"]))
    assert calls == [b"frozen-model"]
    assert [a["actionId"] for a in record["actions"]] == ["move-0", "move-1", "move-2"]
    searches = [params for method, params in service.pool.engine.calls if method == "search"]
    assert len(searches) == 2 and all(params["method"] == "ismcts" for params in searches)
    for params in searches:
        assert sum(item["probability"] for item in params["rootPriors"]) == pytest.approx(1)
        assert params["rootPriors"][0]["probability"] > params["rootPriors"][1]["probability"]


def test_worker_failure_pauses_without_accepting_move_or_inventing_result(service):
    match = create(service)
    service.pool.engine.fail_step = True
    with pytest.raises(RuntimeError, match="worker unavailable"):
        act(service, match)
    state = service.get_private(match["id"])
    assert state["status"] == "paused"
    assert state["revision"] == 0 and state["actions"] == [] and state["score"] == [0, 0]


def test_resource_limit_pauses_engine_without_result_or_new_decision(service, monkeypatch):
    import ptcg_lab.resources as resources
    match = act(service, create(service))
    monkeypatch.setattr(resources, "process_memory", lambda: service.settings.max_memory_bytes + 1)
    state = engine_turn(service, match["id"])
    assert state["status"] == "paused" and "memory target" in state["error"]
    assert state["revision"] == match["revision"] and state["score"] == [0, 0]
    assert len(service.get_private(match["id"])["actions"]) == 1


def test_polling_is_read_only_and_repeated_advance_does_not_duplicate_work(service, monkeypatch):
    match = act(service, create(service))
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    run = service._engine_turn
    calls = []
    def delayed(identifier):
        calls.append(identifier)
        started.set()
        assert release.wait(5)
        run(identifier)
        finished.set()
    monkeypatch.setattr(service, "_engine_turn", delayed)
    service.advance(match["id"])
    assert started.wait(5)
    try:
        for _ in range(5):
            assert service.get(match["id"])["revision"] == match["revision"]
            assert service.advance(match["id"])["thinking"]
        assert calls == [match["id"]]
    finally:
        release.set()
    assert finished.wait(5)
    assert service.get(match["id"])["revision"] == match["revision"] + 2


def test_failed_durable_write_cannot_acknowledge_an_action(service, monkeypatch):
    match = create(service)
    original = service.store.put
    calls = 0
    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 1: raise RuntimeError("disk full")
        return original(*args)
    monkeypatch.setattr(service.store, "put", fail_once)
    with pytest.raises(RuntimeError, match="disk full"):
        act(service, match)
    record = service.get_private(match["id"])
    assert record["actions"] == [] and record["revision"] == 0
    assert record["status"] == "paused" and record["score"] == [0, 0]


def test_changed_engine_cannot_resume(service):
    match = create(service)
    paused = service.command(match["id"], "pause", match["revision"], "pause")
    service.pool.engine.identity["engineBuildHash"] = "different-build"
    with pytest.raises(ValueError, match="recorded build"):
        service.command(match["id"], "resume", paused["revision"], "resume")
    assert service.get(match["id"])["status"] == "paused"


def test_incompatible_benchmark_can_end_incomplete_and_unlock_a_new_match(service):
    match = create(service, mode="benchmark")
    paused = service.command(match["id"], "pause", match["revision"], "pause")
    service.pool.engine.identity["engineBuildHash"] = "new-build"
    with pytest.raises(ValueError, match="recorded build"):
        service.command(match["id"], "resume", paused["revision"], "resume")
    assert service.benchmark_active()
    ended = service.command(match["id"], "abandon", paused["revision"], "end-incomplete")
    assert ended["status"] == "abandoned" and ended["score"] == [0, 0]
    assert service.get_private(match["id"])["abandonment"]["outcome"] is None
    assert not service.benchmark_active()
    with pytest.raises(MatchConflict, match="whole match"):
        service.publish(match["id"])
    assert create(service)["id"] != match["id"]


def test_concession_bo3_and_delayed_replay_publication(service):
    match = create(service, mode="benchmark")
    identifier = match["id"]
    one = service.command(identifier, "concede", match["revision"], "concede-1")
    assert one["status"] == "between-games" and one["score"] == [0, 1]
    assert one["nextStarterChooser"] == 0
    with pytest.raises(ValueError, match="who starts"):
        service.command(identifier, "next-game", one["revision"], "invalid-next")
    assert service.get(identifier)["status"] == "between-games"
    two = service.command(identifier, "next-game", one["revision"], "next", firstPlayer=0)
    complete = service.command(identifier, "concede", two["revision"], "concede-2")
    assert complete["status"] == "completed" and complete["score"] == [0, 2]
    assert not service.benchmark_active()
    result = service.publish(identifier)
    assert len(result["replayIds"]) == 2
    assert service.publish(identifier) == result
    for replay in service.store.list("replays"):
        assert replay["status"] == "truncated" and replay["outcome"] is None
        assert replay["concession"]["playerId"] == 0
        assert replay["evaluationExperiment"].startswith("human-")


def test_live_terminal_and_bookmark_stay_out_of_training(service):
    match = act(service, create(service))
    state = engine_turn(service, match["id"])
    bookmark = service.bookmark(match["id"], "A real decision")
    annotation = teaching.review(service.store, bookmark["id"], review_status="reviewed",
                                 acceptable_action_ids=["move-3"], reasoning="Preserves our only answer.")
    assert not annotation["trainingEligible"]
    finished = act(service, state, "finish")
    assert finished["status"] == "between-games" and finished["score"] == [1, 0]
    assert finished["nextStarterChooser"] == 1


def test_api_benchmark_locks_all_research_analysis_routes(tmp_path, observation, monkeypatch):
    import ptcg_lab.api as module
    engine = ScenarioEngine(observation)
    monkeypatch.setattr(module, "EnginePool", lambda *args: ScenarioPool(engine))
    app = module.create_app(Settings(root=tmp_path, data=tmp_path / "data", min_free_bytes=0))
    with TestClient(app) as client:
        created = client.post("/api/matches", json={"deckId": "human", "opponentArchetype": "Dragapult", "mode": "benchmark"})
        assert created.status_code == 201
        identifier = created.json()["id"]
        for path in ("/api/replays", "/api/replays/example", "/api/positions", "/api/positions/example",
                     "/api/experiments", "/api/teaching/queue", "/api/teaching/curriculum", "/api/teaching/example", "/api/guides"):
            assert client.get(path).status_code == 403, path
        assert client.post("/api/analyze", json={"positionId": "example"}).status_code == 403
        assert client.post(f"/api/matches/{identifier}/analyze").status_code == 403
        assert client.post("/api/guides/retrieve", json={"query": "Crustle"}).status_code == 403
        assert client.get(f"/api/matches/{identifier}").status_code == 200
        assert client.post(f"/api/matches/{identifier}/bookmarks", json={"title": "Review later"}).status_code == 201
