"""API boundaries: projected experimental replays, frozen policies, and locks."""
import json

import pytest
from fastapi.testclient import TestClient

from conftest import make_replay
from test_matches import ScenarioEngine, ScenarioPool
from ptcg_lab.config import Settings
from ptcg_lab.storage import file_digest


@pytest.fixture
def learning_app(tmp_path, observation, monkeypatch):
    import ptcg_lab.api as module
    engine = ScenarioEngine(observation)
    monkeypatch.setattr(module, "EnginePool", lambda *args: ScenarioPool(engine))
    # An explicit analysis override must not replace a learning game's frozen
    # per-seat checkpoints when that decision is revisited.
    override = tmp_path / "override.pt"
    override.write_bytes(b"wrong-policy-for-recorded-game")
    app = module.create_app(Settings(root=tmp_path, data=tmp_path / "data", min_free_bytes=0, analysis_model=override))
    replay = make_replay(observation)
    replay.update(dataTier="experimental", trainingEligible=False, learningRun="run-fixture")
    for frame in replay["frames"]:
        for player, view in enumerate(frame["observations"]):
            view["players"][player]["hand"] = [{"id": f"private-seat-{player}", "name": f"private-seat-{player}"}]
            view["ownDeck"] = [{"cardId": f"private-list-{player}", "count": 60}]
    app.state.learning.store.save_replay(replay)
    policies = []
    for player in (0, 1):
        model = tmp_path / f"frozen-seat-{player}.pt"
        model.write_bytes(f"checkpoint-{player}".encode())
        policies.append({"path": str(model), "hash": file_digest(model), "name": f"Seat {player}"})
    app.state.learning.store.put("learning-games", "game-fixture", {
        "id": "game-fixture", "policies": policies, "replayAvailable": True, "replayId": replay["id"]})
    with TestClient(app) as client:
        yield app, client, policies, engine


def test_experimental_replay_and_bookmark_remain_projected_and_quarantined(learning_app):
    app, client, _, _ = learning_app
    public_id = "experimental-game-fixture"
    replay = client.get(f"/api/replays/{public_id}?playerId=1")
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == public_id and replay.json()["dataTier"] == "experimental"
    encoded = replay.text
    assert "private-seat-0" not in encoded and "private-list-0" not in encoded and '"observations"' not in encoded
    assert "private-seat-1" in encoded
    saved = client.post("/api/positions", json={"replayId": public_id, "playerId": 1, "decisionIndex": 0, "title": "Reviewed perspective"})
    assert saved.status_code == 201, saved.text
    assert saved.json()["dataTier"] == "experimental" and saved.json()["trainingEligible"] is False
    position = app.state.store.get("positions", saved.json()["id"])
    assert position["experimentalAncestry"]["sourceReplayId"] == public_id
    assert "private-seat-0" not in json.dumps(position)


def test_analysis_uses_exact_frozen_seat_and_bookmark_policy(learning_app, monkeypatch):
    import ptcg_lab.api as module
    import ptcg_lab.training as training
    _, client, policies, _ = learning_app
    calls = []
    def analyze(view, registry, model, **kwargs):
        calls.append((view["playerId"], str(model), kwargs))
        return {"warnings": [], "alternatives": [], "evaluation": {"status": "unavailable"}}
    monkeypatch.setattr(module, "analyze_observation", analyze)
    monkeypatch.setattr(training, "load_model", lambda path, **kwargs: (None, {}))
    monkeypatch.setattr(training, "predict", lambda path, view, loaded, **kwargs: ({}, [1.] * len(view["legalActions"])))
    monkeypatch.setattr(training, "export_portable_value", lambda *args, **kwargs: None)
    for player in (0, 1):
        response = client.post("/api/analyze", json={"replayId": "experimental-game-fixture", "playerId": player, "decisionIndex": 0})
        assert response.status_code == 200, response.text
        assert response.json()["dataTier"] == "experimental"
        assert calls[-1] == (player, policies[player]["path"], {"allow_experimental": True})
    position = client.post("/api/positions", json={"replayId": "experimental-game-fixture", "playerId": 1, "decisionIndex": 0, "title": "Bookmark"}).json()
    response = client.post("/api/analyze", json={"positionId": position["id"]})
    assert response.status_code == 200, response.text
    assert calls[-1] == (1, policies[1]["path"], {"allow_experimental": True})


def test_changed_frozen_learning_policy_is_rejected(learning_app):
    from pathlib import Path
    _, client, policies, _ = learning_app
    Path(policies[0]["path"]).write_bytes(b"changed")
    response = client.post("/api/analyze", json={"replayId": "experimental-game-fixture", "playerId": 0, "decisionIndex": 0})
    assert response.status_code == 422
    assert "frozen policy changed" in response.text


def test_learning_routes_and_namespace_locked_during_benchmark(learning_app, monkeypatch):
    app, client, _, _ = learning_app
    monkeypatch.setattr(app.state.matches, "benchmark_active", lambda: True)
    routes = ["/api/learning/runs", "/api/learning/runs/run-fixture", "/api/learning/runs/run-fixture/games",
              "/api/learning/games/game-fixture/frames", "/api/learning/games/game-fixture/replay",
              "/api/replays/experimental-game-fixture"]
    assert all(client.get(path).status_code == 403 for path in routes)
    assert client.post("/api/learning/runs", json={"seed": 42}).status_code == 403
    assert client.post("/api/learning/runs/run-fixture/control/pause", json={"revision": 0, "requestId": "request"}).status_code == 403
    assert client.post("/api/analyze", json={"replayId": "experimental-game-fixture", "playerId": 0, "decisionIndex": 0}).status_code == 403


def test_experimental_model_cannot_start_benchmark(learning_app):
    _, client, _, _ = learning_app
    response = client.post("/api/matches", json={"deckId": "human", "opponentArchetype": "Dragapult", "mode": "benchmark", "modelId": "experimental-fake"})
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "Experimental learning checkpoints are available only in practice matches"


def test_learning_api_configuration_matches_protocol_and_validates_diagnostics():
    from pydantic import ValidationError
    from ptcg_lab.api import LearningRequest
    from ptcg_lab.learning_protocol import DEFAULTS
    assert LearningRequest().model_dump() == DEFAULTS
    assert LearningRequest(comparisonGameLimit=4, maxCycles=1, searchBudgetMs=0).comparisonGameLimit == 4
    for invalid in ({"comparisonGameLimit": -1}, {"maxCycles": True}, {"keepAwake": "true"}, {"seed": -1}):
        with pytest.raises(ValidationError):
            LearningRequest(**invalid)


def test_experimental_practice_freezes_model_and_forwards_explicit_inference_flag(learning_app, monkeypatch):
    from pathlib import Path
    import ptcg_lab.matches as matches
    import ptcg_lab.model_registry as registry
    import ptcg_lab.resources as resources
    import ptcg_lab.training as training
    app, client, policies, engine = learning_app
    calls = []
    monkeypatch.setattr(resources, "process_memory", lambda: 1024**2)
    monkeypatch.setattr(registry, "resolve_model", lambda store, identifier: Path(policies[0]["path"]))
    class FrozenAgent:
        loaded = (None, {})
        def __init__(self, policy, seed, **kwargs): calls.append(("agent", policy, kwargs))
        def choose(self, view): return view["legalActions"][0]["id"]
    def predict(path, view, loaded, **kwargs):
        calls.append(("predict", str(path), kwargs))
        return {}, [0.] * len(view["legalActions"])
    def portable(path, loaded, **kwargs):
        calls.append(("value", str(path), kwargs))
        return None
    monkeypatch.setattr(matches, "Agent", FrozenAgent)
    monkeypatch.setattr(training, "predict", predict)
    monkeypatch.setattr(training, "export_portable_value", portable)
    created = client.post("/api/matches", json={"deckId": "human", "opponentArchetype": "Dragapult", "mode": "practice", "modelId": "experimental-fake", "budgetMs": 1000})
    assert created.status_code == 201, created.text
    match = created.json()
    assert match["dataTier"] == "experimental"
    record = app.state.matches.get_private(match["id"])
    assert record["policy"] != policies[0]["path"]
    assert file_digest(Path(record["policy"])) == policies[0]["hash"]
    accepted = client.post(f"/api/matches/{match['id']}/actions", json={"revision": match["revision"], "requestId": "human-action", "actionId": match["observation"]["legalActions"][0]["id"]})
    assert accepted.status_code == 200, accepted.text
    app.state.matches._engine_turn(match["id"])
    latest = app.state.matches.get(match["id"])
    assert latest["status"] == "active" and latest["observation"]["decisionPlayer"] == 0
    assert {name for name, _, _ in calls} == {"agent", "predict", "value"}
    assert all(kwargs == {"allow_experimental": True} and path == record["policy"] for _, path, kwargs in calls)
    assert any(method == "search" and params["rootPriors"] for method, params in engine.calls)
