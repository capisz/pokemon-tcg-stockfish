import copy
import json
import threading
import time

from fastapi.testclient import TestClient

from ptcg_lab.config import Settings
from ptcg_lab.presentation import FrameStream, project_replay
from ptcg_lab.storage import Store
from conftest import make_replay
from test_matches import ScenarioEngine, ScenarioPool, REGISTRY, create, act, engine_turn, service


def test_replay_projection_redacts_other_views_private_choices_and_chance(observation):
    replay = make_replay(observation)
    replay["chance"] = [{"result": "private-shuffle-order"}]
    replay["frames"][0].update(actor=1, action={"id": "secret-card-id", "label": "Secret choice", "type": "choice"})
    replay["frames"][0]["observations"][1]["ownDeck"] = [{"cardId": "opponent-exact-list"}]
    replay["frames"][0]["observations"][0]["searchPosition"] = {"researchOnly": True}
    projected = project_replay(replay, 0)
    encoded = json.dumps(projected)
    for secret in ("secret-card-id", "Secret choice", "private-shuffle-order", "opponent-exact-list", "searchPosition", "observations"):
        assert secret not in encoded
    assert projected["schemaVersion"] == 2
    assert projected["frames"][1]["priorAction"]["label"].startswith("Opponent choice")
    assert replay["frames"][0]["action"]["id"] == "secret-card-id"


def test_stream_committed_cursor_hides_and_replaces_crash_tail(tmp_path, observation):
    stream = FrameStream(Store(tmp_path))
    frame = {"decisionIndex": 0, "actor": 0, "observation": observation, "priorAction": None}
    assert stream.append("game", frame, -1) == 0
    assert stream.append("game", {**frame, "decisionIndex": 99}, 0) == 1
    restarted = FrameStream(Store(tmp_path))
    assert len(restarted.read("game", after=-1, limit=100, committed=0, player_id=0, status="paused")["frames"]) == 1
    restarted.append("game", {**frame, "decisionIndex": 1}, 0)
    page = restarted.read("game", after=0, limit=1, committed=1, player_id=0, status="running")
    assert [f["decisionIndex"] for f in page["frames"]] == [1]
    assert page["nextCursor"] == 1 and not page["hasMore"]
    assert restarted.read("game", after=1, limit=100, committed=1, player_id=0, status="running")["frames"] == []


def test_match_stream_contains_every_accepted_engine_choice_and_idempotent_ack(service):
    initial = create(service)
    accepted = act(service, initial)
    assert act(service, initial) == accepted
    latest = engine_turn(service, initial["id"])
    page = service.frames(initial["id"])
    assert [f["cursor"] for f in page["frames"]] == [0, 1, 2, 3]
    assert [f["revision"] for f in page["frames"]] == [0, 1, 2, 3]
    assert latest["frameCursor"] == 3
    assert all(f["observation"]["playerId"] == 0 for f in page["frames"])
    assert "private-hand-1" not in json.dumps(page)
    paused = service.command(initial["id"], "pause", latest["revision"], "pause")
    assert service.frames(initial["id"], after=3)["frames"][0]["revision"] == paused["revision"]


def test_job_feed_publishes_durable_frames_before_completion(tmp_path, observation, monkeypatch):
    import ptcg_lab.api as module
    import ptcg_lab.resources as resources
    monkeypatch.setattr(resources, "process_memory", lambda: 1024**2)
    blocked, release = threading.Event(), threading.Event()

    class IncrementalEngine(ScenarioEngine):
        def request(self, method, params=None):
            if method == "choose":
                return self.request("observe")["legalActions"][0]
            if method == "step" and self.index == 1:
                blocked.set()
                assert release.wait(10)
            return super().request(method, params)

    engine = IncrementalEngine(observation)
    monkeypatch.setattr(module, "EnginePool", lambda *args: ScenarioPool(engine))
    app = module.create_app(Settings(root=tmp_path, data=tmp_path / "data", min_free_bytes=0))
    with TestClient(app) as client:
        try:
            response = client.post("/api/games", json={"decks": ["human", "engine-private-list"], "maxDecisions": 4})
            assert response.status_code == 202, response.text
            identifier = response.json()["id"]
            assert blocked.wait(5)
            page = client.get(f"/api/jobs/{identifier}/frames").json()
            assert page["status"] == "running"
            assert len(page["frames"]) == 2
            assert "private-hand-1" not in json.dumps(page)
        finally:
            release.set()
        for _ in range(100):
            job = client.get(f"/api/jobs/{identifier}").json()
            if job["status"] in {"completed", "failed"}: break
            time.sleep(.01)
        assert job["status"] == "completed", job
        tail = client.get(f"/api/jobs/{identifier}/frames?after=1").json()
        assert [f["cursor"] for f in tail["frames"]] == [2, 3, 4]
        assert tail["replayId"] == job["replayId"]
        assert client.post("/api/games", json={"decks": ["human", "heldout"]}).status_code == 422


def test_failed_match_ack_never_publishes_orphan_frame(service, monkeypatch):
    initial = create(service)
    original = service.store.put
    failed = False
    def fail_once(category, identifier, record):
        nonlocal failed
        if category == "private-matches" and not failed:
            failed = True
            raise OSError("disk failure")
        return original(category, identifier, record)
    monkeypatch.setattr(service.store, "put", fail_once)
    import pytest
    with pytest.raises(OSError): act(service, initial)
    assert service.frames(initial["id"])["nextCursor"] == 0
    paused = service.get(initial["id"])
    resumed = service.command(initial["id"], "resume", paused["revision"], "resume")
    assert resumed["frameCursor"] == 1
    assert service.frames(initial["id"])["frames"][1]["priorAction"] is None


def test_live_worker_reused_until_pause_then_reconstructs(service):
    from ptcg_lab.matches import MatchService
    from contextlib import contextmanager
    engines = [ScenarioEngine(service.pool.engine.base), ScenarioEngine(service.pool.engine.base)]
    class HoldingPool:
        clients = engines
        def __init__(self): self.available = list(engines)
        def acquire(self, **_): return self.available.pop(0)
        def release(self, engine): self.available.append(engine)
        @contextmanager
        def lease(self, **_):
            engine = self.acquire()
            try: yield engine
            finally: self.release(engine)
    pool = HoldingPool()
    live = MatchService(service.settings, service.store, pool, service.registry)
    try:
        initial = create(live)
        accepted = act(live, initial)
        assert len(pool.available) == 1
        assert sum(method == "reset" for method, _ in engines[0].calls) == 1
        paused = live.command(initial["id"], "pause", accepted["revision"], "pause")
        assert len(pool.available) == 2
        resumed = live.command(initial["id"], "resume", paused["revision"], "resume")
        assert resumed["observation"] == accepted["observation"]
        assert sum(method == "reset" for method, _ in engines[1].calls) == 1
        assert any(method == "step" for method, _ in engines[1].calls)
    finally:
        live.close()


def test_analysis_search_uses_guide_policy_priors_without_inventing_leaf_value(tmp_path, observation, monkeypatch):
    import ptcg_lab.api as module
    import ptcg_lab.training as training
    engine = ScenarioEngine(observation)
    monkeypatch.setattr(module, "EnginePool", lambda *args: ScenarioPool(engine))
    monkeypatch.setattr(module, "analyze_observation", lambda *args: {"warnings": [], "alternatives": [], "evaluation": {"status": "unavailable"}})
    monkeypatch.setattr(training, "load_model", lambda path: (None, {"modelKind": "policy-only"}))
    monkeypatch.setattr(training, "predict", lambda *args: ({"status": "unavailable"}, [3., -1.]))
    monkeypatch.setattr(training, "export_portable_value", lambda *args: None)
    model = tmp_path / "model.pt"
    model.write_bytes(b"transport-test-only")
    app = module.create_app(Settings(root=tmp_path, data=tmp_path / "data", min_free_bytes=0, analysis_model=model))
    replay = make_replay(observation)
    app.state.store.save_replay(replay)
    with TestClient(app) as client:
        response = client.post("/api/analyze", json={"replayId": replay["id"], "playerId": 0, "decisionIndex": 0})
        assert response.status_code == 200, response.text
        params = next(params for method, params in engine.calls if method == "search")
        assert params["method"] == "ismcts" and "leafModel" not in params
        assert params["rootPriors"][0]["probability"] > .98
        assert response.json()["evaluation"]["status"] == "unavailable"
