from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from ptcg_lab.engine import EngineError
from ptcg_lab.learning_games import run_game, restore_game
from ptcg_lab.presentation import FrameStream
from ptcg_lab.storage import Store, digest


def test_exhausted_worker_budget_raises_without_touching_journal_or_reacquiring(tmp_path):
    store = Store(tmp_path)
    store.put("learning-games", "game", {"id": "game", "status": "running", "workerFailures": 2,
                                          "actions": [{"id": "accepted-action"}], "frameCursor": 1})
    path = store.location("learning-games", "game")
    before = path.read_bytes()
    class Pool:
        def acquire(self, **kwargs):
            raise AssertionError("Must not silently start another automatic retry")
    service = SimpleNamespace(store=store, pool=Pool())
    with pytest.raises(EngineError, match="retry budget.*explicitly resume"):
        run_game(service, "run", "game", 0, None)
    assert path.read_bytes() == before


def test_reconstruction_refuses_changed_engine_before_applying_any_actions():
    class Engine:
        def __init__(self):
            self.calls = []
        def request(self, method, params=None):
            self.calls.append(method)
            return {"engineVersion": "changed", "engineBuildHash": "changed-build"}
    engine = Engine()
    with pytest.raises(ValueError, match="Engine build changed"):
        restore_game(engine, {"actions": [{"id": "never-apply"}]},
                     {"engineVersion": "frozen", "engineBuildHash": "frozen-build"}, lambda: None)
    assert engine.calls == ["health"]


def test_frame_written_before_failed_journal_commit_is_not_published_and_is_replaceable(tmp_path, observation, monkeypatch):
    from ptcg_lab.learning import LearningService
    store = Store(tmp_path)
    stream = FrameStream(store)
    service = SimpleNamespace(store=store, stream=stream,
                              save_game=lambda game: store.put("learning-games", game["id"], game))
    other = copy.deepcopy(observation)
    other.update(playerId=1, legalActions=[])
    views = [observation, other]
    game = {"id": "game", "actions": [], "frameCursor": -1}
    LearningService.commit_frame(service, game, views, None, 0)
    before = store.get("learning-games", "game")
    game["actions"].append({"id": "not-yet-acknowledged"})
    changed = copy.deepcopy(views)
    changed[0]["turn"] += 1
    def interrupted(_):
        raise OSError("Synthetic journal interruption after durable frame append")
    service.save_game = interrupted
    with pytest.raises(OSError, match="Synthetic journal"):
        LearningService.commit_frame(service, game, changed, observation["legalActions"][0], 0)
    persisted = store.get("learning-games", "game")
    assert persisted == before and persisted["observationHash"] == digest(views)
    visible = stream.read("game", after=-1, limit=100, committed=persisted["frameCursor"], player_id=0, status="paused")
    assert len(visible["frames"]) == 1 and not visible["hasMore"]
    # Resuming from the acknowledged journal overwrites the orphan suffix.
    persisted["actions"].append({"id": "replacement"})
    service.save_game = lambda value: store.put("learning-games", value["id"], value)
    LearningService.commit_frame(service, persisted, views, observation["legalActions"][1], 0)
    visible = stream.read("game", after=-1, limit=100, committed=persisted["frameCursor"], player_id=0, status="running")
    assert len(visible["frames"]) == 2 and visible["nextCursor"] == 1
