from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.analysis import analyze_observation, review_decision, select_frame, selected_action_id
from ptcg_lab.dataset import complete_games, examples, game_split
from ptcg_lab.engine import EngineClient, EngineError
from ptcg_lab.evaluation import confidence_interval, promotion_decision
from ptcg_lab.features import action_features, card_tokens, deck_beliefs, resource_features
from ptcg_lab.guides import import_guide, retrieve
from ptcg_lab.selfplay import Agent
from ptcg_lab.storage import Store

from conftest import make_replay


def test_persistent_worker_request_ids(tmp_path):
    code = "import sys,json,os\nfor line in sys.stdin:\n r=json.loads(line);print(json.dumps({'id':r['id'],'result':{'pid':os.getpid(),'method':r['method']}}),flush=True)"
    with EngineClient(tmp_path, command=[sys.executable, "-u", "-c", code]) as engine:
        first = engine.request("first")
        second = engine.request("second")
        assert first["pid"] == second["pid"]
        assert second["method"] == "second"


def test_worker_timeout_is_interruption(tmp_path):
    with EngineClient(tmp_path, timeout=.05, command=[sys.executable, "-u", "-c", "import time;time.sleep(2)"]) as engine:
        with pytest.raises(EngineError, match="interrupted, not a draw"):
            engine.request("slow")


def test_store_rejects_path_traversal_and_fabricated_outcomes(tmp_path, observation):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.get("replays", "../secret")
    replay = make_replay(observation, status="truncated")
    replay["outcome"] = {"winner": None, "reason": "decision_limit"}
    with pytest.raises(ValueError, match="unfinished"):
        store.save_replay(replay)


def test_disk_cap_is_enforced(tmp_path):
    store = Store(tmp_path, max_bytes=20)
    with pytest.raises(RuntimeError, match="disk cap"):
        store.put("guides", "test", {"text": "a" * 100})


def test_unknown_terminal_is_not_draw(tmp_path, observation):
    store = Store(tmp_path)
    replay = make_replay(observation)
    replay["outcome"] = {"winner": None, "reason": "engine-ended-without-winner"}
    with pytest.raises(ValueError, match="not a genuine rules draw"):
        store.save_replay(replay)
    replay["outcome"]["reason"] = "rules-draw"
    store.save_replay(replay)
    assert examples(complete_games(store))[0]["expectedResult"] == .5


def test_analysis_never_uses_opponent_hidden_hand(observation):
    changed = copy.deepcopy(observation)
    changed["players"][1]["hand"] = [{"id": "secret", "name": "Boss's Orders", "kind": "trainer"}]
    np.testing.assert_array_equal(resource_features(observation), resource_features(changed))
    np.testing.assert_array_equal(card_tokens(observation), card_tokens(changed))
    assert analyze_observation(observation, []) == analyze_observation(changed, [])
    assert Agent("heuristic", 42).choose(observation) == Agent("heuristic", 42).choose(changed)


def test_analysis_withholds_untrained_probabilities(observation):
    result = analyze_observation(observation, [])
    assert result["evaluation"]["status"] == "heuristic"
    assert result["evaluation"]["expectedResult"] is None
    assert result["evaluation"]["winProbability"] is None
    assert result["evaluation"]["calibrated"] is False
    assert all(item["visits"] == 0 for item in result["alternatives"])


def test_analysis_boundary_selects_only_predecision_view(observation):
    replay = make_replay(observation)
    selected = copy.deepcopy(select_frame(replay, 0, 0))
    replay["frames"][1]["observations"][0]["players"][0]["hand"] = [{"id": "future"}]
    replay["frames"][0]["observations"][1]["players"][1]["hand"] = [{"id": "secret"}]
    assert selected == select_frame(replay, 0, 0)


def test_decision_review_ignores_future_frames_and_outcome(observation):
    replay = make_replay(observation)
    alternatives = [{"actionId": "action-0", "expectedResult": .4, "visits": 3, "uncertainty": .2},
                    {"actionId": "action-1", "expectedResult": .6, "visits": 4, "uncertainty": .15}]
    original = review_decision(alternatives, selected_action_id(replay, 0, 0))
    replay["outcome"] = {"winner": 1, "reason": "rules-terminal"}
    replay["frames"][1]["observations"][0]["players"][0]["hand"] = [{"id": "future-secret"}]
    replay["frames"].append({"decisionIndex": 99, "actor": 0, "action": {"id": "future-move"}, "observations": []})
    assert review_decision(alternatives, selected_action_id(replay, 0, 0)) == original
    assert original["status"] == "experimental"
    assert original["opportunityLoss"] == pytest.approx(.2)
    assert any("ten samples" in warning for warning in original["warnings"])
    assert any("luck is not used" in warning for warning in original["warnings"])
    assert "blunder" not in original


def test_decision_review_requires_sampled_played_move_and_comparison():
    alternatives = [{"actionId": "best", "expectedResult": .8, "visits": 8, "uncertainty": .1},
                    {"actionId": "played", "expectedResult": None, "visits": 0, "uncertainty": None}]
    assert review_decision(alternatives, "played")["status"] == "unavailable"
    assert review_decision(alternatives, None)["status"] == "unavailable"
    assert review_decision(alternatives[:1], "best")["status"] == "unavailable"


def test_decision_review_same_estimator_has_zero_difference_uncertainty():
    alternatives = [{"actionId": "played", "expectedResult": .8, "visits": 20, "uncertainty": .15},
                    {"actionId": "other", "expectedResult": .4, "visits": 20, "uncertainty": .1}]
    review = review_decision(alternatives, "played")
    assert review["status"] == "experimental"
    assert review["bestTestedActionId"] == review["playedActionId"]
    assert review["opportunityLoss"] == 0
    assert review["differenceStandardError"] == 0


def test_closed_pool_beliefs_warn_about_unknown_cards(observation):
    decks = [{"id": "crustle", "archetype": "Crustle", "cards": [{"cardId": "DRI-007", "count": 4}]}]
    beliefs, warnings = deck_beliefs(observation, decks)
    assert beliefs == [{"archetype": "Crustle", "probability": 1.0}]
    assert "pool" in warnings[0]
    changed = copy.deepcopy(observation)
    changed["players"][1]["active"]["card"]["id"] = "unknown"
    assert deck_beliefs(changed, decks)[0] == []


def test_policy_features_distinguish_prompt_targets_without_opaque_ids():
    first = {"id": "1:2", "type": "prompt", "label": "Choose opponent bench 1: Crustle", "target": "bench1"}
    second = {"id": "1:3", "type": "prompt", "label": "Choose opponent bench 2: Dragapult ex", "target": "bench2"}
    assert not np.array_equal(action_features(first), action_features(second))
    assert np.array_equal(action_features(first), action_features({**first, "id": "different"}))


def test_only_completed_games_become_labels(tmp_path, observation):
    store = Store(tmp_path)
    store.save_replay(make_replay(observation, 0, "finished"))
    store.save_replay(make_replay(observation, 1, "truncated"))
    store.save_replay(make_replay(observation, 2, "error"))
    held_out = make_replay(observation, 3, "finished")
    held_out["evaluationExperiment"] = "held-out-matchup-test"
    store.save_replay(held_out)
    games = complete_games(store)
    assert [game["id"] for game in games] == ["test-0"]
    assert examples(games)[0]["expectedResult"] == 1
    assert examples([held_out]) == []


def test_evaluation_provenance_cannot_overwrite_training_replay(tmp_path, observation):
    store = Store(tmp_path)
    original = make_replay(observation, 0)
    original_id = store.save_replay(original)
    held_out = copy.deepcopy(original)
    held_out["evaluationExperiment"] = "new-evaluation"
    held_out_id = store.save_replay(held_out)
    assert original_id != held_out_id
    assert "evaluationExperiment" not in store.get("replays", original_id)
    assert store.get("replays", held_out_id)["evaluationExperiment"] == "new-evaluation"
    assert len(complete_games(store)) == 1


def test_game_family_split_prevents_seat_pair_leakage(observation):
    games = [make_replay(observation, index) for index in range(20)]
    duplicate = copy.deepcopy(games[0])
    duplicate["id"] = "seat-swapped"
    duplicate["decks"].reverse()
    games.append(duplicate)
    splits = game_split(games)
    family_sets = [{(game["seed"], tuple(sorted(game["decks"]))) for game in group} for group in splits.values()]
    assert not family_sets[0] & family_sets[1]
    assert not family_sets[0] & family_sets[2]
    assert not family_sets[1] & family_sets[2]
    assert sum(len(group) for group in splits.values()) == 21
    original_membership = {game["id"]: split for split, group in splits.items() for game in group}
    expanded = game_split(games + [make_replay(observation, index) for index in range(20, 100)])
    new_membership = {game["id"]: split for split, group in expanded.items() for game in group}
    assert all(new_membership[identifier] == split for identifier, split in original_membership.items())


def test_inconclusive_results_never_promote():
    small = confidence_interval([1, 1])
    eligible, reasons = promotion_decision(small, {"crustle / dragapult": small}, 0)
    assert not eligible
    assert reasons
    assert small["lower"] < .5
    assert promotion_decision(confidence_interval([1] * 100), {"match": confidence_interval([1] * 100)}, 1)[0] is False


def test_evaluation_covers_both_deck_assignments_and_seats(tmp_path, observation, monkeypatch):
    import ptcg_lab.evaluation as evaluation_module
    from ptcg_lab.config import Settings
    calls = []

    class FakeEngine:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method):
            return [{"id": "crustle"}, {"id": "dragapult"}] if method == "decks" else {"version": "test"}

    def fake_play(engine, *, decks, seed, policies, max_decisions):
        calls.append((tuple(decks), policies))
        game = make_replay(observation, len(calls))
        game["decks"], game["seed"], game["startingPlayer"] = decks, seed, 0
        return game

    monkeypatch.setattr(evaluation_module, "EngineClient", FakeEngine)
    monkeypatch.setattr(evaluation_module, "play_game", fake_play)
    report = evaluation_module.evaluate(Settings(tmp_path, tmp_path / "data"), seeds=1)
    assert len(calls) == 8
    assignments = {(decks[policies.index("heuristic")], decks[policies.index("random")]) for decks, policies in calls}
    assert assignments == {("crustle", "crustle"), ("crustle", "dragapult"), ("dragapult", "crustle"), ("dragapult", "dragapult")}
    assert all(counts == {"candidateFirst": 1, "opponentFirst": 1, "unknown": 0} for counts in report["firstPlayerCounts"].values())
    assert not report["promotionEligible"]


def test_guide_import_is_attributed_and_not_training_label(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("Preserve recovery against Crustle.\n\nTrack the opponent's remaining non-ex attackers.")
    store = Store(tmp_path / "data")
    imported = import_guide(store, guide, title="Test strategy", author="Tester", source="user supplied", matchup="Crustle")
    assert imported["reviewStatus"] == "unreviewed"
    passages = retrieve(store, "Crustle recovery")
    assert passages[0]["source"] == "user supplied"
    assert passages[0]["author"] == "Tester"
    assert passages[0]["chunkId"].startswith(imported["id"])


def test_api_validation_and_replay_lookup(tmp_path):
    from fastapi.testclient import TestClient
    from ptcg_lab.api import create_app
    from ptcg_lab.config import Settings
    with TestClient(create_app(Settings(tmp_path, tmp_path / "data"))) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/replays").json() == {"replays": []}
        assert client.post("/api/games", json={"decks": ["a", "b"], "maxDecisions": 3001}).status_code == 422
        assert client.get("/api/replays/missing").status_code == 404


def test_saved_position_has_only_selected_observation(tmp_path, observation):
    from fastapi.testclient import TestClient
    from ptcg_lab.api import create_app
    from ptcg_lab.config import Settings
    app = create_app(Settings(Path.cwd(), tmp_path / "data"))
    replay = make_replay(observation)
    replay["frames"][0]["observations"][1]["players"][1]["hand"] = [{"id": "opponent-secret"}]
    app.state.store.save_replay(replay)
    with TestClient(app) as client:
        saved = client.post("/api/positions", json={"replayId": replay["id"], "decisionIndex": 0,
                                                   "playerId": 0, "title": "Defensive choice"})
        assert saved.status_code == 201
        position = client.get(f"/api/positions/{saved.json()['id']}").json()
        assert position["observation"]["playerId"] == 0
        assert "frames" not in position
        assert "opponent-secret" not in json.dumps(position)
        assert "observation" not in client.get("/api/positions").json()["positions"][0]
        assert client.post("/api/analyze", json={"positionId": position["id"], "replayId": replay["id"]}).status_code == 422
        if (Path.cwd() / "packages/engine/dist/worker.cjs").exists():
            analysis = client.post("/api/analyze", json={"positionId": position["id"], "budgetMs": 20})
            assert analysis.status_code == 200
            assert analysis.json()["evaluation"]["status"] == "heuristic"
            assert analysis.json()["search"]["status"] == "unavailable"


def test_small_model_and_train_resume(tmp_path, observation):
    pytest.importorskip("torch")
    from ptcg_lab.model import PolicyResourceModel
    from ptcg_lab.training import predict, train
    assert sum(parameter.numel() for parameter in PolicyResourceModel().parameters()) < 2_000_000
    store = Store(tmp_path)
    for index in range(12):
        store.save_replay(make_replay(observation, index))
    report = train(store, epochs=1, max_positions=100)
    assert report["status"] == "completed"
    assert report["calibration"]["status"] == "insufficient_data"
    evaluation, scores = predict(Path(report["checkpoint"]), observation)
    assert evaluation["status"] == "trained"
    assert evaluation["winProbability"] is None
    assert len(scores) == len(observation["legalActions"])
    assert sum(component["value"] for component in evaluation["components"]) == pytest.approx(evaluation["score"], abs=1e-5)
    resumed = train(store, epochs=2, max_positions=100, resume=Path(report["checkpoint"]))
    assert resumed["id"] == report["id"]
    with pytest.raises(ValueError, match="hash differs"):
        train(store, epochs=3, seed=7, max_positions=100, resume=Path(report["checkpoint"]))
