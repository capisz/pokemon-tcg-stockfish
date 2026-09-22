from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from ptcg_lab.decision_guard import VERSION as GUARD_VERSION


PILOT_PATH = Path(__file__).parents[2] / "research/strategy_baseline/pilot.py"
SPEC = importlib.util.spec_from_file_location("strategy_baseline_pilot", PILOT_PATH)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class ChoosingAgent:
    def __init__(self, choice_operation: str):
        self.choice_operation = choice_operation
        self.observations = []

    def choose(self, observation: dict) -> str:
        self.observations.append(copy.deepcopy(observation))
        return next(
            action["id"]
            for action in observation["legalActions"]
            if action.get("choiceOperation") == self.choice_operation
        )


def staged_observation(observation: dict) -> dict:
    result = copy.deepcopy(observation)
    result["legalActions"] = [
        {"id": "revision-7:finish", "type": "prompt", "label": "Finish", "choiceOperation": "finish"},
        {"id": "revision-7:undo", "type": "prompt", "label": "Undo last selection", "choiceOperation": "undo"},
    ]
    return result


def test_p3_receives_only_forward_staged_choice_and_resolves_original_action(observation):
    original = staged_observation(observation)
    agent = ChoosingAgent("finish")
    guard_state = {}

    action, event = pilot.select_policy_action("P3", agent, original, guard_state)

    assert [item["choiceOperation"] for item in agent.observations[0]["legalActions"]] == ["finish"]
    assert action is original["legalActions"][0]
    assert event == {
        "version": GUARD_VERSION,
        "filtered": True,
        "searchTargetCreated": False,
    }
    assert guard_state["positions"]
    assert len(original["legalActions"]) == 2


def test_p3_mandatory_action_is_retained_without_fake_terminal(observation):
    original = staged_observation(observation)
    original["legalActions"] = original["legalActions"][:1]

    action, event = pilot.select_policy_action("P3", ChoosingAgent("finish"), original, {})

    assert action is original["legalActions"][0]
    assert event["filtered"] is False
    assert "status" not in event


def test_run_python_game_records_guard_metadata_and_never_searches(monkeypatch, observation):
    running = staged_observation(observation)

    class FakeEngineClient:
        def __init__(self, root: Path, timeout: int):
            self.finished = False

        def request(self, method: str, params: dict | None = None):
            if method == "reset":
                return {"status": "running", "observation": copy.deepcopy(running)}
            if method == "step":
                assert params == {"actionId": "revision-7:finish"}
                self.finished = True
                finished = copy.deepcopy(running)
                finished["status"] = "finished"
                return {"status": "finished", "observation": finished}
            if method == "replay":
                return {"status": "finished", "outcome": {"winner": 0, "reason": "rules-terminal"}}
            raise AssertionError(f"Unexpected engine method: {method}")

        def close(self):
            pass

    class FakeAgent(ChoosingAgent):
        def __init__(self, policy: str, seed: int, allow_experimental: bool):
            assert allow_experimental
            super().__init__("finish")

    def unexpected_search(*args, **kwargs):
        raise AssertionError("A guarded P3 decision must not create a search target")

    monkeypatch.setattr(pilot, "EngineClient", FakeEngineClient)
    monkeypatch.setattr(pilot, "Agent", FakeAgent)
    monkeypatch.setattr(pilot, "search_choice", unexpected_search)

    replay, details = pilot.run_python_game(
        Path("."), {"decks": ["crustle", "dragapult"], "firstPlayer": 0}, 11, "P3", 5,
        Path("guide.pt"),
    )

    assert replay["status"] == "finished"
    assert details["guard"]["version"] == GUARD_VERSION
    assert details["guard"]["filteredDecisions"] == 1
    assert details["guard"]["decisions"] == [
        {"decisionIndex": 0, "actor": 0, "filtered": True, "searchTargetCreated": False}
    ]
    assert details["search"]["searchAttempts"] == 0
