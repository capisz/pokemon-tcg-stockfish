from __future__ import annotations

import copy

import pytest


@pytest.fixture
def observation():
    card = {"id": "DRI-007", "name": "Crustle", "kind": "pokemon", "hp": 150,
            "stage": "stage1", "prizeValue": 1, "attacks": [{"name": "Attack", "damage": "120", "cost": ["grass", "colorless"]}]}
    board = {"card": card, "damage": 0, "energy": ["grass", "colorless"], "tools": [], "conditions": []}
    return {"schemaVersion": 1, "playerId": 0, "decisionPlayer": 0, "turn": 3, "phase": "player-turn", "status": "running",
            "players": [{"id": 0, "name": "A", "active": copy.deepcopy(board), "bench": [], "hand": [], "handCount": 4,
                         "deckCount": 35, "prizesRemaining": 6, "discard": []},
                        {"id": 1, "name": "B", "active": copy.deepcopy(board), "bench": [], "hand": [], "handCount": 6,
                         "deckCount": 28, "prizesRemaining": 4, "discard": []}],
            "ownDeck": [{"cardId": "DRI-007", "name": "Crustle", "count": 4}],
            "legalActions": [{"id": "action-0", "type": "attack", "label": "Attack", "cardId": "DRI-007"},
                             {"id": "action-1", "type": "end", "label": "End turn"}],
            "history": [], "warnings": []}


def make_replay(observation, index=0, status="finished"):
    other = copy.deepcopy(observation)
    other["playerId"] = 1
    other["legalActions"] = []
    return {"schemaVersion": 1, "id": f"test-{index}", "seed": index, "decks": ["crustle", "dragapult"],
            "engineVersion": "test-fixture-only", "status": status,
            "trainingEligible": True, "deckRoles": ["main", "main"],
            "outcome": {"winner": index % 2, "reason": "test_fixture"} if status == "finished" else None,
            "frames": [{"decisionIndex": 0, "actor": 0, "action": observation["legalActions"][0],
                        "observations": [copy.deepcopy(observation), other]},
                       {"decisionIndex": 1, "actor": 0, "action": None, "observations": [copy.deepcopy(observation), other]}],
            "warnings": []}
