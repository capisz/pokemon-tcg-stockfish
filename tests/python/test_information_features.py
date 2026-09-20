from copy import deepcopy

from ptcg_lab.features import deck_beliefs


def view():
    return {"playerId": 0, "players": [
        {"id": 0, "active": None, "bench": [], "discard": [], "hand": []},
        {"id": 1, "active": {"card": {"id": "answer", "name": "Revealed answer"}},
         "bench": [], "discard": [], "hand": []},
    ]}


def deck(identifier, role, count=1, archetype="Crustle"):
    return {"id": identifier, "role": role, "archetype": archetype,
            "cards": [{"cardId": "answer", "count": count}]}


def test_heldout_and_historical_lists_cannot_resolve_unknown_opponent():
    observation = view()
    registry = [deck("main", "main", 0), deck("reserved", "heldout"), deck("old", "historical")]
    beliefs, warnings = deck_beliefs(observation, registry)
    assert beliefs == []
    assert any("unsupported" in warning for warning in warnings)


def test_adding_secret_test_manifest_does_not_change_prior():
    registry = [deck("main", "main"), deck("training", "training-variant", 2)]
    before = deck_beliefs(view(), registry)
    after = deck_beliefs(view(), registry + [deck("secret", "heldout", 4, "Dragapult")])
    assert before == after
    assert before[0] == [{"archetype": "Crustle", "probability": 1.0}]


def test_duplicate_archetypes_are_aggregated_and_unknown_fields_are_not_used():
    registry = [deck("a", "main"), deck("b", "training-variant"), deck("c", "main", archetype="Dragapult")]
    observation = view()
    prior, _ = deck_beliefs(observation, registry)
    assert len(prior) == 2
    assert abs(sum(item["probability"] for item in prior) - 1) < 1e-12
    altered = deepcopy(observation)
    altered["privateOpponentDeck"] = "secret"
    altered["futureChance"] = [1, 2, 3]
    assert deck_beliefs(altered, registry)[0] == prior
