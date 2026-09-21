from __future__ import annotations

import json
from pathlib import Path

from ptcg_lab.strategy_contract import ACTION_TIERS, REQUIRED_PERSPECTIVES, validate_strategy_contract


ROOT = Path(__file__).resolve().parents[2]


def test_strategy_contract_schema_is_valid_json_and_targets_both_document_types():
    schema = json.loads((ROOT / "research/strategy/strategy-contract.schema.json").read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert {"contract", "playbook"} <= set(schema["$defs"])


def test_strategy_contract_is_scoped_and_cross_referenced():
    result = validate_strategy_contract(ROOT)
    assert set(result["playbooks"]) == {"crustle-v1", "dragapult-v1"}
    perspectives = {
        matchup["perspectiveId"]
        for playbook in result["playbooks"].values()
        for matchup in playbook["matchups"]
    }
    assert perspectives == REQUIRED_PERSPECTIVES


def test_strategy_contract_preserves_review_and_abstention_gates():
    result = validate_strategy_contract(ROOT)
    contract = result["contract"]
    assert [tier["id"] for tier in contract["actionAssessment"]["tiers"]] == ACTION_TIERS
    assert contract["confidencePolicy"]["gradeable"] == ["high", "medium"]
    assert contract["confidencePolicy"]["abstentionLabel"] == "insufficient-confidence"
    assert contract["correctionsAndPromotion"]["automaticPromotion"] is False
    assert contract["status"] == "approved"
    assert all(playbook["status"] == "approved" for playbook in result["playbooks"].values())


def test_playbooks_only_reference_cards_from_frozen_manifests_as_executable_identity():
    result = validate_strategy_contract(ROOT)
    crustle = result["playbooks"]["crustle-v1"]
    assert any(item["id"] == "crustle-missing-guide-cards" for item in crustle["versionConflicts"])
    assert "Crushing Hammer" not in " ".join(item["statement"] for item in crustle["principles"])
    deck = json.loads((ROOT / "decks/crustle.json").read_text())
    counts = {item["cardId"]: item["count"] for item in deck["cards"]}
    assert deck["version"] == 3 and counts["TEF-146"] == 2 and counts["SFA-64"] == 2
    dragapult = result["playbooks"]["dragapult-v1"]
    assert any(item["id"] == "dragapult-updated-list-precedence" for item in dragapult["versionConflicts"])
