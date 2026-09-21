from __future__ import annotations

import json
from pathlib import Path


ACTION_TIERS = ["preferred", "acceptable", "inaccuracy", "mistake", "critical-error"]
REQUIRED_PERSPECTIVES = {
    "crustle-vs-dragapult",
    "dragapult-vs-crustle",
    "crustle-mirror",
    "dragapult-mirror",
}
REQUIRED_CONCEPTS = {
    "win-condition",
    "prize-race",
    "tempo",
    "offense",
    "defense",
    "resource-plan",
    "information-sequencing",
    "plan-switch",
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def validate_strategy_contract(root: Path) -> dict:
    strategy = root / "research" / "strategy"
    contract = read_json(strategy / "contract-v1.json")
    if contract.get("schemaVersion") != 1 or contract.get("id") != "strategy-contract-v1":
        raise ValueError("Unsupported strategy contract identity")
    specialists = contract.get("scope", {}).get("specialists", [])
    if {item.get("id") for item in specialists} != {"crustle-v1", "dragapult-v1"}:
        raise ValueError("Strategy v1 must contain only the Crustle and Dragapult specialists")
    perspectives = contract.get("scope", {}).get("matchupPerspectives", [])
    if {item.get("id") for item in perspectives} != REQUIRED_PERSPECTIVES:
        raise ValueError("Strategy v1 matchup perspectives changed")
    concepts = {item.get("id") for item in contract.get("strategicConcepts", [])}
    if concepts != REQUIRED_CONCEPTS:
        raise ValueError("Strategy v1 concept vocabulary changed")
    tiers = contract.get("actionAssessment", {}).get("tiers", [])
    if [item.get("id") for item in tiers] != ACTION_TIERS or [item.get("rank") for item in tiers] != list(range(5)):
        raise ValueError("Action assessment tiers must remain ordered and complete")
    confidence = contract.get("confidencePolicy", {})
    if set(confidence.get("gradeable", [])) != {"high", "medium"} or confidence.get("abstentionLabel") != "insufficient-confidence":
        raise ValueError("Low-confidence decisions must abstain")
    if contract.get("correctionsAndPromotion", {}).get("automaticPromotion") is not False:
        raise ValueError("Strategy candidates may not auto-promote")

    playbooks = {}
    referenced_perspectives = set()
    for specialist in specialists:
        playbook_path = root / specialist["playbook"]
        playbook = read_json(playbook_path)
        deck = read_json(root / "decks" / f"{specialist['deckId']}.json")
        reference = playbook.get("referenceDeck", {})
        for key, expected in (("id", deck["id"]), ("version", deck["version"]), ("listHash", deck["listHash"]),
                              ("sourceSha256", deck["source"]["sourceSha256"])):
            if reference.get(key) != expected:
                raise ValueError(f"{playbook_path} referenceDeck.{key} does not match the frozen main manifest")
        if playbook.get("id") != specialist["id"]:
            raise ValueError(f"{playbook_path} specialist identity mismatch")
        if contract.get("status") == "approved" and playbook.get("status") != "approved":
            raise ValueError(f"{playbook_path} must be approved with the approved contract")
        if playbook.get("source", {}).get("contentHash") != reference["sourceSha256"]:
            raise ValueError(f"{playbook_path} guide source does not match the deck provenance")
        if not playbook.get("principles") or not playbook.get("matchups"):
            raise ValueError(f"{playbook_path} requires principles and matchups")
        ids = [item.get("id") for item in playbook["principles"]]
        if len(ids) != len(set(ids)) or any(not identifier for identifier in ids):
            raise ValueError(f"{playbook_path} principle IDs must be unique and nonempty")
        for principle in playbook["principles"]:
            unknown = set(principle.get("concepts", [])) - REQUIRED_CONCEPTS
            if unknown:
                raise ValueError(f"{playbook_path} uses unknown strategic concepts: {sorted(unknown)}")
            if principle.get("reviewStatus") not in {"needs-human-review", "approved", "corrected", "rejected"}:
                raise ValueError(f"{playbook_path} has an invalid principle review status")
            if playbook.get("status") == "approved" and principle.get("reviewStatus") == "needs-human-review":
                raise ValueError(f"{playbook_path} retains an unreviewed principle")
            if not principle.get("sourcePages"):
                raise ValueError(f"{playbook_path} principles require source pages")
        for matchup in playbook["matchups"]:
            perspective = matchup.get("perspectiveId")
            if perspective not in REQUIRED_PERSPECTIVES or perspective in referenced_perspectives:
                raise ValueError(f"Duplicate or unknown matchup perspective {perspective}")
            referenced_perspectives.add(perspective)
            if playbook.get("status") == "approved" and matchup.get("reviewStatus") == "needs-human-review":
                raise ValueError(f"{playbook_path} retains an unreviewed matchup")
        playbooks[playbook["id"]] = playbook
    if referenced_perspectives != REQUIRED_PERSPECTIVES:
        raise ValueError("Every approved v1 matchup perspective must have one playbook entry")
    return {"contract": contract, "playbooks": playbooks}
