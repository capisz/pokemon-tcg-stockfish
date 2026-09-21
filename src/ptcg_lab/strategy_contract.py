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


def _deck_cards(root: Path, deck_id: str) -> dict[str, int]:
    deck = read_json(root / "decks" / f"{deck_id}.json")
    return {item["cardId"]: item["count"] for item in deck["cards"]}


def _validate_card_refs(path: Path, refs: list[dict], specialist: dict[str, int], opponent: dict[str, int]) -> None:
    for ref in refs:
        if ref.get("executable") is True:
            card_id = ref.get("cardId")
            allowed = specialist if ref.get("owner") == "specialist" else opponent if ref.get("owner") == "opponent" else {}
            if not card_id or card_id not in allowed:
                raise ValueError(f"{path} has an executable card reference outside the declared specialist/opponent lists: {ref}")
            if "cardName" in ref or "reason" in ref:
                raise ValueError(f"{path} executable card references use only cardId, owner and executable")
        elif ref.get("executable") is False:
            if not ref.get("cardName") or not ref.get("reason"):
                raise ValueError(f"{path} historical card references require cardName and reason")
        else:
            raise ValueError(f"{path} card references require an explicit executable boolean")


def validate_strategy_revision(root: Path) -> dict:
    """Validate the draft v1.1 overlay without mutating or weakening approved v1."""
    approved = validate_strategy_contract(root)
    if approved["contract"].get("status") != "approved":
        raise ValueError("Strategy v1 must remain approved while v1.1 is under review")
    strategy = root / "research" / "strategy"
    revision = read_json(strategy / "contract-v1.1.json")
    if revision.get("id") != "strategy-contract-v1.1" or revision.get("release") != "1.1":
        raise ValueError("Unsupported strategy revision identity")
    if revision.get("status") != "draft-awaiting-human-review":
        raise ValueError("Strategy v1.1 remains draft until explicit human approval")
    if revision.get("base") != {"contractId": "strategy-contract-v1", "approvedCommit": "b176362"}:
        raise ValueError("Strategy v1.1 must identify the approved v1 base")
    opening = revision.get("openingInformationPolicy", {})
    if opening.get("blindChoice", {}).get("preference") != "go-first":
        raise ValueError("Blind opening choice must not use hidden matchup knowledge")
    if revision.get("displayLabels", {}).get("critical-error") != "Blunder":
        raise ValueError("The fifth strategic tier must expose the approved display label")
    if revision.get("approvalGate", {}).get("automaticPromotion") is not False:
        raise ValueError("Strategy revisions may not auto-promote")

    expected = {
        "research/strategy/crustle-v1.1.json": ("crustle-v1", "crustle", "dragapult"),
        "research/strategy/dragapult-v1.1.json": ("dragapult-v1", "dragapult", "crustle"),
    }
    if set(revision.get("revisionFiles", [])) != set(expected):
        raise ValueError("Strategy v1.1 revision file set changed")
    perspectives = set()
    revisions = {}
    for relative, (base_id, deck_id, default_opponent_id) in expected.items():
        path = root / relative
        playbook = read_json(path)
        if playbook.get("status") != "draft-awaiting-human-review" or playbook.get("basePlaybook") != base_id:
            raise ValueError(f"{path} must remain a draft overlay of {base_id}")
        if playbook.get("specialistDeckId") != deck_id or playbook.get("defaultOpponentDeckId") != default_opponent_id:
            raise ValueError(f"{path} changed its specialist or default opponent")
        specialist = _deck_cards(root, deck_id)
        opponent = _deck_cards(root, default_opponent_id)
        for card_id, count in playbook.get("deckCountAssertions", {}).items():
            if specialist.get(card_id) != count:
                raise ValueError(f"{path} count assertion for {card_id} does not match the frozen list")
        patch_ids = []
        for patch in playbook.get("principlePatches", []):
            if patch.get("operation") not in {"add", "replace"} or not patch.get("id"):
                raise ValueError(f"{path} has a malformed principle patch")
            patch_ids.append(patch["id"])
            if set(patch.get("concepts", [])) - REQUIRED_CONCEPTS or not patch.get("sourcePages"):
                raise ValueError(f"{path} principles require known concepts and source pages")
            if "cardRefs" not in patch:
                raise ValueError(f"{path} principles require structured cardRefs")
            _validate_card_refs(path, patch["cardRefs"], specialist, opponent)
        if len(patch_ids) != len(set(patch_ids)):
            raise ValueError(f"{path} principle patch IDs must be unique")
        for matchup in playbook.get("matchupPatches", []):
            perspective = matchup.get("perspectiveId")
            if perspective not in REQUIRED_PERSPECTIVES or perspective in perspectives:
                raise ValueError(f"Duplicate or unknown revision perspective {perspective}")
            perspectives.add(perspective)
            if matchup.get("operation") != "replace" or not matchup.get("sourcePages") or "cardRefs" not in matchup:
                raise ValueError(f"{path} matchup patches require replacement semantics, source pages and cardRefs")
            matchup_opponent_id = deck_id if perspective.endswith("mirror") else default_opponent_id
            _validate_card_refs(path, matchup["cardRefs"], specialist, _deck_cards(root, matchup_opponent_id))
        revisions[playbook["id"]] = playbook
    if perspectives != REQUIRED_PERSPECTIVES:
        raise ValueError("Strategy v1.1 must revise all four approved matchup perspectives")
    return {"approved": approved, "revision": revision, "playbooks": revisions}
