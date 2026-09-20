from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

FEATURE_NAMES = (
    "prize_race", "attacker_health", "bench_development", "attached_energy",
    "hand_access", "deck_reserve", "evolution_development", "ready_attackers",
    "recovery_access", "draw_search_access", "mobility_access", "disruption_access",
    "exposed_prizes", "defensive_protection", "non_ex_answers", "resource_exhaustion",
)
CARD_BUCKETS, MAX_VISIBLE_CARDS, ACTION_DIM = 2048, 128, 32


def card_bucket(identifier: str) -> int:
    return 1 + int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % (CARD_BUCKETS - 1)


def pokemon(player: dict) -> list[dict]:
    return ([player["active"]] if player.get("active") else []) + player.get("bench", [])


def visible_cards(observation: dict) -> list[dict]:
    """The caller supplies one already-redacted player observation only."""
    cards = []
    for player in observation.get("players", []):
        cards.extend(item["card"] for item in pokemon(player))
        cards.extend(player.get("discard", []))
        if player["id"] == observation["playerId"]:
            cards.extend(player.get("hand", []))
    return cards


def card_tokens(observation: dict) -> np.ndarray:
    ids = []
    for player in observation.get("players", []):
        owner = "own" if player["id"] == observation["playerId"] else "opponent"
        zones = [("active", [player["active"]["card"]] if player.get("active") else []),
                 ("bench", [item["card"] for item in player.get("bench", [])]),
                 ("discard", player.get("discard", []))]
        if owner == "own":
            zones.append(("hand", player.get("hand", [])))
        for zone, cards in zones:
            ids.extend(card_bucket(f"{owner}:{zone}:{card.get('id', card.get('name', ''))}") for card in cards)
    ids = ids[:MAX_VISIBLE_CARDS]
    return np.array(ids + [0] * (MAX_VISIBLE_CARDS - len(ids)), dtype=np.int64)


def _resources(player: dict, own: bool) -> np.ndarray:
    board = pokemon(player)
    hand = player.get("hand", []) if own else []
    names = [card.get("name", "").lower() for card in hand]

    def count_words(words: tuple[str, ...]) -> float:
        return sum(any(word in name for word in words) for name in names) / 4

    health = sum(max(0, item["card"].get("hp", 0) - item.get("damage", 0)) for item in board) / 1000
    energy = sum(len(item.get("energy", [])) for item in board) / 12
    evolved = sum(str(item["card"].get("stage", "")).lower() not in {"", "basic", "0"} for item in board) / 6
    # Readiness is a resource proxy, not a legal-cost checker (special energies differ).
    ready = sum(bool(item.get("energy")) and bool(item["card"].get("attacks")) for item in board) / 6
    liabilities = sum(max(0, item["card"].get("prizeValue", 1) - 1) for item in board) / 6
    crustle = sum(item["card"].get("name", "").lower() == "crustle" for item in board) / 4
    non_ex = sum(item["card"].get("prizeValue", 1) == 1 and bool(item.get("energy")) for item in board) / 6
    return np.array([
        (6 - player.get("prizesRemaining", 6)) / 6, health, len(board) / 6, energy,
        player.get("handCount", len(hand)) / 10, player.get("deckCount", 0) / 60,
        evolved, ready, count_words(("rod", "recovery", "recycler", "stretcher")),
        count_words(("research", "iono", "ultra ball", "nest ball", "poffin", "arven")),
        count_words(("switch", "jet energy", "rescue board")),
        count_words(("boss", "iono", "catcher", "stamp")), liabilities, crustle, non_ex,
        len(player.get("discard", [])) / 60,
    ], dtype=np.float32)


def resource_features(observation: dict) -> np.ndarray:
    own = next((p for p in observation.get("players", []) if p["id"] == observation["playerId"]), {})
    opponent = next((p for p in observation.get("players", []) if p["id"] != observation["playerId"]), {})
    if not own or not opponent:
        raise ValueError("Observation requires both player views")
    result = _resources(own, True) - _resources(opponent, False)
    # Opponent hidden hand features remain unknown; no fabricated cards are supplied.
    return np.clip(result, -4, 4)


def action_features(action: dict) -> np.ndarray:
    text = f"{action.get('type', '')} {action.get('label', '')}".lower()
    vocabulary = ("attack", "energy", "evolve", "basic", "supporter", "item", "ability", "retreat",
                  "pass", "end", "confirm", "cancel", "draw", "search", "switch", "discard")
    features = [float(word in text) for word in vocabulary]
    # Signed semantic token hashing distinguishes target and prompt/card choices.
    # Never hash the opaque id: it is a decision-local index, not move semantics.
    semantic = f"{text} {action.get('cardId', '')} target:{action.get('target', '')}".lower()
    tokens = re.findall(r"[a-z0-9_-]+", semantic)
    hashed = np.zeros(16, dtype=np.float32)
    for token in tokens + [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]:
        encoded = hashlib.sha256(token.encode()).digest()
        hashed[encoded[0] % len(hashed)] += 1 if encoded[1] % 2 else -1
    hashed /= max(1, float(np.linalg.norm(hashed)))
    features.extend(hashed.tolist())
    return np.array(features, dtype=np.float32)


def heuristic_action_score(action: dict, observation: dict | None = None) -> float:
    # Staged selection controls are operations, not card names. Scoring an Undo
    # containing "Energy" by keyword alone can select/undo forever.
    if action.get("type") == "choice":
        if action.get("choiceOperation") == "undo" or action.get("label") == "Cancel":
            return -100.0
        if action.get("choiceOperation") == "finish":
            return 20.0 if (observation or {}).get("prompt", {}).get("type") == "Choose energy" else 0.0
        return 10.0
    weights = np.array([3, 2, 2.5, 1.2, 1, .5, 1.3, -.8, -2, -2, .1, -.5, 1, .7, .1, -.3] + [0] * 16)
    return float(action_features(action) @ weights)


def deck_beliefs(observation: dict, decks: list[dict]) -> tuple[list[dict], list[str]]:
    # The registry also serves the experiment runner. Its withheld lists must
    # never become an inference oracle merely because the API passed that registry.
    decks = [deck for deck in decks if deck.get("role", "main") not in {"heldout", "historical"}]
    opponent = next((p for p in observation.get("players", []) if p["id"] != observation["playerId"]), None)
    warnings = ["Opponent beliefs are conditional on the supported five-deck pool, not the full metagame."]
    if opponent is None or not decks:
        return [], warnings + ["No deck pool is available for inference."]
    cards = [item["card"] for item in pokemon(opponent)] + opponent.get("discard", [])
    observed = Counter(str(card.get("id", card.get("name"))) for card in cards)
    log_weights = []
    for deck in decks:
        counts = Counter({str(card["cardId"]): card["count"] for card in deck.get("cards", [])})
        if any(count > counts[identifier] for identifier, count in observed.items()):
            log_weights.append(-math.inf)
            continue
        log_weights.append(sum(math.log(max(1, counts[identifier] - index) / max(1, 60 - index))
                               for identifier, count in observed.items() for index in range(count)))
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        return [], warnings + ["Visible cards are inconsistent with all frozen decklists; unsupported deck or variant."]
    weights = [math.exp(value - max(finite)) if math.isfinite(value) else 0 for value in log_weights]
    total = sum(weights)
    grouped: dict[str, float] = {}
    for deck, weight in zip(decks, weights):
        archetype = deck.get("archetype", deck["id"])
        grouped[archetype] = grouped.get(archetype, 0.) + weight / total
    return [{"archetype": archetype, "probability": probability}
            for archetype, probability in sorted(grouped.items()) if probability > 0], warnings + [
                "Composition-only reference-list prior; strategic action likelihoods and unknown-variant mass are not modeled here.",
                "Held-out and historical exact lists are excluded. These priors are separate from sampled search hypotheses.",
            ]
