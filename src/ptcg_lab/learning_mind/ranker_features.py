from __future__ import annotations

import hashlib
import math

import numpy as np

from .schema import identity_hash

MACRO_FEATURE_SCHEMA = {
    "version": "macro-state-plan-features-v2",
    "dimension": 640,
    "global": "64 actor-visible context and own-hand count-sketch features",
    "board": "12 fixed actor-visible active/bench slots with 32 numeric and identity features each",
    "candidate": "32 semantic plan features plus 160 deterministic semantic-token hash features",
    "hiddenInformation": "opponent private hand contents are never read",
}
MACRO_FEATURE_SCHEMA_HASH = identity_hash(MACRO_FEATURE_SCHEMA)


def _feature_bucket(value: object, width: int) -> tuple[int, float]:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return digest[0] % width, 1.0 if digest[1] & 1 else -1.0


def _safe_count(value: object, scale: float, cap: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number / scale, 0.0), cap)


def _card_identity(card: object) -> str:
    if not isinstance(card, dict):
        return "unknown"
    return str(card.get("id") or card.get("cardId") or card.get("name") or card.get("kind") or "unknown")


def _semantic_tokens(candidate: dict) -> list[str]:
    tokens = []
    for key in ("turn_intent", "intended_attack", "attack_target", "supporter", "pivot_destination",
                "energy_source", "energy_destination", "disruption_intent", "protected_pokemon", "setup_target"):
        value = candidate.get(key)
        if value is not None:
            tokens.append(f"{key}={value}")
    for action in candidate.get("action_sequence") or []:
        if not isinstance(action, dict):
            continue
        for key in ("type", "cardId", "target", "amount", "choice", "energyType"):
            value = action.get(key)
            if value is not None:
                tokens.append(f"action.{key}={value}")
        target = action.get("targetRef")
        if isinstance(target, dict):
            tokens.append("targetRef=" + ":".join(str(target.get(key, ""))
                for key in ("playerId", "zone", "index")))
    return tokens


def candidate_features_v2(observation: dict, candidate: dict) -> np.ndarray:
    """Rank a macro plan from the actor-visible board and action semantics only."""
    players = observation.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("macro ranker requires a two-player actor-visible observation")
    actor_id = observation.get("playerId")
    own_matches = [player for player in players if player.get("id") == actor_id]
    other_matches = [player for player in players if player.get("id") != actor_id]
    if len(own_matches) != 1 or len(other_matches) != 1:
        raise ValueError("macro ranker observation does not identify one actor and one opponent")
    own, other = own_matches[0], other_matches[0]
    vector = np.zeros(MACRO_FEATURE_SCHEMA["dimension"], dtype=np.float32)

    vector[:16] = [
        _safe_count(observation.get("turn"), 50),
        _safe_count(own.get("handCount", len(own.get("hand", []))), 20),
        _safe_count(other.get("handCount"), 20),
        _safe_count(own.get("prizesRemaining", own.get("prizes")), 6),
        _safe_count(other.get("prizesRemaining", other.get("prizes")), 6),
        _safe_count(own.get("deckCount"), 60), _safe_count(other.get("deckCount"), 60),
        _safe_count(len(own.get("bench", [])), 5), _safe_count(len(other.get("bench", [])), 5),
        _safe_count(len(own.get("discard", [])), 60), _safe_count(len(other.get("discard", [])), 60),
        _safe_count(len(own.get("stadium", [])), 1), _safe_count(len(other.get("stadium", [])), 1),
        _safe_count(len(own.get("lostzone", [])), 60), _safe_count(len(other.get("lostzone", [])), 60),
        1.0 if observation.get("prompt") is not None else 0.0,
    ]
    for card in own.get("hand", []):
        bucket, sign = _feature_bucket(_card_identity(card), 48)
        vector[16 + bucket] += sign / max(1, len(own.get("hand", [])))
    for card in own.get("discard", []):
        bucket, sign = _feature_bucket("discard:" + _card_identity(card), 16)
        vector[48 + bucket] += sign / max(1, len(own.get("discard", [])))

    for side_index, player in enumerate((own, other)):
        slots = [player.get("active")] + list(player.get("bench", []))[:5]
        for slot_index in range(6):
            pokemon = slots[slot_index] if slot_index < len(slots) else None
            if not isinstance(pokemon, dict):
                continue
            offset = 64 + (side_index * 6 + slot_index) * 32
            card = pokemon.get("card") if isinstance(pokemon.get("card"), dict) else pokemon
            hp = card.get("hp", pokemon.get("hp", 0))
            damage = pokemon.get("damage", 0)
            energies = pokemon.get("energy", pokemon.get("energies", [])) or []
            tools = pokemon.get("tools", []) or []
            conditions = pokemon.get("conditions", []) or []
            vector[offset:offset + 12] = [
                1.0, 1.0 if slot_index == 0 else 0.0, float(side_index),
                _safe_count(hp, 400), _safe_count(damage, 400),
                _safe_count(max(0, float(hp or 0) - float(damage or 0)), 400),
                _safe_count(len(energies), 8), _safe_count(len(tools), 4),
                _safe_count(len(conditions), 8), _safe_count(card.get("prizeValue"), 3),
                _safe_count(len(card.get("attacks", []) or []), 8),
                _safe_count(len(card.get("powers", []) or []), 4),
            ]
            bucket, sign = _feature_bucket(_card_identity(card), 16)
            vector[offset + 12 + bucket] = sign
            for energy in energies:
                energy_bucket, energy_sign = _feature_bucket("energy:" + _card_identity(energy), 4)
                vector[offset + 28 + energy_bucket] += energy_sign / max(1, len(energies))

    candidate_offset = 448
    sequence = candidate.get("action_sequence") or []
    action_types = [str(action.get("type", "unknown")) for action in sequence if isinstance(action, dict)]
    role_fields = ("intended_attack", "supporter", "pivot_destination", "energy_source",
                   "energy_destination", "disruption_intent", "protected_pokemon", "setup_target")
    vector[candidate_offset:candidate_offset + 18] = [
        1.0 if candidate.get("turn_intent") == "attack" else 0.0,
        1.0 if candidate.get("turn_intent") == "no-attack" else 0.0,
        1.0 if candidate.get("turn_intent") == "incomplete" else 0.0,
        _safe_count(len(sequence), 5),
        *[1.0 if candidate.get(field) is not None else 0.0 for field in role_fields],
        *[_safe_count(action_types.count(kind), 5) for kind in
          ("attack", "play-trainer", "retreat", "attach-energy", "switch", "pass")],
    ]
    player_slots = {own.get("id"): 0, other.get("id"): 1}
    for action in sequence:
        if not isinstance(action, dict) or not isinstance(action.get("targetRef"), dict):
            continue
        target = action["targetRef"]
        side = player_slots.get(target.get("playerId"))
        zone = target.get("zone")
        if side is None:
            continue
        if zone == "active":
            slot = side * 6
        elif zone == "bench" and isinstance(target.get("index"), int) and 0 <= target["index"] < 5:
            slot = side * 6 + 1 + target["index"]
        else:
            continue
        action_type = str(action.get("type", ""))
        semantic_slot = (0 if action_type == "attack" else
                         1 if action_type in {"retreat", "switch"} else
                         2 if action_type in {"attach-energy", "energy-switch"} else None)
        if semantic_slot is not None:
            vector[candidate_offset + 18 + semantic_slot] = slot / 11
    for token in _semantic_tokens(candidate):
        bucket, sign = _feature_bucket(token.lower(), 160)
        vector[candidate_offset + 32 + bucket] += sign
    norm = np.linalg.norm(vector[candidate_offset + 32:candidate_offset + 192])
    if norm:
        vector[candidate_offset + 32:candidate_offset + 192] /= norm
    return vector
