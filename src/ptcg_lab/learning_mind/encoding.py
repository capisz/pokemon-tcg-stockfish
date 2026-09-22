from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .schema import EQUIVALENCE_VERSION, LIMITS, SCHEMA_VERSION, UnsupportedPosition, identity_hash

RAW_DIM = 64
OPTION_DIM = 48
CARD_BUCKETS = 8192
TOKEN_TYPES = {"global": 1, "pokemon": 2, "stadium": 3, "summary": 4, "context": 5, "effect": 6}
ACTION_TYPES = {name: index + 1 for index, name in enumerate((
    "attack", "play-trainer", "attach-energy", "ability", "retreat", "bench", "evolve",
    "stadium", "choice", "pass", "prompt", "other", "stop"))}


def _bucket(value: str, maximum: int = CARD_BUCKETS) -> int:
    return 1 + int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") % (maximum - 1)


def _hash_features(text: str, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    words = re.findall(r"[a-z0-9_-]+", text.lower())
    for token in words + [" ".join(words[i:i + 2]) for i in range(len(words) - 1)]:
        digest = hashlib.sha256(token.encode()).digest()
        result[digest[0] % size] += 1 if digest[1] & 1 else -1
    norm = float(np.linalg.norm(result))
    return result / norm if norm else result


def _card_id(card: dict) -> str:
    return str(card.get("id") or card.get("cardId") or card.get("name") or "unknown")


def _static_card_features(card: dict) -> np.ndarray:
    value = np.zeros(RAW_DIM, dtype=np.float32)
    value[0] = min(float(card.get("hp", 0)) / 350, 1)
    value[1] = min(float(card.get("prizeValue", 1)) / 3, 1)
    value[2] = {"pokemon": 1, "trainer": .5, "energy": -.5}.get(str(card.get("kind", "")).lower(), 0)
    value[3] = min(len(card.get("attacks") or []) / 3, 1)
    value[4] = min(len(card.get("powers") or []) / 3, 1)
    value[16:] = _hash_features(" ".join((str(card.get("name", "")), str(card.get("stage", "")),
                                           " ".join(card.get("types") or []))), RAW_DIM - 16)
    return value


@dataclass(frozen=True)
class ActionClass:
    semantic_key: str
    actions: tuple[dict, ...]
    feature: np.ndarray
    type_id: int
    source_token: int
    target_token: int

    @property
    def multiplicity(self) -> int:
        return len(self.actions)


@dataclass
class EncodedDecision:
    state_card_ids: np.ndarray
    state_features: np.ndarray
    state_type_ids: np.ndarray
    action_features: np.ndarray
    action_type_ids: np.ndarray
    source_indices: np.ndarray
    target_indices: np.ndarray
    action_classes: list[ActionClass]
    token_keys: list[str]
    identity: str


def _action_semantics(action: dict) -> dict:
    def ref(value):
        if not isinstance(value, dict):
            return None
        zone = value.get("zone")
        # Equal copies in hand/prompt are interchangeable by card identity.
        index = None if zone in {"hand", "prompt"} else value.get("index")
        return {"playerId": value.get("playerId"), "zone": zone, "index": index}
    return {"type": action.get("type"), "label": re.sub(r"\s+", " ", str(action.get("label", "")).strip().lower()),
            "cardId": action.get("cardId"), "target": action.get("target"),
            "choiceOperation": action.get("choiceOperation"), "sourceRef": ref(action.get("sourceRef")),
            "targetRef": ref(action.get("targetRef")),
            "choiceRefs": [{"sourceRef": ref(item.get("sourceRef")), "targetRef": ref(item.get("targetRef")),
                            "cardId": item.get("cardId"), "amount": item.get("amount")}
                           for item in action.get("choiceRefs") or []]}


def _ref_key(ref: dict | None) -> str | None:
    if not ref:
        return None
    return f"p{ref.get('playerId')}:{ref.get('zone')}:{ref.get('index', '')}"


def encode_decision(observation: dict, tracker: dict) -> EncodedDecision:
    actor = observation.get("playerId")
    if actor not in (0, 1) or tracker.get("viewer") != actor:
        raise ValueError("encoder requires the acting seat's tracker snapshot")
    cards: list[int] = []
    features: list[np.ndarray] = []
    types: list[int] = []
    keys: list[str] = []

    def add(key: str, token_type: str, card: dict | None = None, numeric: dict[int, float] | None = None):
        vector = _static_card_features(card or {})
        for index, value in (numeric or {}).items():
            vector[index] = value
        keys.append(key); cards.append(_bucket(_card_id(card)) if card else 0)
        features.append(vector); types.append(TOKEN_TYPES[token_type])

    own = next(player for player in observation.get("players", []) if player.get("id") == actor)
    other = next(player for player in observation.get("players", []) if player.get("id") != actor)
    add("global", "global", numeric={0: min(observation.get("turn", 0) / 50, 1), 1: 1 if observation.get("decisionPlayer") == actor else -1,
                                      2: own.get("prizesRemaining", 6) / 6, 3: other.get("prizesRemaining", 6) / 6,
                                      4: own.get("handCount", 0) / 20, 5: other.get("handCount", 0) / 20})
    for player in (own, other):
        owner = "own" if player is own else "opponent"
        board = [("active", 0, player.get("active"))] + [("bench", i, item) for i, item in enumerate(player.get("bench", []))]
        for zone, index, item in board:
            if not item:
                continue
            add(f"p{player['id']}:{zone}:{index}", "pokemon", item.get("card", {}),
                {6: min(item.get("damage", 0) / 350, 1), 7: min(len(item.get("energy", [])) / 8, 1),
                 8: min(len(item.get("tools", [])) / 3, 1), 9: 1 if owner == "own" else -1})
    stadium = observation.get("stadium")
    if stadium:
        add("stadium", "stadium", stadium.get("card", {}), {9: 1 if stadium.get("owner") == actor else -1})
    summaries = [("own-hand", own.get("hand", [])), ("own-discard", own.get("discard", [])),
                 ("opponent-discard", other.get("discard", []))]
    for zone, zone_cards in summaries:
        for identifier, count in sorted(Counter(_card_id(card) for card in zone_cards).items()):
            card = next(card for card in zone_cards if _card_id(card) == identifier)
            add(f"summary:{zone}:{identifier}", "summary", card, {10: min(count / 4, 1)})
    for owner, facts in sorted((tracker.get("seats") or {}).items()):
        for card_id, count in sorted((facts.get("revealedHand") or {}).items()):
            add(f"known-hand:{owner}:{card_id}", "context", {"id": card_id}, {10: min(count / 4, 1)})
        for index, card_id in enumerate(facts.get("knownDeckOrder") or []):
            add(f"known-deck:{owner}:{index}", "context", {"id": card_id}, {11: 1 / (index + 1)})
    for index, effect in enumerate(tracker.get("lingeringEffects") or []):
        add(f"effect:{index}", "effect", numeric={12: 1, **{16 + i: x for i, x in enumerate(_hash_features(effect, RAW_DIM - 16))}})
    if len(keys) > LIMITS["stateTokens"]:
        raise UnsupportedPosition(f"state token cap exceeded: {len(keys)} > {LIMITS['stateTokens']}")

    index_by_key = {key: index for index, key in enumerate(keys)}
    grouped: dict[str, list[dict]] = {}
    semantics: dict[str, dict] = {}
    for action in observation.get("legalActions", []):
        semantic = _action_semantics(action)
        key = identity_hash({"version": EQUIVALENCE_VERSION, **semantic})
        grouped.setdefault(key, []).append(action); semantics[key] = semantic
    if len(grouped) > LIMITS["legalActionClasses"]:
        raise UnsupportedPosition(f"legal action cap exceeded: {len(grouped)} > {LIMITS['legalActionClasses']}")
    classes = []
    for key in sorted(grouped):
        semantic = semantics[key]
        action = grouped[key][0]
        text = f"{action.get('type', '')} {action.get('label', '')} {action.get('cardId', '')} {action.get('target', '')}"
        vector = np.zeros(OPTION_DIM, dtype=np.float32)
        vector[:8] = [min(len(grouped[key]) / 4, 1), min(float(action.get("selectionCount", 0)) / 8, 1),
                      1 if action.get("choiceOperation") == "finish" else 0,
                      1 if action.get("choiceOperation") == "undo" else 0, 0, 0, 0, 0]
        vector[8:] = _hash_features(text, OPTION_DIM - 8)
        source = index_by_key.get(_ref_key(semantic.get("sourceRef")), 0)
        target = index_by_key.get(_ref_key(semantic.get("targetRef")), 0)
        action_type = str(action.get("type", "other"))
        classes.append(ActionClass(key, tuple(grouped[key]), vector,
                                   ACTION_TYPES.get(action_type, ACTION_TYPES["other"]), source, target))
    record = {"schemaVersion": SCHEMA_VERSION, "trackerHash": tracker.get("hash"), "tokenKeys": keys,
              "actionKeys": [item.semantic_key for item in classes]}
    return EncodedDecision(np.asarray(cards, dtype=np.int64), np.asarray(features, dtype=np.float32),
                           np.asarray(types, dtype=np.int64), np.asarray([item.feature for item in classes], dtype=np.float32),
                           np.asarray([item.type_id for item in classes], dtype=np.int64),
                           np.asarray([item.source_token for item in classes], dtype=np.int64),
                           np.asarray([item.target_token for item in classes], dtype=np.int64), classes, keys,
                           identity_hash(record))


def collate(decisions: list[EncodedDecision]) -> dict[str, np.ndarray]:
    if not decisions:
        raise ValueError("cannot collate an empty batch")
    state_n = max(len(item.token_keys) for item in decisions)
    action_n = max(len(item.action_classes) for item in decisions) + 1  # STOP
    batch = len(decisions)
    result = {
        "state_card_ids": np.zeros((batch, state_n), np.int64),
        "state_features": np.zeros((batch, state_n, RAW_DIM), np.float32),
        "state_type_ids": np.zeros((batch, state_n), np.int64),
        "state_mask": np.zeros((batch, state_n), bool),
        "option_features": np.zeros((batch, action_n, OPTION_DIM), np.float32),
        "option_type_ids": np.full((batch, action_n), ACTION_TYPES["stop"], np.int64),
        "source_indices": np.zeros((batch, action_n), np.int64),
        "target_indices": np.zeros((batch, action_n), np.int64),
        "option_mask": np.zeros((batch, action_n), bool),
    }
    for row, item in enumerate(decisions):
        ns, na = len(item.token_keys), len(item.action_classes)
        result["state_card_ids"][row, :ns] = item.state_card_ids
        result["state_features"][row, :ns] = item.state_features
        result["state_type_ids"][row, :ns] = item.state_type_ids
        result["state_mask"][row, :ns] = True
        result["option_features"][row, :na] = item.action_features
        result["option_type_ids"][row, :na] = item.action_type_ids
        result["source_indices"][row, :na] = item.source_indices
        result["target_indices"][row, :na] = item.target_indices
        result["option_mask"][row, :na] = True
        result["option_mask"][row, -1] = True
    return result
