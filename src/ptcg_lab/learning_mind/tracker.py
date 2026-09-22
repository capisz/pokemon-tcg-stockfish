from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

from .schema import TRACKER_VERSION, identity_hash


def _card_key(card: dict) -> str:
    return str(card.get("id") or card.get("cardId") or card.get("name") or "unknown")


def _public_cards(player: dict) -> Counter:
    cards = list(player.get("discard", []))
    if player.get("active"):
        cards.append(player["active"].get("card", {}))
    cards.extend(item.get("card", {}) for item in player.get("bench", []))
    return Counter(_card_key(card) for card in cards)


@dataclass
class SeatFacts:
    revealed_hand: Counter = field(default_factory=Counter)
    known_deck_order: list[str] = field(default_factory=list)
    exact_prize_deductions: Counter = field(default_factory=Counter)
    once_per_turn: set[str] = field(default_factory=set)

    def record(self) -> dict:
        return {"revealedHand": dict(sorted(self.revealed_hand.items())),
                "knownDeckOrder": list(self.known_deck_order),
                "exactPrizeDeductions": dict(sorted(self.exact_prize_deductions.items())),
                "oncePerTurn": sorted(self.once_per_turn)}


class ObservableHistoryTracker:
    """Facts reconstructed from one seat's already-redacted stream only."""

    version = TRACKER_VERSION

    def __init__(self, seat: int):
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        self.seat = seat
        self.facts = {0: SeatFacts(), 1: SeatFacts()}
        self.recent_public_actions: list[str] = []
        self.lingering_effects: list[str] = []
        self._last_turn: int | None = None
        self._last_history: list[str] = []
        self._last_public = {0: Counter(), 1: Counter()}

    def update(self, observation: dict) -> dict:
        if observation.get("playerId") != self.seat:
            raise ValueError("tracker received another seat's private observation")
        turn = int(observation.get("turn", 0))
        if self._last_turn is not None and turn != self._last_turn:
            for facts in self.facts.values():
                facts.once_per_turn.clear()
        self._last_turn = turn

        history = [str(item) for item in observation.get("history", [])]
        prefix = 0
        while prefix < min(len(history), len(self._last_history)) and history[prefix] == self._last_history[prefix]:
            prefix += 1
        additions = history[prefix:] if prefix == len(self._last_history) else history
        self.recent_public_actions = (self.recent_public_actions + additions)[-24:]
        self._last_history = history

        for player in observation.get("players", []):
            pid = int(player.get("id", -1))
            if pid not in self.facts:
                continue
            visible = _public_cards(player)
            # A newly public card ceases to be merely a known hidden hand card.
            for card, count in (visible - self._last_public[pid]).items():
                self.facts[pid].revealed_hand[card] -= count
                if self.facts[pid].revealed_hand[card] <= 0:
                    self.facts[pid].revealed_hand.pop(card, None)
            self._last_public[pid] = visible

        for fact in observation.get("knowledge") or []:
            self._consume_knowledge(fact)
        prompt = observation.get("prompt") or {}
        for card in prompt.get("cards") or []:
            # Prompt cards are private only when this actor can see them.
            if prompt.get("type") in {"Reveal hand", "Opponent hand"}:
                self.facts[1 - self.seat].revealed_hand[_card_key(card)] += 1
        self.lingering_effects = sorted({line for line in history[-16:]
                                         if any(word in line.lower() for word in ("during", "until", "can't", "cannot"))})
        return self.snapshot()

    def _consume_knowledge(self, fact: object) -> None:
        if not isinstance(fact, dict):
            return
        owner = fact.get("playerId", fact.get("owner"))
        if owner not in (0, 1):
            return
        kind = str(fact.get("type", "")).lower().replace("_", "-")
        cards = fact.get("cards") or ([fact["card"]] if isinstance(fact.get("card"), dict) else [])
        keys = [_card_key(card) for card in cards if isinstance(card, dict)]
        if "hand" in kind and "reveal" in kind:
            self.facts[owner].revealed_hand.update(keys)
        elif "deck" in kind and any(word in kind for word in ("top", "order", "known")):
            self.facts[owner].known_deck_order = keys
        elif "prize" in kind:
            self.facts[owner].exact_prize_deductions.update(keys)
        elif "used" in kind or "once" in kind:
            marker = str(fact.get("name") or fact.get("effect") or kind)
            self.facts[owner].once_per_turn.add(marker)

    def snapshot(self) -> dict:
        record = {"version": self.version, "viewer": self.seat,
                  "seats": {str(key): value.record() for key, value in self.facts.items()},
                  "recentPublicActions": list(self.recent_public_actions),
                  "lingeringEffects": list(self.lingering_effects)}
        return {**deepcopy(record), "hash": identity_hash(record)}
