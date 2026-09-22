from __future__ import annotations

import hashlib
from dataclasses import dataclass

ARCHETYPES = ("crustle", "dragapult", "raging-bolt", "grimmsnarl", "mega-lucario")


def assignment(index: int, historical_policies: list[str]) -> dict:
    if index < 0 or not historical_policies: raise ValueError("curriculum needs an index and historical policies")
    # One in twenty is a mirror. The remaining slots rotate every archetype uniformly.
    own = ARCHETYPES[index % len(ARCHETYPES)]
    mirror = index % 20 == 0
    opponent = own if mirror else ARCHETYPES[(index // len(ARCHETYPES) + index + 1) % len(ARCHETYPES)]
    historical = index % 5 == 0  # exactly 20%
    return {"ownArchetype": own, "opponentArchetype": opponent, "mirror": mirror,
            "opponentPolicy": historical_policies[(index // 5) % len(historical_policies)] if historical else "current",
            "policyFamily": "historical" if historical else "current"}


def specialist_for_deck(deck_hash: str, registry: dict, generalist: str) -> str:
    record = registry.get(deck_hash)
    return record["checkpoint"] if isinstance(record, dict) and record.get("approved") is True else generalist


def promotion_seed_namespace_disjoint(seed: int, purpose: str, ordinal: int) -> int:
    if purpose not in {"training", "development", "promotion"}: raise ValueError("unknown namespace")
    offset = {"training": 0, "development": 1_000_000_000, "promotion": 2_000_000_000}[purpose]
    value = int.from_bytes(hashlib.sha256(f"{seed}|{purpose}|{ordinal}".encode()).digest()[:8], "big")
    return offset + value % 900_000_000
