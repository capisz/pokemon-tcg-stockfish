"""Dedicated deterministic seed derivation for heldout macro labels."""

from __future__ import annotations

import hashlib


HELDOUT_SEED_VERSION = "heldout-sha256-namespace-v1"


def heldout_rollout_seed(position_hash: str, rollout_index: int, rollout_identity: str) -> int:
    if (not isinstance(position_hash, str) or not position_hash
            or type(rollout_index) is not int or rollout_index < 0
            or not isinstance(rollout_identity, str) or not rollout_identity):
        raise ValueError("heldout rollout seed inputs are malformed")
    material = f"learning-mind-v1|heldout|{rollout_identity}|{position_hash}|{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
