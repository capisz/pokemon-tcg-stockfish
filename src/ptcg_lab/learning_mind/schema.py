from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

SCHEMA_VERSION = "learning-mind-visible-tokens-v1"
TRACKER_VERSION = "observable-history-v1"
EQUIVALENCE_VERSION = "legal-action-equivalence-v1"
CARD_METADATA_VERSION = "engine-static-card-view-v1"
LIMITS = {"stateTokens": 160, "legalActionClasses": 128, "stopActions": 1}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def identity_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class IdentityManifest:
    engine_build_hash: str
    deck_manifest_hash: str
    feature_schema_hash: str
    tracker_rules_hash: str
    card_metadata_hash: str
    action_equivalence_hash: str

    @classmethod
    def create(cls, *, engine_build_hash: str, deck_manifests: object,
               card_metadata: object, extra_schema: Mapping[str, object] | None = None):
        schema = {"version": SCHEMA_VERSION, "limits": LIMITS, **(extra_schema or {})}
        return cls(
            engine_build_hash=engine_build_hash,
            deck_manifest_hash=identity_hash(deck_manifests),
            feature_schema_hash=identity_hash(schema),
            tracker_rules_hash=identity_hash({"version": TRACKER_VERSION}),
            card_metadata_hash=identity_hash({"version": CARD_METADATA_VERSION, "cards": card_metadata}),
            action_equivalence_hash=identity_hash({"version": EQUIVALENCE_VERSION}),
        )

    def record(self) -> dict:
        value = asdict(self)
        return {"schemaVersion": 1, **value, "identityHash": identity_hash(value)}

    def require_match(self, frozen: Mapping[str, object]) -> None:
        current = self.record()
        if current != dict(frozen):
            changed = sorted(key for key in set(current) | set(frozen) if current.get(key) != frozen.get(key))
            raise IdentityError(f"Learning-mind identity drift: {', '.join(changed)}")


class IdentityError(RuntimeError):
    pass


class UnsupportedPosition(RuntimeError):
    """Fail-closed signal: the policy must not silently discard information."""
