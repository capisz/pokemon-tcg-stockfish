from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import asdict, dataclass
from typing import Callable

from .schema import UnsupportedPosition, identity_hash

MAX_CANDIDATES = 128


@dataclass(frozen=True)
class MacroCandidateV1:
    intended_attack: str | None = None
    attack_target: str | None = None
    supporter: str | None = None
    pivot_destination: str | None = None
    energy_source: str | None = None
    energy_destination: str | None = None
    disruption_intent: str | None = None
    protected_pokemon: str | None = None
    setup_target: str | None = None
    action_ids: tuple[str, ...] = ()

    def key(self) -> str:
        return identity_hash(asdict(self))


class MacroExecutionFailure(RuntimeError):
    pass


def _kind(action: dict) -> str:
    text = f"{action.get('type', '')} {action.get('label', '')}".lower()
    if action.get("type") == "attack": return "attack"
    if "supporter" in text or any(name in text for name in ("judge", "research", "lillie", "boss")): return "supporter"
    if action.get("type") == "retreat" or "switch" in text: return "pivot"
    if action.get("type") == "attach-energy": return "energy"
    if any(name in text for name in ("judge", "hammer", "eri", "stamp", "red card", "fan")): return "disruption"
    return "other"


def _name(action: dict) -> str:
    return str(action.get("cardId") or action.get("label") or action.get("id"))


def generate_candidates(observation: dict, *, cap: int = MAX_CANDIDATES) -> list[MacroCandidateV1]:
    legal = list(observation.get("legalActions") or [])
    buckets = {key: [] for key in ("attack", "supporter", "pivot", "energy", "disruption")}
    for action in legal:
        kind = _kind(action)
        if kind in buckets:
            buckets[kind].append(action)
    # None is always a deliberate plan choice.  Candidate action IDs are an
    # executable subset, not a fabricated whole turn.
    choices = [[None] + sorted(values, key=lambda item: (_name(item), str(item.get("id"))))
               for values in buckets.values()]
    candidates: dict[str, MacroCandidateV1] = {}
    for attack, supporter, pivot, energy, disruption in itertools.product(*choices):
        actions = tuple(item for item in (supporter, pivot, energy, disruption, attack) if item)
        # A single action can occupy two semantic roles (e.g. Judge); execute it once.
        ids = tuple(dict.fromkeys(str(item.get("id")) for item in actions))
        energy_name = _name(energy) if energy else None
        candidate = MacroCandidateV1(
            intended_attack=_name(attack) if attack else None,
            attack_target=str(attack.get("target")) if attack and attack.get("target") is not None else None,
            supporter=_name(supporter) if supporter else None,
            pivot_destination=str(pivot.get("target") or _name(pivot)) if pivot else None,
            energy_source=energy_name,
            energy_destination=str(energy.get("target")) if energy and energy.get("target") is not None else None,
            disruption_intent=_name(disruption) if disruption else None,
            protected_pokemon=str((energy or {}).get("target")) if energy and "mist" in _name(energy).lower() else None,
            setup_target=str((energy or pivot or {}).get("target")) if energy or pivot else None,
            action_ids=ids)
        candidates[candidate.key()] = candidate
        if len(candidates) > cap:
            raise UnsupportedPosition(f"macro candidate cap exceeded: > {cap}")
    return [candidates[key] for key in sorted(candidates)]


def execute_candidate(candidate: MacroCandidateV1, legal_actions: list[dict]) -> list[dict]:
    by_id = {str(action.get("id")): action for action in legal_actions}
    missing = [identifier for identifier in candidate.action_ids if identifier not in by_id]
    if missing:
        raise MacroExecutionFailure(f"declared macro actions are no longer legal: {missing}")
    return [by_id[identifier] for identifier in candidate.action_ids]


def rollout_seed(namespace: str, position_hash: str, rollout_index: int) -> int:
    if namespace not in {"training", "development", "promotion"}:
        raise ValueError("unknown seed namespace")
    token = f"learning-mind-v1|{namespace}|{position_hash}|{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def label_candidates(candidates: list[MacroCandidateV1], position_hash: str,
                     rollout: Callable[[MacroCandidateV1, int], dict], *,
                     namespace: str = "training", initial: int = 16,
                     maximum: int = 64, close_margin: float = .10) -> list[dict]:
    if namespace == "promotion":
        raise ValueError("label generation may not consume promotion seeds")
    if not candidates or not 1 <= initial <= maximum <= 64:
        raise ValueError("invalid rollout allocation")
    records = {item.key(): [] for item in candidates}

    def run(indices, selected):
        for index in indices:
            seed = rollout_seed(namespace, position_hash, index)
            for candidate in selected:
                outcome = rollout(candidate, seed)
                status = outcome.get("status")
                if status == "finished" and outcome.get("score") in {0, .5, 1}:
                    records[candidate.key()].append(float(outcome["score"]))
                elif status not in {"truncated", "error"}:
                    raise ValueError("rollout returned an invalid status")

    run(range(initial), candidates)
    means = {key: sum(values) / len(values) if values else -math.inf for key, values in records.items()}
    best = max(means.values())
    close = [candidate for candidate in candidates if best - means[candidate.key()] <= close_margin]
    if maximum > initial and len(close) > 1:
        run(range(initial, maximum), close)
    finite_means = [sum(values) / len(values) for values in records.values() if values]
    center = max(finite_means) if finite_means else 0.0
    output = []
    for candidate in candidates:
        scores = records[candidate.key()]
        mean = sum(scores) / len(scores) if scores else None
        uncertainty = math.sqrt(max((mean or 0) * (1 - (mean or 0)), .25) / len(scores)) if scores else None
        output.append({"candidate": asdict(candidate), "candidateHash": candidate.key(),
                       "completedRollouts": len(scores), "attemptedRollouts": maximum if candidate in close and len(close) > 1 else initial,
                       "expectedResult": mean, "relativeResult": mean - center if mean is not None else None,
                       "uncertainty": uncertainty, "weight": 0 if not scores else len(scores) / (1 + uncertainty)})
    return output
