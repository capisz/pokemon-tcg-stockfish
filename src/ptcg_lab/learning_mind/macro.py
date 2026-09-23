from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .schema import UnsupportedPosition, identity_hash

MAX_CANDIDATES = 128
CANDIDATE_GENERATOR_VERSION = "transition-aware-public-determinization-v5-cjs-runtime"


@dataclass(frozen=True)
class MacroCandidateV1:
    turn_intent: str = "incomplete"
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
    action_sequence: tuple[dict, ...] = ()

    def key(self) -> str:
        return identity_hash(asdict(self))


class MacroExecutionFailure(RuntimeError):
    pass


def _kind(action: dict) -> str:
    text = f"{action.get('type', '')} {action.get('label', '')}".lower()
    if action.get("type") == "attack": return "attack"
    if "supporter" in text or any(name in text for name in ("judge", "research", "lillie", "boss")): return "supporter"
    if "energy switch" in text: return "energy"
    if action.get("type") == "retreat" or "switch" in text: return "pivot"
    if action.get("type") == "attach-energy": return "energy"
    if any(name in text for name in ("judge", "hammer", "eri", "stamp", "red card", "fan")): return "disruption"
    return "other"


def _name(action: dict) -> str:
    return str(action.get("cardId") or action.get("label") or action.get("id"))


def generate_candidates(observation: dict, *, cap: int = MAX_CANDIDATES) -> list[MacroCandidateV1]:
    legal = list(observation.get("legalActions") or [])
    if cap < 1 or len(legal) > cap:
        raise UnsupportedPosition(f"root legal-action candidate cap exceeded: {len(legal)} > {cap}")
    candidates: dict[str, MacroCandidateV1] = {}
    # The observation contains only root legal actions. Combining two such IDs
    # does not prove they remain legal in sequence, so this conservative
    # generator evaluates one executable root action per candidate. Multi-step
    # turn plans require a transition-aware generator and are intentionally not
    # synthesized from this static action list.
    for action in sorted(legal, key=lambda item: (_name(item), str(item.get("id")))):
        kind = _kind(action)
        candidate = MacroCandidateV1(
            intended_attack=_name(action) if kind == "attack" else None,
            attack_target=str(action.get("target")) if kind == "attack" and action.get("target") is not None else None,
            supporter=_name(action) if kind == "supporter" else None,
            pivot_destination=str(action.get("target") or _name(action)) if kind == "pivot" else None,
            energy_source=_name(action) if kind == "energy" else None,
            energy_destination=str(action.get("target")) if kind == "energy" and action.get("target") is not None else None,
            disruption_intent=_name(action) if kind == "disruption" else None,
            protected_pokemon=str(action.get("target")) if kind == "energy" and "mist" in _name(action).lower() else None,
            setup_target=str(action.get("target")) if kind in {"energy", "pivot"} and action.get("target") is not None else None,
            action_ids=(str(action.get("id")),))
        candidates[candidate.key()] = candidate
    return [candidates[key] for key in sorted(candidates)]


def candidates_from_transition_plans(plans: Iterable[dict], *, cap: int = MAX_CANDIDATES) -> list[MacroCandidateV1]:
    candidates: dict[str, MacroCandidateV1] = {}
    for plan in plans:
        actions = tuple(plan.get("actions") or ())
        if not actions or any(not isinstance(action, dict) or not action.get("id") for action in actions):
            raise MacroExecutionFailure("transition planner returned an empty or malformed action sequence")
        completion = plan.get("completion", "incomplete")
        final_type = actions[-1].get("type")
        if completion not in {"attack", "no-attack", "incomplete"}:
            raise MacroExecutionFailure("transition planner returned an unknown terminal intent")
        if (completion == "attack" and final_type != "attack") or (completion == "no-attack" and final_type != "pass"):
            raise MacroExecutionFailure("transition planner terminal intent does not match its final legal action")
        if completion == "incomplete" and final_type in {"attack", "pass"}:
            raise MacroExecutionFailure("transition planner marked a terminal action as an incomplete prefix")
        typed = [(_kind(action), action) for action in actions]
        attack = next((action for kind, action in typed if kind == "attack"), None)
        supporter = next((action for kind, action in typed if kind == "supporter"), None)
        pivot = next((action for kind, action in typed if kind == "pivot"), None)
        energy = next((action for kind, action in typed if kind == "energy"), None)
        disruption = next((action for kind, action in typed if kind == "disruption"), None)
        candidate = MacroCandidateV1(
            turn_intent=str(completion),
            intended_attack=_name(attack) if attack else None,
            attack_target=str(attack.get("target")) if attack and attack.get("target") is not None else None,
            supporter=_name(supporter) if supporter else None,
            pivot_destination=str(pivot.get("target") or _name(pivot)) if pivot else None,
            energy_source=_name(energy) if energy else None,
            energy_destination=str(energy.get("target")) if energy and energy.get("target") is not None else None,
            disruption_intent=_name(disruption) if disruption else None,
            protected_pokemon=str(energy.get("target")) if energy and "mist" in _name(energy).lower() else None,
            setup_target=next((str(action.get("target")) for kind, action in typed
                               if kind in {"energy", "pivot"} and action.get("target") is not None), None),
            action_ids=tuple(str(action["id"]) for action in actions),
            action_sequence=actions)
        candidates[candidate.key()] = candidate
        if len(candidates) > cap:
            raise UnsupportedPosition(f"transition macro candidate cap exceeded: > {cap}")
    if not candidates:
        raise UnsupportedPosition("transition macro planner returned no executable candidates")
    return [candidates[key] for key in sorted(candidates)]


def execute_candidate(candidate: MacroCandidateV1, legal_actions: list[dict]) -> list[dict]:
    by_id = {str(action.get("id")): action for action in legal_actions}
    missing = [identifier for identifier in candidate.action_ids if identifier not in by_id]
    if missing:
        raise MacroExecutionFailure(f"declared macro actions are no longer legal: {missing}")
    return [by_id[identifier] for identifier in candidate.action_ids]


def rollout_seed(namespace: str, position_hash: str, rollout_index: int,
                 rollout_identity: str | None = None) -> int:
    if namespace not in {"training", "development", "promotion"}:
        raise ValueError("unknown seed namespace")
    if rollout_identity is not None and (not isinstance(rollout_identity, str) or not rollout_identity):
        raise ValueError("rollout identity must be a non-empty string when provided")
    if rollout_identity is None:
        token = f"learning-mind-v1|{namespace}|{position_hash}|{rollout_index}".encode()
    else:
        token = f"learning-mind-v1|{namespace}|{rollout_identity}|{position_hash}|{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def label_candidates(candidates: list[MacroCandidateV1], position_hash: str,
                     rollout: Callable[[MacroCandidateV1, int], dict], *,
                     namespace: str = "training", initial: int = 16,
                     maximum: int = 64, close_margin: float = .10,
                     extension_batch_size: int = 8,
                     rollout_workers: int = 1, resume_state: dict | None = None,
                     checkpoint: Callable[[dict], None] | None = None,
                     rollout_identity: str | None = None) -> list[dict]:
    if namespace == "promotion":
        raise ValueError("label generation may not consume promotion seeds")
    if not candidates or not 1 <= initial <= maximum <= 64:
        raise ValueError("invalid rollout allocation")
    if not isinstance(close_margin, (int, float)) or isinstance(close_margin, bool) or not math.isfinite(close_margin) or not 0 <= close_margin <= 1:
        raise ValueError("close margin must be finite and between 0 and 1")
    if not isinstance(extension_batch_size, int) or isinstance(extension_batch_size, bool) or not 1 <= extension_batch_size <= 64:
        raise ValueError("extension batch size must be an integer from 1 to 64")
    if not isinstance(rollout_workers, int) or isinstance(rollout_workers, bool) or not 1 <= rollout_workers <= 8:
        raise ValueError("rollout workers must be an integer from 1 to 8")
    candidate_hashes = [item.key() for item in candidates]
    if len(set(candidate_hashes)) != len(candidate_hashes):
        raise ValueError("macro candidates must have unique hashes")
    records = {key: {"scores": [], "finished": 0, "truncated": 0, "error": 0,
                     "reasons": {}, "decisionCounts": {}} for key in candidate_hashes}
    completed_initial: set[int] = set()
    completed_extension: set[int] = set()
    close_candidate_hashes: list[str] | None = None
    allocation = {"initial": initial, "maximum": maximum,
                  "closeMargin": close_margin, "extensionBatchSize": extension_batch_size}
    if resume_state is not None:
        if resume_state.get("allocation") != allocation:
            raise ValueError("macro rollout checkpoint allocation mismatch")
        if resume_state.get("candidateHashes") != candidate_hashes:
            raise ValueError("macro rollout checkpoint candidate set mismatch")
        prior_records = resume_state.get("records")
        if not isinstance(prior_records, dict) or set(prior_records) != set(candidate_hashes):
            raise ValueError("macro rollout checkpoint records mismatch")
        records = prior_records
        completed_initial = set(resume_state.get("completedInitialIndices", []))
        completed_extension = set(resume_state.get("completedExtensionIndices", []))
        if (any(type(index) is not int or not 0 <= index < initial for index in completed_initial)
                or any(type(index) is not int or not initial <= index < maximum for index in completed_extension)):
            raise ValueError("macro rollout checkpoint contains invalid sample indices")
        close_candidate_hashes = resume_state.get("closeCandidateHashes")
        if close_candidate_hashes is not None and (not isinstance(close_candidate_hashes, list)
                or any(key not in records for key in close_candidate_hashes)):
            raise ValueError("macro rollout checkpoint close-candidate set mismatch")

    def save_progress():
        if checkpoint is not None:
            checkpoint({"allocation": allocation, "candidateHashes": candidate_hashes, "records": records,
                        "completedInitialIndices": sorted(completed_initial),
                        "completedExtensionIndices": sorted(completed_extension),
                        "closeCandidateHashes": close_candidate_hashes})

    def consume(candidate, outcome):
        status = outcome.get("status")
        score = outcome.get("score")
        if (status == "finished" and isinstance(score, (int, float))
                and not isinstance(score, bool) and math.isfinite(score)
                and 0 <= score <= 1):
            record = records[candidate.key()]
            record["scores"].append(float(score))
            record["finished"] += 1
            decision_count = outcome.get("decisionCount")
            if isinstance(decision_count, int) and not isinstance(decision_count, bool) and decision_count >= 0:
                key = str(decision_count)
                record["decisionCounts"][key] = record["decisionCounts"].get(key, 0) + 1
        elif status in {"truncated", "error"}:
            record = records[candidate.key()]
            record[status] += 1
            reason = outcome.get("reason")
            if isinstance(reason, str) and reason:
                record["reasons"][reason] = record["reasons"].get(reason, 0) + 1
            decision_count = outcome.get("decisionCount")
            if isinstance(decision_count, int) and not isinstance(decision_count, bool) and decision_count >= 0:
                key = str(decision_count)
                record["decisionCounts"][key] = record["decisionCounts"].get(key, 0) + 1
        else:
            raise ValueError("rollout returned an invalid status")

    executor = ThreadPoolExecutor(max_workers=rollout_workers) if rollout_workers > 1 else None
    def run(indices, selected, *, extension: bool):
        for index in indices:
            completed = completed_extension if extension else completed_initial
            if index in completed:
                continue
            seed = rollout_seed(namespace, position_hash, index, rollout_identity)
            outcomes = ([rollout(candidate, seed) for candidate in selected] if executor is None
                        else list(executor.map(lambda candidate: rollout(candidate, seed), selected)))
            for candidate, outcome in zip(selected, outcomes):
                consume(candidate, outcome)
            completed.add(index)
            save_progress()

    def close_candidates(active_hashes: list[str]) -> list[str]:
        intervals = {}
        stage_count = 1 + math.ceil((maximum - initial) / extension_batch_size)
        # Union-bound the per-candidate intervals across every candidate and
        # every planned look. This keeps monotone pruning conservative despite
        # repeated stage-boundary checks.
        alpha = .05 / (len(candidate_hashes) * stage_count)
        log_term = math.log(2 / alpha)
        for key in active_hashes:
            scores = records[key]["scores"]
            if not scores:
                intervals[key] = (0.0, 1.0)
                continue
            mean = sum(scores) / len(scores)
            radius = math.sqrt(log_term / (2 * len(scores)))
            intervals[key] = (max(0.0, mean - radius), min(1.0, mean + radius))
        best_lower = max(lower for lower, _upper in intervals.values())
        return [key for key in active_hashes
                if intervals[key][1] >= best_lower - close_margin]

    try:
        run(range(initial), candidates, extension=False)
        if close_candidate_hashes is None:
            # W/D/L-derived scores are bounded in [0, 1]. Use conservative
            # 95% Hoeffding intervals so tiny completed samples do not make a
            # candidate look confidently worse merely because many attempts
            # were truncated.
            close_candidate_hashes = close_candidates(candidate_hashes)
            save_progress()
        elif completed_extension:
            last = max(completed_extension)
            stage_start = initial + ((last - initial) // extension_batch_size) * extension_batch_size
            stage_end = min(maximum, stage_start + extension_batch_size)
            if all(index in completed_extension for index in range(stage_start, stage_end)):
                # Recover a crash after the last per-index checkpoint but before
                # the stage's confidence-set checkpoint.
                close_candidate_hashes = close_candidates(close_candidate_hashes)
                save_progress()
        while maximum > initial and close_candidate_hashes:
            missing = [index for index in range(initial, maximum) if index not in completed_extension]
            if not missing:
                close_candidate_hashes = close_candidates(close_candidate_hashes)
                save_progress()
                break
            start = missing[0]
            stage_end = min(maximum, initial +
                            (((start - initial) // extension_batch_size) + 1) * extension_batch_size)
            active = set(close_candidate_hashes)
            run(range(start, stage_end),
                [candidate for candidate in candidates if candidate.key() in active],
                extension=True)
            # Pruning is monotone so active plans retain a common seed history.
            # Re-evaluate after each complete batch and persist the new active set.
            close_candidate_hashes = close_candidates(close_candidate_hashes)
            save_progress()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    finite_means = [sum(record["scores"]) / len(record["scores"])
                    for record in records.values() if record["scores"]]
    center = max(finite_means) if finite_means else 0.0
    output = []
    attempted = {key: record["finished"] + record["truncated"] + record["error"]
                 for key, record in records.items()}
    for candidate in candidates:
        record = records[candidate.key()]
        scores = record["scores"]
        mean = sum(scores) / len(scores) if scores else None
        uncertainty = math.sqrt(max((mean or 0) * (1 - (mean or 0)), .25) / len(scores)) if scores else None
        output.append({"candidate": asdict(candidate), "candidateHash": candidate.key(),
                       "completedRollouts": len(scores), "attemptedRollouts": attempted[candidate.key()],
                       "outcomes": {key: record[key] for key in ("finished", "truncated", "error")},
                       "outcomeReasons": dict(sorted(record["reasons"].items())),
                       "decisionCountDistribution": dict(sorted(record["decisionCounts"].items(), key=lambda item: int(item[0]))),
                       "expectedResult": mean, "relativeResult": mean - center if mean is not None else None,
                       "uncertainty": uncertainty, "weight": 0 if not scores else len(scores) / (1 + uncertainty)})
    return output
