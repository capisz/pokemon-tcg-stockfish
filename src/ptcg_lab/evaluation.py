from __future__ import annotations

import itertools
import os
import shutil
import uuid
from pathlib import Path

import numpy as np

from .config import Settings
from .engine import EngineClient
from .selfplay import play_game, admitted_decks
from .storage import Store, digest, file_digest, terminal_score
from .resources import check_storage


def confidence_interval(scores: list[float], seed: int = 0) -> dict:
    if not scores:
        return {"pairs": 0, "mean": None, "lower": None, "upper": None}
    values = np.asarray(scores)
    # Hoeffding bounds remain conservative for tiny/all-identical samples where
    # an ordinary bootstrap would give a misleading zero-width interval.
    radius = np.sqrt(np.log(40) / (2 * len(values)))
    mean = float(np.mean(values))
    return {"pairs": len(scores), "mean": mean, "lower": max(0., mean - float(radius)),
            "upper": min(1., mean + float(radius)), "method": "95% bounded-sample Hoeffding interval"}


def promotion_decision(overall: dict, matchups: dict, excluded: int) -> tuple[bool, list[str]]:
    reasons = []
    if overall["pairs"] < 30 or overall["lower"] is None or overall["lower"] <= .5:
        reasons.append("Overall improvement is statistically inconclusive or has fewer than 30 complete pairs.")
    if any(result["pairs"] < 5 for result in matchups.values()):
        reasons.append("Every matchup needs at least five complete seat-balanced pairs.")
    if any(result["upper"] is not None and result["upper"] < .5 for result in matchups.values()):
        reasons.append("A matchup shows a supported regression and needs investigation.")
    if excluded:
        reasons.append("Truncated/error pairs remain; investigate completion bias before promotion.")
    return not reasons, reasons


REQUIRED_ARCHETYPES = {"dragapult", "raging-bolt", "grimmsnarl", "mega-lucario", "crustle"}
REQUIRED_ROLES = {"main", "training-variant", "heldout"}


def coverage_failures(registered: list[dict], admitted: list[dict]) -> list[str]:
    """The denominator is the frozen registry, never its eligible subset."""
    required = [deck for deck in registered if deck.get("role") in REQUIRED_ROLES]
    missing_lists = {deck["id"] for deck in required} - {deck["id"] for deck in admitted}
    missing_groups = {(archetype, role) for archetype in REQUIRED_ARCHETYPES for role in REQUIRED_ROLES} - {
        (deck.get("archetype"), deck.get("role")) for deck in admitted}
    reasons = []
    if missing_lists:
        reasons.append("Promotion requires every registered competitive list; omitted: " + ", ".join(sorted(missing_lists)))
    if missing_groups:
        reasons.append("Promotion requires all five archetypes and all main/training/heldout roles; missing: " +
                       ", ".join(f"{archetype}:{role}" for archetype, role in sorted(missing_groups)))
    return reasons


def _verified_checkpoint_copy(store: Store, source: Path, destination: Path, expected_hash: str, *, replace: bool = False) -> None:
    if destination.exists() and not replace:
        raise ValueError("Immutable evaluation checkpoint already exists")
    check_storage(store.path, store.max_bytes, store.min_free_bytes, source.stat().st_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if file_digest(temporary) != expected_hash:
            raise ValueError("Checkpoint changed during evaluation copy; no checkpoint was published")
        with temporary.open("rb") as artifact:
            os.fsync(artifact.fileno())
        check_storage(store.path, store.max_bytes, store.min_free_bytes)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_frozen_models(policies: dict, identities: dict) -> None:
    for role, policy in policies.items():
        if policy not in {"random", "heuristic"} and file_digest(Path(policy)) != identities[role]:
            raise ValueError("Frozen evaluation checkpoint changed; this experiment cannot publish results or promote")


def evaluate(settings: Settings, *, candidate: str = "heuristic", opponent: str = "random", seeds: int = 2,
             seed_start: int = 1_000_000_000, max_decisions: int = 1000, promote: bool = False,
             allow_unverified: bool = False, guard=None) -> dict:
    if not 1 <= seeds <= 1000 or not 1 <= max_decisions <= settings.max_decisions:
        raise ValueError("Evaluation requires 1..1000 paired seeds and 1..3000 decisions")
    if not 0 <= seed_start < 2**32:
        raise ValueError("Evaluation seed-start must be uint32")
    store = Store(settings.data, settings.max_disk_bytes, settings.min_free_bytes)
    excluded_seeds: set[int] = set()
    models, frozen_policies, snapshots = {}, {}, {}
    identifier = uuid.uuid4().hex
    for name, policy in (("candidate", candidate), ("opponent", opponent)):
        if policy not in {"random", "heuristic"}:
            from .training import load_model
            source = Path(policy).resolve()
            models[name] = file_digest(source)
            frozen = store.path / "evaluation-models" / f"{identifier}-{name}-{models[name][:16]}.pt"
            _verified_checkpoint_copy(store, source, frozen, models[name])
            _, checkpoint = load_model(frozen)
            frozen_policies[name] = str(frozen)
            snapshots[name] = {"source": str(source), "path": frozen.relative_to(store.path).as_posix(),
                               "sha256": models[name]}
            from .training import checkpoint_lineage
            excluded_seeds.update(checkpoint_lineage(checkpoint)["seenSeeds"])
        else:
            models[name] = frozen_policies[name] = policy
    scores: list[float] = []
    by_matchup: dict[str, list[float]] = {}
    first_player_counts: dict[str, dict[str, int]] = {}
    pair_records = []
    excluded = 0
    with EngineClient(settings.root, settings.engine_timeout) as engine:
        registry = engine.request("decks")
        registry = registry["decks"] if isinstance(registry, dict) else registry
        registered = registry
        registry = admitted_decks(registry, allow_unverified, evaluation=True)
        coverage_reasons = coverage_failures(registered, registry)
        deck_ids = [deck["id"] for deck in registry]
        health = engine.request("health")
        trusted = not allow_unverified and all("role" not in deck or deck.get("validation", {}).get("trainingEligible") is True for deck in registry)
        protocol = {"candidate": models["candidate"], "opponent": models["opponent"], "seeds": seeds,
                    "checkpointSnapshots": snapshots,
                    "seedStart": seed_start, "deckHash": digest(registered),
                    "admittedDeckHash": digest(registry), "omittedCoverage": coverage_reasons, "engine": health,
                    "maxDecisions": max_decisions, "gate": "95% lower expected-result bound >0.5, >=30pairs, >=5/matchup, no excluded pairs or supported regressions"}
        store.put("evaluation-protocols", identifier, {"id": identifier, "protocol": protocol, "hash": digest(protocol)})
        # The candidate must pilot EACH deck against EACH opponent deck. Merely
        # swapping seats of unordered pairs leaves half those assignments untested.
        for pair_index, pair in enumerate(itertools.product(deck_ids, repeat=2)):
            key = " / ".join(pair)
            by_matchup[key] = []
            first_player_counts[key] = {"candidateFirst": 0, "opponentFirst": 0, "unknown": 0}
            for offset in range(seeds):
                seed = (seed_start + pair_index * seeds + offset) % 2**32
                if seed in excluded_seeds:
                    raise ValueError("Evaluation seed overlaps a model's train/calibration/test data; choose new seeds")
                results = []
                replay_ids = []
                first_players = []
                for seat in (0, 1):
                    if guard:
                        guard.check()
                    _verify_frozen_models(frozen_policies, models)
                    policies = ((frozen_policies["candidate"], frozen_policies["opponent"]) if seat == 0
                                else (frozen_policies["opponent"], frozen_policies["candidate"]))
                    ordered_pair = list(pair) if seat == 0 else list(reversed(pair))
                    # When the v2 worker reports first-player control, fixing
                    # seat0 and swapping the candidate balances starts exactly.
                    options = {"first_player": 0} if health.get("firstPlayerControl") else {}
                    if guard:
                        options["guard"] = guard
                    replay = play_game(engine, decks=ordered_pair, seed=seed, policies=policies, max_decisions=max_decisions, **options)
                    replay["evaluationExperiment"] = identifier
                    replay["trainingEligible"] = False
                    replay["deckRoles"] = [next(deck.get("role", "main") for deck in registry if deck["id"] == deck_id) for deck_id in ordered_pair]
                    replay_ids.append(store.save_replay(replay))
                    first = replay.get("startingPlayer")
                    first_players.append("unknown" if first is None else "candidateFirst" if first == seat else "opponentFirst")
                    if replay["status"] == "finished" and replay.get("outcome"):
                        results.append(terminal_score(replay, seat))
                if len(results) == 2:
                    score = float(np.mean(results))
                    scores.append(score)
                    by_matchup[key].append(score)
                    for first in first_players:
                        first_player_counts[key][first] += 1
                else:
                    excluded += 1
                pair_records.append({"decks": list(pair), "seed": seed, "replays": replay_ids,
                                     "firstPlayers": first_players,
                                     "score": float(np.mean(results)) if len(results) == 2 else None})
                store.put("experiments", identifier, {"id": identifier, "type": "paired-evaluation", "status": "running",
                          "candidate": candidate, "opponent": opponent, "completedPairs": len(pair_records), "excludedPairs": excluded})
    _verify_frozen_models(frozen_policies, models)
    summary = confidence_interval(scores)
    matchup_results = {key: confidence_interval(values) for key, values in by_matchup.items()}
    eligible, reasons = promotion_decision(summary, matchup_results, excluded)
    if coverage_reasons:
        eligible = False
        reasons.extend(coverage_reasons)
    if not trusted:
        eligible = False
        reasons.append("Unverified deck legality or interactions make this a rules-QA experiment; promotion is blocked.")
    if any("role" in deck for deck in registry) and not any(deck.get("role") == "heldout" for deck in registry):
        eligible = False
        reasons.append("Reserved variant coverage is required before promotion.")
    champion_path = settings.data / "models" / "champion.pt"
    if champion_path.exists():
        if opponent in {"random", "heuristic"} or models["opponent"] != file_digest(champion_path):
            reasons.append("Champion promotion requires comparison against the installed champion checkpoint.")
            eligible = False
    elif opponent != "heuristic":
        reasons.append("The initial learned champion must beat the strategy heuristic; no other opponent can satisfy the bootstrap gate.")
        eligible = False
    if any(counts["unknown"] or counts["candidateFirst"] != counts["opponentFirst"] for counts in first_player_counts.values()):
        reasons.append("Actual first-player counts are unknown or imbalanced in at least one matchup; swapped seats alone do not guarantee first-player balance.")
        eligible = False
    promoted = False
    if promote and eligible:
        if candidate in {"random", "heuristic"}:
            reasons.append("Only a learned checkpoint can be installed as model champion.")
        else:
            _verified_checkpoint_copy(store, Path(frozen_policies["candidate"]), champion_path,
                                      models["candidate"], replace=True)
            promoted = True
    report = {"id": identifier, "type": "paired-evaluation", "status": "completed", "candidate": candidate,
              "opponent": opponent, "models": models, "checkpointSnapshots": snapshots, "engine": health, "deckHash": digest(registry),
              "seedStart": seed_start, "protocolHash": digest(protocol), "summary": summary, "matchups": matchup_results,
              "trustedRulesCoverage": trusted,
              "firstPlayerCounts": first_player_counts,
              "pairs": pair_records, "excludedPairs": excluded, "promotionEligible": eligible,
              "promoted": promoted, "promotionNotes": reasons,
              "limitations": ["Scores measure only these frozen decklists, opponents and decision budgets.",
                              "Seats are balanced by construction; actual first-player choices are separately measured.",
                              "Incomplete seat pairs are excluded and block automatic promotion."]}
    store.put("experiments", identifier, report)
    return report
