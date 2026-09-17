from __future__ import annotations

import itertools
import shutil
import uuid
from pathlib import Path

import numpy as np

from .config import Settings
from .engine import EngineClient
from .selfplay import play_game
from .storage import Store, digest, file_digest, terminal_score


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


def evaluate(settings: Settings, *, candidate: str = "heuristic", opponent: str = "random", seeds: int = 2,
             seed_start: int = 1_000_000_000, max_decisions: int = 1000, promote: bool = False) -> dict:
    if not 1 <= seeds <= 1000 or not 1 <= max_decisions <= settings.max_decisions:
        raise ValueError("Evaluation requires 1..1000 paired seeds and 1..3000 decisions")
    if not 0 <= seed_start < 2**32:
        raise ValueError("Evaluation seed-start must be uint32")
    store = Store(settings.data, settings.max_disk_bytes)
    excluded_seeds: set[int] = set()
    models = {}
    for name, policy in (("candidate", candidate), ("opponent", opponent)):
        if policy not in {"random", "heuristic"}:
            from .training import load_model
            _, checkpoint = load_model(Path(policy))
            models[name] = file_digest(Path(policy))
            for group in checkpoint["manifest"].values():
                for game in group:
                    excluded_seeds.add(store.get("replays", game["id"])["seed"])
        else:
            models[name] = policy
    identifier = uuid.uuid4().hex
    scores: list[float] = []
    by_matchup: dict[str, list[float]] = {}
    first_player_counts: dict[str, dict[str, int]] = {}
    pair_records = []
    excluded = 0
    with EngineClient(settings.root, settings.engine_timeout) as engine:
        registry = engine.request("decks")
        registry = registry["decks"] if isinstance(registry, dict) else registry
        deck_ids = [deck["id"] for deck in registry]
        health = engine.request("health")
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
                    policies = (candidate, opponent) if seat == 0 else (opponent, candidate)
                    ordered_pair = list(pair) if seat == 0 else list(reversed(pair))
                    replay = play_game(engine, decks=ordered_pair, seed=seed, policies=policies, max_decisions=max_decisions)
                    replay["evaluationExperiment"] = identifier
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
    summary = confidence_interval(scores)
    matchup_results = {key: confidence_interval(values) for key, values in by_matchup.items()}
    eligible, reasons = promotion_decision(summary, matchup_results, excluded)
    champion_path = settings.data / "models" / "champion.pt"
    if champion_path.exists():
        if opponent in {"random", "heuristic"} or file_digest(Path(opponent)) != file_digest(champion_path):
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
            champion = champion_path
            temporary = champion.with_suffix(".tmp")
            shutil.copyfile(candidate, temporary)
            temporary.replace(champion)
            promoted = True
    report = {"id": identifier, "type": "paired-evaluation", "status": "completed", "candidate": candidate,
              "opponent": opponent, "models": models, "engine": health, "deckHash": digest(registry),
              "seedStart": seed_start, "summary": summary, "matchups": matchup_results,
              "firstPlayerCounts": first_player_counts,
              "pairs": pair_records, "excludedPairs": excluded, "promotionEligible": eligible,
              "promoted": promoted, "promotionNotes": reasons,
              "limitations": ["Scores measure only these frozen decklists, opponents and decision budgets.",
                              "Seats are balanced by construction; actual first-player choices are separately measured.",
                              "Incomplete seat pairs are excluded and block automatic promotion."]}
    store.put("experiments", identifier, report)
    return report
