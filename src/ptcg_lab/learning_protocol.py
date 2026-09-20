"""Pure schedules and adoption checks for the explicitly experimental lane."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from .evaluation import confidence_interval, REQUIRED_ARCHETYPES


DEFAULTS = {"seed": 42, "keepAwake": True, "gamesPerBatch": 50, "searchBudgetMs": 200,
            "investigationPositions": 20, "investigationBudgetMs": 2000, "maxPositions": 20000,
            "comparisonSeeds": 2, "comparisonGameLimit": 0, "maxDecisions": 3000, "maxCycles": 0}


def configuration(values=None):
    values = values or {}
    if set(values) - set(DEFAULTS):
        raise ValueError("Unknown learning configuration field")
    result = {**DEFAULTS, **values}
    bounds = {"seed": (0, 2**32-1), "gamesPerBatch": (10, 500), "searchBudgetMs": (0, 2000),
              "investigationPositions": (0, 100), "investigationBudgetMs": (1, 5000),
              "maxPositions": (100, 20000), "comparisonSeeds": (1, 10), "comparisonGameLimit": (0, 2000),
              "maxDecisions": (1, 3000), "maxCycles": (0, 100000)}
    for key, (minimum, maximum) in bounds.items():
        if type(result[key]) is not int or not minimum <= result[key] <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
    if type(result["keepAwake"]) is not bool:
        raise ValueError("keepAwake must be boolean")
    return result


def collection_assignment(index, deck_ids, opponents):
    """Latin ordering visits every list early, then all pairs and seat/start cells.

    Keeping index global (not batch-local) is essential: fifty-game batches must
    not repeatedly discard the second half of the pair schedule.
    """
    _schedule_inputs(index, deck_ids)
    count = len(deck_ids)
    if not opponents:
        raise ValueError("A schedule needs decks and frozen opponents")
    pair_index, rotation = index % (count * count), index // (count * count)
    left = pair_index % count
    right = (pair_index // count + left) % count
    seat = (rotation + pair_index) % 2
    first = (rotation // 2 + pair_index // 2) % 2
    pair = [deck_ids[left], deck_ids[right]]
    return {"decks": pair if seat == 0 else pair[::-1], "candidateSeat": seat,
            "firstPlayer": first, "pairKey": " / ".join(pair),
            "opponent": opponents[(rotation // 4 + pair_index) % len(opponents)]}


def comparison_assignment(index, deck_ids, seeds):
    _schedule_inputs(index, deck_ids)
    if type(seeds) is not int or seeds < 1:
        raise ValueError("Comparison needs a positive seed count")
    count = len(deck_ids)
    pair_index, offset = divmod(index, seeds * 2)
    left, right = divmod(pair_index, count)
    if left >= count:
        raise ValueError("Comparison schedule exhausted")
    seed_offset, seat = divmod(offset, 2)
    pair = [deck_ids[left], deck_ids[right]]
    return {"decks": pair if seat == 0 else pair[::-1], "candidateSeat": seat,
            "firstPlayer": seed_offset % 2, "pairKey": " / ".join(pair), "pairIndex": pair_index,
            "seedOffset": seed_offset, "pairNumber": pair_index * seeds + seed_offset}


def game_seed(run_seed, purpose, ordinal):
    # Phase ranges are disjoint; within a range this is a deterministic hash,
    # not a promise that arbitrarily many ordinals can never collide.
    if type(run_seed) is not int or not 0 <= run_seed < 2**32:
        raise ValueError("Run seed must be a uint32 integer")
    offsets = {"collection": 0, "comparison": 1_000_000_000, "investigation": 2_000_000_000}
    if purpose not in offsets or not (type(ordinal) is int and ordinal >= 0 or isinstance(ordinal, str) and ordinal):
        raise ValueError("Choose a supported seed purpose and a nonnegative ordinal or nonempty key")
    value = int.from_bytes(hashlib.sha256(f"{run_seed}:{purpose}:{ordinal}".encode()).digest()[:8], "big")
    return value % 900_000_000 + offsets[purpose]


def _schedule_inputs(index, deck_ids):
    if type(index) is not int or index < 0:
        raise ValueError("Schedule index must be a nonnegative integer")
    if (not isinstance(deck_ids, (list, tuple)) or not deck_ids or
            any(not isinstance(deck, str) or not deck for deck in deck_ids) or len(set(deck_ids)) != len(deck_ids)):
        raise ValueError("A schedule needs distinct, nonempty deck IDs")


def _terminal_score(game):
    score = game.get("score")
    return (game.get("status") == "finished" and type(score) in {int, float}
            and math.isfinite(score) and score in {0, .5, 1})


def comparison_summary(records, expected_games, tactical, *, deck_ids=None, seeds=None, archetypes=None):
    """Adopt only a complete, exact, frozen comparison schedule.

    Merely collecting the right number of records or naming 25 matchups does not
    establish coverage. Legacy summaries without their schedule remain diagnostic.
    """
    if type(expected_games) is not int or expected_games < 2 or expected_games % 2:
        raise ValueError("Expected comparisons must be a positive number of seat pairs")
    notes, schedule = [], None
    if deck_ids is None or seeds is None or archetypes is None:
        notes.append("Exact frozen deck/seed schedule metadata is missing.")
    else:
        _schedule_inputs(0, deck_ids)
        if type(seeds) is not int or seeds < 1:
            raise ValueError("Comparison needs a positive seed count")
        total = len(deck_ids)**2 * seeds * 2
        schedule = [comparison_assignment(index, deck_ids, seeds) for index in range(total)]
        if expected_games != total:
            notes.append("Expected game count differs from the frozen schedule.")
        if seeds < 2:
            notes.append("At least two seeds are needed for all four seat/start cells.")
        if (set(archetypes) != set(deck_ids) or set(archetypes.values()) != REQUIRED_ARCHETYPES
                or len(deck_ids) != 10 or any(list(archetypes.values()).count(a) != 2 for a in REQUIRED_ARCHETYPES)):
            notes.append("The frozen schedule must contain two lists for each of the five supported archetypes.")
    identities = [game.get("gameId") for game in records]
    if any(not isinstance(identifier, str) or not identifier for identifier in identities) or len(set(identities)) != len(identities):
        notes.append("Comparison game identities are missing or duplicated.")
    pairs = defaultdict(list)
    invalid_records = 0
    for game in records:
        if type(game.get("pairNumber")) is not int or game["pairNumber"] < 0 or type(game.get("candidateSeat")) is not int or game["candidateSeat"] not in {0, 1}:
            invalid_records += 1
            continue
        pairs[game["pairNumber"]].append(game)
    if schedule is not None and set(pairs) != set(range(len(schedule)//2)):
        notes.append("Comparison pair IDs differ from the exact scheduled set.")
    scores, by_matchup, excluded = [], defaultdict(list), 0
    seen_seeds = set()
    for pair_number, games in pairs.items():
        valid = len(games) == 2 and {g["candidateSeat"] for g in games} == {0, 1} and all(_terminal_score(g) for g in games)
        if valid:
            valid = all(g.get("archetypePair") == games[0].get("archetypePair") for g in games) and bool(games[0].get("archetypePair"))
        if valid and schedule is not None:
            for game in games:
                index = pair_number * 2 + game["candidateSeat"]
                if index >= len(schedule):
                    valid = False
                    break
                expected = schedule[index]
                if any(game.get(key) != value for key, value in expected.items()):
                    valid = False
                    break
                pair_decks = expected["decks"] if expected["candidateSeat"] == 0 else expected["decks"][::-1]
                if game["archetypePair"] != " / ".join(archetypes.get(deck, "unknown") for deck in pair_decks):
                    valid = False
                    break
            seed = games[0].get("seed")
            if (type(seed) is not int or not 1_000_000_000 <= seed < 1_900_000_000
                    or games[1].get("seed") != seed or seed in seen_seeds):
                valid = False
            else:
                seen_seeds.add(seed)
        if not valid:
            excluded += 1
            continue
        score = sum(g["score"] for g in games) / 2
        scores.append(score)
        by_matchup[games[0]["archetypePair"]].append(score)
    overall = confidence_interval(scores)
    matchups = {key: confidence_interval(values) for key, values in by_matchup.items()}
    if len(records) != expected_games or excluded or invalid_records:
        notes.append("Comparison has missing, failed, or truncated seat pairs.")
    if overall["lower"] is None or overall["lower"] <= .5 or len(scores) < 30:
        notes.append("Aggregate experimental improvement is inconclusive.")
    expected_matchups = {f"{left} / {right}" for left in REQUIRED_ARCHETYPES for right in REQUIRED_ARCHETYPES}
    if set(matchups) != expected_matchups or any(item["pairs"] < 5 for item in matchups.values()):
        notes.append("All 25 archetype matchups need at least five complete pairs.")
    if any(item["upper"] < .5 for item in matchups.values()):
        notes.append("A matchup has a supported regression.")
    regressions = tactical.get("regressions")
    if (tactical.get("status") != "measured" or type(tactical.get("positions")) is not int or tactical["positions"] < 1
            or not (regressions == [] or type(regressions) is int and regressions == 0)
            or tactical.get("passes") is False):
        notes.append("Reviewed tactical checks are unavailable or regressed.")
    return {"status": "completed" if len(records) == expected_games and not excluded and not invalid_records else "incomplete", "completedGames": sum(_terminal_score(game) for game in records), "totalGames": expected_games,
            **overall, "matchups": matchups, "excludedPairs": excluded,
            "adopted": not notes, "notes": notes, "invalidRecords": invalid_records,
            "scope": "Experimental simulator comparison, not trusted champion promotion."}
