from __future__ import annotations

import math
from collections import Counter, defaultdict

from .macro import rollout_seed
from .curriculum import ARCHETYPES

PROMOTION_MATCHUPS = frozenset((own, opponent)
    for own in ARCHETYPES for opponent in ARCHETYPES)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0: return {"low": None, "high": None}
    p = k / n; denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return {"low": center - margin, "high": center + margin}


def promotion_seed(position: str, index: int) -> int:
    return rollout_seed("promotion", position, index)


def sequential_decision(records: list[dict], *, non_regression_margin: float = .05) -> dict:
    complete = [row for row in records if row.get("status") == "finished" and row.get("score") in {0, .5, 1}]
    counts = Counter(row.get("status", "unknown") for row in records)
    n = len(complete)
    wins = sum(row["score"] == 1 for row in complete)
    losses = sum(row["score"] == 0 for row in complete)
    decisive = wins + losses
    interval = wilson(wins, decisive)
    if n < 100:
        status = "continue-to-100"
    elif decisive < 100 and n < 250:
        status = "continue-to-250"
    elif decisive < 100 and n < 500:
        status = "continue-to-500"
    elif decisive < 100:
        status = "inconclusive-at-cap"
    elif interval["low"] is not None and interval["low"] > .5:
        status = "supported-improvement"
    elif interval["high"] is not None and interval["high"] < .5 - non_regression_margin:
        status = "supported-regression"
    elif interval["low"] is not None and interval["low"] >= .5 - non_regression_margin:
        status = "supported-non-regression"
    elif n < 250:
        status = "continue-to-250"
    elif n < 500:
        status = "continue-to-500"
    else:
        status = "inconclusive-at-cap"
    return {"status": status, "completed": n, "decisive": decisive,
            "wins": wins, "draws": n - decisive, "losses": losses,
            "unfinished": {key: counts[key] for key in ("truncated", "error")}, "wilson95DecisiveWinRate": interval}


def promotion_gate(*, aggregate: dict, matchups: list[dict], strategy: dict,
                   blind_family_passed: bool, identities_match: bool, human_approved: bool) -> dict:
    reasons = []
    if aggregate.get("status") != "supported-improvement": reasons.append("aggregate improvement unsupported")
    if type(aggregate.get("completed")) is not int or aggregate["completed"] < 100:
        reasons.append("aggregate minimum of 100 completed games not met")
    if type(aggregate.get("decisive")) is not int or aggregate["decisive"] < 100:
        reasons.append("aggregate minimum of 100 decisive games for Wilson support not met")
    if not isinstance(matchups, list):
        reasons.append("ordered matchup evidence is malformed")
    else:
        observed = {}
        malformed = False
        for row in matchups:
            if not isinstance(row, dict):
                malformed = True
                continue
            key = (row.get("ownArchetype"), row.get("opponentArchetype"))
            if (key not in PROMOTION_MATCHUPS or key in observed
                    or type(row.get("completed")) is not int
                    or row["completed"] < 100
                    or type(row.get("decisive")) is not int
                    or row["decisive"] < 100
                    or row.get("status") not in {"supported-improvement", "supported-non-regression"}
                    or type(row.get("regressionPoints")) not in {int, float}
                    or not math.isfinite(row["regressionPoints"])):
                malformed = True
                continue
            observed[key] = row
        if malformed or set(observed) != PROMOTION_MATCHUPS:
            reasons.append("all 25 ordered matchups need unique evidence, at least 100 completed games, and a resolved confidence gate")
        if any(row.get("regressionPoints", math.inf) > 5 for row in observed.values()):
            reasons.append("critical matchup regressed over five points")
    if not isinstance(strategy, dict) or strategy.get("severityThreeRegressions") != []:
        reasons.append("severity-three probe regression status is missing or failed")
    if blind_family_passed is not True: reasons.append("blind opponent-policy family failed")
    if identities_match is not True: reasons.append("evaluation identity drift")
    if human_approved is not True: reasons.append("explicit human approval missing")
    return {"promotable": not reasons, "reasons": reasons, "automaticPromotion": False}
