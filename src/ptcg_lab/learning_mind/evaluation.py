from __future__ import annotations

import math
from collections import Counter, defaultdict

from .macro import rollout_seed


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
    elif interval["low"] is not None and interval["low"] > .5:
        status = "supported-improvement"
    elif interval["high"] is not None and interval["high"] < .5 - non_regression_margin:
        status = "supported-regression"
    elif n < 250:
        status = "continue-to-250"
    elif n < 500:
        status = "continue-to-500"
    else:
        status = "inconclusive-at-cap"
    return {"status": status, "completed": n, "wins": wins, "draws": n - decisive, "losses": losses,
            "unfinished": {key: counts[key] for key in ("truncated", "error")}, "wilson95DecisiveWinRate": interval}


def promotion_gate(*, aggregate: dict, matchups: list[dict], strategy: dict,
                   blind_family_passed: bool, identities_match: bool, human_approved: bool) -> dict:
    reasons = []
    if aggregate.get("status") != "supported-improvement": reasons.append("aggregate improvement unsupported")
    if any(row.get("regressionPoints", 0) > 5 for row in matchups): reasons.append("critical matchup regressed over five points")
    if strategy.get("severityThreeRegressions"): reasons.append("severity-three probe regression")
    if not blind_family_passed: reasons.append("blind opponent-policy family failed")
    if not identities_match: reasons.append("evaluation identity drift")
    if not human_approved: reasons.append("explicit human approval missing")
    return {"promotable": not reasons, "reasons": reasons, "automaticPromotion": False}
