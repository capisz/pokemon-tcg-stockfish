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
    if not isinstance(records, list):
        raise ValueError("promotion records must be a list")
    if (isinstance(non_regression_margin, bool)
            or not isinstance(non_regression_margin, (int, float))
            or not math.isfinite(non_regression_margin)
            or not 0 <= non_regression_margin <= 1):
        raise ValueError("non-regression margin must be finite and between zero and one")
    counts = Counter({"finished": 0, "truncated": 0, "error": 0})
    complete = []
    seen_game_ids = set()
    for row in records:
        if not isinstance(row, dict) or row.get("status") not in counts:
            raise ValueError("promotion record has an unknown or malformed outcome status")
        game_id = row.get("gameId")
        if game_id is not None:
            if not isinstance(game_id, str) or not game_id or game_id in seen_game_ids:
                raise ValueError("promotion game IDs must be nonempty and unique")
            seen_game_ids.add(game_id)
        counts[row["status"]] += 1
        score = row.get("score")
        if row["status"] == "finished":
            if (type(score) not in {int, float} or not math.isfinite(score)
                    or score not in {0, .5, 1}):
                raise ValueError("finished promotion records require an exact win/draw/loss score")
            complete.append(row)
        elif score is not None:
            raise ValueError("truncated and errored promotion records may not contain scores")
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


def matched_sequential_decision(candidate_records: list[dict], control_records: list[dict], *,
                                non_regression_margin: float = .05) -> dict:
    """Compare a candidate with its control only on exactly matched scheduled game pairs."""
    metadata_fields = ("seed", "ownArchetype", "opponentArchetype", "learnerSeat",
                       "firstPlayer", "opponentPolicyFamily", "schedulerIdentity")

    def index(records: list[dict], label: str) -> dict[str, dict]:
        if not isinstance(records, list):
            raise ValueError(f"{label} promotion records must be a list")
        sequential_decision(records, non_regression_margin=non_regression_margin)
        indexed: dict[str, dict] = {}
        for row in records:
            pair_id, game_id = row.get("pairId"), row.get("gameId")
            if (not isinstance(pair_id, str) or not pair_id or pair_id in indexed
                    or not isinstance(game_id, str) or not game_id):
                raise ValueError(f"{label} promotion records require unique pair and game IDs")
            if (type(row.get("seed")) is not int
                    or row.get("ownArchetype") not in ARCHETYPES
                    or row.get("opponentArchetype") not in ARCHETYPES
                    or type(row.get("learnerSeat")) is not int or row["learnerSeat"] not in (0, 1)
                    or type(row.get("firstPlayer")) is not int or row["firstPlayer"] not in (0, 1)
                    or not isinstance(row.get("opponentPolicyFamily"), str)
                    or not row["opponentPolicyFamily"]
                    or not isinstance(row.get("schedulerIdentity"), str)
                    or not row["schedulerIdentity"]):
                raise ValueError(f"{label} promotion record lacks valid frozen matchup assignments")
            indexed[pair_id] = row
        return indexed

    candidates = index(candidate_records, "candidate")
    controls = index(control_records, "control")
    if not candidates or candidates.keys() != controls.keys():
        raise ValueError("candidate and control records do not cover the same scheduled game pairs")
    paired_records = []
    candidate_statuses = Counter()
    control_statuses = Counter()
    for pair_id in sorted(candidates):
        candidate, control = candidates[pair_id], controls[pair_id]
        if any(candidate[field] != control[field] for field in metadata_fields):
            raise ValueError("candidate and control game pair has mismatched frozen assignments")
        candidate_statuses[candidate["status"]] += 1
        control_statuses[control["status"]] += 1
        if candidate["status"] == control["status"] == "finished":
            delta = candidate["score"] - control["score"]
            paired_records.append({"gameId": pair_id, "status": "finished",
                                   "score": 1 if delta > 0 else 0 if delta < 0 else .5})
        else:
            status = ("error" if "error" in {candidate["status"], control["status"]}
                      else "truncated")
            paired_records.append({"gameId": pair_id, "status": status})
    result = sequential_decision(paired_records, non_regression_margin=non_regression_margin)
    return {**result, "matchedPairs": len(paired_records),
            "candidateOutcomes": {key: candidate_statuses[key] for key in ("finished", "truncated", "error")},
            "controlOutcomes": {key: control_statuses[key] for key in ("finished", "truncated", "error")}}


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
