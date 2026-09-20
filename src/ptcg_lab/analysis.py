from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, deck_beliefs, heuristic_action_score, resource_features

HEURISTIC_WEIGHTS = np.array([2.5, .8, .4, .9, .25, .1, .4, .8, .15, .15, .1, .2, -.5, .2, .3, -.1])


def analyze_observation(observation: dict, decks: list[dict], model_path: Path | None = None) -> dict:
    """Only a single player's view enters the analysis boundary.

    In particular this function has no replay, future frames, RNG state, or hidden
    deck order. A budget cannot manufacture rollouts when branching is unavailable.
    """
    beliefs, warnings = deck_beliefs(observation, decks)
    warnings.extend(observation.get("warnings", []))
    if model_path and model_path.exists():
        from .training import predict
        evaluation, scores = predict(model_path, observation)
        label = "Learned policy preference, not a searched continuation or mistake estimate."
    else:
        contributions = resource_features(observation) * HEURISTIC_WEIGHTS
        evaluation = {"status": "heuristic", "score": round(float(sum(contributions)), 4),
                      "expectedResult": None, "winProbability": None, "drawProbability": None, "lossProbability": None,
                      "modelVersion": "resource-heuristic-v1", "calibrated": False,
                      "components": [{"name": name, "value": round(float(value), 4)} for name, value in zip(FEATURE_NAMES, contributions)],
                      "description": "Untrained resource index. These hand-set contributions are not advantage units or probabilities; opponent private resources are unknown."}
        scores = [heuristic_action_score(action, observation) for action in observation.get("legalActions", [])]
        label = "Static action heuristic; no simulation visits, expected-result estimate, or optimality claim."
    alternatives = [{"actionId": action["id"], "label": action["label"], "score": round(float(score), 4),
                     "visits": 0, "description": label}
                    for action, score in zip(observation.get("legalActions", []), scores)]
    alternatives.sort(key=lambda item: item["score"], reverse=True)
    warnings.append("Search and calibrated mistake grading are unavailable in this foundation build; alternatives are policy rankings.")
    return {"evaluation": evaluation, "alternatives": alternatives[:20], "beliefs": beliefs, "warnings": list(dict.fromkeys(warnings))}


def select_frame(replay: dict, decision_index: int, player_id: int) -> dict:
    if player_id not in (0, 1):
        raise ValueError("Player id must be 0 or 1")
    frame = next((frame for frame in replay.get("frames", []) if frame["decisionIndex"] == decision_index), None)
    if frame is None:
        raise ValueError("Decision index is not present in this replay")
    # Do not forward the frame (which contains BOTH private observations) to analysis.
    observation = frame["observations"][player_id]
    if observation["playerId"] != player_id:
        raise ValueError("Replay observation perspective is inconsistent")
    return observation


def selected_action_id(replay: dict, decision_index: int, player_id: int) -> str | None:
    """Read only the action attached to this exact pre-decision frame."""
    frame = next((item for item in replay.get("frames", []) if item["decisionIndex"] == decision_index), None)
    if frame is None or frame.get("actor") != player_id or not frame.get("action"):
        return None
    return frame["action"]["id"]


def review_decision(alternatives: list[dict], played_action_id: str | None) -> dict:
    """Compare sampled values without any subsequent game data or outcome.

    This is an experimental opportunity-loss estimate, never a definitive mistake
    classification. Search values may include untrained heuristic horizon leaves.
    """
    notes = ["Subsequent luck is not used or separately quantified.",
             "Sampled estimates may use heuristic horizon values; the best tested move is not proven optimal."]

    def unavailable(reason: str) -> dict:
        return {"status": "unavailable", "playedActionId": played_action_id,
                "opportunityLoss": None, "description": reason, "warnings": notes}

    if played_action_id is None:
        return unavailable("No recorded move belongs to this selected player and decision; saved positions have no recorded continuation.")
    visited = [item for item in alternatives if item.get("visits", 0) > 0
               and isinstance(item.get("expectedResult"), (int, float))
               and math.isfinite(item["expectedResult"])]
    played = next((item for item in visited if item.get("actionId") == played_action_id), None)
    if played is None:
        return unavailable("The recorded move has no sampled estimate within this search budget.")
    best = max(visited, key=lambda item: item["expectedResult"])
    if len(visited) < 2:
        return unavailable("Fewer than two distinct legal moves were sampled; no opportunity comparison is available.")
    delta = max(0., float(best["expectedResult"] - played["expectedResult"]))
    if min(played["visits"], best["visits"]) < 10:
        notes.append("Fewer than ten samples support the played move or best tested alternative; the estimate is especially uncertain.")
    standard_errors = [item.get("uncertainty") for item in (played, best)]
    uncertainty = None
    if played["actionId"] == best["actionId"]:
        uncertainty = 0.
    elif all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in standard_errors):
        uncertainty = math.sqrt(sum(float(value) ** 2 for value in standard_errors))
        if delta <= 1.96 * uncertainty:
            notes.append("The estimated difference is small relative to sampling uncertainty; no reliable separation is established.")
    else:
        notes.append("Sampling uncertainty is unavailable for at least one compared move.")
    notes.append("Selecting the largest sampled estimate can exaggerate differences; no blunder or mistake label is assigned.")
    return {"status": "experimental", "playedActionId": played_action_id, "bestTestedActionId": best["actionId"],
            "playedExpectedResult": float(played["expectedResult"]), "bestTestedExpectedResult": float(best["expectedResult"]),
            "playedVisits": played["visits"], "bestTestedVisits": best["visits"],
            "opportunityLoss": delta, "differenceStandardError": uncertainty,
            "description": "Estimated expected-result opportunity loss relative to the best sampled alternative, using pre-decision information only.",
            "warnings": notes}
