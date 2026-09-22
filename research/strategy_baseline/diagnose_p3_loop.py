"""Research-only trace for the frozen P3 guide-policy repetition failure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ptcg_lab.decision_guard import action_key, forward_choices, position_key, record_choice
from ptcg_lab.engine import EngineClient
from ptcg_lab.features import heuristic_action_score
from ptcg_lab.selfplay import Agent
from ptcg_lab.storage import digest, file_digest
from ptcg_lab.training import predict


CASES = [
    {"id": "cross-crustle-seat0-first0", "decks": ["crustle", "dragapult"],
     "firstPlayer": 0, "seed": 219283457},
    {"id": "crustle-mirror-first0", "decks": ["crustle", "crustle"],
     "firstPlayer": 0, "seed": 219283461},
]


def observation_for_actor(engine: EngineClient, state: dict) -> dict:
    observation = state["observation"]
    if observation["playerId"] != observation["decisionPlayer"]:
        observed = engine.request("observe", {"playerId": observation["decisionPlayer"]})
        observation = observed.get("observation", observed)
    return observation


def action_summary(action: dict, score: float | None = None) -> dict:
    result = {key: action.get(key) for key in
              ("id", "type", "label", "choiceOperation", "cardId", "target") if action.get(key) is not None}
    if score is not None:
        result["score"] = score
    return result


def prompt_summary(prompt: dict | None) -> dict | None:
    if not prompt:
        return None
    result = {key: prompt.get(key) for key in
              ("type", "message", "selectionCount", "min", "max", "canFinish", "canUndo")
              if prompt.get(key) is not None}
    result["cardIds"] = [card.get("id") for card in prompt.get("cards", [])]
    return result


def ranked_actions(checkpoint: Path, agent: Agent, observation: dict) -> tuple[dict, list[dict], list[dict]]:
    legal = observation["legalActions"]
    _, p3_scores = predict(checkpoint, observation, agent.loaded, allow_experimental=True)
    chosen_id = agent.choose(observation)
    chosen = next(action for action in legal if action["id"] == chosen_id)
    p3 = sorted((action_summary(action, float(score)) for action, score in zip(legal, p3_scores)),
                key=lambda item: item["score"], reverse=True)[:5]
    heuristic = sorted((action_summary(action, heuristic_action_score(action, observation)) for action in legal),
                       key=lambda item: item["score"], reverse=True)[:5]
    return chosen, p3, heuristic


def trace(root: Path, checkpoint: Path, case: dict, *, guarded: bool, max_decisions: int) -> dict:
    engine = EngineClient(root, timeout=300)
    agent_seed = case["seed"] * 2
    agents = [Agent(str(checkpoint), agent_seed + player, allow_experimental=True) for player in range(2)]
    repetitions: dict = {}
    pair_counts: Counter = Counter()
    position_counts: Counter = Counter()
    representatives = {}
    examples = []
    filtered_decisions = 0
    undo_choices = 0
    turns = []
    failure = None
    try:
        state = engine.request("reset", {"seed": case["seed"], "decks": case["decks"],
                                          "firstPlayer": case["firstPlayer"]})
        executed = 0
        for decision in range(max_decisions):
            if state.get("status") == "finished" or state.get("observation", {}).get("status") == "finished":
                break
            observation = observation_for_actor(engine, state)
            actor = observation["playerId"]
            turns.append(observation.get("turn"))
            allowed, guard_position, filtered = forward_choices(observation, repetitions)
            policy_observation = allowed if guarded else observation
            try:
                chosen, p3_top, heuristic_top = ranked_actions(checkpoint, agents[actor], policy_observation)
                original = next(action for action in observation["legalActions"] if action["id"] == chosen["id"])
                pos = position_key(observation)
                act = action_key(original)
                token = f"{pos}:{act}"
                pair_counts[token] += 1
                position_counts[pos] += 1
                representatives.setdefault(token, {
                    "positionKey": pos,
                    "actionKey": act,
                    "action": action_summary(original),
                    "p3Top": p3_top,
                    "heuristicTop": heuristic_top,
                    "actor": actor,
                    "turn": observation.get("turn"),
                    "phase": observation.get("phase"),
                    "prompt": prompt_summary(observation.get("prompt")),
                })
                occurrence = pair_counts[token]
                if occurrence in {1, 2, 3, 10, 50, 100}:
                    examples.append({"decisionIndex": decision, "occurrence": occurrence,
                                     **representatives[token]})
                if original.get("choiceOperation") == "undo" or original.get("label") == "Cancel":
                    undo_choices += 1
                # Maintain the exact existing guard state even in the unguarded
                # trace so its counterfactual filtering point is observable.
                record_choice(repetitions, guard_position, original)
                if filtered:
                    filtered_decisions += 1
                state = engine.request("step", {"actionId": original["id"]})
                executed += 1
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break
        replay = engine.request("replay")
    finally:
        engine.close()
    top_pairs = []
    for token, count in pair_counts.most_common(10):
        top_pairs.append({"count": count, **representatives[token]})
    return {
        "case": case,
        "guarded": guarded,
        "maxDecisions": max_decisions,
        "executedDecisions": executed,
        "status": replay.get("status") if replay.get("status") == "finished" else "truncated",
        "outcome": replay.get("outcome") if replay.get("status") == "finished" else None,
        "failure": failure,
        "firstTurn": turns[0] if turns else None,
        "lastTurn": turns[-1] if turns else None,
        "uniqueVisiblePositions": len(position_counts),
        "repeatedVisiblePositions": sum(count > 1 for count in position_counts.values()),
        "guardFilteredDecisions": filtered_decisions,
        "undoOrCancelChoices": undo_choices,
        "topRepeatedPositionActions": top_pairs,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=250)
    args = parser.parse_args()
    root, checkpoint = args.root.resolve(), args.checkpoint.resolve()
    health_client = EngineClient(root)
    try:
        health = health_client.request("health")
    finally:
        health_client.close()
    traces = []
    for case in CASES:
        traces.append(trace(root, checkpoint, case, guarded=False, max_decisions=args.max_decisions))
        traces.append(trace(root, checkpoint, case, guarded=True, max_decisions=args.max_decisions))
    result = {
        "schemaVersion": 1,
        "id": "strategy-baseline-v1-p3-loop-diagnosis",
        "engineFingerprint": health["engineVersion"],
        "engineBuildHash": health["engineBuildHash"],
        "checkpoint": {"path": str(checkpoint), "sha256": file_digest(checkpoint)},
        "method": "Two cells, each traced unchanged and with the existing visible-repetition guard as a counterfactual.",
        "traces": traces,
    }
    result["resultHash"] = digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in traces:
        top = item["topRepeatedPositionActions"][0] if item["topRepeatedPositionActions"] else None
        print(json.dumps({"case": item["case"]["id"], "guarded": item["guarded"],
                          "status": item["status"], "decisions": item["executedDecisions"],
                          "lastTurn": item["lastTurn"], "guardFiltered": item["guardFilteredDecisions"],
                          "topRepeat": top}), flush=True)


if __name__ == "__main__":
    main()
