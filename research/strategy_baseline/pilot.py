"""Research-only Strategy Baseline v1 timing pilot.

This module freezes the approved strategy inputs and times one game per sample
cell for each currently loadable policy. It does not write to data/competitive
or change production policy, search, gameplay, features, models, or training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from ptcg_lab.engine import EngineClient
from ptcg_lab.selfplay import Agent, search_choice
from ptcg_lab.strategy_contract import validate_strategy_revision_v12


PILOT_SEEDS = [219283457, 219283458, 219283459, 219283460,
               219283461, 219283462, 219283463, 219283464]
CELLS = [
    {"id": "cross-crustle-seat0-first0", "decks": ["crustle", "dragapult"], "firstPlayer": 0},
    {"id": "cross-crustle-seat0-first1", "decks": ["crustle", "dragapult"], "firstPlayer": 1},
    {"id": "cross-crustle-seat1-first0", "decks": ["dragapult", "crustle"], "firstPlayer": 0},
    {"id": "cross-crustle-seat1-first1", "decks": ["dragapult", "crustle"], "firstPlayer": 1},
    {"id": "crustle-mirror-first0", "decks": ["crustle", "crustle"], "firstPlayer": 0},
    {"id": "crustle-mirror-first1", "decks": ["crustle", "crustle"], "firstPlayer": 1},
    {"id": "dragapult-mirror-first0", "decks": ["dragapult", "dragapult"], "firstPlayer": 0},
    {"id": "dragapult-mirror-first1", "decks": ["dragapult", "dragapult"], "firstPlayer": 1},
]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_patches(items: list[dict], patches: list[dict], key: str) -> list[dict]:
    result = copy.deepcopy(items)
    positions = {item[key]: index for index, item in enumerate(result)}
    for raw in patches:
        patch = copy.deepcopy(raw)
        operation = patch.pop("operation")
        identifier = patch[key]
        if operation == "replace":
            if identifier not in positions:
                raise ValueError(f"Cannot replace missing {key} {identifier}")
            result[positions[identifier]] = patch
        elif operation == "add":
            if identifier in positions:
                raise ValueError(f"Cannot add duplicate {key} {identifier}")
            positions[identifier] = len(result)
            result.append(patch)
        else:
            raise ValueError(f"Unsupported patch operation {operation}")
    return result


def effective_contract(root: Path, fingerprint: str, build_hash: str) -> dict:
    base = read_json(root / "research/strategy/contract-v1.json")
    playbooks = {}
    for deck in ("crustle", "dragapult"):
        current = read_json(root / f"research/strategy/{deck}-v1.json")
        for release in ("v1.1", "v1.2"):
            overlay = read_json(root / f"research/strategy/{deck}-{release}.json")
            current["principles"] = apply_patches(current["principles"], overlay.get("principlePatches", []), "id")
            current["matchups"] = apply_patches(current["matchups"], overlay.get("matchupPatches", []), "perspectiveId")
        current["id"] = f"{deck}-effective-v1.2"
        current["status"] = "approved"
        current["effectiveRevision"] = "v1.2"
        playbooks[deck] = current
    return {
        "schemaVersion": 1,
        "id": "strategy-contract-effective-v1.2",
        "status": "approved-generated",
        "sourceRevisions": ["v1", "v1.1", "v1.2"],
        "engineFingerprint": fingerprint,
        "engineBuildHash": build_hash,
        "contract": base,
        "playbooks": playbooks,
    }


class RecordingEngine:
    def __init__(self, client: EngineClient):
        self.client = client
        self.searches: list[dict] = []

    def request(self, method: str, params: dict | None = None):
        result = self.client.request(method, params)
        if method == "search":
            self.searches.append(result)
        return result


def normalized_observation(engine: RecordingEngine, state: dict) -> dict:
    observation = state["observation"]
    if observation["playerId"] != observation["decisionPlayer"]:
        observed = engine.request("observe", {"playerId": observation["decisionPlayer"]})
        observation = observed.get("observation", observed)
    return observation


def run_python_game(root: Path, cell: dict, seed: int, policy: str, max_decisions: int,
                    guide_checkpoint: Path) -> tuple[dict, dict]:
    client = EngineClient(root, timeout=300)
    engine = RecordingEngine(client)
    agent_policy = str(guide_checkpoint) if policy == "P3" else "heuristic"
    agents = [Agent(agent_policy, seed * 2 + player, allow_experimental=policy == "P3") for player in range(2)]
    search_tags = []
    failure = None
    try:
        state = engine.request("reset", {"seed": seed, "decks": cell["decks"], "firstPlayer": cell["firstPlayer"]})
        decisions = 0
        for decisions in range(max_decisions):
            if state.get("status") == "finished" or state.get("observation", {}).get("status") == "finished":
                break
            observation = normalized_observation(engine, state)
            actor = observation["playerId"]
            try:
                if policy == "P4":
                    before = len(engine.searches)
                    action, _ = search_choice(
                        engine,
                        observation,
                        agents[actor],
                        seed=(seed + decisions) % 2**32,
                        budget_ms=200,
                        method="ismcts",
                    )
                    if len(engine.searches) == before:
                        search_tags.append({"decisionIndex": decisions, "actor": actor, "kind": "fallback", "reason": "no-search-position"})
                    else:
                        response = engine.searches[-1]
                        legal = {item["id"] for item in observation.get("legalActions", [])}
                        measured = [item for item in response.get("alternatives", [])
                                    if item.get("actionId") in legal and item.get("visits", 0) > 0
                                    and item.get("score") is not None and math.isfinite(item["score"])]
                        search_tags.append({
                            "decisionIndex": decisions,
                            "actor": actor,
                            "kind": "searched" if measured else "fallback",
                            "reason": None if measured else "search-returned-no-measured-action",
                            "iterations": response.get("iterations", 0),
                        })
                else:
                    action = agents[actor].choose(observation)
                state = engine.request("step", {"actionId": action})
            except Exception as exc:  # Preserve an honest pilot result rather than reclassifying it.
                failure = f"{type(exc).__name__}: {exc}"
                break
        replay = engine.request("replay")
    finally:
        client.close()
    if replay.get("status") != "finished":
        replay["status"] = "error" if failure else "truncated"
        replay["outcome"] = None
    searched = sum(tag["kind"] == "searched" for tag in search_tags)
    return replay, {
        "decisions": decisions + 1 if replay.get("status") != "finished" else decisions,
        "failure": failure,
        "search": {
            "fallbackPolicy": "Python heuristic_action_score" if policy == "P4" else None,
            "tags": search_tags,
            "searchedDecisions": searched,
            "fallbackDecisions": len(search_tags) - searched,
            "searchAttempts": len(engine.searches),
            "totalIterations": sum(item.get("iterations", 0) or 0 for item in engine.searches),
        },
    }


def run_cell(root: Path, policy: str, cell: dict, seed: int, max_decisions: int,
             guide_checkpoint: Path) -> dict:
    started = time.perf_counter()
    if policy == "P1":
        with EngineClient(root, timeout=300) as engine:
            replay = engine.request("run", {
                "seed": seed,
                "decks": cell["decks"],
                "firstPlayer": cell["firstPlayer"],
                "policy": "heuristic",
                "maxDecisions": max_decisions,
            })
        details = {"decisions": len(replay.get("frames", [])), "failure": None, "search": None}
    else:
        replay, details = run_python_game(root, cell, seed, policy, max_decisions, guide_checkpoint)
    elapsed = time.perf_counter() - started
    return {
        "policy": policy,
        "cell": cell,
        "seed": seed,
        "elapsedSeconds": elapsed,
        "status": replay.get("status"),
        "outcome": replay.get("outcome"),
        "warnings": replay.get("warnings", []),
        **details,
    }


def summarize_policy(rows: list[dict]) -> dict:
    durations = [row["elapsedSeconds"] for row in rows]
    finished = [row for row in rows if row["status"] == "finished"]
    winners = {0: 0, 1: 0}
    for row in finished:
        winner = row.get("outcome", {}).get("winner")
        if winner in winners:
            winners[winner] += 1
    # The main plan weights cross-matchup cells at 8 games each and mirror
    # cells at 16 games each; a flat 12x multiplier would distort this mix.
    projected = None
    if len(finished) == len(rows):
        projected = sum(
            row["elapsedSeconds"] * (8 if row["cell"]["id"].startswith("cross-") else 16)
            for row in rows
        )
    return {
        "pilotGames": len(rows),
        "finished": len(finished),
        "truncated": sum(row["status"] == "truncated" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "seatWins": {"seat0": winners[0], "seat1": winners[1]},
        "elapsedSeconds": sum(durations),
        "meanGameSeconds": statistics.mean(durations),
        "medianGameSeconds": statistics.median(durations),
        "projected96GameSeconds": projected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guide-checkpoint", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--policies", nargs="+", choices=["P1", "P2", "P3", "P4"])
    parser.add_argument("--prior-results", type=Path,
                        help="Merge previously timed policy rows without rerunning them")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    validated = validate_strategy_revision_v12(root)
    if validated["activeRevision"] != "v1.2":
        raise RuntimeError("Strategy v1.2 is not active")

    with EngineClient(root, timeout=300) as engine:
        health = engine.request("health")
        decks = engine.request("decks")
    fingerprint = health["engineVersion"]
    build_hash = health["engineBuildHash"]
    relevant_decks = {deck["id"]: deck["listHash"] for deck in decks if deck["id"] in {"crustle", "dragapult"}}

    effective = effective_contract(root, fingerprint, build_hash)
    effective_hash = hashlib.sha256(canonical_bytes(effective)).hexdigest()
    effective_record = {**effective, "effectiveContractHash": effective_hash}
    (output / "effective-contract.json").write_text(json.dumps(effective_record, indent=2) + "\n", encoding="utf-8")

    checkpoint = args.guide_checkpoint.resolve()
    checkpoint_record = {"path": str(checkpoint), "sha256": file_hash(checkpoint), "loadStatus": "pending"}
    available_policies = ["P1", "P2", "P4"]
    try:
        from ptcg_lab.training import load_model
        load_model(checkpoint, allow_experimental=True)
        checkpoint_record["loadStatus"] = "loaded"
        available_policies.insert(2, "P3")
    except Exception as exc:
        checkpoint_record["loadStatus"] = "dropped"
        checkpoint_record["loadError"] = f"{type(exc).__name__}: {exc}"

    executed_policies = args.policies or available_policies
    unavailable = sorted(set(executed_policies) - set(available_policies))
    if unavailable:
        raise RuntimeError(f"Requested policies are unavailable: {unavailable}")

    prior_rows = []
    prior_policies = []
    if args.prior_results:
        prior = read_json(args.prior_results.resolve())
        if (prior.get("engineFingerprint") != fingerprint
                or prior.get("engineBuildHash") != build_hash
                or prior.get("effectiveContractHash") != effective_hash):
            raise ValueError("Prior pilot results use different frozen inputs")
        prior_rows = prior.get("games", [])
        prior_policies = prior.get("activePolicies", [])
    combined = set(prior_policies) | set(executed_policies)
    active_policies = [policy for policy in ["P1", "P2", "P3", "P4"] if policy in combined]

    frozen = {
        "schemaVersion": 1,
        "id": "strategy-baseline-v1-pilot-inputs",
        "engineFingerprint": fingerprint,
        "engineBuildHash": build_hash,
        "strategyRevision": validated["activeRevision"],
        "effectiveContractHash": effective_hash,
        "guideCheckpoint": checkpoint_record,
        "deckListHashes": relevant_decks,
        "pilotSeeds": PILOT_SEEDS,
        "cells": CELLS,
        "activePolicies": active_policies,
        "executedPolicies": executed_policies,
        "priorResults": str(args.prior_results.resolve()) if args.prior_results else None,
        "droppedPolicies": ["P3"] if "P3" not in available_policies else [],
        "maxDecisions": args.max_decisions,
        "searchBudgetMs": 200,
    }
    (output / "frozen-inputs.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    rows = list(prior_rows)
    for policy in executed_policies:
        for cell, seed in zip(CELLS, PILOT_SEEDS):
            row = run_cell(root, policy, cell, seed, args.max_decisions, checkpoint)
            rows.append(row)
            print(json.dumps({"policy": policy, "cell": cell["id"], "seconds": round(row["elapsedSeconds"], 3),
                              "status": row["status"], "outcome": row["outcome"]}), flush=True)

    summaries = {policy: summarize_policy([row for row in rows if row["policy"] == policy]) for policy in active_policies}
    projectable = {policy: summary for policy, summary in summaries.items()
                   if summary["projected96GameSeconds"] is not None}
    blocking = [policy for policy in active_policies if policy not in projectable]
    completed_policy_seconds = sum(summary["projected96GameSeconds"] for summary in projectable.values())
    projection_available = not blocking
    full_games = 96 * len(active_policies)
    reduced = completed_policy_seconds > 3600 if projection_available else None
    result = {
        "schemaVersion": 1,
        "id": "strategy-baseline-v1-pilot-results",
        "engineFingerprint": fingerprint,
        "engineBuildHash": build_hash,
        "effectiveContractHash": effective_hash,
        "activePolicies": active_policies,
        "droppedPolicies": frozen["droppedPolicies"],
        "games": rows,
        "policySummaries": summaries,
        "projection": {
            "originalPromptGames": 384,
            "executableFullPlanGames": full_games,
            "projectablePolicyGames": 96 * len(projectable),
            "completedPolicyProjectionSeconds": completed_policy_seconds,
            "completedPolicyProjectionMinutes": completed_policy_seconds / 60,
            "projectionAvailable": projection_available,
            "blockingPolicies": blocking,
            "serialEquivalentSeconds": completed_policy_seconds if projection_available else None,
            "serialEquivalentMinutes": completed_policy_seconds / 60 if projection_available else None,
            "exceeds60Minutes": reduced,
            "recommendedMainGames": (full_games // 2 if reduced else full_games) if projection_available else None,
            "recommendedGamesPerOriginalCell": ("half" if reduced else "unchanged") if projection_available else None,
            "note": ("Projection is timing-only. Pilot outcomes are not strength estimates."
                     if projection_available else
                     "The complete projection is unavailable because at least one policy had no terminal pilot games."),
        },
    }
    (output / "pilot-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["projection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
