"""Resumable games for the experimental supervisor, using its shared workers."""
from __future__ import annotations

import time
from pathlib import Path

from .engine import EngineError
from .decision_guard import forward_choices, record_choice, VERSION as GUARD_VERSION
from .selfplay import Agent, search_choice
from .storage import digest, file_digest


class LearningPaused(RuntimeError):
    pass


class LearningYield(LearningPaused):
    pass


def frozen_policy(store, policy):
    if policy in {"heuristic", "random"}:
        return {"path": policy, "hash": policy, "name": policy}
    from .checkpoint_files import freeze_checkpoint
    target = freeze_checkpoint(store, Path(policy))
    return {"path": str(target), "hash": file_digest(target), "name": f"checkpoint {file_digest(target)[:8]}"}


def validate_policy(policy):
    if policy["path"] not in {"heuristic", "random"} and file_digest(Path(policy["path"])) != policy["hash"]:
        raise ValueError("A frozen policy changed; this run cannot continue")


def restore_game(engine, game, identity, check):
    health = engine.request("health")
    if any(health.get(key) != identity.get(key) for key in ("engineVersion", "engineBuildHash")):
        raise ValueError("Engine build changed; keep this run paused and create a new run")
    for policy in game["policies"]:
        validate_policy(policy)
    engine.request("reset", {"decks": game["decks"], "seed": game["seed"], "firstPlayer": game["firstPlayer"]})
    for action in game["actions"]:
        check()
        engine.request("step", {"actionId": action["id"]})
    if game.get("observationHash"):
        current = [engine.request("observe", {"playerId": player}) for player in (0, 1)]
        if digest(current) != game["observationHash"]:
            raise ValueError("Reconstruction differs from the last acknowledged position")


def run_game(service, run_id, game_id, worker_index, guard):
    """A failed worker may replay one acknowledged journal; no fake terminal."""
    game = service.store.get("learning-games", game_id)
    if game["status"] in {"finished", "truncated", "error"}:
        return game
    if game.get("workerFailures", 0) >= 2:
        raise EngineError("Worker retry budget is exhausted; explicitly resume after resolving the failure. The accepted game journal is preserved.")
    engine = service.pool.acquire(wait_timeout=5)
    previous_timeout = engine.timeout
    engine.timeout = min(previous_timeout, 20)
    try:
        for attempt in range(game.get("workerFailures", 0), 2):
            try:
                service.check(run_id, guard)
                restore_game(engine, game, service.private(run_id)["engine"], lambda: service.check(run_id, guard))
                agents = [Agent(policy["path"], game["seed"] + player, allow_experimental=True)
                          for player, policy in enumerate(game["policies"])]
                game.update(status="running", workerIndex=worker_index)
                if game["frameCursor"] < 0:
                    observations = [engine.request("observe", {"playerId": p}) for p in (0, 1)]
                    service.commit_frame(game, observations, None, observations[0]["decisionPlayer"])
                while True:
                    service.check(run_id, guard)
                    observations = [engine.request("observe", {"playerId": p}) for p in (0, 1)]
                    if observations[0]["status"] == "finished" or len(game["actions"]) >= game["maxDecisions"]:
                        replay = engine.request("replay")
                        replay.update(id=game_id, dataTier="experimental", experimentalLearning=game["purpose"] == "collection",
                                      trainingEligible=False, learningRun=run_id, learningCycle=game["cycle"],
                                      seed=game["seed"], decks=game["decks"], deckRoles=game["deckRoles"], deckHashes=game["deckHashes"],
                                      searchTargets=game["searchTargets"], decisionGuard={"version": GUARD_VERSION, "fallbackDecisions": game.get("guardDecisions", 0)}, policies=[p["hash"] for p in game["policies"]],
                                      policyContext=service.policy_context(game), startingPlayer=game["firstPlayer"])
                        if replay.get("status") != "finished":
                            replay.update(status="truncated", outcome=None)
                        if game["purpose"] == "comparison":
                            replay["evaluationExperiment"] = f"{run_id}-{game['cycle']}"
                        replay_id = service.store.save_replay(replay)
                        game.update(status=replay["status"], outcome=replay.get("outcome"), replayId=replay_id,
                                    replayAvailable=True, decisionIndex=len(game["actions"]), turn=observations[0]["turn"])
                        service.save_game(game)
                        service.compact_game(game)
                        return game
                    actor = observations[0]["decisionPlayer"]
                    observation = observations[actor]
                    turn_key = f"{observation['turn']}:{actor}"
                    spent = game["spentMs"].get(turn_key, 0.)
                    budget = min(game["searchBudgetMs"], max(0, int(120000-spent)))
                    agent = agents[actor]
                    # Counter-based RNG survives continuation without drawing on
                    # the real game's chance stream or replaying policy inference.
                    agent.rng.seed(f"{game['seed']}:{len(game['actions'])}:{actor}")
                    started = time.monotonic()
                    safe, position, filtered = forward_choices(observation, game.setdefault("decisionRepetitions", {}))
                    if not safe["legalActions"]:
                        raise ValueError("Staged selection has no forward legal action")
                    target = None
                    if budget and not observation.get("prompt") and not filtered:
                        action_id, target = search_choice(engine, observation, agent,
                            seed=(game["seed"] + len(game["actions"])) % 2**32, budget_ms=budget)
                        game["searchDecisions"] += 1
                    else:
                        if spent >= 120000:
                            fallback = Agent("heuristic", game["seed"]+len(game["actions"]))
                            action_id = fallback.choose(safe)
                            game["budgetFallbackDecisions"] = game.get("budgetFallbackDecisions", 0)+1
                        else:
                            action_id = agent.choose(safe)
                    action = next(a for a in observation["legalActions"] if a["id"] == action_id)
                    if action.get("choiceOperation") == "undo":
                        action_id, target = agent.choose(safe), None
                        action = next(a for a in safe["legalActions"] if a["id"] == action_id)
                    record_choice(game["decisionRepetitions"], position, action)
                    if filtered:
                        game["guardDecisions"] = game.get("guardDecisions", 0)+1
                    game["spentMs"][turn_key] = spent + (time.monotonic()-started)*1000
                    engine.request("step", {"actionId": action_id})
                    next_views = [engine.request("observe", {"playerId": p}) for p in (0, 1)]
                    if target:
                        target["decisionIndex"] = len(game["actions"])
                        game["searchTargets"].append(target)
                    game["actions"].append({"id": action_id})
                    service.commit_frame(game, next_views, action, actor)
            except EngineError as exc:
                engine.close()
                # Reload only committed decisions after a failed/partial request.
                game = service.store.get("learning-games", game_id)
                game["workerFailures"] = attempt + 1
                service.save_game(game)
                if attempt:
                    raise EngineError("Worker failed twice; the accepted game journal is preserved") from exc
    finally:
        engine.timeout = previous_timeout
        service.pool.release(engine)
