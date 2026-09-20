from __future__ import annotations

import math
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Settings
from .engine import EngineClient, EngineError
from .features import heuristic_action_score
from .resources import ResourceGuard, ResourceLimit
from .storage import Store, digest, file_digest


class Agent:
    def __init__(self, policy: str, seed: int):
        self.policy, self.rng, self.loaded = policy, random.Random(seed), None
        if policy not in {"random", "heuristic"}:
            from .training import load_model
            self.loaded = load_model(Path(policy))

    def choose(self, observation: dict) -> str:
        legal = observation.get("legalActions", [])
        if not legal:
            raise ValueError("Nonterminal observation contains no enumerated legal action")
        if self.policy == "random":
            return self.rng.choice(legal)["id"]
        if self.policy == "heuristic":
            scores = [heuristic_action_score(action) for action in legal]
        else:
            from .training import predict
            _, scores = predict(Path(self.policy), observation, self.loaded)
        best = max(scores)
        return self.rng.choice([action for action, score in zip(legal, scores) if score == best])["id"]


def admitted_decks(registry: list[dict], allow_unverified: bool = False, evaluation: bool = False) -> list[dict]:
    roles = {"main", "training-variant", "heldout"} if evaluation else {"main", "training-variant"}
    result = [deck for deck in registry if deck.get("role", "main") in roles
              and (allow_unverified or ("role" in deck and deck.get("validation", {}).get("trainingEligible") is True))]
    if not result:
        raise ValueError("No rules-validated competitive decks are training eligible. Use --allow-unverified only for rules QA; its games cannot train or promote a model.")
    return result


def policy_identity(policy: str) -> str:
    return policy if policy in {"random", "heuristic"} else file_digest(Path(policy))


def population(store: Store, limit: int = 4) -> list[str]:
    result = ["random", "heuristic"]
    champion = store.path / "models" / "champion.pt"
    seen = set()
    for path in [champion, *sorted((store.path / "models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)]:
        if not path.exists():
            continue
        identity = file_digest(path)
        if identity in seen:
            continue
        from .training import load_model
        try:
            load_model(path)
        except (ValueError, RuntimeError, KeyError):
            continue
        result.append(str(path.resolve()))
        seen.add(identity)
        if len(seen) >= limit:
            break
    return result


def search_choice(engine, observation: dict, fallback: Agent, *, seed: int, budget_ms: int, method: str = "ismcts",
                  known_opponent_deck_id: str | None = None, prior_revealed_cards: list[str] | None = None):
    params = {"observation": observation, "seed": seed, "budgetMs": budget_ms,
              "method": method, "iterations": 100, "maxRolloutDecisions": 16}
    if known_opponent_deck_id is not None:
        params["knownOpponentDeckId"] = known_opponent_deck_id
    if prior_revealed_cards is not None:
        params["priorRevealedCards"] = list(prior_revealed_cards)
    if fallback.loaded is not None:
        from .training import predict
        _, scores = predict(Path(fallback.policy), observation, fallback.loaded)
        if scores and all(math.isfinite(score) for score in scores):
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            total = sum(weights)
            params["rootPriors"] = [{"actionId": action["id"], "probability": weight / total}
                                   for action, weight in zip(observation.get("legalActions", []), weights)]
    result = engine.request("search", params)
    legal = {action["id"] for action in observation.get("legalActions", [])}
    measured = [item for item in (result.get("alternatives", []) if result.get("status") == "complete" else []) if item.get("actionId") in legal
                and item.get("visits", 0) > 0 and item.get("score") is not None and math.isfinite(item["score"])]
    choice = max(measured, key=lambda item: item["score"])["actionId"] if measured else fallback.choose(observation)
    target = None
    # A soft target is admitted only when every legal action was sampled at least
    # twice. Missing candidates are not incorrectly labelled as bad actions.
    if (len(measured) == len(legal) and {item["actionId"] for item in measured} == legal
            and all(item["visits"] >= 2 for item in measured)):
        maximum = max(item["score"] for item in measured)
        weights = {item["actionId"]: math.exp((item["score"] - maximum) / .15) for item in measured}
        total = sum(weights.values())
        target = {"schemaVersion": 1, "actor": observation["playerId"], "observationHash": digest(observation),
                  "probabilities": {action: value / total for action, value in weights.items()},
                  "method": method, "budgetMs": budget_ms, "iterations": result.get("iterations", 0),
                  "rootPolicy": fallback.policy if fallback.loaded is None else policy_identity(fallback.policy),
                  "semantics": "Search policy target from approximate cutoffs; never an outcome label"}
    return choice, target


def play_game(engine: EngineClient, *, decks: list[str], seed: int, policies: tuple[str, str], max_decisions: int,
              first_player: int | None = None, guard: ResourceGuard | None = None, search_budget_ms: int = 0,
              search_method: str = "ismcts") -> dict:
    agents = [Agent(policy, seed * 2 + player) for player, policy in enumerate(policies)]
    config = {"seed": seed, "decks": decks}
    if first_player is not None:
        config["firstPlayer"] = first_player
    state = engine.request("reset", config)
    failure, targets = None, []
    for decision in range(max_decisions):
        if guard:
            guard.check(storage=decision % 25 == 0)
        if state.get("status") == "finished" or state.get("observation", {}).get("status") == "finished":
            break
        observation = state["observation"]
        if observation["playerId"] != observation["decisionPlayer"]:
            observation = engine.request("observe", {"playerId": observation["decisionPlayer"]})
            observation = observation.get("observation", observation)
        try:
            agent = agents[observation["playerId"]]
            if search_budget_ms:
                action, target = search_choice(engine, observation, agent, seed=(seed + decision) % 2**32,
                                               budget_ms=search_budget_ms, method=search_method)
                if target:
                    target["decisionIndex"] = observation.get("decisionIndex", state.get("decisionIndex", decision))
                    targets.append(target)
            else:
                action = agent.choose(observation)
            state = engine.request("step", {"actionId": action})
        except (EngineError, ValueError) as exc:
            failure = str(exc)
            break
    replay = engine.request("replay")
    if replay.get("status") != "finished":
        replay["status"] = "error" if failure else "truncated"
        replay["outcome"] = None
        replay.setdefault("warnings", []).append(f"Agent/engine failure: {failure}" if failure else "Python decision budget reached; no terminal game result is assigned.")
    replay["policies"], replay["searchTargets"] = list(policies), targets
    replay["startingPlayer"] = next((frame["actor"] for frame in replay.get("frames", [])
        if frame.get("observations") and frame["observations"][frame["actor"]].get("phase") == "PLAYER_TURN"
        and frame["observations"][frame["actor"]].get("turn") == 1
        and not frame["observations"][frame["actor"]].get("prompt")), None)
    return replay


def selfplay(settings: Settings, *, games: int = 20, seed: int = 42, max_decisions: int = 1000,
             policy: str = "heuristic", resume: str | None = None, stop_after: int | None = None,
             opponents: list[str] | None = None, search_budget_ms: int = 0, search_method: str = "ismcts",
             allow_unverified: bool = False, on_created=None) -> dict:
    if not 1 <= games <= 10000 or not 1 <= max_decisions <= settings.max_decisions:
        raise ValueError("Self-play limits: 1..10000 games and 1..3000 decisions per game")
    if not 0 <= seed < 2**32 or (stop_after is not None and stop_after < 0):
        raise ValueError("Seed must be uint32 and stop-after must be nonnegative")
    if not 0 <= search_budget_ms <= 10000 or search_method not in {"rollout", "ismcts"}:
        raise ValueError("Search per decision must be 0..10000 ms and rollout or ismcts")
    if not 1 <= settings.workers <= 8:
        raise ValueError("Workers must be 1..8; increase above two only after diagnostics")
    store = Store(settings.data, settings.max_disk_bytes, settings.min_free_bytes)
    with EngineClient(settings.root, settings.engine_timeout) as metadata:
        registry = metadata.request("decks")
        registry = registry["decks"] if isinstance(registry, dict) else registry
        decks = admitted_decks(registry, allow_unverified)
        deck_ids = [deck["id"] for deck in decks]
        choices = opponents or [policy]
        config = {"games": games, "seed": seed, "maxDecisions": max_decisions, "policy": policy,
                  "policyHash": policy_identity(policy), "opponents": choices,
                  "opponentHashes": [policy_identity(value) for value in choices],
                  "deckHash": digest(decks), "decks": deck_ids, "engine": metadata.request("health"),
                  "searchBudgetMs": search_budget_ms, "searchMethod": search_method,
                  "allowUnverified": allow_unverified}
    if resume:
        state = store.get("selfplay", resume)
        if state["configuration"] != config:
            raise ValueError("Resume requires identical configuration, engine and deck hashes")
    else:
        state = {"id": uuid.uuid4().hex, "configuration": config, "nextGame": 0, "replayIds": [],
                 "finished": 0, "truncated": 0, "errors": 0, "status": "running"}
    workers = [EngineClient(settings.root, settings.engine_timeout) for _ in range(settings.workers)]
    completed = 0
    # Persist before resource checks so a resource pause always has a resumable ID.
    store.put("selfplay", state["id"], state)
    if on_created:
        on_created(state["id"])
    try:
        with ResourceGuard(settings) as guard, ThreadPoolExecutor(max_workers=settings.workers) as pool:
            while state["nextGame"] < games:
                guard.check()
                if (settings.data / f"pause-{state['id']}").exists() or (stop_after is not None and completed >= stop_after):
                    state["status"] = "paused"
                    break
                count = min(settings.workers, games - state["nextGame"], (stop_after - completed) if stop_after is not None else games)
                futures = []
                for offset in range(count):
                    index = state["nextGame"] + offset
                    pair = [deck_ids[index % len(deck_ids)], deck_ids[(index // len(deck_ids)) % len(deck_ids)]]
                    opponent = choices[index % len(choices)]
                    policies = (policy, opponent) if index % 2 == 0 else (opponent, policy)
                    futures.append((index, pair, pool.submit(play_game, workers[offset], decks=pair, seed=(seed + index) % 2**32,
                        policies=policies, max_decisions=max_decisions, guard=guard,
                        search_budget_ms=search_budget_ms, search_method=search_method)))
                for index, pair, future in futures:
                    replay = future.result()
                    manifests = [next(deck for deck in decks if deck["id"] == identifier) for identifier in pair]
                    replay["deckRoles"] = [deck.get("role", "main") for deck in manifests]
                    replay["deckHashes"] = [deck.get("listHash", digest(deck)) for deck in manifests]
                    replay["trainingEligible"] = not allow_unverified and all("role" not in deck or deck.get("validation", {}).get("trainingEligible") is True for deck in manifests)
                    replay["selfplayRun"] = state["id"]
                    state["replayIds"].append(store.save_replay(replay))
                    state["nextGame"] = index + 1
                    counter = {"finished": "finished", "truncated": "truncated", "error": "errors"}[replay["status"]]
                    state[counter] += 1
                    completed += 1
                    store.put("selfplay", state["id"], state)
            else:
                state["status"] = "completed"
            state["peakProcessTreeRssBytes"] = guard.peak
    except (KeyboardInterrupt, ResourceLimit, EngineError, OSError) as exc:
        state["status"] = "paused"
        state["pauseReason"] = str(exc) or "Interrupted by user"
        state["note"] = "Unfinished games have no outcome and restart from their original seeds on resume."
    finally:
        for worker in workers:
            worker.close()
    # Keep a small emergency reserve for the pause manifest; never delete replays.
    try:
        store.put("selfplay", state["id"], state)
    except (ResourceLimit, RuntimeError):
        state["persistenceWarning"] = "Resource limit prevented a final manifest update; the last durable batch can be resumed."
    return state
