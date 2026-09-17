from __future__ import annotations

import random
import time
import uuid
from pathlib import Path

from .config import Settings
from .engine import EngineClient, EngineError
from .features import heuristic_action_score
from .storage import Store, digest


class Agent:
    def __init__(self, policy: str, seed: int):
        self.policy = policy
        self.rng = random.Random(seed)
        self.loaded = None
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
        candidates = [action for action, score in zip(legal, scores) if score == best]
        return self.rng.choice(candidates)["id"]


def play_game(engine: EngineClient, *, decks: list[str], seed: int, policies: tuple[str, str], max_decisions: int) -> dict:
    agents = [Agent(policy, seed * 2 + player) for player, policy in enumerate(policies)]
    state = engine.request("reset", {"seed": seed, "decks": decks})
    failure = None
    for _ in range(max_decisions):
        if state.get("status") == "finished" or state.get("observation", {}).get("status") == "finished":
            break
        observation = state["observation"]
        if observation["playerId"] != observation["decisionPlayer"]:
            observation = engine.request("observe", {"playerId": observation["decisionPlayer"]})
            if "observation" in observation:
                observation = observation["observation"]
        try:
            action = agents[observation["playerId"]].choose(observation)
            state = engine.request("step", {"actionId": action})
        except (EngineError, ValueError) as exc:
            failure = str(exc)
            break
    replay = engine.request("replay")
    if replay.get("status") != "finished":
        replay["status"] = "error" if failure else "truncated"
        replay["outcome"] = None
        replay.setdefault("warnings", []).append(f"Agent/engine failure: {failure}" if failure else "Python decision budget reached; no terminal game result is assigned.")
    replay["policies"] = list(policies)
    replay["startingPlayer"] = next((frame["actor"] for frame in replay.get("frames", [])
        if frame.get("observations") and frame["observations"][frame["actor"]].get("phase") == "PLAYER_TURN"
        and frame["observations"][frame["actor"]].get("turn") == 1
        and not frame["observations"][frame["actor"]].get("prompt")), None)
    return replay


def selfplay(settings: Settings, *, games: int = 20, seed: int = 42, max_decisions: int = 1000,
             policy: str = "heuristic", resume: str | None = None, stop_after: int | None = None) -> dict:
    if not 1 <= games <= 10000 or not 1 <= max_decisions <= settings.max_decisions:
        raise ValueError("Self-play limits: 1..10000 games and 1..3000 decisions per game")
    if not 0 <= seed < 2**32 or (stop_after is not None and stop_after < 0):
        raise ValueError("Seed must be uint32 and stop-after must be nonnegative")
    store = Store(settings.data, settings.max_disk_bytes)
    with EngineClient(settings.root, settings.engine_timeout) as engine:
        registry = engine.request("decks")
        registry = registry["decks"] if isinstance(registry, dict) else registry
        deck_ids = [deck["id"] for deck in registry]
        if not deck_ids:
            raise ValueError("No deck manifests available")
        config = {"games": games, "seed": seed, "maxDecisions": max_decisions, "policy": policy,
                  "deckHash": digest(registry), "decks": deck_ids,
                  "engine": engine.request("health")}
        if resume:
            state = store.get("selfplay", resume)
            if state["configuration"] != config:
                raise ValueError("Resume requires identical configuration, engine and deck hashes")
        else:
            state = {"id": uuid.uuid4().hex, "configuration": config, "nextGame": 0, "replayIds": [],
                     "finished": 0, "truncated": 0, "errors": 0, "status": "running"}
        completed_this_run = 0
        try:
            while state["nextGame"] < games:
                if (settings.data / f"pause-{state['id']}").exists() or (stop_after is not None and completed_this_run >= stop_after):
                    state["status"] = "paused"
                    break
                index = state["nextGame"]
                # Cover every ordered pairing, including mirrors, before repeating.
                pair = [deck_ids[index % len(deck_ids)], deck_ids[(index // len(deck_ids)) % len(deck_ids)]]
                state["status"] = "running"
                store.put("selfplay", state["id"], state)
                game_seed = (seed + index) % 2**32
                if policy in {"random", "heuristic"}:
                    replay = engine.request("run", {"seed": game_seed, "decks": pair,
                                                     "policy": policy, "maxDecisions": max_decisions})
                else:
                    replay = play_game(engine, decks=pair, seed=game_seed, policies=(policy, policy), max_decisions=max_decisions)
                identifier = store.save_replay(replay)
                state["replayIds"].append(identifier)
                state["nextGame"] += 1
                counter = {"finished": "finished", "truncated": "truncated", "error": "errors"}[replay["status"]]
                state[counter] = state.get(counter, 0) + 1
                completed_this_run += 1
                store.put("selfplay", state["id"], state)
            else:
                state["status"] = "completed"
        except KeyboardInterrupt:
            state["status"] = "paused"
            state["note"] = "Interrupted game has no outcome and will be replayed from its seed on resume."
        except Exception as exc:
            state["status"] = "interrupted"
            state["error"] = str(exc)
            store.put("selfplay", state["id"], state)
            raise
        store.put("selfplay", state["id"], state)
        return state
