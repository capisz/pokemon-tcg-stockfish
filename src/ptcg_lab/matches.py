"""Durable local matches. Private journals never share the research replay namespace."""
from __future__ import annotations

import copy
import math
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .selfplay import Agent
from .storage import digest, file_digest, terminal_score


class MatchConflict(ValueError):
    pass


def policy_implementation() -> str:
    directory = Path(__file__).parent
    return digest({name: file_digest(directory / name) for name in ("matches.py", "selfplay.py", "features.py", "training.py")})


class MatchService:
    def __init__(self, settings, store, pool, registry):
        self.settings, self.store, self.pool, self.registry = settings, store, pool, registry
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ptcg-match")
        self.pending: set[str] = set()

    def close(self):
        self.executor.shutdown(wait=True)

    def recover(self):
        for record in self.store.list("private-matches"):
            if record["status"] == "active":
                record.update(status="paused", error="Server restarted. Resume reconstructs the accepted decisions.")
                self._save(record)

    def _save(self, record):
        record["updatedAt"] = datetime.now(timezone.utc).isoformat()
        self.store.put("private-matches", record["id"], record)

    def get_private(self, identifier):
        return self.store.get("private-matches", identifier)

    def benchmark_active(self):
        return any(r["mode"] == "benchmark" and r["status"] not in {"completed", "abandoned"} for r in self.store.list("private-matches"))

    def _public(self, record, *, summary=False):
        result = {key: copy.deepcopy(record[key]) for key in ("id", "schemaVersion", "revision", "mode", "status", "gameNumber", "score", "createdAt", "updatedAt", "warnings", "modelVersion")}
        result.update(thinking=record["id"] in self.pending, engineTurnBudgetMs=record["budgetMs"],
                      engineTurnRemainingMs=max(0, record["budgetMs"] - record.get("spentMs", 0)),
                      knownList=record["knownList"], ownDeckId=record["decks"][0],
                      error=record.get("error"), nextStarterChooser=record.get("nextStarterChooser"),
                      gameResult=record.get("gameResult"))
        if not summary:
            result["observation"] = copy.deepcopy(record["observation"])
            result["matchKnowledge"] = copy.deepcopy(record.get("publicKnowledge", []))
            if record["knownList"]:
                result["opponentList"] = next(d for d in self.registry() if d["id"] == record["decks"][1])["cards"]
            if record["status"] == "completed":
                result["replayIds"] = record.get("replayIds", [])
        return result

    def get(self, identifier):
        with self.lock:
            return self._public(self.get_private(identifier))

    def list(self):
        with self.lock:
            return [self._public(r, summary=True) for r in self.store.list("private-matches")]

    def create(self, deck_id, opponent_archetype, mode="practice", known_list=False, budget_ms=120000):
        with self.lock:
            if mode not in {"practice", "benchmark"} or not 1 <= budget_ms <= 120000:
                raise ValueError("Use practice or benchmark mode and a turn budget between 1 and 120000 ms.")
            if any(r["status"] not in {"completed", "abandoned"} for r in self.store.list("private-matches")):
                raise MatchConflict("Finish the current match before starting another.")
            registry = self.registry()
            available = [d for d in registry if d.get("role") not in {"heldout", "historical"} and d.get("playable", True)]
            if deck_id not in {d["id"] for d in available}:
                raise ValueError("Choose a playable main or training deck.")
            candidates = [d for d in available if d["archetype"] == opponent_archetype]
            if not candidates:
                raise ValueError("No playable deck for the selected opponent archetype.")
            identifier = uuid.uuid4().hex
            selected = secrets.choice(candidates)
            model = self.settings.analysis_model
            if model is None and (self.settings.data / "models/champion.pt").exists():
                model = self.settings.data / "models/champion.pt"
            policy, model_version = "heuristic", "heuristic-rollout-v1"
            if model is not None:
                model_version = file_digest(model)
                target = self.settings.data / "private-models" / f"{model_version}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    import os
                    from .resources import check_storage
                    minimum = getattr(self.settings, "min_free_bytes", 20 * 1024**3)
                    check_storage(self.settings.data, self.settings.max_disk_bytes, minimum, additional=model.stat().st_size)
                    temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
                    try:
                        with model.open("rb") as source, temporary.open("wb") as output:
                            import shutil
                            shutil.copyfileobj(source, output)
                            output.flush(); os.fsync(output.fileno())
                        if file_digest(temporary) != model_version:
                            raise ValueError("Checkpoint changed while the frozen copy was being created. Retry with an immutable checkpoint.")
                        check_storage(self.settings.data, self.settings.max_disk_bytes, minimum)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                policy = str(target)
            now = datetime.now(timezone.utc).isoformat()
            record = {"id": identifier, "schemaVersion": 1, "revision": 0, "mode": mode, "knownList": known_list,
                      "status": "active", "gameNumber": 1, "score": [0, 0], "createdAt": now, "updatedAt": now,
                      "decks": [deck_id, selected["id"]], "registryHash": digest(registry), "budgetMs": budget_ms,
                      "spentMs": 0, "budgetTurn": None, "seed": secrets.randbits(32), "searchSeed": secrets.randbits(32),
                      "actions": [], "requests": {}, "games": [], "publicKnowledge": [],
                      "publicKnowledgeByPlayer": [[], []], "policy": policy,
                      "policyImplementation": policy_implementation(),
                      "modelVersion": model_version, "warnings": ["Experimental playing strength. Search may use untrained cutoff values.",
                      "A turn's exhausted thinking budget uses a legal heuristic fallback to finish mandatory choices.",
                      "Cross-game beliefs retain cards seen on the public boards and in discards; private prior-game hands are never used.",
                      "Guide-derived competitive lists remain experimental until their rules and legality audit passes."], "replayIds": []}
            with self.pool.lease() as engine:
                record["engineIdentity"] = engine.request("health")
                engine.request("reset", {"seed": record["seed"], "decks": record["decks"]})
                record["observation"] = engine.request("observe", {"playerId": 0})
            self._save(record)
            return self._public(record)

    def _verify_identity(self, engine, record):
        if engine.request("health") != record["engineIdentity"] or digest(self.registry()) != record["registryHash"]:
            raise ValueError("Engine or deck registry changed. Restore the recorded build before resuming this match.")
        if record["policy"] != "heuristic" and file_digest(Path(record["policy"])) != record["modelVersion"]:
            raise ValueError("The frozen model changed. Restore the recorded checkpoint before resuming this match.")
        if record.get("policyImplementation") != policy_implementation():
            raise ValueError("The policy implementation changed. Restore the recorded build, or end this match as incomplete.")

    def _restore(self, engine, record):
        self._verify_identity(engine, record)
        params = {"seed": record["seed"], "decks": record["decks"]}
        if record.get("firstPlayer") is not None:
            params["firstPlayer"] = record["firstPlayer"]
        engine.request("reset", params)
        for action in record["actions"]:
            engine.request("step", {"actionId": action["actionId"]})
        current = engine.request("observe", {"playerId": 0})
        if current != record["observation"]:
            raise ValueError("Reconstruction differs from the durable observation; the match is paused.")

    def _check(self, record, revision, request_id, payload):
        previous = record["requests"].get(request_id)
        if previous:
            if previous["payload"] != payload:
                raise MatchConflict("This request identifier was already used for a different operation.")
            return previous["response"]
        if revision != record["revision"]:
            raise MatchConflict("The position changed. Reload it before choosing an action.")
        if record["id"] in self.pending:
            raise MatchConflict("The engine is thinking; wait for the accepted decision.")
        return None

    def _ack(self, record, request_id, payload):
        record["revision"] += 1
        response = self._public(record)
        record["requests"][request_id] = {"payload": payload, "response": response}
        self._save(record)
        return response

    def _capture_knowledge(self, record):
        by_player = record.setdefault("publicKnowledgeByPlayer", [[], []])
        for player in record["observation"]["players"]:
            seen = set(by_player[player["id"]])
            for card in player["discard"]:
                seen.add(card["id"])
            for pokemon in [player.get("active"), *player["bench"]]:
                if pokemon:
                    seen.add(pokemon["card"]["id"])
            by_player[player["id"]] = sorted(seen)
        record["publicKnowledge"] = sorted(set(by_player[0]) | set(by_player[1]))

    def _finish_game(self, record, engine, concession=False):
        replay = engine.request("replay")
        result = {"winner": 1, "reason": "human-concession"} if concession else replay.get("outcome")
        if not result:
            raise ValueError("Missing terminal outcome; no match result assigned.")
        if not concession:
            terminal_score(replay, 0)
        if concession:
            replay.update(status="truncated", outcome=None, concession={"playerId": 0, "reason": "human-concession"})
        replay["humanMatch"] = {"matchId": record["id"], "gameNumber": record["gameNumber"], "mode": record["mode"],
                                "modelVersion": record["modelVersion"], "reviewed": False}
        # Excluded from automatic training, even after later review.
        replay["evaluationExperiment"] = f"human-{record['id']}"
        record["games"].append(replay)
        record["gameResult"] = result
        winner = result.get("winner")
        if winner in (0, 1):
            record["score"][winner] += 1
            record["nextStarterChooser"] = 1 - winner
        else:
            record["nextStarterChooser"] = None
        self._capture_knowledge(record)
        record["status"] = "completed" if max(record["score"]) >= 2 else "between-games"

    def command(self, identifier, operation, revision, request_id, **arguments):
        with self.lock:
            record = self.get_private(identifier)
            payload = {"operation": operation, "revision": revision, **arguments}
            previous = self._check(record, revision, request_id, payload)
            if previous is not None:
                return previous
            if record["status"] in {"completed", "abandoned"}:
                raise MatchConflict("The match has ended.")
            if operation == "next-game" and record["status"] == "between-games" and record.get("nextStarterChooser") == 0 and arguments.get("firstPlayer") not in (0, 1):
                raise ValueError("Choose who starts the next game.")
            try:
                if operation == "abandon":
                    if record["status"] not in {"paused", "between-games"}:
                        raise MatchConflict("Pause the match before ending it as incomplete.")
                    record.update(status="abandoned", error=None,
                                  abandonment={"reason": "user-ended-incomplete-match", "outcome": None})
                elif operation == "pause":
                    if record["status"] != "active": raise MatchConflict("Only an active game can be paused.")
                    record["status"] = "paused"
                elif operation == "resume":
                    if record["status"] != "paused": raise MatchConflict("The game is not paused.")
                    with self.pool.lease() as engine:
                        self._restore(engine, record)
                    record.update(status="active", error=None)
                elif operation == "next-game":
                    if record["status"] != "between-games": raise MatchConflict("Finish this game first.")
                    first = arguments.get("firstPlayer")
                    if record.get("nextStarterChooser") == 0 and first not in (0, 1):
                        raise ValueError("Choose who starts the next game.")
                    if record.get("nextStarterChooser") == 1: first = 1
                    record.update(gameNumber=record["gameNumber"] + 1, seed=secrets.randbits(32), actions=[],
                                  firstPlayer=first, budgetTurn=None, spentMs=0, status="active", gameResult=None)
                    with self.pool.lease() as engine:
                        self._verify_identity(engine, record)
                        params = {"seed": record["seed"], "decks": record["decks"]}
                        if first is not None: params["firstPlayer"] = first
                        engine.request("reset", params)
                        record["observation"] = engine.request("observe", {"playerId": 0})
                elif operation in {"action", "concede"}:
                    if record["status"] != "active": raise MatchConflict("Resume the active game before acting.")
                    if operation == "action":
                        observation = record["observation"]
                        if observation["decisionPlayer"] != 0 or arguments["actionId"] not in {a["id"] for a in observation["legalActions"]}:
                            raise MatchConflict("This is not a legal action in the current position.")
                    with self.pool.lease() as engine:
                        self._restore(engine, record)
                        if operation == "action":
                            engine.request("step", {"actionId": arguments["actionId"]})
                            record["actions"].append({"actionId": arguments["actionId"], "playerId": 0})
                            record["observation"] = engine.request("observe", {"playerId": 0})
                        if operation == "concede" or record["observation"]["status"] == "finished":
                            self._finish_game(record, engine, operation == "concede")
                else:
                    raise ValueError("Unknown match operation.")
                self._capture_knowledge(record)
                return self._ack(record, request_id, payload)
            except MatchConflict:
                raise
            except Exception as exc:
                # Do not persist the in-memory action if durable acceptance failed.
                saved = self.get_private(identifier)
                saved.update(status="paused" if saved["status"] == "active" else saved["status"], error=str(exc)[:1000])
                try: self._save(saved)
                except Exception: pass
                raise

    def advance(self, identifier):
        with self.lock:
            record = self.get_private(identifier)
            if record["status"] != "active" or record["observation"]["decisionPlayer"] == 0:
                return self._public(record)
            if identifier not in self.pending:
                self.pending.add(identifier)
                self.executor.submit(self._engine_turn, identifier)
            return self._public(record)

    def _engine_turn(self, identifier):
        try:
            from .resources import ResourceGuard
            guard_context = ResourceGuard(self.settings) if hasattr(self.settings, "max_memory_bytes") else nullcontext()
            with self.pool.lease() as engine, guard_context as guard:
                with self.lock:
                    record = self.get_private(identifier)
                    self._restore(engine, record)
                agent = None
                # A bounded batch protects against accidental endless effect loops.
                for _ in range(256):
                    if guard is not None:
                        guard.check()
                    observation = engine.request("observe", {"playerId": 1})
                    if observation["decisionPlayer"] != 1 or observation["status"] != "running": break
                    with self.lock:
                        record = self.get_private(identifier)
                    turn = observation["turn"]
                    if turn != record.get("budgetTurn"):
                        record.update(budgetTurn=turn, spentMs=0)
                    remaining = max(0, record["budgetMs"] - record["spentMs"])
                    started = time.monotonic()
                    if remaining <= 0:
                        # Resolving a mandatory prompt after the thinking budget expires
                        # still needs a legal answer, but must not start another search.
                        action_id = Agent("heuristic", record["searchSeed"] + len(record["actions"])).choose(observation)
                    else:
                        if agent is None:
                            agent = Agent(record["policy"], record["searchSeed"] + len(record["actions"]))
                        if hasattr(agent, "rng"):
                            agent.rng.seed(record["searchSeed"] + len(record["actions"]))
                        action_id = agent.choose(observation)
                    elapsed = math.ceil((time.monotonic() - started) * 1000)
                    search_remaining = remaining - elapsed
                    if search_remaining > 100 and len(observation["legalActions"]) > 1 and observation.get("searchPosition"):
                        params = {"observation": observation, "budgetMs": min(1000, search_remaining),
                            "method": "rollout" if record["policy"] == "heuristic" else "ismcts",
                            "seed": (record["searchSeed"] + len(record["actions"])) % 2**32,
                            "priorRevealedCards": record.get("publicKnowledgeByPlayer", [[], []])[0]}
                        if record["policy"] != "heuristic":
                            from .training import predict
                            _, scores = predict(Path(record["policy"]), observation, agent.loaded)
                            if len(scores) != len(observation["legalActions"]) or not all(math.isfinite(score) for score in scores):
                                raise ValueError("Frozen policy produced invalid action scores; the match is paused.")
                            weights = [math.exp(score - max(scores)) for score in scores]
                            total = sum(weights)
                            params["rootPriors"] = [{"actionId": action["id"], "probability": weight / total}
                                                    for action, weight in zip(observation["legalActions"], weights)]
                        if record["knownList"]:
                            params["knownOpponentDeckId"] = record["decks"][0]
                        search = engine.request("search", params)
                        alternatives = search.get("alternatives", []) if search.get("status") == "complete" else []
                        legal = {a["id"] for a in observation["legalActions"]}
                        measured = [a for a in alternatives if a.get("actionId") in legal and a.get("visits", 0) > 0
                                    and isinstance(a.get("score"), (int, float)) and math.isfinite(a["score"])]
                        if measured:
                            action_id = max(measured, key=lambda a: a["score"])["actionId"]
                    record["spentMs"] += math.ceil((time.monotonic() - started) * 1000)
                    engine.request("step", {"actionId": action_id})
                    record["actions"].append({"actionId": action_id, "playerId": 1})
                    record["observation"] = engine.request("observe", {"playerId": 0})
                    record["revision"] += 1
                    self._capture_knowledge(record)
                    if record["observation"]["status"] == "finished": self._finish_game(record, engine)
                    with self.lock: self._save(record)
                    if record["status"] != "active": break
                else:
                    raise ValueError("Engine decision batch limit reached; resume to continue, no result assigned.")
        except Exception as exc:
            with self.lock:
                record = self.get_private(identifier)
                record.update(status="paused", error=str(exc)[:1000])
                try: self._save(record)
                except Exception: pass
        finally:
            with self.lock: self.pending.discard(identifier)

    def bookmark(self, identifier, title):
        with self.lock:
            record = self.get_private(identifier)
            item = {"id": uuid.uuid4().hex, "schemaVersion": 1, "familyId": f"human-{identifier}",
                    "partition": "test", "title": title, "sourceMatchId": identifier, "gameNumber": record["gameNumber"],
                    "revision": record["revision"], "observation": record["observation"], "playerId": 0,
                    "engineVersion": record["engineIdentity"].get("engineVersion"), "reviewStatus": "draft",
                    "deckVersion": record["registryHash"], "formatDate": "2026-09-17",
                    "source": {"kind": "human-match", "matchId": identifier, "modelVersion": record["modelVersion"]},
                    "rulesValidation": "legal-position", "positionHash": digest(record["observation"]), "trainingEligible": False, "reason": "human-bookmark",
                    "createdAt": datetime.now(timezone.utc).isoformat()}
            self.store.put("teaching", item["id"], item)
            return {k: v for k, v in item.items() if k != "observation"}

    def publish(self, identifier):
        with self.lock:
            record = self.get_private(identifier)
            if record["status"] != "completed": raise MatchConflict("Full replays become available after the whole match finishes.")
            if not record["replayIds"]:
                record["replayIds"] = [self.store.save_replay(copy.deepcopy(game)) for game in record["games"]]
                self._save(record)
            return {"replayIds": record["replayIds"]}
