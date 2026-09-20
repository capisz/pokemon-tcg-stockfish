"""One manually launched, durable experimental learning supervisor per API."""
from __future__ import annotations

import copy
import os
import random
import platform
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from itertools import islice

from .config import Settings
from .engine import EngineError
from .features import resource_features
from .learning_games import LearningPaused, LearningYield, frozen_policy, validate_policy, run_game
from .learning_protocol import configuration, collection_assignment, comparison_assignment, comparison_summary, game_seed
from .presentation import FrameStream, project_replay
from .resources import ResourceGuard, check_storage, process_memory
from .storage import Store, digest, file_digest, terminal_score


def now():
    return datetime.now(timezone.utc).isoformat()


class KeepAwake:
    def __init__(self):
        self.process = None

    def set(self, enabled):
        if enabled and self.process is None and platform.system() == "Darwin":
            self.process = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif not enabled and self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None


class LearningService:
    def __init__(self, settings: Settings, source: Store, pool, registry, interactive_busy=lambda: False):
        self.settings, self.source, self.pool, self.registry = settings, source, pool, registry
        self.store = Store(settings.data / "experimental", settings.max_disk_bytes, settings.min_free_bytes)
        self.stream = FrameStream(self.store)
        self.interactive_busy = interactive_busy
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ptcg-learning")
        self.future = None
        self.shutdown = threading.Event()
        self.foreground = 0
        self.awake = KeepAwake()
        self.last_resource_check = 0.
        self.implementation = digest({name: file_digest(Path(__file__).parent / name) for name in
                                      ("learning.py", "learning_games.py", "learning_protocol.py", "selfplay.py", "training.py", "experimental.py", "dataset.py", "features.py", "teaching_audits.py", "model.py", "storage.py", "resources.py", "presentation.py", "engine.py", "checkpoint_files.py", "decision_guard.py")})

    def private(self, identifier):
        return self.store.get("learning-runs", identifier)

    def update(self, identifier, **values):
        with self.lock:
            record = self.private(identifier)
            if record["desired"] == "stopped":
                if "desired" in values:
                    values["desired"] = "stopped"
                if "status" in values:
                    values["status"] = "stopped"
            if values.get("status") == "running" and record["desired"] != "running":
                values.pop("status")
            record.update(values, updatedAt=now())
            self.store.put("learning-runs", identifier, record)
            return record

    def public(self, record):
        allowed = ("id", "schemaVersion", "revision", "status", "phase", "cycle", "createdAt", "updatedAt",
                   "metrics", "error", "pauseReason")
        result = {key: copy.deepcopy(record[key]) for key in allowed if key in record}
        result["configuration"] = {key: value for key, value in record["configuration"].items() if key != "seed"}
        for key in ("incumbent", "candidate", "guidePolicy"):
            policy = record.get(key)
            result[key] = {"version": policy["hash"], "name": policy["name"], "dataTier": "experimental"} if policy else None
        result["activeGames"] = [self.game_summary(self.store.get("learning-games", identifier))
                                 for identifier in record.get("pendingGames", [])]
        if result.get("error"):
            safe = ("Aggregate project", "Local data disk", "Destination free-space", "Implementation changed", "Engine build changed", "Too many truncated")
            if "request" in result["error"].lower() and ("exceeds" in result["error"].lower() or "exceeded" in result["error"].lower()):
                result["error"] = "A simulator message exceeded the supported size. Update the app before resuming; retrying alone cannot fix this."
            elif not result["error"].startswith(safe):
                result["error"] = "Learning paused after a local failure. Inspect its private run journal and resume after resolving the cause."
        result["dataTier"] = "experimental"
        return result

    def list(self):
        return [self.public(record) for record in self.store.list("learning-runs")]

    def get(self, identifier):
        return self.public(self.private(identifier))

    def recover(self):
        for record in self.store.list("learning-runs"):
            if record["status"] in {"running", "pausing", "waiting-for-play"}:
                self.update(record["id"], status="paused", desired="paused", revision=record["revision"] + 1,
                            pauseReason="Application restarted. Resume restores the acknowledged journals.")

    def create(self, options=None):
        config = configuration(options)
        if config["maxDecisions"] > self.settings.max_decisions:
            raise ValueError("Decision limit exceeds this machine's configured limit")
        with self.lock:
            if any(r["status"] != "stopped" for r in self.store.list("learning-runs")):
                raise ValueError("Resume or stop the existing learning run before creating another")
            decks = [d for d in self.registry() if d.get("role") in {"main", "training-variant"}]
            if len(decks) != 10 or len({d["archetype"] for d in decks}) != 5:
                raise ValueError("Continuous learning requires all five main and training-variant pairs")
            with self.pool.lease() as engine:
                identity = engine.request("health")
            identifier = uuid.uuid4().hex
            record = {"schemaVersion": 1, "id": identifier, "revision": 0, "status": "running", "desired": "running",
                      "phase": "initializing", "cycle": 0, "configuration": config, "createdAt": now(), "updatedAt": now(),
                      "engine": identity, "decks": decks, "deckHash": digest(decks), "implementationHash": self.implementation,
                      "collectionCursor": 0, "pendingGames": [], "collectedReplayIds": [], "batchReplayIds": [],
                      "incumbent": frozen_policy(self.store, "heuristic"), "candidate": None, "guidePolicy": None,
                      "history": [], "controls": {}, "metrics": {"completedGames": 0, "truncatedGames": 0, "errors": 0,
                       "searchDecisions": 0, "searchTargets": 0, "peakMemoryBytes": 0, "managedBytes": 0, "freeBytes": None,
                       "coverage": {"scheduledPairs": 0, "completedPairs": 0, "totalPairs": 100, "archetypes": 5},
                       "comparison": None, "guideAgreement": {"status": "unavailable", "positions": 0, "acceptableActionAccuracy": None}, "loss": None}}
            self.store.put("learning-runs", identifier, record)
            self.future = self.executor.submit(self._run, identifier)
            return self.public(record)

    def control(self, identifier, operation, revision, request_id):
        if operation not in {"pause", "resume", "stop"}:
            raise ValueError("Unknown learning control")
        with self.lock:
            record = self.private(identifier)
            fingerprint = digest({"operation": operation, "revision": revision})
            prior = record["controls"].get(request_id)
            if prior:
                if prior != fingerprint:
                    raise ValueError("Control request ID was already used differently")
                return self.public(record)
            if revision != record["revision"]:
                raise ValueError("Learning run changed; refresh before submitting another control")
            if record["status"] == "stopped":
                raise ValueError("Stopped runs remain immutable; create a new run")
            desired = "running" if operation == "resume" else "stopped" if operation == "stop" else "paused"
            if operation == "resume" and record["status"] in {"running", "waiting-for-play"}:
                raise ValueError("Learning is already enabled")
            if operation == "resume" and record["status"] == "pausing":
                raise ValueError("Pause is still checkpointing; resume after the run reports paused")
            if operation == "resume" and record["implementationHash"] != self.implementation:
                raise ValueError("Implementation changed; stop this preserved run and start a new one")
            record["controls"][request_id] = fingerprint
            record["controls"] = dict(list(record["controls"].items())[-100:])
            record.update(desired=desired, revision=revision+1, status="running" if operation == "resume" else "pausing",
                          error=None, pauseReason=None, updatedAt=now())
            running = self.future is not None and not self.future.done()
            if not running and operation != "resume":
                record["status"] = desired
            if operation == "resume":
                for game_id in record.get("pendingGames", []):
                    game = self.store.get("learning-games", game_id)
                    if game.get("workerFailures"):
                        game["lifetimeWorkerFailures"] = game.get("lifetimeWorkerFailures", 0)+game["workerFailures"]
                        game["workerFailures"] = 0
                        self.save_game(game)
            self.store.put("learning-runs", identifier, record)
            if operation == "resume":
                # Queue after any final cleanup of the previous invocation.
                self.future = self.executor.submit(self._run, identifier)
            return self.public(record)

    @contextmanager
    def priority(self):
        with self.lock:
            self.foreground += 1
        try:
            yield
        finally:
            with self.lock:
                self.foreground -= 1

    def check(self, identifier, guard=None):
        if self.shutdown.is_set():
            raise LearningPaused("Application shutting down; accepted progress is saved")
        state = self.private(identifier)
        if state["desired"] != "running":
            raise LearningPaused("Paused by user" if state["desired"] == "paused" else "Stopped by user")
        if self.foreground or self.interactive_busy():
            raise LearningYield("Interactive play or analysis has priority")
        if guard:
            guard.check(storage=False)
        if time.monotonic() - self.last_resource_check >= 5:
            with self.lock:
                resources = check_storage(self.settings.data, self.settings.max_disk_bytes, self.settings.min_free_bytes)
                record = self.private(identifier)
                record["metrics"].update(resources)
                record["metrics"]["peakMemoryBytes"] = max(record["metrics"]["peakMemoryBytes"], process_memory(), getattr(guard, "peak", 0))
                self.store.put("learning-runs", identifier, record)
                self.last_resource_check = time.monotonic()

    def save_game(self, game):
        with self.lock:
            game["updatedAt"] = now()
            self.store.put("learning-games", game["id"], game)

    def commit_frame(self, game, observations, action, actor):
        game.update(observationHash=digest(observations), decisionIndex=len(game["actions"]), turn=observations[0]["turn"])
        cursor = self.stream.append(game["id"], {"decisionIndex": len(game["actions"]), "actor": actor,
                    "priorAction": action, "observations": observations}, game["frameCursor"])
        game["frameCursor"] = cursor
        self.save_game(game)

    def compact_game(self, game):
        # The complete immutable replay is the canonical saved feed. Only its
        # redundant runner-owned stream is disposable; user artifacts never are.
        if not game.get("replayAvailable"):
            return
        replay = self.store.get("replays", game["replayId"])
        if len(replay["frames"]) != game["frameCursor"]+1 or digest(replay["frames"][-1]["observations"]) != game["observationHash"]:
            raise ValueError("Saved replay differs from acknowledged frames; retained stream for inspection")
        with self.stream.lock:
            self.stream._path(game["id"]).unlink(missing_ok=True)
            self.stream.offsets.pop(game["id"], None)

    def game_summary(self, game):
        return {key: game.get(key) for key in ("id", "workerIndex", "status", "purpose", "archetypes",
                    "decisionIndex", "turn", "frameCursor", "replayAvailable", "createdAt")}

    def games(self, identifier):
        return [self.game_summary(g) for g in islice((g for g in self.store.iter_records("learning-games") if g["runId"] == identifier), 200)]

    def policy_context(self, game, player_id=None):
        return {"policy": "model" if any(p["path"] not in {"heuristic", "random"} for p in game["policies"]) else "heuristic",
                "modelVersion": game["policies"][player_id]["hash"] if player_id is not None else " / ".join(p["hash"] for p in game["policies"]), "dataTier": "experimental",
                "decisionGuard": "visible-repetition-v1", "trainingStatus": "experimental-frozen-game", "learnsDuringRun": False,
                "opponentPopulation": "Experimental heuristics, reviewed policy, incumbent and historical checkpoints",
                "computeBudget": f"{game['searchBudgetMs']} ms per decision; 120 seconds per turn ceiling",
                "runId": game["runId"], "cycle": game["cycle"], "purpose": game["purpose"]}

    def frames(self, identifier, player_id=0, after=-1, limit=100):
        game = self.store.get("learning-games", identifier)
        run = self.private(game["runId"])
        status = game["status"] if run["status"] == "running" or game.get("replayAvailable") else run["status"]
        if player_id not in (0, 1) or after < -1 or not 1 <= limit <= 100:
            raise ValueError("Invalid frame cursor, limit or player perspective")
        if game.get("replayAvailable"):
            frames = []
            if after < game["frameCursor"]:
                projected = project_replay(self.store.get("replays", game["replayId"]), player_id)
                frames = projected["frames"][after+1:after+1+limit]
            cursor = frames[-1]["cursor"] if frames else after
            result = {"schemaVersion": 1, "frames": frames, "nextCursor": cursor,
                      "hasMore": cursor < game["frameCursor"], "status": status}
        else:
            result = self.stream.read(identifier, after=after, limit=limit, committed=game["frameCursor"], player_id=player_id, status=status)
        result.update(policyContext=self.policy_context(game, player_id), runId=game["runId"], gameId=identifier)
        if game.get("replayAvailable"):
            result["replayId"] = "experimental-" + identifier
        return result

    def replay(self, identifier, player_id=0):
        game = self.store.get("learning-games", identifier)
        if not game.get("replayAvailable"):
            raise ValueError("This learning game has not completed its replay")
        result = project_replay(self.store.get("replays", game["replayId"]), player_id)
        result.update(id="experimental-"+identifier, dataTier="experimental", policyContext=self.policy_context(game, player_id))
        return result

    def reserve_comparison_seed(self, run, key):
        seed = game_seed(run["configuration"]["seed"], "comparison", key)
        owner = digest({"runId": run["id"], "key": key})
        # A permanent ledger makes fresh comparison seeds a checked invariant,
        # including the unlikely collision of two truncated hash values.
        with self.lock:
            while self.store.location("comparison-seeds", str(seed)).exists():
                if self.store.get("comparison-seeds", str(seed))["owner"] == owner:
                    return seed
                seed = 1_000_000_000 + (seed-1_000_000_000+1) % 900_000_000
            self.store.put("comparison-seeds", str(seed), {"id": str(seed), "owner": owner,
                "runId": run["id"], "dataTier": "experimental", "excludedFromTraining": True})
        return seed

    def _new_game(self, run, assignment, ordinal, purpose, worker_index):
        key = f"{run['cycle']}:{ordinal}" if purpose == "comparison" else str(ordinal)
        identifier = digest({"run": run["id"], "purpose": purpose, "index": key})[:32]
        if self.store.location("learning-games", identifier).exists():
            return identifier
        decks = {deck["id"]: deck for deck in run["decks"]}
        learner = run["candidate"] if purpose == "comparison" else (run["incumbent"] if run["incumbent"]["path"] != "heuristic" else run["guidePolicy"])
        opponent = run["incumbent"] if purpose == "comparison" else assignment["opponent"]
        policies = [learner, opponent] if assignment["candidateSeat"] == 0 else [opponent, learner]
        archetypes = [decks[d]["archetype"] for d in assignment["decks"]]
        seed_index = f"{run['id']}:{run['cycle']}:{assignment['pairNumber']}" if purpose == "comparison" else f"{run['id']}:{ordinal}"
        game = {"id": identifier, "runId": run["id"], "cycle": run["cycle"], "purpose": purpose,
                "workerIndex": worker_index, "createdAt": now(), "status": "queued", "frameCursor": -1,
                "replayAvailable": False, "seed": self.reserve_comparison_seed(run, seed_index) if purpose == "comparison" else game_seed(run["configuration"]["seed"], purpose, seed_index),
                **{k: v for k, v in assignment.items() if k != "opponent"},
                "policies": copy.deepcopy(policies), "archetypes": archetypes,
                "archetypePair": " / ".join(archetypes if assignment["candidateSeat"] == 0 else archetypes[::-1]),
                "deckRoles": [decks[d]["role"] for d in assignment["decks"]],
                "deckHashes": [decks[d]["listHash"] for d in assignment["decks"]],
                "maxDecisions": run["configuration"]["maxDecisions"], "searchBudgetMs": run["configuration"]["searchBudgetMs"],
                "actions": [], "searchTargets": [], "searchDecisions": 0, "spentMs": {}, "decisionIndex": 0, "turn": 0}
        self.save_game(game)
        return identifier

    def _play_pending(self, identifier, guard):
        ids = self.private(identifier)["pendingGames"]
        with ThreadPoolExecutor(max_workers=min(self.settings.workers, len(ids)), thread_name_prefix="learning-game") as workers:
            futures = [workers.submit(run_game, self, identifier, game_id, slot, guard) for slot, game_id in enumerate(ids)]
            return [future.result() for future in futures]

    def _collect(self, identifier, guard):
        while True:
            self.check(identifier, guard)
            run = self.private(identifier)
            if len(run["batchReplayIds"]) >= run["configuration"]["gamesPerBatch"]:
                self.update(identifier, phase="investigating", pendingGames=[])
                return
            if not run["pendingGames"]:
                opponents = [frozen_policy(self.store, "heuristic"), run["guidePolicy"], run["incumbent"], *run.get("history", [])[-3:]]
                ids = []
                count = min(self.settings.workers, run["configuration"]["gamesPerBatch"]-len(run["batchReplayIds"]))
                deck_ids = [d["id"] for d in run["decks"]]
                for slot in range(count):
                    index = run["collectionCursor"] + slot
                    ids.append(self._new_game(run, collection_assignment(index, deck_ids, opponents), index, "collection", slot))
                self.update(identifier, pendingGames=ids, collectionCursor=run["collectionCursor"]+count)
            games = self._play_pending(identifier, guard)
            with self.lock:
                run = self.private(identifier)
                completed_pairs = set(run.get("completedPairKeys", []))
                for game in games:
                    if game["status"] == "finished":
                        run["batchReplayIds"].append(game["replayId"])
                        run["collectedReplayIds"].append(game["replayId"])
                        run["metrics"]["completedGames"] += 1
                        completed_pairs.add(game["pairKey"])
                    else:
                        run["metrics"]["truncatedGames"] += 1
                    run["metrics"]["searchDecisions"] += game["searchDecisions"]
                    run["metrics"]["searchTargets"] += len(game["searchTargets"])
                run["pendingGames"] = []
                run["completedPairKeys"] = sorted(completed_pairs)
                run["metrics"]["coverage"].update(scheduledPairs=min(100, run["collectionCursor"]), completedPairs=len(completed_pairs))
                self.store.put("learning-runs", identifier, run)
            if run["metrics"]["truncatedGames"] > max(20, run["metrics"]["completedGames"]):
                raise ValueError("Too many truncated games to learn outcomes; inspect the saved games before resuming")

    def _investigate(self, identifier, guard):
        from .experimental import experimental_partition
        from .selfplay import Agent, search_choice
        run = self.private(identifier)
        if "investigationQueue" not in run:
            from .features import heuristic_action_score
            from .training import predict
            policy = run["incumbent"] if run["incumbent"]["path"] != "heuristic" else run["guidePolicy"]
            scout = Agent(policy["path"], run["configuration"]["seed"], allow_experimental=True)
            candidates = []
            for replay_id in run["batchReplayIds"]:
                self.check(identifier, guard)
                replay = self.store.get("replays", replay_id)
                if experimental_partition(replay) != "train":
                    continue
                seen_groups = set()
                indices = list(range(len(replay["frames"])))
                random.Random(f"{run['id']}:{run['cycle']}:{replay_id}").shuffle(indices)
                for offset, frame_index in enumerate(indices[:256]):
                    if offset % 16 == 0:
                        self.check(identifier, guard)
                    frame = replay["frames"][frame_index]
                    obs = frame["observations"][frame["actor"]]
                    if not frame.get("action") or not obs.get("searchPosition") or len(obs["legalActions"]) < 2:
                        continue
                    features = resource_features(obs).tolist()
                    group = digest({"decks": replay["decks"], "resources": [round(x*3) for x in features], "turn": min(obs["turn"]//4, 5)})
                    if group in seen_groups:
                        continue
                    seen_groups.add(group)
                    scores = predict(Path(scout.policy), obs, scout.loaded, allow_experimental=True)[1] if scout.loaded else [heuristic_action_score(a, obs) for a in obs["legalActions"]]
                    ranked = sorted(scores, reverse=True)
                    margin = ranked[0] - ranked[1]
                    heuristic = max(range(len(scores)), key=lambda i: heuristic_action_score(obs["legalActions"][i], obs))
                    disagreement = scores[heuristic] < ranked[0]
                    candidates.append({"replayId": replay_id, "decisionIndex": frame["decisionIndex"], "actor": frame["actor"],
                                       "group": group, "priority": (0 if disagreement else 1, margin, sum(abs(x) for x in features)),
                                       "disagreement": disagreement, "policyMargin": margin})
            # Favor close resource positions, round-robin across games; never copy
            # an action target from a similar but distinct information set.
            candidates.sort(key=lambda x: (x["priority"], x["group"]))
            chosen, per_game = [], {}
            for item in candidates:
                if per_game.get(item["replayId"], 0) >= 3 or any(row["group"] == item["group"] for row in chosen):
                    continue
                chosen.append(item)
                per_game[item["replayId"]] = per_game.get(item["replayId"], 0)+1
                if len(chosen) >= run["configuration"]["investigationPositions"]:
                    break
            if run["configuration"]["investigationPositions"] == 0:
                chosen = []
            run = self.update(identifier, investigationQueue=chosen, investigationCursor=0)
        policy = run["incumbent"] if run["incumbent"]["path"] != "heuristic" else run["guidePolicy"]
        agent = Agent(policy["path"], run["configuration"]["seed"], allow_experimental=True)
        for index in range(run["investigationCursor"], len(run["investigationQueue"])):
            self.check(identifier, guard)
            item = run["investigationQueue"][index]
            replay = self.store.get("replays", item["replayId"])
            frame = next(f for f in replay["frames"] if f["decisionIndex"] == item["decisionIndex"])
            observation = frame["observations"][item["actor"]]
            targets = []
            with self.pool.lease(wait_timeout=5) as engine:
                identity = engine.request("health")
                if any(identity.get(key) != run["engine"].get(key) for key in ("engineVersion", "engineBuildHash")):
                    raise ValueError("Engine build changed before investigation; start a new run")
                for draw in (0, 1):
                    self.check(identifier, guard)
                    _, target = search_choice(engine, observation, agent,
                        seed=game_seed(run["configuration"]["seed"], "investigation", f"{run['cycle']}:{index}:{draw}"),
                        budget_ms=max(1, run["configuration"]["investigationBudgetMs"]//2))
                    if target:
                        targets.append(target)
            key = digest({"replayId": item["replayId"], "decisionIndex": item["decisionIndex"], "policyHash": policy["hash"]})
            evidence = {"id": key, **item, "observationHash": digest(observation), "policyHash": policy["hash"],
                        "dataTier": "experimental", "status": "unsupported", "independentSeeds": 2,
                        "budgetMs": run["configuration"]["investigationBudgetMs"]}
            if len(targets) == 2:
                evidence.update(status="supported", method="ismcts", source="reanalysis", probabilities={
                    action: sum(t["probabilities"][action] for t in targets)/2 for action in targets[0]["probabilities"]})
                self.store.put("search-targets", key, evidence)
            self.store.put("position-bank", key, evidence)
            metrics = self.private(identifier)["metrics"]
            metrics["investigatedPositions"] = index+1
            metrics["investigationTargets"] = metrics.get("investigationTargets", 0) + int(evidence["status"] == "supported")
            self.update(identifier, investigationCursor=index+1, metrics=metrics)
        self.update(identifier, phase="training")

    def buffer_replays(self, run):
        selected, size = [], 0
        for identifier in reversed(run["collectedReplayIds"][-2000:]):
            amount = self.store.location("replays", identifier).stat().st_size
            if size + amount > 240 * 1024**2:
                break
            selected.append(identifier)
            size += amount
        if not selected:
            raise ValueError("A completed replay exceeds the bounded training input size")
        return selected[::-1]

    def _train(self, identifier, guard):
        from .experimental import snapshot_teaching, prepare_dataset, train_experimental, InsufficientExperimentalData
        run = self.private(identifier)
        if not run.get("datasetId"):
            snapshot_id = f"{identifier}-cycle-{run['cycle']}"
            snapshot_teaching(self.source, self.store, snapshot_id)
            replay_ids = self.buffer_replays(run)
            try:
                manifest = prepare_dataset(self.store, replay_ids, teaching_snapshot_id=snapshot_id,
                                           seed=(run["configuration"]["seed"]+run["cycle"]) % 2**32, max_rows=run["configuration"]["maxPositions"])
            except InsufficientExperimentalData:
                self.update(identifier, phase="collecting", batchReplayIds=[],
                            pauseReason="Collecting another batch to obtain usable whole-game training partitions")
                return
            run = self.update(identifier, datasetId=manifest["id"], teachingSnapshotId=snapshot_id,
                              trainingId=uuid.uuid4().hex)
        path = self.store.path / "models" / f"{run['trainingId']}.pt"
        parent = run["incumbent"] if run["incumbent"]["path"] != "heuristic" else run["guidePolicy"]
        validate_policy(parent)
        report = train_experimental(self.store, dataset_id=run["datasetId"], run_id=run["trainingId"],
                    seed=(run["configuration"]["seed"]+run["cycle"]) % 2**32, checkpoint=path if path.exists() else Path(parent["path"]),
                    resume=path.exists(), guard=guard)
        self.check(identifier, guard)
        if report["status"] != "completed":
            raise LearningPaused(report.get("pauseReason", "Training checkpointed"))
        metrics = self.private(identifier)["metrics"]
        metrics["loss"] = report.get("history", [{}])[-1].get("loss")
        self.update(identifier, phase="comparing", candidate=frozen_policy(self.store, report["checkpoint"]),
                    metrics=metrics, comparisonCursor=0, comparisonRecords=[])

    def _compare(self, identifier, guard):
        from .experimental import tactical_regressions
        run = self.private(identifier)
        deck_ids = [d["id"] for d in run["decks"]]
        total = len(deck_ids)**2 * run["configuration"]["comparisonSeeds"] * 2
        limit = min(total, run["configuration"].get("comparisonGameLimit") or total)
        protocol_id = f"{identifier}-{run['cycle']}"
        if not self.store.location("comparison-protocols", protocol_id).exists():
            self.store.put("comparison-protocols", protocol_id, {"id": protocol_id, "dataTier": "experimental",
                "engine": run["engine"], "deckHash": run["deckHash"], "candidate": run["candidate"], "incumbent": run["incumbent"],
                "configuration": run["configuration"], "expectedGames": total, "createdAt": now(),
                "schedule": "ordered pairs; independent seeds; exchanged seats; paired Hoeffding 95% interval",
                "adoption": "complete coverage, lower > 0.5, no supported matchup or reviewed tactical regression"})
        while run["comparisonCursor"] < limit or run["pendingGames"]:
            self.check(identifier, guard)
            if not run["pendingGames"]:
                ids = []
                for slot in range(min(self.settings.workers, limit-run["comparisonCursor"])):
                    index = run["comparisonCursor"] + slot
                    assignment = comparison_assignment(index, deck_ids, run["configuration"]["comparisonSeeds"])
                    ids.append(self._new_game(run, assignment, index, "comparison", slot))
                run = self.update(identifier, pendingGames=ids, comparisonCursor=run["comparisonCursor"]+len(ids))
            games = self._play_pending(identifier, guard)
            records = list(run["comparisonRecords"])
            for game in games:
                replay = self.store.get("replays", game["replayId"])
                records.append({**{key: game[key] for key in ("decks", "pairKey", "pairIndex", "seedOffset", "firstPlayer", "status", "seed")},
                                "gameId": game["id"], "pairNumber": game["pairNumber"], "candidateSeat": game["candidateSeat"],
                                "archetypePair": game["archetypePair"], "score": terminal_score(replay, game["candidateSeat"]) if replay["status"] == "finished" else None})
            metrics = self.private(identifier)["metrics"]
            metrics["comparison"] = {"status": "running", "completedGames": len(records), "totalGames": total,
                                     "mean": None, "lower": None, "upper": None, "adopted": False, "notes": []}
            run = self.update(identifier, pendingGames=[], comparisonRecords=records, metrics=metrics)
        validate_policy(run["candidate"])
        validate_policy(run["incumbent"])
        tactical = tactical_regressions(self.store, snapshot_id=run["teachingSnapshotId"],
                                       candidate=Path(run["candidate"]["path"]), incumbent=run["incumbent"]["path"])
        validate_policy(run["candidate"])
        validate_policy(run["incumbent"])
        summary = comparison_summary(run["comparisonRecords"], total, tactical, deck_ids=deck_ids,
                    seeds=run["configuration"]["comparisonSeeds"], archetypes={d["id"]: d["archetype"] for d in run["decks"]})
        self.store.put("comparisons", f"{identifier}-{run['cycle']}", {"id": f"{identifier}-{run['cycle']}",
                       "dataTier": "experimental", "engine": run["engine"], "deckHash": run["deckHash"],
                       "configuration": run["configuration"], "candidate": run["candidate"], "incumbent": run["incumbent"],
                       "records": run["comparisonRecords"], "tactical": tactical, "summary": summary})
        metrics = run["metrics"]
        metrics.update(comparison=summary, guideAgreement={"status": tactical.get("status", "unavailable"),
            "positions": tactical.get("positions", 0), "acceptableActionAccuracy": tactical.get("candidateAccuracy")})
        if summary["adopted"]:
            run["history"].append(run["incumbent"])
            run["incumbent"] = run["candidate"]
            self.store.put("incumbents", identifier, {"id": identifier, "dataTier": "experimental", "policy": run["incumbent"],
                                                      "comparison": f"{identifier}-{run['cycle']}"})
        self.update(identifier, phase="checkpointed", metrics=metrics, incumbent=run["incumbent"], history=run["history"])

    def _run(self, identifier):
        try:
            with ResourceGuard(self.settings) as resource_guard:
                service = self
                class Guard:
                    def check(self, storage=True):
                        service.check(identifier, resource_guard)
                guard = Guard()
                while True:
                    try:
                        self.check(identifier, resource_guard)
                        run = self.private(identifier)
                        self.awake.set(run["configuration"]["keepAwake"])
                        if run["status"] == "waiting-for-play":
                            run = self.update(identifier, status="running", pauseReason=None)
                        if run["phase"] == "initializing":
                            from .experimental import snapshot_teaching, bootstrap_policy
                            snapshot = identifier + "-bootstrap"
                            snapshot_teaching(self.source, self.store, snapshot)
                            report = bootstrap_policy(self.store, snapshot_id=snapshot, run_id=identifier+"-guide", guard=guard)
                            self.check(identifier, resource_guard)
                            if report["status"] != "completed":
                                raise LearningPaused("Guide-policy initialization checkpointed")
                            metrics = self.private(identifier)["metrics"]
                            positions = report.get("teachingDiagnostic", {}).get("positions", [])
                            metrics["guideAgreement"] = {"status": "measured" if positions else "unavailable",
                                "positions": len(positions), "acceptableActionAccuracy": sum(p["after"]["selectedIsAcceptable"] for p in positions)/len(positions) if positions else None}
                            self.update(identifier, guidePolicy=frozen_policy(self.store, report["checkpoint"]), phase="collecting", metrics=metrics)
                        elif run["phase"] == "collecting":
                            self._collect(identifier, guard)
                        elif run["phase"] == "investigating":
                            self._investigate(identifier, guard)
                        elif run["phase"] == "training":
                            self._train(identifier, guard)
                        elif run["phase"] == "comparing":
                            self._compare(identifier, guard)
                        else:
                            cycle = run["cycle"]+1
                            if run["configuration"]["maxCycles"] and cycle >= run["configuration"]["maxCycles"]:
                                self.update(identifier, cycle=cycle, desired="stopped", status="stopped")
                                return
                            run.update(cycle=cycle, phase="collecting", batchReplayIds=[], candidate=None, datasetId=None)
                            for key in ("investigationQueue", "investigationCursor", "trainingId", "comparisonRecords", "comparisonCursor"):
                                run.pop(key, None)
                            with self.lock:
                                latest = self.private(identifier)
                                run.update(desired=latest["desired"], revision=latest["revision"], controls=latest["controls"], status=latest["status"])
                                self.store.put("learning-runs", identifier, run)
                    except LearningYield as exc:
                        self.awake.set(False)
                        self.update(identifier, status="waiting-for-play", pauseReason=str(exc))
                        self.shutdown.wait(.25)
        except LearningPaused as exc:
            record = self.private(identifier)
            status = "stopped" if record["desired"] == "stopped" else "paused"
            self.update(identifier, status=status, desired=status, pauseReason=str(exc))
        except Exception as exc:
            try:
                record = self.private(identifier)
                record["metrics"]["errors"] += 1
                self.update(identifier, status="paused", desired="paused", error=str(exc)[:1000], pauseReason="Learning paused safely; inspect the recorded error.", metrics=record["metrics"])
            except (OSError, RuntimeError):
                try:
                    self.store.emergency_control_pause(identifier, {**self.private(identifier),
                        "status": "paused", "desired": "paused", "updatedAt": now(), "error": str(exc),
                        "pauseReason": "Resource or local failure; acknowledged progress is preserved. Resolve the recorded cause before resuming."})
                except (OSError, RuntimeError, ValueError):
                    pass  # A physically full/disconnected volume cannot acknowledge another write.
        finally:
            self.awake.set(False)

    def close(self):
        self.shutdown.set()
        self.executor.shutdown(wait=True)
        self.awake.set(False)
