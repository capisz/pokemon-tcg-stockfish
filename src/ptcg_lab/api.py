from __future__ import annotations

import threading
import math
import uuid
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from .analysis import analyze_observation, review_decision, select_frame, selected_action_id
from .config import Settings
from .engine import EngineError, EnginePool
from .storage import Store, file_digest
from .presentation import FrameStream, project_replay
from .matches import MatchService, MatchConflict
from . import teaching, guides


class MatchRequest(BaseModel):
    deckId: str = Field(min_length=1, max_length=160)
    opponentArchetype: str = Field(min_length=1, max_length=160)
    mode: str = Field(default="practice", pattern="^(practice|benchmark)$")
    knownList: bool = False
    budgetMs: int = Field(default=120000, ge=1, le=120000)
    modelId: str | None = Field(default="heuristic", pattern=r"^[A-Za-z0-9_-]{1,160}$")


class MatchActionRequest(BaseModel):
    revision: int = Field(ge=0)
    requestId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,160}$")
    actionId: str | None = Field(default=None, max_length=160)
    firstPlayer: int | None = Field(default=None, ge=0, le=1)


class BookmarkRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class TeachingReviewRequest(BaseModel):
    reviewStatus: str = Field(pattern="^(draft|reviewed|rejected)$")
    acceptableActionIds: list[str] = Field(default_factory=list, max_length=1000)
    rejectedActionIds: list[str] = Field(default_factory=list, max_length=1000)
    conditionalReasoning: str = Field(default="", max_length=10000)
    criticalResources: str = Field(default="", max_length=2000)
    confidence: str = Field(default="uncertain", pattern="^(uncertain|likely|confident)$")


class TeachingPositionRequest(BaseModel):
    positionId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,160}$")
    familyId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,160}$")


class GuideQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class GameRequest(BaseModel):
    decks: tuple[str, str]
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    maxDecisions: int = Field(default=1000, ge=1, le=3000)
    policy: str = Field(default="heuristic", pattern="^(random|heuristic|model)$")
    modelId: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,160}$")
    laboratory: bool = False

    @model_validator(mode="after")
    def model_selection(self):
        if (self.policy == "model") != bool(self.modelId):
            raise ValueError("Select a modelId exactly when policy is model")
        return self


class AnalysisRequest(BaseModel):
    replayId: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,160}$")
    decisionIndex: int | None = Field(default=None, ge=0)
    playerId: int | None = Field(default=None, ge=0, le=1)
    positionId: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,160}$")
    budgetMs: int = Field(default=250, ge=1, le=5000)

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.replayId) == bool(self.positionId):
            raise ValueError("Choose exactly one replayId or positionId")
        if self.replayId and (self.decisionIndex is None or self.playerId is None):
            raise ValueError("Replay analysis requires decisionIndex and playerId")
        if self.positionId and (self.decisionIndex is not None or self.playerId is not None):
            raise ValueError("Saved positions already specify their decision and player perspective")
        return self


class PositionRequest(BaseModel):
    replayId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,160}$")
    decisionIndex: int = Field(ge=0)
    playerId: int = Field(ge=0, le=1)
    title: str = Field(min_length=1, max_length=120)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # The research Store adds a reserve while preserving the legacy constructor.
    store = Store(settings.data, settings.max_disk_bytes, settings.min_free_bytes) if hasattr(settings, "min_free_bytes") else Store(settings.data, settings.max_disk_bytes)
    pool = EnginePool(settings.root, settings.workers, settings.engine_timeout)
    streams = FrameStream(store)
    executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="ptcg-job")
    capacity = threading.BoundedSemaphore(settings.max_jobs)
    deck_cache: list[dict] = []
    deck_cache_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # A process restart is not a game outcome. Preserve completed replays.
        for job in store.list("jobs"):
            if job["status"] in {"queued", "running"}:
                store.put("jobs", job["id"], {**job, "status": "interrupted", "error": "Server restarted; rerun the recorded seed and configuration."})
        matches.recover()
        yield
        matches.close()
        executor.shutdown(wait=True, cancel_futures=False)
        pool.close()

    app = FastAPI(title="Pokémon TCG Local Research Lab", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.state.store, app.state.pool = store, pool

    def decks() -> list[dict]:
        if deck_cache:
            return deck_cache
        try:
            with deck_cache_lock:
                if not deck_cache:
                    with pool.lease() as engine:
                        result = engine.request("decks")
                    deck_cache.extend(result["decks"] if isinstance(result, dict) else result)
            return deck_cache
        except (EngineError, FileNotFoundError) as exc:
            raise HTTPException(503, f"Engine unavailable. Run npm run engine:build. {exc}") from exc

    matches = MatchService(settings, store, pool, decks)
    app.state.matches = matches

    def hypothesis_decks():
        return [deck for deck in decks() if deck.get("role") not in {"heldout", "historical"}]

    def protect_benchmark():
        if matches.benchmark_active():
            raise HTTPException(403, "Analysis and private research views are locked until the benchmark match finishes.")

    def match_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Match or teaching record not found") from exc
        except MatchConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (EngineError, RuntimeError, OSError) as exc:
            raise HTTPException(503, "Local match operation failed; the durable position is preserved. Check local diagnostics and resume.") from exc

    @app.post("/api/matches", status_code=201)
    def create_match(request: MatchRequest):
        return match_call(matches.create, request.deckId, request.opponentArchetype, request.mode, request.knownList, request.budgetMs, request.modelId)

    @app.get("/api/matches")
    def list_matches():
        return {"matches": matches.list()}

    @app.get("/api/matches/{identifier}")
    def get_match(identifier: str):
        return match_call(matches.get, identifier)

    @app.get("/api/matches/{identifier}/frames")
    def match_frames(identifier: str, after: int = Query(-1, ge=-1), limit: int = Query(100, ge=1, le=100)):
        return match_call(matches.frames, identifier, after, limit)

    @app.post("/api/matches/{identifier}/actions")
    def match_action(identifier: str, request: MatchActionRequest):
        if request.actionId is None: raise HTTPException(422, "actionId is required")
        return match_call(matches.command, identifier, "action", request.revision, request.requestId, actionId=request.actionId)

    @app.post("/api/matches/{identifier}/advance", status_code=202)
    def advance_match(identifier: str):
        return match_call(matches.advance, identifier)

    @app.post("/api/matches/{identifier}/control/{operation}")
    def control_match(identifier: str, operation: str, request: MatchActionRequest):
        if operation not in {"pause", "resume", "concede", "next-game", "abandon"}: raise HTTPException(404, "Unknown match operation")
        arguments = {"firstPlayer": request.firstPlayer} if operation == "next-game" else {}
        return match_call(matches.command, identifier, operation, request.revision, request.requestId, **arguments)

    @app.post("/api/matches/{identifier}/bookmarks", status_code=201)
    def bookmark_match(identifier: str, request: BookmarkRequest):
        if not request.title.strip(): raise HTTPException(422, "Bookmark title cannot be blank")
        return match_call(matches.bookmark, identifier, request.title.strip())

    @app.post("/api/matches/{identifier}/replays")
    def publish_match(identifier: str):
        return match_call(matches.publish, identifier)

    @app.post("/api/matches/{identifier}/analyze")
    def analyze_match(identifier: str):
        protect_benchmark()
        record = match_call(matches.get_private, identifier)
        return analyze_observation(record["observation"], hypothesis_decks(), Path(record["policy"]) if record["policy"] != "heuristic" else None)

    @app.get("/api/teaching/curriculum")
    def get_curriculum():
        protect_benchmark()
        return {"families": teaching.curriculum(settings.root)}

    @app.get("/api/teaching/queue")
    def teaching_queue():
        protect_benchmark()
        return {"items": teaching.queue(store)}

    @app.get("/api/teaching/{identifier}")
    def get_teaching(identifier: str):
        protect_benchmark()
        return load("teaching", identifier)

    @app.post("/api/teaching/{identifier}/review")
    def review_teaching(identifier: str, request: TeachingReviewRequest):
        protect_benchmark()
        return match_call(teaching.review, store, identifier, review_status=request.reviewStatus,
                          acceptable_action_ids=request.acceptableActionIds, rejected_action_ids=request.rejectedActionIds,
                          reasoning=request.conditionalReasoning, critical_resources=request.criticalResources, confidence=request.confidence)

    @app.post("/api/teaching", status_code=201)
    def create_teaching(request: TeachingPositionRequest):
        protect_benchmark()
        return match_call(teaching.from_position, store, settings.root, request.positionId, request.familyId)

    @app.get("/api/guides")
    def list_guides():
        protect_benchmark()
        return {"guides": [{key: value for key, value in guide.items() if key != "chunks"}
                           for guide in store.list("guides")]}

    @app.post("/api/guides/retrieve")
    def retrieve_guides(request: GuideQueryRequest):
        protect_benchmark()
        return {"passages": match_call(guides.retrieve, store, request.query, limit=request.limit)}

    def load(category: str, identifier: str) -> dict:
        try:
            return store.get(category, identifier)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, "Record not found") from exc

    @app.get("/api/health")
    def health():
        return {"status": "ok", "engineBuilt": settings.worker.exists(), "version": "0.1.0", "workers": settings.workers,
                "limits": {"maxDecisions": settings.max_decisions, "maxJobs": settings.max_jobs,
                           "maxDiskBytes": settings.max_disk_bytes},
                "warnings": ["Private research data is stored locally; serve on loopback only."]}

    @app.get("/api/decks")
    def get_decks():
        return {"decks": decks()}

    @app.get("/api/models")
    def get_models():
        from .model_registry import list_models
        return {"models": list_models(store)}

    def run_job(identifier: str, request: GameRequest, model_path: Path | None = None) -> None:
        job = {"id": identifier, "status": "running", "configuration": request.model_dump(), "frameCursor": -1,
               "policyContext": {"policy": request.policy, "modelVersion": file_digest(model_path) if model_path else f"{request.policy}-v1",
                                 "trainingStatus": "frozen-checkpoint" if model_path else "handwritten-baseline", "learnsDuringRun": False}}
        job["policyContext"].update(opponentPopulation="Selected experimental deck pair", computeBudget="One direct policy decision; no search")
        try:
            store.put("jobs", identifier, job)
            with pool.lease() as engine:
                from .resources import ResourceGuard
                from .selfplay import Agent
                agent = Agent(str(model_path), request.seed) if model_path else None
                engine.request("reset", {"seed": request.seed, "decks": request.decks})
                prior_action, prior_actor = None, 0
                with ResourceGuard(settings) as guard:
                    for index in range(request.maxDecisions + 1):
                        guard.check()
                        observations = [engine.request("observe", {"playerId": player}) for player in (0, 1)]
                        actor = observations[0]["decisionPlayer"]
                        job["frameCursor"] = streams.append(identifier, {
                            "decisionIndex": index, "actor": prior_actor if prior_action else actor,
                            "priorAction": prior_action, "observations": observations}, job["frameCursor"])
                        job["progress"] = index
                        store.put("jobs", identifier, job)
                        if observations[0]["status"] != "running" or index == request.maxDecisions:
                            break
                        if agent:
                            action_id = agent.choose(observations[actor])
                            action = next(a for a in observations[actor]["legalActions"] if a["id"] == action_id)
                        else:
                            action = engine.request("choose", {"policy": request.policy, "seed": (request.seed ^ 0x13579bdf) + index & 0xffffffff})
                        engine.request("step", {"actionId": action["id"]})
                        prior_action, prior_actor = action, actor
                replay = engine.request("replay")
            # Interactive research runs may include lists still under audit.
            # They are explicit QA data, never silently admitted to training.
            replay["trainingEligible"] = False
            replay["dataPurpose"] = "interactive-qa"
            replay["policyContext"] = job["policyContext"]
            replay_id = store.save_replay(replay)
            store.put("jobs", identifier, {**job, "status": "completed", "replayId": replay_id,
                                          "resultStatus": replay["status"]})
        except Exception as exc:
            try:
                store.put("jobs", identifier, {**job, "status": "failed", "error": str(exc)[:2000]})
            except Exception:
                # Disk exhaustion cannot create a false completed job.
                pass
        finally:
            capacity.release()

    @app.post("/api/games", status_code=202)
    def create_game(request: GameRequest):
        protect_benchmark()
        if request.maxDecisions > settings.max_decisions:
            raise HTTPException(422, "Decision limit exceeds this machine's configured limit")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "The bounded local job queue is full")
        identifier = uuid.uuid4().hex
        try:
            registry = decks()
            known = {deck["id"] for deck in registry}
            if any(deck not in known for deck in request.decks):
                raise HTTPException(422, "Unknown deck id")
            if not request.laboratory and any(d["id"] in request.decks and d.get("role") in {"heldout", "historical"} for d in registry):
                raise HTTPException(422, "Reserved and historical lists require explicit laboratory mode")
            model_path = None
            if request.modelId:
                from .model_registry import resolve_model
                from .checkpoint_files import freeze_checkpoint
                model_path = match_call(resolve_model, store, request.modelId)
                model_path = freeze_checkpoint(store, model_path)
            store.put("jobs", identifier, {"id": identifier, "status": "queued", "configuration": request.model_dump()})
            executor.submit(run_job, identifier, request, model_path)
        except Exception:
            capacity.release()
            raise
        return {"id": identifier, "status": "queued"}

    @app.get("/api/jobs/{identifier}")
    def get_job(identifier: str):
        protect_benchmark()
        return load("jobs", identifier)

    @app.get("/api/jobs/{identifier}/frames")
    def job_frames(identifier: str, playerId: int = Query(0, ge=0, le=1), after: int = Query(-1, ge=-1), limit: int = Query(100, ge=1, le=100)):
        protect_benchmark()
        job = load("jobs", identifier)
        result = match_call(streams.read, identifier, after=after, limit=limit, committed=job.get("frameCursor", -1),
                            player_id=playerId, status=job["status"])
        if job.get("replayId"):
            result["replayId"] = job["replayId"]
        if job["status"] == "failed":
            # Engine diagnostics can mention concealed cards; the frame feed
            # exposes only a public interruption message.
            result["error"] = "Simulation stopped before completion. Accepted positions remain available."
        result["policyContext"] = job.get("policyContext")
        return result

    @app.get("/api/replays")
    def get_replays():
        protect_benchmark()
        return {"replays": [{**item, "decks": [deck if index == 0 else "opponent" for index, deck in enumerate(item.get("decks", []))]}
                            for item in store.list("replay-index")]}

    @app.get("/api/replays/{identifier}")
    def get_replay(identifier: str, playerId: int = Query(0, ge=0, le=1)):
        protect_benchmark()
        return project_replay(load("replays", identifier), playerId)

    @app.post("/api/analyze")
    def analyze(request: AnalysisRequest):
        protect_benchmark()
        try:
            if request.positionId:
                observation = load("positions", request.positionId)["observation"]
                played_action_id = None
            else:
                replay = load("replays", request.replayId)
                observation = select_frame(replay, request.decisionIndex, request.playerId)
                played_action_id = selected_action_id(replay, request.decisionIndex, request.playerId)
            # A champion pointer is only installed after a conclusive evaluation.
            model_path = settings.analysis_model
            champion = settings.data / "models" / "champion.pt"
            if model_path is not None and not model_path.exists():
                raise ValueError("PTCG_ANALYSIS_MODEL does not point to an existing local checkpoint")
            if model_path is None and request.replayId:
                context = replay.get("policyContext", {})
                fingerprint = context.get("modelVersion", "")
                if context.get("policy") == "model" and len(fingerprint) == 64 and all(c in "0123456789abcdef" for c in fingerprint):
                    frozen = settings.data / "private-models" / f"{fingerprint}.pt"
                    if frozen.exists():
                        if file_digest(frozen) != fingerprint:
                            raise ValueError("Replay checkpoint checksum differs from its frozen policy")
                        model_path = frozen
            if model_path is None and champion.exists():
                model_path = champion
            result = analyze_observation(observation, hypothesis_decks(), model_path)
            if settings.analysis_model is not None:
                result["warnings"].append("Explicit experimental analysis checkpoint; training does not establish superior playing strength or champion status.")
            try:
                params = {"observation": observation, "budgetMs": request.budgetMs, "method": "rollout", "seed": 42}
                if model_path:
                    from .training import export_portable_value, load_model, predict
                    loaded = load_model(model_path)
                    _, scores = predict(model_path, observation, loaded)
                    legal = observation.get("legalActions", [])
                    if legal:
                        if len(scores) != len(legal) or not all(math.isfinite(score) for score in scores):
                            raise ValueError("Analysis policy produced invalid action preferences")
                        weights = [math.exp(score - max(scores)) for score in scores]
                        total = sum(weights)
                        params.update(method="ismcts", rootPriors=[{"actionId": action["id"], "probability": weight / total}
                                                                 for action, weight in zip(legal, weights)])
                    leaf_model = export_portable_value(model_path, loaded)
                    if leaf_model:
                        params["leafModel"] = leaf_model
                with pool.lease(wait_timeout=.1) as engine:
                    search = engine.request("search", params)
                result["search"] = {key: search.get(key) for key in ("status", "method", "iterations", "elapsedMs")}
                result["search"]["valueContext"] = search.get("valueContext")
                result["warnings"].extend(search.get("warnings", []))
                if search.get("status") == "complete" and search.get("alternatives"):
                    result["alternatives"] = search["alternatives"]
                    result["warnings"] = [warning for warning in result["warnings"]
                                          if not warning.startswith("Search and calibrated mistake grading are unavailable")]
                    result["warnings"].append("Sampled search estimates are separate from the resource evaluation; calibrated mistake grading remains unavailable.")
            except EngineError as exc:
                result["search"] = {"status": "unavailable", "method": "rollout", "iterations": 0}
                result["warnings"].append(f"Search unavailable: {exc}")
            result["warnings"] = list(dict.fromkeys(result["warnings"]))
            result["decisionReview"] = review_decision(result["alternatives"], played_action_id)
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(503, "Install the training extra to load the selected model") from exc

    @app.post("/api/positions", status_code=201)
    def save_position(request: PositionRequest):
        protect_benchmark()
        replay = load("replays", request.replayId)
        try:
            observation = select_frame(replay, request.decisionIndex, request.playerId)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Deliberate allowlist: never save the full frame, chance record, future
        # frames, true opponent deck manifest, or opposite player's private view.
        allowed = {"schemaVersion", "playerId", "decisionPlayer", "turn", "phase", "status", "players", "ownDeck",
                   "legalActions", "history", "prompt", "warnings", "searchPosition", "searchUnavailableReason", "stadium", "knowledge"}
        position = {"id": uuid.uuid4().hex, "schemaVersion": 1, "title": request.title.strip(),
                    "createdAt": datetime.now(timezone.utc).isoformat(), "playerId": request.playerId,
                    "decisionIndex": request.decisionIndex, "sourceReplayId": request.replayId,
                    "engineVersion": replay["engineVersion"],
                    "observation": {key: value for key, value in observation.items() if key in allowed}}
        if not position["title"]:
            raise HTTPException(422, "Position title cannot contain only whitespace")
        store.put("positions", position["id"], position)
        return {key: value for key, value in position.items() if key != "observation"}

    @app.get("/api/positions")
    def list_positions():
        protect_benchmark()
        return {"positions": [{key: value for key, value in position.items() if key != "observation"}
                              for position in store.list("positions")]}

    @app.get("/api/positions/{identifier}")
    def get_position(identifier: str):
        protect_benchmark()
        return load("positions", identifier)

    @app.get("/api/experiments")
    def get_experiments():
        protect_benchmark()
        return {"experiments": store.list("experiments")}

    return app


app = create_app()
