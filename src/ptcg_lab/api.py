from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from .analysis import analyze_observation, review_decision, select_frame, selected_action_id
from .config import Settings
from .engine import EngineError, EnginePool
from .storage import Store


class GameRequest(BaseModel):
    decks: tuple[str, str]
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    maxDecisions: int = Field(default=1000, ge=1, le=3000)
    policy: str = Field(default="heuristic", pattern="^(random|heuristic)$")


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
    store = Store(settings.data, settings.max_disk_bytes)
    pool = EnginePool(settings.root, settings.workers, settings.engine_timeout)
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
        yield
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
                "warnings": ["Research replays contain both players' private views; serve on loopback only."]}

    @app.get("/api/decks")
    def get_decks():
        return {"decks": decks()}

    def run_job(identifier: str, request: GameRequest) -> None:
        job = {"id": identifier, "status": "running", "configuration": request.model_dump()}
        try:
            store.put("jobs", identifier, job)
            with pool.lease() as engine:
                replay = engine.request("run", request.model_dump())
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
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "The bounded local job queue is full")
        identifier = uuid.uuid4().hex
        try:
            known = {deck["id"] for deck in decks()}
            if any(deck not in known for deck in request.decks):
                raise HTTPException(422, "Unknown deck id")
            store.put("jobs", identifier, {"id": identifier, "status": "queued", "configuration": request.model_dump()})
            executor.submit(run_job, identifier, request)
        except Exception:
            capacity.release()
            raise
        return {"id": identifier, "status": "queued"}

    @app.get("/api/jobs/{identifier}")
    def get_job(identifier: str):
        return load("jobs", identifier)

    @app.get("/api/replays")
    def get_replays():
        return {"replays": store.list("replay-index")}

    @app.get("/api/replays/{identifier}")
    def get_replay(identifier: str):
        return load("replays", identifier)

    @app.post("/api/analyze")
    def analyze(request: AnalysisRequest):
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
            if model_path is None and champion.exists():
                model_path = champion
            result = analyze_observation(observation, decks(), model_path)
            if settings.analysis_model is not None:
                result["warnings"].append("Explicit experimental analysis checkpoint; training does not establish superior playing strength or champion status.")
            try:
                with pool.lease(wait_timeout=.1) as engine:
                    search = engine.request("search", {"observation": observation, "budgetMs": request.budgetMs,
                                                       "method": "rollout", "seed": 42})
                result["search"] = {key: search.get(key) for key in ("status", "method", "iterations", "elapsedMs")}
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
        replay = load("replays", request.replayId)
        try:
            observation = select_frame(replay, request.decisionIndex, request.playerId)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Deliberate allowlist: never save the full frame, chance record, future
        # frames, true opponent deck manifest, or opposite player's private view.
        allowed = {"schemaVersion", "playerId", "decisionPlayer", "turn", "phase", "status", "players", "ownDeck",
                   "legalActions", "history", "prompt", "warnings", "searchPosition", "searchUnavailableReason"}
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
        return {"positions": [{key: value for key, value in position.items() if key != "observation"}
                              for position in store.list("positions")]}

    @app.get("/api/positions/{identifier}")
    def get_position(identifier: str):
        return load("positions", identifier)

    @app.get("/api/experiments")
    def get_experiments():
        return {"experiments": store.list("experiments")}

    return app


app = create_app()
