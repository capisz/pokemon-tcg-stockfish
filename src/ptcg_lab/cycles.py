"""Manually launched bounded research cycles. No daemon or scheduled startup."""
from __future__ import annotations

import uuid
from pathlib import Path

from .config import Settings
from .resources import ResourceGuard, ResourceLimit
from .selfplay import population, selfplay
from .storage import Store, file_digest


def continuous(settings: Settings, *, batches: int = 1, games: int = 50, seed: int = 42,
               max_decisions: int = 1000, search_budget_ms: int = 200, resume: str | None = None,
               device: str = "cpu", max_positions: int = 10000, evaluate_every: int = 1) -> dict:
    """Zero batches means until manually stopped; each batch is independently bounded."""
    if not 0 <= batches <= 100000 or not 10 <= games <= 10000 or not 1 <= evaluate_every <= 1000:
        raise ValueError("Cycles need 0..100000 batches,10..10000 games/batch, evaluation every1..1000 batches")
    if device != "cpu" and settings.profile == "windows":
        raise ValueError("Windows starts on CPU; AMD acceleration is not enabled by this prototype")
    store = Store(settings.data, settings.max_disk_bytes, settings.min_free_bytes)
    config = {"batches": batches, "games": games, "seed": seed, "maxDecisions": max_decisions,
              "searchBudgetMs": search_budget_ms, "device": device, "maxPositions": max_positions,
              "evaluateEvery": evaluate_every}
    if resume:
        state = store.get("cycles", resume)
        if state["configuration"] != config:
            raise ValueError("Repeat the original cycle configuration to resume")
    else:
        state = {"id": uuid.uuid4().hex, "configuration": config, "nextBatch": 0,
                 "status": "running", "history": [], "current": None, "candidate": "heuristic"}
    store.put("cycles", state["id"], state)
    try:
        while batches == 0 or state["nextBatch"] < batches:
            if (settings.data / f"pause-{state['id']}").exists():
                state["status"] = "paused"
                break
            with ResourceGuard(settings) as guard:
                index = state["nextBatch"]
                if state["current"] is None:
                    state["current"] = {"index": index, "opponents": population(store), "selfplay": None,
                                        "trainingId": uuid.uuid4().hex, "trained": False,
                                        "candidate": state["candidate"]}
                    store.put("cycles", state["id"], state)
                batch = state["current"]

                def link(identifier):
                    batch["selfplay"] = identifier
                    store.put("cycles", state["id"], state)

                played = selfplay(settings, games=games, seed=(seed + index * games) % 2**32,
                                  max_decisions=max_decisions, policy=batch["candidate"], opponents=batch["opponents"],
                                  resume=batch["selfplay"], search_budget_ms=search_budget_ms, on_created=link)
                if played["status"] != "completed":
                    state["status"], state["pauseReason"] = "paused", played.get("pauseReason", "Self-play paused")
                    break
                model = store.path / "models" / f"{batch['trainingId']}.pt"
                if not batch["trained"]:
                    from .training import train
                    report = train(store, epochs=1, seed=seed, device=device, max_positions=max_positions,
                                   resume=model if model.exists() else None, run_id=batch["trainingId"], guard=guard,
                                   warm_start=Path(batch["candidate"]) if not model.exists() and batch["candidate"] != "heuristic" else None)
                    batch["trained"] = True
                    batch["modelHash"] = report["modelHash"]
                    store.put("cycles", state["id"], state)
                if (index + 1) % evaluate_every == 0 and not batch.get("evaluation"):
                    from .evaluation import evaluate
                    champion = store.path / "models" / "champion.pt"
                    result = evaluate(settings, candidate=str(model), opponent=str(champion) if champion.exists() else "heuristic",
                                      seeds=1, seed_start=(1_000_000_000 + index * 10000) % 2**32,
                                      max_decisions=max_decisions, guard=guard)
                    batch["evaluation"] = result["id"]
                    # Evaluation-only automatic reports do not promote: one seed
                    # cannot meet the statistical promotion gate.
                state["candidate"] = str(model)
                state["history"].append(dict(batch))
                state["nextBatch"] += 1
                state["current"] = None
                store.put("cycles", state["id"], state)
        else:
            state["status"] = "completed"
    except (KeyboardInterrupt, ResourceLimit, RuntimeError, ValueError, OSError) as exc:
        state["status"] = "paused"
        state["pauseReason"] = str(exc) or "Interrupted by user; continue from the last checkpoint"
    try:
        store.put("cycles", state["id"], state)
    except (RuntimeError, OSError):
        state["persistenceWarning"] = "Resume the last durable manifest after freeing space"
    return state
