"""Bounded real API integration: observation-only proxy operates the human seat.

Use an isolated API data directory. This measures product execution, not human
playing strength. No forced outcomes, concessions, or privileged observations.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import httpx
from ptcg_lab.selfplay import Agent


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8766")
parser.add_argument("--seconds", type=int, default=240)
parser.add_argument("--output", type=Path, default=Path("artifacts/natural-match.json"))
args = parser.parse_args()
client = httpx.Client(base_url=args.url, timeout=30)


def get(path):
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def post(path, body=None):
    response = client.post(path, json=body or {})
    response.raise_for_status()
    return response.json()


if any(m["status"] not in {"completed", "abandoned"} for m in get("/api/matches")["matches"]):
    raise SystemExit("Use an isolated server with no ongoing match")
state = post("/api/matches", {"deckId": "crustle", "opponentArchetype": "mega-lucario", "budgetMs": 1, "modelId": "heuristic"})
identifier = state["id"]
agent = Agent("heuristic", 914)
started = time.monotonic()
report = {"matchId": identifier, "purpose": "product-integration-only", "engineTurnBudgetMs": 1,
          "humanSeat": "observation-only heuristic proxy", "naturalOutcome": None, "nextGameVerified": False}
last_report = 0
try:
    while time.monotonic() - started < args.seconds:
        if state["status"] != "active": break
        if state["thinking"]:
            time.sleep(.05)
            state = get(f"/api/matches/{identifier}")
            continue
        if state["observation"]["decisionPlayer"] == 0:
            action_id = agent.choose(state["observation"])
            state = post(f"/api/matches/{identifier}/actions", {"revision": state["revision"], "requestId": uuid.uuid4().hex, "actionId": action_id})
        else:
            state = post(f"/api/matches/{identifier}/advance")
        elapsed = time.monotonic() - started
        if elapsed - last_report >= 30:
            print(json.dumps({"seconds": round(elapsed, 1), "revision": state["revision"], "turn": state["observation"]["turn"]}), flush=True)
            last_report = elapsed
    if state["status"] == "between-games":
        report["naturalOutcome"] = state["gameResult"]
        report["score"] = state["score"]
        state = post(f"/api/matches/{identifier}/control/next-game", {
            "revision": state["revision"], "requestId": uuid.uuid4().hex,
            "firstPlayer": 0 if state["nextStarterChooser"] == 0 else None})
        report["nextGameVerified"] = state["gameNumber"] == 2 and state["status"] == "active"
finally:
    # Keep the journal; an interrupted probe never manufactures a match result.
    state = get(f"/api/matches/{identifier}")
    while state["thinking"]:
        time.sleep(.1)
        state = get(f"/api/matches/{identifier}")
    if state["status"] == "active":
        state = post(f"/api/matches/{identifier}/control/pause", {"revision": state["revision"], "requestId": uuid.uuid4().hex})
    if state["status"] in {"paused", "between-games"}:
        state = post(f"/api/matches/{identifier}/control/abandon", {"revision": state["revision"], "requestId": uuid.uuid4().hex})
    report.update(seconds=round(time.monotonic()-started, 3), finalStatus=state["status"], revision=state["revision"], error=state.get("error"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    client.close()
if not report["naturalOutcome"] or not report["nextGameVerified"]:
    raise SystemExit(1)
