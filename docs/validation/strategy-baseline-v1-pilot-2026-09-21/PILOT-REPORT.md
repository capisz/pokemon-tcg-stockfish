# Strategy Baseline v1 pilot

Status: completed, measurement only.

Engine fingerprint: `twinleaf-adapter-0.1.0+bf3a9a0f133b99bd`  
Engine build SHA-256: `474549dda1d98fea5a7435c3abfdcccb4baa2663237ce8bb4aab99c40b36906e`

The pilot timed one game in each of the eight matchup, deck-to-seat and first-player cells. P1, P2 and P4 used the same seed for each corresponding cell. All 24 executable games reached a rules-terminal result; none was truncated and none reported an engine error.

P3 was dropped as required. Its frozen checkpoint exists at `/Users/admin/Documents/ChatGPT/Pokemon Ai project/data/competitive/experimental/models/09572b53a4e74d9193c101bab05b30da-guide.pt` with SHA-256 `1ff34f893540373ebb2da6433b3b67a0cdbcb2b711220cf08767cdeb85a08e6f`, but the approved environment cannot load it because the optional local PyTorch dependency is absent. No replacement checkpoint or dependency installation was used.

| Policy | Pilot games | Finished | Mean seconds/game | Projected 96-game time |
|---|---:|---:|---:|---:|
| P1 — TypeScript heuristic | 8 | 8 | 5.86 | 9.05 min |
| P2 — Python heuristic | 8 | 8 | 6.88 | 11.56 min |
| P4 — ISMCTS 200 ms | 8 | 8 | 10.00 | 17.63 min |

The supported 288-game plan projects to 38.25 serial-equivalent minutes, below the 60-minute reduction threshold. This projection applies the actual plan weights: eight games for each cross-matchup cell and sixteen for each mirror cell. The recommendation is therefore to retain the original per-cell counts for P1, P2 and P4. The original 384-game plan is not executable until P3 can load; no runtime was imputed for it.

P4 used the unchanged Python `heuristic_action_score` fallback. Across 2,079 decisions, 93 search calls were attempted, 80 decisions selected a measured search action, and 1,999 used fallback. That is a 3.85% searched-decision fraction and a 4.47% search-attempt fraction. The search calls completed 82 total iterations. These figures show search availability and coverage, not playing strength.

Pilot cross-matchup outcomes were: P1 Crustle 3–1 Dragapult; P2 Crustle 1–3 Dragapult; P4 Crustle 1–3 Dragapult. Mirror seat results and individual game records are in `pilot-results.json`. With only one game per cell, none of these outcomes is a strength estimate or a basis for changing strategy.

No production policy, search, gameplay behavior, feature, model, training data, teaching label, or file under `data/competitive` was changed.
