# Continuous experimental learning v1

The manually started service shares the API's two-worker pool. It owns persistent
run/game journals under `data/competitive/experimental`; this is not trusted
outcome-training data. No existing QA replay is relabeled or imported implicitly.

- `GET/POST /api/learning/runs`: `{runs:[...]}` / create a run. Defaults: seed 42,
  keepAwake true, 50 completed games/batch, 200 ms decision search, 20 investigated
  positions at 2000 ms, 20,000 training rows, two comparison seeds, 3000 decisions,
  maxCycles 0 (until stopped). comparisonGameLimit 0 means complete the protocol;
  a nonzero diagnostic cap deliberately stops short, cannot pass adoption, and
  retains the full expected coverage denominator. Smaller diagnostic configurations
  stay explicit.
- `GET /api/learning/runs/{id}` returns schemaVersion, id, revision, status, phase,
  cycle, public configuration, metrics, incumbent/candidate/guidePolicy metadata,
  activeGames, error/pauseReason. Revisions guard controls, not progress polling.
- `POST /api/learning/runs/{id}/control/{pause|resume|stop}` takes revision and
  requestId. Duplicate requests are idempotent; stale conflicting requests fail.
- `GET /api/learning/runs/{id}/games` returns at most 200 recent game summaries.
  Each has id, workerIndex, status, purpose, archetypes, decisionIndex, turn,
  frameCursor and replayAvailable. Exact opponent lists/seeds stay private.
- `GET /api/learning/games/{id}/frames?playerId=0&after=-1&limit=100` uses the
  existing projected feed. `GET .../{id}/replay?playerId=0` returns a completed
  projected replay with public ID `experimental-<gameId>`. Existing analysis and
  saved-position APIs resolve that namespace on the server.

Run status: running, pausing, paused, stopped, waiting-for-play, failed. Phases:
initializing, collecting, investigating, training, comparing, checkpointed.
Metrics include completedGames/errors/truncatedGames/searchDecisions/searchTargets,
peakMemoryBytes/managedBytes/freeBytes, coverage, loss, comparison, guideAgreement.
Coverage has scheduledPairs/totalPairs/completedPairs/archetypes; comparison has
status/completedGames/totalGames/mean/lower/upper/adopted/notes; guideAgreement has
status/positions/acceptableActionAccuracy. Unknown numbers are null, not zero claims.

Models and every game are frozen by bytes. Experimental ancestry is permanent and
requires explicit inference opt-in. Only experimental incumbent pointers advance;
champion.pt and trusted admission remain untouched. A completed terminal supplies
a value label; only reviewed alternatives or adequately covered searches supply
policy labels. Comparison families never enter training.

An accepted decision is journaled before its frame cursor is published. Restart
pauses and requires explicit resume. Playback pause never alters computation.
Human play and foreground analysis preempt learning at safe boundaries. The
browser can close without stopping the backend. A requested keep-awake assertion
exists only while actively computing, never as an OS startup schedule.
