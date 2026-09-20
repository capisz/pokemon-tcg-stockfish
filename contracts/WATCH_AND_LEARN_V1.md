# Watch and learn integration contract

This additive contract applies to the competitive integration branch. Existing private research replays remain schema 1 on disk. They are never sent directly to a browser.

## Player-projected presentation

`GET /api/replays/{id}?playerId=0` returns schema 2 with one `observation` per frame, selected `playerId`, and no private randomness or second observation. `action` describes the move from that frame; `priorAction` describes the move into it. Opponent private choices are redacted server-side. Analysis continues to read private records on the server.

`GET /api/jobs/{id}/frames?playerId=0&after=-1&limit=100` and `GET /api/matches/{id}/frames?after=-1&limit=100` return `{schemaVersion:1,frames,nextCursor,hasMore,status,replayId?}`. Limits are 1..100. `after` is the last consumed cursor, initially -1. Frames are immutable, monotonically numbered snapshots: `{cursor,decisionIndex,actor,priorAction,observation,revision?,gameNumber?}`. `priorAction` produced the snapshot; it is null initially and for controls without a game action. Match perspective is always the human player. A durable journal acknowledgement bounds visible cursors; an unacknowledged tail is never returned. Polling does not advance a game. A one-second client poll is independent of playback speed.

Match responses add `frameCursor`. All accepted match revisions publish frames, including pause/resume, so historical playback cannot submit against the live revision. Restart can resume the same stream. Legacy matches start a stream at their first new acknowledged position.

Legal actions add optional `sourceRef` and `targetRef`: `{playerId,zone,index?}`, with zone `hand|active|bench|discard|prompt`. References are valid only at their observation/revision and come from actual simulator bindings. Browser controls never derive legality from display labels. Pokémon add `attachments?:CardView[]`; legacy Energy/Tool names remain supported. Choice operation and selection metadata pass through to browser types.

## Policy and teaching

`GET /api/models` lists locally admitted checkpoint metadata; clients supply their own heuristic/random baseline options. `POST /api/games` accepts `policy:heuristic|random|model`, optional `modelId`, and `laboratory` (required for reserved/historical lists). `POST /api/matches` accepts optional `modelId`; `heuristic` explicitly selects the baseline. Checkpoints are resolved by opaque ID, verified, and frozen for a run. Policy-only demonstrations do not establish a value estimate.

All interactive runs remain QA data. Trusted outcome training, audited demonstrations, and supported search targets have separate admission checks. Human review remains pending until actually supplied. Existing family partitions and held-out list exclusions are preserved.
