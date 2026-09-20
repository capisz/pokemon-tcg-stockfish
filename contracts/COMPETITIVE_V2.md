# Competitive extension contract (2026-09-17)

The v1 JSON-lines transport and existing REST endpoints remain compatible. Optional observation additions use v1 until a breaking change is required. All artifacts carry their own schema version. The lead owns this contract; implementation owners must communicate additions before depending on them.

## Engine and list registry

Decks retain `id`, `name`, `archetype`, `formatDate`, `cards`, and `validation`. Add `role: main|training-variant|heldout|historical`, `source`, and immutable `listHash`. Heldout lists are available to the experiment runner, not the default agent hypothesis registry or training pool. Unverified legality or unresolved rules coverage must be explicit and exclude trusted training.

`reset({seed,decks,firstPlayer?})`, `observe({playerId?})`, `step({actionId})`, `replay`, and `search` are the integration primitives. `firstPlayer` is optional 0/1 for controlled experiments; omit for normal setup. Legal actions remain opaque, decision-local IDs. Staged choices may introduce `type: choice` actions; choosing a stage must preserve all legal completions, require no hidden information, and be reproducible by the action journal. No silent action cap is allowed.

Player observations may add Stadium/card details and player knowledge. Known order/reveals must be honored in search or explicitly cause `searchUnavailableReason`; never reconstruct them from true hidden state. Search only consumes player information. Historical public BO3 observations may inform list beliefs, never hidden hands/order from prior private replays.

## Live matches (API / UI owner)

`POST /api/matches` accepts `{deckId, opponentArchetype, mode: practice|benchmark, knownList: false, budgetMs: 120000}`. The server chooses the opponent list privately. Optional known-list lab mode is explicit. Return only the selected human observation and match summary, never seeds, opponent manifest IDs, private action journals, or full research replays while a match is active.

`GET /api/matches` returns redacted summaries. `GET /api/matches/{id}` returns `{id,schemaVersion:1,revision,mode,status,gameNumber,score:[human,engine],observation,engineTurnBudgetMs,thinking,error?,warnings}`. Current `revision` and unique `requestId` are required by `POST /api/matches/{id}/actions {actionId,revision,requestId}`. Exact retries are idempotent; stale or conflicting requests get 409. Engine work may run in a background task; a poll must not apply another move. Provide explicit advance, concede, pause/resume and bookmark operations as needed, documented by the owner.

Durably journal accepted decisions and private seed/config before acknowledging. Reconstruct with reset + accepted actions under the identical engine fingerprint; reject mismatched builds. Worker failures pause a session without assigning a result. Shared 120s engine turn budget includes prompts and cannot reset per action. Benchmark model hash is frozen; hints/analysis/takebacks are blocked while the benchmark is active. Completed match replay access is explicit.

## Teaching and analysis

Teaching records contain `schemaVersion`, `id`, `familyId`, `partition: train|validation|test`, source author/title/hash and page/timestamp, deck/format versions, a real position/fixture reference, player perspective, acceptable action IDs or tested conditions, conditional reasoning, `reviewStatus: draft|reviewed|rejected`, and rules-validation state. Only reviewed legal concrete positions may supply demonstrations. Prose curricula and LLM notes remain draft/ineligible.

Weekly review returns no more than ten items. Split whole fixture families; never infer preferred actions from future hidden information. Keep source files/private chunks in ignored local data. Guide metadata may be committed; source passages may not.

Analysis retains existing resource score, W/D/L, alternatives and review fields. Add source/model/budget context when available. Missing calibration, comparisons, or luck decomposition remain null/unavailable. Search-cutoff values are not terminal outcome probabilities. Live benchmark routes must not bypass analysis restrictions via saved-position or replay endpoints.

## Research execution (learning owner)

Store/EngineClient/EnginePool remain service boundaries consumed by the API owner; preserve constructor compatibility. API owner exclusively owns `api.py` and new live-session modules. Learning owner owns configuration, storage, transport, CLI, self-play, datasets/model/training/evaluation and portable bundles.

Bundle manifest v1 records relative paths/checksums, engine/deck/feature/model hashes, immutable partitions, run provenance and parent checkpoint. Import rejects traversal, incompatible identities, corruption and split conflicts before promotion; staged export/import is atomic. Trackers remain local, not copied live. Cross-device continuation receives a new linked run ID.

Profiles: Mac starts with 2 workers, an 8 GiB aggregate memory target and 25 GiB data budget; Windows starts with 2 workers, a 40 GiB aggregate memory target and 200 GiB data budget. Preserve 20 GiB free per destination. Desktop tuning limit is `max(1,min(8,logicalCPUs-2))`; increase only with measured benefit. CPU is the Windows baseline; Mac benchmarks CPU/MPS. Jobs are manually launched and resumable, with no scheduler. Quota, drive, or process failures pause work rather than invent results.

## Implemented additions (September 19 integration)

- Deck construction diagnostics are under `support: {playable, errors, trainingEligible}`. `decks` returns all fifteen competitive manifests; `getAgentDecks()` excludes heldout/historical lists. Eligibility is explicit and currently false for every competitive list.
- Search accepts `knownOpponentDeckId`, `priorRevealedCards: string[]` (presence across games, never summed copy counts), and normalized `rootPriors: {actionId,probability}[]`. It returns `hypothesisWeights: {id,archetype,kind,weight}[]`, with kinds `listed`, `unknown-variant`, or `known-list`. Only laboratory mode may supply an exact heldout list.
- `CardView` optionally adds `text`, `imageUrl`, powers and attack text. Observations optionally add `stadium` and observer-only `knowledge`. Temporary-zone peeks and other unsupported histories cause an explicit `searchUnavailableReason`. They never authorize inspection of actual hidden state.
- `POST /api/matches/{id}/advance` schedules engine work; polling is read-only. `POST /api/matches/{id}/control/{operation}` accepts revision/requestId and operations `pause`, `resume`, `concede`, `next-game`, `abandon`; next-game may include `firstPlayer`. Abandonment preserves an incomplete private journal without a winner and releases benchmark locks.
- Live responses include `engineTurnRemainingMs`, `modelVersion`, `knownList`, `ownDeckId`, `matchKnowledge`, and next-starter information. `opponentList` appears only in known-list mode. Status is `active`, `paused`, `between-games`, `completed`, or `abandoned`.
- `POST /api/matches/{id}/bookmarks` saves a selected-perspective teaching draft. `POST /api/matches/{id}/analyze` is practice-only. `POST /api/matches/{id}/replays` publishes private research replays only after completion. Human matches remain excluded from automatic training, even after annotation.
- Teaching endpoints are `GET /api/teaching/curriculum`, `GET /api/teaching/queue`, `GET /api/teaching/{id}`, `POST /api/teaching`, and `POST /api/teaching/{id}/review`. The server fixes family partitions, validates action IDs and the original position hash, and controls rules-audit eligibility. The twenty prose families do not count as concrete demonstrations.
- Replay training admission requires literal `trainingEligible: true` and two supported training roles. Family quarantine records persist in `partitions/` and travel with bundles. Trusted checkpoints carry verified-corpus metadata and cumulative training-seed/family lineage. Legacy smoke artifacts without those records remain historical.
- Evaluation snapshots candidate/opponent weights under managed storage and verifies them through play and promotion. Promotion requires coverage of the complete registered competitive pool, balanced starts, untouched ancestral seeds, statistical evidence and regression review. A partial eligible registry cannot promote a candidate.

Resource accounting currently covers the launching Python process tree and managed data directory. Separate processes and external model caches remain outside this monitored scope; this is a documented implementation gap, not an aggregate machine quota.
