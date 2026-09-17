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
