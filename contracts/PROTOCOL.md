# Local engine protocol v1

Node worker reads one JSON request per stdin line: `{id, method, params}`. Replies on stdout only: `{id, result}` or `{id, error: {code, message}}`. Diagnostics go to stderr. Python owns persistent workers; no network dependency.

Methods: `health`, `decks`, `reset({seed, decks: [deckId, deckId]})`, `observe({playerId?})`, `step({actionId})`, `replay`, `run({seed, decks, policy: 'random'|'heuristic', maxDecisions})`. `reset`/`step` return `{observation, status, decisionIndex}`. `run` returns a replay. Player IDs are 0 and 1 at this boundary (engine may use 1/2 internally). `observe` defaults to current decision player. `run` limits are truncations, never fabricated terminal results.

Observation: `{schemaVersion: 1, playerId, decisionPlayer, turn, phase, status, players: PlayerView[], ownDeck: {cardId,name,count}[], legalActions: LegalAction[], history: string[], prompt?: {type,message}, warnings: string[]}`.

PlayerView: `{id,name,active: PokemonView|null,bench: PokemonView[],hand: CardView[],handCount,deckCount,prizesRemaining,discard: CardView[]}`. Opponent hand is an empty array with count only. Own prizes and both deck orders are always hidden. Public history must not expose hidden choices.

CardView: `{id,name,kind:'pokemon'|'trainer'|'energy',types?: string[],hp?:number,stage?:string,prizeValue?:number,attacks?:{name,damage,cost:string[]}[]}`. PokemonView: `{card:CardView,damage:number,energy:string[],tools:string[],conditions:string[]}`. Card IDs are printed-card identifiers, not persistent hidden-object IDs.

LegalAction: `{id,label,type,cardId?:string,target?:string}`. ID is an opaque identifier valid only at this exact decision. Candidate enumeration must report any incompleteness in warnings. Policies receive only the observation. Mutation occurs only through validated candidate actions.

Replay: `{schemaVersion:1,id,seed,decks:string[],engineVersion:string,status:'finished'|'truncated'|'error',outcome:{winner:0|1|null,reason:string}|null,frames: ReplayFrame[],warnings:string[]}`. ReplayFrame: `{decisionIndex,actor:0|1,action:LegalAction|null,observations:[Observation,Observation]}`. Each frame is BEFORE its action; include a final frame with null action. Full research replays contain both private views and are explicitly not public exports.

Deck registry JSON files live in `decks/`, schema `{id,name,archetype,formatDate,cards:[{cardId,name,count}],validation:{status,notes:[]}}`. Initial status is experimental until a complete legality and interaction audit. Engine `decks` returns registry entries with playable support diagnostics. Python/UI must preserve status and warnings, not silently promote them to validated.

Python REST contract: `GET /api/health`, `GET /api/decks` returns `{decks:[]}`, `POST /api/games` body `{decks:[id,id],seed,maxDecisions,policy}` returns `{id,status}` for a background job; `GET /api/jobs/{id}` returns `{id,status,progress?,error?,replayId?}`; `GET /api/replays` returns `{replays:[{id,decks,status,frames}]}`; `GET /api/replays/{id}` returns Replay; `POST /api/analyze` body `{replayId,decisionIndex,playerId,budgetMs?}` returns `{evaluation,alternatives,beliefs,warnings}`; `GET /api/experiments` returns `{experiments:[]}`.

Evaluation: `{status:'heuristic'|'trained'|'unavailable',score:number|null,expectedResult:number|null,winProbability:number|null,drawProbability:number|null,lossProbability:number|null,modelVersion:string,components:{name,value}[],calibrated:boolean,description:string}`. Untrained resource scores are NEVER called trained/calibrated probabilities. Alternatives: `{actionId,label,score:number|null,visits:number,description:string}[]`. Beliefs: `{archetype,probability}[]`, always labeled pool-limited.

## Research extensions in v1

- Replay includes `visibility:'private-research'` and a private chance-event log. Only `rules-draw` with a null winner is a genuine draw; an engine finish without a rules winner is an error with no outcome.
- `branch` reconstructs the current seeded action history in an independent environment. It is a research replay operation, not an agent information input.
- Observations may include `searchPosition` (the whitelisted `PublicPosition` type in `belief-state.ts`) or `searchUnavailableReason`. The projection includes only public zones/effects and the selected player's own known hand. Authorized reveal prompts can expose `cards` or `hands`; other private prompt content remains hidden.
- `search({observation,method:'rollout'|'ismcts',budgetMs,seed,iterations,maxRolloutDecisions})` returns `{status,method,iterations,elapsedMs,alternatives,warnings,hypotheses?,treeNodes?}`. No true game state or replay is accepted. An individual engine step can overrun the wall-clock budget.
- Search alternatives add `expectedResult:number|null`, `uncertainty:number|null` (sample standard error), and optional `continuation`. A continuation has `conditional:true`, `representative:true`, `description`, `opponentArchetype`, at most eight `steps:{playerId,label,type}[]`, `end:'terminal'|'cutoff'`, and optional outcome. It is one illustrative sampled line, not a forced sequence or a proven best continuation. Opponent private choices are redacted.
- `POST /api/positions` accepts `{replayId,decisionIndex,playerId,title}`. `GET /api/positions` lists positions; `GET /api/positions/{id}` returns a single selected observation and source identifiers, without future frames or the other player's private view.
- `POST /api/analyze` accepts either `{positionId,budgetMs?}` OR `{replayId,decisionIndex,playerId,budgetMs?}`. It adds `search` metadata and `decisionReview` to the base response. `decisionReview.opportunityLoss` is available only when the played action and another action received samples; it remains experimental and never uses future frames/results. Saved positions with no recorded move receive no mistake estimate. Later luck is not quantified by this field.
- The Python worker client records a SHA256 bundle hash alongside the source-fingerprinted engine version. Resume checks reject incompatible builds/configurations.
