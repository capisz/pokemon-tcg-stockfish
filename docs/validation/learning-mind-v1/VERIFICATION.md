# Learning Mind v1 verification

Date: 2026-09-22

Branch: `codex/learning-mind-v1`

Base: `7543298`

## Completed gates

- Strategy v1.2 validator: passed; v1.2 remains the active approved revision.
- Frozen representation audit: 192/192 replay hashes verified, 41,675 actor
  decisions encoded, zero unsupported positions, and every selected legal action
  represented exactly once.
- Full Python suite: 283 passed.
- Full engine suite: 67 passed.
- TypeScript typecheck: passed.
- Focused new plus baseline/pilot/decision-guard suite: 58 passed.
- Engine transport and portable feature parity retry: 11 passed. The initial
  full-suite invocation began before the isolated worktree's temporary
  dependency link was available; no source change was needed, and the final
  full Python rerun passed 283/283.
- `git diff --check`: passed.

## Supervised orchestration smoke

- The immutable builder produced four actor-visible policy-labelled positions:
  three train, one development, and zero held-out. They comprise two unique
  compatible review positions and two exact search distributions. Ordinary
  self-play contributed zero policy labels; all four stale reviews remain
  excluded.
- The resumable macro collector completed a bounded one-position root-action
  proxy smoke with common random numbers. This is not yet a high-confidence
  turn-plan label and was not added to Transformer policy loss.
- XGBoost fit successfully, but its only query contained three candidates. Its
  pairwise ordering accuracy was 0.0 and development, archetype, and policy
  family holdouts were all insufficient.
- A one-epoch 1,585,714-parameter Transformer smoke checkpoint trained from
  three approved examples. On the sole development position it tied the
  heuristic at 1/1; there were no held-out positions and no strategy-probe
  result. The supervised acceptance gate did not pass.
- Current validation after the orchestration changes: 289 Python tests and 67
  engine tests passed; TypeScript typechecking and `git diff --check` passed.
- Compact hashes and results are frozen in
  `supervised-smoke-2026-09-22.json`; raw rows, labels, models, and checkpoints
  remain ignored under `artifacts/learning-mind-v1/`.

## Deliberately closed gates

- The smoke dataset has no held-out row, broad archetype coverage, or blind
  opponent-policy-family evidence. The macro smoke does not enforce complete
  turn plans. More approved labels must be collected before a serious fit.
- The required held-out-label and targeted-probe wins are not established.
  PPO, specialists, continuous running, and promotion remain disabled.
- The launchd file is an uninstalled example. A process initialized from it
  persists `PAUSED`; it cannot start training without explicit human enablement.

## Raging Bolt evidence expansion

- A deterministic actor-visible pool now contains 18 unlabeled Raging Bolt
  positions: 12 train, three development, and three held-out, spanning all five
  opponent archetypes and two frozen opponent-policy families.
- A six-position, 2-to-4 rollout smoke completed and trained a ranker over 35
  training candidates. The single development position had 0.0415 top-1
  relative regret, zero top-3 recall, and 0.554 pairwise accuracy.
- Archetype and policy-family holdout machinery produced measured results, but
  the sample is intentionally too small and uses root-action proxy semantics.
  It is not a high-confidence macro-plan label and changes no promotion gate.
- Frozen hashes are recorded in `raging-bolt-macro-smoke-2026-09-22.json`.

## Executable Raging Bolt macro evidence

- The 18-position frozen pool completed at the default 16-to-64 rollout
  allocation: 174 candidate plans, 10,000 completed rollouts, 80 typed
  execution failures, and zero truncations. All 80 failures came from
  multi-action plans; their outcomes were excluded from ranker labels.
- The ranker used 12 training positions and 98 candidates. On three development
  positions it measured 0.0020 mean top-1 relative regret, 0.667 top-3 recall,
  and 0.712 pairwise ordering accuracy. This sample is too small to establish
  generalization.
- Frozen policy-family folds were weaker: held-out regret was 0.032 / 0.021,
  top-3 recall 0.167 / 0.500, and pairwise accuracy 0.404 / 0.575. The fitter
  did not report a primary held-out split score, and the result is explicitly
  not evidence of playing-strength improvement.
- A subsequent runtime audit found that the earlier transition-macro planner
  ran under TSX/ESM while search ran from the CJS engine worker. A same-position
  comparison reproduced 56/81 non-executable plans in that mixed runtime; the
  CJS planner/search smoke generated 36 plans and executed all 36. All 36 then
  hit the 80-decision horizon, yielding zero scored candidates. Treat the
  earlier executable-plan failure count as a runtime mismatch, not a strategy
  metric; re-run the historical 18-position collection under v5 before relying
  on its aggregate failure count. See
  `macro-planner-cjs-parity-v5-2026-09-22.json`.
- A single 300-decision rollout under the pinned public hypothesis reached a
  rules terminal after the v5 parity smoke. This establishes that a longer
  fixed research horizon can produce a terminal sample on this position only;
  it is not a candidate comparison, label set, or strength result.
- The executable sequence harness improves on root-action proxy semantics, but
  still uses heuristic prompt resolution and has incomplete multi-action
  execution. Fix that coverage, expand the frozen pool, then rerun before
  considering strategy-probe evaluation. PPO and continuous operation remain
  disabled.
- Compact results and all frozen identities are recorded in
  `executable-raging-bolt-full-2026-09-22.json`; raw rollout artifacts and the
  model remain ignored under `artifacts/learning-mind-v1/`.

## Candidate-generation safety correction

- The full-run artifacts above were generated before the candidate-generator
  correction. Inspection showed that combinations of root legal-action IDs do
  not prove those actions remain legal in sequence. The old multi-action plans
  were therefore not valid turn-plan candidates, even though typed failures
  were excluded from their labels.
- The generator now emits exactly one candidate per currently legal root
  action, rejects cap overflow, and records a version in the resumable collector
  settings. Its semantics explicitly say these are not complete turn plans.
- Focused generator and collector tests pass (7 passed). A broader focused
  pytest invocation reached a native segmentation fault in the existing
  XGBoost test path; the isolated macro tests pass. No new collection or ranker
  fit has been run under generator v2, so all prior metrics remain historical
  and do not validate the corrected candidate set.
- Next: collect under a new output directory with generator v2, then implement
  a transition-aware multi-step planner before describing results as turn-plan
  learning.
- A live one-position generator-v2 smoke completed 15 root actions × two
  rollouts (30/30 complete, zero typed errors, zero truncations). Its compact
  manifest and hashes are recorded in `root-action-generator-smoke-2026-09-22.json`.
  This validates the corrected collector path only; a full fresh pool and
  transition-aware planner are still required.
- The full corrected 18-position batch is now complete: 136 single-action
  candidates and 7,984/7,984 completed rollouts, with zero typed errors and zero
  truncations. A ranker trained on 12 positions measured 0.0029 mean top-1
  regret on three development positions, but 0.0557 regret on the three
  untouched held-out positions (top-3 recall 1.0; pairwise accuracy 0.475).
  This mixed, tiny held-out result does not establish a policy or playing
  strength improvement.
- The ranker now reports the dedicated held-out split in addition to
  development and fold metrics. Full frozen identities and ranker checksums are
  in `root-action-ranker-v2-2026-09-22.json`.

## Transition-aware planner implementation

- Added a research-only planner that samples opponent hypotheses from the
  actor-visible public search position and expands ordered action prefixes by
  replaying each prefix from the same determinization. Every next action is
  re-resolved against the updated legal observation; the 128-candidate cap and
  three-action depth fail closed rather than dropping branches. The Python
  collector freezes planner/adapter hashes and records unsupported positions
  immutably.
- The planner now uses complete action bindings (including source/target refs,
  staged choice refs, and selection operation) when matching sampled actions.
  Five positions are rejected because the existing search action key aliases
  distinct downstream bindings; this is fail-closed. One supported full
  position generated nine candidates, four multi-action, through depth two.
- The end-to-end one-position collector smoke correctly recorded its first
  selected position as unsupported because full branching exceeded 128; it
  produced no fabricated labels. The updated frozen-pool audit supports 2/18
  at depth three and 16/18 at depth two (zero errors, zero rollouts): 11 depth-
  three positions overflow the cap and five hit the search-binding ambiguity.
- `npm run typecheck`, all 71 engine tests, and the seven targeted Python
  macro/orchestration tests pass. A broad Python invocation still hits a native
  XGBoost segmentation fault outside the touched tests. PPO, ranker refitting,
  policy promotion, and continuous operation remain disabled.
- Exact fixture, smoke-output, and implementation checksums are recorded in
  `transition-macro-smoke-2026-09-22.json`; the full collector artifact stays
  ignored under `artifacts/learning-mind-v1/`.
- The earlier (coarse-key) audit reported depth-three support on 5/18. Binding-
  aware revalidation reduced safe support to 2/18: 11 hit the 128-candidate cap
  and five would be ambiguous under the existing research search matcher. A
  matcher fix changes internal search behavior; it needs explicit scope
  approval before implementation. The planner remains fail-closed meanwhile.

This evidence establishes implementation and representation parity, not playing
strength or autonomous improvement.

## Bound-action search and planner v2

- Research search now keys choices by action type/card, target, label, source
  and target references, staged choice references/operation/count, and amount.
  Physical copies in hand/prompt remain interchangeable; board indices do not.
  This changed internal research search behavior only; no public API or
  production policy changed.
- The planner uses the same semantic action key and retains a runtime parity
  check. A full 18-position, no-rollout audit found depth-three support on 4/18
  and depth-two support on 16/18, with zero errors. The five prior binding
  ambiguity rejections are resolved; 14 depth-three positions still exceed the
  128-candidate cap.
- A fresh one-position collector smoke recorded over-cap status with zero
  rollouts. Full pool evidence and artifact hashes are in
  `transition-macro-bound-actions-v2-2026-09-22.json`.
- The shared-key change passed `npm run typecheck`, all 74 engine tests, and
  seven targeted Python tests. No new rollout labels, ranker fit, policy
  promotion, or PPO run was started.

## Explicit attack / no-attack intent in planner v3

- Harness commit `c54c93b` adds explicit terminal intent: attack candidates end
  in a legal attack; deliberate no-attack candidates end in legal pass. The
  adapter rejects any mismatch, and only those complete candidates may receive
  rollout labels. Incomplete prefixes remain available for enumeration but are
  excluded from labels.
- Against the same frozen 18-position pool, depth-three support is 4/18; the
  other 14 fail closed at the 128-candidate cap. Supported positions produced
  25, 87, 119, and 87 candidates including incomplete prefixes. This ran no
  rollouts and does not establish model quality. Candidate branching remains
  the immediate blocker.
- All 74 engine tests pass, `npm run typecheck` passes, and the focused Python
  macro/orchestration tests pass (16; the XGBoost native metadata test was
  excluded after its known environment segfault). Full checksums and pool
  identity are in `transition-macro-terminal-intent-v3-2026-09-22.json`.
- No new training labels or ranker fit were started. PPO, promotion, and
  continuous operation remain disabled.

## Bounded rollout cutoff accounting

- Macro collection accepts a frozen `--rollout-budget-ms` (default 1000),
  includes it in resume identity, and reports search budget cutoffs distinctly
  from decision-horizon cutoffs. Neither kind receives an outcome label.
- A current-identity Raging Bolt position generated 42 executable candidates.
  With one rollout each, a 300-decision horizon, and a 1-second per-rollout
  cap, all 42 calls completed in about 52 seconds: 42 budget truncations, zero
  horizon truncations, zero errors, and zero scored candidates. This validates
  bounded collection and no-fabricated-label handling only; it is not usable
  ranker data or strategy evidence.
- Evidence and checksums are appended to
  `macro-planner-cjs-parity-v5-2026-09-22.json`. The focused orchestration suite
  passes (7 tests), engine typecheck/build pass, and `git diff --check` passes.
- Continuation diagnostics now report simulated decision counts without adding
  reward shaping. On the same position at a 5-second cap, 30/42 candidates
  reached terminal and 12/42 remained budget-truncated (zero errors); decision
  counts ranged 81–142, median 121. Since this is only one rollout per
  candidate, it remains runtime evidence and is not used for ranker fitting.
- Opt-in two-worker execution preserved matched seed assignment and candidate
  result order. On this position, serial and parallel runs agreed on each
  candidate's terminal/truncated status and completed score; decision counts
  differed only for budget-limited continuations. Approximate wall time fell
  from 151 seconds to 98 seconds (~1.5x) in this single run. Worker count stays
  frozen per dataset and defaults to one; repeat a benchmark before scaling up.

## Complete-candidate cap accounting in planner v4

- The prior cap check incorrectly charged intermediate traversal prefixes and
  queued search work against the 128-candidate limit. Planner v4 counts only
  completed attack/pass plans as candidates; intermediate prefixes are
  traversal-only and are independently bounded at 4,096 unique sequences.
  Either bound fails closed; candidates are never silently clipped.
- The same frozen 18-position pool now supports 11 positions at depth three;
  seven exceed the 128 complete-candidate cap. Those 11 yield 761 completed
  candidates after exploring 1,686 unique prefixes. No position hit the
  expansion-prefix cap, and no candidate rollouts or labels were produced.
- Full details and checksums are in
  `transition-macro-complete-cap-v4-2026-09-22.json`. Harness commit
  `6b26454` passed typecheck, all 75 engine tests, and the focused Python suite
  (16 passed; the known XGBoost native metadata test was excluded).
- This corrects the accounting gate and improves support measurement; it does
  not establish policy quality. PPO, promotion, and 24/7 operation remain off.
- A one-position, one-rollout-per-candidate smoke at the maximum 80-decision
  horizon evaluated 81 complete candidates with common random seeds. It
  correctly produced 25 horizon truncations and 56 typed action-resolution
  errors, with zero finished outcomes and zero candidate scores. The raw local
  record is ignored; checksums and reason counts are in
  `macro-label-horizon80-smoke-v4-2026-09-22.json`.
- Do not collect broader rollout labels yet. Exact planned action sequences
  generated under one sampled state do not reliably resolve during search
  rollouts. A likely cause is independently sampled hidden state between the
  planner and search. Confirming this requires an explicitly authorized
  research-search change to support matched determinization and chance streams;
  no such engine/search change was made here.

## Position-stage availability and resumable sampling (2026-09-22)

- Audited the 12 read-only Raging Bolt source replays using only each frame's
  actor observation. They contain 177 opening, 343 midgame, and 938 late
  actor-visible states. Only 37 opening positions pass the pool's strict
  eligibility gate; every later-stage state carries a historical
  search-unavailable marker for unreconstructed revealed-card/known-order
  history or an unsupported modified-state flag. The 18-position v8 pool is
  opening-only for this compatibility reason, not a sampler defect. Do not
  bypass the gate or reconstruct from private replay fields.
- A 34-candidate opening position was sampled 16 times per candidate (544
  rollouts, 5-second budget, two workers). Six rollouts finished and 538 were
  budget-truncated; no candidate had more than one completed score. This is
  runtime/development evidence only and is not ranker-eligible. Unlike the
  earlier v4 smoke, these matched-hypothesis rollouts had zero typed
  action-resolution errors across all 544 attempts; this closes that
  execution-reliability blocker for this opening position, but does not solve
  terminal sample scarcity or historical later-stage incompatibility.
- Added per-matched-sample-index atomic resume checkpoints and provenance/stage
  fields for future immutable pools. Focused Python orchestration tests: 9
  passed, including interruption/resume bit-equivalence and source-game split
  isolation. The strategy validator passed; focused Python orchestration,
  baseline-pilot, and decision-guard tests passed (15 total). `npm run
  typecheck` and `npm run engine:build` passed with engine fingerprint
  `c212653686a59248`. The full engine suite passed (76/76), including all
  fifteen competitive lists reaching terminal games. `git diff --check`
  passes.
- No training, PPO, promotion, service installation, or production-policy
  change was made. Historical later-stage records remain ineligible until an
  independently tested reconstruction is available.

## Fresh actor-only current-engine position collection (2026-09-22)

- Added the research CLI command `collect-fresh-positions`. It freezes engine,
  deck/feature identity, collector version, policy, max decisions, schedule,
  and source code hash; checkpoints each completed game with a replay SHA-256;
  and refuses identity drift, changed replay hashes, unexpected files, or
  uncheckpointed replay artifacts on resume. It accepts only the TS or Python
  frozen heuristic and creates no policy labels or training targets.
- Collected six serial TypeScript-heuristic games (two each for Raging Bolt vs
  Crustle, Dragapult, and a Raging Bolt mirror): six finished, zero truncations,
  zero errors, 1,161 decisions, and 23 engine-provided searchable actor
  positions. The original v1 schedule put absolute first player in seat 0 for
  every game; target-deck seat was balanced in the cross-matchups. This is
  explicitly position coverage, not matched playing-strength evidence.
- Each stored replay contains only the decision actor's observation in every
  frame; the opposite seat slot is null and chance records were discarded. All
  six compressed replay hashes and all 1,167 actor-only frames were verified.
- Built an immutable 13-position pool from the six current-engine replays.
  All 13 positions were opening/turn 1, and the transition planner supported
  all 13 without hitting the complete-candidate or traversal caps (2–68
  candidates per position). No rollout labels were collected. Candidate and
  replay hashes are captured in
  `fresh-position-coverage-v1-2026-09-22.json`; local compressed replays remain
  ignored under `artifacts/learning-mind-v1/`.
- The committed v2 collector source hash matches harness commit `6d66340`.
  The v2 schedule alternated mirror first player and fully balanced the four
  Raging Bolt seat × first-player cells in each cross-matchup. Its seed token
  did not yet include the collector version, so its first two seeds per
  matchup overlap the separate exploratory v1 run; the two pools were not
  combined. Collector v3 now includes its version in seed derivation, and v1/v2
  outputs must not be resumed under v3.
- The new collector tests plus the existing focused orchestration, baseline
  pilot, and decision-guard tests pass (21 total); `git diff --check` passes.
- Fresh mid/late positions are still blocked by engine-side knowledge
  restrictions after effects such as temporary-zone reveals. Do not bypass
  them. The next implementation target is a tested, actor-visible reconstruction
  contract for those facts; until then this pool is too narrow for ranker fit.

## Fresh actor-only position collection v3 (2026-09-22)

- After the research engine began preserving the two exact public once-per-turn
  flags used by supported determinization, the v3 collector ran 12 games: four
  each in Raging Bolt vs Crustle, Raging Bolt vs Dragapult, and Raging Bolt
  mirror. All 12 finished (zero truncations/errors), with 2,079 decisions.
- The seed namespace is versioned. The 12 unique v3 seeds overlap neither the
  exploratory v1 nor balanced v2 seed sets. Cross-matchup seat/first-player
  assignments remain balanced; mirror first player alternates.
- Verified all 12 compressed replay hashes and the run-manifest checksum. The
  2,091 stored frames contain only the decision actor's observation; opposite
  private views are null and chance records are empty. Raw local replay bytes:
  14,114,328.
- The immutable pool contains 18 game-disjoint rows: eight opening, eight
  midgame, and two late; 12 train, three development, and three held out by
  source game. The transition candidate audit supported all 18 positions with
  2–116 candidates each. No candidate rollout labels were collected.
- Exact run, pool, replay, and identity hashes are recorded in
  `fresh-position-coverage-v3-2026-09-22.json`. These are coverage artifacts,
  not evidence of strategic improvement. The pool still contains only one
  opponent-policy family and does not establish blind generalization.
- No ranker fit, supervised training, PPO, policy promotion, or service
  installation occurred. Continue to treat unsupported actor-visible history
  reconstruction fail-closed; v3 does not complete the representation or
  learning milestone.

## Matched v4 position-coverage check (2026-09-22)

- Recollected the same 12 deterministic seeds under the new engine identity.
  All games finished; decisions, action sequences, and outcomes match v3
  exactly. No policy or game mechanic changed in this check.
- Actor-searchable decision frames increased from 225 to 252 (+27, or 12% in
  this small sample). This is evidence that the three explicitly whitelisted
  public state fields removed some unsupported-state rejections; it does not
  establish full history support.
- Verified all 12 replay hashes, 2,091 actor-only frames, null opposite views,
  and empty chance arrays. The new immutable 18-row pool again spans eight
  opening, eight midgame, and two late positions; its source-game split is
  12/3/3. No rollout labels were collected.
- Candidate support was not rerun under the new engine identity. Do not reuse
  v3 candidate-support counts or train from the v4 pool. Complete hashes and
  boundaries are in `fresh-position-coverage-v4-2026-09-22.json`.

## Public state transport extension (2026-09-22)

- Replay inspection showed additional unsupported public-state flags in later
  positions: `cannotPlayItemCards`, `rocketSupporter`, and
  `cannotRetreatNextTurn`. Added these exact scalar fields to the research
  belief-state whitelist; hidden-history restrictions were not changed.
- A determinization round-trip fixture verifies all three flags survive into
  the sampled state. TypeScript typecheck passes; engine build fingerprint is
  `3dddf99eb4911076`, worker bundle SHA-256 is
  `1b7f3347aa2dbb548490288b4d62a76c5980c5acd48a7b3679b18922d4816749`, and
  the full engine suite passes (78/78).
- Existing v3 positions and generated evidence retain their prior engine
  identity and are not treated as data from this new build. Recollect under a
  new output identity before using positions with the updated sampler. This
  narrow state transport repair does not claim to remove any knowledge-history
  blocker or validate sampled strategic quality.

## Balanced fresh-position batch v2 (2026-09-22)

- Ran the committed v2 collector after the harness commit, with four games in
  each of three cells: Raging Bolt vs Crustle, Raging Bolt vs Dragapult, and
  Raging Bolt mirror. All 12 finished (0 truncations, 0 errors), totaling
  2,263 decisions and 38 engine-provided searchable actor positions. Cross
  matchups covered all four Raging Bolt-seat × first-player assignments; the
  mirror alternated first player. This remains position-coverage data, not
  policy evaluation.
- Verified the run-manifest checksum and all 12 compressed replay hashes.
  The 2,275 stored frames retain only the decision actor's observation; all
  opposite-seat slots are null and the replay chance arrays are empty. Raw
  artifact size was 15,453,726 bytes. The frozen collector hash in the run
  manifest matches `6d66340`.
- The 18-position game-disjoint pool contains all three matchups but all rows
  are turn 1. Candidate generation supports 18/18 positions, producing 2–89
  complete plans each without rollout or expansion-cap failures. No rollout
  labels were generated. Pool split: 11 train, 3 development, 4 held out by
  source game; there is still only one opponent policy family, so this is not
  a blind policy-family holdout.
- The v2 seed namespace reused its first two seeds per matchup from the
  separate v1 exploratory run because collector version was not in the seed
  token. The v2 run itself has 12 unique seeds, and the v1 and v2 pools were
  not mixed. v3 corrects the namespace for future batches.
- Full checksums, schedule and compact coverage metrics are in
  `fresh-position-coverage-v2-2026-09-22.json`. Current engine-side
  knowledge-history restrictions still exclude mid/late states; no guard was
  bypassed. No model training, PPO, policy promotion, or service installation
  was started.

## Actor-visible history coverage v5-v7 (2026-09-23)

- Three matched 12-game collections used the same deterministic seeds, policy,
  schedule, action sequences, and outcomes: 12 finished games per run, zero
  truncations/errors, and 2,079 decisions. The research-only engine changes
  affected information transport, not observed play. Full identities and
  checksums are in
  `fresh-position-coverage-v5-v7-2026-09-23.json`.
- Searchable actor decision frames rose from 266 (v5) to 532 (v6) to 929 (v7).
  The v6 change safely reconstructs Ultra Ball's own-deck reveal as known
  non-Prize identities without inventing deck order. The v7 change records
  Crispin's opponent-visible reveal only after its prompts resolve and the
  final zone is observable. The v7 engine suite passes 80/80; typechecking
  passes.
- The v7 privacy audit verified 12 replay hashes and 2,091 actor-only frames;
  opposite private views and chance records are absent. Its 18-position pool
  spans opening/midgame/late, but has only three source games (one per split)
  and one opponent-policy family. Candidate support was not re-audited under
  the v7 identity and no rollout labels exist. These rows must not be fit as a
  ranker dataset or treated as independent held-out evidence.
- The generic revealed-card/known-order rejection is absent for stable v7
  decisions in this sample. There are 1,059 non-stable awaiting-choice frames
  and 91 hidden-marker-provenance frames. The latter remain fail-closed because
  actor observations do not expose the hidden marker source; do not recover it
  from the opposite view or private replay data.
- This is sample-specific representation coverage, not complete representation
  parity, strategic improvement, or playing-strength evidence. No ranker fit,
  supervised training, PPO, promotion, or service installation occurred.
  Next gate: add diverse actor-visible source games/policy families, then
  re-audit v7 candidate support and tracker parity before collecting labels.

## v7 transition-candidate support audit (2026-09-23)

- Revalidated the immutable 18-position actor-visible pool against its exact
  current engine/feature identity and planner bundle. All 18 positions generated
  complete attack/no-attack candidate sets: 435 candidates total, 1–116 per
  position under the cap of 128. Candidate depths were 19 at depth one, 100 at
  depth two, and 316 at depth three.
- This ran candidate generation only. No rollouts, labels, ranker fitting, or
  training occurred. Support is not candidate quality or strategy evidence.
- Source limitations are unchanged: three games (one per split), one opponent
  policy family. Broaden independent source-game and policy-family coverage
  before collecting labels. Checksums and exact identities are in
  `macro-support-v7-2026-09-23.json`; detailed ignored local audit is retained
  under `artifacts/learning-mind-v1/fresh-actor-positions-v7/`.

## Second frozen policy-family coverage batch (2026-09-23)

- The Python heuristic collector completed a separate balanced 12-game batch
  (four per Crustle, Dragapult, and Raging Bolt mirror cell): 12 finished,
  zero truncations/errors, and 2,447 decisions. All 12 compressed replay file
  hashes and embedded frame checksums passed. The 2,459 decoded frames contain
  only the decision actor's view; opposite views are null and chance arrays are
  empty. Raw local replay bytes: 22,928,657.
- Built separate 72-position actor-visible pools for the TypeScript and Python
  heuristic families under the same exact engine/feature identity. Each has 24
  opening, 24 midgame, and 24 late rows across six source games, split by game
  into four train, one development, and one held-out game. Pools are not merged.
- On the Python family's initial 18-row subset, candidate generation supported
  15/18 positions and produced 379 complete candidates; the other three failed
  closed at the 128-candidate cap. This audit does not apply to the expanded
  72-row pool. The expanded pools still need candidate-support and split-quality
  audits; one held-out game per family is not robust generalization evidence.
- Run, pool, and support-report hashes are in
  `fresh-position-coverage-v8-2026-09-23.json`. No rollout labels, ranker fit,
  supervised training, PPO, or promotion occurred.

## Expanded-pool integrity and split audit (2026-09-23)

- Re-encoded all 144 rows under the exact frozen identity: 72/72 per policy
  family matched. Each source game appears in only one split, and the two
  families have no duplicate actor-visible positions.
- The resulting row split is too imbalanced for evaluation: TypeScript has
  67/4/1 train/development/held-out rows; Python has 49/1/22. These come from
  only six selected source games per family. Source-game disjointness alone is
  insufficient; development and held-out row counts are not defensible evidence.
- Candidate support was not audited on the expanded pools. Do not collect
  rollout labels or fit a ranker from them. First fix and test balanced
  game-level pool selection, then rerun feature and candidate-support audits.
- Counts and per-game provenance are frozen in
  `expanded-pool-integrity-v8-2026-09-23.json`.
- Stage-by-game analysis showed the root cause: only three games per family
  contribute any midgame/late rows; the other three contribute opening rows
  only. Keep game boundaries intact. The next data gate is additional
  independent games with stable mid/late observations, not moving positions
  across train/dev/held-out. See
  `expanded-pool-stage-provenance-v8-2026-09-23.json`.

## Fresh coverage epoch B (2026-09-23)

- Added a tested `--collection-namespace` slug to derive deterministic new
  seeds and replay IDs; the default `main` schedule preserves its prior seed
  derivation. The named `coverage-2026-09b` epoch completed 12/12 games for each
  heuristic family, with no truncations/errors or overlap with that family's
  prior seeds. All 24 replay hashes and compressed frame checksums passed;
  actor-only views and empty chance records were verified.
- Each family's new 144-row pool has 48 opening, 48 midgame, and 48 late rows.
  Seven selected source games per family contribute midgame/late states, up
  from three in the prior expanded pools. However, each family still has only
  one development and one held-out source game, with imbalanced row counts.
- Candidate support has not been audited on these 288 exact rows. No rollout
  labels, ranker fit, supervised training, PPO, or promotion occurred. Run and
  pool hashes are in `fresh-position-coverage-v9-2026-09-23.json`.

## Game-balanced macro-pool support audit v2 (2026-09-23)

- Rebuilt the exact epoch-B pools with source-game-disjoint balanced splits:
  Python 91/29/24 rows from 6/2/2 games and TypeScript 68/39/37 rows from
  5/2/2 games (train/development/heldout); each pool has 48 rows per stage.
- Audited all 144 rows per family using the committed
  `audit-macro-candidate-support` CLI. It re-encodes observations against the
  frozen feature identity and generates transition candidates only. Status is
  `no-rollouts-no-labels`; no outcomes, labels, fitting, training, or promotion
  were produced.
- Python: 138 supported, 6 fail-closed due to the 128-candidate cap, 2,519
  complete candidates. TypeScript: 142 supported, 2 fail-closed due to that
  cap, 3,561 complete candidates. Every one of the 288 positions had at least
  one complete candidate before cap enforcement. Do not silently drop the ten
  unsupported positions or call the support rate a strategic result.
- Both audits share identity
  `e32fd5094ae1db4c053dde7b1e7a04b080535426eb0df4dc77ff05c1cadcafa8` and
  planner bundle SHA-256
  `ffc636cbd7d439ecf3726eaaf991339ec49adda6ad12ac6363c96aefb5a47a21`.
  Python pool manifest/rows SHA-256 are
  `51f37bb3b148d9d66348b54a2fd6f85ad97bb05213e7fc6735a83cd1cf879f99` /
  `2037c7a6af38e15ac6ad4227b1c08e1677b09f52e378ecda62979b911ecb80d6`;
  TypeScript values are
  `846c167d498ad4f16791b03a922bb635f479d88ad7338c5d0c3f0d448a1caefc` /
  `4eb9a50dcdf2f67686fab872d9faffb8381fc2226d39535d3a64c797c30e04f5`.
- The complete ignored reports and their file checksums are recorded in
  `macro-support-game-balanced-v2-2026-09-23.json`. Current gate: investigate
  principled candidate-cap handling and collect more independent source games
  before rollout labeling or model fitting.
