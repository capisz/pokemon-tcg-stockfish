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
- Bumped the collector identity to v2 for future runs: mirror first-player
  assignment now alternates, four cross-matchup games balance both Raging Bolt
  seat and first-player factors, and the collector source hash is frozen.
  Existing v1 run output is immutable and must not be resumed under v2.
- The new collector tests plus the existing focused orchestration, baseline
  pilot, and decision-guard tests pass (21 total); `git diff --check` passes.
- Fresh mid/late positions are still blocked by engine-side knowledge
  restrictions after effects such as temporary-zone reveals. Do not bypass
  them. The next implementation target is a tested, actor-visible reconstruction
  contract for those facts; until then this pool is too narrow for ranker fit.
