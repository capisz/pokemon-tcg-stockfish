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

## Supervised checkpoint provenance hardening (2026-09-28)

- Supervised checkpoint schema v2 binds each checkpoint to SHA-256 identities
  for the trainer, Transformer, encoder, and observable-history tracker, plus
  the Python/PyTorch CPU runtime settings. Auditing, safety receipts, and resume
  reject implementation drift.
- Checkpoints now include sorted ranker-teacher hashes and a parent checkpoint
  hash. Resume is same-path only; fresh training refuses to overwrite an
  existing checkpoint. Exact parent-chain and no-overwrite behavior are tested.
- The complete Python suite passes: 400 tests. The strategy-contract validator
  confirms approved v1.2 remains active; `git diff --check` passes.
- This closes a provenance gap only. No training run, collector, PPO, promotion,
  or autonomous service was started or enabled; the active collector was not
  inspected or polled.

## Independent supervised-evidence gate

- Supervised held-out evaluation is accepted only after the audit command
  recomputes policy actions from the frozen checkpoint and actor-visible rows,
  verifies the dataset/checkpoint/evaluation identities, and exactly covers
  every non-training position.
- The comparison unit is an independent held-out game-side (or review-family
  side when no source game exists), not a decision row. The gate requires at
  least 20 such units and at least 20 decisive paired units; the two-sided
  Wilson 95% lower bound of model wins among decisive units must exceed 0.5.
  Ties remain reported and do not count as decisive wins or losses.
- Blind-family evidence is separately aggregated only for held-out game-sides
  whose frozen opponent-policy family is absent from every supervised training
  row; review-only and frozen-replay source labels do not count as opponent
  policy families, and missing family metadata fails closed. At least one
  individual unseen opponent family must pass the same independent-side
  minimum and Wilson support rule; evidence is not pooled across undersized
  families.
- PPO enablement additionally requires explicit passes for legal-action
  coverage, autoregressive legality, candidate-cap integrity, exact evaluation
  identity, and human review of representative disagreements. These are
  separate evidence receipts; none is inferred from the label audit.
- `audit-candidate-safety` re-encodes every non-training evaluation position,
  compares the exact multiset of engine legal-action IDs to encoded action
  classes, checks token/action caps and checkpoint logits, and exercises the
  one-engine-action autoregressive decoder (`minimum=maximum=1`). Its immutable
  receipt is tied to the supervised audit, checkpoint, and evaluation hashes.
- `build-disagreement-review` creates an immutable packet containing every
  held-out position where the model and heuristic selected different action
  classes. It copies only `observation` for the acting player, includes the
  two legal action classes and frozen labels, and binds itself to the exact
  audited dataset, checkpoint, and evaluation hashes. The review form must
  cover every packet position with a named reviewer, a finding, and rationale;
  the receipt marks the PPO review prerequisite true only when all findings are
  `acceptable`. Concerns and follow-ups remain unresolved and keep the gate shut.
- Before producing safety or review receipts, downstream commands now rerun
  the supervised audit from its frozen inputs and compare the complete report;
  a recomputed self-hash alone cannot make a forged pass status valid.
- This gate only strengthens supervised evidence. It does not enable PPO,
  automatic promotion, or continuous operation; those still require all
  existing stage gates and explicit human authorization.
- CLI: `python -m ptcg_lab.learning_mind audit-supervised-evaluation
  --dataset DATASET --checkpoint CHECKPOINT --evaluation EVALUATION --output
  NEW_AUDIT_JSON`. The output is immutable and includes the verified hashes
  and per-game-side comparisons.

## Five-archetype generalist pools and label collection (2026-09-23)

- A fresh matched epoch collected 60/60 games for each heuristic policy family
  across all 15 five-archetype matchup cells. Both families had zero
  truncations, engine errors, missing replays, or replay-hash mismatches.
- Generalist pool v13 contains 250 actor-visible positions per policy family,
  balanced to 50 for each target deck. Game-disjoint splits contain 172 train,
  39 development, and 39 held-out rows. Exact pool manifests and source replay
  identities remain under ignored local artifacts.
- Target-deck-stratified 10-position support screens passed 9/10 positions for
  each family; one position per family failed closed at the 128-candidate cap.
  These small screens are diagnostic, not support-rate estimates.
- A second exact-position audit selected five supported train positions, one
  per target deck, from five distinct source games. Candidate counts are 18,
  70, 5, 20, and 8. The adaptive 16-to-64 rollout collection subsequently
  completed; the compact evidence and file hashes are in
  `macro-labels-v13-generalist-five-train-2026-09-23.json`. No ranker has been
  fit from it and no development or held-out rows enter labels.
- Harness increment `cc3802e` adds target-deck round-robin pool selection,
  deterministic target-deck-stratified support sampling, exact repeated
  position-hash CLI selection, split guards, and the engine's 500-step horizon
  cap. The focused orchestration suite passed 21 tests, TypeScript typecheck
  passed, and `git diff --check` passed before commit. The commit is pushed to
  `codex/learning-mind-v1`.
- PPO, continuous operation, and trusted promotion remain disabled. The next
  gate is completed label evidence plus independent, game-disjoint ranker
  evaluation; this batch alone is not strength or policy-improvement evidence.

## Five-position generalist macro-label pilot (2026-09-23)

- The resumable v13 training-only collector completed all five selected,
  source-game-disjoint positions under identity
  `e32fd5094ae1db4c053dde7b1e7a04b080535426eb0df4dc77ff05c1cadcafa8` and
  rollout identity
  `c4646ed7f00c80e3f54105ac2b266579ed64f5820ac08a2637cd4d0651a483b3`.
  The local manifest and per-position file hashes are summarized in
  `macro-labels-v13-generalist-five-train-2026-09-23.json`; raw records remain
  ignored local artifacts.
- All 121 candidate plans received 64 attempts each. The collector recorded
  7,731 terminal outcomes, 13 budget truncations, and zero engine errors.
  Truncations occurred only in the 70-candidate Dragapult mirror position;
  they remain censored and were not converted into draws. All five records
  explicitly report high-confidence policy eligibility as false.
- The sample covers each target archetype and five distinct source games, but
  every source is from the Python heuristic family and there are only five
  independent positions. No development or held-out positions were labeled.
  This is resume/hash/runtime evidence, not enough breadth to fit or accept a
  ranker and not evidence of strategy improvement or playing strength.
- No ranker fit, policy training, PPO, continuous operation, or trusted
  promotion occurred. The next gate is a much broader training label set
  balanced across both heuristic policy families and target archetypes, plus
  independently labeled development positions and an untouched held-out set.

## Full v13 candidate-support audit and CLI repair (2026-09-24)

- Audited all 250 positions in each frozen v13 heuristic-family pool under
  the same engine/feature identity and transition-aware planner bundle. Python
  support: 231/250, including train/development/held-out counts 159/172,
  37/39, and 35/39. TypeScript support: 235/250, with 162/172, 36/39, and
  37/39. Combined: 466/500 supported and 13,954 complete candidate plans.
- The remaining 34 positions fail closed at the declared candidate cap; zero
  supported positions lacked a complete candidate. These are coverage results,
  not strategic quality measures. No games, rollouts, labels, or model fits
  were performed by the audits. Full report identities, sizes, and checksums
  are in `macro-support-v13-generalist-full-2026-09-24.json`; detailed reports
  remain ignored local artifacts.
- Fixed the candidate-support CLI's default empty position-hash list, which
  previously made ordinary all-row audits fail immediately. The focused
  regression subset passed 2 tests and the full orchestration module passed
  21 tests. The full Python suite hit the known XGBoost native segfault in
  `xgboost/data.py`; rerunning with only that ranker test excluded passed
  326 tests (one deselected). JSON validation and `git diff --check` passed.
- The next data gate is a preregistered source-game-balanced subset across
  both policy families and all five decks, with development labels separated
  and held-out positions left untouched until the evaluator is frozen. The
  existing five training labels do not justify ranker fitting. All promotion
  gates remain disabled.

## Fresh source-game-balanced coverage and macro labeler diagnostic (2026-09-23)

- Epoch `coverage-2026-09c` completed 60/60 games across the Python and
  TypeScript heuristic families: 30 each, all finished, with zero truncations
  and zero engine errors. Each family used 30 source games with game-disjoint
  train/development/heldout splits of 20/5/5; selected pools contain 96/24/24
  rows (opening/midgame/late coverage is approximately balanced). No seed
  overlap with epoch v9, cross-family seed overlap, cross-family duplicate
  positions, or source-game split violations were found.
- The v3 balanced pool builder round-robins through source games within each
  matchup/policy/stage bucket. Full support audits covered all 144 rows per
  family. Python supports 125 rows and TypeScript 132; the remaining 19 and 12
  rows respectively fail closed from candidate-cap overflow or infeasible
  public-zone-count belief hypotheses. No candidates were silently dropped.
- On one 11-candidate Raging Bolt position, the frozen 15-second budget and 16
  common-seed rollouts per candidate yielded 174 terminal outcomes, two budget
  truncations, and zero errors. Uncertainty remained as high as 0.1291, and no
  high-confidence policy labels were emitted. This is a bounded runtime
  diagnostic, not ranker, training, or playing-strength evidence.
- Artifact identities and checksums are in
  `fresh-coverage-v10-macro-label-diagnostic-2026-09-23.json`; full raw games,
  pool rows, audit detail, and label output remain ignored locally under
  `artifacts/learning-mind-v1/`.
- Current gate: collect broader independent source-game coverage and improve
  candidate support and label precision before serious ranker fitting. PPO,
  continuous operation, and promotion remain disabled.

## Adaptive 64-rollout allocation diagnostic (2026-09-23)

- A single 11-candidate Raging Bolt position completed 700/704 rollouts at a
  15-second per-rollout budget: 700 terminal outcomes, four budget truncations,
  and zero errors. Each candidate received 62–64 completed outcomes.
- The top two observed mean results were 0.8125 and 0.6719. A post-hoc
  per-candidate 95% Hoeffding radius at 64 outcomes is about 0.1699, so these
  intervals still overlap. High-confidence policy eligibility is deliberately
  disabled; this is not a policy label or a strategy result.
- Harness inspection found that the close-candidate set is selected after the
  initial 16 outcomes and then frozen through the 64-outcome maximum. It is not
  reevaluated between extension stages, so every initially close candidate
  receives the full extension even if later outcomes separate it.
- Exact identities and hashes are recorded in
  `macro-adaptive64-diagnostic-2026-09-23.json`; rollout outputs remain local
  and ignored. Next, implement and test staged reevaluation/pruning with
  resumable checkpoints, then compare cost and top-plan ordering on frozen
  development positions. Do not scale labeling or enable PPO on this evidence.

## Staged simultaneous-bound allocation validation (2026-09-23)

- Harness commit `0ee5e2e` adds resumable eight-rollout extension stages,
  monotone close-set reevaluation, a simultaneous Bonferroni-union Hoeffding
  budget over candidates and planned looks, exact attempted-rollout counts,
  and allocation settings in checkpoint/output identity.
- Tests prove distant candidates can be pruned only after the declared bound
  separates them, active candidates retain matched seed indices, interrupted
  mid-stage collection resumes without replaying completed batches, and
  results match an uninterrupted run.
- Full validation passed: 321 Python tests, 80 engine tests, strategy v1.2
  contract validation, TypeScript typecheck, and `git diff --check`.
- The same frozen Raging Bolt position under the new collector completed
  699/704 rollouts with five budget truncations and zero errors. All 11
  candidates remained close through 64 attempts, so this diagnostic saved no
  rollout compute. The top two observed means tied at 0.65625. The prior
  adaptive64 run used a different configuration-bound seed stream and is not
  directly comparable. No high-confidence labels, ranker fit, or training
  resulted.
- Checksums and identities are in
  `macro-staged64-diagnostic-2026-09-23.json`; raw outputs remain ignored. Next
  investigate per-index common-random-number paired differences as a
  descriptive variance reduction on multiple frozen positions. Do not scale
  label collection from this single-position result.

## Common-random-number paired-results diagnostic (2026-09-23)

- Harness commit `264fcff` preserves each completed score by rollout index and
  reports candidate-versus-observed-leader paired differences and standard
  errors as explicitly descriptive values. They are not confidence bounds and
  do not control pruning or label eligibility. A follow-up marks paired counts
  below 20 as insufficient.
- One 11-candidate Raging Bolt diagnostic attempted 176 rollouts at a 15-second
  budget: 87 finished, 89 were budget-truncated, and zero errored. Common
  finished pairs ranged only from two to eight; all 11 comparisons are below
  the 20-sample minimum. Zero paired standard errors on this tiny, incomplete
  sample are not evidence of zero variance or a quality signal.
- The report and artifact hashes are in
  `macro-paired16-diagnostic-2026-09-23.json`; outputs remain ignored locally.
  Do not use this as a macro training label or a reason to scale collection.
- Next gate: attain at least 20 finished matched seeds on multiple frozen
  development positions, preserve truncation accounting, and test whether
  incomplete pairs bias the descriptive comparisons. High-confidence policy
  eligibility, PPO, continuous operation, and trusted promotion remain off.

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

## Paired rollout completion across frozen development positions (2026-09-23)

- Two late-game Crustle positions from different source games completed every
  20-seed matched rollout at a 60-second budget: 460/460 across 23 candidates,
  and 140/140 across seven candidates. Both have 20 common finished outcomes
  for every candidate.
- Dragapult was not uniformly tractable at the same cap. A 21-candidate
  position completed 416/420 outcomes but only 18–19 paired finishes per
  comparison. An eight-candidate position completed 115/160, with 45 budget
  cutoffs and only 7–14 common finishes. Neither reaches the predeclared
  20-sample minimum.
- Paired variability is descriptive and selected-leader-relative, not a
  confidence interval or policy label. These development-only diagnostics did
  not train or fit any model. Complete identities, outcomes, and checksums are
  in `macro-paired-development-budget60s-2026-09-23.json`; raw artifacts remain
  ignored locally.
- Next gate: collect only game-disjoint training positions across approved
  archetypes and opponent families, choose a position-complexity-aware but
  identity-bound rollout/censoring protocol, and preserve development
  separation. Do not fit the ranker until there is sufficient independent
  training breadth. PPO, continuous operation, and promotion remain disabled.

## Five-archetype actor-position coverage and stratified planner screen (2026-09-23)

- Completed 60-game Python-heuristic and 60-game TypeScript-heuristic coverage
  runs: five archetypes, all 15 unordered matchup/mirror cells, four games per
  cell, and 120/120 finished outcomes with zero truncations or engine errors.
  All 120 replay records had a present file with a matching SHA-256. The Python
  and TypeScript run-manifest file hashes are
  `f96ab09610515840a11d3dec8b740a1181da48424258e0a7bc0438ee9052893f` and
  `6ffc2a6c1bf024f7e57eb13a923dd9acf13983471697665a2a6856fc99761190`;
  both bind identity `e32fd5094ae1db4c053dde7b1e7a04b080535426eb0df4dc77ff05c1cadcafa8`.
- Built ten unlabeled, source-game-disjoint pools (one per policy family and
  target archetype), 50 positions per pool. Their ordinary heuristic actions
  were not converted to policy labels.
- Added deterministic stratification when `audit-macro-candidate-support`
  receives a row limit: prioritize underrepresented opponent archetypes, then
  position stage, split, and source game. Unit coverage verifies repeatability
  and breadth across those dimensions. This avoids a misleading first-N sample
  dominated by the earliest matchup.
- Screened five positions per pool, with one sampled opponent matchup for each
  target deck and a mixture of position stages/splits. Python support was
  21/25 and TypeScript support 23/25 (44/50 overall); all six unsupported
  positions failed closed at the existing 128-candidate cap. These are small,
  stratified diagnostic samples, not estimates of population support and not
  evidence of strategic quality. Every sampled position that was supported
  had at least one complete executable candidate; no rollout, label, ranker,
  supervised training, PPO, or promotion was run.
- The cap overflows occurred in late positions with large branching factors
  across several archetypes, confirming that planner support—not data capture—
  is the immediate macro-learning blocker. Do not raise the cap or drop plans
  without a separately versioned and tested coverage policy.
- Full audit summaries, report hashes, and file checksums are in
  `macro-support-stratified-screen-v11-2026-09-23.json`; raw replays, pools,
  and detailed support reports remain ignored local artifacts. PPO,
  continuous operation, and trusted promotion remain disabled.
### TypeScript-family macro-rollout label pilot (2026-09-23)

- Completed five train-only positions from five distinct TypeScript-heuristic
  source games, one position per target deck, using rollout identity
  `86d7bf7ac54bebe7f844a57d923b4134df642b1ebdbd785db1b9036031b1785f`.
- The immutable local manifest is
  `artifacts/learning-mind-v1/macro-labels-v16-generalist-train-five-targets/typescript/manifest.json`
  (manifest hash `bc87dc9d580c57a8218990fd7a7d7bf5b220c85ca0bcbcaecfba88520e718f9b`).
  All five record hashes verified against that manifest.
- The 31 candidates received 1,984 rollout attempts: 1,980 finished, four
  horizon cutoffs, zero engine errors. The cutoffs are censored, not draws.
  No record is eligible as a high-confidence policy label.
- Compact provenance, per-position outcomes, record hashes, and limitations
  are in `macro-labels-v16-typescript-five-train-2026-09-23.json`; raw records
  remain ignored local artifacts.
- Combined with the five-position Python-family pilot, the evidence now covers
  10 source games, 152 candidates, 9,711 finished outcomes, 17 truncations,
  zero engine errors, and zero high-confidence policy labels. This remains
  insufficient for ranker fitting, supervised policy promotion, or strength
  claims. PPO, continuous operation, and trusted promotion remain disabled.

### Game-balanced macro-label selection (2026-09-23)

- Added `freeze-macro-label-selection`, which verifies both v13 pool/support
  identities and hashes, selects supported rows only, pins existing train
  labels, and chooses one row per source game using deterministic greedy
  balance across target deck, opponent archetype, and stage. Heldout is not a
  selectable split.
- Frozen selection is in
  `macro-label-selection-v17-generalist-balanced-2026-09-23.json` (selection
  hash `c1fcb21d10ad38a799490fce3714fe18f72f0dc7e08dadf4431ca6ad1551ae4d`).
  It includes 40 Python-family train games and 39 TypeScript-family train
  games, plus nine development games per family. All five decks, three stages,
  and five opponent archetypes are represented in each family/split.
- The training selections contain 2,022 complete candidate plans and the
  development selections 560. Ten existing pilot positions are pinned to
  avoid replacing them in the selection, but will be regenerated under the
  complete frozen run identity. No heldout positions were selected.
- Selection regression tests: 22 orchestration tests passed. The report's
  selection hash was recomputed and verified; both underlying source-pool
  manifest/support hashes were validated by the CLI. No rollouts were run by
  the selection command.
- This is a precollection design gate only. Collect labels next; do not fit or
  promote a ranker from the ten pilot positions, and keep PPO, continuous
  operation, and trusted promotion disabled.

### Cross-family macro-label aggregation

- Added `combine-macro-label-runs` for the separate per-family collectors. It
  verifies each manifest self-hash, frozen experiment identity, record hashes,
  per-record dataset and rollout identities, filenames, and single-family
  provenance before atomically publishing a combined ranker input. It rejects
  duplicate positions and requires both approved policy families.
- Aggregation tests cover successful train/development merging, corrupt record
  rejection without publishing partial output, and frozen-identity rejection.
  Combined with the selection/orchestration tests, 25 tests passed.
- The combiner has not yet been run on experiment outputs; the expanded
  Python-family label collector is active. PPO and all promotion gates remain
  disabled.

### Held-out split isolation hardening (2026-09-25)

- Review of the ranker path found that `fit_ranker` reports held-out metrics
  from any held-out records in its input, while the v17 selection intentionally
  excludes those rows until the evaluation protocol is frozen. The combined
  ranker-input publisher now rejects held-out and unknown split labels; only
  train/development records can cross that boundary.
- Added regression coverage for both rejected split classes. Aggregation plus
  macro-orchestration tests pass (27 passed), and `git diff --check` passes.
- No collection, fitting, or held-out evaluation was started by this change.
  The active label collector and its artifacts were left untouched; PPO,
  continuous operation, and trusted promotion remain disabled.

### Ranker input verification (2026-09-25)

- Closed the direct-CLI bypass around the combiner: before fitting, the ranker
  now requires the combined manifest format and verifies its self-hash, both
  policy families, every safe in-directory record path and file hash, record
  identity/status/split, unique position IDs, and split counts. Held-out and
  unknown splits are rejected before model fitting.
- Added tests for valid two-family input, held-out/unknown split rejection,
  post-freeze record corruption, and manifest tampering. Combined aggregation,
  ranker-input, and macro-orchestration tests pass (31 passed); `git diff
  --check` passes.
- No label fitting or collection was started. The active collector remains
  untouched, and all PPO, continuous-operation, and trusted-promotion gates
  remain disabled.

### Immutable ranker artifacts (2026-09-25)

- The ranker fit path now rejects an existing model or sidecar-manifest path
  before reading inputs, calculates metrics before publication, and uses an
  atomic no-replace model-file publication. Each frozen iteration must use a
  fresh output path.
- Regression tests verify existing model and manifest artifacts are retained
  byte-for-byte and fitting is rejected before the input path is read. The
  combined aggregation, ranker-input, and macro-orchestration tests pass
  (33 passed); `git diff --check` passes.
- This did not start a fit or collection. PPO, continuous operation, and
  trusted promotion remain disabled.

### Frozen-selection completeness binding (2026-09-28)

- Further review found that self-consistent run manifests could still describe a
  partial subset of the frozen v17 positions. The combiner and ranker input
  verifier now require the original selection manifest, validate its identity
  and self-hash, and require exact train/development position coverage for both
  approved policy families before publishing or fitting. The combined manifest
  records the selection hash, file hash, and family/split counts.
- CLI usage now requires `--selection` for both `combine-macro-label-runs` and
  `fit-macro-ranker`. Regression tests cover incomplete selection rejection at
  both boundaries. The focused aggregation and ranker-input tests pass (7).
- This change did not inspect or modify collector checkpoints or replay
  artifacts, and it did not run simulations, combine active results, fit a
  ranker, or enable later learning stages.
- A broader learning-mind check passed 36 tests across aggregation, ranker
  input, representation, supervision, orchestration-adjacent modules, and
  supervisor checks. The separate strategy suite reaches its existing
  XGBoost ranker smoke test, where the Python process segfaults inside the
  installed XGBoost NumPy metadata path; the 22 orchestration tests pass when
  run independently. This environment failure is not treated as a passing
  ranker-training verification.

### Weighted XGBoost ranking runtime fix (2026-09-28)

- Isolated the crash to the installed XGBoost 3.2.0 sklearn ranker wrapper's
  weighted `fit` path. The same inputs trained successfully without weights,
  and the core `DMatrix` plus `xgboost.train` path accepted ranking groups and
  per-group weights. The research ranker now uses that core API, retaining
  completed-rollout weights as group weights.
- The full strategy module now passes (18 tests), including weighted fitting
  and prediction. No macro labels were combined and no ranker was fit from
  experiment data; PPO, continuous operation, and trusted promotion remain
  disabled.
- Full Python validation passes (335 tests), the learning-mind modules pass
  (76 tests), the TypeScript engine suite passes (80 tests), and `npm run
  typecheck` and `git diff --check` pass.

### Ranker holdout split isolation (2026-09-28)

- The frozen ranker input excludes the separate heldout split by design, so
  requiring a measured `heldout` metric made the ranker acceptance result
  permanently insufficient. The prior family/archetype holdout loop also
  allowed training rows into its evaluation subset. It now reports the
  separate split as `not-included`; each holdout model trains only on train
  positions outside the held-out group and evaluates only development
  positions inside that group. Missing coverage remains `insufficient`.
- Added regression coverage for family and archetype separation, and for
  train/development split isolation. The targeted strategy, orchestration, and
  ranker-input tests pass (45); the full Python suite passes (337), and
  `git diff --check` passes. Ranker evidence now explicitly requires both
  holdout axes and can report only `insufficient` or `review-required`; it
  cannot claim a win or promotion. The existing heldout label pool is not read
  or modified, and no ranker was fit from experiment data.

### Held-out strategy-probe evaluation (2026-09-28)

- `evaluate-supervised` previously compared held-out label hits against the
  frozen heuristic but hard-coded `targetProbeWin` to false. It now evaluates
  all registered v1.2 probes using model and heuristic choices from the same
  actor-visible held-out decisions. Headline counts use the first qualifying
  decision per probe/game-side; secondary counts include every qualifying
  decision; both report Wilson 95% intervals; fewer than 20 headline game-sides
  is insufficient.
- Probe evaluation requires its own immutable corpus built from every actor
  decision in held-out source games; the engine's final `action: null` terminal
  frame is explicitly excluded as an outcome boundary, not a policy decision.
  It does not misuse the sparse search/review
  training rows. The corpus excludes accepted actions and the opposite private
  observation, binds to the same frozen source manifest as the supervised
  dataset, verifies source identity/hash, and hashes both probe definitions and
  evaluator implementation.
- Severity-three coverage is reported separately from regression; all
  severity-three probes need adequate held-out coverage before the model can
  pass this evidence gate. Only a target-probe win, held-out label win, full
  severity-three coverage, and no severity-three regression produce
  `acceptance: passed`; automatic promotion remains false.
- Corrected the opening-selection evaluator to treat a null prompt as
  nonqualifying. Tests cover actor-view extraction, terminal-frame exclusion,
  sparse-corpus rejection, chronological headline deduplication, minimum
  sample size, and full registry output. The full Python suite passes (343);
  CLI help checks, TypeScript
  typecheck, and `git diff --check` pass. No games were simulated and no label
  collector/checkpoint was read or changed.

### Supervised resume and promotion-evidence integrity (2026-09-28)

- A pause after the final minibatch of an epoch previously saved a cursor past
  the end of the batch order without preserving partial-epoch loss totals; a
  resume could then divide by an empty loss list. The checkpoint now persists
  batch/loss progress and final-batch cursors, and a final-minibatch interrupt
  resumes bit-equivalently to an uninterrupted run.
- Resume now rejects changes to seed, batch size, epoch target, optimizer/model
  configuration, or source dataset-manifest hash. Dataset loading verifies the
  manifest self-hash as well as the rows hash, and trained checkpoints bind to
  the exact supervised dataset manifest that evaluation must use.
- PPO's evidence prerequisite now requires adequate severity-three probe
  coverage, a passed Raging Bolt macro-plan-fidelity gate, a held-out label
  win, a targeted probe win, and an explicit no-regression result. Gate values
  must be exact booleans/statuses, not merely truthy/falsy substitutes. Human
  enablement remains independently required, and the optimizer entry point
  refuses updates unless that full gate record passes.
- Verification: full Python suite 348 passed; TypeScript typecheck and
  `git diff --check` passed. No model fit, games, or collector activity was
  started by this change.

### Continuous-operation supervisor gate (2026-09-28)

- The supervisor's `start` method previously accepted only a human boolean and
  could persist `RUNNING` without checking whether the learning stages had
  passed. It now requires exact accepted-stage flags for PPO, specialization,
  continuous operation, and a separate human continuous-operation approval.
- Startup after reboot remains `PAUSED`. Tests verify every individual gate,
  reject truthy non-boolean substitutes, and retain the three-strike/failure
  pause behavior. The protocol records these prerequisites and keeps the
  continuous-operation default disabled.
- Added running-only JSON cursor checkpoints and sequential collection →
  training → evaluation → retention phase transitions. Cursor values must be
  finite JSON data and survive pause/restart; phase transitions are rejected
  while paused, when skipped, or when malformed. The service remains
  uninstalled and its default state remains paused.
- Verification: learning-mind supervisor tests, full Python suite, typecheck,
  and `git diff --check` pass. No service was started or installed.

### Actor-visible macro-ranker v2 path (2026-09-28)

- Added an isolated 640-value state/plan feature schema and a separate
  `fit-macro-ranker-v2` path. It encodes actor-visible global context, the
  actor's own hand, fixed public board slots, attached Energy, semantic macro
  roles, and referenced target slots. Opponent private hand contents and
  simulation-generated action IDs do not enter the features.
- The v2 report binds its model to the feature-schema hash and feature-source
  hash and portable inference implementation hash. The existing
  `experiment.py` collector/ranker-v1 module is unchanged
  to preserve the active label collector's frozen source identity. A synthetic
  fixture fits the complete v2 path and exercises all archetype and family
  holdouts; no finalized labels were consumed and no real-data model-quality
  claim is made. The old ranker remains available as a separate baseline.
- A synthetic native XGBoost deserialization attempt caused a segmentation
  fault in the configured runtime. To keep future inference out of that native
  load path, v2 exports a portable JSON tree ensemble with a Python scorer.
  Its synthetic prediction scores match native XGBoost to 1e-6, and artifact
  verification checks report/schema/feature-source/inference-source/model
  hashes and feature dimension. Prediction rejects a scorer-source mismatch.
  No ranker was fit from the finalized game labels, so real-data quality and
  held-out performance remain unverified.
- Ranker-v2 top-1 regret now includes reproducible 95% source-game
  cluster-bootstrap intervals, overall and by archetype and policy family; the
  report records each interval's deterministic seed, replicate count, and
  independent source-game count. This quantifies uncertainty within the
  frozen sample only.
- The fitter verifies every record's source game against the frozen selection,
  rejects duplicate source games within a family/split and train/development
  overlap, and clusters cross-family reuse for confidence intervals.

### Portable ranker inference identity (2026-09-28)

- Moved the portable ranker-v2 tree scorer into its own module and bound its
  exact source hash into both the fitted model artifact and report. The
  verifier and prediction API reject scorer-hash drift; the CLI returns the
  verified inference hash for audit records. The active `experiment.py` source
  remains unchanged.
- Verification: focused ranker-v2 tests pass (11); full Python suite passes
  (381, with two existing deprecation warnings); strategy contract validator
  passes; `git diff --check` passes. No finalized experiment labels were fit,
  and no collector, PPO run, continuous service, or promotion was started.
- Before fitting, v2 also independently validates candidate identities,
  actor-visible legal root actions, rollout count/reason reconciliation,
  result and uncertainty bounds, evidence-derived weights, and the per-position
  relative-result center. It binds each family to its source-run rollout
  identity and recomputes recorded seeds by split/position/index; promotion
  seeds and altered streams fail closed. Truncated/error-only candidates remain
  unlabelled.
- Verification: full Python suite 381 passed, strategy contract v1.2 validator,
  new CLI help, and `git diff --check` passed. Feature tests prove deterministic
  encoding, visible-board and plan sensitivity, target-slot sensitivity, and
  invariance to changed hidden opponent-hand contents. No games, labels,
  collector checkpoint, PPO update, service, or promotion were run.

### Supervised target validation (2026-09-28)

- Tightened the policy-label boundary: each supervised row must provide exactly
  one nonempty target form, and soft action distributions must be one-dimensional,
  numeric, finite, nonnegative, normalized within `1e-6`, and mass only on legal
  options. Acceptable-action labels must be integer indices. This prevents
  malformed or contradictory evidence from silently entering policy loss.
- Verification: focused model/strategy tests pass (39); full Python suite passes
  (388, with two existing deprecation warnings); `git diff --check` passes.
  Collector-bound `experiment.py` is unchanged; no labels were fit and no PPO,
  autonomous service, or promotion was started.

### Supervised dataset publication and source identity (2026-09-28)

- The supervised dataset builder now verifies its source replay manifest hash
  and exact experiment identity before reading search labels. Dataset rows and
  manifest are staged in a temporary sibling directory and published by one
  rename; any conversion failure cleans up the staging directory instead of
  leaving an unusable output path behind.
- New fixtures prove actor-view-only row construction, search-distribution
  coverage including zero-mass legal actions, source hash and identity rejection,
  and cleanup after conversion failure.
- The supervised loader now validates row schema, actor/player ownership,
  opponent-hand redaction, encoded feature identity, target dimensionality and
  legality, manifest counts, unique positions, and game/family split isolation.
  Hash-consistent edits that inject an opponent hand or move a family across
  splits are explicitly rejected.
- Verification: focused dataset/orchestration/macro-fidelity tests pass (32);
  full Python suite passes (394, with two existing deprecation warnings). No
  real dataset was frozen and no collector, PPO, service, or promotion was
  started.

### Verified macro-ranker policy distillation (2026-09-28)

- Added a research-only path from a verified portable ranker-v2 model to
  train-only supervised legal-action distributions. It binds the exact ranker,
  report, combined labels, frozen selection, and macro-position-pool artifacts;
  checks measured archetype and policy-family holdouts; revalidates source
  candidate seeds/legal root actions; and aggregates candidate softmax mass
  into the encoder's semantic action classes. The result can be included when
  creating a new immutable supervised dataset. It does not change the teacher's
  `review-required` status or enable promotion.
- Verification: focused distillation/dataset tests pass (8), including a
  synthetic end-to-end ranker-to-action distribution and rejection of an
  unmeasured holdout. CLI help renders; full Python suite passes (396, with two
  existing deprecation warnings); `git diff --check` passes.
- No experiment data was consumed, no ranker fit or game simulation was run,
  and no collector, PPO, autonomous service, or promotion was started.

### Promotion evidence gate hardening (2026-09-28)

- Promotion readiness now fails closed unless the aggregate has at least 100
  finished games and all 25 ordered archetype matchups each have at least 100
  finished games plus a resolved confidence status. Unresolved cells cannot be
  treated as non-regressions; malformed, duplicate, or missing matchup records
  are rejected.
- Sequential evaluation now identifies supported non-regression when the
  decisive-game Wilson lower bound clears the frozen five-point margin, but
  will not issue any supported status until at least 100 decisive games exist;
  draw-heavy cells continue to 250/500 and become inconclusive at the cap if
  decisive evidence remains sparse. The
  gate also requires an explicit empty severity-three regression list, passing
  blind-family evidence, exact identities, and human approval. Automatic
  promotion remains impossible.
- Full Python suite: 400 passed; strategy-contract validator passed with v1.2
  active; `git diff --check` passed. The frozen collector implementation is
  unchanged. No collector was inspected or polled, and no games, training,
  promotion, or service were started.

### Ranker-teacher provenance enforcement at training boundary (2026-09-28)

- The supervised trainer now independently requires exactly the ranker-model
  and report SHA-256 hashes on every ranker-distilled row. Explicit checkpoint
  teacher hashes must exactly match the union of row provenance; non-distilled
  rows cannot claim ranker teachers. Invalid or mismatched provenance fails
  before an output checkpoint is created.
- Focused dataset, distillation, strategy, and model tests pass (51); the full
  Python suite passes (401), and the v1.2 strategy validator passes. The frozen
  collector implementation and active run remain untouched.

### Actor-view terminal value targets and value-only supervised loss (2026-09-28)

- Added a separate immutable value-target dataset builder. It verifies the
  source manifest and exact runtime identity, accepts only explicitly
  training-eligible finished experimental games with valid rules outcomes,
  derives `+1/0/-1` from each acting player's seat, and stores only
  `frame.observations[frame.actor]`. Ineligible games are recorded as excluded;
  hidden opposite-seat observations and selected actions do not become labels.
- The trainer now accepts value-only rows, computes terminal-outcome MSE on
  the blind value head with the frozen `0.5` coefficient, and continues to
  train policy loss only from approved policy-label sources. Checkpoints bind
  both dataset manifest hashes and the dataset/training CLI source hashes, and
  report policy loss and value MSE separately. Value manifests pin their exact
  builder source hash.
  The CLI exposes `build-value-target-dataset` and optional `--value-dataset`
  on `train-supervised`.
- Fixtures cover actor-view isolation, win/loss/draw targets, rejection of
  ineligible outcomes and identity drift, joint policy/value losses, and
  bit-equivalent value-only resume. All 408 Python tests, the approved v1.2
  validator, CLI help checks, TypeScript typecheck, and `git diff --check` pass.
  No real value dataset was built and no training run, collector, PPO update,
  service, or promotion was started.

### Hash-bound held-out value-head diagnostics (2026-09-28)

- Added a separate development/held-out value evaluator. It verifies the
  checkpoint's exact value-dataset hash and implementation identity, refuses
  the training split, and emits an immutable report tied to the dataset,
  checkpoint, and evaluator source hashes. Metrics include per-record and
  per-unique-position MSE/MAE, 10-bin bounded-outcome calibration, and
  game-side summaries; it makes no promotion decision.
- A synthetic train/held-out dataset and value-only checkpoint exercise the
  full evaluator and its training-split rejection. No live or real value
  dataset was read.
- Full Python suite: 409 passed; strategy-contract v1.2 validator, value CLI
  help checks, TypeScript typecheck, `git diff --check`, and the frozen
  `experiment.py` source-identity check passed.

### Require measured macro-ranker evidence before PPO stage authorization (2026-09-28)

- The runbook requires frozen ranker evidence before Transformer training, but
  the PPO-stage verifier did not consume or bind a ranker artifact. A fully
  satisfied supervised gate plus the explicit human flag could therefore
  authorize PPO while strategic-plan ranker evidence was absent.
- PPO-stage verification now checks the portable ranker model and report
  checksum, requires the ranker and supervised checkpoint to share the exact
  frozen identity, and requires measured development, leave-one-archetype-out,
  and frozen-policy-family holdout results. `review-required` remains a
  research status, not ranker acceptance; this check only closes the documented
  stage-order prerequisite. PPO still requires every existing stage condition
  and the separate explicit human authorization.
- The gate also revalidates the exact combined train/development label bundle
  and its independent confidence report, then compares their manifest, frozen
  selection, report, implementation, and file hashes against the ranker. A
  stale report from another valid run can no longer satisfy the gate.
- The supervised dataset must carry the exact ranker model, ranker report, and
  confidence-report hashes in `teacherHashes`; otherwise an older Transformer
  checkpoint trained without the strategic-plan teacher cannot pass the PPO
  gate, even when a valid ranker report is supplied afterward.
- The immutable stage receipt hashes the ranker model/report and records the
  label and confidence inputs plus measured holdout axes. Regression tests
  reject missing/insufficient evidence, mismatched identities, and stale
  confidence/teacher provenance. The low-level `ppo_enablement` function now
  independently rechecks the complete prerequisite and ranker fields, closing
  a capability-level bypass where the stage report could be false but the
  older subset of fields still enabled PPO. Stage, ranker-v2, and PPO
  experience tests pass (39); CLI help and `git diff --check` pass. No
  collector, training, PPO, promotion, or service operation was started.

### Continuous-operation capability mapping check (2026-09-28)

- `VerifiedContinuousOperationRecord` freezes its values as a `MappingProxyType`,
  but the enablement predicate incorrectly required the unwrapped values to be
  a mutable `dict`. Thus a legitimate verified capability could never pass,
  even with every prerequisite and explicit human authorization true.
- The predicate now reads the immutable mapping directly. Regression coverage
  proves plain editable dictionaries still fail, and a valid immutable
  capability can satisfy the check. The capability now also requires passed
  PPO and promotion evidence, exact-deck specialist acceptance, a human-
  reviewed 24-hour soak, every required failure drill, separate human run
  approval, and automatic promotion explicitly false. Each of these conditions
  has a fail-closed regression case. All 26 supervisor tests and
  `git diff --check` pass. No continuous-operation capability issuer exists
  yet; no supervisor was started.

### v17 label finalization, Raging Bolt plan fidelity, and ranker-v2 evidence (2026-09-29)

- Finalized the existing immutable v17 Python and TypeScript label runs after
  source audit confirmed the two exact collector modules share the registered
  byte-identical collection surface. The combined bundle contains 97 selected
  positions and preserves each family’s separate source-module hash, source
  commit, manifest hash, and rollout identity. The confidence reevaluation is
  analysis-only at familywise 95%; it creates no policy-eligible labels.
- Raging Bolt fidelity audit passed for all 17 selected positions and 300
  candidates: 300 declared plans executed, with zero typed plan failures and
  zero untyped errors. This verifies plan executability, not strategic quality.
- Ranker-v2 now fills source-game bootstrap metadata from the exact frozen
  selection when label records omit that field; it rejects any conflicting
  record metadata and does not edit immutable inputs. This resolved the
  initial fail-closed fit attempt. New regression coverage tests this join.
- Fit and portable verification completed for exploratory ranker v2 iteration
  3 (the same deterministic model content, refit after audited holdout-coverage
  reporting changes). Model SHA-256 is
  `3c659dd6f993127d99acd354d285f2f60d2253a5e64a068cde4a1de0fa1285b9`; report
  hash is `018dc274ddd518e73661a8ba7afe90633fae057941e33a9cc5e8f84dd50fd6fa`.
  Training covered 76/79 positions; development covered 18/18 with mean top-1
  relative regret 0.0817 (source-game bootstrap 95% interval 0.0365–0.1418).
  All seven archetype/family holdouts are measured using available comparable
  training positions; per-fold training coverage is 95.0%–97.4% and is
  explicitly reported. Archetype test cells contain only 3–6 positions and
  family cells nine, so these are small-sample descriptive results. Ranker
  acceptance remains `insufficient` because the full training set is incomplete,
  not an improvement claim or PPO evidence.
- Three training positions contain exactly one candidate each, despite 64
  completed rollouts per candidate: `115cda2e…f54ae9` (Python-family Crustle
  vs Raging Bolt, midgame), `92d06e83…6e9580` (Python-family Raging Bolt vs
  Grimmsnarl, late), and `1eece6ac…a7439b85` (TypeScript-family Grimmsnarl
  mirror, opening). More rollouts on these same candidates cannot create
  comparisons. A new selection must replace unsupported roots with independently
  sourced positions that have at least two distinct executable plans, selected
  before outcomes are observed.
- The fit joins `sourceGameId` only from the frozen selection for in-memory
  source-game clustering; raw combined label records remain unchanged. The
  post-validation confidence audit hash is
  `990fc3f2e2cfca3481645382b45a0ff72f124a95ffae27af846bd3a62ddb060d`.
- Focused ranker-v2 tests pass (17); the full Python suite previously passed
  (538, with two dependency deprecation warnings), TypeScript typecheck,
  strategy-contract v1.2 validation, and `git diff --check` pass. No collector,
  supervised training, PPO, continuous service, or promotion was started. The
  next evidence gate is a new frozen selection that prequalifies at least two
  distinct executable plans per root and balances independent source games
  across archetypes and both policy families. Preserve v17 artifacts; do not
  recollect the same single-candidate positions or infer acceptance from the
  small holdout cells.

### Candidate-qualified pool audit and independent-data gate (2026-09-29)

- The ranker report was refit and reverified after adding exact per-fold
  training-position hashes to its coverage receipt. Iteration 4 report hash is
  `9b979a0e50125498997587e6613b45c66b41bc18f855def7b2d79e8f0ccc4f09`;
  acceptance remains `insufficient`, with 76/79 train and 18/18 development
  positions, but all seven holdout tests now report measured metrics and
  auditable 95.0%–97.4% train coverage. This does not enable distillation or
  PPO because the complete-train gate remains unmet.
- A no-rollout candidate-support audit covered all 250 frozen v13 positions
  per policy family. Python: 231 supported, 19 unsupported, zero supported
  roots without a complete plan, and 221 roots with at least two plans. Its
  support-report file SHA-256 is
  `1d112cee351276567f9f60daa70bb1b2bca54a023c7b60fa1509f5e87d205583` and
  report hash is `5f2f7501ef12c596675d82ece21696541a8761a701d8368dd71bbcea816993eb`.
  TypeScript: 235 supported, 15 unsupported, zero supported roots without a
  complete plan, and 231 roots with at least two plans. Its report file SHA-256
  is `72f27d1663bc7532a0d42effbe3082118cbb59f222ba3f46ce23f32882900936` and
  report hash is `a8fd45c41e3f7d0e329a50000ab0215ac18f1dff3c7f396a1a0df6bbbc068e60`.
- Future frozen selections now require at least two distinct complete
  executable plans per root. A draft selection from the existing v13 pools
  has 40/9 Python train/development and 39/9 TypeScript train/development
  source games (selection hash
  `08a1221e0cb8c468805078111252273d1eed81630d0fa748c4f9773845ea0761`). It
  retains 1,200 / 344 Python train/development candidates and 865 / 216
  TypeScript candidates.
- The draft is **not independent expansion**: it reuses all 97 v17 source-game
  IDs and 74/97 exact position hashes. It contains only one or two development
  positions against Raging Bolt per family. Therefore it cannot be used to
  claim new game-level training expansion or sufficient per-opponent
  development evaluation. A fresh source-game pool with a new seed namespace
  remains the route for broader training coverage; v17 and v13 remain
  immutable. No games, rollouts, labels, training, PPO, or service work were
  started by these support audits.
- Verification after the coverage-receipt and selection-eligibility changes:
  focused ranker/distillation tests pass (23), orchestration tests pass (23),
  and the full Python suite passes (541, two existing dependency warnings).
  TypeScript typecheck, strategy-contract v1.2 validator, and `git diff --check`
  pass. The next step requires separate authorization for a fresh 60-game
  coverage-only pool (30 games per policy family: two games per each of the 15
  archetype matchup cells, using a new namespace); this creates positions only,
  not macro rollouts or labels. Runtime is unknown until measured. The prior
  no-new-collector boundary remains in force until the user approves this
  exact scope.

### Reserved heldout positions frozen for future evaluation (2026-09-29)

- The v13 Python and TypeScript source pools already include heldout partitions
  that v17 deliberately left unused. A separate evaluator-only selection now
  freezes every supported heldout position with at least two complete plans,
  rather than selecting one position per game. It contains 34 Python positions
  from nine games and 36 TypeScript positions from nine games, spanning all
  five target decks and all five opponent archetypes. The selection hash is
  `b265bb6d2d24b9577bc48dedc68a1fc779b99fb72648e8c279edd1b1a3b01a49`; file
  SHA-256 is
  `2c1acd71e7aefd7b2a078c33e0d52f88aa5a278e5f76734d06544823728947cf`.
- Both heldout pools share the exact v13/v17 identity; their dataset row hashes
  match the candidate-support reports. Their nine source games per family have
  zero overlap with the corresponding v17 train/development game IDs. The
  evaluator selection is explicitly `trainingEligible: false`; rows remain
  unlabeled and no rollouts or model evaluation were performed.
- This removes fresh-game collection as a prerequisite merely to create a
  game-disjoint heldout position set. An external heldout evaluator and a
  separately authorized label-collection run are still needed; repeated
  positions from each game must be clustered by source game. This does not
  repair the ranker's 76/79 train coverage, establish a policy-family blind
  test, or satisfy promotion's fresh seed namespace. Reassess the previously
  proposed 60-game coverage-only pool as a training-coverage expansion, not as
  the only way to obtain heldout positions.
- The full Python suite passes 541 tests, including the new heldout-selection
  isolation test. TypeScript typecheck, strategy-contract v1.2 validation,
  stage-gate and selection JSON parsing, and `git diff --check` pass. No games,
  macro rollouts, labels, ranker fit, training, PPO, or service were run.

### Heldout macro-ranker evaluator implementation (2026-09-29)

- Added an offline evaluator for a verified frozen macro-ranker artifact. It
  binds heldout labels to the separately frozen all-supported selection and
  exact v13 datasets/support audits; checks run-manifest and record hashes,
  actor-visible position hashes, heldout-only rollout seeds, legal candidate
  roots, candidate/outcome consistency, collector compatibility, and disjoint
  train/development source-game and position identities.
- Reports source-game-clustered top-1 relative regret, top-three recall, and
  pairwise ordering with per-family/deck/opponent summaries. The immutable
  report is descriptive only (`trainingEligible: false`,
  `automaticPromotion: false`). It is not a policy-strength or promotion gate.
- Added tests for heldout selection rejection, source-game/split leakage
  rejection, immutable report behavior, heldout seed namespace isolation, and
  actor-visible legal-root enforcement. Focused validation passed; the full
  Python suite passes (549 tests, two existing dependency deprecation warnings).
  TypeScript typecheck, strategy-contract v1.2 validation, selection/stage-gate
  JSON parsing, and `git diff --check` pass. A first full-suite run correctly
  caught an accidental change to the historically hashed train/development
  collector; that edit was removed, and the exact audited collector-surface
  hash again matches `a3f0225f...e0811f9`. The original macro adapter hash also
  matches both frozen v13 support reports.
- No heldout label collection, model fitting, games, training, PPO, autonomous
  operation, or promotion was started. The evaluator cannot produce a result
  until separately authorized heldout-only labels and compatible frozen
  ranker artifacts are supplied.

### Heldout-only macro-label collection harness (2026-09-29)

- Added a separate resumable runner and CLI for the frozen 34-position Python
  and 36-position TypeScript heldout selections. It validates exact runtime,
  dataset, support, candidate-generator, and selection identities before work;
  uses an independently hashed heldout seed implementation; checkpoints after
  each common-random-number seed; preserves terminal/truncated/error outcomes;
  and publishes immutable result files and a checksummed run manifest.
- The historical train/development collector source surface and `macro.py`
  adapter remain byte-identical to their audited hashes; heldout behavior lives
  in separate modules. Unsupported positions are retained as explicit records
  and cannot silently count as scored evaluation positions.
- Adaptive sampling, common seeds, interrupted-seed resume, identity mismatch,
  and unfinished-outcome accounting pass using deterministic fake rollouts.
  The complete focused heldout collector/evaluator suite passes (10 tests),
  and the full Python suite passes (551 tests). TypeScript typecheck,
  CLI help, strategy-contract v1.2 validation, JSON parsing, and diff checks
  pass. No real engine rollout, heldout label, ranker fit, or training run was
  started.
- Separate user authorization is still required before invoking the collector.

### Goal continuation check (2026-09-29)

- The user authorized resuming the existing frozen v17 label checkpoint. A
  read-only artifact check found that it is already finalized: Python has 49/49
  positions and TypeScript has 48/48, with immutable manifests and no remaining
  per-position progress checkpoints. There was nothing to resume, and no
  completed result was replaced.
- Implemented a distinct heldout-only collector and verified it offline through
  the real evaluator loader using a fake engine. The full Python suite passes
  (551 tests); TypeScript typecheck, strategy-contract v1.2 validation, CLI
  help, and `git diff --check` pass.
- The 70-position heldout selection remains unlabeled. This confirmation only
  applied to resuming v17, which is complete; it did not authorize a new
  heldout rollout collection. Gate values remain unchanged: heldout evidence
  insufficient, blind opponent-family evidence insufficient, PPO disabled,
  continuous operation disabled, and trusted promotion disabled.

### Heldout per-seed evidence reconciliation (2026-09-29)

- Each candidate label now persists its exact attempted seed-index prefix and
  per-index finished/truncated/error outcome (including score or typed reason
  and observed decision count). The heldout evaluator verifies the full sample
  set, status totals, unfinished reasons, decision-count distribution, and
  mean score against the aggregate label before ranking it. This makes the
  resulting evidence auditable at seed granularity and gives future progress
  reports actual completed extension indices instead of inferring them from
  sample totals.
- Added a tampering test that removes a per-seed receipt and recomputes record
  and manifest hashes; the evaluator still rejects the incomplete evidence.
  Focused heldout collector/evaluator tests pass (10). No engine rollouts or
  labels were started; previously completed v17 evidence is unchanged.

### Heldout evaluator input readiness audit (2026-09-29)

- Independently verified the frozen ranker-v2 iteration 4 portable model/report
  pair. Its selection hash and selection-file checksum match the v17 frozen
  train/development selection; all 97 training positions/source-game clusters
  are disjoint from the 70 reserved heldout positions. Both heldout datasets
  and support-report hashes reconcile with the frozen selection.
- This audit found the runbook supplied `macro-support-v13-generalist` report
  paths, while the frozen heldout selection is bound to the audited
  `macro-support-v13-python-all` and `macro-support-v13-typescript-all` report
  files. Updated the evaluator and collector examples to the exact matching
  support reports. This is a documentation/input-readiness repair; no
  collection command or engine rollout was run.
- Stage status is unchanged: ranker acceptance remains insufficient (76/79
  training positions), heldout labels are absent, and PPO, continuous
  operation, and trusted promotion remain disabled.

### Read-only heldout collection progress interface (2026-09-29)

- Added `progress-heldout-macro-labels`, bound to the frozen family selection.
  It validates immutable result records and manifest hashes, validates active
  checkpoint identity/hash and per-seed outcome receipts, and reports selected
  versus finalized positions, in-progress/not-started positions, completed
  initial/extension indices, highest extension index, active candidate count,
  active sample-count range, and outcome totals. It never edits checkpoints,
  starts a collector, or claims process liveness from file state.
- Two focused tests pass for real adaptive-checkpoint reporting and tampered
  per-seed state rejection. The selection-aware progress command remains
  separate from the previously completed v17 run and does not enable any
  training or promotion gate.
- Final verification after the read-only progress CLI change: all 555 Python
  tests pass (two dependency deprecation warnings); TypeScript typecheck,
  strategy-contract v1.2 validation, CLI help, and `git diff --check` pass.
- A completed-manifest integration fixture verifies finalized/unsupported
  position counts and rejects a self-consistently rehashed but false count.
  Heldout progress tests now pass (3).

### Game-balanced heldout ranking summary (2026-09-29)

- Heldout positions are not evenly distributed across source games. The prior
  headline interval resampled source-game clusters but then averaged all
  position rows, allowing games with more selected positions to carry more
  weight. The headline top-1 regret and interval now average per-game means
  equally; a separately named position-weighted clustered estimate remains for
  comparison. The report also declares the weighting method.
- An uneven-cluster fixture (three positions from one game and one from
  another) verifies the game-balanced mean is 0.5 while the supplementary
  position-weighted mean is 0.25. No heldout labels or rollouts were used.

### Fresh-state heldout progress check (2026-09-29)

- The read-only progress command now accepts a not-yet-created output directory
  and reports a valid zero-progress snapshot without creating files. Focused
  progress tests pass, including the no-directory-creation assertion.
- Checked the frozen selection for both families: Python has 0/34 finalized
  positions and TypeScript has 0/36; both have zero checkpoints and report all
  selected positions not started. The shared selection hash is
  `b265bb6d2d24b9577bc48dedc68a1fc779b99fb72648e8c279edd1b1a3b01a49`.
  The CLI's process status is `unknown`; this filesystem snapshot does not
  establish whether a collector process is running.
- No labels were collected and no PPO, continuous operation, or trusted
  promotion gate was enabled. The heldout label authorization question remains
  unanswered, so collection was not started.

### Frozen heldout collection preflight (2026-09-29)

- Ran the no-write preflight against both approved policy-family datasets and
  support reports. Both matched the frozen selection hash
  `b265bb6d2d24b9577bc48dedc68a1fc779b99fb72648e8c279edd1b1a3b01a49`, the
  active runtime identity, candidate-generator identity, and frozen 16-to-64
  rollout settings. Each reports a fresh output path, zero existing results,
  zero checkpoints, `writesArtifacts=false`, and `startsEngine=false`.
- Python: 34/34 positions supported, 854 candidate plans, 13,664 minimum and
  54,656 maximum candidate-seed rollout work units; rollout identity
  `e613ec76b3e8ff620c17502a9dffe5cf3a89c55c43fd9a93e6888d63a4fb96bb`.
  TypeScript: 36/36 supported, 1,167 plans, 18,672 minimum and 74,688 maximum
  work units; rollout identity
  `d04de2c85d826053ba2640a3e4dd1240e7e13e195dfa7c35447c22236a61bffe`.
  These are candidate-seed work-unit bounds, not elapsed-time estimates.
- No simulation, labeling, or artifact write occurred. Collection remains a
  separate authorized action; ranker training coverage remains 76/79 and the
  existing PPO, continuous-operation, and trusted-promotion gates stay closed.

### v19 training-root repair reproducibility audit (2026-09-29)

- Rebuilt the v19 proposal from the frozen v17 parent selection, both source
  datasets, and both support reports into a temporary audit location. Its full
  selection hash exactly matches the committed draft:
  `6d13ae378f8078d09636d0636a950c913d312c3afd5ee40cf1c6df628ce61e05`.
- The proposal replaces exactly three unsupported single-candidate training
  roots: two Python and one TypeScript. Each replacement is from the same
  source game and has at least two complete supported candidates (40, 70, and
  7 respectively). It retains 38 pinned training labels per family, preserves
  all 18 development positions, and selects no heldout position.
- This proves the draft is reproducible and source-game coverage is repaired;
  it does not create the three replacement labels or change ranker acceptance.
  The repair proposal remains unlabeled, and training/promotion gates remain
  closed.

### No-write v19 repair-label preflight (2026-09-29)

- Added `preflight-macro-label-training-repair`. It validates the draft hash,
  immutable parent-selection hash/file receipt, unchanged development split,
  exact same-source-game train replacements, runtime identity, dataset/support
  hashes, candidate-generator receipt, minimum candidate support, frozen
  collector settings, and an empty/fresh output path without initializing an
  engine or writing a directory. A re-hashed draft with a false source-game
  receipt and symlinked source evidence are rejected. All four focused tests
  pass.
- Both real family preflights passed with selection hash
  `6d13ae378f8078d09636d0636a950c913d312c3afd5ee40cf1c6df628ce61e05`.
  Python verifies 2 replacements / 110 plans (1,760 initial and 7,040 maximum
  candidate-seed work units); TypeScript verifies 1 / 7 (112 and 448). Both
  use initial 16, maximum 64, extension batch 8, horizon 500, 60,000 ms per
  rollout, eight workers, and the parent run's `position-hash-list` method with
  no split filter. Both output states are fresh and report no ETA.
- The runbook now lists the exact hashes/settings for the separately
  authorized collector commands. No labels or engine rollouts were started;
  this preflight does not grant collection authority or change any gate.
- Final verification after parent-lineage and filesystem-input hardening:
  565 Python tests pass
  (two existing dependency deprecation warnings); TypeScript typecheck,
  strategy-contract v1.2 validation, JSON parsing, and `git diff --check` pass.
- After the metric update, the complete Python suite passes (555 tests, two
  existing dependency deprecation warnings); TypeScript typecheck,
  strategy-contract v1.2 validation, and `git diff --check` pass.
- A read-only join of the frozen heldout selection and matching support audits
  confirms 34/34 Python and 36/36 TypeScript positions have at least two
  complete plans: 854 and 1,167 plans respectively. At the frozen 16-to-64
  allocation, that is 32,336 initial and 129,344 maximum candidate-seed
  rollout work units before adaptive pruning. This yields no elapsed-time
  estimate, and no simulations were launched.

### Heldout collection preflight (2026-09-29)

- Added `preflight-heldout-macro-labels`, sharing the exact frozen-input and
  settings construction used by the collector. Its offline fixture verifies
  that preflight and collection resolve to the same rollout identity while
  preflight creates no output directory and never constructs an engine pool.
- Ran preflight against the actual frozen artifacts. Python identity
  `a487942f…52f7063` verifies 34/34 selected positions and 854 candidate plans
  (13,664 initial, 54,656 maximum rollouts). TypeScript identity
  `2d38f817…07a35ee` verifies 36/36 positions and 1,167 plans (18,672 initial,
  74,688 maximum). Both report elapsed-time estimate `unknown`, `startsEngine`
  false, fresh output state, and zero result/checkpoint files. Both target
  output directories remain absent.
- Final verification after the preflight/collector refactor: the full Python
  suite passes (555 tests, two existing dependency deprecation warnings),
  TypeScript typecheck passes, strategy-contract v1.2 validation passes, and
  `git diff --check` passes. No engine simulations were started.

### Heldout collector concurrency guard (2026-09-29)

- Added an advisory operating-system lock around each heldout output directory.
  A second process targeting the same output fails before input verification,
  engine construction, or sampling; a process exit releases the lock while
  preserving resumable per-position checkpoints.
- A subprocess test confirms a separate process cannot acquire the lock held
  by the test process. The focused heldout collection/progress suite passes
  (8 tests). No heldout runs, labels, or engine simulations were started.
- Refreshed read-only preflights after the source change verify the same frozen
  selection (`b265bb6d…b01a49`) and work bounds: Python 34 positions/854 plans
  (rollout identity `e613ec76…a4fb96bb`), TypeScript 36/1,167
  (`d04de2c8…6a1bffe`). Both output states are fresh; no runtime estimate is
  available, and neither preflight created output or started the engine.
- Full Python suite passes (556 tests, two existing dependency warnings),
  TypeScript typecheck and strategy-contract validation pass, and
  `git diff --check` passes. Existing frozen v17 artifacts and identities are
  unaffected; no label collection was started.

### v17 training-coverage repair feasibility (2026-09-29)

- The v17 ranker has 76/79 comparable training positions because three
  selected roots have only one executable candidate. A read-only join of the
  immutable v17 selection, v13 candidate-support audits, and the existing
  candidate-qualified selector found same-source-game replacements for all
  three: Python Crustle vs Raging Bolt (`115cda2e…f54ae9` →
  `b6b4e613…c43cd11`, 40 complete plans); Python Raging Bolt vs Grimmsnarl
  (`92d06e83…06e9580` → `e3e93f9f…b88056a`, 70); TypeScript Grimmsnarl mirror
  (`1eece6ac…a7439b85` → `84b70000…2503bd`, 7).
- Added and generated the draft-only CLI `freeze-macro-label-training-repair`
  and selection artifact
  `macro-label-selection-v19-train-coverage-repair-draft-2026-09-29.json`
  (selection hash
  `6d13ae378f8078d09636d0636a950c913d312c3afd5ee40cf1c6df628ce61e05`). It
  pins all 76 currently comparable v17 training roots, replaces only the
  three unsupported roots, preserves 40/39 train source games, and leaves all
  18 v17 development roots exactly unchanged. The replacements contain 117
  candidate plans total (1,872 initial and 7,488 maximum 16-to-64
  candidate-seed work units before pruning). No candidate rollouts or labels
  were produced.
- This is a repair feasibility result, not an accepted selection or ranker
  result. The current ranker-input merger binds every record to one exact
  selection hash and expects full family runs, so a separately reviewed
  lineage-aware supplement/merge path is still needed before collecting or
  combining these three replacement positions. Do not rewrite v17 artifacts.

### Training-repair merge and real v17 provenance recheck (2026-09-29)

- The prior bullet describes the state before the lineage-aware merge was
  implemented. The v19 draft has three replacements **total** (two Python,
  one TypeScript), not three per family. The merge retains 76 usable parent
  training records, replaces the three single-candidate records, and keeps all
  18 original development records. Its integration test verifies this lineage,
  preserves the original run directories, and models the actual collector
  record schema, where `sourceGameId` is established by the hashed selected
  dataset row rather than duplicated in each label record.
- The previous v17 combined bundle predates the current explicit source-run
  compatibility receipt and is rejected by the current strict ranker-input
  loader. The immutable Python and TypeScript v17 family runs were re-combined,
  without relabeling or editing either source run, into the ignored local view
  `artifacts/learning-mind-v1/combined-macro-labels-v17-provenance-recheck-2026-09-29`.
- The current combiner verified source-record checksums and exact selection
  coverage, preserving 49 Python and 48 TypeScript positions and 79 train / 18
  development positions. The strict ranker-input loader accepted all 97
  records and both distinct rollout identities. The receipt classifies the
  collectors as `audited-same-collection-surface`, with surface hash
  `a3f0225ffb11a69b87b6ee786dcaaab4e5ebe484b333424fc88bdee61e0811f9` and
  source commits `cc3802e38c9e8b41e49196309bf03976eac7251b` and
  `45620268a1f085184b58d834c0a547164c507649`.
- Derived manifest hash:
  `6f3c836e949e014dfb2db631b8e94599b9cc168418288201e6a9e0c155f1beb6`;
  manifest-file SHA-256:
  `18ef38032027c16506cdf7f9a07e7702f4098f5eaed08e2061a5165ceec50904`.
  This is format/provenance validation only. Ranker acceptance remains
  `insufficient` (76/79 comparable training positions); heldout labels remain
  uncollected, and PPO, continuous operation, and trusted promotion remain
  disabled.
- The old iteration-4 model report no longer verifies against the current
  evaluation implementation hash, so it was not treated as current evidence.
  A new iteration-5 research ranker was fitted from the re-derived v17 bundle
  and fresh analysis-only confidence report, then passed the current portable
  artifact verifier. Model SHA-256 is
  `3c659dd6f993127d99acd354d285f2f60d2253a5e64a068cde4a1de0fa1285b9`;
  report hash is `c8625885114f2afed49ccb789c482b20fa55c9a61e79437f77e1641f98d8abb4`.
  It deterministically matches the prior model bytes, but now binds the current
  input, confidence-audit, and evaluation implementation receipts. Development
  mean top-1 relative regret remains 0.0817 (18 independent games; 95% source-
  game cluster interval 0.0365–0.1418); training remains 76/79 and overall
  acceptance is `insufficient`. This is a provenance/verifiability refresh,
  not a model-strength gain or permission to distill.

### Heldout evaluator protocol binding (2026-09-29)

- The current heldout evaluator now requires each label-run manifest's candidate
  generator version to match the frozen candidate-generator identity, and
  enforces the settings used by the reviewed v2 preflights: 16 initial / 64
  maximum rollouts, extension batch 8, horizon 500, 60,000 ms per rollout,
  eight workers, the heldout seed namespace, staged allocation v3, frozen
  heldout selection, and heldout split. Equal-but-altered settings no longer
  pass as the same evaluation.
- Heldout evaluation reports now pin their evaluator source SHA-256:
  `ae72f5000aa4a247d54ae95eb39679eff20f2f5cef9347bc6e3153a1841b9c6c`.
  Tests cover altered generator version, altered settings, and current
  collector-to-evaluator fixture integration with a fake pool. The actual
  preflight starts no engine, and no heldout labels were collected.
- Verification: 560 Python tests pass (two existing dependency deprecation
  warnings); TypeScript typecheck, strategy-contract v1.2 validator, and
  `git diff --check` pass. This improves evidence reproducibility only; the
  heldout result is still absent, ranker acceptance is still insufficient,
  and PPO/continuous operation/trusted promotion remain disabled.
- Added `verify-macro-ranker-heldout-evaluation`. Given the report, frozen
  selections, ranker files, and original family runs, it replays the descriptive
  evaluation from those saved inputs in a temporary directory and requires an
  exact report match. It independently recomputes aggregate metrics from
  per-position rows and rejects a re-hashed, altered summary. A tampering test
  recomputes the outer report hash after changing regret and is still rejected.
  The command has no engine or training path; its temporary output is removed
  when verification ends.
