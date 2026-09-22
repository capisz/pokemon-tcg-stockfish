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

This evidence establishes implementation and representation parity, not playing
strength or autonomous improvement.
