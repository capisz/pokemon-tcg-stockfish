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

This evidence establishes implementation and representation parity, not playing
strength or autonomous improvement.
