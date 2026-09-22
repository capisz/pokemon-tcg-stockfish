# Strategy Baseline v1 verification

Accepted on 2026-09-22 as measurement evidence only. These results do not establish playing strength.

## Frozen execution

- Gameplay harness commit: `2cb0e01d8d35840b95bdc54090fa6218f987e0a7`.
- 192 unique scheduled games: 48 each for P1, P2, P3, and P4.
- 192 finished; 0 truncated; 0 engine errors; 0 replacements.
- 48 matched cell/game seed slots, each shared by all four policies.
- 192 replay artifacts independently verified against both compressed and decoded SHA-256 values.
- Verified compressed replay size: 195,408,979 bytes.
- 24 tested probes cover all 23 effective principles.
- P3 recorded 2,043 guard-filtered decisions and zero search targets.
- P4 recorded 574 searched decisions, 9,513 fallback decisions, 612 search attempts, and 574 iterations. Per-decision fallback reasons remain in `games.jsonl`.

## Checks

- Strategy contract validator: approved v1.2 selected.
- Baseline, pilot, and decision-guard Python suites: 36 passed.
- TypeScript typecheck: passed.
- Focused pre-baseline engine rules: 12 passed.
- `git diff --check`: passed.
- Research TypeScript evaluator parity with the worker heuristic is covered by the baseline suite.
- Private-view isolation, deterministic seeds, identity rejection, resume behavior, replay hashing, Wilson intervals, and gap ranking are covered by the baseline suite.

## Finalization note

The games were run from the clean gameplay harness commit above. Two later research-only commits corrected report finalization without rerunning or replacing games: `337df8b` fixed the local `tsx` loader path, and `7102e46` added the required first-player structural-unmeasurability label.
