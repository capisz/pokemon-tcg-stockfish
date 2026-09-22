# Strategy Baseline v1

Status: completed measurement; no policy improvement was made.

This is a fixed 192-game simulator baseline against the approved v1.2 playbooks. Outcomes are context, not established playing strength.

## Frozen execution

- Harness commit: `2cb0e01d8d35840b95bdc54090fa6218f987e0a7`
- Engine: `twinleaf-adapter-0.1.0+bf3a9a0f133b99bd` / `474549dda1d98fea5a7435c3abfdcccb4baa2663237ce8bb4aab99c40b36906e`
- Effective contract: `c13d14d4af34dbab8d2f52c3755281fb57d9aa7755f582c3656668f7cb32954b`
- Guide checkpoint: `1ff34f893540373ebb2da6433b3b67a0cdbcb2b711220cf08767cdeb85a08e6f`
- Games: 192 scheduled; 192 finished; 0 truncated; 0 errors.
- Plan: 48 matched seeds per policy; four cross games and eight mirror games per cell.

## Measurement summary

- Mean/median game length: 216.1 / 221.0 decisions.
- Deck-outs: 0.
- P3 guard-filtered decisions: 2043.
- P4 searched/fallback decisions: 574 / 9513.
- First-player preference is structurally unmeasurable because the benchmark scheduler fixes first player rather than presenting a policy decision.
- W/D/L/unfinished Wilson intervals by perspective, seat, and first player are in `context-metrics.json`.

## Largest measured strategic gaps

| Rank | Policy | Probe | Failures / headline n | Severity | Gap score |
|---:|---|---|---:|---:|---:|
| 1 | P2 | crustle-fan-active-kangaskhan | 16 / 18 | 3 | 1.000 |
| 2 | P4 | crustle-fan-active-kangaskhan | 16 / 17 | 3 | 1.000 |
| 3 | P1 | crustle-fan-active-kangaskhan | 15 / 17 | 3 | 0.938 |
| 4 | P3 | crustle-fan-active-kangaskhan | 15 / 20 | 3 | 0.938 |
| 5 | P1 | dragapult-large-hand-judge | 20 / 24 | 2 | 0.833 |

Rates with fewer than 20 qualifying headline cases are marked insufficient in the machine-readable results.

## Teaching and review queue

- Graded four exact-compatible review records covering 2 unique positions.
- Preferred-action rate is unavailable because those reviews contain acceptable sets but no preferred-action annotation.
- Four stale Crustle records remain listed for human re-review and were not graded or changed.
- Proposed 4 unlabeled draft scenarios from the highest-ranked observed failures.

## Evidence

Tracked machine-readable files are under `docs/validation/strategy-baseline-v1-2026-09-21/`. Every private raw replay remains in ignored local artifacts and is checksummed by `replay-manifest.json`.

No engine, heuristic, search, feature, checkpoint, training data, teaching review, or file under `data/competitive` was modified. The next step requires a separate human choice among the measured gaps.
