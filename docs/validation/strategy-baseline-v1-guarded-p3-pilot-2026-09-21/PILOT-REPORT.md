# Strategy Baseline v1 guarded P3 pilot

Status: completed, measurement only.

The research-only P3 measurement adapter now applies the existing
`visible-repetition-v1` guard before checkpoint inference, resolves the selected
ID against the engine's original legal actions, and records the accepted choice.
It does not change the engine legal-action set, production behavior, checkpoint,
search, features, training, labels, or files under `data/competitive`.

## Verification

Focused adapter and existing guard tests:

```text
......                                                                   [100%]
6 passed in 0.13s
```

Frozen identities remained unchanged:

- Engine fingerprint: `twinleaf-adapter-0.1.0+bf3a9a0f133b99bd`
- Engine build SHA-256: `474549dda1d98fea5a7435c3abfdcccb4baa2663237ce8bb4aab99c40b36906e`
- Effective contract SHA-256: `c13d14d4af34dbab8d2f52c3755281fb57d9aa7755f582c3656668f7cb32954b`
- Guide checkpoint SHA-256: `1ff34f893540373ebb2da6433b3b67a0cdbcb2b711220cf08767cdeb85a08e6f`
- Crustle v3 list SHA-256: `f454a2c2f3bd329f6d75bb8344ba09000db93e220e8a07c3e0502e5a7ddd8343`
- Dragapult list SHA-256: `2e40a150bc968b9e67a9c7e57f6b9ebddb72591a41ae7e3a9f7b855db92318b9`

The original cells and seeds were reused. The P1, P2, and P4 game rows in the
merged result compare exactly equal to the preserved prior result; none was
rerun or changed.

## P3 results

| Cell | Seed | Status | Decisions | Guard-filtered | Seconds |
|---|---:|---|---:|---:|---:|
| cross-crustle-seat0-first0 | 219283457 | rules-terminal | 193 | 48 | 6.832 |
| cross-crustle-seat0-first1 | 219283458 | rules-terminal | 205 | 43 | 8.891 |
| cross-crustle-seat1-first0 | 219283459 | rules-terminal | 247 | 58 | 12.417 |
| cross-crustle-seat1-first1 | 219283460 | rules-terminal | 230 | 53 | 8.983 |
| crustle-mirror-first0 | 219283461 | rules-terminal | 243 | 15 | 19.707 |
| crustle-mirror-first1 | 219283462 | rules-terminal | 285 | 24 | 25.493 |
| dragapult-mirror-first0 | 219283463 | rules-terminal | 266 | 52 | 15.492 |
| dragapult-mirror-first1 | 219283464 | rules-terminal | 273 | 65 | 15.806 |

All 8 games finished; 0 truncated and 0 errored. The guard filtered 358
decisions in total. Each P3 decision carries its filtered flag and
`searchTargetCreated: false` in `pilot-results.json`.

P3's mean game time was 14.203 seconds and its median was 13.955 seconds. The
weighted 96-game P3 projection is 1,520.969 seconds, or 25.35 minutes.

## Complete projection and frozen plan

| Policy | Projected 96-game seconds | Projected minutes |
|---|---:|---:|
| P1 | 543.281 | 9.05 |
| P2 | 693.840 | 11.56 |
| P3 | 1,520.969 | 25.35 |
| P4 | 1,057.897 | 17.63 |
| Combined | 3,815.987 | 63.60 |

The honest complete projection exceeds 60 minutes. The frozen main-baseline
recommendation is therefore **192 games**, halving every cell before the run:
four games per cross-matchup cell and eight games per mirror cell for each of
the four policies. Pilot outcomes are timing data, not strength estimates.

The full Strategy Baseline v1 was not started.
