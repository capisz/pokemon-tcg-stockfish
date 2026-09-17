# Implementation and evidence — September 17, 2026

This checkout implements a working local research loop. It does **not** establish a strong Pokémon player, exhaustive rules correctness, or completion of six weeks of research. All five archetypes, including Crustle, are supported by authored starter lists and exercised in real games. The next work is correctness and strategic quality, not adding infrastructure.

## Verified behavior

| Check | Observed result |
|---|---|
| TypeScript engine tests | 18 passed: deterministic replay, branch continuation, information isolation, five complete games, selected Crustle interactions, recovery, card conservation, search, and numeric shuffle validation |
| Python tests | 22 passed: terminal labels, immutable splits, held-out evaluation exclusion, storage, model decomposition, checkpoints, promotion, guides, API positions, and pre-decision review |
| TypeScript / React build | Typecheck and Vite production build passed |
| Actual browser workflow | Crustle vs Mega Lucario seed 17 completed; selected player view, trained analysis, sampled search, save and reopen position passed; no page errors; 390px viewport had no horizontal overflow |
| Baseline self-play | 25/25 games finished, one game for every ordered deck pairing; zero truncations or errors |
| Baseline evaluation | 50 finished games, 25 paired assignments, no excluded pairs; actual candidate/opponent going-first counts were balanced in this run |
| Pause/resume | A two-game run paused after game one and resumed to two finished games on the same build/configuration |
| Learned agent | Experimental checkpoint played a complete real game; this is an execution check, not a strength result |
| Data export | 2,997 decision examples in Zstandard-compressed Parquet, with whole-game partitions |
| Training | 38,532-parameter model, two epochs, local SQLite MLflow logging and checkpoint created |
| Device microbenchmark | About 1,388 CPU vs 600 MPS forward/backward steps/sec at batch 32 for the small model; CPU selected |
| Two-worker probe | 342 decisions across two completed Crustle games in about 5.0 s, about 68.4 decisions/sec; sampled aggregate Python/model + worker RSS about 529 MiB |

Memory excludes browser, API, operating system, and Ollama. It is a small sampled workload, not a worst-case promise. Device timings are short model microbenchmarks, not full training throughput. Two Python dependency deprecation warnings remain in the passing test output.

The 25-game/50-game experiments used source fingerprint `69a47ff0a07bc8db`. Later fingerprint `be0c29e9637ba8b3` adds illustrative conditional search lines only; rules and transition behavior were unchanged. Each evidence JSON preserves its actual engine identity. Raw replays and model artifacts stay under ignored `data/`.

## First experiments

Reproduce the main smoke corpus and training:

```sh
.venv/bin/python -m ptcg_lab.cli --data data/current-smoke selfplay --games 25 --seed 17092026 --max-decisions 800
.venv/bin/python -m ptcg_lab.cli --data data/current-smoke train --epochs 2 --max-positions 2000 --mlflow
.venv/bin/python -m ptcg_lab.cli --data data/current-smoke export-parquet data/current-smoke/trajectories.parquet
.venv/bin/python -m ptcg_lab.cli --data data/evaluation-smoke evaluate --candidate heuristic --opponent random --seeds 1 --max-decisions 500
```

The heuristic's mean result against the random policy was **0.74**, with a conservative 95% paired interval **[0.4684, 1.0]**. No champion was promoted. One pair per matchup is far too little to infer metagame performance or conclude that the heuristic is reliably stronger.

Crustle piloted by the candidate had mean results 1.0 vs Dragapult, 1.0 vs Raging Bolt, 1.0 vs Grimmsnarl, 0.5 vs Mega Lucario, and 0.5 in the mirror. Every one of those matchup intervals was [0,1]. These numbers verify coverage; they are **not matchup win-rate estimates** suitable for strategic advice. The reverse candidate/opponent deck assignments are separately recorded.

For the final small-model smoke run, training loss changed from 1.4099 to 0.9328. The held-out expected-result MSE was 0.1141 and outcome Brier score 0.2159 over only two test games/284 positions. Those positions are correlated; they are not 284 independent tests. Calibration had five games and remained insufficient, so W/D/L probabilities are withheld. Training loss reduction does not demonstrate stronger play. The checkpoint is `data/current-smoke/models/67ffd2a88d6a49f0b0ae7feffbac079a.pt`; it is deliberately experimental.

Both rollout and ISMCTS executed on the same three saved player observations at matched 1,000 ms budgets. The runs are recorded in `evidence/search-budget.json`. Sample counts are small and some candidates remain unvisited. All sampled leaf estimates in that probe used heuristic cutoffs. This is an implementation/performance comparison; it does not select a superior search method.

## Remaining acceptance gates

1. **Rules and competitive deck coverage.** Validate complete interactions and exact rulings with strategy review. The 256-choice cap is reported, but must be replaced with complete/lazy legal-choice handling before treating every game as trusted training data. The legal lists prioritize coverage rather than competitive ratios. No claim of covering all Standard cards is made.
2. **Search and beliefs.** Search currently supports stable whitelisted positions, excludes unsupported live callbacks, and lacks complete knowledge of previously revealed cards/order. Opponent beliefs use card compatibility, not a learned likelihood of strategic actions. Search uses untrained leaf values and is not yet a default self-play policy. Conditional lines are illustrative samples, not forced lines.
3. **Learning quality.** Policy targets currently clone recorded actions. Search-target distillation, champion/historical population scheduling, tactical demonstrations, replay-buffer eviction, and controlled ablations still need work. Train for strength only after simulator gates pass. Static resource features are coarse proxies; learn/refine defensive answers and remaining recovery with expert positions.
4. **Evaluation reliability.** Larger held-out opponent populations, matchup confidence, calibration, strategic regression review, and stronger baselines are required. Null probabilities and unavailable mistake estimates must stay unavailable until evidence supports them. Opportunity-loss estimates do not separately measure later luck.
5. **Guides and explanations.** Attributed import, lexical retrieval, and citation-validation tests work. Sentence Transformers and Ollama integration code exists, but their model weights were not installed and inference was not exercised. Reviewed guides have not yet supplied demonstrations. No generated strategy is accepted as ground truth.
6. **Operations.** JSON storage caps and per-worker heap caps are implemented, but they are not global memory/disk quotas. Long-run profiling, atomic Parquet export, streaming datasets, and a complete process-level quota policy remain needed before prolonged unattended training.

## Independent next assignments

The three files under `docs/handoffs/` are ready to paste into separate chats. They share interfaces and separate file ownership. Do not run competing large experiments simultaneously on the 16 GB Mac. None authorizes deployment, paid services, or an upstream contribution.

## Evidence files

- `evidence/selfplay.json`: seeds, deck and engine hashes, completed replay IDs.
- `evidence/baseline-evaluation.json`: all ordered matchups, confidence intervals, first-player counts, promotion decision.
- `evidence/training.json`: model/data hashes, whole-game split assignments, losses, metrics, calibration status.
- `evidence/parquet.json`: dataset row counts and partitions.
- `evidence/devices.json`, `evidence/memory.json`: bounded hardware measurements.
- `evidence/search-budget.json`: matched-budget executable search comparison.
- `evidence/resume.json`: successful game-boundary pause and resume.
- `artifacts/browser/desktop.png` and `mobile.png`: ignored local screenshots from the browser test.

The Git remote is `https://github.com/capisz/pokemon-tcg-stockfish.git`; local work is on `codex/local-research-prototype`. The supplied remote was empty when inspected. No commit, push, or deployment has been made.
