# Assignment 2 — Learning, reproducible experiments, and evaluation

Continue the Pokémon TCG engine project in `/Users/admin/Documents/ChatGPT/Pokemon Ai project`. Repository: https://github.com/capisz/pokemon-tcg-stockfish. Inspect the real checkout, Git status, and latest implementation evidence before changing anything; preserve unrelated local work.

The objective is measurable learning and defensible ML research evidence on the user's M4 / 16 GB machine. Paid compute, hosted trackers, cloud fallbacks, and automatic model downloads are out of scope. Do not push or deploy without the user's authorization. A trained checkpoint is not a strong or promoted engine merely because training completes.

## Ownership and current implementation

Own Python features, model, dataset, training, self-play, evaluation, storage, and their tests under `src/ptcg_lab/` and `tests/python/`. Coordinate worker/API changes and `contracts/PROTOCOL.md` with the project lead. The rules agent owns `packages/engine/`; the analysis agent owns `web/` and `src/ptcg_lab/guides.py`. Coordinate shared imports and schema changes instead of overwriting their work.

Read `README.md`, `docs/ARCHITECTURE.md`, `contracts/PROTOCOL.md`, and `docs/IMPLEMENTATION_STATUS.md` if present. Inspect the actual experiment manifests under the local ignored `data/` directory before repeating a run.

The current foundation includes bounded resumable self-play, complete-game datasets, whole seed/deck-family splits, Parquet export, a compact PyTorch resource/value and legal-action model, optional local SQLite MLflow tracking, separate calibration data, and a paired evaluator over all 25 ordered deck assignments. The resource score sums learned terms, baseline, and interaction, then divides the raw logit by ln(2). The first policy objective clones recorded actions. Search distillation and a historical opponent curriculum are not yet established learning results. Evaluation artifacts are excluded from training by default. Tiny pipeline runs cannot establish strength or calibration.

## Next work, in order

1. Run the existing checks and local device benchmark. Verify the actual checkpoint feature schema, engine fingerprint, deck hash, and corpus before reusing any model. Preserve incompatible historical checkpoints as evidence; do not reinterpret old features or silently resume against changed data.
2. Make train/calibration/test family assignment persistent as the corpus grows. Keep paired seats and duplicate trajectories together. Reserve evaluation seeds independently and continue excluding evaluator-generated games from training. Demonstrate that adding new training data cannot move an old held-out family into training.
3. Run one bounded pilot using the commands below, then inspect completion rate, per-deck coverage, target distribution, memory, disk, and throughput. Fix invalid labels or data gaps before scaling. A cap, engine error, missing winner, or interrupted job is never a draw; only an actual rules draw receives 0.5.
4. Connect verified search targets to policy training through an explicit versioned dataset field. Retain the behavior-cloning baseline and compare against it. Add a reproducible mixture of current and historical policies instead of training exclusively against the newest checkpoint. Guidance from the guide agent may enter only as reviewed, attributable demonstrations, never raw prose or generated claims treated as game truth.
5. Evaluate strength separately from loss reduction. Use all ordered deck assignments with swapped seats, fresh seeds, fixed opponent/model hashes, paired uncertainty, per-matchup results, completion bias checks, and the measured going-first distribution. Swapped seats alone do not prove first-player balance. Compare against the installed champion and frozen baselines before promotion.
6. Calibrate and report resource-derived expected result separately from W/D/L. The current W/D/L gate requires sufficient held-out games and all outcome classes. Do not fabricate draws or call missing draws a calibrated zero probability. If introducing scalar temperature calibration, fit it on calibration families only, evaluate on untouched families, version the score semantics, and coordinate the frontend labels. Expose uncertainty and keep unsupported probabilities null.

## Bounded commands

Run from the project directory. Inspect existing runs before generating duplicates; use a new documented seed namespace if any example seed overlaps an existing corpus.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ptcg_lab.cli doctor --benchmark
.venv/bin/python -m ptcg_lab.cli selfplay --games 25 --seed 27092026 --max-decisions 800
.venv/bin/python -m ptcg_lab.cli export-parquet data/pilot-trajectories.parquet
.venv/bin/python -m ptcg_lab.cli train --epochs 3 --max-positions 10000 --device cpu --mlflow
.venv/bin/python -m ptcg_lab.cli evaluate --candidate heuristic --opponent random --seeds 1 --seed-start 1200000000 --max-decisions 800
```

The final command checks the evaluator pipeline; it is not a learned-model strength claim. Training prints a real checkpoint path. Substitute that exact absolute path as `--candidate` for a separately seeded learned-policy evaluation. Do not add `--promote` to a smoke experiment. Choose CPU or MPS from the measured local benchmark, not from the presence of Apple acceleration alone.

Self-play resumes with its run ID and identical original arguments. Training resumes with its checkpoint and compatible configuration/data hash. Stop after the pilot to assess results before increasing volume. Respect the two-worker, bounded-queue and local-storage safeguards; monitor actual process memory and all artifact disk use because a heap limit is not a system quota.

## Completion report

Deliver reproducible run manifests, exact dataset/model/engine/deck hashes, split checks, train and held-out metrics, complete/incomplete game counts, matchup and going-first coverage, calibration status, and measured performance/cost. Explain what was learned versus imitated and whether playing strength improved. Keep ML résumé claims limited to implemented and verified work; never call a small smoke model superhuman, optimal, fully calibrated, or a successful champion without evidence.
