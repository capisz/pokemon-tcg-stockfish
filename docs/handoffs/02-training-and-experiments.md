# Assignment 2: learning and portable execution

Repository: https://github.com/capisz/pokemon-tcg-stockfish. Canonical local checkout: `/Users/admin/Documents/ChatGPT/Pokemon Ai project`. Use the isolated `codex/portable-learning` worktree and inspect current status/evidence. The shared foundation is `f6e9ac1`; continue from subsequent integration work when present.

Read `contracts/COMPETITIVE_V2.md`, `docs/COMPETITIVE_IMPLEMENTATION.md`, and current implementation evidence. This assignment implements measurable learning, not a promise of superhuman performance.

## Ownership

Own Python transport, configuration, storage, self-play, datasets, features/models, training, evaluation, CLI, portable execution scripts and research tests. Preserve transport/storage service compatibility. FastAPI, live sessions, guides, teaching and React belong to assignment 3. Rules, registry and search belong to assignment 1. Shared dependency and interface changes go through the lead.

## Required behavior

Support macOS M4/16 GB and native Windows Ryzen/64 GB. Windows CPU is the baseline; do not depend on the likely RX 580. Diagnose actual hardware and benchmark Mac CPU/MPS. Start with two simulation workers. Monitor aggregate processes and all managed artifacts. Profiles begin at 8/40 GiB memory targets and 25/200 GiB managed data, keeping 20 GiB free per destination. Increase desktop workers only after measured benefit, up to `max(1,min(8,logicalCPUs-2))`; control library thread oversubscription.

Provide explicitly launched resumable self-play/training/evaluation cycles with bounded batches, safe interruption and checkpoints. No scheduler or automatic model download. Unknown results remain incomplete, never fabricated draws. Mix baselines, champion and historical opponents. Preserve behavior-cloning comparisons, add search policy/value targets and concrete reviewed demonstrations, and keep the network below two million parameters initially.

Exclude held-out exact lists, evaluation runs, and held-out guide families from training and agent hypotheses. Preserve permanent partitions as data grows. Stream compressed records and bound replay retention without deleting permanent benchmark evidence or the only recovery checkpoint.

Export/import immutable bundles with relative paths, identity hashes, partitions and checksums. Stage atomically; reject traversal, corruption, missing artifacts and split conflicts. Do not copy live tracker databases or raw private guides by default. Cross-device training continuation creates a linked experiment with explicit parent/device provenance.

Calibrate on separate complete-game families. Report learned resource score, calibrated W/D/L and sampled search values distinctly. Champion promotion requires a predeclared paired held-out comparison with a positive lower 95% confidence bound on aggregate improvement and investigated strategic regressions. Training loss alone is not strength.

## Completion evidence

Provide actual CLI commands and tests for pause/resume, continuous-run bounds, profiles, device continuation, corrupt/interrupted bundle rejection, streaming and immutable partitions. Exercise an actual small end-to-end run before scaling. Native Windows execution must be identified as unverified until run there; platform-aware code or mocks are insufficient proof.

Deliver commits, tests, manifests, hashes, resource measurements, matchup confidence and calibration status. Commits/pushes to `codex/` branches and draft PRs are authorized. Merges/deployment remain user-controlled; no paid services or cloud work.
