# Assignment 3: visual play, teaching and analysis

Repository: https://github.com/capisz/pokemon-tcg-stockfish. Canonical checkout: `/Users/admin/Documents/ChatGPT/Pokemon Ai project`. Start an isolated worktree from the latest reviewed `codex/competitive-integration` commit. The three original work branches are preserved as implementation history; do not restart from their old foundation. Inspect status and preserve unrelated edits.

Read `contracts/COMPETITIVE_V2.md`, `docs/COMPETITIVE_IMPLEMENTATION.md`, `PRODUCT.md` and `DESIGN.md`. Use the approved familiar 2D table and preserve existing replay navigation. Do not restart design discovery.

## Ownership

Own React, FastAPI, live-match persistence/session modules, guide/teaching modules and browser/session tests. Assignment 2 supplies transport/storage/training services; assignment 1 supplies legal rules, observations and search. The lead owns common contracts, integration, and attributed `research/curriculum.json`.

## Required behavior

Build a playable 2D table with Active/Bench Pokémon, expanded benches, attachments, damage, conditions, hand, prizes, discard, Stadium and inspectable text/art. Derive all actions from validated prompts; use a readable fallback for missing art rather than inventing assets.

Add untimed best-of-three against the engine. Use closed lists by default, explicit known-list laboratory mode, and a shared maximum 120 seconds of engine thinking per turn across all prompts. Persist accepted decisions before acknowledging. Check revisions and idempotency keys. Recover identical seeded games after restart; incompatible builds pause rather than silently alter history. Engine failures or resource limits never award wins.

Keep seeds, true opponent lists, hidden actions and private replays out of live responses and linked endpoints. Carry only legitimate revealed information between games. Benchmark models stay frozen and hints/takebacks/analysis are unavailable until the match ends. Practice supports study. Provide concession, next game, reconnection and bookmarks.

Import private written guides and supplied video/transcript material with author, source hash, page/timestamp, format and deck version. The initial PDFs are already privately extracted into `data/competitive/`; their source metadata and twenty draft families are in `research/`. Those families contain sixty controlled variations and permanent train/validation/test partitions. They are not yet reviewed demonstrations.

Link a family variation to a reproducible legal position, validate candidates, and record acceptable alternatives plus conditions that reverse the preference. Only reviewed, rules-validated concrete training positions may be admitted by assignment 2. Preserve page metadata in retrieval. Local Qwen drafts remain hypotheses. The review queue has at most ten items and favors disagreements, uncertainty, suspected rule errors and discoveries.

Show signed engine advantage units, learned baseline/components/interactions, calibration status, conditional lines, uncertainty, coverage and version. Compare played actions using pre-decision information only. Keep unavailable probabilities and luck estimates unavailable.

## Completion evidence

Exercise a complete human match, reconnect/restart, duplicate/stale submissions, private-information isolation, bookmark/annotation/review and post-match analysis using the real API. Check laptop and narrow layouts, keyboard/focus, reduced motion and absent card art. Screenshots must show real data.

Deliver scoped commits, actual commands/results, screenshots and limitations. Branch pushes and draft PRs are authorized; merges/deployment remain user-controlled. No paid services, automatic large model downloads, or private source contents in Git.
