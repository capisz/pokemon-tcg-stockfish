# Assignment 1: competitive rules and search

Repository: https://github.com/capisz/pokemon-tcg-stockfish. Canonical local checkout: `/Users/admin/Documents/ChatGPT/Pokemon Ai project`. Work on `codex/competitive-rules` in its isolated worktree, or create an isolated worktree from the latest agreed integration commit. Inspect status and preserve unrelated edits.

Read `contracts/COMPETITIVE_V2.md`, `docs/COMPETITIVE_IMPLEMENTATION.md`, and the latest implementation status before editing. The shared foundation is commit `f6e9ac1`; newer integration commits may already contain completed work. Continue from current evidence rather than restarting.

## Ownership

Own the TypeScript environment/search, vendor patches and provenance, deck/format manifests, catalogue/build scripts, and engine tests. The learning assignment owns Python transport/storage/training. The play assignment owns FastAPI, live sessions, guides/teaching, and React. The lead owns shared contracts and combined verification. Communicate contract changes before relying on them.

## Required behavior

Register five exact competitive main lists, one training variant and one held-out variant per archetype. Preserve historical engineering decks separately. Main lists are the user's Lucario/Hariyama, Grimmsnarl/Froslass, and Bolt/Kangaskhan lists; Dragapult guide page 48; Crustle guide page 27. Counts and sources belong in immutable manifests. Verify exact printings, release dates, legal reprints, and Standard legality as of September 17, 2026. Vendor presence alone is insufficient certification.

Complete legal-choice handling, including staged multi-card/target choices, without silently dropping actions. Preserve seeded chance, callback reconstruction, conservation, and branch independence. Search may sample actions, but the environment must allow every legal action. Unsupported states must be explicit and excluded from trusted training.

Audit protective effects, Energy suppression/removal, counters versus damage, item lock, recovery, simultaneous checkup knockouts, bench expansion/shrinking, draw triggers, and genuine deck-out. Record every vendor fix with before/after hashes. Test observed issues before changing behavior: Meowth bench entry after ability use, Special Red Card shuffle, Gwynn choosing fewer than two, Hammer tails, and Spiky retaliation when damage is prevented.

Preserve legally known cards and order (including Ciphermaniac) or refuse affected searches. Sample legal unknown-list variants consistent with observations, without consulting the true opponent list or future random events. Exclude held-out exact lists from agent hypotheses. Every simulated player acts from its own information. Compare rollout and information-set search at equal budgets and connect the learned evaluator through the agreed interface.

## Completion evidence

Run engine build, TypeScript checking, focused interaction tests, then the full engine suite. Record complete games for every registered list, observed failures, coverage and eligibility status. Verify indistinguishable hidden states produce identical player inputs and fixed-seed search results. Do not label unsupported or unverified decks fully validated.

Deliver commits, exact commands/results, engine/list hashes, coverage matrix, search-budget evidence, and remaining gaps. Commits, pushes to `codex/` branches and draft PRs are authorized; merges and deployment remain user-controlled. No paid services, Kaggle-derived restricted assets, or raw private guides in Git.
