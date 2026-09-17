# Assignment 1 — Rules correctness and trustworthy search

Continue the Pokémon TCG engine project in `/Users/admin/Documents/ChatGPT/Pokemon Ai project`. Repository: https://github.com/capisz/pokemon-tcg-stockfish. Work from the actual local checkout and inspect Git status before editing; local work may be ahead of the remote. Preserve unrelated changes.

The objective is a reliable simulator and increasingly useful imperfect-information search for the five-deck research pool. This is a local M4 / 16 GB project with no paid services. Do not push, deploy, provision services, or expand the card pool unless the user has authorized that work. Do not replace the independent Twinleaf adapter with Kaggle competition assets.

## Ownership and starting point

Own `packages/engine/`, its tests, and the rules/search sections of the research documentation. Coordinate changes to `decks/`, `formats/`, the pinned `vendor/twinleaf/` source, and `contracts/PROTOCOL.md` with the project lead. Record every necessary vendor change in the existing provenance/patch record. The Python training/evaluation agent owns `src/ptcg_lab/`; the analysis agent owns `web/` and guide processing. Do not edit their files concurrently.

Read `README.md`, `docs/ARCHITECTURE.md`, `contracts/PROTOCOL.md`, and `docs/IMPLEMENTATION_STATUS.md` if present. Then inspect `environment.ts`, `choices.ts`, `belief-state.ts`, and `search.ts` under `packages/engine/src/`.

Current code contains five frozen 60-card engineering decks: Dragapult, Grimmsnarl, Mega Lucario, Raging Bolt, and Crustle. It implements seeded real rules, validated bounded legal choices, replay-based callback reconstruction, selected-view observations, stable public-position snapshots, flat rollouts, and experimental information-set UCB search. The deck pool is not a certified tournament metagame. Tests cover selected Crustle interactions, conservation, replay independence, and hidden-state search invariance; passing those does not certify every interaction.

## Next work, in order

1. Establish a fresh baseline with the commands below. Record the current engine fingerprint and exact failures before changing code. Read existing fixtures so you extend coverage instead of rewriting working checks.
2. Create an explicit card/interaction coverage matrix for this closed pool. Add the highest-impact missing fixtures first: damage versus counters, ability suppression/bypass, attack and evolution restrictions, search/discard/recovery choices, knockout/prize handling, retreat, failed draws, and long-game resource exhaustion. Include Crustle's non-ex counterplay. Fix demonstrated rules failures without silently broadening supported behavior.
3. Audit every choice family against the legal rules and document selective enumeration. Never let a cap silently become a claim of exhaustive legal search. Unsupported prompts or state must return a reason, not invent a move, winner, or draw.
4. Improve belief reconstruction's treatment of previously revealed information and known card order. The current sampler deliberately omits that history. Preserve legally known information or explicitly refuse the affected search; never inspect the real hidden hand, prizes, deck order, opponent deck identity, or original RNG future. Reconcile Python's displayed deck beliefs with the distribution actually sampled by search through an agreed interface.
5. Benchmark flat rollouts and information-set search at equal compute against the frozen heuristic. Report terminal versus heuristic-cutoff samples, excluded continuations, coverage of legal candidates, latency, and matchup results. Promote a search choice only when the measurement supports it. The current leaf evaluator is untrained; deterministic sampling and a search tree alone do not establish stronger play or equilibrium behavior.

## Checks and commands

Run from the project directory:

```sh
npm run engine:build
npm run typecheck
npm run test:engine
.venv/bin/python -m ptcg_lab.cli doctor
```

The JSONL worker is `packages/engine/dist/worker.cjs`; its request methods and observation/search types are defined in `contracts/PROTOCOL.md`. Use fixed iterations and sampling seeds for deterministic search comparisons; separately measure wall-clock behavior. Check that equal observable positions with different true hidden allocations produce equal public search inputs and equal sampled results. Verify replayed and hypothetical branches never mutate their source and that every simulated actor acts from its own observation.

If protocol changes are necessary, agree on them with the Python/API owner before changing payloads. Preserve schema compatibility or bump versions and reject incompatible historical artifacts explicitly.

## Completion report

Deliver the changed interaction matrix, focused fixes, exact commands/results, engine/deck fingerprints, equal-budget search evidence, and remaining unsupported cases. Link real local artifacts. Distinguish rules correctness, search execution, and demonstrated playing strength. Do not claim complete card coverage, calibrated winning probability, optimal play, or a public release from local tests.
