# Competitive rules implementation status

The registry includes five competitive mains, five training variants, five heldout variants, and five retained historical starters. All competitive manifests are structurally checked and hashed. Their external legality and full interaction audits remain incomplete, so `validation.trainingEligible` stays false. `formats/rules-coverage.json` distinguishes focused fixtures from execution-only coverage.

## Environment and choices

Selections for combinatorial prompts are staged without the former 256-action cutoff. Each prefix has append/undo options and a validated Finish. The journal records stages, and completed effects clear stage state even when upstream prompt IDs are reused. Prize choices address the upstream compressed remaining-Prize list. Setup, callbacks, random events, first-player overrides, and complete 15-list games have regression checks. Simulator-only `Math.random` draws are now included in private replay chance events.

Opponent hands, deck order, Prize identities, deck manifest IDs, and future chance stay out of player observations. Artwork URLs come from optional pinned `formats/card-art.json`; missing artwork has a text fallback.

## Search interface additions

`search` accepts optional `knownOpponentDeckId` for an explicit laboratory session, `priorRevealedCards: string[]` containing legitimately revealed printed IDs from prior games, and `rootPriors: {actionId, probability}[]`. Presence across games is a lower bound, not an accumulated copy count. Exact heldout lists enter only explicit known-list mode.

The default registry is the ten main/training lists. Bounded unknown hypotheses repair archetype mains against observed cards, preserving structural deck construction; they do not read heldout manifests. When compatible exact lists exist, 15% prior mass is assigned to unknown variants. `hypothesisWeights` reports the actual normalized distribution. These hand-selected priors are experimental.

Information-set search can use normalized learned root priors through PUCT; deeper choices and leaf evaluation still use the baseline heuristic. Rollout scores are not calibrated outcome probabilities. Equal-budget test calls do not establish stronger play.

A player's legal deck search infers the multiset of their remaining Prizes; only their own hand and publicly known zones are subtracted from their own decklist. Ordered top cards constrain independent sampled decks. Resolved own prize pickups reconcile the inferred multiset when the transition is unambiguous. Stale Prize knowledge, opponent hand revelations, unsupported markers, and other unmodeled knowledge refuse search explicitly. Search needs more knowledge-history coverage before broad research claims are justified.

## Verification commands

```
npm run engine:build
npm run typecheck
npm run test:engine
```

Tests are bundled as CommonJS with esbuild, matching production. Under Node 24, directly mixing ESM tsx tests with the vendored CommonJS package loaded duplicate CardManager/effect classes; the bundled runner prevents those false failures.
