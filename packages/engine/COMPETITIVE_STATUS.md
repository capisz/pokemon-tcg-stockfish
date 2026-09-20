# Competitive rules implementation status

The registry includes five competitive mains, five training variants, five heldout variants, and five retained historical starters. All competitive manifests are structurally checked and hashed. Their external legality and full interaction audits remain incomplete, so `validation.trainingEligible` stays false. `formats/rules-coverage.json` distinguishes focused fixtures from execution-only coverage.

## Environment and choices

Selections for combinatorial prompts are staged without the former 256-action cutoff. Each prefix has append/undo options and a validated Finish. The journal records stages, and completed effects clear stage state even when upstream prompt IDs are reused. Prize choices address the upstream compressed remaining-Prize list. Setup, callbacks, random events, first-player overrides, and complete 15-list games have regression checks. Simulator-only `Math.random` draws are now included in private replay chance events.

Unobserved opponent hand cards, hidden deck order, hidden Prize identities, opponent deck manifest IDs, and future chance stay out of player observations. Legally revealed identities live in the entitled player's knowledge history. Artwork URLs come from optional pinned `formats/card-art.json`; missing artwork has a text fallback.

Actions expose revision-scoped `sourceRef` and `targetRef` from actual simulator bindings. Compound choices use `choiceRefs`. Staged prompts expose `selectedChoices`, limits, and Finish/Undo availability. `PokemonView.attachments` contains inspectable deduplicated Energy/Tool cards; `slotIndex` identifies the actual Bench slot in compact board observations. `choose({policy,seed?})` returns a current legal action without advancing the game or chance generator.

## Search interface additions

`search` accepts optional `knownOpponentDeckId` for an explicit laboratory session, `priorRevealedCards: string[]` containing legitimately revealed printed IDs from prior games, and `rootPriors: {actionId, probability}[]`. Presence across games is a lower bound, not an accumulated copy count. Exact heldout lists enter only explicit known-list mode.

The default registry is the ten main/training lists. Bounded unknown hypotheses repair archetype mains against observed cards, preserving structural deck construction; they do not read heldout manifests. When compatible exact lists exist, 15% prior mass is assigned to unknown variants. `hypothesisWeights` reports the actual normalized distribution. These hand-selected priors are experimental.

Information-set search can use normalized learned root priors through PUCT. The research integration may supply a validated portable learned leaf evaluator; otherwise cutoffs use the baseline heuristic. Neither path implies calibration or stronger play without held-out evidence.

A player's legal deck search infers the multiset of their remaining Prizes; only their own hand and publicly known zones are subtracted from their own decklist. Joint sampling respects known deck membership, exact top/bottom order, and revealed opponent-hand lower bounds. Actual Recon Directive, Pokégear, Ciphermaniac, Eri, repeated search reveals, and Judge have regression fixtures. Own known cards returned to the deck cannot silently become Prizes. Revealed copy lower bounds constrain listed and unknown opponent hypotheses. Independent continuations retain the observer's acquired knowledge; information-set keys include that history.

Stale Prize knowledge, unfamiliar reveal destinations, unsupported markers, and unmodeled ordered segments still refuse search. In particular, Special Red Card creates a known but unordered bottom segment and remains unsupported for affected search observers. Resolved own Prize pickups reconcile inferred identities only when unambiguous. No history is reconstructed using later hidden revelations.

## Tactical admission and focused rules coverage

`fixture({fixtureId,variationId})` materializes twelve constructed analogues from four guide families: Crustle delay-prize/energy-function and Dragapult information-order/hammer-target. It returns canonical curriculum IDs, an immutable fixture hash, observation, and `mechanicsAudit.validatedActionIds`. Each admitted root is exercised through real legal prompts with sixty-card conservation and narrow transition assertions. Hammer fixtures are explicitly conditioned on a real heads result. These receipts certify the listed mechanical assertions only; they do not certify guide strategy, full card implementations, optimal actions, or whole-deck training eligibility. Raw guide material is not included.

Focused checks cover Froslass sources/exclusions and ordinary simultaneous checkup knockouts; Area Zero Tera capacity, attachment discard, and original-owner order on replacement; Aura Jab zero/split recovery; Premium Power Pro stacking/expiry; Mega Brave's next-turn restriction; and Hariyama's hand-evolution gust. Area Zero replacement previously prompted Player 1 first regardless of ownership; the adapter now preserves the Stadium owner's discard priority. Suppression combinations, all removal/HP/recovery interactions, final-Prize simultaneous terminal cases, and complete matchup closure remain incomplete.

The project lead verified the official [2026 rotation announcement](https://www.pokemon.com/uk/pokemon-news/2026-pokemon-tcg-standard-format-rotation-announcement) and [Pitch Black release showcase](https://www.pokemon.com/uk/news/pokemon-tcg-mega-evolution-pitch-black-product-showcase) on September 19, 2026. These support format-level dates and marks; they do not complete the per-print legality audit. All competitive manifests remain experimental and `trainingEligible:false`.

## Verification commands

```
npm run engine:build
npm run typecheck
npm run test:engine
```

Tests are bundled as CommonJS with esbuild, matching production. Under Node 24, directly mixing ESM tsx tests with the vendored CommonJS package loaded duplicate CardManager/effect classes; the bundled runner prevents those false failures.

September 19 milestone evidence is in `evidence/watch-learn-verification.json`: final build `c01ee150cbe47377`, 53 passing engine tests, fifteen complete fixed-seed list games, twelve tactical recipes, twenty-eight scoped validated roots, and twenty-two verified vendor patch checksums. A regression confirms that drawing indistinguishable copies produces identical observations regardless of which remembered physical card was drawn. These checks do not change the whole-deck training gate.

September 20 adds a separate `auditFixture` method for the three reviewed Energy attachments in Crustle delay-prize variation 2. It compares the complete saved JSON observation and action bindings against the current recipe, checks Grass-only HP gain, exact target transfer, no immediate Spiky damage, the manual Energy limit, conservation, and independent reconstruction. Python stores a checksum-addressed `teaching-audits` attestation without rewriting the original teaching record, human review snapshot, or fixture receipt. Demonstration manifests bind both the original review and the new attestation. Five saved reviews are now effectively eligible for policy demonstrations; whole-game outcome admission remains unchanged. Focused evidence is in `evidence/teaching-admission-verification.json`.
