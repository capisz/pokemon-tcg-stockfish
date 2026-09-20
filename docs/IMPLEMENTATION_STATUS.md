# Competitive implementation checkpoint — September 19, 2026

The three implementation assignments are integrated on `codex/competitive-integration`. This is a playable research checkpoint, not completion of the six-week strength study. Fifteen competitive manifests replace the five engineering starters for play; historical starters are preserved separately. **Every competitive manifest remains experimental and training-ineligible until its legality and interaction audit passes.** No champion, calibrated competitive evaluator, regional-level strength or discovered superior strategy is claimed.

## What works

- **Rules:** 98 registered printings, immutable 60-card manifests, staged choices, deterministic action/chance reconstruction, controlled first player, player observations, and complete execution for all fifteen lists. Focused fixes address Crustle/Spiky/Mist, Hammer tails, Munkidori selection, Meowth restrictions, Special Red Card, reused prompt IDs and compressed Prize choices. Recon Directive must take a card; unsupported temporary-zone knowledge explicitly blocks search.
- **Search:** rollout and information-set methods consume a selected player's information. Reserved exact lists stay out of default hypotheses. A bounded unknown-variant component and prior-game public reveals constrain opponent possibilities. Own Prize inference and supported known top order constrain sampled states. Learned root priors are accepted; leaves and deeper policies remain heuristic.
- **Play:** a 2D table, optional verified art, card inspection, staged actions, durable untimed best-of-three, concessions, restart recovery, idempotent submission, shared turn budget, public knowledge across games and bookmarks. Practice and frozen-policy benchmark modes are separate. Abandonment preserves an incomplete journal after an incompatible build without inventing a result.
- **Teaching:** private guide retrieval, twenty attributed draft families and sixty variations, fixed family partitions, saved-position binding, review annotations and a queue capped at ten. Raw PDFs and extracted passages stay outside Git. Draft prose is never an automatic training label.
- **Research:** portable setup/profiles, diagnostics, bounded manual cycles, population opponents, sufficiently sampled search targets, periodic optimizer checkpoints, streaming compressed Parquet, bounded replay loading and atomic checksummed bundles with linked continuation.
- **Integrity:** explicit replay/checkpoint admission; persistent benchmark-family quarantine; cumulative ancestor data lineage; frozen evaluation weight snapshots; complete registry coverage before promotion. Legacy smoke models and incomplete/subset comparisons cannot bypass those gates.

## Verification and evidence

The integrated engine fingerprint is `6dae18d12adf6214`. Rules/browser checks used `310ae24b3cc8cbbc`; the final change advertises the already implemented first-player control to Python. Actual JSON-lines probes verified both requested starting players on the final build. Replays retain their actual build identity; earlier branch fixtures have earlier fingerprints. The metadata-only artwork map is part of the fingerprint.

| Check | Observed result |
|---|---|
| Integrated TypeScript engine suite | 31 passed, including complete seed-5 games for all fifteen competitive lists against Mega Lucario |
| Final Drakloak / knowledge changes | 13 focused competitive tests passed after the full suite; build and typecheck passed |
| Integrated Python suite | 76 passed, two upstream dependency deprecation warnings |
| Web | TypeScript and Vite production build passed |
| Real match browser | Legal setup action, bookmark, pause/reload/resume, card inspection, two concession outcomes with next-game choice, explicit replay publication and reviewed annotation passed |
| Actual API crash/restart | A durably acknowledged decision survived SIGKILL; restart paused the session, preserved identical player observation and budget, and reconstructed on resume |
| Real replay browser | Competitive Crustle vs Lucario seed 17 finished; selected-player view, saved-position roundtrip and uncalibrated analysis passed; unsupported search correctly withheld |
| Browser layout | 1440px desktop and 390px narrow checks; no JavaScript errors or horizontal page overflow |
| Recovery / bundles | Two-game starter smoke paused and resumed; 3.66 MB / five-artifact bundle roundtrip. Unit integration additionally trains a tiny model, transfers its checkpoint and continues optimizer progress |
| Device probe | Small-model CPU approximately 702 vs MPS 303 steps/sec on this Mac; CPU is the measured starting choice. Windows hardware remains untested |
| Card metadata | All 98 names and available regulation marks matched provider metadata; 85 art URLs available, thirteen Basic Energy printings require text fallback |

A longer live-API probe exposed a Python baseline selection/undo loop. The fix gives staged controls explicit priorities and has two regression tests. The corrected 180-second live API probe advanced through eleven turns and ended incomplete with no assigned result; a full natural-outcome live best-of-three is still unverified. The browser best-of-three concession test verifies product state transitions, not strategic playing strength. Full simulator games and natural terminal reasons are tested separately. Engine test counts are not an assertion of exhaustive card coverage.

Evidence and reproduction:

- `packages/engine/evidence/competitive-verification.json` and `recon-directive-review.json`: branch rules checks and source identities.
- `formats/rules-coverage.json`: per-printing coverage and unresolved interactions.
- `docs/evidence/card-metadata.json`: provider cross-check and missing art; no artwork files downloaded.
- `docs/PORTABLE_RESEARCH.md`: setup, profiles, pause/resume, bundle commands and resource limits.
- `tests/browser/play-smoke.mjs`, `tests/browser/smoke.mjs`, and `card-resilience.mjs`: actual UI/API integration scripts. Use isolated local data with no active match.
- `artifacts/browser/`: ignored local screenshots; private integration replays stay in `data/integration-ui/`.
- `docs/BASELINE_EVIDENCE_2026-09-17.md`: preserved historical starter experiments. Their model/data cannot silently enter the new trusted corpus.

## Remaining acceptance gates, in order

1. **Finish competitive rules certification.** Highest-risk fixtures are Froslass checkup/simultaneous knockouts; Area Zero expansion/shrinking; Lucario damage modifiers and evolution-triggered gust; item-lock timing; recovery, healing and HP combinations; genuine exhaustion lines. Complete official per-print/reprint legality evidence. All fifteen lists must pass, including Crustle and reserved variants.
2. **Complete information tracking.** Opponent hand reveals, generic peeks and known bottom order currently refuse search. Drakloak and Pokégear make this a material search-coverage limitation. Preserve every legally acquired constraint and prove indistinguishable hidden worlds give identical seeded decisions. Unknown-list priors are authored approximations, not learned opponent plans.
3. **Materialize and review guide fixtures.** The twenty families are not yet twenty legal simulator positions. Build concrete variants, validate their mechanics, obtain strategic review, then implement a measured guide-imitation stage. Current training does not yet consume guide demonstrations.
4. **Measure strength and evaluation quality.** Run the complete held-out matchup matrix with baselines, historical checkpoints and balanced starts after admission passes. Add contextual defensive-answer/recovery features, learned search leaves, matched-compute ablations and sufficient calibration. A deployed Stockfish-like bar must document its opponent population and compute context; the present untrained display is a labeled resource index.
5. **Complete platform/operational acceptance.** Run native Windows install, long self-play, crash recovery and cross-device continuation on the actual Ryzen desktop. Current monitors cover one Python process tree and managed data, not separately launched heavy processes or external model caches. Add shared accounting before claiming an aggregate project quota. Interrupted evaluations restart their bounded matrix. Live actions currently reconstruct prior accepted actions, so long-match latency needs profiling and worker-state reuse before scaling.
6. **Complete product research workflow.** Practice study branches, full video/transcript intake and reviewed-match test-set retirement need fuller implementation. Optional Sentence Transformers/Ollama inference is not installed or verified. Keep generated explanations subordinate to simulator/model evidence.

The next useful user contribution is strategic review of reproducible Crustle and Dragapult positions once their rules fixtures are verified. The existing guides and supplied lists are already preserved; no repeated upload is needed.

## Review and handoffs

Commits and feature branches are preserved; the integration is submitted as a draft PR. Merges and deployment remain user-controlled. Start additional work from the current integration commit, not the original `f6e9ac1` foundation. The three scoped assignments in `docs/handoffs/` remain the ownership boundaries.
