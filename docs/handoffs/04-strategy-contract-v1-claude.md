# Claude update: Strategy Contract v1

The user narrowed the project before further engine work. Do not execute the earlier engine-core prompt from M0 through M3 as one task.

## Decisions now approved

- Codex work proceeds one bounded milestone per task, with a stop and report before the next milestone.
- The initial product goal is strategically competent play first and calibrated decision grading later.
- v1 has two distinct specialists over a shared strategic and fair-information core: Crustle and Dragapult.
- The authoritative lists are exactly `decks/crustle.json` and `decks/dragapult.json`; training and held-out variants are not strategy authorities.
- Initial coverage is Crustle vs Dragapult, Dragapult vs Crustle, Crustle mirror and Dragapult mirror.
- The specialists may infer opponent possibilities from public information but may not read hidden state or treat a scheduled exact list as player-visible.
- Guide-derived paraphrases may be committed. Raw guide text is not used at runtime.
- Scenarios may rank multiple actions as preferred or acceptable. Low-confidence positions abstain.
- Rules are hard constraints; strategic corrections become versioned playbook edits and reviewed weighted scenarios.
- Learning produces versioned candidates. Promotion is manual and may receive user corrections.

## Implemented in this milestone

- `research/strategy/contract-v1.json`: eight strategic concepts, information policy, five action tiers, abstention and manual-promotion rules.
- `research/strategy/crustle-v1.json`: frozen deck provenance, identity, six guide-derived principles and two matchup perspectives.
- `research/strategy/dragapult-v1.json`: frozen deck provenance, identity, six guide-derived principles and two matchup perspectives.
- `research/strategy/strategy-contract.schema.json`: JSON Schema for contract/playbook documents.
- `src/ptcg_lab/strategy_contract.py` and `scripts/validate-strategy-contract.py`: semantic validation and cross-checks against the frozen deck manifests.
- `tests/python/test_strategy_contract.py`: scope, provenance, abstention, promotion and version-boundary regression tests.
- `docs/STRATEGY-CONTRACT-V1.md`: contract explanation and curriculum audit.
- `docs/STRATEGY-CONTRACT-V1-REVIEW.md`: six focused user-review questions.

No engine, search, heuristic, model, trusted admission, champion, replay or UI behavior was changed. No training or long benchmark was started.

## Material finding

The Crustle PDF is an evolving guide. The frozen v1 manifest does not contain Crushing Hammer, Psyduck, Cornerstone Ogerpon ex, Bianca's Devotion or Hand Trimmer, although earlier guide sections and draft curriculum entries reference them. The playbook excludes unavailable-card instructions and retains only compatible strategic principles. The Dragapult frozen list aligns with the later post-Worlds section, so later list-specific guidance takes precedence over earlier count assumptions.

## User corrections and approval

The contract is now approved. The Crustle main manifest was corrected to the user-confirmed post-Worlds count of two Eri and two Xerosic's Machinations. The Crustle mirror plan now explicitly preserves Crustle, promotes the opposing Mega Kangaskhan ex to suppress damage, and times Xerosic so a three-card hand with three cards left in deck cannot use Lillie's Determination to reset the deck clock.

The next separate task should build an honest baseline showing how the existing policy behaves against these approved guide scenarios. Do not begin search optimization until that baseline is reported.
