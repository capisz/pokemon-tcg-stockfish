# Strategy Contract v1

Status: draft awaiting human review. This milestone defines strategy; it does not change the engine, run training, assign win probabilities, or promote a model.

## Frozen scope

The first strategic system contains two specialists:

- `crustle-v1`, tied to `decks/crustle.json` version 2 and its exact `listHash`.
- `dragapult-v1`, tied to `decks/dragapult.json` version 2 and its exact `listHash`.

Only four perspectives are in the initial benchmark contract: Crustle into Dragapult, Dragapult into Crustle, and each mirror. Training and held-out variants are not strategy authorities.

The machine-readable files are:

- `research/strategy/contract-v1.json`: shared vocabulary, information policy, action tiers, abstention and promotion gates.
- `research/strategy/crustle-v1.json`: Crustle identity, principles and the two in-scope matchup perspectives.
- `research/strategy/dragapult-v1.json`: Dragapult identity, principles and the two in-scope matchup perspectives.
- `research/strategy/strategy-contract.schema.json`: a JSON Schema for contract and playbook documents.

## What strategic understanding means

The shared vocabulary makes eight concepts explicit: win condition, prize race, tempo, offense, defense, resource plan, information sequencing and plan switching. A later policy or evaluator must represent these concepts with current evidence rather than merely attach a strategic word to a move.

Each specialist has a primary plan, alternate plans, failure conditions and guide-derived principles. Each matchup entry defines opening intent, credible win conditions, represented threats and questions that must be answered before ranking an action.

## Fair-information contract

Normal evaluation may use the acting player's private state and public history. It may infer opponent possibilities from public cards, actions, deck-construction relationships and prior matchup knowledge, but must keep three categories separate:

1. Observed facts.
2. Confidence-weighted inferences with a public evidence trail.
3. Material unresolved possibilities.

An exact opponent list used by the scheduler is not automatically player-visible. Later reveals cannot be used to grade an earlier decision. A separately labelled hindsight analysis may eventually use more information, but it is outside v1.

## Action rankings and abstention

Reviewed scenarios can place more than one legal action in the same tier:

1. Preferred.
2. Acceptable.
3. Inaccuracy.
4. Mistake.
5. Critical error.

These are strategic ranks, not calibrated win-probability losses. Every grade eventually needs an active win condition, concepts advanced and risked, plus observed, inferred and unresolved evidence.

Only high- and medium-confidence positions are gradeable. Unsupported interactions, missing opponent hypotheses, shallow search, evaluator disagreement or hindsight dependence must produce `insufficient-confidence` with assumptions rather than a mistake label.

## Corrections and promotion

Rules, legality, information boundaries and deterministic replay are hard constraints. Strategy corrections are versioned playbook edits plus reviewed weighted scenarios; they do not become unconditional move rules.

Learning produces candidate specialists. Promotion requires a scenario regression report, the four-perspective benchmark, changed-decision examples, disclosed regressions and explicit human approval. Automatic promotion is prohibited.

## Existing curriculum audit

`research/curriculum.json` contains 20 families and 60 variations. Four families (12 variations) have reproducible engine positions; all remain unreviewed. Sixteen families still need legal positions and human strategy review. Several are outside the newly approved four-perspective scope, so they remain evidence for later work rather than v1 acceptance tests.

The curriculum is not rewritten by this milestone. The playbooks sit above it: playbook principles describe the strategy to review, while curriculum records hold eventual position-level evidence.

## Validation

Run:

```bash
uv run --extra test python scripts/validate-strategy-contract.py
uv run --extra test pytest -q tests/python/test_strategy_contract.py
```

Validation cross-checks specialist identity, frozen deck hashes and guide provenance; requires the exact four perspectives and eight shared concepts; keeps action tiers ordered; and enforces abstention and manual promotion gates.
