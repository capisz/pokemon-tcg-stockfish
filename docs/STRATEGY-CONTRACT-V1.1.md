# Strategy Contract v1.1

Status: approved by the user on September 21, 2026. Approved v1 remains unchanged and identifiable at commit `b176362`.

## Purpose

Version 1.1 is a reviewable overlay on the approved Crustle and Dragapult playbooks. It incorporates the latest compatible guide lines without silently changing the frozen deck identities or treating outdated card advice as executable.

The revision files are:

- `research/strategy/contract-v1.1.json`
- `research/strategy/crustle-v1.1.json`
- `research/strategy/dragapult-v1.1.json`
- `research/strategy/strategy-contract-v1.1.schema.json`

The overlays contain only additions and replacements. Approved v1 remains the immutable base, and approved v1.1 is now the active strategy contract.

## Shared changes

- A blind opening choice defaults to going first.
- Matchup-specific opening preferences apply only after the opponent archetype is public or in an explicitly labelled known-matchup lab run.
- The fifth internal tier remains `critical-error`; its display label is `Blunder`.
- Card provenance uses structured references. Executable specialist cards must appear in the specialist's frozen main list, and executable opponent cards must appear in the relevant opponent list.
- Historical missing-card references are non-executable and require a reason.
- Prose scanning is deliberately rejected as a provenance mechanism.

## Crustle revision

The overlay adds or corrects:

- Handheld Fan on Active Mega Kangaskhan ex against Dragapult, including the damage, Energy and Bench conditions required for the Tool to matter.
- Eri timing without importing the outdated three-copy count. The frozen list remains two Eri and two Xerosic's Machinations.
- Conditional Special Red Card use after the opponent reaches three or fewer Prize cards.
- Lumiose City as a non-Item Dwebble route under Budew, with its turn-ending cost explicit.
- The three-attached-Energy-card threshold for Jumbo Ice Cream.
- Mist and Spiky Energy roles before additional Growing Grass Energy against Hammer Dragapult.
- One Crustle by default against Dragapult; a second needs a named protected role or publicly exhausted trap route.
- Separate mirror plans for Kangaskhan and Dwebble starts.
- Precise Xerosic/Lillie language: Lillie's Determination may remain legal, but a sufficiently small combined hand/deck can leave no card for the next required draw.

## Dragapult revision

The overlay adds:

- Default Hammer timing after Crustle's second attachment.
- Exceptions for the only Mist, relevant Spiky, or the Jumbo Ice Cream threshold.
- Racing to Phantom Dive against a Kangaskhan opening and evaluating Judge on the relevant knockout.
- Pressure on an underprotected benched Crustle and immediate pressure after turn-one Ascension.
- Dudunsparce as a non-ex attack route when Spiky retaliation and attachment tempo permit it.
- A low-confidence known-matchup preference for going second in the mirror; blind game one still follows the shared go-first rule.

## What v1.1 does not do

It does not alter engine, search, heuristics, models, replay data, trusted admission, UI code or training. It does not assign action IDs or outcome labels. No benchmark or simulation run is part of this revision.

## Validation

Run:

```bash
uv run --extra test python scripts/validate-strategy-contract.py
uv run --extra test pytest -q tests/python/test_strategy_contract.py
```

The validator checks both the unchanged approved v1 and the approved v1.1 overlay, frozen deck counts, all four perspectives, page citations, fair opening information, structured card references, abstention/promotion gates and the `Blunder` display mapping.
