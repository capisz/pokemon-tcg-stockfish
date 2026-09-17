# Competitive implementation agreement

Approved September 17, 2026. Goal: a visual opponent and research engine capable of measurable improvement against a regional competitor, with new strategies treated as hypotheses until held-out validation. Six weeks is a milestone horizon, not evidence of superhuman strength.

## Main lists

- Dragapult: Andrew Hedrick guide page 48, post-Worlds/pre-Baltimore; 19 Pokemon / 32 Trainers / 9 Energy. Guide source remains private.
- Crustle: Rahul Reddy / Carter Malnaik guide page 27, Baltimore candidate; 10 Pokemon / 37 Trainers / 13 Energy. Exclude later Japanese/future Azurill proposals.
- Mega Lucario: user-supplied MEG Riolu/Lucario, Solrock/Lunatone, Hariyama list; 16 / 33 / 11.
- Grimmsnarl: user-supplied Munkidori/Froslass list with Gwynn; 22 / 29 / 9.
- Raging Bolt: user-supplied Mega Kangaskhan / Area Zero toolbox; 20 / 26 / 14.

Each archetype has a main, curated training variant, and reserved held-out variant. Verify exact printings and legality against an immutable international Standard snapshot dated 2026-09-17. Vendor metadata alone is not legality certification. Preserve historical starter lists. Keep held-out exact lists out of training and agent list hypotheses.

## Product and research

Untimed best-of-three; engine thinking budget of 120 seconds total per turn; closed lists by default, with explicit known-list lab mode; familiar 2D card table; practice hints separate from frozen-model benchmarks; selected-player information only until completed-match replay. No paid services. All runs are user-launched; continuous runs checkpoint bounded batches. The user reviews about two hours weekly, with a queue of at most ten positions.

M4 / 16 GB for play and review; Windows Ryzen / 64 GB, likely RX 580, for longer runs with a CPU baseline. Manual immutable bundle transfer; external SSDs for archives. No automatic scheduling or implicit model downloads. Existing local retrieval remains useful when optional LLM weights are absent.

## Ownership

1. Rules owner: packages/engine, vendor patches, decks/formats, engine tests, scripts/create-catalog.py and build-engine.mjs.
2. Learning owner: Python modules except api.py/guides.py/live-session/teaching modules, research tests, portable scripts (except catalog/build/UI), and config/dependencies coordinated with the lead.
3. Play owner: web, api.py, live sessions, guides/teaching, browser tests, PRODUCT/DESIGN.
4. Lead: sharedcontracts, integration, documentation/handoffs, research curriculum source metadata.

Commits and pushes on codex branches and draft PRs are authorized. Merges and deployment remain user-controlled. Preserve the foundation commit before parallel work. Private guides, data, models, and caches are never committed. Correctness gates precede scale; incomplete runs are not draws; trained models are not automatically champions.
