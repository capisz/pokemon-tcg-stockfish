# Pokémon TCG Stockfish research lab

A local Pokémon TCG simulator, playable card table, and reproducible learning laboratory. The competitive pool contains Dragapult, Crustle, Mega Lucario, Marnie's Grimmsnarl, and Raging Bolt / Mega Kangaskhan: one main, one training variant, and one reserved test variant each.

**Current checkpoint:** manually started, watchable experimental learning now connects reviewed positions, self-play, training, and gated candidate comparisons. Rules audits remain in progress. All fifteen competitive lists remain experimental and excluded from trusted outcome training. This is not a completed strength study or a claim of regional-level play. See [implementation evidence](docs/IMPLEMENTATION_STATUS.md) and [remaining assignments](docs/handoffs/).

## Install and play

Requires Node.js 22+ and Python 3.10+. No paid service or API key is required.

```sh
python scripts/setup-local.py --training
npm run dev
```

On macOS, use `python3` if that is your Python 3.10+ command. The installer creates `.venv`, installs dependencies, builds the simulator and checks the installation. Windows starts with CPU PyTorch; an AMD GPU is not required. Open **http://127.0.0.1:5173**. FastAPI listens on 127.0.0.1:8765. Stop the launcher with Ctrl+C.

The Play tab provides a 2D table with simulator-generated choices, inspectable cards, durable best-of-three sessions, bookmarks and a shared engine turn budget up to two minutes. Closed lists are the default. Practice allows hints; benchmark play freezes its policy and locks research assistance until completion. Choose known-list laboratory mode explicitly when you want both lists available.

Replay analysis now watches simulations live on the same playmat, with pause, speed, decision/turn stepping, and saved positions. Browser responses contain only the selected player's information; private research replays remain server-side. Publish completed matches explicitly before opening their projected replays. [Watch, play and teaching instructions](docs/WATCH_AND_LEARN.md) cover the complete workflow.

To use the already imported private guides in this Mac checkout, set the data directory before launching:

```sh
# macOS / Linux shell
PTCG_LAB_DATA=data/competitive npm run dev
```

```powershell
# Windows PowerShell, after privately importing guides on that computer
$env:PTCG_LAB_DATA = "data/competitive"
npm run dev
```

Artwork is loaded optionally by the browser from [TCGdex](https://tcgdex.dev/assets); offline or missing images retain readable card details. No artwork files are bundled. Card data and artwork do not establish rules correctness.

## Continuous experimental learning

On **Replay analysis**, use **Continuous learning → Start learning**. Both workers'
real games appear on the existing table. Playback pause is independent of learning;
Pause/Resume/Stop control the runner. Browser closure does not stop the local API.
An application restart restores progress paused and requires explicit resume.

This separately authorized lane learns from unverified simulator outcomes and keeps
all descendants quarantined under `data/competitive/experimental`. It never alters
the trusted champion. Default comparisons require 400 complete games and regression
checks before automatic experimental adoption. Five saved reviews now supply
admitted policy examples with their original review evidence preserved.

See [operating instructions](docs/CONTINUOUS_LEARNING.md),
[experimental research semantics](research/EXPERIMENTAL_LEARNING.md), and
[bounded integration evidence](docs/validation/continuous-learning/RESULTS.md).
A completed loop demonstrates the learning process, not stronger play. The full
24-hour soak and complete candidate comparison remain separately recorded gates.

## Reproducible research

Use `.venv/bin/python` on macOS or `.venv\Scripts\python.exe` in PowerShell. Examples below abbreviate that interpreter as `python`.

```sh
python -m ptcg_lab.cli doctor --benchmark
# Experimental rules QA; resulting games cannot train or promote a model:
python -m ptcg_lab.cli --data data/rules-qa selfplay --games 15 --seed 390920260 --allow-unverified
```

Trusted `selfplay`, `train`, and `continuous` work is gated by verified deck/rules eligibility. After that gate passes:

```sh
python -m ptcg_lab.cli selfplay --games 50 --population --search-budget-ms 200
python -m ptcg_lab.cli train --epochs 3 --max-positions 10000 --mlflow
python -m ptcg_lab.cli continuous --batches 1 --games 50
python -m ptcg_lab.cli export-bundle /path/to/mounted/ssd/run-bundle
python -m ptcg_lab.cli import-bundle /path/to/mounted/ssd/run-bundle
```

[Portable research instructions](docs/PORTABLE_RESEARCH.md) describe pause/resume, immutable bundles, linked optimizer continuation, profile limits and Windows commands. A manually launched `continuous --batches 0` continues in bounded batches; no startup is scheduled. Actual Windows execution still needs verification on the desktop.

The small policy/value model exposes learned baseline, resource and interaction terms. Engine advantage units equal outcome logit divided by `ln(2)`; one unit doubles expected-result odds. Untrained scores are labeled heuristic. Calibrated win/draw/loss probabilities remain unavailable without enough held-out evidence. Search cutoffs, guide agreement and lower training loss do not prove stronger play.

## Guides and review

Supplied PDFs remain private under ignored local storage. `research/curriculum.json` records twenty attributed guide families with sixty variations and permanent family partitions. Twelve positions from the first four families now have reproducible simulator recipes and scoped transition audits; five saved reviews currently supply admitted policy examples; the remaining positions await review. The other sixteen families remain specifications. Only reviewed, rules-checked training-family choices may train a policy. `prepare-teaching` creates the review queue, and `train-policy` learns acceptable-action sets without inventing outcome labels. Running a simulation does not train a model.

Local lexical retrieval works without model downloads. Optional Sentence Transformers requires an already installed local model. The optional Ollama/Qwen adapter drafts cited hypotheses; it cannot establish rules or numerical labels. Run language-model work separately from heavy training. These optional model runtimes have not been validated here.

## Verify

```sh
npm run engine:build
npm run typecheck
npm run test:engine
python -m pytest -q
npm run web:build
# With the app running and Playwright Chromium installed:
npm run test:ui
node tests/browser/play-smoke.mjs
```

## Project map

| Path | Responsibility |
|---|---|
| `packages/engine/`, `vendor/twinleaf/` | Pinned rules, staged choices, player knowledge, deterministic reconstruction and search |
| `decks/`, `formats/` | Fifteen competitive manifests, historical fixtures and frozen format metadata |
| `src/ptcg_lab/` | Workers, live sessions, datasets, training, evaluation, bundles and FastAPI |
| `web/` | Play, replay, analysis and teaching review |
| `contracts/`, `docs/` | Interfaces, evidence, limitations and independent assignments |
| `research/` | Attributed curriculum metadata; no raw guide passages |
| `data/` | Private sources, replays, local tracking, models and caches; ignored by Git |

Twinleaf is pinned to `b26ec9c1c5ea62849f2c248b17fec171aff108ca`. Its provenance, original notices and local patch hashes are retained under `vendor/twinleaf/`. No Kaggle competition SDK, restricted datasets or derived models are included. The project is independent of Pokémon, Twinleaf and Kaggle.
