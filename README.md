# Pokémon TCG Stockfish research lab

A local five-deck simulation, self-play, training, and replay-analysis prototype. It runs real Twinleaf card effects behind a deterministic headless adapter. Crustle is in the default training pool and every matchup matrix.

**Experimental research software.** This is a working first implementation of the six-week plan, not a completed six-week strength study. The starter decks are authored 60-card engineering baselines, not tournament-winning lists. Complete interaction certification, expert strategy review, probability calibration, and demonstrated playing-strength gains remain acceptance gates. No human-level or optimal-play claim is made.

## Run locally

Requires Node 22+, Python 3.10+, and npm. No paid service, account, or API key is required. On this Mac, `.venv` and both npm installs have already been created.

```sh
npm ci
npm --prefix web ci
python3.10 -m venv .venv
.venv/bin/python -m pip install -e '.[training,test]'
npm run engine:build
npm run dev
```

For the locked Python dependency set, use `uv sync --locked --python 3.10 --extra training --extra test` instead of the Python installation steps. Open **http://127.0.0.1:5173**. The API binds to 127.0.0.1:8765. Stop both with Ctrl+C. Ports are fixed; the frontend refuses to silently move to another port.

Pick two decks and a seed, run a game, and step through the replay. Choose a player perspective to see that player's information. Opponent private choices are redacted in the timeline. Saved positions retain a single player's observation. Research replay files contain both private views and should not be treated as public game logs.

## Reproducible experiments

```sh
.venv/bin/python -m ptcg_lab.cli doctor --benchmark
.venv/bin/python -m ptcg_lab.cli selfplay --games 25 --seed 17092026 --max-decisions 800
.venv/bin/python -m ptcg_lab.cli export-parquet data/trajectories.parquet
.venv/bin/python -m ptcg_lab.cli train --epochs 3 --max-positions 10000 --mlflow
.venv/bin/python -m ptcg_lab.cli evaluate --candidate heuristic --opponent random --seeds 2
```

Training prints the experimental checkpoint path. Pass it as `--candidate /absolute/path/model.pt` or `--policy /absolute/path/model.pt` to evaluate or generate games with it. Set `PTCG_ANALYSIS_MODEL=/absolute/path/model.pt npm run dev` to inspect an experimental model in the application. This does not promote it. The application otherwise uses a statistically promoted champion if present, or an explicitly untrained resource index.

Run `selfplay --stop-after 2` to pause a configured run after two games. Resume by supplying `--resume RUN_ID` and repeating its original arguments. Creating `data/pause-RUN_ID` pauses at the next game boundary. Training writes epoch checkpoints and accepts `--resume PATH`; incompatible feature schemas or data/configuration hashes are rejected. Failed or capped games are not fabricated draws and never supply terminal training targets.

`evaluate` tests all 25 ordered candidate/opponent deck assignments in both seats, including mirrors. Use held-out seeds disjoint from the model's data. Tiny evaluation runs only verify the pipeline. Promotion requires enough completed pairs, an improvement bound above 0.5, matchup coverage, no unresolved excluded pairs, and the required champion/baseline opponent. Seat balance is distinct from going-first balance; see the research limitations.

## Verification

```sh
npm run typecheck
npm run test:engine
.venv/bin/python -m pytest -q
npm --prefix web run build
# With npm run dev running in another terminal and Playwright Chromium installed:
npm run test:ui
```

Engine tests cover deterministic replay, callback reconstruction, branch independence, hidden information, actual complete games, search, and selected Crustle interactions. Python tests cover storage, dataset isolation, true outcomes, model decomposition, checkpoints, promotion gates, guides, and API behavior. Browser smoke evidence and first experiments are documented in `docs/IMPLEMENTATION_STATUS.md`.

For bounded performance probes, run `.venv/bin/python scripts/benchmark-local.py` and `.venv/bin/python scripts/benchmark-search.py --data data/current-smoke`. These measure throughput/memory and search sample coverage; they do not establish playing strength.

## Guides

```sh
.venv/bin/python -m ptcg_lab.cli import-guide /path/guide.md \
  --title 'Crustle matchup notes' --author 'Guide author' \
  --source 'Original URL or user-authored source' --matchup 'Crustle vs Dragapult'
.venv/bin/python -m ptcg_lab.cli retrieve 'preserving non-ex answers'
```

Local lexical retrieval works immediately. Optional semantic retrieval uses `pip install -e '.[guides]'` and `retrieve ... --embedding-model /path/to/local/sentence-transformer`. The model must already exist locally; the application never silently downloads weights.

The optional Ollama adapter targets `http://127.0.0.1:11434` and `qwen3:4b`. Once Ollama and that model are installed separately, use `draft-notes 'your matchup question'`. Generated notes require valid source chunk citations and remain unreviewed, training-ineligible hypotheses. Ollama/model inference has not been verified in this checkout. Do not run a large language model concurrently with heavy training on the 16 GB machine.

## Project map

| Path | Responsibility |
|---|---|
| `packages/engine/` | TypeScript rules adapter, seeded RNG, legal choices, public observations, search, JSONL worker |
| `vendor/twinleaf/` | Pinned independently adapted rules source and attribution |
| `decks/`, `formats/` | Five frozen lists and closed Standard legality allowlist |
| `src/ptcg_lab/` | Persistent workers, self-play, data, models, evaluation, guides, FastAPI |
| `web/` | React replay and saved-position application |
| `contracts/PROTOCOL.md` | Versioned simulator/replay/API interfaces |
| `docs/` | Architecture, evidence, limitations, and three independent handoffs |
| `data/` | Local private replays, experiments, models, SQLite tracking; ignored by Git |

Twinleaf is pinned to `b26ec9c1c5ea62849f2c248b17fec171aff108ca`. See its `PROVENANCE.json`, `LICENSE.ryuu-play`, and `LOCAL_PATCHES.json`. Upstream declares MIT; the original RyuuPlay notice is retained. No Kaggle competition SDK, datasets, replay files, models, or Pokémon card artwork are included. The project is independent of Pokémon, Twinleaf, and Kaggle.
