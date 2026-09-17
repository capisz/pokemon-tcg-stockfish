# Assignment 3 — Analysis workspace, saved positions, and expert guidance

Continue the Pokémon TCG engine project in `/Users/admin/Documents/ChatGPT/Pokemon Ai project`. Repository: https://github.com/capisz/pokemon-tcg-stockfish. Inspect Git status and the actual local implementation first; the remote may lag local work. Preserve unrelated edits and the existing interaction model.

The objective is a clear replay/study tool that explains actual engine evidence and uses attributed expert guidance as a teacher. Work locally on the M4 / 16 GB machine. No paid services, remote model inference, automatic model downloads, push, or deployment without user authorization. Do not change the compact analysis layout into a dashboard or begin another design-concept round.

## Ownership and current surface

Own `web/`, `PRODUCT.md`, `DESIGN.md`, and guide processing in `src/ptcg_lab/guides.py`. Coordinate tests and shared modules with the Python research owner. The backend owner controls FastAPI, persistence, training eligibility, and model/evaluation semantics; the rules agent controls simulation/search. Agree on necessary contract/API additions before touching their files or changing `contracts/PROTOCOL.md` concurrently.

Read `README.md`, `PRODUCT.md`, `DESIGN.md`, `contracts/PROTOCOL.md`, `docs/ARCHITECTURE.md`, and `docs/IMPLEMENTATION_STATUS.md` if present.

The React app at `http://127.0.0.1:5173/` proxies `/api` to `127.0.0.1:8765`. It has five real deck selectors, seeded background game jobs, saved replay selection, selected-player boards, keyboard decision navigation, resource terms, candidate analysis, pool-limited beliefs, and saved single-view positions. Board and timeline stay left; analysis stays right on desktop and stacks below on narrow screens. Warm offwhite, restrained green, familiar system type, and no remote assets are the approved direction.

Opponent prompt labels are redacted in the timeline, including accessible labels, and the board does not reveal the real opponent deck name. Full research replays retain both private observations; they are not public exports. Saved positions retain only the chosen observation and require an explicit action to reopen the source replay. Candidate details can show conditional sampled continuations; experimental decision review can show opportunity loss against the best tested action when both were sampled. Search estimates, heuristic/learned resource scores, calibrated W/D/L, and sample uncertainty are different quantities and must remain distinctly labeled. Many positions legitimately have unavailable probabilities, comparisons, or unsupported search.

Guide processing currently imports attributed UTF-8/Markdown files, retrieves local passages lexically, optionally uses an already-downloaded local sentence-transformer, and can request cited draft notes from local Ollama at `127.0.0.1:11434`. Drafts are training-ineligible and require review. Import/retrieval code is not proof that Ollama inference has been installed or tested. There is no completed reviewed-guide-to-demonstration pipeline yet.

## Next work, in order

1. Exercise the actual game → replay → decision → save-position → reopen-position workflow. Verify desktop, a laptop-height viewport, phone width, keyboard controls/focus, reduced motion, loading/error/empty states, and analysis refresh. Use real local API data; never mock a successful game or evaluation for screenshots. Preserve warnings and make unsupported search useful to understand.
2. Improve saved-position explanations using only the selected predecision view. Render legally revealed prompt cards when the protocol supplies them; do not import the other observation to fill gaps. Show the supported candidate alternatives, rollout counts, standard error, cutoff caveats, and model status. Do not call a bad random draw a mistake. Full calibrated mistake grading remains future work until the backend supplies comparable predecision counterfactual values.
3. Add a minimal local guide-study workflow around the existing importer/retriever. Keep author, source, format date, matchup, content hash, and exact chunk citations visible. Preserve imported source text and flag stale or mismatched formats. Coordinate the smallest API endpoints needed with the backend owner; do not bypass its storage or fabricate returned content.
4. Implement an explicit review step for cited strategy notes. A reviewed teaching example must link to a real saved position or reproducible tactical fixture, identify the proposed legal action/claim, preserve citations, record review status, and be validated against the rules interface. Human guidance initializes priors or demonstrations; it does not become a hard constraint preventing discoveries. Generated summaries remain hypotheses until checked, and only the training owner may admit reviewed examples into a versioned training dataset.
5. Validate optional local model inference only if the user has already supplied/installed the model. If unavailable, keep lexical retrieval usable and report the missing local dependency. Do not silently fall back to a paid API or run an LLM alongside heavy training on this machine.

## Commands and interfaces

Run from the project directory:

```sh
npm run dev
```

In a second terminal:

```sh
npm --prefix web run build
.venv/bin/python -m pytest -q
.venv/bin/python -m ptcg_lab.cli retrieve 'preserving non-ex answers'
```

To import a user-supplied guide, replace the path and attribution values with its real details:

```sh
.venv/bin/python -m ptcg_lab.cli import-guide /absolute/path/guide.md \
  --title 'Actual guide title' --author 'Actual author' \
  --source 'Original source URL or user-authored source' --matchup 'Actual matchup'
```

Relevant existing endpoints are `/api/games`, `/api/jobs/{id}`, `/api/replays`, `/api/replays/{id}`, `/api/positions`, `/api/positions/{id}`, and `/api/analyze`. Analysis accepts either a replay/decision/player selection or a saved position ID. The frontend must not silently substitute future frames, a true opponent list, or a full replay for a saved single-view position. The `Refresh` control clears cached evaluation so an explicitly selected newer local model can be inspected.

## Completion report

Provide changed files, the exact local route, actual end-to-end and accessibility checks, screenshots from real data, saved-position/privacy checks, cited-guide examples, and any backend coordination still required. State which explanation values are heuristic, learned, searched, calibrated, or unavailable. Do not describe a trained model as strong, a cited note as proven strategy, or a successful local run as a deployment.
