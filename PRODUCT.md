# TCG Engine Lab

A local research tool for understanding Pokémon TCG decisions. The first interface runs the real supported simulator, opens saved games, and shows each decision alongside the information available to the selected player.

## First workflow

1. Choose two decks from the server's research registry, a deterministic seed, decision limit, and baseline policy.
2. Run a game. The interface follows the background job and opens its persisted replay when available.
3. Move through the decision timeline using buttons, the range control, the decision list, or Left / Right / Home / End keys outside form inputs.
4. Switch player perspective to inspect that player's hand, both public boards, resource terms, candidate actions, and pool-limited opponent beliefs.
5. Save a position from the timeline. Reopen it using the saved-position selector; a saved position contains only the selected observation. Open source replay is a separate, explicit action.

## Truth and information boundaries

- Deck support and validation status come from the registry. Experimental decks remain labeled experimental.
- Every board comes from a real replay frame. Frames describe the position before the displayed action; the final frame has no action.
- The UI consumes one player observation at a time. The opponent's hand is represented by its count; both full private observations are stored in the local research replay, which is not a public export.
- Opponent prompt choices are redacted in the timeline and its accessible labels. The board does not disclose the opponent's actual deck identity; opponent inference comes from the analysis response. Deck selections remain visible as local run metadata.
- Heuristic scores are labeled heuristic and uncalibrated. Missing probabilities display as unavailable. Resource terms describe the evaluator, not universal card values or exact causal explanations.
- A game stopped by a decision limit has an unknown outcome. It is never displayed as a draw or a completed win.
- Candidate action scores and beliefs are rendered as returned, including descriptions and limitations. The interface does not fabricate games, probabilities, evaluation curves, or trained status.
- Search candidate values use approximate expected result and show sampled continuations and reported standard error. Heuristic cutoff estimates are not completed game outcomes, and search values are not resource-score units or calibrated winning probabilities.
- Candidate details may include an illustrative sampled continuation, explicitly conditional on a sampled opponent assumption and future chance. Decision review, when available, reports opportunity loss against the best tested move in expected-result percentage points, with sampling caveats; it is not a definitive mistake grade or a measure of later luck.

## Local surface

The Play tab adds a real 2D card table, private server-journaled best-of-three sessions, and separate practice and benchmark modes. Engine decisions share one turn budget; accepted actions survive browser reconnects and server restart. The selected model is copied to an immutable private checkpoint at match creation. Analysis and research views are locked during an unfinished benchmark, including paused matches. Full research replays require explicit publication after the entire match ends.

Teaching review shows the bookmarked player view and legal choices, accepts multiple sound actions with conditional reasoning, and caps each queue at ten positions. Saved replay positions can be attached to a fixed guide family. Human benchmark records stay in the test partition; unaudited rules, prose lessons, and unreviewed annotations cannot become demonstrations. Local guide retrieval displays attributed passages and page/timestamp references.

The Vite development server serves `/` on `127.0.0.1:5173` and proxies `/api` to the local backend at `127.0.0.1:8765`. The REST boundary is defined in `contracts/PROTOCOL.md`. Production files build into `web/dist`; deployment is not part of this interface work.

The browser shows connection recovery, simulation progress, failed job recovery, unreadable replay recovery, and analysis loading/error/empty states. On a failed status poll, Retry checks the existing job; it does not start a duplicate game.
