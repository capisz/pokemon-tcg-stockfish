# Watch, play, and learning checkpoint — September 19, 2026

Latest milestone: [continuous experimental learning](validation/continuous-learning/RESULTS.md) now has a completed bounded real cycle and a manually started normal-budget soak. Trusted deck admission and champion promotion remain locked.

The three assignments are integrated on `codex/competitive-integration`. Real
simulations, saved replays, human play and teaching review share a card table.
The guide-to-policy pipeline is implemented, but the user's twelve initial
positions remain **drafts awaiting strategic review**. No competitive checkpoint,
strength improvement, calibrated probability or superior strategy is claimed.
All fifteen competitive lists remain experimental and outcome-training-ineligible.

## Implemented in this checkpoint

- **Watch and play:** observable card zones, expanded/sparse Benches, card art/text,
  attachments, damage, conditions, inspection and simulator-bound source/target
  choices. Staged prompts retain Finish/Undo and keyboard action choices. Live
  follow, pause, speed, decision/turn navigation and seeking share immutable frames.
  Pausing playback leaves simulation computation running.
- **Durability and privacy:** ordered cursor feeds publish only durably committed
  decisions, ignore orphaned crash tails and redact other-player private choices
  before transmission. Browser replay schema 2 selects one perspective; legacy
  private replay files remain readable. Accepted human actions and engine choices
  remain journaled, with revision and idempotency checks. A retained match worker
  avoids rebuilding every prior decision; restart/replacement reconstructs it.
- **Human sessions:** untimed best-of-three, shared engine turn budget, practice
  assistance, frozen benchmark policies, bookmarks, concessions, restart recovery
  and explicit incomplete abandonment. Historical playback cannot submit a move
  or grade a different current position. Benchmark research access remains locked.
- **Rules and knowledge:** new focused checks for Froslass, Area Zero, Lucario and
  the twelve teaching positions. Area Zero replacement now preserves the original
  Stadium owner's Bench-reduction priority. Revealed hands, own deck membership,
  supported top/bottom order and movement/shuffle invalidation constrain beliefs.
  Ordinary Recon Directive and Pokégear resolution have search regressions.
  Unsupported information states still refuse search rather than discard knowledge.
- **Teaching:** four existing guide families now have three concrete variations
  each. Receipts identify the exact engine, recipe, observation and mechanically
  checked root actions. Review remains separate. Immutable review snapshots,
  permanent family partitions and contamination quarantine gate demonstrations.
  The queue is capped at ten; the other sixteen families remain unmaterialized.
- **Learning:** multiple acceptable actions share a policy-only loss with no
  invented outcome label. Policy training, optimizer resume, linked continuation,
  checksummed bundle transfer and model selection are implemented. Policy-only
  checkpoints cannot supply a value score. Verified outcome training can warm-start
  from an admitted guide policy and mix demonstrations with game/search targets.
- **Evaluation:** learned root preferences and a checksummed portable value network
  can enter search. Python/TypeScript features, card tokens and numerical outputs
  have parity tests. Deeper rollout policies remain heuristic. The UI distinguishes
  heuristic resource index, learned outcome logit contributions and independently
  calibrated W/D/L availability. No actual competitive outcome model exists yet.

## Verification and evidence

`docs/evidence/watch-and-learn.json` records the final tested engine identity,
commands and results. Earlier evidence retains its original build fingerprints;
it is not relabeled as testing this build.

- Engine fixtures cover complete games across all fifteen lists, legal bindings,
  knowledge constraints and the new mechanical interaction groups. They do not
  constitute exhaustive card or matchup certification.
- Python checks cover admission, multiple-action learning, quarantine, resume,
  portable value parity, bundles, private projection, cursor recovery and retained
  match workers. Two dependency deprecation warnings are upstream.
- Real browser checks exercise a live simulation while playback is paused,
  exact saved-position seeking, private perspectives, legal human setup, history
  guards, pause/reload/resume, bookmarks, two concession games and teaching review.
  Isolated synthetic annotations are not user strategy approvals.
- A real live-API game reached a natural rules terminal and started game two with
  the correct score. The human seat was an observation-only heuristic proxy and
  the engine budget was 1 ms. This verifies product execution, not human play or
  competitive strength. Game two's unfinished journal was preserved without
  assigning a match result.
- Browser checks include 1440×900 and 390px layouts, reduced motion, no horizontal
  overflow, failed/slow artwork, sparse targets and an eight-Pokémon Bench.

Reproduction and local evidence:

- `docs/WATCH_AND_LEARN.md`: launch, teaching materialization and training commands.
- `contracts/WATCH_AND_LEARN_V1.md`: private projection, action references and feeds.
- `packages/engine/COMPETITIVE_STATUS.md` and `formats/rules-coverage.json`: exact
  mechanics coverage and remaining gaps.
- `tests/browser/{smoke,play-smoke,watch-layout,board-bindings,card-resilience}.mjs`:
  real integration and separately labeled component checks.
- `scripts/check-natural-match.py`: bounded natural-outcome product probe; use an
  isolated server with no ongoing match.
- `artifacts/browser/` and `artifacts/natural-match.json`: ignored local captures
  and integration evidence; QA journals stay in `data/watch-play-qa/`.
- `docs/evidence/competitive-integration.json`: preserved previous checkpoint,
  including actual SIGKILL/restart recovery evidence and device measurements.

## Remaining acceptance gates

1. **Strategic review and the first real policy experiment.** Review the prepared
   positions, accepting all sound alternatives and recording conditions/resources.
   Then run `train-policy` and examine its decision comparison. Test-only reviews
   cannot stand in for the user's review. No additional guides or hardware are
   needed. Whole held-out teaching families remain separate.
2. **Complete rules and legality admission.** Finish per-print legality/reprint
   evidence and full interaction closure, including suppression, protection/HP/
   recovery combinations, item-lock timing, final-Prize simultaneous terminals
   and exhaustion. Format-level sources are not per-print certification. Admission
   must match engine, deck and covered mechanics; all mains and variants are gated.
3. **Close information gaps.** Special Red Card's unordered known bottom segment,
   unfamiliar reveal destinations, unsupported markers and ambiguous inferred
   Prize changes still block affected searches. Continue equal-information tests.
   Authored unknown-list priors are approximations, not learned opponent plans.
4. **Measure actual strength and calibration.** After admission, run bounded
   self-play, balanced held-out matchup comparisons and guide-initialization
   ablations. Promote only through the predeclared improvement/regression gate.
   Current static Energy coverage and Crustle protection capability features do
   not resolve all actual readiness/suppression/remaining-answer conditions.
   More contextual features and fresh tactical review remain necessary.
5. **Finish curriculum and broader product workflows.** Materialize the other
   sixteen families, add practice study branches and complete video/transcript
   intake and reviewed-match test retirement. Optional local LLM/retrieval models
   are not installed or verified; their prose cannot establish numeric labels.
6. **Complete machine acceptance.** Native Windows setup, sustained operation,
   crash recovery and cross-device continuation need the actual Ryzen desktop.
   Monitors cover one Python process tree and managed artifacts, not independently
   launched heavy processes or external model caches. Keep heavy jobs separate.

The next user contribution is review in **Teaching review**. Existing guide files,
earlier replays and incomplete human journals are preserved. An incompatible old
match stays paused; use explicit incomplete abandonment before starting another.

## Integration boundaries

Commits, pushes to `codex/` branches and the existing draft PR are authorized.
Merges and deployment remain user-controlled. Future assignments start from the
latest integration commit and read this checkpoint plus `docs/handoffs/`.
