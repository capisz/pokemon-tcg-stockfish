# Autonomous Learning Mind v1 — Operator Runbook

This guide describes how to take the research implementation from
representation parity through its first supervised candidate and,
only after that candidate passes its gates, toward bounded PPO and supervised
continuous operation.

This is not currently a one-command autonomous learner. The safety components,
dataset builder, macro-rollout orchestration, model, ranker, evaluation CLI,
PPO update, and supervisor exist. The first bounded smoke is complete, but its
four positions are insufficient for a held-out or strategy win. Notification
adapters and long-running operation remain deliberately disconnected.

## 1. Non-negotiable boundaries

Keep these rules in force during every phase:

1. Work only in the isolated worktree and branch:

   ```text
   /Users/admin/.codex/worktrees/learning-mind-v1/Pokemon Ai project
   codex/learning-mind-v1
   ```

2. Do not modify the canonical checkout or its untracked `Claude outputs/`.
3. Never feed `frame.observations[1 - frame.actor]`, replay chance records, the
   engine store, or an opponent deck identity into a learner.
4. Never turn an engine error or truncation into a draw, loss, reward, or value
   bootstrap.
5. Never treat an ordinary self-play move as a correct policy label.
6. Keep training, development, and promotion seeds disjoint.
7. Do not automatically promote a checkpoint. Promotion always requires human
   approval after the frozen evaluation.
8. Stop on identity drift, replay corruption, legal-action omission, private
   information leakage, or a non-finite tensor.
9. Preserve the prior trusted checkpoint so rollback is atomic.
10. Do not install a reboot service until the 24-hour supervised soak passes.

## 2. Current readiness

The following is complete:

- all 192 frozen baseline replays and 41,675 actor decisions encode safely;
- `StrategyTransformerV1` is within the frozen parameter range;
- legal actions and autoregressive `STOP` are represented;
- the supervised trainer resumes bit-equivalently on the same device;
- macro candidates, rollout seed sharing, XGBoost boundary, holdouts,
  sequential evaluation, PPO update guards, specialist routing, rollback, and
  pause-safe supervisor behavior are tested;
- Python, engine, TypeScript, and strategy-contract tests pass.
- an immutable four-position supervised smoke manifest;
- resumable macro collection and XGBoost fit/report commands;
- a one-epoch supervised smoke checkpoint and development evaluation.

The following is not complete:

- held-out label and targeted-probe wins;
- enough approved positions for archetype and policy-family holdouts;
- complete turn-plan execution rather than root-action proxy labels;
- a candidate-specific 400-game screen or sequential promotion evaluation;
- an installed notification transport or reboot service.

The authoritative gate record is
`docs/validation/learning-mind-v1/stage-gates.json`.

### Latest actor-visible coverage boundary (2026-09-23)

The v5-v7 matched coverage report is
`docs/validation/learning-mind-v1/fresh-position-coverage-v5-v7-2026-09-23.json`.
The narrow reveal-history fixes increased searchable frames in the sampled
12-game batch from 266 to 929, while preserving actor-only replay storage and
identical play. This is not complete representation parity. In particular,
91 v7 frames still require hidden-marker provenance unavailable to the actor;
they remain unsupported. The v7 pool has only three source games, one opponent
family, and candidate support has not been re-audited under its current engine
identity. Do not use it for ranker fitting or claim independent held-out
performance.

The next safe gate is to broaden fresh, privacy-audited coverage across source
games and frozen opponent families. The v7 candidate-support audit is now
complete: all 18 positions produce complete candidates under the exact current
identity (435 candidates total; at most 116 per position). This is support
only, with no rollout labels, and does not resolve the three-game/one-policy
diversity limitation. See
`docs/validation/learning-mind-v1/macro-support-v7-2026-09-23.json`. Keep
unsupported positions fail-closed. Broaden independent sources before deciding
whether there is enough diverse support to collect labels. PPO, 24/7 operation,
promotion, and service installation remain disabled.

### Second policy-family coverage result

A separate Python-heuristic batch is now frozen alongside the TypeScript family.
Each has a separate 72-row pool with six source games (four train, one
development, one held out) and balanced opening/midgame/late coverage. This is
useful for coverage expansion, but one held-out game per family is not enough
for generalization. The initial Python 18-row support sample had three
fail-closed candidate-cap overflows; do not transfer that support rate to the
expanded pools. Full identities and caveats are in
`docs/validation/learning-mind-v1/fresh-position-coverage-v8-2026-09-23.json`.
The next gate is a tested multi-source pool/split audit plus support checks on
the exact expanded rows—not rollouts or model fitting yet.

The expanded-pool audit re-encoded all 144 rows and confirmed game-disjoint
splits and zero cross-family duplicate positions, but exposed severe row
imbalance (TypeScript 67/4/1 and Python 49/1/22 train/development/held-out).
Therefore do not use these tiny dev/held-out subsets as generalization tests.
The next implementation task is to repair and test selection/splitting so
source games remain disjoint while each split gets enough balanced rows; then
rerun identity and candidate-support audits. Details are in
`docs/validation/learning-mind-v1/expanded-pool-integrity-v8-2026-09-23.json`.
The imbalance is partly structural: only three games per family yielded
midgame/late positions. Preserve game-level splits and collect additional
independent games with stable later-turn observations rather than moving rows
between splits; stage provenance is recorded in
`docs/validation/learning-mind-v1/expanded-pool-stage-provenance-v8-2026-09-23.json`.

A new `coverage-2026-09b` epoch has since produced 144-row pools for each
heuristic family, with 48 rows per stage and seven source games per family
contributing midgame/late positions. This improves coverage but not evaluation
power: each family has only one held-out and one development source game, and
candidate support has not been rerun for these exact pools. Keep labels and
fitting disabled. See
`docs/validation/learning-mind-v1/fresh-position-coverage-v9-2026-09-23.json`.

The v2 game-level allocator is now implemented and tested. For pools with at
least eight source games, it assigns at least two games to development and two
to held-out while optimizing row balance and matchup representation; smaller
collections retain the one-game-per-evaluation-split minimum. Every source
game remains wholly within one split. The rebuilt epoch-B pools now split as
TypeScript 68/39/37 rows across 5/2/2 games and Python 91/29/24 rows across
6/2/2 games (train/development/held-out). This is more useful but still too
small for robust policy-family generalization.

The exact v2 pools have now passed a full transition-candidate support audit:
288/288 positions were checked under the recorded engine, feature, tracker,
deck, and planner identities. Python supports 138/144 positions (2,519 complete
candidates); TypeScript supports 142/144 (3,561 complete candidates). The ten
unsupported positions all fail closed because the complete candidate set
exceeds the configured 128-candidate cap; none has zero candidates. These rows
must not be silently pruned or treated as labeled. Keep rollout labels and
model fitting gated until candidate-cap handling is explicitly reviewed and
the small game-heldout sets are expanded. Full counts, hashes, and artifact
locations are in
`docs/validation/learning-mind-v1/macro-support-game-balanced-v2-2026-09-23.json`.
No rollout, label generation, ranker fit, supervised training, PPO, or promotion
was run as part of this audit.

### Source-game sampling follow-up

The next 30-game-per-family epoch showed that split balancing alone was not
enough: the prior row selector chose only 8 TypeScript and 10 Python source
games from 30 available games because it cycled through matchup/stage buckets
without cycling through games inside each bucket. Pool-builder v3 fixes this
by round-robining across distinct source games within every matchup/policy/stage
bucket before taking a second position from a game. New immutable pools must be
built with v3 and should show broad source-game coverage in every split before
labels are considered. The v2 pools and their audits remain historical and
unchanged.

The first v4 support-audit attempt exposed an infeasible determinization: the
selected actor-visible deck hypothesis could not reconcile the public zone
counts. The planner adapter now classifies this exact failure as
`unsupported-position`, allowing the audit to report the affected positions
without swallowing unrelated planner defects. Generic planner errors still
abort the audit.

## 3. Prepare an isolated environment

Open Terminal and enter the isolated worktree:

```bash
cd "/Users/admin/.codex/worktrees/learning-mind-v1/Pokemon Ai project"
git status --short --branch
git rev-parse HEAD
```

Expected branch: `codex/learning-mind-v1`. The working tree should be clean.

Create an environment inside this worktree. Do not reuse or modify the
canonical checkout's environment:

```bash
uv venv .venv
uv sync --extra test --extra training --extra mind
npm ci
npm run engine:build
```

`engine:build` also produces the ignored
`packages/engine/dist/learning-mind-planner.cjs` research CLI. Macro collection
requires this CJS bundle so candidate generation and the engine search worker
use consistent module semantics; do not invoke the TypeScript planner directly
with `tsx` for label collection.

Ordinary search retains its 80-decision horizon cap. Research search may request
up to 500 decisions only when it pins a public hypothesis with
`researchHypothesisId`; include that horizon in the frozen collection identity.
Macro collection also freezes a per-rollout wall-clock cap. The CLI defaults to
`--rollout-budget-ms 1000`; increase it only for a named experiment and expect
longer collection. A budget cutoff is recorded as `truncated` with no outcome
label, separately from a fixed-horizon cutoff. Each record includes a
`decisionCountDistribution` so runtime budgets can be estimated from observed
progress; this is diagnostic metadata, not a reward or value target. For example, a one-position
runtime smoke can use `--initial 1 --maximum 1 --horizon 300
--rollout-budget-ms 1000`; do not fit a ranker from a smoke set dominated by
truncations.
Rollouts can be parallelized with `--rollout-workers 2` (maximum 8); the
default remains one worker until a matched throughput comparison is recorded.
Worker count is frozen into the collection manifest and resume identity.
Within each position, collection atomically checkpoints after each fully
completed matched-seed batch. A restart replays at most the interrupted batch;
the position, candidate set, rollout config, and checkpoint hash must all match
before resume. Final position files remain immutable.

The reviewed lockfile includes the `mind` extra and XGBoost. On macOS, prefix
combined Torch/XGBoost test or experiment commands with
`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` to avoid competing native thread
runtimes.

Verify the environment:

```bash
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind model-info
PYTHONPATH=src .venv/bin/python -c "import torch, xgboost; print(torch.__version__, xgboost.__version__)"
npm run typecheck
```

Expected model parameter count: `1585714`.

### Environment checkpoint

Proceed only if:

- the branch and base commit are correct;
- `git status --porcelain` is empty before generated artifacts;
- PyTorch and XGBoost import successfully;
- the engine build succeeds;
- the model reports exactly 1,585,714 parameters.

## 4. Reconfirm frozen representation parity

The committed replay manifest points to the private ignored artifacts in the
strategy-baseline worktree. Confirm those paths still exist, then run:

```bash
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind audit-baseline \
  --manifest docs/validation/strategy-baseline-v1-2026-09-21/replay-manifest.json \
  --output artifacts/learning-mind-v1/representation-parity.json
```

Check the compact fields:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/learning-mind-v1/representation-parity.json")
data = json.loads(path.read_text())
for key in ("scheduled", "audited", "decisions", "unsupportedPositions", "representationParity"):
    print(f"{key}: {data[key]}")
PY
```

Expected values:

```text
scheduled: 192
audited: 192
decisions: 41675
unsupportedPositions: 0
representationParity: True
```

Any mismatch is a stop condition. Do not rebuild a missing replay by silently
substituting another game.

## 5. Build and freeze the supervised dataset

This is the next engineering milestone. Implement a dataset builder before
attempting training. It should write immutable JSONL or Parquet rows plus a
manifest containing the source hashes and split assignments.

### 5.1 Permitted policy labels

Only these sources may contribute policy loss:

- `exact-search-distribution`;
- `compatible-reviewed-acceptable-set`;
- `high-confidence-macro-plan`;
- `macro-ranker-distillation`.

Ordinary completed self-play decisions may contribute terminal value targets,
but their selected actions must not become policy labels.

### 5.2 Required row fields

Each policy-labelled row should include at least:

```json
{
  "positionHash": "...",
  "familyId": "...",
  "sourceGameId": "...",
  "sourceDecisionIndex": 0,
  "actor": 0,
  "deckHash": "...",
  "opponentArchetype": "...",
  "opponentPolicyFamily": "...",
  "featureIdentityHash": "...",
  "policyLabelSource": "exact-search-distribution",
  "acceptableActionIndices": [0],
  "policyDistribution": null,
  "split": "train"
}
```

Store the actor-visible observation or its encoded tensors, but never store the
opposite observation in the training row.

### 5.3 Deduplication and split rules

- Deduplicate by actor-visible position hash, not replay frame number.
- Keep every variation of one teaching family in one split.
- Keep one opponent archetype out at a time for archetype transfer tests.
- Reserve at least one frozen opponent-policy family that contributes neither
  labels nor PPO experience.
- Keep promotion seeds and positions entirely outside training and development.
- Record duplicates and exclusions in the manifest rather than silently
  discarding them.

### 5.4 Reviews

Grade only the four exact-compatible teaching review hashes from the baseline.
The four stale Crustle reviews remain excluded and queued for human re-review.
Do not edit a stale review merely to make it compatible.

### 5.5 Dataset manifest

Freeze:

- engine version and build hash;
- deck hashes;
- feature, tracker, card-metadata, and action-equivalence hashes;
- source replay hashes;
- review hashes;
- search checkpoint and configuration hashes;
- split algorithm and seed;
- counts by source, family, archetype, policy family, and split;
- every exclusion reason.

Commit the manifest and compact statistics. Keep large encoded arrays and raw
replays in ignored `artifacts/learning-mind-v1/` paths.

### Dataset checkpoint

Proceed only if a test proves that ordinary self-play actions cannot enter the
policy-labelled subset and that changing the opposite hidden observation cannot
change an encoded row.

## 6. Generate strategic macro labels

The rollout orchestration CLI consumes frozen dataset positions and calls the
transition-aware TypeScript planner plus `label_candidates` from
`ptcg_lab.learning_mind.macro`. The planner reconstructs one actor-visible
public determinization, enumerates bounded ordered prefixes (up to three
actions), and re-resolves every later action against the updated legal list.
Candidate overflow is an explicit unsupported-position outcome, not silent
pruning. A smoke run found one such overflow, so support-rate measurement and
branching reduction under the same legality guarantees are required before
large-scale collection.

The initial (coarse-key) 18-position audit found depth-three support on 5/18.
The binding-aware audit conservatively found 2/18 because the old search key
could alias different targets. The research search now shares a semantic key
that preserves target refs, slot indices, staged choice bindings and amounts;
same-card hand/prompt copies alone are interchangeable. The v2 audit resolves
all five former alias cases: depth-three support is now 4/18 and depth-two
support is 16/18. All remaining unsupported positions exceed 128 candidates.
Do not blindly reduce depth or silently prune; design and test an abstraction
that preserves plan intent and all meaningful targets before collecting broadly.

For every selected position:

1. Generate candidates deterministically with the frozen public planner.
2. Stop and record `unsupported-position` if more than 128 candidates exist.
3. Reconstruct hidden possibilities from actor-visible information and the
   approved deck-hypothesis model only.
4. Give every candidate the exact same determinization and chance seeds.
5. Run 16 rollouts per candidate.
6. Extend close candidates to no more than 64 rollouts.
7. Record finished, truncated, and error attempts separately.
8. Calculate within-position relative results only from completed rollouts.
9. Persist the candidate set, rollout seeds, result counts, uncertainty, and
   artifact hashes before moving to the next position.

Use the `training` namespace for ranker labels and `development` for ranker
selection. The labeler intentionally rejects `promotion` seeds.

Start with a small smoke set covering:

- Raging Bolt plan fidelity;
- Crustle Handheld Fan to active Mega Kangaskhan ex;
- Dragapult Judge against a large opposing hand.

Do not scale collection until candidates reproduce their declared action plan
or emit a typed `MacroExecutionFailure` on every smoke fixture, and the frozen
pool's unsupported-position rate is measured and accepted.

Use the support-only audit before rollout collection. It re-encodes every row
against the frozen feature identity and runs only candidate generation; it
does not invoke search, rollouts, or labels. Outputs are immutable, so select a
new directory for each audit:

```bash
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind \
  audit-macro-candidate-support \
  --root . \
  --dataset artifacts/learning-mind-v1/fresh-actor-positions-v9-python/macro-pool-game-balanced-v2 \
  --output artifacts/learning-mind-v1/fresh-actor-positions-v9-python/support-audit-v2 \
  --workers 4
```

The report freezes dataset row/manifest hashes, engine/features, planner bundle,
per-position support/failure reasons, and counts by matchup, policy family,
stage, and split. Unsupported positions remain unsupported; never truncate the
candidate list to force support.

## 7. Fit and evaluate the XGBoost macro ranker

Use `XGBoostMacroRanker`. If XGBoost is unavailable, stop; do not silently use a
different estimator.

For each position, supply candidate features, relative labels, group sizes, and
completed-rollout/uncertainty weights. Evaluate:

- top-1 expected-result regret;
- pairwise ordering accuracy;
- top-k recall of the best completed candidate;
- results per archetype and opponent policy family;
- every leave-one-opponent-archetype-out split;
- the separately frozen policy-family holdout.

Freeze each teacher iteration with `FrozenIteration`. Permit at most six
iterations. A new iteration must use a frozen prior teacher and a new manifest;
it must not overwrite earlier labels or metrics.

Stop if improvement exists only against one buggy opponent policy or disappears
under an archetype/policy-family holdout.

## 8. Train StrategyTransformerV1

Train the Transformer only after the supervised manifest and ranker evidence are
frozen.

Inputs may include:

- exact search distributions;
- reviewed acceptable-action sets;
- high-confidence macro executions;
- the frozen ranker's candidate distribution.

Value targets may additionally use completed self-play outcomes. Truncated and
error games supply neither terminal rewards nor reset-crossing bootstraps.

The trainer must persist:

- model and optimizer state;
- exact identity manifest;
- RNG state;
- epoch and next-batch cursor;
- dataset and split hashes;
- label-source counts;
- loss history;
- parent/teacher hashes.

Use a small deterministic smoke run first. Interrupt after a batch, resume it,
and compare every model tensor with an uninterrupted run on the same device.

Do not compare training accuracy alone. Report held-out acceptable-action
accuracy per record and per unique position, cross-entropy for distribution
labels, and calibration/value metrics separately.

## 9. First supervised acceptance gate

Freeze a candidate checkpoint before evaluation. Do not train during evaluation.

The candidate must satisfy all of the following:

1. Higher held-out acceptable-action accuracy than the frozen heuristic.
2. Improvement on at least one targeted strategy probe.
3. No new severity-three probe regression.
4. No legal-action omission, illegal autoregressive selection, or cap overflow.
5. Blind policy-family holdout passes.
6. Exact model, feature, engine, deck, scheduler, and opponent identities match.
7. Representative disagreements have been reviewed by a human.

If it fails, diagnose whether the problem is representation, candidate
generation, label quality, data balance, or optimization. Do not jump to PPO to
hide a failed supervised foundation.

Update `stage-gates.json` only from generated evaluation evidence. Never edit a
gate to `true` based on expectation.

## 10. Bounded PPO experiment

PPO remains disabled unless the supervised acceptance gate passes and a human
sets `humanEnablePPO` for that experiment.

The frozen configuration is:

```text
AdamW learning rate       1e-4
weight decay              1e-4
gamma                     1.0
GAE lambda                0.95
clip                      0.2
entropy coefficient       0.01
value coefficient         0.5
optimization epochs       1
actor decisions/update    8192
minibatch                  512
current self-play         80%
historical opponents      20%
mirrors                    5%
```

Use all five supported archetypes uniformly: Crustle, Dragapult, Raging Bolt,
Grimmsnarl, and Mega Lucario.

For every rollout boundary:

- terminal win/draw/loss reward is `+1/0/-1`;
- engine errors and truncations end the trace and are excluded;
- never bootstrap across a reset;
- never add prize, damage, or card-value shaping.

Before mutating parameters, `ppo_update` calculates its KL and value loss. A
minibatch is rejected if approximate KL exceeds `0.05` or value loss exceeds
`0.5`. Non-finite values are an immediate pause. Three rejected updates or
worker restarts in one hour pause the supervisor.

Begin with one manually launched update. Inspect its replay and metrics before
authorizing a longer batch.

## 11. Screening and promotion evaluation

Use frozen candidates and three disjoint seed namespaces.

### Smoke

- tracker invariants;
- complete legal-option coverage;
- autoregressive termination;
- ten games per policy;
- W/D/L/truncated/error reported separately.

### Screening

Run the existing exact 400-game, five-archetype matrix with matched seats and
first-player assignments. Treat it as a screen, not final playing strength.

### Strategy

Rerun every v1.2 probe. Require improvement on the targeted gap and no new
severity-three regression.

### Sequential promotion

For each ordered matchup:

1. Run at least 100 completed games.
2. If the 95% interval overlaps the non-regression boundary, continue to 250.
3. If still overlapping, continue to 500.
4. Stop early only for supported improvement or supported regression.

Promotion requires aggregate improvement, no critical matchup regression over
five percentage points, no severity-three probe regression, the blind policy
family, exact identities, and explicit human approval.

Keep the previous trusted checkpoint available through
`AtomicRollbackRegistry`. A candidate checkpoint is not trusted merely because
it was queued or won one local matrix.

## 12. Specialists

Specialize only after a generalist passes its gates. Fine-tune against the
frozen generalist and a progressive historical ladder.

A specialist is selected only by an exact approved deck hash. An unknown or
modified deck must fall back to the generalist. Never route using a deck name,
archetype guess, or partial card match.

Each specialist must independently pass the same information, legality,
strategy, matchup, blind-policy, and human-review requirements.

## 13. Supervised continuous operation

Do not begin here until all earlier sections pass.

Use four independently resumable phases:

1. collection;
2. training;
3. evaluation;
4. retention.

Persist completed games, phase cursors, optimizer cursor, RNG state, opponent
assignment, candidate sets, checkpoint manifests, and artifact hashes.

The CPU profile is:

```text
workers = min(physical CPU cores - 2, 12), with a minimum of 1
one shared batched inference process
8192 actor decisions per PPO update
checkpoint every 10 updates
```

Use the existing data cap and free-space reserve. Raw replays remain ignored;
commit only manifests and compact evidence.

Notifications are limited to:

- pause;
- failure;
- review-ready candidate;
- completed milestone.

Do not put email credentials in the repository or chat. Configure a credential
reference in the local service environment after the notification adapter is
implemented and tested.

### Reboot behavior

The example launchd file is:

```text
research/learning_mind/com.openai.ptcg-learning-mind.plist.example
```

It is intentionally uninstalled. Any new supervisor process reloads in
`PAUSED`, even if the previous persisted state said `RUNNING`. A human must
explicitly enable a new run.

Before installation, complete a manually supervised 24-hour soak. Verify pause,
disk exhaustion, corrupted replay, worker restart, rejected update,
notification, and rollback drills.

## 14. Failure response

Pause immediately for:

- identity drift;
- replay hash mismatch;
- missing or duplicated legal action;
- private-view leakage;
- non-finite tensors;
- repeated engine failure;
- data-cap or free-space threshold;
- three rejected updates or restarts in one hour.

After a pause:

1. Preserve the state and artifacts.
2. Record the phase, cursor, checkpoint hash, last completed game, and error.
3. Reproduce with the same identities and seed.
4. Classify the problem as data, model, engine, infrastructure, or evidence.
5. Add a failing test before changing behavior.
6. Resume only from the last durable boundary.

Never delete or replace a completed error/truncation record merely to improve a
reported result.

## 15. Final test checklist

Before accepting any milestone, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate-strategy-contract.py
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/python
npm run typecheck
npm run test:engine
git diff --check
git status --short --branch
```

Also verify the generated experiment manifest, replay hashes, exact scheduled
game count, unfinished-outcome accounting, seed namespaces, and disk reserve.

## 16. Recommended next evidence task

The orchestration layer below is implemented and smoke-tested. The next task
should expand its evidence without changing the acceptance rules:

1. collect approved positions for the Raging Bolt plan, Crustle Fan target, and
   Dragapult large-hand Judge families;
2. preserve a real held-out split and a blind opponent-policy family;
3. improve the transition planner's candidate abstraction without merging
   board targets or dropping meaningful action bindings: v3 now requires an
   explicit attack or pass ending. Only complete plans count against the
   128-candidate cap (intermediate traversal has a separate 4,096-prefix hard
   bound); the corrected accounting supports 11/18 frozen positions;
4. first fix action-sequence execution across matched determinization and
   chance samples. The one-position 80-decision smoke scored 0/81 candidates
   (25 cutoffs, 56 typed action-resolution errors); do not start broad label
   collection until a targeted matched-sample test passes;
5. refit and require measured archetype/policy holdouts;
6. only after broad, balanced support and label quality are demonstrated, train
   a new immutable candidate and evaluate held-out labels plus every frozen
   v1.2 probe.

### Position-stage support audit (2026-09-22)

The current Raging Bolt source replay set contains actor-visible states across
all three stages, but the pool builder intentionally admits only positions
without a search-unavailable flag. In the 12 source games, 177 actor-visible
states were opening, 343 midgame, and 938 late; only 37 opening states passed
the frozen eligibility gate. All sampled midgame/late states were rejected
because historical observations lack the revealed-card/known-order history
required by current determinization, or carry unsupported modified-state
flags. The stage-balanced selector is therefore behaving correctly; its
opening-only output is an evidence-coverage limitation, not a selection bug.

Do not remove this gate or infer hidden-card history from private replay data.
Use only freshly collected positions whose actor-visible tracker state is
complete, or separately build and validate an observable-history reconstruction
before admitting historical midgame/late positions. The 16-matched-sample
opening experiment also produced only 6 terminal outcomes across 544 rollouts
(538 budget cutoffs; at most one completed outcome for any candidate), so it
does not meet the minimum evidence threshold for ranker fitting. Preserve it as
development/runtime evidence only.

Stop after producing the supervised acceptance report. Do not enable PPO,
install launchd, or promote the candidate in that task.

### Fresh current-engine position collection (2026-09-22)

Use the research-only collector to get positions from a frozen current build;
do not use its ordinary heuristic choices as policy labels:

```bash
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind collect-fresh-positions \
  --root . \
  --output artifacts/learning-mind-v1/fresh-actor-positions-v3 \
  --games-per-matchup 4 \
  --policy typescript-heuristic
```

The command is serial and game-checkpointed. Its immutable identity covers the
engine/deck/feature identity, policy, schedule, game cap, and collector version;
resume only at the same output path with all replay hashes intact. It retains
only `frame.observations[frame.actor]` and removes chance records. Finished
games are exposed in `run-manifest.json` as the source manifest for
`build-macro-position-pool`; truncated/error games stay in the run record and
are not silently replaced. The v2 schedule balances the Raging Bolt seat and
first-player assignments when four games are scheduled in each cross-matchup,
and alternates first player in mirror games. Seed derivation includes the
collector version, so future batches do not repeat the exploratory v1/v2
namespace. These are position-coverage games, not a performance benchmark.

To collect a later deterministic seed epoch without colliding with the `main`
schedule, supply a stable ASCII slug and use a new immutable output directory:

```bash
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind collect-fresh-positions \
  --root . \
  --output artifacts/learning-mind-v1/fresh-actor-positions-epoch-b-python \
  --games-per-matchup 4 \
  --policy python-heuristic \
  --collection-namespace coverage-2026-09b
```

The namespace is part of the seed, replay IDs, schedule, and run identity.
Omitting it or using `main` preserves the original deterministic schedule; do
not reuse an epoch slug for a different experiment.

The first exploratory six-game capture is frozen at
`artifacts/learning-mind-v1/fresh-actor-positions-v1-2026-09-22`. It used six
finished TypeScript-heuristic games (two each against Crustle, Dragapult, and
Raging Bolt mirror), 1,161 decisions, and 23 engine-provided searchable actor
positions. The candidate pool contained 13 unique positions, all opening;
transition-plan generation supported 13/13, with 2–68 complete candidates
per position. This capture was for coverage diagnostics only: the original v1
schedule fixed absolute first player to seat 0 and did not provide full
factorial matchup balance. Do not treat its outcomes as policy-strength
evidence or resume it with the current collector.

The local replay files are under the ignored artifact directory and are
checksummed in its `run-manifest.json`; the compact pool manifest is
`macro-pool/manifest.json`. Candidate support has not yet been rollout-scored,
and all later states remain blocked by engine knowledge-reconstruction gates.

The committed v2 collector was exercised with a balanced 12-game run at
`artifacts/learning-mind-v1/fresh-actor-positions-v2`. All 12 finished, with
2,263 decisions and 38 engine-provided searchable actor positions. Its
18-position capped pool is still all turn 1, despite covering all three
matchups and every cross-matchup seat/first-player cell. Candidate generation
supports all 18 positions (2–89 complete plans), but no rollout labels were
collected. The v2 schedule reused the first two seed entries per matchup from
exploratory v1 because collector version was not yet part of the seed token;
the v2 pool does not mix in the separate v1 data. Current v3 fixes that
namespace issue and refuses to resume these v2 outputs as if they were v3.

The v3 run completed at `artifacts/learning-mind-v1/fresh-actor-positions-v3`.
It finished all 12 scheduled games, with zero truncations/errors and 2,079
decisions. All replay and manifest hashes were verified; 2,091 stored frames
contain only the acting player's observation, with opposite private views null
and chance records removed. Its new 18-position pool spans eight opening,
eight midgame, and two late positions, with 12/3/3 source-game-disjoint
train/development/heldout splits. The candidate support audit covered all 18
positions (2–116 candidates each); it produced no rollout labels. This fixes
the earlier opening-only coverage limitation, but the pool still uses only the
TypeScript heuristic opponent family and does not establish policy strength or
blind generalization. Checksums and exact metrics are in
`docs/validation/learning-mind-v1/fresh-position-coverage-v3-2026-09-22.json`.

The next gate remains a tested actor-visible reconstruction contract for
revealed-card and known-order history. Do not weaken determinization's
fail-closed restrictions to increase pool size. Until reconstruction fixtures
pass, these new positions are coverage evidence only; do not start a ranker fit
or policy training from them.

After the research sampler added exact transport for three observable public
flags, a fresh v4 run reused the same deterministic 12 seeds. The 12 action
sequences and game outcomes matched v3 exactly, while actor-searchable decision
frames increased from 225 to 252. A new, identity-matched 18-row pool was built;
candidate support has **not** yet been re-audited under this engine identity.
This is a narrow +27-frame coverage change, not general history reconstruction.
See `docs/validation/learning-mind-v1/fresh-position-coverage-v4-2026-09-22.json`.
