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

The rollout orchestration CLI still needs to be implemented. It should consume
the frozen dataset positions and call `generate_candidates` and
`label_candidates` from `ptcg_lab.learning_mind.macro`.

For every selected position:

1. Generate candidates deterministically.
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
or emit a typed `MacroExecutionFailure` on every smoke fixture.

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
3. replace root-action proxy labels with candidates whose declared turn plan is
   fully executed or fails with a typed error;
4. run the default 16-to-64 common-random-number allocation;
5. refit and require measured archetype/policy holdouts;
6. train a new immutable candidate and evaluate held-out labels plus every
   frozen v1.2 probe.

Stop after producing the supervised acceptance report. Do not enable PPO,
install launchd, or promote the candidate in that task.
