# Autonomous Learning Mind v1

This is an experimental, research-only lane based on commit `7543298`. It does
not replace a production policy, promote a checkpoint, import a competition
dataset, build decks, or start an unattended training process.

## Evidence decisions

The four supplied Kaggle writeups support a compute-efficient order: visible
representation, supervised universal legal-action policy, strategic turn-plan
ranking, then PPO. Action abstraction is adopted with common random numbers,
relative within-position labels, adaptive rollouts, and opponent holdouts.
Uniform hidden-state resampling is rejected. The legal-option pointer encoder,
history tracking, terminal reward, historical self-play, specialization, and
failure guards are adapted clean-room. Hosted text embeddings, an oracle critic,
silent truncation resets, search-assisted inference, and deck building are
deferred.

The Team Unown Gradiant repository is MIT, but its release excludes the
competition simulator and card sheet. No source or dataset from that repository
is imported by this implementation.

## Safety boundary

Only `frame.observations[frame.actor]` enters the tracker and encoder. The
opposite observation, chance records, engine store, and opponent deck identity
are prohibited. Token and action limits fail closed. Raw replays stay in the
existing ignored artifact directory and are verified against the committed
manifest before encoding.

PPO stays disabled until representation parity is complete and the frozen
Transformer beats the heuristic on held-out labels and at least one targeted
probe without a severity-three regression. Even then a human must explicitly
enable the run. Checkpoints never become trusted automatically.

## Commands

Use the project environment and private manifest paths explicitly:

```bash
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind model-info
PYTHONPATH=src .venv/bin/python -m ptcg_lab.learning_mind audit-baseline \
  --manifest docs/validation/strategy-baseline-v1-2026-09-21/replay-manifest.json \
  --output artifacts/learning-mind-v1/representation-parity.json
```

Install the `mind` extra before training the macro ranker. XGBoost absence is an
error; the implementation will not substitute a different learner.

## Operational states

Collection, training, evaluation, and retention have independent cursors. A
new supervisor process always restores as `PAUSED`, including after reboot.
Identity drift, corruption, option omission, view leakage, and non-finite
tensors pause immediately. Three rejected updates or worker restarts in one hour
also pause. Installing a launch agent or configuring email remains an explicit
operations step; this repository only supplies the safe supervisor behavior.
