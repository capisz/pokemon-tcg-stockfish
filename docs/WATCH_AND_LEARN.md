# Watch, play, and teach

The shared playmat runs on real simulator observations in Play, Replay analysis,
and Teaching review. Live streams and saved replays expose one player perspective.
Opponent private choices are redacted on the server; hidden views and random
shuffle records are never presentation events.

## Use the prototype

Start with the existing private guide data on this Mac:

```sh
PTCG_LAB_DATA=data/competitive npm run dev
```

Open `http://127.0.0.1:5173/`. In **Replay analysis**, select two main/training
lists and Run game. The board updates while the simulator runs. Pause playback,
change speed, step by decision or turn, and return to the live position. Pausing
the viewer does not stop computation. A saved replay or position uses the same
table. Inspect a card for enlarged art and text. Missing art retains its text face.

In **Play**, select your deck and the opponent archetype. Click a card to inspect
its legal actions; select an action and a highlighted target. Finish and Undo
operate the simulator's staged selections. The action list remains available for
keyboard use and choices without a board target. Historical playback cannot
submit moves or annotate the current decision. Practice permits assistance;
benchmark play locks research assistance until the match ends.

Every run shows its actual policy. A simulation does **not** update a model.
Heuristic/random policies are baselines. An admitted checkpoint is frozen for the
run and selectable explicitly. A reserved list requires laboratory selection and
its games remain excluded from training.

## Make the guides teach the policy

Twelve constructed positions implement three variations of four existing guide
families: Crustle prize timing, Crustle Energy function, Dragapult information
order, and Dragapult Hammer targeting. They conserve both 60-card manifests and
have checked root transitions. They are teaching analogues, not historical games
or certified strategic answers. Hammer targeting begins after an actual heads
flip; it does not label the earlier choice to play Hammer.

```sh
.venv/bin/python -m ptcg_lab.cli --data data/competitive prepare-teaching
```

The command is safe to repeat: it preserves annotations and supersedes obsolete
draft receipts after engine/recipe changes. In **Teaching review**, choose all
sound alternatives and explain the conditions and resources that matter. The
queue shows at most ten positions; the next positions appear as reviews finish.
An alternative outside the checked transition set remains excluded until its
mechanics are checked. A rules test is never treated as your strategy review.

After you have approved examples:

```sh
.venv/bin/python -m ptcg_lab.cli --data data/competitive train-policy --epochs 20
```

This creates an experimental **policy-only** checkpoint, with immutable review
hashes, dataset partitions, optimizer state, and recorded losses. The acceptable
action loss rewards the total probability assigned to sound alternatives. It
never fabricates an outcome label for a tactical position. The value heads are
frozen and their scores remain unavailable. Reload the app to select the resulting
checkpoint. Without approved examples this command stops before creating a model.

The experiment report includes a before/after teaching diagnostic for up to one
hundred fixed training positions: selected actions, acceptable-action probability
mass, and observation/review hashes. Resume preserves the original baseline.
These changes measure what the policy learned from examples; they do not establish
better play or generalization.

Resume uses `--resume <checkpoint> --epochs <total>`. A transferred checkpoint
uses `--linked-resume` to preserve optimizer state in a child experiment. Bundles
carry teaching receipts, family bindings and immutable review snapshots; raw
private guide files remain excluded. Corrections cannot erase the annotations
that produced an older checkpoint.

## From preferences to strength and evaluation

All competitive lists still require full legality/interaction admission before
trusted outcome training. After that gate, the existing bounded self-play and
evaluation commands can supply outcome targets and supported search targets.
Use `train --warm-start <policy-checkpoint>` to initialize outcome training from
the reviewed policy, then compare it with the same process without initialization.
Keep the declared game/guide partitions and reserved variants untouched.

The engine now accepts a checksummed portable value model for search cutoffs.
PyTorch and TypeScript use the same versioned features, card tokens and small
network; numerical parity is tested. Policy-only checkpoints cannot supply this
value model. Deeper rollout action policies remain heuristic. Learned cutoffs are
uncalibrated expected-result estimates, not terminal results or proof of optimal
play. A model's baseline, resource and interaction terms sum to the outcome logit;
advantage units are that logit divided by `ln(2)`. W/D/L estimates have a separate
calibration gate.

The current Energy feature measures identified static cost coverage; it does not
establish legal readiness under cost changes, conditions or suppression. Crustle's
protection feature describes a capability against a visible opposing ex, not a
complete rules evaluation. Unknown provision and remaining coverage are explicit.
More contextual features and complete audits remain necessary before strength or
calibration claims.

## Verification and boundaries

```sh
npm run test:engine
.venv/bin/python -m pytest tests/python -q
npm run typecheck
npm run web:build
# Against isolated local data and a running test API:
.venv/bin/python scripts/check-natural-match.py --url http://127.0.0.1:8766
```

Cursor tests cover ordered delivery before completion, duplicate submissions,
reconnection, failed writes, ignored crash tails and worker reuse/reconstruction.
Browser checks use real simulations and matches, plus clearly separated component
fixtures for sparse/expanded Benches and artwork failures. The natural-game probe
uses an observation-only heuristic proxy for the human seat and a 1 ms engine
turn budget: evidence of product execution, not human or tournament strength.

This milestone does not supply an approved competitive champion. Human strategic
review, the other sixteen concrete curriculum families, remaining per-print
legality/interaction audits, all-pair strength comparisons, and native Windows
execution remain explicit acceptance work. Nothing here merges or deploys the
project.
