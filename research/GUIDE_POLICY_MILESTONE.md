# Guide-informed policy milestone

Imported guides are private attributed evidence. They do not change a simulation's
policy, establish rules, label outcomes, or produce an evaluation bar. The default
simulation baseline remains the explicit hand-written heuristic.

## Reproducible teaching positions

The first four curriculum families have three simulator-generated variations each:

| Family | Question for review | Mechanics exercised |
| --- | --- | --- |
| Crustle delay prize | Attach defensively, pass, or take the knockout? | Attack/pass transitions and the terminal-prize condition |
| Crustle energy function | Which Energy function matters in this public position? | Legal attachments and Growing [G] HP changes; attack-effect protection and retaliation need separate rule coverage |
| Dragapult information order | Which source of information or selective search goes first? | Drakloak, Dudunsparce, and Poké Pad decisions |
| Dragapult Hammer target | Which target should lose Energy after a successful flip? | Target and Energy choice continuations conditioned on heads |

These are supported-deck analogues of the guide lessons, not claims that an exact
published board was reconstructed. `research/curriculum.json` records the original
attribution/pages and each `fixture` worker request. Each response contains its
observation, recipe hash, engine version, tested action IDs, test scope, and
limitations. No hidden deck order is part of the teaching observation.

Materialization stores a private immutable receipt and a **draft** teaching record.
The review queue returns at most ten records at once. The user must choose legal
acceptable actions and write conditional reasoning. An accepted action must also
be among the transitions exercised by the receipt to enter policy training.
This individual position audit does not certify the deck for outcome training.
The other sixteen curriculum families remain unmaterialized drafts.

## Learning and model admission

`train-policy` learns from reviewed, audited training-family decisions. Its loss
is negative log probability of the entire acceptable-action set. Thus two sound
alternatives can share probability without either being labeled a mistake. This
loss contains no terminal result, hand-authored value, generated prose label, or
invented reward. Raw guides are never passed to the network.

Policy-only checkpoints freeze their unused value heads. Inference returns legal
action preferences and an unavailable evaluation; no score or probability is
shown. Validation and test metrics use whole separate reviewed families, or remain
unavailable when none exist. These metrics measure guide agreement, not strength.

Each policy-only report also compares up to 100 admitted training positions before
and after learning, chosen by immutable teaching ID. It records the observation
and review hashes, the preferred legal action, and total probability assigned to
the acceptable-action set. The initial policy tensor hash and initial predictions
are saved in the first checkpoint. Resuming locally or on another device keeps
that same baseline; a missing or altered baseline blocks continuation rather than
silently redefining “before.” The report labels these comparisons as teaching
diagnostics. They are not held-out accuracy, playing strength, or win probabilities.

When verified complete games become available, normal policy/value training can
warm-start from this policy and mix reviewed demonstrations with game/search
targets. Outcome losses are masked off for demonstration rows. Cumulative game
seeds, teaching families, review hashes, source hashes, and optimizer state are
retained. Reviews have immutable snapshots, so a later correction cannot erase the
annotation used by an earlier checkpoint. Linked continuation creates a child
experiment; cross-device bitwise equivalence is not promised.

Only completed local training experiments whose checksums, schemas, model IDs,
and lineage agree appear in the model registry. Their status remains experimental
until held-out playing-strength evaluation supports promotion. Interrupted or
historical incompatible models are not offered as selectable completed models.

## Resource evaluation and search

Feature version `visible-energy-coverage-v3` checks identified visible Energy
types against printed attack costs. It exposes unknown provision and does not
pretend to resolve every cost modifier, condition, or ability suppression. The
Crustle feature depends on an opposing Active Pokémon ex, and represents a
protection capability rather than a guaranteed resolved effect. Feature metadata
states those limits explicitly.

Reviewed policy checkpoints can supply ISMCTS root priors. Only checkpoints trained
on verified game outcomes can export the portable value evaluator used at search
leaves. The export contains checksummed exact JSON bytes, feature/model versions,
and finite network parameters. Its sigmoid predicts expected result from the raw
outcome logit; it is not a calibrated W/D/L distribution. Search results remain
approximate and are not proof of an optimal move.

Actual local guides and their initial fixture records still require user review.
No human review, trained checkpoint, strength result, or calibrated advantage bar
is established by the implementation tests. Tests use explicitly synthetic review
annotations in temporary directories to exercise the pipeline.
