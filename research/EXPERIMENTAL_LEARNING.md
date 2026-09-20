# Quarantined continuous learning

The manually launched runner uses `data/competitive/experimental`. Simulator
outcomes are experimental evidence until the relevant rules gates pass. Neither
this path nor its descendants may publish `champion.pt` or enter trusted training,
calibration, benchmark evaluation, or the trusted opponent population.

## Pinned inputs and targets

`snapshot_teaching` records the effective reviewed training decisions alongside
only their referenced original family markers, review snapshots, fixture receipts,
and mechanical attestations. Original review bytes remain unchanged. Source reads
are bounded before JSON allocation; snapshots are checksummed. `bootstrap_policy`
uses these frozen records for twenty CPU policy-only epochs and retains its initial
before/after diagnostic through interruption and resumption. It supplies no value
or probability estimate.

`prepare_dataset` accepts explicit replay IDs, not an implicit scan of past QA
runs. A replay must declare experimental learning, retain `trainingEligible=false`,
use main/training variants, identify its engine/decks/run, and finish with a genuine
simulator terminal. Comparison, human, reserved-list, and incomplete games cannot
supply targets. Whole-family partitions remain fixed. Calibration and test
partitions are excluded from optimization.

Each immutable dataset pins replay hashes, selected decision indexes, observation
hashes, search-target hashes, teaching provenance, and feature-row hashes. Input
limits are 2,000 replay IDs, 256 MiB of source replay bytes, 20,000 feature rows, and
64 MiB of feature rows. Selection balances ordered matchups and prompt types;
at most 256 candidate decisions and four observations from one coarse position
group are considered per game. Grouping prioritizes sampling; it never transfers
an action label between similar positions.

Completed-game rows train the outcome/value heads. They incur **no policy loss**
without an exact observation-matched search distribution. Two independently
supported reanalysis searches take precedence over collection search targets.
Reviewed demonstrations provide acceptable-action-set policy targets and at most
10% of the total rows. They never receive invented game-result labels.

## Checkpoints, inference, and comparisons

`train_experimental` runs one CPU epoch with optimizer checkpoints and an exact
pinned batch cursor. Resuming requires the same data, configuration, and run ID;
warm-starting produces a descendant with inherited seed/family/review provenance.
Experimental ancestry is permanent. Normal `load_model`, `predict`, and portable
value export reject it unless the caller explicitly opts in, even when passed an
already loaded model tuple.

Completed models appear as `experimental-<id>` through the local model registry.
Outcome-trained models return **Experimental learned** with uncalibrated logit
terms; win/draw/loss and expected-result probabilities remain unavailable. The
portable simulator evaluator preserves the tier in its search context. A value
score is an estimate under this unverified simulator and training population, not
a verified strategic advantage or proof of playing strength.

`tactical_regressions` measures retention on pinned reviewed training positions
against a frozen incumbent or the heuristic baseline. Its accuracy is a teaching
diagnostic, not held-out strength. The runner separately controls fresh paired
game comparisons and can advance only its experimental incumbent.

Bundles preserve the nested experimental namespace, pinned datasets, teaching
audits, frozen models, comparison protocols, and private decision streams.
Importing starts no process. Imported runner journals need startup recovery to a
paused state and explicit continuation; archived absolute source paths must be
rebased and checksum-validated before cross-machine continuation. Private stream
files contain research state and are only exposed through the existing player
projection API.

## Verification boundary

Synthetic tests cover target admission, absence of behavior cloning on unlabelled
moves, anchor limits, data tampering, exact search-label matching, interrupted
optimizer equivalence, permanent ancestry, teaching retention, and nested bundle
round trips. Python/TypeScript evaluator parity includes both data tiers. These
tests do not establish the strength of an actual trained model or certify the
five competitive deck lists.
