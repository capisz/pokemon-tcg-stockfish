# Continuous experimental learning

Start the API and browser with `PTCG_LAB_DATA=data/competitive npm run dev`, open
**Replay analysis**, and use **Continuous learning → Start**. Keep the terminal running.
Closing the browser stops neither games nor training. Stop or pause learning from
Replay analysis before shutting down; an interrupted application recovers paused and needs
an explicit Resume. There is no scheduled startup or background login service.

The Mac defaults are two shared simulation workers, CPU training, an 8 GiB process
memory target, 25 GiB managed data and a 20 GiB free-space reserve. These are safe
boundary checks, not OS-enforced hard caps. Keep awake uses a process-bound `caffeinate -i -w PID`, permits
display sleep, and is released during pause, human play, stop, or application exit.
Leave the laptop open and connected to power for an unattended run.

## What it learns

The runner snapshots admitted reviews, trains a policy initializer, then repeats:

1. Collect 50 new completed games. A persisted schedule covers all five mains and
   five training variants, including mirrors, exchanged seats and starting players.
2. Revisit up to 20 diverse supported decisions with two independent searches
   sharing two seconds per position. Similarity selects positions; labels remain
   bound to the exact observation and legal actions.
3. Train one CPU epoch on at most 20,000 stratified positions. Genuine game results
   train value; reviewed alternatives and sufficiently covered search distributions
   train policy. Ordinary played moves receive no automatic imitation loss. Reviewed
   demonstrations occupy at most 10% of outcome-stage rows.
4. Compare the frozen candidate to the incumbent in 400 fresh games at identical
   search budgets. Paired uncertainty, exact schedule coverage, matchup regressions
   and reviewed tactical retention all gate automatic experimental adoption.

The heuristic begins as incumbent. Inconclusive candidates remain experimental.
A recorded visible-position repetition guard retires a twice-tried action when
an alternative exists, preventing an untrained policy from repeatedly reopening
and cancelling an optional effect. It applies equally to both comparison policies,
uses no hidden cards, resets each turn, and supplies no search label.

The model is frozen during each game; changes apply only between games. The
independent comparison seeds and whole-game calibration/test partitions never
supply training rows. Reserved deck variants and held-out guide families stay out
of the routine loop. Additional reviews enter the next training snapshot.

Experimental data and descendants remain in `data/competitive/experimental`.
They cannot enter trusted training, trusted export promotion, or `champion.pt`.
Card rules are still being certified. An experimental win is evidence about this
simulator and opponent population, not a claim of competitive strength.

## Watch and control

Select either worker to follow its actual collection or comparison game. Playback
pause holds the viewed position while learning continues. Live resumes following
new frames. A saved game uses the same table and player-specific projection. No
watch operation starts another simulation. Practice matches and foreground
analysis receive priority; learning checkpoints at a decision/training boundary.

Training loss, comparison results and reviewed-position agreement are different
measurements. The experimental value bar is outcome logit / ln(2), with separate
resource contributions. It does not publish calibrated probabilities or a luck
score. Before outcome training, policy-only models withhold learned value estimates.

The following commands control the *existing* API; they do not launch more workers:

```sh
.venv/bin/python -m ptcg_lab.runner_client list
.venv/bin/python -m ptcg_lab.runner_client status RUN_ID
.venv/bin/python -m ptcg_lab.runner_client pause RUN_ID
.venv/bin/python -m ptcg_lab.runner_client resume RUN_ID
.venv/bin/python -m ptcg_lab.runner_client stop RUN_ID
```

To start from the terminal, use `python -m ptcg_lab.runner_client start` with the
same virtual environment. `--config FILE.json` pins a bounded diagnostic override;
smaller comparisons cannot pass normal coverage gates. For a bounded integration
check, `comparisonGameLimit: 4` deliberately leaves the required 400-game comparison
incomplete and therefore cannot adopt a candidate; production defaults use 0 (no
extra comparison cap). A stopped run is immutable;
starting again creates a new experiment. Windows execution is not yet validated for
this milestone; use CPU initially and leave Mac awake behavior disabled there.

## Recovery, retention and evidence

Accepted actions, policy randomness counters, spent turn budgets, frame cursors,
training inputs, optimizer state and phase cursors are durable. A worker is restored
once from its last acknowledged journal; a repeated failure pauses with an error.
Different engine or runner builds require a new run. An incomplete game remains
incomplete and supplies neither a win/loss label nor a fabricated draw.

The training buffer is capped by position count and input bytes. New experimental
replay frames use checksummed gzip compression and decode transparently for viewing
and training; legacy files remain readable. Redundant completed
runner frame streams are removed only after their immutable replay reproduces the
acknowledged position. Reviews, human games, comparison evidence and checkpoint
ancestry are protected. If protected data reaches a limit, the run pauses for
archiving; it does not erase evidence to continue. Imported bundles preserve their
experimental ancestry. Archived absolute source paths are provenance; a different
machine must validate/rebase the import before using it for continuation.

A soak record must measure elapsed operation, not assume it from an intended run:

```sh
.venv/bin/python -m ptcg_lab.runner_client soak RUN_ID \
  --seconds 86400 --output data/competitive/soak-24h.jsonl
```

This observer records 30-second status samples without starting or resuming work.
Inspect games, completed cycles, actual memory/disk growth, errors and recovery.
An observation window containing a paused run is not a successful continuous
learning soak. The 24-hour acceptance gate remains open until that evidence exists.
