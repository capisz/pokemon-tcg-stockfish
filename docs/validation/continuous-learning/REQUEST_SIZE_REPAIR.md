# Resume failure repair — September 20, 2026

The normal run paused during comparison because a search message exceeded the
worker's former 1 MiB limit. Its encoded request was 1,049,432 bytes; the worker
rejected it before parsing the request ID. The client then reported a misleading
request-ID mismatch. Repeated Resume could not change that deterministic failure.

## Repair

- Client and worker accept at most 32 MiB of UTF-8 JSON, with compact encoding and
  a client-side check before transmission. Wrong non-null response IDs still fail.
- Pre-parse worker errors retain their original diagnostic. Repeated failures
  record the cause and last accepted decision in the private game journal.
- Positions without supported information reconstruction use their frozen policy
  directly. They produce no invented search label and send no unnecessary model.
- Changed builds still cannot resume old run identities. An explicit linked
  recovery preserves the original run snapshot, pending journals, data and models,
  and starts a fresh comparison under new identities and independently reserved
  seeds. No earlier comparison result enters the new adoption decision.

## Verification

Worker source fingerprint: `bc02ed13677fb185`.

Both pending games reconstructed with exact observation hashes: the already
finished game at 248 decisions and the failed game at 157. A read-only continuation
of the failed position completed naturally at decision 175, with no worker error.
The original run and both pending game files retained their byte hashes. This
in-memory regression check was not saved as a training game or comparison result.

Focused transport tests include a real portable-model request larger than the old
limit, exact 32 MiB acceptance, UTF-8 oversize rejection, malformed input recovery,
and original-error propagation. Recovery tests cover immutable snapshots, frozen
data and model validation, interrupted preparation, explicit resume, and bundle
preservation of the linked ancestry. Unsupported probabilities and experimental
adoption restrictions remain unchanged.

This repair does not establish playing strength or a successful 24-hour soak.

## Local recovery observed

Recovery preserved the byte hashes of 678 existing game, teaching, model and data
files. The parent run metadata was archived separately with its exact prior
snapshot in an immutable receipt. The linked successor retained all 50 collected
games, the dataset and candidate checkpoint, then resumed through the normal
browser control on the repaired server.

At 06:04 UTC, ten fresh comparison games had completed and both workers were
playing subsequent games. No additional failure had been recorded; the displayed
two historical errors were inherited from the parent. Live-table playback was
verified against those actual worker games. A new read-only 24-hour observer is
recording the successor; the interrupted observer's evidence is retained.

The full Python suite passed 215 tests before the final recovery-hardening case;
all seven recovery tests passed after that addition. TypeScript checking and the
rebuilt worker passed. No card mechanic was changed by this repair, so the prior
full mechanics suite was not repeated.
