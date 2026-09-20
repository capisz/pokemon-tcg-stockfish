# Continuous learning verification — September 20, 2026

The bounded real integration completed the learning loop. The first normal-budget
run collected 50 games, trained a candidate, and recorded 50 comparison results
before a deterministic transport failure paused it. The 24-hour acceptance gate
remains open. See [the request-size repair](REQUEST_SIZE_REPAIR.md) for diagnosis,
regression verification and preservation of its experimental evidence.

## Real bounded cycle

Run `2eb936b60f6b48c78a764b6e00b6b847` in isolated `data/learning-qa` finished in
148.5 seconds with no errors or truncations:

- 10 completed collection games, covering mains and training variants across all
  five archetypes (the schedule's first block contains mirrors).
- Two positions investigated with two independent seeds each. The diagnostic's
  50 ms investigation budget supported neither target; neither became a label.
- One epoch on 1,000 rows: 900 outcome rows and 100 reviewed policy demonstrations.
  The model has 38,532 parameters. Ordinary bot moves received no imitation loss.
- Four naturally completed comparison games. This deliberately capped diagnostic
  did **not** satisfy the required 400-game protocol and did **not** adopt a model.
  Its apparent result is not a strength finding; the lower confidence bound was
  approximately 0.04, far below the required 0.5.
- Recorded peak process-tree RSS was about 1.30 GiB. Managed bytes included earlier
  isolated diagnostics, so they are not attributed as this cycle's disk growth.

Five original reviewed positions supplied the bootstrap. Three selected actions
changed; four of five final choices matched an accepted alternative. This is a
training-position diagnostic, not independent strategic validation. The fifth
review's separate mechanics attestation leaves its original review, timestamp,
reasoning and receipt unchanged.

See [bounded-run.json](bounded-run.json) for configuration, hashes and metrics.
Diagnostic search/collection limits differ from normal defaults. The first
attempt exposed an optional-ability cancellation loop and was stopped with its
journal intact. A tested, visible-information repetition guard then allowed the
successful cycle; no loop or interrupted game became an outcome label.

## Automated and browser checks

- 198 Python tests passed; two existing dependency deprecation warnings remain.
- 55 engine tests passed, including genuine terminal games for all fifteen
  competitive lists. This full run preceded the small experimental leaf-label
  change; focused Python/TypeScript evaluator tests covered that change.
- Final worker build `4958d864a552fb17`, TypeScript checking and web production build
  passed. The metadata change does not certify additional deck mechanics.
- Real isolated browser operation verified both worker views, perspective
  isolation, playback pause independent of learning, pause/reload/resume/stop,
  mobile width and no JavaScript errors. Its operational test run was stopped.
- Mocked browser coverage verifies following subsequent games, preserving a paused
  view, idempotent retry payloads, reload restoration and separate metric displays.
- Durable tests cover orphan frames, failed-worker reconstruction, repeated failure,
  storage-limit pause records, exact comparison schedules, seed collisions,
  regression rejection, optimizer resumption, model quarantine, compressed replay
  checksums/bounds, and nested experimental bundles.

Browser captures are local ignored artifacts under `artifacts/browser/`:
`learning-recorded-desktop.png` and `learning-recorded-mobile.png`.

## Normal-budget soak in progress

Manually started run `09572b53a4e74d9193c101bab05b30da` uses the agreed defaults:
two shared workers, 200 ms search per decision, 50 completed collection games,
20 investigations at two seconds, one CPU epoch, and the complete 400-game
comparison. It is unlimited until paused/stopped or a resource/invariant gate.

The process-bound Mac idle-sleep assertion was verified, with display sleep
permitted. Actual observations are recorded every 30 seconds in the ignored file
`data/competitive/soak-2026-09-20.jsonl`. The observer does not start or resume work.
Completed live games use compressed replay storage; the uncompressed diagnostic
artifacts were preserved. A read-only API check verifies saved frame/replay parity
and both player perspectives on an actual compressed learning replay.

Remaining gates: the full 24-hour observation window, a complete normal-budget
comparison, demonstrated strength improvement, trusted rules admission and native
Windows execution. No checkpoint has been promoted to trusted champion.
