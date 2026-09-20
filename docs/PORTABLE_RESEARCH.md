# Portable local research

The commands below run locally and never schedule startup, promote a model automatically, or download language-model weights. Rules coverage is an admission gate: a competitive deck with unresolved legality or interactions cannot supply trusted training games. `--allow-unverified` is a rules-QA override and always makes its games ineligible for training and promotion.

## Install and diagnose

Use Python 3.10 or newer and Node.js 22 LTS. From the repository:

```sh
python scripts/setup-local.py --training
```

On macOS, replace `python` with your Python 3.10+ executable if needed. The script creates `.venv`, installs the declared Python and locked Node dependencies, builds the simulator, and runs diagnostics. On Windows it installs CPU PyTorch; an AMD GPU is not required. Installation downloads dependencies but no guide/LLM models.

Run commands below with `.venv/bin/python` on macOS, or `.venv\Scripts\python.exe` in Windows PowerShell. The examples use `python` as shorthand for that project interpreter.

```sh
python -m ptcg_lab.cli doctor --benchmark
python -m ptcg_lab.cli selfplay --games 20 --seed 390920260 --population --search-budget-ms 200
python -m ptcg_lab.cli continuous --batches 1 --games 50 --seed 490920260
```

Diagnostics measure the platform, process-tree memory, free storage, a bounded simulator probe, and available CPU/MPS model throughput. The two workers remain unchanged by diagnostics. The Windows upper tuning bound is reported, not automatically selected. Benchmark an increase before configuring it.

The simulator probe is explicitly QA and may truncate. Its throughput is not evidence of playing strength. The Windows launch path and CPU configuration are covered by implementation and unit tests; an actual run on the user's Windows hardware remains necessary.

## Manual runs and recovery

`continuous --batches 0` runs successive bounded batches until interrupted. Each batch freezes an opponent population containing random/heuristic baselines and compatible champion/historical checkpoints. It gathers complete games with optional search-policy targets, trains for one epoch, and periodically records a paired evaluation. Evaluation is experimental and does not automatically install a champion. Explicit promotion requires coverage of every registered competitive list and all five archetypes in main, training-variant, and held-out roles; validating only a subset cannot satisfy the gate.

Only search decisions that sampled every legal action at least twice become soft policy targets. An observation hash ties each target to the actor's pre-decision information. Learned checkpoints can supply legal root-action priors to ISMCTS; deeper nodes and leaf evaluation remain heuristic. This is not a claim that learned search is stronger.

Press Ctrl+C to pause. The printed run or cycle ID identifies the durable checkpoint. Resume with the original arguments and `--resume ID`. Self-play resumes at the last durable game; a game interrupted before it was saved restarts with the same seed. Training saves optimizer state at epoch boundaries and every 100 batches or 60 seconds. An interrupted evaluation currently restarts that evaluation stage rather than continuing its partial match matrix.

A file named `pause-RUN_ID` in the data directory pauses a self-play run at a game boundary or a cycle at a batch boundary. Remove that file before resuming. No artificial computation limit supplies a win, draw, or loss.

```sh
python -m ptcg_lab.cli selfplay --games 20 --seed 390920260 --population --search-budget-ms 200 --resume RUN_ID
python -m ptcg_lab.cli continuous --batches 1 --games 50 --seed 490920260 --resume CYCLE_ID
```

Population paths/hashes and original parameters are immutable within a run. Resume fails if they change. For ongoing collection after changing policies or machines, start a new linked training experiment and a new cycle rather than rewriting historical manifests.

## Storage and memory

Defaults are two workers, 8 GiB process-tree target and 25 GiB managed data on Mac; two workers, 40 GiB target and 200 GiB managed data on Windows. Both preserve 20 GiB of free storage. Environment overrides are `PTCG_PROFILE`, `PTCG_WORKERS`, `PTCG_MAX_MEMORY_GIB`, `PTCG_MAX_DATA_GIB`, and `PTCG_LAB_DATA`.

The memory monitor covers the launching Python process and its descendants, not unrelated independent API/training processes. Limits are checked targets, not OS hard caps. Run heavyweight learning and language-model work separately. Model caches or Ollama stores outside managed data are not included in its disk quota. If process inspection is unavailable, monitored work pauses instead of pretending its memory is accounted for.

The training replay buffer reads at most 2,000 recent complete games and 256 MiB of source JSON. Original artifacts are retained. Parquet export streams records and writes compressed chunks rather than loading the entire collection. Exported rows retain stable complete-game-family splits. Evaluation, benchmark, held-out-list, historical-list, and unverified games never supply training labels. Eligibility must be explicit; missing legacy metadata is not approval. A permanent family-quarantine marker prevents a trusted-looking copy of a benchmark or QA game from entering training, even after the original source is archived. These markers travel with bundles.

## Transfer between computers

Keep active data and tracker databases on the computer's local drive. Use an already-mounted external SSD directory for finished bundles. An export is published only after all copied files, their checksums, relative paths, partitions, and engine/deck/feature compatibility pass validation. Private guides and live tracker databases are excluded.

```sh
python -m ptcg_lab.cli --data data export-bundle /path/to/mounted/ssd/run-bundle
python -m ptcg_lab.cli --data data import-bundle /path/to/mounted/ssd/run-bundle
```

Windows accepts a normal drive path such as `E:\ptcg-archives\run-bundle`. The destination must not already exist. Import returns an independent `dataRoot`; it does not merge imported files into an active experiment. Use that root explicitly:

```sh
python -m ptcg_lab.cli --data IMPORTED_ROOT train --epochs 4 --max-positions 10000 --resume IMPORTED_ROOT/models/CHECKPOINT.pt --linked-resume
```

Repeat the checkpoint's seed and maximum-position settings. Its data corpus must be identical. `--linked-resume` permits a changed device and assigns a new experiment ID with the parent checkpoint hash, restoring optimizer progress. Each checkpoint also preserves checksummed cumulative seed/family lineage across warm starts and resumes, so held-out evaluation cannot reuse an ancestor's training or validation seeds after the buffer moves on. Historical checkpoints without verified eligibility and complete lineage remain evidence and cannot be used as trusted policies or warm starts. Floating-point equivalence across machines is not promised. To train on a larger/new corpus use `--warm-start CHECKPOINT.pt`, which starts a new experiment and fresh optimizer instead.

Source absolute paths in historical reports remain provenance, not executable destinations. Select imported checkpoints by their path within the returned data root. A transfer interrupted before final publication leaves no successful bundle. Corrupt, incompatible, traversing, symlinked, or partition-conflicting bundles are rejected.

## Verification evidence

The September 19 implementation check ran 42 Python tests, including a real tiny-model training → export → import → linked optimizer continuation roundtrip. A separate actual simulator run completed two engineering-starter games, paused after the first, resumed the same run, and roundtripped five artifacts totaling 3.66 MB. Its monitored process-tree peak was approximately 337 MB. The starter engine fingerprint was `be0c29e9637ba8b3`; those games do not establish competitive-deck correctness or strength.

The same Mac diagnostic measured approximately 702 CPU versus 303 MPS training steps/second on the 38,532-parameter microbenchmark. These figures depend on concurrent load and are not long-run performance guarantees. CPU is the measured starting choice.
