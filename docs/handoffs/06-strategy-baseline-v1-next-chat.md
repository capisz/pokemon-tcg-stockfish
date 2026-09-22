# Strategy Baseline v1 — next-chat execution plan

Date: 2026-09-21  
Status: ready for a fresh Codex chat  
Immediate milestone: correct only the research measurement adapter's missing P3 repetition guard, rerun P3's eight-cell pilot, update the full baseline projection, and stop.

## 1. Repository state

- Canonical repository: `/Users/admin/Documents/ChatGPT/Pokemon Ai project`
- Continue in the existing isolated worktree: `/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project`
- Branch: `codex/strategy-contract-v1.2-prebaseline`
- Current HEAD: `ec8e153` (`Compact P3 diagnostic evidence`)
- Working tree was clean when this handoff was written.
- Do not switch back to or edit the canonical `codex/competitive-integration` checkout. It has an unrelated untracked `Claude outputs/` directory that must be preserved.
- Nothing in this worktree has been merged or pushed.

Relevant commits, newest last in logical order:

1. `752f0ce` — draft strategy v1.2 and pre-baseline rule fixes.
2. `831c56b` — make the v1.2 status test approval-safe.
3. `14689d3` — approve v1.2; validator selects v1.2.
4. `828d207` — initial P1/P2/P4 timing pilot.
5. `557abcd` — P3 compatibility pilot; checkpoint loads but all games truncate.
6. `71c1f45` — diagnose the P3 setup repetition loop.
7. `ec8e153` — compact the diagnostic evidence.

## 2. Frozen facts — do not rediscover

Strategy v1.2 is approved and active. The validator previously reported:

```text
Validated strategy-contract-v1.2: 2 specialist overlays, status approved.
Active strategy revision: v1.2
```

Engine and lists:

- Engine fingerprint: `twinleaf-adapter-0.1.0+bf3a9a0f133b99bd`
- Worker SHA-256: `474549dda1d98fea5a7435c3abfdcccb4baa2663237ce8bb4aab99c40b36906e`
- Crustle v3 list hash: `f454a2c2f3bd329f6d75bb8344ba09000db93e220e8a07c3e0502e5a7ddd8343`
- Dragapult list hash: `2e40a150bc968b9e67a9c7e57f6b9ebddb72591a41ae7e3a9f7b855db92318b9`

Guide checkpoint:

- Path: `/Users/admin/Documents/ChatGPT/Pokemon Ai project/data/competitive/experimental/models/09572b53a4e74d9193c101bab05b30da-guide.pt`
- SHA-256: `1ff34f893540373ebb2da6433b3b67a0cdbcb2b711220cf08767cdeb85a08e6f`
- Compatible interpreter: `/Users/admin/Documents/ChatGPT/Pokemon Ai project/.venv/bin/python`
- Set `PYTHONPATH` to the isolated worktree's `src` when using that interpreter.
- Python 3.10.20; PyTorch 2.14.0.
- Checkpoint: policy-only, 38,532 parameters, no value head, feature version `visible-energy-coverage-v3`.
- Do not install or update dependencies.

Initial timing pilot:

- P1/P2/P4: 24/24 rules-terminal games, no errors or truncations.
- Valid 288-game projection: 38.25 serial-equivalent minutes.
- P4 selected a measured search action on 80 of 2,079 decisions (3.85%); its fallback is Python `heuristic_action_score`.
- These pilot outcomes are timing data, not strength estimates.

P3 failure and diagnosis:

- All eight unguarded P3 pilot cells hit the 1,000-decision cap.
- Two traced cells alternate `Select <Basic>` and `Undo last selection` without leaving setup turn 0.
- Cross trace: P3 scores Undo `0.00862`, Finish `-0.00301`.
- Mirror trace: P3 scores Undo `0.01752`, Finish `-0.02794`.
- Python heuristic scores Undo `-100` and Finish `0`.
- The engine is correct. The checkpoint supplies the bad immediate preference.
- The research pilot becomes nonterminal because it bypasses the existing `visible-repetition-v1` guard.
- The normal experimental learning driver already uses that guard.
- Counterfactual guarded traces reached genuine terminal outcomes in 193 and 243 decisions.
- Existing decision-guard tests: 3 passed.

Primary evidence:

- `docs/validation/strategy-baseline-v1-p3-loop-diagnosis-2026-09-21/DIAGNOSIS.md`
- `docs/validation/strategy-baseline-v1-p3-loop-diagnosis-2026-09-21/diagnosis.json`
- `docs/validation/strategy-baseline-v1-pilot-2026-09-21/`
- `docs/validation/strategy-baseline-v1-complete-pilot-2026-09-21/`

## 3. Immediate milestone — research adapter correction

This is the only implementation authorized in the next chat.

### Scope

Modify only the research measurement path, principally:

- `research/strategy_baseline/pilot.py`
- new focused research-adapter tests if needed
- a new validation output directory for the corrected P3 pilot

Do not change:

- `src/ptcg_lab/decision_guard.py` behavior;
- the checkpoint or any model weights;
- TypeScript or Python production heuristics;
- engine rules or gameplay;
- search behavior or budgets;
- features, training, labels, or files under `data/competitive`;
- approved strategy v1, v1.1, or v1.2 content;
- historical pilot or diagnostic evidence.

### Required implementation

Use the existing guard, not a new loop heuristic:

```python
from ptcg_lab.decision_guard import (
    VERSION as GUARD_VERSION,
    forward_choices,
    record_choice,
)
```

For P3 decisions in `run_python_game`:

1. Maintain one guard state per game.
2. Call `forward_choices` on the unmodified actor observation before P3 chooses.
3. Give P3 the returned safe observation.
4. Resolve the chosen ID back to the original legal action.
5. Call `record_choice` after the choice is accepted.
6. Record guard version, whether the decision was filtered, and an aggregate filtered-decision count in the pilot result.
7. Never create a search target from a guarded decision.
8. Preserve truncation if a genuinely mandatory cycle remains.
9. Leave P1, P2, and P4 frozen and do not rerun them.

The guard excludes Undo from forward staged-choice actions and retires a twice-tried unchanged action while another forward action exists. It must not mutate the engine's legal-action set.

### Tests first

Add focused tests that fail before the adapter correction and pass afterward:

1. A staged selection containing Finish and Undo sends only the forward choice to P3.
2. The chosen guarded action is resolved against the original legal-action list and recorded.
3. Guard version and filtered-decision count appear in the result.
4. A guarded P3 decision does not create a search target.
5. A single mandatory legal action is retained; no fake terminal result is created.
6. Existing `tests/python/test_decision_guard.py` remains green.

Tests should use small fakes where possible so the worktree's lightweight `.venv` can run them without PyTorch.

### Corrected P3 pilot

Use the original eight cells and seeds. Do not rerun P1/P2/P4. Merge P3 with the original valid result file:

`docs/validation/strategy-baseline-v1-pilot-2026-09-21/pilot-results.json`

Run P3 with the compatible canonical interpreter and isolated source:

```bash
cd "/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project"

PYTHONPATH="/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project/src" \
  "/Users/admin/Documents/ChatGPT/Pokemon Ai project/.venv/bin/python" \
  research/strategy_baseline/pilot.py \
  --root "/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project" \
  --output "/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project/docs/validation/strategy-baseline-v1-guarded-p3-pilot-2026-09-21" \
  --guide-checkpoint "/Users/admin/Documents/ChatGPT/Pokemon Ai project/data/competitive/experimental/models/09572b53a4e74d9193c101bab05b30da-guide.pt" \
  --policies P3 \
  --prior-results "/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project/docs/validation/strategy-baseline-v1-pilot-2026-09-21/pilot-results.json"
```

### Stop conditions

Stop without starting the main baseline if any of these occurs:

- any P3 pilot game truncates or errors;
- checkpoint, worker, list, effective-contract, cell, or seed identity differs;
- the guard changes P1/P2/P4 data;
- the complete runtime projection cannot be computed honestly;
- the required fix expands beyond the research adapter.

### Completion report

Report and commit:

- focused test output;
- P3 8-cell terminal/truncated/error table;
- P3 guard-filter counts;
- P3 mean/median timing and weighted 96-game projection;
- combined P1–P4 projected time;
- the frozen final sample-plan decision:
  - 384 games if projected time is at most 60 minutes;
  - 192 games, halving every cell before the run, if it exceeds 60 minutes;
- exact new artifact paths and commit hash.

Then stop. Do not start the full baseline in the same milestone.

## 4. Following milestone — full Strategy Baseline v1

Only begin this after the corrected P3 pilot passes and the user explicitly approves the frozen 384- or 192-game plan.

### Freeze inputs

- Generate effective v1 → v1.1 → v1.2 contract and hash.
- Preserve the checkpoint path/hash, engine fingerprint/build hash, deck hashes, full seed list, cells, and per-cell counts.
- Never alter the sample plan after observing outcomes.

### Four frozen policies

- P1: TypeScript `choose` heuristic.
- P2: Python `heuristic_action_score`.
- P3: frozen guide-policy checkpoint through the approved guarded research adapter.
- P4: unchanged ISMCTS at 200 ms through the existing driver; record searched/fallback decision tags and iterations.
- Each policy plays itself in both seats with matched cell/game seeds.

### Principle probes

- Build 1–3 automatic trigger/behavior checks for every principle in the effective contract.
- Use only acting-player information.
- Unit-test probes on fixed positions.
- Mark dependencies lacking focused rules tests as `unverified rules`.
- Headline rate: first qualifying decision per probe per game-side.
- Secondary rate: all qualifying decisions.
- Report `k/n` with Wilson 95% intervals; `n < 20` is insufficient.
- Include 2–3 replay/decision examples per probe.
- Split P4 searched versus fallback adherence where sample size permits.

### Teaching reviews

Grade only the four exact-compatible review records:

- `530af431…`
- `83223955…`
- `b6a6c652…`
- `da82cf91…`

Do not grade or relabel these four stale Crustle records; list them for human re-review:

- `1e803b1a…`
- `49f4021a…`
- `c0d1cfe4…`
- `f34b1c1b…`

The stale records differ because they encode the older 3 Eri / 1 Xerosic list rather than approved Crustle v3's 2 Eri / 2 Xerosic counts.

### Context metrics and reporting

- W/D/L separately—never fold truncations into draws.
- Wilson intervals by perspective, seat, and first player.
- Game length, deck-outs, guard interventions, search availability, searched fraction, and iterations/search.
- Preferred/acceptable review hit rate and abstentions, per record and per unique position.
- Up to ten unlabeled draft scenario positions, prioritized by the worst probe failures.
- Five largest strategic gaps ranked by frequency × severity, each with a proposed next milestone.

Deliver:

- `docs/BASELINE-STRATEGY-V1.md`
- frozen-input record and generated effective contract
- machine-readable results
- probe tests
- ranked strategic-gap list

No heuristic, search, feature, model, training, or production gameplay changes are allowed during measurement.

## 5. Human decision after the baseline

After reviewing the baseline, choose one bounded improvement milestone. Likely candidates are:

1. staged-choice policy representation or policy training correction if P3 still systematically mishandles forward/undo/finish semantics despite the safety guard;
2. search-position availability if P4 remains mostly fallback;
3. the highest-frequency Crustle or Dragapult principle failure;
4. manual re-review of the four stale Crustle teaching positions.

Do not train or promote a new checkpoint until the user approves the selected gap, its success metric, its regression set, and its promotion gate.

## 6. Paste this into the new chat

```text
Continue the Pokémon TCG Stockfish lab from the existing isolated worktree:
/Users/admin/.codex/worktrees/strategy-v12-prebaseline/Pokemon Ai project

Read docs/handoffs/06-strategy-baseline-v1-next-chat.md completely before acting. Current branch is codex/strategy-contract-v1.2-prebaseline at ec8e153 or a direct descendant. Preserve the canonical checkout and its unrelated untracked files.

Execute only Section 3, “Immediate milestone — research adapter correction.” Write failing focused tests first, integrate the existing visible-repetition-v1 guard into the research-only P3 measurement adapter, rerun only P3's original eight pilot cells with the frozen checkpoint and seeds, merge those measurements with the preserved P1/P2/P4 pilot results, compute the honest complete projection and freeze the 384- or 192-game recommendation. Do not start the full baseline, change production behavior, modify the checkpoint, train, relabel teaching reviews, touch data/competitive, merge, or push. Commit the bounded work, report evidence, and stop.
```
