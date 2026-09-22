# P3 nonterminal-loop diagnosis

Status: diagnosed; no fix implemented.

Engine fingerprint: `twinleaf-adapter-0.1.0+bf3a9a0f133b99bd`  
Engine build SHA-256: `474549dda1d98fea5a7435c3abfdcccb4baa2663237ce8bb4aab99c40b36906e`

## Finding

P3 loops during the staged opening-Pokémon prompt. In both inspected cells it selects the only valid Basic Pokémon, then prefers `Undo last selection` over `Finish selection (1)`, returning to the identical visible setup position.

In the Crustle→Dragapult cell, P3 scored Undo at `0.00862` and Finish at `-0.00301`. In the Crustle mirror, it scored Undo at `0.01752` and Finish at `-0.02794`. The Python heuristic scored Undo at `-100` and Finish at `0` in both states.

The unguarded traces each executed 250 decisions without leaving turn 0: 125 selections and 125 undo operations. This reproduces the 1,000-decision pilot truncations.

## Attribution

- **Checkpoint:** supplies the immediate bad preference by ranking Undo above Finish in these staged-selection states.
- **Measurement adapter:** converts that preference into an infinite loop because its P3 path calls the agent directly and does not invoke the repository's existing `visible-repetition-v1` decision guard.
- **Decision guard:** works as intended. With the unchanged existing guard applied counterfactually, the cross-matchup game finished after 193 decisions on turn 21 and the mirror finished after 243 decisions on turn 77.
- **Engine:** is not the cause. It exposes legal append, undo and finish operations and correctly returns to the prior setup state after Undo. Both guarded traces reached genuine terminal results on the same worker build.

The current experimental learning driver already calls `forward_choices`, prevents Undo from being selected as a forward action, records attempts with `record_choice`, and withholds search targets on guarded fallback decisions. The research pilot adapter omitted that integration.

## Smallest proposed correction

In the research-only baseline measurement adapter, apply the existing `visible-repetition-v1` guard exactly as the learning-game driver does before asking P3 to choose. Record the guard version and filtered-decision count, never create a search target for a guarded choice, and continue to classify a genuinely mandatory cycle as truncated.

This requires no engine, checkpoint, feature, search, production-policy, or training change. After review, rerun only P3's eight pilot cells and recompute the full projection. No correction was implemented in this diagnosis milestone.
