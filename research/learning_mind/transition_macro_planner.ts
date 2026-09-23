import { Environment } from '../../packages/engine/src/environment';
import { legalActionKey } from '../../packages/engine/src/action-key';
import { CARD_FACTORIES } from '../../packages/engine/src/generated-catalog';
import { opponentHypotheses } from '../../packages/engine/src/hypotheses';
import { chooseAction } from '../../packages/engine/src/policies';
import { SeededRandom } from '../../packages/engine/src/random';
import type { LegalAction, Observation } from '../../packages/engine/src/types';
import { TrainerType } from '../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';

export const TRANSITION_MACRO_PLANNER_VERSION = 'transition-aware-public-determinization-v5-cjs-runtime';
export const TRANSITION_MACRO_MAX_CANDIDATES = 128;
export const TRANSITION_MACRO_MAX_STEPS = 3;
export const TRANSITION_MACRO_MAX_EXPANSION_PREFIXES = 4096;

export interface PlannedCandidate {
  actions: LegalAction[];
  semanticSlots: string[];
  completion: 'attack' | 'no-attack' | 'incomplete';
}

/** Mirrors the search action identity; equality is checked at every planned step. */
function searchActionKey(action: LegalAction): string {
  const ref = (value: LegalAction['sourceRef']) => value
    ? [value.playerId, value.zone, ['hand', 'prompt'].includes(value.zone) ? null : value.index ?? null]
    : null;
  return JSON.stringify([action.type, action.cardId ?? null, action.target ?? null, action.label,
    action.choiceOperation ?? null, action.selectionCount ?? null, action.amount ?? null, ref(action.sourceRef),
    ref(action.targetRef), (action.choiceRefs ?? []).map(choice => [
      ref(choice.sourceRef), ref(choice.targetRef), choice.cardId ?? null, choice.amount ?? null,
    ])]);
}

function candidateSlot(action: LegalAction): string | null {
  if (action.type === 'attack') return 'attack';
  if (action.type === 'attach-energy') return 'energy';
  if (/energy switch/i.test(action.label)) return 'energy';
  if (action.type === 'retreat' || /\bswitch\b/i.test(action.label)) return 'pivot';
  if (/fan|hammer|eri|stamp|red card/i.test(action.label)) return 'disruption';
  const card: any = action.cardId ? CARD_FACTORIES[action.cardId]?.() : undefined;
  if (card?.trainerType === TrainerType.SUPPORTER) return 'supporter';
  if (['ability', 'bench', 'evolve', 'attach-tool'].includes(action.type)) return action.type;
  if (action.type === 'play-trainer') return 'item';
  if (action.type === 'pass') return 'pass';
  return null;
}

function settlePrompts(env: Environment, actor: number, turn: number, seed: number): Observation | null {
  const promptRng = new SeededRandom(seed ^ 0x6a09e667);
  for (let count = 0; count < 64 && env.status === 'running'; count++) {
    const observation = env.observe();
    if (observation.playerId !== actor || observation.turn !== turn) return null;
    if (!observation.prompt) return observation;
    const action = chooseAction(observation, 'heuristic', promptRng);
    env.step(action.id);
  }
  return null;
}

function publicDeterminization(observation: Observation, seed: number) {
  if (!observation.searchPosition) throw new Error('macro-plan generation requires actor-visible searchPosition');
  const hypotheses = opponentHypotheses(observation.searchPosition);
  if (!hypotheses.length) throw new Error('macro-plan generation found no compatible public opponent hypothesis');
  const rng = new SeededRandom(seed);
  let sample = rng.uint32() / 0x100000000;
  const selected = hypotheses.find(item => (sample -= item.weight) < 0) ?? hypotheses.at(-1)!;
  const decisionIndex = Number(observation.legalActions[0]?.id.split(':')[0] ?? 0);
  const deck = selected.kind === 'unknown-variant' ? selected.deck : selected.id;
  return {
    selectedHypothesis: selected.id,
    create: () => Environment.fromPublicPosition(observation.searchPosition!, seed, deck, decisionIndex),
  };
}

function replayPlan(
  create: () => Environment,
  observation: Observation,
  actions: LegalAction[],
  seed: number,
): {env: Environment; decision: Observation | null} {
  const env = create();
  for (const [step, intended] of actions.entries()) {
    const current = settlePrompts(env, observation.playerId, observation.turn, seed);
    if (!current) return {env, decision: null};
    const intendedKey = legalActionKey(intended);
    if (searchActionKey(intended) !== intendedKey)
      throw new Error(`unsupported position: planner/search action-key mismatch for ${JSON.stringify(intended.label)}`);
    const matches = current.legalActions.filter(candidate => legalActionKey(candidate) === intendedKey);
    if (matches.length !== 1) return {env, decision: null};
    if (step > 0) {
      const searchMatches = current.legalActions.filter(candidate => searchActionKey(candidate) === searchActionKey(intended));
      const distinctBindings = new Set(searchMatches.map(legalActionKey));
      if (distinctBindings.size > 1)
        throw new Error(`unsupported position: search cannot uniquely re-resolve bound action ${JSON.stringify(intended.label)}`);
    }
    const action = matches[0];
    if (!action) return {env, decision: null};
    env.step(action.id);
  }
  return {env, decision: settlePrompts(env, observation.playerId, observation.turn, seed)};
}

/** Enumerate bounded legal plans by replaying each prefix from the same actor-visible determinization. */
export function generateTransitionMacroPlans(
  observation: Observation,
  seed: number,
  maxCandidates = TRANSITION_MACRO_MAX_CANDIDATES,
  maxSteps = TRANSITION_MACRO_MAX_STEPS,
): {version: string; hypothesisId: string; exploredPrefixCount: number; candidates: PlannedCandidate[]} {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('macro-plan seed must be uint32');
  if (!Number.isInteger(maxCandidates) || maxCandidates < 1 || maxCandidates > TRANSITION_MACRO_MAX_CANDIDATES)
    throw new Error('macro candidate cap must be between 1 and 128');
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > TRANSITION_MACRO_MAX_STEPS)
    throw new Error(`macro plan depth must be between 1 and ${TRANSITION_MACRO_MAX_STEPS}`);
  if (!observation.legalActions.length || observation.playerId !== observation.decisionPlayer)
    throw new Error('macro-plan generation requires the acting player and at least one legal action');
  const determinization = publicDeterminization(observation, seed);
  const root = determinization.create().observe(observation.playerId);
  const suppliedRoot = new Set(observation.legalActions.map(legalActionKey));
  const sampledRoot = new Set(root.legalActions.map(legalActionKey));
  const missingFromSample = observation.legalActions.filter(action => !sampledRoot.has(legalActionKey(action))).map(action => action.label);
  if (root.turn !== observation.turn || missingFromSample.length)
    throw new Error(`public determinization omitted actor-visible root actions (missing=${JSON.stringify(missingFromSample)})`);

  const candidates: PlannedCandidate[] = [];
  const queue: PlannedCandidate[] = [];
  const visitedPrefixes = new Set<string>();
  const pendingPrefixKeys = new Set<string>();
  const candidateKeys = new Set<string>();
  let deduplicatedPrefixes = 0;
  const candidateDepths = new Map<number, number>();
  const candidateIntents = new Map<string, number>();
  const expansionsBySlot = new Map<string, number>();
  const capFailure = () => {
    const diagnostic = {
      emitted: candidates.length,
      queued: queue.length,
      visitedPrefixes: visitedPrefixes.size,
      deduplicatedPrefixes,
      candidateDepths: Object.fromEntries([...candidateDepths].sort(([a], [b]) => a - b)),
      candidateIntents: Object.fromEntries([...candidateIntents].sort(([a], [b]) => a.localeCompare(b))),
      expansionsBySlot: Object.fromEntries([...expansionsBySlot].sort(([a], [b]) => a.localeCompare(b))),
    };
    return new Error(`unsupported position: complete macro candidate cap exceeded (${maxCandidates}); diagnostic=${JSON.stringify(diagnostic)}`);
  };
  const expansionFailure = () => new Error(`unsupported position: transition macro expansion prefix cap exceeded (${TRANSITION_MACRO_MAX_EXPANSION_PREFIXES}); diagnostic=${JSON.stringify({
    emittedCompleteCandidates: candidates.length, queued: queue.length, visitedPrefixes: visitedPrefixes.size,
  })}`);
  const enqueuePrefix = (prefix: PlannedCandidate) => {
    const key = JSON.stringify(prefix.actions.map(legalActionKey));
    if (visitedPrefixes.has(key) || pendingPrefixKeys.has(key)) {
      deduplicatedPrefixes++;
      return;
    }
    pendingPrefixKeys.add(key);
    queue.push(prefix);
    if (visitedPrefixes.size + pendingPrefixKeys.size > TRANSITION_MACRO_MAX_EXPANSION_PREFIXES)
      throw expansionFailure();
  };
  for (const action of root.legalActions.filter(action => suppliedRoot.has(legalActionKey(action)))
    .sort((left, right) => left.label.localeCompare(right.label) || left.id.localeCompare(right.id))) {
    const slot = candidateSlot(action) ?? 'other';
    enqueuePrefix({actions: [action], semanticSlots: [slot],
      completion: slot === 'attack' ? 'attack' : slot === 'pass' ? 'no-attack' : 'incomplete'});
  }
  while (queue.length) {
    const plan = queue.shift()!;
    const key = JSON.stringify(plan.actions.map(legalActionKey));
    pendingPrefixKeys.delete(key);
    if (visitedPrefixes.has(key)) continue;
    visitedPrefixes.add(key);
    const addCandidate = (candidate: PlannedCandidate) => {
      const candidateKey = JSON.stringify(candidate.actions.map(legalActionKey));
      if (candidateKeys.has(candidateKey)) return;
      candidateKeys.add(candidateKey);
      if (candidates.length >= maxCandidates) throw capFailure();
      candidates.push(candidate);
      candidateDepths.set(candidate.actions.length, (candidateDepths.get(candidate.actions.length) ?? 0) + 1);
      candidateIntents.set(candidate.completion, (candidateIntents.get(candidate.completion) ?? 0) + 1);
    };
    if (plan.completion !== 'incomplete') addCandidate(plan);
    if (plan.completion !== 'incomplete' || plan.actions.length >= maxSteps) continue;

    const {decision} = replayPlan(determinization.create, observation, plan.actions, seed);
    if (!decision) continue;
    const occupied = new Set(plan.semanticSlots.filter(slot => slot !== 'other'));
    const terminalOptions = decision.legalActions
      .filter(action => action.type === 'attack' || action.type === 'pass')
      .sort((left, right) => left.type.localeCompare(right.type)
        || left.label.localeCompare(right.label) || left.id.localeCompare(right.id));
    for (const action of terminalOptions) {
      const slot = candidateSlot(action)!;
      const completed: PlannedCandidate = {actions: [...plan.actions, action],
        semanticSlots: [...plan.semanticSlots, slot], completion: slot === 'attack' ? 'attack' : 'no-attack'};
      addCandidate(completed);
    }
    const options = decision.legalActions
      .filter(action => !action.type.includes('prompt') && action.type !== 'choice')
      .map(action => ({action, slot: candidateSlot(action)}))
      .filter((item): item is {action: LegalAction; slot: string} => !!item.slot && !occupied.has(item.slot)
          && item.slot !== 'attack' && item.slot !== 'pass')
      .sort((left, right) => left.slot.localeCompare(right.slot)
        || left.action.label.localeCompare(right.action.label) || left.action.id.localeCompare(right.action.id));
    for (const option of options) {
      enqueuePrefix({actions: [...plan.actions, option.action], semanticSlots: [...plan.semanticSlots, option.slot],
        completion: 'incomplete'});
      expansionsBySlot.set(option.slot, (expansionsBySlot.get(option.slot) ?? 0) + 1);
    }
  }
  if (!candidates.length) throw new Error('unsupported position: no complete attack or deliberate no-attack macro candidate');
  return {version: TRANSITION_MACRO_PLANNER_VERSION, hypothesisId: determinization.selectedHypothesis,
    exploredPrefixCount: visitedPrefixes.size, candidates};
}

if (process.argv[1]?.endsWith('/transition_macro_planner.ts')) {
  let input = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => input += chunk);
  process.stdin.on('end', () => {
    try {
      const request = JSON.parse(input);
      const response = generateTransitionMacroPlans(request.observation, request.seed,
        request.maxCandidates, request.maxSteps);
      process.stdout.write(JSON.stringify(response) + '\n');
    } catch (error) {
      process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    }
  });
}
