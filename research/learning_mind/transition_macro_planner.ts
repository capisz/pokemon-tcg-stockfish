import { Environment } from '../../packages/engine/src/environment';
import { legalActionKey } from '../../packages/engine/src/action-key';
import { CARD_FACTORIES } from '../../packages/engine/src/generated-catalog';
import { opponentHypotheses } from '../../packages/engine/src/hypotheses';
import { chooseAction } from '../../packages/engine/src/policies';
import { SeededRandom } from '../../packages/engine/src/random';
import type { LegalAction, Observation } from '../../packages/engine/src/types';
import { TrainerType } from '../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';

export const TRANSITION_MACRO_PLANNER_VERSION = 'transition-aware-public-determinization-v3-terminal-intent';
export const TRANSITION_MACRO_MAX_CANDIDATES = 128;
export const TRANSITION_MACRO_MAX_STEPS = 3;

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
): {version: string; hypothesisId: string; candidates: PlannedCandidate[]} {
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
  const queue: PlannedCandidate[] = root.legalActions.filter(action => suppliedRoot.has(legalActionKey(action)))
    .sort((left, right) => left.label.localeCompare(right.label) || left.id.localeCompare(right.id))
    .map(action => {
      const slot = candidateSlot(action) ?? 'other';
      return {actions: [action], semanticSlots: [slot],
        completion: slot === 'attack' ? 'attack' : slot === 'pass' ? 'no-attack' : 'incomplete'};
    });
  const visitedPrefixes = new Set<string>();
  const candidateKeys = new Set<string>();
  while (queue.length) {
    const plan = queue.shift()!;
    const key = JSON.stringify(plan.actions.map(legalActionKey));
    if (visitedPrefixes.has(key)) continue;
    visitedPrefixes.add(key);
    const addCandidate = (candidate: PlannedCandidate) => {
      const candidateKey = JSON.stringify(candidate.actions.map(legalActionKey));
      if (candidateKeys.has(candidateKey)) return;
      candidateKeys.add(candidateKey);
      if (candidates.length >= maxCandidates)
        throw new Error(`unsupported position: transition-aware macro cap exceeded (${maxCandidates})`);
      candidates.push(candidate);
    };
    addCandidate(plan);
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
      queue.push({actions: [...plan.actions, option.action], semanticSlots: [...plan.semanticSlots, option.slot],
        completion: 'incomplete'});
      if (candidates.length + queue.length > maxCandidates)
        throw new Error(`unsupported position: transition-aware macro cap exceeded (${maxCandidates})`);
    }
  }
  return {version: TRANSITION_MACRO_PLANNER_VERSION, hypothesisId: determinization.selectedHypothesis, candidates};
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
