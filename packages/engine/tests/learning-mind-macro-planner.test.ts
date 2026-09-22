import test from 'node:test';
import assert from 'node:assert/strict';
import { Environment } from '../src/environment';
import { chooseAction } from '../src/policies';
import { SeededRandom } from '../src/random';
import { generateTransitionMacroPlans } from '../../../research/learning_mind/transition_macro_planner';

function readyObservation() {
  const env = new Environment();
  env.reset(5, ['crustle', 'mega-lucario']);
  const rng = new SeededRandom(5);
  let observation = env.observe();
  for (let count = 0; count < 50; count++) {
    observation = env.observe();
    if (observation.searchPosition) break;
    env.step(chooseAction(observation, 'heuristic', rng).id);
  }
  assert.ok(observation.searchPosition);
  assert.ok(observation.legalActions.length >= 3);
  return observation;
}

test('transition-aware macro planner emits deterministic legal prefixes with root coverage', () => {
  const observation = readyObservation();
  const roots = observation.legalActions.slice(0, 3);
  const narrowed = {...observation, legalActions: roots};
  const generated = generateTransitionMacroPlans(narrowed, 42, 128, 2);
  assert.deepEqual(generated, generateTransitionMacroPlans(narrowed, 42, 128, 2));
  assert.deepEqual(generated.candidates.filter(candidate => candidate.actions.length === 1)
    .map(candidate => candidate.actions[0].id).sort(), roots.map(action => action.id).sort());
  assert.ok(generated.candidates.every(candidate => candidate.actions.length >= 1 && candidate.actions.length <= 2));
});

test('transition-aware macro planner fails closed when the plan cap is exceeded', () => {
  const observation = readyObservation();
  assert.throws(() => generateTransitionMacroPlans(observation, 42, 1, 2), /cap exceeded/);
});
