import test from 'node:test';
import assert from 'node:assert/strict';
import { Environment } from '../src/environment';
import { search, continuationStep } from '../src/search';
import { chooseAction } from '../src/policies';
import { SeededRandom } from '../src/random';

function ready() {
  const env = new Environment(); env.reset(5, ['crustle', 'mega-lucario']);
  const rng = new SeededRandom(5);
  for (let i = 0; i < 50 && !env.observe().searchPosition; i++) env.step(chooseAction(env.observe(), 'heuristic', rng).id);
  assert.ok(env.observe().searchPosition); return env;
}

test('indistinguishable true hidden states produce identical public search input and samples', () => {
  const env = ready(); const viewer = env.actor; const observation = env.observe();
  const a = Environment.fromPublicPosition(observation.searchPosition!, 22, viewer === 0 ? 'mega-lucario' : 'crustle', env.decisionIndex);
  const opponent = env.store.state.players[1 - viewer];
  // Move a genuinely different hidden identity between hand and deck; preserve all public counts.
  [opponent.hand.cards[0], opponent.deck.cards[0]] = [opponent.deck.cards[0], opponent.hand.cards[0]];
  opponent.deck.cards.reverse(); opponent.prizes.reverse();
  const changed = env.observe();
  assert.deepEqual(changed, observation);
  const b = Environment.fromPublicPosition(changed.searchPosition!, 22, viewer === 0 ? 'mega-lucario' : 'crustle', env.decisionIndex);
  assert.deepEqual(a.observe(), b.observe());
  assert.deepEqual(a.store.state.players.map(p => p.deck.cards.map(c => c.fullName)), b.store.state.players.map(p => p.deck.cards.map(c => c.fullName)));
});

test('flat rollouts and information-set UCB execute bounded sampled games', {timeout: 30000}, () => {
  const env = ready(); const observation = env.observe(); const original = env.replay();
  for (const method of ['rollout', 'ismcts'] as const) {
    const result = search({observation, method, budgetMs: 10000, iterations: 3, maxRolloutDecisions: 4, seed: 44});
    assert.equal(result.status, 'complete'); assert.equal(result.iterations, 3);
    assert.equal(result.alternatives.reduce((n, a) => n + a.visits, 0), 3);
    assert.ok(result.alternatives.every(a => a.score === null || a.score >= 0 && a.score <= 1));
    assert.ok(result.warnings.some(w => /not calibrated/.test(w)));
    for (const alternative of result.alternatives) if (alternative.visits) {
      assert.equal(alternative.continuation?.conditional, true);
      assert.equal(alternative.continuation?.representative, true);
      assert.ok(alternative.continuation!.steps.length >= 1 && alternative.continuation!.steps.length <= 8);
      assert.equal(alternative.continuation!.steps[0].label, alternative.label);
      assert.ok(['cutoff', 'terminal'].includes(alternative.continuation!.end));
      assert.ok(alternative.continuation!.opponentArchetype);
    }
  }
  assert.deepEqual(env.replay(), original);
});

test('unfinished effects explicitly refuse search rather than expose original hidden state', () => {
  const env = new Environment(); env.reset(1, ['crustle', 'dragapult']);
  const result = search({observation: env.observe(), budgetMs: 100});
  assert.equal(result.status, 'unavailable'); assert.equal(result.iterations, 0);
});


test('sampled continuation redacts opponent private prompt selections', () => {
  const observation = ready().observe();
  const privateObservation = {...observation, playerId: 1 - observation.playerId, prompt: {type: 'Choose cards', message: 'CHOOSE_CARD_TO_HAND'}};
  const hiddenChoice = {id: '1:0', type: 'prompt', label: 'Choose SECRET_OPPONENT_HAND_CARD'};
  const shown = continuationStep(privateObservation, hiddenChoice, observation.playerId);
  assert.ok(!shown.label.includes('SECRET_OPPONENT_HAND_CARD'));
  assert.match(shown.label, /private choice omitted/);
  const publicAction = continuationStep(privateObservation, {id: '1:1', type: 'attack', label: 'Attack: Aura Jab'}, observation.playerId);
  assert.equal(publicAction.label, 'Attack: Aura Jab');
});
