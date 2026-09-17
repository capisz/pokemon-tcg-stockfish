import test from 'node:test';
import assert from 'node:assert/strict';
import { SeededRandom } from '../src/random';
import { ShuffleDeckPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/shuffle-prompt';
import { State } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';

test('sixty-card permutations validate and deterministic generator does not overwrite host randomness', () => {
  const rng = new SeededRandom(88); const a = rng.shuffle(60);
  assert.equal(new Set(a).size, 60); assert.deepEqual([...a].sort((a, b) => a - b), Array.from({length: 60}, (_, i) => i));
  const state = new State(); const p = new Player(); p.id = 1; p.deck.cards = Array(60).fill(null); state.players = [p];
  assert.equal(new ShuffleDeckPrompt(1).validate(a, state), true);
  const previous = Math.random; assert.throws(() => rng.scoped(() => {throw new Error('test');})); assert.equal(Math.random, previous);
});
