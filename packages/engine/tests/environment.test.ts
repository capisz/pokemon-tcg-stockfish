import test from 'node:test';
import assert from 'node:assert/strict';
import { Environment } from '../src/environment';
import { chooseAction } from '../src/policies';
import { SeededRandom } from '../src/random';

function play(env: Environment, n: number, seed = 51) {
  const rng = new SeededRandom(seed);
  for (let i = 0; i < n && env.status === 'running'; i++) env.step(chooseAction(env.observe(), 'heuristic', rng).id);
}

test('seeded chance and decision replay produce identical results', () => {
  const a = new Environment(); const b = new Environment();
  a.reset(12, ['crustle', 'mega-lucario']); b.reset(12, ['crustle', 'mega-lucario']);
  play(a, 35); play(b, 35);
  assert.deepEqual(a.replay(), b.replay());
  const c = new Environment(); c.reset(13, ['crustle', 'mega-lucario']); play(c, 35);
  assert.notDeepEqual(a.replay().chance, c.replay().chance);
});

test('branch reconstructs live prompt callbacks and does not mutate source', () => {
  const env = new Environment(); env.reset(9, ['dragapult', 'grimmsnarl']);
  const rng = new SeededRandom(9);
  for (let i = 0; i < 40 && (!env.observe().prompt || env.decisionIndex < 10); i++) env.step(chooseAction(env.observe(), 'heuristic', rng).id);
  const before = env.replay(); const branch = env.branch();
  assert.deepEqual(branch.observe(), env.observe());
  const id = branch.observe().legalActions[0].id;
  branch.step(id); assert.deepEqual(env.replay(), before);
  env.step(id); assert.deepEqual(branch.observe(), env.observe());
});

test('opponent hand, prizes and deck order never appear in observations', () => {
  const env = new Environment(); env.reset(15, ['crustle', 'mega-lucario']); play(env, 12);
  const viewer = 1 - env.actor;
  const before = env.observe(viewer);
  const other = env.store.state.players[1 - viewer];
  other.hand.cards.reverse(); other.deck.cards.reverse(); other.prizes.reverse();
  // Reorder private state and invalidate cache without changing any observable quantities.
  assert.deepEqual(env.observe(viewer), before);
  assert.equal(before.players[1 - viewer].hand.length, 0);
  assert.equal(before.legalActions.length, 0);
  assert.ok(!('prizes' in before.players[viewer]));
  assert.ok(!('deck' in before.players[viewer]));
});

test('bounded runs are truncated and stale actions leave the state intact', () => {
  const env = new Environment(); env.reset(0, ['crustle', 'crustle']);
  const before = env.replay();
  assert.equal(before.status, 'truncated'); assert.equal(before.outcome, null);
  assert.throws(() => env.step('invalid'));
  assert.deepEqual(env.replay(), before);
});

test('all five archetypes reach real terminal games with the baseline', {timeout: 120000}, () => {
  const decks = ['dragapult', 'raging-bolt', 'grimmsnarl', 'mega-lucario', 'crustle'];
  for (let i = 0; i < decks.length; i++) {
    const env = new Environment(); env.reset(5, [decks[i], decks[(i + 1) % decks.length]]);
    play(env, 600, (5 ^ 0x13579bdf) >>> 0);
    const replay = env.replay();
    assert.equal(replay.status, 'finished', decks[i]);
    assert.ok(replay.outcome?.winner === 0 || replay.outcome?.winner === 1, decks[i]);
    assert.ok(replay.frames.length > 10);
  }
});


test('all card identities are conserved at stable decisions', () => {
  const env = new Environment(); env.reset(14, ['crustle', 'grimmsnarl']);
  const rng = new SeededRandom(14);
  for (let i = 0; i < 80 && env.status === 'running'; i++) {
    if (!env.observe().prompt) for (const p of env.store.state.players) {
      const cards = [...p.deck.cards, ...p.hand.cards, ...p.discard.cards, ...p.lostzone.cards, ...p.supporter.cards, ...p.stadium.cards,
        ...p.prizes.flatMap(pr => pr.cards), ...p.active.cards, ...p.active.tools, ...p.active.energies.cards,
        ...p.bench.flatMap(b => [...b.cards, ...b.tools, ...b.energies.cards])];
      assert.equal(new Set(cards).size, 60);
    }
    env.step(chooseAction(env.observe(), 'heuristic', rng).id);
  }
});
