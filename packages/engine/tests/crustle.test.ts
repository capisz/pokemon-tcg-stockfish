import test from 'node:test';
import assert from 'node:assert/strict';
import { Store } from '../../../vendor/twinleaf/ptcg-server/src/game/store/store';
import { State, GamePhase, GameWinner } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { AttackEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/game-effects';
import { PutDamageEffect, PutCountersEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/attack-effects';
import { initNextTurn } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effect-reducers/game-phase-effect';
import { WalkingWakeex } from '../../../vendor/twinleaf/ptcg-server/src/sets/10-scarlet-and-violet/set-temporal-forces/walking-wake-ex';
import { PlayCardAction, PlayerType, SlotType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { ResolvePromptAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/resolve-prompt-action';
import { promptChoices } from '../src/choices';
import { CARD_FACTORIES, registerCards } from '../src/catalog';

registerCards();
function board(sourceId = 'TEF-123') {
  const store = new Store({onStateChange: () => {}}); const state = new State();
  const attacker = new Player(); attacker.id = 1;
  const defender = new Player(); defender.id = 2;
  attacker.active.cards = [CARD_FACTORIES[sourceId]()];
  defender.active.cards = [CARD_FACTORIES['DRI-12']()];
  attacker.prizes = Array.from({length: 6}, () => {const p = new CardList(); p.cards = [CARD_FACTORIES['SVE-1']()]; return p;});
  defender.prizes = Array.from({length: 6}, () => {const p = new CardList(); p.cards = [CARD_FACTORIES['SVE-1']()]; return p;});
  state.players = [attacker, defender]; state.phase = GamePhase.ATTACK; state.turn = 3;
  store.state = state;
  const attack = () => new AttackEffect(attacker, defender, attacker.active.getPokemonCard()!.attacks[0]);
  return {store, state, attacker, defender, attack};
}

test('Crustle prevents attack damage from opposing ex but not damage counters', () => {
  const b = board();
  const damage = new PutDamageEffect(b.attack(), 100); b.store.reduceEffect(b.state, damage);
  assert.equal(damage.preventDefault, true); assert.equal(b.defender.active.damage, 0);
  b.store.reduceEffect(b.state, new PutCountersEffect(b.attack(), 30));
  assert.equal(b.defender.active.damage, 30);
});

test('Crustle takes non-ex damage and protection disappears under ability suppression', () => {
  const nonEx = board('TWM-107');
  nonEx.store.reduceEffect(nonEx.state, new PutDamageEffect(nonEx.attack(), 40));
  assert.equal(nonEx.defender.active.damage, 40);
  const suppressed = board(); suppressed.defender.active.noAbilities = true;
  suppressed.store.reduceEffect(suppressed.state, new PutDamageEffect(suppressed.attack(), 40));
  assert.equal(suppressed.defender.active.damage, 40);
});

test('actual Walking Wake ex ability bypasses Crustle protection', () => {
  const b = board(); b.attacker.active.cards = [new WalkingWakeex()];
  b.store.reduceEffect(b.state, b.attack());
  assert.equal(b.defender.active.damage, 120);
});

test('genuine failed draw ends game; a merely empty deck does not', () => {
  const b = board(); assert.equal(b.state.winner, GameWinner.NONE);
  b.state.phase = GamePhase.BETWEEN_TURNS; b.state.activePlayer = 1;
  initNextTurn(b.store, b.state);
  assert.equal(b.state.phase, GamePhase.FINISHED);
  assert.equal(b.state.winner, GameWinner.PLAYER_2);
});


test('Night Stretcher recovers Crustle from public discard and cannot fail to find', () => {
  const b = board(); b.state.phase = GamePhase.PLAYER_TURN;
  const crustle = CARD_FACTORIES['DRI-12'](); crustle.id = 101;
  b.attacker.discard.cards = [crustle]; b.attacker.hand.cards = [CARD_FACTORIES['SFA-61']()];
  b.store.dispatch(new PlayCardAction(1, 0, {player: PlayerType.BOTTOM_PLAYER, slot: SlotType.ACTIVE, index: 0}));
  let p = b.state.prompts.find(p => p.result === undefined)!;
  assert.equal(p.type, 'Choose cards');
  const choices = promptChoices(p, b.state).choices;
  assert.ok(choices.every(c => c.raw.length === 1));
  b.store.dispatch(new ResolvePromptAction(p.id, p.decode(choices[0].raw, b.state)));
  p = b.state.prompts.find(p => p.result === undefined)!;
  assert.equal(p.type, 'Show cards'); b.store.dispatch(new ResolvePromptAction(p.id, true));
  assert.ok(b.attacker.hand.cards.includes(crustle));
  assert.ok(!b.attacker.discard.cards.includes(crustle));
  assert.ok(b.attacker.discard.cards.some(c => c.name === 'Night Stretcher'));
});
