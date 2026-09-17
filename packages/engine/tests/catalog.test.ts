import test from 'node:test';
import assert from 'node:assert/strict';
import { getDecks, getDeck, validateDeck, CARD_FACTORIES } from '../src/catalog';

test('all five distinct starter lists contain 60 supported H/I/basic cards', () => {
  const decks = getDecks();
  assert.deepEqual(decks.map(d=>d.id), ['dragapult','raging-bolt','grimmsnarl','mega-lucario','crustle']);
  for (const deck of decks) {
    assert.equal(deck.cardCount,60);
    assert.deepEqual(validateDeck(deck),[]);
    assert.equal(deck.validation.status,'experimental');
  }
});

test('structural validation rejects missing cards, too many copies, and wrong engine mappings', () => {
  const deck = structuredClone(getDeck('crustle'));
  deck.cards[0].count=5;
  assert.ok(validateDeck(deck).some(e=>e.includes('60')));
  assert.ok(validateDeck(deck).some(e=>e.includes('four')));
  deck.cards[0].engineName='unknown';
  assert.ok(validateDeck(deck).some(e=>e.includes('mapping')));
  deck.formatDate='2027-09-17';
  assert.ok(validateDeck(deck).some(e=>e.includes('format freeze')));
});

test('Crustle registry resolves the requested Destined Rivals card and official labels', () => {
  const card = CARD_FACTORIES['DRI-12']();
  assert.equal(card.name,'Crustle');
  assert.equal(card.powers[0].name,'Mysterious Rock Inn');
  assert.equal(card.attacks[0].name,'Superb Scissors');
  assert.equal(card.regulationMark,'I');
});
