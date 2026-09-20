import test from 'node:test';
import assert from 'node:assert/strict';
import { getDecks, getDeck, validateDeck, CARD_FACTORIES, getAgentDecks, listHash } from '../src/catalog';

test('five main lists and ten variants have immutable legal constructions and explicit audit gates', () => {
  const decks = getDecks();
  assert.equal(decks.length, 15);
  assert.equal(decks.filter(d => d.role === 'main').length, 5);
  assert.equal(decks.filter(d => d.role === 'training-variant').length, 5);
  assert.equal(decks.filter(d => d.role === 'heldout').length, 5);
  assert.equal(new Set(decks.map(d => d.listHash)).size, 15);
  for (const deck of decks) {
    assert.equal(deck.cardCount, 60, deck.id);
    assert.deepEqual(validateDeck(deck), [], deck.id);
    assert.equal(deck.listHash, listHash(deck), deck.id);
    assert.equal(deck.validation.trainingEligible, false, deck.id);
    assert.equal(deck.validation.legalityVerified, false, deck.id);
  }
  assert.equal(getAgentDecks().length, 10);
  assert.ok(getAgentDecks().every(d => d.role !== 'heldout' && d.role !== 'historical'));
  assert.equal(getDecks({includeHistorical: true}).filter(d => d.role === 'historical').length, 5);
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
