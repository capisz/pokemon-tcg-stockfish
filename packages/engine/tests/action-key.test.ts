import test from 'node:test';
import assert from 'node:assert/strict';
import { legalActionKey } from '../src/action-key';
import type { LegalAction } from '../src/types';

const bound = (overrides: Partial<LegalAction> = {}): LegalAction => ({
  id: '4:0', type: 'ability', label: 'Use Example Ability', cardId: 'EXAMPLE-1', target: 'bench-0',
  sourceRef: {playerId: 0, zone: 'active', index: 0},
  targetRef: {playerId: 0, zone: 'bench', index: 0},
  ...overrides,
});

test('semantic action identity preserves distinct board target bindings', () => {
  assert.notEqual(legalActionKey(bound()), legalActionKey(bound({
    targetRef: {playerId: 0, zone: 'bench', index: 1},
  })));
});

test('semantic action identity collapses interchangeable hand copies only', () => {
  assert.equal(legalActionKey(bound({sourceRef: {playerId: 0, zone: 'hand', index: 0}})),
    legalActionKey(bound({sourceRef: {playerId: 0, zone: 'hand', index: 2}})));
  assert.notEqual(legalActionKey(bound()), legalActionKey(bound({
    sourceRef: {playerId: 0, zone: 'active', index: 1},
  })));
});

test('semantic action identity preserves staged selection semantics', () => {
  const original = bound({choiceOperation: 'append', selectionCount: 1,
    choiceRefs: [{sourceRef: {playerId: 0, zone: 'hand', index: 0}, cardId: 'BASIC-ENERGY', amount: 1}]});
  const changed = bound({choiceOperation: 'append', selectionCount: 1,
    choiceRefs: [{sourceRef: {playerId: 0, zone: 'hand', index: 0}, cardId: 'BASIC-ENERGY', amount: 2}]});
  assert.notEqual(legalActionKey(original), legalActionKey(changed));
  assert.notEqual(legalActionKey(original), legalActionKey(bound({...original, choiceOperation: 'finish'})));
});
