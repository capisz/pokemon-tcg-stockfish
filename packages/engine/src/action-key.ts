import type { LegalAction } from './types';

function refKey(ref: LegalAction['sourceRef']): unknown {
  if (!ref) return null;
  // Hand/prompt copies of the same card are interchangeable; board indices are not.
  return [ref.playerId, ref.zone, ['hand', 'prompt'].includes(ref.zone) ? null : ref.index ?? null];
}

/** Stable semantic identity for a legal action, preserving board and staged-choice bindings. */
export function legalActionKey(action: LegalAction): string {
  return JSON.stringify([
    action.type, action.cardId ?? null, action.target ?? null, action.label,
    action.choiceOperation ?? null, action.selectionCount ?? null, action.amount ?? null,
    refKey(action.sourceRef), refKey(action.targetRef),
    (action.choiceRefs ?? []).map(choice => [
      refKey(choice.sourceRef), refKey(choice.targetRef), choice.cardId ?? null, choice.amount ?? null,
    ]),
  ]);
}
