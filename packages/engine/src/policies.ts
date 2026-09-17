import type { LegalAction, Observation } from './types';
import { SeededRandom } from './random';

/** Baseline deliberately accepts ONLY an observation. No Store or opponent hand access. */
export function chooseAction(observation: Observation, policy: 'random' | 'heuristic', rng: SeededRandom): LegalAction {
  const actions = observation.legalActions;
  if (!actions.length) throw new Error('Policy has no legal action.');
  if (policy === 'random') return actions[rng.int(actions.length)];
  const own = observation.players[observation.playerId];
  const opponent = observation.players[1 - observation.playerId];
  const score = (a: LegalAction): number => {
    if (a.type === 'prompt') {
      if (a.label === 'Cancel' || /no cards|no energy|Discard 0/.test(a.label)) return -10;
      if (a.label === 'Yes') return 8;
      if (a.label === 'No') return 0;
      const amount = Number(a.label.match(/(?:Attach|Discard) (\d+) energy/)?.[1] ?? 0);
      if (amount) return 5 + Math.min(amount, 5);
      if (observation.prompt?.message.includes('CHOOSE_STARTING')) {
        const n = a.label.split(', ').length;
        const keyBasic = /Dreepy|Dwebble|Riolu|Raging Bolt|Impidimp/.test(a.label) ? 2 : 0;
        return n * 3 + keyBasic;
      }
      if (/Choose cards/.test(observation.prompt?.type ?? '')) return a.label.split(', ').length;
      return 1;
    }
    if (a.type === 'evolve') return 50;
    if (a.type === 'attach-energy') {
      const pokemon = a.target === 'active' ? own.active : own.bench[Number(a.target?.split(' ')[1]) - 1];
      if (!pokemon) return 0;
      const count = pokemon.energy.length;
      return 40 - count * 8 + (a.target === 'active' ? 5 : 0);
    }
    if (a.type === 'bench') return own.bench.length < 3 ? 34 : 4;
    if (a.type === 'ability') return 30;
    if (a.type === 'attack') {
      const attack = own.active?.card.attacks?.find(x => a.label === `Attack: ${x.name}`);
      const damage = attack?.damage ?? 0;
      const ko = opponent.active && damage >= (opponent.active.card.hp ?? 999) - opponent.active.damage;
      return 20 + damage / 50 + (ko ? 30 : 0);
    }
    if (a.type === 'play-trainer') {
      if (/Switch|Boss's Orders|Hand Trimmer/.test(a.label)) return 1;
      if (/Lillie|Carmine/.test(a.label)) return own.handCount <= 4 ? 32 : 3;
      if (/Crispin|Rare Candy|Poffin|Ball|Gong|Signal/.test(a.label)) return 27;
      return 9;
    }
    if (a.type === 'attach-tool') return 15;
    if (a.type === 'retreat') return -5;
    if (a.type === 'pass') return -20;
    return 0;
  };
  const ranked = actions.map(action => ({action, value: score(action)}));
  const best = Math.max(...ranked.map(r => r.value));
  const tied = ranked.filter(r => r.value === best);
  return tied[rng.int(tied.length)].action;
}
