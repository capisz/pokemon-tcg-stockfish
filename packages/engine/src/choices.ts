import { State } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Prompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/prompt';
import { PlayerType, SlotType, CardTarget } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { StateUtils } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state-utils';
import { matchesPromptFilter } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/prompt-card-filter';
import { SuperType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';

export interface PromptChoice { raw: any; label: string }
export const CHOICE_LIMIT = 256;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

export function targetsFor(state: State, prompt: any): CardTarget[] {
  const result: CardTarget[] = [];
  for (const side of [PlayerType.BOTTOM_PLAYER, PlayerType.TOP_PLAYER]) {
    if (prompt.playerType !== PlayerType.ANY && side !== prompt.playerType) continue;
    const p = state.players.find(p => side === PlayerType.BOTTOM_PLAYER ? p.id === prompt.getPerspectivePlayerId() : p.id !== prompt.getPerspectivePlayerId());
    if (!p) continue;
    for (const slot of prompt.slots ?? [SlotType.ACTIVE, SlotType.BENCH]) {
      if (slot === SlotType.ACTIVE && p.active.cards.length) result.push({player: side, slot, index: 0});
      if (slot === SlotType.BENCH) p.bench.forEach((b, index) => { if (b.cards.length) result.push({player: side, slot, index}); });
    }
  }
  return result;
}
export function targetLabel(state: State, prompt: any, target: CardTarget): string {
  const p = state.players.find(p => p.id === prompt.getPerspectivePlayerId())!;
  const card = StateUtils.getTarget(state, p, target).getPokemonCard();
  return `${target.player === PlayerType.BOTTOM_PLAYER ? 'own' : 'opponent'} ${target.slot === SlotType.ACTIVE ? 'active' : `bench ${target.index + 1}`} ${card?.name ?? ''}`.trim();
}

/** Bounded combinator; reports restricted action coverage, never pretends exhaustive search. */
export function promptChoices(prompt: Prompt<any>, state: State): {choices: PromptChoice[]; warnings: string[]} {
  const p: any = prompt;
  const choices: PromptChoice[] = [];
  const warnings: string[] = [];
  let overflow = false;
  const add = (raw: any, label: string) => {
    if (choices.length >= CHOICE_LIMIT) { overflow = true; return; }
    try {
      const decoded = p.decode(raw, state);
      if (p.validate(decoded, state) && !choices.some(c => same(c.raw, raw))) choices.push({raw, label});
    } catch { /* rejected candidates are never presented as legal */ }
  };
  let work = 0;
  const subsets = <T>(items: T[], min: number, max: number, cb: (xs: T[]) => void) => {
    const visit = (start: number, count: number, chosen: T[]) => {
      if (choices.length >= CHOICE_LIMIT || ++work > 20000) { overflow = true; return; }
      if (chosen.length === count) { cb(chosen); return; }
      for (let i = start; i <= items.length - (count - chosen.length); i++) visit(i + 1, count, [...chosen, items[i]]);
    };
    for (let count = Math.min(max, items.length); count >= min; count--) visit(0, count, []);
  };
  const opts = p.options ?? {};
  const cancel = () => { if (opts.allowCancel) add(null, 'Cancel'); };
  switch (p.type) {
    case 'Confirm': add(true, 'Yes'); add(false, 'No'); break;
    case 'Select': case 'SelectOption':
      p.values.forEach((v: string, i: number) => { if (!opts.disabled?.[i]) add(i, String(v)); }); cancel(); break;
    case 'Alert': case 'Confirm cards': case 'Show cards': case 'Show mulligan':
      add(true, 'Acknowledge'); break;
    case 'Choose cards': {
      const eligible = p.cards.cards.map((card: any, i: number) => ({card, i})).filter(({card, i}: any) => !opts.blocked?.includes(i) && matchesPromptFilter(card, p.filter));
      subsets(eligible, opts.min, opts.max, (selected: any[]) => {
        const submit = (xs: any[]) => add(xs.map(x => x.i), xs.length ? `Choose ${xs.map(x => opts.isSecret ? 'hidden card' : x.card.name).join(', ')}` : 'Choose no cards');
        submit(selected);
        // Setup order determines the active Pokémon. Cover each possible active.
        if (String(p.message).includes('CHOOSE_STARTING')) for (let i = 1; i < selected.length; i++) submit([selected[i], ...selected.filter((_, j) => j !== i)]);
      });
      cancel(); break;
    }
    case 'Choose pokemon': {
      const ts = targetsFor(state, p).filter(t => !opts.blocked?.some((b: any) => same(b, t)));
      subsets(ts, opts.min, opts.max, xs => add(xs, xs.length ? `Choose ${xs.map(t => targetLabel(state, p, t)).join(', ')}` : 'Choose no Pokémon'));
      cancel(); break;
    }
    case 'Choose prize': {
      const owner = state.players.find(pl => opts.useOpponentPrizes ? pl.id !== p.getPerspectivePlayerId() : pl.id === p.getPerspectivePlayerId())!;
      const indexes = owner.prizes.filter(pr => pr.cards.length).map((_, i) => i).filter(i => !opts.blocked?.includes(i));
      const n = Math.min(opts.count, indexes.length);
      subsets(indexes, n, n, xs => add(xs, `Take prize slots ${xs.map(i => i + 1).join(', ')}`)); cancel(); break;
    }
    case 'Choose energy': {
      subsets(p.energy.map((_: any, i: number) => i), 0, p.energy.length, (xs: number[]) => add(xs, `Pay with ${xs.map(i => p.energy[i].card.name).join(', ') || 'no energy'}`)); cancel(); break;
    }
    case 'Choose attack':
      p.cards.forEach((c: any, i: number) => c.attacks.forEach((a: any) => add({index: i, attack: a.name}, `Use ${c.name}: ${a.name}`))); cancel(); break;
    case 'Order cards': {
      const indexes = p.cards.cards.map((_: any, i: number) => i);
      const permute = (rest: number[], chosen: number[]) => {
        if (choices.length >= CHOICE_LIMIT) { overflow = true; return; }
        if (!rest.length) { add(chosen, `Order ${chosen.map(i => p.cards.cards[i].name).join(', ')}`); return; }
        rest.forEach((x, i) => permute(rest.filter((_, j) => i !== j), [...chosen, x]));
      }; permute(indexes, []); cancel(); break;
    }
    case 'Attach energy': {
      const ts = targetsFor(state, p).filter(t => !opts.blockedTo?.some((b: any) => same(b, t)));
      const energy = p.cardList.cards.map((card: any, i: number) => ({card, i})).filter(({card, i}: any) => card.superType === SuperType.ENERGY && !opts.blocked?.includes(i) && matchesPromptFilter(card, p.filter));
      // Enumerate grouped allocations first so high attachment counts are usable even when capped.
      subsets(energy, opts.min, opts.max, (selected: any[]) => {
        if (!selected.length) { add([], 'Attach no energy'); return; }
        const emit = (assign: any[]) => add(assign, `Attach ${selected.length} energy: ${assign.map(a => `${p.cardList.cards[a.index].name} to ${targetLabel(state, p, a.to)}`).join('; ')}`);
        for (const t of ts) emit(selected.map(e => ({index: e.i, to: t})));
        const assign = (index: number, assigned: any[]) => {
          if (choices.length >= CHOICE_LIMIT || ++work > 20000) { overflow = true; return; }
          if (index === selected.length) { emit(assigned); return; }
          ts.forEach(t => assign(index + 1, [...assigned, {index: selected[index].i, to: t}]));
        };
        if (!opts.sameTarget) assign(0, []);
      });
      if (opts.min === 0) add([], 'Attach no energy'); cancel(); break;
    }
    case 'Discard energy': {
      const owner = state.players.find(pl => pl.id === p.getPerspectivePlayerId())!;
      const cards: any[] = [];
      targetsFor(state, p).filter(t => !opts.blockedFrom?.some((b: any) => same(t, b))).forEach(t => {
        const source = StateUtils.getTarget(state, owner, t);
        source.cards.forEach((c, index) => {
          const blocked = opts.blockedMap?.find((m: any) => same(m.source, t))?.blocked ?? [];
          if (c.superType === SuperType.ENERGY && !blocked.includes(index) && matchesPromptFilter(c, p.filter)) cards.push({from: t, index});
        });
      });
      subsets(cards, opts.min, opts.max ?? cards.length, xs => add(xs, `Discard ${xs.length} energy${xs.length ? ` from ${xs.map(x => targetLabel(state, p, x.from)).join(', ')}` : ''}`));
      cancel(); break;
    }
    case 'Move energy': {
      const owner = state.players.find(pl => pl.id === p.getPerspectivePlayerId())!;
      const ts = targetsFor(state, p);
      const transfers: any[] = [];
      for (const from of ts.filter(t => !opts.blockedFrom?.some((b: any) => same(t, b)))) {
        const source = StateUtils.getTarget(state, owner, from);
        source.cards.forEach((card, index) => {
          const blocked = opts.blockedMap?.find((m: any) => same(m.source, from))?.blocked ?? [];
          if (card.superType !== SuperType.ENERGY || blocked.includes(index) || !matchesPromptFilter(card, p.filter)) return;
          for (const to of ts.filter(t => !same(t, from) && !opts.blockedTo?.some((b: any) => same(t, b)))) transfers.push({from, to, index});
        });
      }
      subsets(transfers, opts.min, opts.max ?? transfers.length, xs => {
        const identities = xs.map(x => JSON.stringify([x.from, x.index]));
        if (new Set(identities).size !== identities.length) return;
        add(xs, `Move ${xs.length} energy: ${xs.map(x => `${targetLabel(state, p, x.from)} to ${targetLabel(state, p, x.to)}`).join('; ')}`);
      }); cancel(); break;
    }
    case 'Remove damage': case 'Move damage': {
      const owner = state.players.find(pl => pl.id === p.getPerspectivePlayerId())!;
      const ts = targetsFor(state, p);
      const froms = ts.filter(t => !opts.blockedFrom?.some((b: any) => same(t, b)));
      const tos = ts.filter(t => !opts.blockedTo?.some((b: any) => same(t, b)));
      for (const from of froms) for (const to of tos) {
        if (same(from, to)) continue;
        const available = Math.floor(StateUtils.getTarget(state, owner, from).damage / (opts.damageMultiple ?? 10));
        for (let n = Math.min(available, opts.max ?? available); n >= Math.max(1, opts.min); n--) {
          add(Array.from({length: n}, () => ({from, to})), `Move ${n} damage counters from ${targetLabel(state, p, from)} to ${targetLabel(state, p, to)}`);
        }
      }
      if (opts.min === 0) add([], 'Move no damage counters');
      if (p.type === 'Move damage' && !(opts.singleSourceTarget && opts.singleDestinationTarget)) warnings.push('Damage movement enumeration is selective: one source and destination per candidate.');
      cancel(); break;
    }
    case 'Put damage': {
      const ts = targetsFor(state, p).filter(t => !opts.blocked?.some((b: any) => same(t, b)));
      const unit = opts.damageMultiple ?? 10;
      const allocate = (index: number, left: number, assigned: any[]) => {
        if (choices.length >= CHOICE_LIMIT) { overflow = true; return; }
        if (index === ts.length) {
          if (left === 0 || opts.allowPlacePartialDamage) add(assigned, `Place counters: ${assigned.map(a => `${a.damage} on ${targetLabel(state, p, a.target)}`).join('; ') || 'none'}`);
          return;
        }
        const cap = p.maxAllowedDamage?.find((a: any) => same(a.target, ts[index]))?.damage ?? left;
        for (let n = Math.min(left, cap); n >= 0; n -= unit) allocate(index + 1, left - n, n ? [...assigned, {target: ts[index], damage: n}] : assigned);
      };
      allocate(0, p.damage, []); cancel(); break;
    }
    default: throw new Error(`Unsupported prompt type '${p.type}'; game must stop for implementation, not fabricate a result.`);
  }
  if (overflow) warnings.push(`Action enumeration for ${p.type} is selective (maximum ${CHOICE_LIMIT} candidates).`);
  if (!choices.length) throw new Error(`No validated choice for '${p.type}' (${p.message ?? ''}).`);
  return {choices, warnings};
}
