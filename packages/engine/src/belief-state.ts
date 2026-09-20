import { State, GamePhase } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { Card } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { Marker } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-marker';
import { CARD_FACTORIES, getDeck, printedId } from './catalog';
import { SeededRandom } from './random';
import type { DeckManifest } from './catalog';
export type DeckHypothesis = string | DeckManifest;

// Deliberately explicit: additions to upstream state do not silently become policy inputs.
const PLAYER_FIELDS = ['supporterTurn', 'retreatedTurn', 'energyPlayedTurn', 'stadiumPlayedTurn', 'stadiumUsedTurn',
  'usedTableTurner', 'pokemonKnockedOutDuringOpponentsLastTurn', 'pokemonKnockedOutByAttackDuringOpponentsLastTurn',
  'pokemonKnockedOutLastTurnEntries', 'prizesTaken', 'prizesTakenThisTurn', 'prizesTakenLastTurn', 'canEvolve',
  'ancientPokemonAttackedLastTurn'];
const BOARD_FIELDS = ['damage', 'hp', 'hpBonus', 'specialConditions', 'poisonDamage', 'burnDamage', 'confusionDamage',
  'pokemonPlayedTurn', 'abilityLockActivationOrder', 'sleepFlips', 'boardEffect', 'attacksThisTurn',
  'cannotUseAttacksNextTurn', 'cannotUseAttacksNextTurnPending', 'noAbilities', 'noAbilitiesAttackerId', 'noAbilitiesClearArmed',
  'cannotAttackNextTurn', 'cannotAttackNextTurnPending', 'healedThisTurn', 'maxTools'];
const PLAYER_IGNORE = new Set(['id', 'name', 'hand', 'deck', 'prizes', 'active', 'bench', 'discard', 'lostzone', 'stadium', 'supporter', 'marker',
  'playableCardIds', 'playableHandAbilityCardIds', 'gameStats', 'movedToActiveThisTurn', 'movedFromActiveToBenchThisTurn']);
const BOARD_IGNORE = new Set(['cards', 'tools', 'energies', 'marker', 'isSecret', 'isPublic', 'isActivatingCard', 'triggerEvolutionAnimation',
  'showBasicAnimation', 'triggerAttackAnimation', '__uniqueId']);
const serial = (x: unknown) => JSON.stringify(x);
function copyAllowed(source: any, dest: any, allowed: string[], ignored: Set<string>, label: string) {
  for (const key of allowed) if (source[key] !== undefined) dest[key] = JSON.parse(JSON.stringify(source[key]));
  for (const key of Object.keys(source)) {
    if (allowed.includes(key) || ignored.has(key)) continue;
    if (serial(source[key]) !== serial(dest[key])) throw new Error(`Belief search does not support modified ${label}.${key}.`);
  }
}

/**
 * Build from a public-state whitelist + the observer's own hand; never clone true hidden zones.
 * The caller supplies the opponent deck hypothesis. No field consults the original opponent decklist.
 * Stable positions only: callback closures cannot be transplanted into an independently sampled game.
 */
export function sampleBeliefState(source: State, observer: number, ownDeckId: DeckHypothesis, opponentDeckId: DeckHypothesis, rng: SeededRandom): State {
  if (source.phase !== GamePhase.PLAYER_TURN || source.prompts.some(p => p.result === undefined)) throw new Error('Belief search requires a stable player-turn decision.');
  if (source.activePlayer !== observer) throw new Error('Belief search is available only for the current decision player.');
  const state = new State();
  state.phase = source.phase; state.turn = source.turn; state.activePlayer = source.activePlayer;
  state.abilityLockOrderCounter = source.abilityLockOrderCounter;
  const cardMap = new Map<Card, Card>();
  let nextId = 0;
  const freshCard = (id: string) => {
    const make = CARD_FACTORIES[id]; if (!make) throw new Error(`Belief search does not support ${id}.`);
    const card = make(); card.id = nextId++; state.cardNames[card.id] = card.fullName; return card;
  };
  const publicCard = (old: Card): Card => {
    if (!cardMap.has(old)) {
      const card = freshCard(printedId(old));
      if (serial(old.tags) !== serial(card.tags)) throw new Error('Belief search does not support modified card tags.');
      cardMap.set(old, card);
    }
    return cardMap.get(old)!;
  };
  const markers: {old: Marker; fresh: Marker}[] = [];
  const list = (old: CardList): CardList => {
    const fresh = new CardList(); fresh.isPublic = true; fresh.cards = old.cards.map(publicCard); return fresh;
  };
  const board = (old: PokemonCardList): PokemonCardList => {
    if (old.isSecret) throw new Error('Belief search does not support face-down board cards.');
    const fresh = new PokemonCardList();
    copyAllowed(old, fresh, BOARD_FIELDS, BOARD_IGNORE, 'board');
    fresh.isPublic = true; fresh.cards = old.cards.map(publicCard); fresh.tools = old.tools.map(publicCard);
    fresh.energies.cards = old.energies.cards.map(publicCard);
    markers.push({old: old.marker, fresh: fresh.marker});
    return fresh;
  };
  source.players.forEach((old, index) => {
    const fresh = new Player(); fresh.id = index + 1; fresh.name = `Player ${index + 1}`;
    copyAllowed(old, fresh, PLAYER_FIELDS, PLAYER_IGNORE, 'player');
    fresh.active = board(old.active); fresh.bench = old.bench.map(board);
    fresh.discard = list(old.discard); fresh.lostzone = list(old.lostzone); fresh.stadium = list(old.stadium); fresh.supporter = list(old.supporter);
    markers.push({old: old.marker, fresh: fresh.marker});
    // Only observer hand card identities are read here. Opponent hand supplies its public count only.
    if (index === observer) fresh.hand.cards = old.hand.cards.map(publicCard);
    const reference = index === observer ? ownDeckId : opponentDeckId;
    const deck = typeof reference === 'string' ? getDeck(reference) : reference;
    const available = new Map<string, number>(deck.cards.map(c => [c.cardId, c.count]));
    const known = new Set<Card>([...fresh.active.cards, ...fresh.active.tools, ...fresh.active.energies.cards,
      ...fresh.bench.flatMap(b => [...b.cards, ...b.tools, ...b.energies.cards]), ...fresh.discard.cards,
      ...fresh.lostzone.cards, ...fresh.stadium.cards, ...fresh.supporter.cards, ...fresh.hand.cards]);
    for (const card of known) {
      const id = printedId(card); const n = available.get(id) ?? 0;
      if (n < 1) throw new Error(`Deck hypothesis ${deck.id} conflicts with visible card ${id}.`);
      available.set(id, n - 1);
    }
    const remaining = [...available].flatMap(([id, count]) => Array.from({length: count}, () => id));
    const shuffled = rng.shuffle(remaining.length).map(i => remaining[i]);
    const handCount = index === observer ? 0 : old.hand.cards.length;
    const prizeCount = old.prizes.filter(p => p.cards.length > 0).length;
    if (shuffled.length !== handCount + prizeCount + old.deck.cards.length) throw new Error('Belief pool does not match public zone counts.');
    if (index !== observer) fresh.hand.cards = shuffled.splice(0, handCount).map(freshCard);
    fresh.prizes = Array.from({length: prizeCount}, () => {const pr = new CardList(); pr.isSecret = true; pr.cards = [freshCard(shuffled.shift()!)]; return pr;});
    fresh.deck.isSecret = true; fresh.deck.cards = shuffled.map(freshCard);
    state.players.push(fresh);
  });
  // Marker provenance must point to a public or observer-known card, never a hidden original object.
  for (const {old, fresh} of markers) fresh.markers = old.markers.map(m => {
    if (m.source && !cardMap.has(m.source)) throw new Error('Belief search does not support hidden marker provenance.');
    return {...m, source: m.source ? cardMap.get(m.source) : undefined};
  });
  for (let i = 0; i < 2; i++) {
    for (const key of ['movedToActiveThisTurn', 'movedFromActiveToBenchThisTurn'] as const) {
      state.players[i][key] = source.players[i][key].map(id => {
        const old = [...cardMap.keys()].find(c => c.id === id);
        if (!old) throw new Error('Belief search does not support unknown movement provenance.');
        return cardMap.get(old)!.id;
      });
    }
  }
  // The five-deck baseline has no attack-copy card. Reject active copy-dependent history explicitly.
  state.lastAttack = null; state.playerLastAttack = {};
  return state;
}

export interface PublicPosition {
  version: 1; observer: number; ownDeckId: string; ownPrizeCards?: string[]; ownDeckTop?: string[]; turn: number; activePlayer: number; abilityLockOrderCounter: number;
  cards: {key: number; cardId: string}[];
  players: any[];
}
/** Transportable whitelist. Contains no true hidden hand, prize identities, or deck order. */
export function projectPublicPosition(source: State, observer: number, ownDeckId: string): PublicPosition {
  if (source.phase !== GamePhase.PLAYER_TURN || source.prompts.some(p => p.result === undefined)) throw new Error('Search is unavailable while an effect is awaiting a choice.');
  if (observer !== source.activePlayer) throw new Error('Search is available for the decision player only.');
  const keys = new Map<Card, number>();
  const cards: PublicPosition['cards'] = [];
  const cardKey = (card: Card) => {
    if (!keys.has(card)) {
      const cardId = printedId(card); if (!CARD_FACTORIES[cardId]) throw new Error(`Unsupported visible card ${cardId}.`);
      const key = keys.size; keys.set(card, key); cards.push({key, cardId});
    }
    return keys.get(card)!;
  };
  const fields = (value: any, allowed: string[]) => Object.fromEntries(allowed.filter(k => value[k] !== undefined).map(k => [k, JSON.parse(JSON.stringify(value[k]))]));
  const board = (b: PokemonCardList) => {
    if (b.isSecret) throw new Error('Face-down board cards are not supported by search.');
    copyAllowed(b, new PokemonCardList(), BOARD_FIELDS, BOARD_IGNORE, 'board');
    return {fields: fields(b, BOARD_FIELDS), cards: b.cards.map(cardKey), tools: b.tools.map(cardKey), energies: b.energies.cards.map(cardKey)};
  };
  const players = source.players.map((p, i) => {
    copyAllowed(p, new Player(), PLAYER_FIELDS, PLAYER_IGNORE, 'player');
    return {fields: fields(p, PLAYER_FIELDS), active: board(p.active), bench: p.bench.map(board),
      discard: p.discard.cards.map(cardKey), lostzone: p.lostzone.cards.map(cardKey), stadium: p.stadium.cards.map(cardKey), supporter: p.supporter.cards.map(cardKey),
      hand: i === observer ? p.hand.cards.map(cardKey) : [], handCount: p.hand.cards.length, deckCount: p.deck.cards.length,
      prizes: p.prizes.filter(pr => pr.cards.length).length};
  });
  const marker = (m: Marker) => m.markers.map(item => {
    if (item.source && !keys.has(item.source)) throw new Error('Search does not support hidden marker provenance.');
    return {name: item.name, sourceKey: item.source ? keys.get(item.source) : undefined, sourceType: item.sourceType, targetScope: item.targetScope};
  });
  players.forEach((p: any, i) => {
    const old = source.players[i]; p.markers = marker(old.marker); p.active.markers = marker(old.active.marker);
    p.bench.forEach((b: any, j: number) => { b.markers = marker(old.bench[j].marker); });
    p.movedToActiveThisTurn = old.movedToActiveThisTurn.map(id => {const card = [...keys.keys()].find(c => c.id === id); if (!card) throw new Error('Unsupported movement provenance.'); return keys.get(card);});
    p.movedFromActiveToBenchThisTurn = old.movedFromActiveToBenchThisTurn.map(id => {const card = [...keys.keys()].find(c => c.id === id); if (!card) throw new Error('Unsupported movement provenance.'); return keys.get(card);});
  });
  return {version: 1, observer, ownDeckId, turn: source.turn, activePlayer: source.activePlayer, abilityLockOrderCounter: source.abilityLockOrderCounter, cards, players};
}

/** Hydrates only known cards and zone counts, then samples every unknown card from the hypothesis. */
export function samplePublicPosition(position: PublicPosition, opponentDeckId: DeckHypothesis, rng: SeededRandom): State {
  if (position.version !== 1 || ![0, 1].includes(position.observer) || position.players.length !== 2) throw new Error('Invalid search position.');
  const source = new State(); source.phase = GamePhase.PLAYER_TURN; source.turn = position.turn;
  source.activePlayer = position.activePlayer; source.abilityLockOrderCounter = position.abilityLockOrderCounter;
  const cards = new Map(position.cards.map(c => {
    const make = CARD_FACTORIES[c.cardId]; if (!make) throw new Error(`Unsupported card ${c.cardId}.`);
    const card = make(); card.id = c.key; return [c.key, card] as const;
  }));
  const take = (indexes: number[]) => indexes.map(i => {const card = cards.get(i); if (!card) throw new Error('Unknown public card reference.'); return card;});
  const marker = (items: any[]) => items.map(m => ({name: m.name, source: m.sourceKey === undefined ? undefined : cards.get(m.sourceKey), sourceType: m.sourceType, targetScope: m.targetScope}));
  const board = (b: any) => {
    const p = new PokemonCardList();
    for (const k of BOARD_FIELDS) if (b.fields[k] !== undefined) (p as any)[k] = b.fields[k];
    p.isPublic = true; p.cards = take(b.cards); p.tools = take(b.tools); p.energies.cards = take(b.energies);
    p.marker.markers = marker(b.markers); return p;
  };
  source.players = position.players.map((p: any, i: number) => {
    const fresh = new Player(); fresh.id = i + 1;
    for (const k of PLAYER_FIELDS) if (p.fields[k] !== undefined) (fresh as any)[k] = p.fields[k];
    fresh.active = board(p.active); fresh.bench = p.bench.map(board);
    for (const k of ['discard', 'lostzone', 'stadium', 'supporter'] as const) {fresh[k].cards = take(p[k]); fresh[k].isPublic = true;}
    if (![p.handCount, p.deckCount, p.prizes].every(n => Number.isInteger(n) && n >= 0 && n <= 60)) throw new Error('Invalid public zone count.');
    fresh.hand.cards = i === position.observer ? take(p.hand) : Array(p.handCount).fill(null);
    fresh.deck.cards = Array(p.deckCount).fill(null);
    fresh.prizes = Array.from({length: p.prizes}, () => {const c = new CardList(); c.cards = [null as any]; return c;});
    fresh.marker.markers = marker(p.markers); fresh.movedToActiveThisTurn = p.movedToActiveThisTurn; fresh.movedFromActiveToBenchThisTurn = p.movedFromActiveToBenchThisTurn;
    return fresh;
  });
  const sampled = sampleBeliefState(source, position.observer, position.ownDeckId, opponentDeckId, rng);
  const own = sampled.players[position.observer];
  // The supplied identities were inferred from a legal deck search, never read from
  // the true Prize zone. Randomize their slot allocation rather than inventing it.
  if (position.ownPrizeCards) {
    if (position.ownPrizeCards.length !== own.prizes.length) throw new Error('Known Prize counts are stale.');
    const pool = [...own.deck.cards, ...own.prizes.flatMap(p => p.cards)];
    const takeKnown = (id: string) => {const i=pool.findIndex(c=>printedId(c)===id);if(i<0)throw new Error('Known Prize identities conflict with own deck.');return pool.splice(i,1)[0];};
    const prizes=position.ownPrizeCards.map(takeKnown);
    rng.shuffle(prizes.length).forEach((index, slot)=>{own.prizes[slot].cards=[prizes[index]];});
    own.deck.cards = rng.shuffle(pool.length).map(i=>pool[i]);
  }
  if (position.ownDeckTop?.length) {
    const pool=[...own.deck.cards];
    const top=position.ownDeckTop.map(id=>{const i=pool.findIndex(c=>printedId(c)===id);if(i<0)throw new Error('Known deck order conflicts with own deck.');return pool.splice(i,1)[0];});
    own.deck.cards=[...top,...pool];
  }
  return sampled;
}
