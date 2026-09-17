import { createHash } from 'node:crypto';
import { Store } from '../../../vendor/twinleaf/ptcg-server/src/game/store/store';
import { State, GamePhase, GameWinner } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Card } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { CardType, CardTag, Stage, SuperType, TrainerType, SpecialCondition } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';
import { deepClone } from '../../../vendor/twinleaf/ptcg-server/src/utils/utils';
import { AddPlayerAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/add-player-action';
import { ResolvePromptAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/resolve-prompt-action';
import { Action } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/action';
import { PlayCardAction, PlayerType, SlotType, CardTarget } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { AttackAction, UseAbilityAction, RetreatAction, PassTurnAction, UseStadiumAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/game-actions';
import { CoinFlipPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/coin-flip-prompt';
import { ShuffleDeckPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/shuffle-prompt';
import { ShuffleHandPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/shuffle-hand-prompt';
import { ShufflePrizesPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/shuffle-prizes-prompt';
import { SeededRandom } from './random';
import { sampleBeliefState, projectPublicPosition, samplePublicPosition } from './belief-state';
import type { PublicPosition } from './belief-state';
import { promptChoices } from './choices';
import { registerCards, getDeck, getDecks, deckNames } from './catalog';
import type { CardView, LegalAction, Observation, PokemonView, Replay, ReplayFrame } from './types';

declare const __ENGINE_BUILD__: string;
export const ENGINE_VERSION = 'twinleaf-adapter-0.1.0+' + (typeof __ENGINE_BUILD__ === 'string' ? __ENGINE_BUILD__ : 'development');
interface Candidate { view: LegalAction; action?: Action; raw?: any }
const ownTarget = (slot: SlotType, index = 0): CardTarget => ({player: PlayerType.BOTTOM_PLAYER, slot, index});
const warning = 'Experimental decks and upstream rules are not yet certified for the frozen Standard format.';

export function cardView(card: Card): CardView {
  const c: any = card;
  return {
    id: `${c.set}-${c.setNumber}`, name: c.name,
    kind: c.superType === SuperType.POKEMON ? 'pokemon' : c.superType === SuperType.ENERGY ? 'energy' : 'trainer',
    ...(c.superType === SuperType.POKEMON ? {
      types: (Array.isArray(c.cardType) ? c.cardType : [c.cardType]).map((t: number) => CardType[t]),
      hp: c.hp, stage: Stage[c.stage],
      prizeValue: c.hasTag(CardTag.POKEMON_SV_MEGA) ? 3 : c.hasTag(CardTag.POKEMON_ex) ? 2 : 1,
      attacks: c.attacks.map((a: any) => ({name: a.name, damage: a.damage, cost: a.cost.map((t: number) => CardType[t])})),
    } : {}),
  };
}
function pokemonView(list: PokemonCardList, owner: boolean): PokemonView | null {
  const card = list.getPokemonCard();
  if (!card || (list.isSecret && !owner)) return null;
  return {card: cardView(card), damage: list.damage,
    energy: list.cards.filter(c => c.superType === SuperType.ENERGY).map(c => c.name),
    tools: list.tools.map(c => c.name), conditions: list.specialConditions.map(c => SpecialCondition[c]),
  };
}

export class Environment {
  store!: Store;
  random!: SeededRandom;
  seed = 0;
  decks: [string, string] = ['', ''];
  decisionIndex = 0;
  private candidates: Candidate[] | null = null;
  private candidateWarnings: string[] = [];
  private replayWarnings = new Set<string>();
  private recorded: ReplayFrame[] = [];
  private actionIds: string[] = [];
  private publicHistory: string[] = [];
  private chance: Replay['chance'] = [];
  private warnings = new Set<string>([warning]);
  private failure: string | null = null;
  private hypothetical = false;

  constructor() { registerCards(); }
  reset(seed: number, decks: [string, string]) {
    if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('Seed must be a uint32 integer.');
    if (!Array.isArray(decks) || decks.length !== 2) throw new Error('Exactly two deck IDs are required.');
    decks.forEach(id => getDeck(id));
    this.seed = seed; this.decks = [...decks]; this.random = new SeededRandom(seed);
    this.decisionIndex = 0; this.candidates = null; this.candidateWarnings = []; this.replayWarnings = new Set(); this.recorded = []; this.actionIds = [];
    this.publicHistory = []; this.chance = []; this.warnings = new Set([warning]); this.failure = null;
    this.store = new Store({onStateChange: () => {}});
    this.random.scoped(() => {
      decks.forEach((id, index) => this.store.dispatch(new AddPlayerAction(index + 1, `Player ${index + 1}`, deckNames(id))));
    });
    if (this.store.state.players.length !== 2) throw new Error('Engine rejected a deck: structural validation failed.');
    this.resolveChance();
    return this.result();
  }
  private pending() { return this.store.state.prompts.find(p => p.result === undefined); }
  get actor(): number { return (this.pending()?.playerId ?? this.store.state.players[this.store.state.activePlayer]?.id ?? 1) - 1; }
  get status(): 'running' | 'finished' | 'error' {
    if (this.failure) return 'error';
    if (this.store.state.phase === GamePhase.FINISHED && this.store.state.winner === GameWinner.NONE) return 'error';
    return this.store.state.phase === GamePhase.FINISHED ? 'finished' : 'running';
  }
  private resolveChance() {
    for (let count = 0; count < 10000; count++) {
      const pending = this.store.state.prompts.filter(p => p.result === undefined);
      const prompt = pending.find(p => p instanceof CoinFlipPrompt || p instanceof ShuffleDeckPrompt || p instanceof ShufflePrizesPrompt || p instanceof ShuffleHandPrompt || p.type === 'WaitPrompt');
      if (!prompt) {
        if (!pending.length && !this.store.hasPrompts()) this.store.state.prompts = [];
        this.store.state.logs = []; // upstream wall-clock logs are not an observation or replay source
        return;
      }
      const owner = this.store.state.players.find(p => p.id === prompt.getPerspectivePlayerId())!;
      let raw: boolean | number[];
      if (prompt.type === 'WaitPrompt') raw = true;
      else if (prompt instanceof CoinFlipPrompt) raw = this.random.int(2) === 1;
      else if (prompt instanceof ShuffleHandPrompt) throw new Error('Upstream ShuffleHandPrompt validator uses prize count; unsupported until audited.');
      else if (prompt instanceof ShufflePrizesPrompt) raw = this.random.shuffle(owner.prizes.reduce((n, p) => n + p.cards.length, 0));
      else raw = this.random.shuffle(owner.deck.cards.length);
      const decoded = prompt.decode(raw, this.store.state);
      if (!prompt.validate(decoded, this.store.state)) throw new Error(`Invalid generated chance result for ${prompt.type}.`);
      this.chance.push({decisionIndex: this.decisionIndex, type: prompt.type, result: raw});
      this.random.scoped(() => this.store.dispatch(new ResolvePromptAction(prompt.id, decoded)));
    }
    throw new Error('Chance resolution limit reached; no artificial terminal outcome created.');
  }
  private normalCandidates(): Candidate[] {
    const state = this.store.state;
    if (state.phase !== GamePhase.PLAYER_TURN) throw new Error(`Unsupported non-prompt phase ${GamePhase[state.phase]}.`);
    const player = state.players[state.activePlayer];
    const raw: Candidate[] = [];
    const add = (action: Action, type: string, label: string, card?: Card, target?: string) => raw.push({action, view: {id: '', type, label, ...(card ? {cardId: cardView(card).id} : {}), ...(target ? {target} : {})}});
    const slots = [{target: ownTarget(SlotType.ACTIVE), list: player.active}, ...player.bench.map((list, i) => ({target: ownTarget(SlotType.BENCH, i), list}))];
    const seenCards = new Set<string>();
    player.hand.cards.forEach((card: any, i) => {
      if (seenCards.has(card.fullName)) return; seenCards.add(card.fullName);
      if (card.superType === SuperType.TRAINER && card.trainerType !== TrainerType.TOOL) {
        add(new PlayCardAction(player.id, i, ownTarget(SlotType.ACTIVE)), 'play-trainer', `Play ${card.name}`, card); return;
      }
      for (const {target, list} of slots) {
        if (card.superType === SuperType.ENERGY && !list.getPokemonCard()) continue;
        const at = target.slot === SlotType.ACTIVE ? 'active' : `bench ${target.index + 1}`;
        const type = card.superType === SuperType.ENERGY ? 'attach-energy' : card.superType === SuperType.POKEMON ? (list.getPokemonCard() ? 'evolve' : 'bench') : 'attach-tool';
        add(new PlayCardAction(player.id, i, target), type, `${type === 'attach-energy' ? 'Attach' : 'Play'} ${card.name} to ${at}`, card, at);
      }
    });
    for (const {target, list} of slots) {
      const card = list.getPokemonCard();
      card?.powers.forEach(power => { if (power.useWhenInPlay === true) add(new UseAbilityAction(player.id, power.name, target), 'ability', `Use ${card.name}: ${power.name}`, card); });
    }
    player.active.getPokemonCard()?.attacks.forEach(attack => add(new AttackAction(player.id, attack.name), 'attack', `Attack: ${attack.name}`, player.active.getPokemonCard()));
    player.bench.forEach((b, i) => { if (b.getPokemonCard()) add(new RetreatAction(player.id, i), 'retreat', `Retreat to bench ${i + 1}: ${b.getPokemonCard()!.name}`); });
    if (state.players.some(p => p.stadium.cards.length)) add(new UseStadiumAction(player.id), 'stadium', 'Use stadium');
    add(new PassTurnAction(player.id), 'pass', 'End turn');
    // At this point no callbacks are pending. Probe deep-cloned states in isolated Stores.
    // Never use the upstream Simulator clone for continuation through a live prompt.
    return raw.filter(candidate => {
      const probe = new Store({onStateChange: () => {}});
      probe.state = deepClone(state);
      try {
        const rng = new SeededRandom(this.random.state);
        rng.scoped(() => probe.dispatch(candidate.action!));
        for (let n = 0; n < 100; n++) {
          const wait = probe.state.prompts.find(p => p.result === undefined && p.type === 'WaitPrompt');
          if (!wait) break;
          rng.scoped(() => probe.dispatch(new ResolvePromptAction(wait.id, true)));
          if (n === 99) throw new Error('Probe automatic resolution limit');
        }
        return true;
      }
      catch { return false; }
    });
  }
  private getCandidates(): Candidate[] {
    if (this.candidates) return this.candidates;
    if (this.status !== 'running') return [];
    const pending = this.pending();
    this.candidateWarnings = [];
    let candidates: Candidate[];
    if (pending) {
      const resolved = promptChoices(pending, this.store.state);
      this.candidateWarnings = resolved.warnings;
      resolved.warnings.forEach(w => this.replayWarnings.add(w));
      candidates = resolved.choices.map(({raw, label}) => ({raw, view: {id: '', type: 'prompt', label}}));
    } else candidates = this.normalCandidates();
    if (!candidates.length) throw new Error('No validated actions; cannot advance the engine.');
    candidates.forEach((c, i) => { c.view.id = `${this.decisionIndex}:${i}`; });
    this.candidates = candidates;
    return candidates;
  }
  observe(playerId = this.actor): Observation {
    if (playerId !== 0 && playerId !== 1) throw new Error('Player ID must be 0 or 1.');
    const actor = this.actor;
    const actions = this.getCandidates();
    const state = this.store.state;
    const prompt: any = this.pending();
    let searchPosition: PublicPosition | undefined; let searchUnavailableReason: string | undefined;
    if (playerId === actor && this.status === 'running') {
      try { searchPosition = projectPublicPosition(state, playerId, this.decks[playerId]); }
      catch (error) { searchUnavailableReason = error instanceof Error ? error.message : String(error); }
    }
    return {
      ...(searchPosition ? {searchPosition} : {}), ...(searchUnavailableReason ? {searchUnavailableReason} : {}),
      schemaVersion: 1, playerId, decisionPlayer: actor, turn: state.turn, phase: GamePhase[state.phase], status: this.status,
      players: state.players.map((p, index) => ({
        id: index, name: p.name, active: pokemonView(p.active, index === playerId),
        bench: p.bench.map(b => pokemonView(b, index === playerId)).filter((b): b is PokemonView => !!b),
        hand: index === playerId ? p.hand.cards.map(cardView) : [], handCount: p.hand.cards.length,
        deckCount: p.deck.cards.length, prizesRemaining: p.prizes.filter(pr => pr.cards.length).length,
        discard: p.discard.cards.map(cardView),
      })),
      ownDeck: getDeck(this.decks[playerId]).cards.map((c: any) => ({cardId: c.cardId, name: c.name, count: c.count})),
      legalActions: playerId === actor ? actions.map(c => ({...c.view})) : [],
      history: [...this.publicHistory],
      ...(prompt && actor === playerId ? {prompt: {type: prompt.type, message: String(prompt.message ?? ''), ...(['Show cards', 'Confirm cards'].includes(prompt.type) ? {cards: prompt.cards.map(cardView)} : {}), ...(prompt.type === 'Show mulligan' ? {hands: prompt.hands.map((h: Card[]) => h.map(cardView))} : {})}} : {}),
      warnings: [...this.warnings, ...(playerId === actor ? this.candidateWarnings : [])],
    };
  }
  private frame(action: LegalAction | null): ReplayFrame {
    return {decisionIndex: this.decisionIndex, actor: this.actor, action, observations: [this.observe(0), this.observe(1)]};
  }
  private result() { return {observation: this.observe(), status: this.status, decisionIndex: this.decisionIndex}; }
  step(actionId: string) {
    if (this.status !== 'running') throw new Error('Game is not running.');
    const chosen = this.getCandidates().find(c => c.view.id === actionId);
    if (!chosen) throw new Error('Unknown or stale action ID.');
    const before = this.frame({...chosen.view});
    const oldIds = [...this.actionIds];
    const seed = this.seed; const decks: [string, string] = [...this.decks];
    try {
      const prompt = this.pending();
      if (prompt) {
        const decoded = prompt.decode(chosen.raw, this.store.state);
        if (!prompt.validate(decoded, this.store.state)) throw new Error('Prompt candidate failed validation.');
        this.random.scoped(() => this.store.dispatch(new ResolvePromptAction(prompt.id, decoded)));
      } else this.random.scoped(() => this.store.dispatch(chosen.action!));
      this.publicHistory.push(`P${before.actor + 1}: ${prompt ? `resolved ${prompt.type}` : chosen.view.label}`);
      this.recorded.push(before); this.actionIds.push(actionId); this.decisionIndex++; this.candidates = null;
      this.resolveChance();
      return this.result();
    } catch (error) {
      // Dispatch may mutate its state before throwing. Reconstruct all callback closures as well.
      if (this.hypothetical) { this.fail('Hypothetical continuation failed'); throw error; }
      this.reset(seed, decks);
      for (const id of oldIds) this.step(id);
      throw error;
    }
  }
  static fromPublicPosition(position: PublicPosition, seed: number, opponentDeckId: string, decisionIndex = 0): Environment {
    const copy = new Environment(); copy.seed = seed; copy.random = new SeededRandom(seed); copy.hypothetical = true;
    copy.decks = position.observer === 0 ? [position.ownDeckId, opponentDeckId] : [opponentDeckId, position.ownDeckId];
    copy.store = new Store({onStateChange: () => {}});
    copy.store.state = samplePublicPosition(position, opponentDeckId, copy.random);
    copy.decisionIndex = decisionIndex; return copy;
  }
  /** Experimental stable-position determinization, built without original hidden-state identities. */
  forkForBelief(seed: number, opponentDeckId: string, observerPlayerId = this.actor): Environment {
    const rng = new SeededRandom(seed);
    const copy = new Environment();
    copy.seed = seed; copy.hypothetical = true;
    copy.decks = observerPlayerId === 0 ? [this.decks[0], opponentDeckId] : [opponentDeckId, this.decks[1]];
    copy.random = rng;
    copy.store = new Store({onStateChange: () => {}});
    copy.store.state = sampleBeliefState(this.store.state, observerPlayerId, this.decks[observerPlayerId], opponentDeckId, rng);
    copy.decisionIndex = this.decisionIndex;
    copy.publicHistory = [...this.publicHistory];
    copy.warnings.add('Belief sampling uses public board and own hand; previous hidden-card revelations and deck-order knowledge are not modeled.');
    return copy;
  }
  branch(): Environment {
    const copy = new Environment(); copy.reset(this.seed, this.decks);
    for (const id of this.actionIds) copy.step(id);
    return copy;
  }
  replay(status?: Replay['status']): Replay {
    const finished = this.status === 'finished';
    const winner = this.store.state.winner;
    const actualStatus = status ?? (finished ? 'finished' : this.status === 'error' ? 'error' : 'truncated');
    if (actualStatus === 'finished' && !finished) throw new Error('Cannot mark a nonterminal game finished.');
    const id = createHash('sha256').update(JSON.stringify({seed: this.seed, decks: this.decks, actions: this.actionIds, engine: ENGINE_VERSION})).digest('hex').slice(0, 20);
    let finalFrame: ReplayFrame;
    try { finalFrame = this.frame(null); } catch { this.failure ??= 'Unable to enumerate current decision'; finalFrame = {decisionIndex: this.decisionIndex, actor: this.actor, action: null, observations: this.recorded.at(-1)?.observations ?? [] as any}; }
    return {
      schemaVersion: 1, id, seed: this.seed, decks: [...this.decks], engineVersion: ENGINE_VERSION,
      status: actualStatus, outcome: finished ? {winner: winner === GameWinner.PLAYER_1 ? 0 : winner === GameWinner.PLAYER_2 ? 1 : null, reason: winner === GameWinner.DRAW ? 'rules-draw' : winner === GameWinner.NONE ? 'engine-ended-without-winner' : 'rules-terminal'} : null,
      frames: [...this.recorded, finalFrame], warnings: [...this.warnings, ...this.replayWarnings, ...(this.status === 'error' && !this.failure ? ['Engine ended without a valid winner; this is not a draw.'] : []), ...(this.failure ? [this.failure] : [])], visibility: 'private-research', chance: [...this.chance],
    };
  }
  fail(message: string) { this.failure = message; this.candidates = []; }
}
export { getDecks };
