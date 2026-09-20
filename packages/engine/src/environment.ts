import { readFileSync } from 'node:fs';
import { join } from 'node:path';
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
import type { PublicPosition, DeckHypothesis } from './belief-state';
import { promptChoices, stagedChoices, STAGED_PROMPTS } from './choices';
import type { PromptChoice } from './choices';
import { choiceBinding, targetRef } from './action-bindings';
import { registerCards, getDeck, getDecks, deckNames, printedId } from './catalog';
import type { CardView, LegalAction, Observation, PokemonView, Replay, ReplayFrame } from './types';

declare const __ENGINE_BUILD__: string;
export const ENGINE_VERSION = 'twinleaf-adapter-0.1.0+' + (typeof __ENGINE_BUILD__ === 'string' ? __ENGINE_BUILD__ : 'development');
interface Candidate { view: LegalAction; action?: Action; raw?: any; stage?: PromptChoice['stage'] }
const ownTarget = (slot: SlotType, index = 0): CardTarget => ({player: PlayerType.BOTTOM_PLAYER, slot, index});
const warning = 'Experimental decks and upstream rules are not yet certified for the frozen Standard format.';

// Optional, pinned and verified provider metadata. Missing art uses the card's
// readable text view; never manufacture provider paths for unverified printings.
let art: Record<string,{imageUrl:string}> = {};
try { art = JSON.parse(readFileSync(join(process.cwd(),'formats/card-art.json'),'utf8')).cards ?? {}; } catch { /* optional local catalog */ }
function cardImageUrl(set:string,number:string){return art[`${set}-${number}`]?.imageUrl;}
export function cardView(card: Card): CardView {
  const c: any = card;
  return {
    id: `${c.set}-${c.setNumber}`, name: c.name, text: c.text,
    imageUrl: cardImageUrl(c.set,c.setNumber),
    kind: c.superType === SuperType.POKEMON ? 'pokemon' : c.superType === SuperType.ENERGY ? 'energy' : 'trainer',
    ...(c.superType === SuperType.POKEMON ? {
      types: (Array.isArray(c.cardType) ? c.cardType : [c.cardType]).map((t: number) => CardType[t]),
      hp: c.hp, stage: Stage[c.stage],
      powers: c.powers.map((p:any)=>({name:p.name,text:p.text})),
      prizeValue: c.hasTag(CardTag.POKEMON_SV_MEGA) ? 3 : c.hasTag(CardTag.POKEMON_ex) ? 2 : 1,
      attacks: c.attacks.map((a: any) => ({name: a.name, damage: a.damage, text:a.text, cost: a.cost.map((t: number) => CardType[t])})),
    } : {}),
  };
}
function pokemonView(list: PokemonCardList, owner: boolean): PokemonView | null {
  const card = list.getPokemonCard();
  if (!card || (list.isSecret && !owner)) return null;
  const attachments=[...new Set([...list.cards.filter(c=>c.superType===SuperType.ENERGY),...list.energies.cards,...list.tools])];
  return {card: cardView(card), damage: list.damage, attachments:attachments.map(cardView),
    energy: attachments.filter(c => c.superType === SuperType.ENERGY).map(c => c.name),
    tools: list.tools.map(c => c.name), conditions: list.specialConditions.map(c => SpecialCondition[c]),
  };
}

export class Environment {
  store!: Store;
  random!: SeededRandom;
  seed = 0;
  firstPlayer:0|1|undefined;
  private selection:any[]=[];
  private selectionPrompt:number|undefined;
  private knowledge: [any[],any[]]=[[],[]];
  private seenKnowledge=new WeakSet<object>();
  private knowledgeRestricted=[false,false];
  private ownPrizeKnowledge: (string[]|undefined)[]=[undefined,undefined];
  private knownTop: Card[][]=[[],[]];
  private knownBottom: Card[][]=[[],[]];
  private knownNonPrizeCounts:Map<string,number>[]=[new Map(),new Map()];
  private knownOpponentHand: Map<string,number>[]=[new Map(),new Map()];
  private opponentRevealedCounts: Map<string,number>[]=[new Map(),new Map()];
  private activePeek: {viewer:number;kind:'recon'|'pokegear'}|undefined;
  private effectStartHands:Set<Card>[]|undefined;
  private effectReveals:Set<Card>[]=[new Set(),new Set()];
  private activeEffectCardId:string|undefined;
  private hypotheticalOpponent: import('./catalog').DeckManifest|undefined;
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
  private reconstructFixture:(()=>Environment)|undefined;

  constructor() { registerCards(); }
  reset(seed: number, decks: [string, string], firstPlayer?:0|1) {
    if(firstPlayer!==undefined&&firstPlayer!==0&&firstPlayer!==1)throw new Error("firstPlayer must be 0 or 1.");
    this.firstPlayer=firstPlayer;this.selection=[];this.selectionPrompt=undefined;this.knowledge=[[],[]];this.seenKnowledge=new WeakSet();this.knowledgeRestricted=[false,false];this.ownPrizeKnowledge=[undefined,undefined];this.knownTop=[[],[]];
    this.knownBottom=[[],[]];this.knownNonPrizeCounts=[new Map(),new Map()];this.knownOpponentHand=[new Map(),new Map()];this.opponentRevealedCounts=[new Map(),new Map()];this.activePeek=undefined;
    this.effectStartHands=undefined;this.effectReveals=[new Set(),new Set()];
    this.activeEffectCardId=undefined;
    if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('Seed must be a uint32 integer.');
    if (!Array.isArray(decks) || decks.length !== 2) throw new Error('Exactly two deck IDs are required.');
    decks.forEach(id => getDeck(id));
    this.seed = seed; this.decks = [...decks]; this.random = new SeededRandom(seed,value=>this.chance.push({decisionIndex:this.decisionIndex,type:'upstream-random',result:value}));
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
      const prompt = pending.find(p => p instanceof CoinFlipPrompt || p instanceof ShuffleDeckPrompt || p instanceof ShufflePrizesPrompt || p instanceof ShuffleHandPrompt || p.type === 'WaitPrompt' || (this.firstPlayer!==undefined&&String((p as any).message)==='GO_FIRST'));
      if (!prompt) {
        if (!pending.length && !this.store.hasPrompts()) this.store.state.prompts = [];
        this.store.state.logs = []; // upstream wall-clock logs are not an observation or replay source
        return;
      }
      const owner = this.store.state.players.find(p => p.id === prompt.getPerspectivePlayerId())!;
      let raw: boolean | number[];
      if (prompt.type === 'WaitPrompt') raw = true;
      else if(this.firstPlayer!==undefined&&String((prompt as any).message)==='GO_FIRST') raw=prompt.playerId-1===this.firstPlayer;
      else if (prompt instanceof CoinFlipPrompt) raw = this.random.int(2) === 1;
      else if (prompt instanceof ShuffleHandPrompt) {
        raw = this.random.shuffle(owner.hand.cards.length);
        // Special Red Card creates a known but unordered bottom segment. Until
        // that segment is represented, neither observer may forget its location.
        if(owner.hand.cards.length)this.knowledgeRestricted[owner.id-1]=true;
        const viewer=1-(owner.id-1);
        if([...this.knownOpponentHand[viewer].values()].some(n=>n>0))this.knowledgeRestricted[viewer]=true;
        if(this.activeEffectCardId==='CRI-82')this.knownOpponentHand[viewer].clear();
      }
      else if (prompt instanceof ShufflePrizesPrompt) raw = this.random.shuffle(owner.prizes.reduce((n, p) => n + p.cards.length, 0));
      else {raw = this.random.shuffle(owner.deck.cards.length);this.knownTop[owner.id-1]=[];this.knownBottom[owner.id-1]=[];if(owner.hand.cards.length===0)this.knownOpponentHand[1-(owner.id-1)].clear();}
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
    const add = (action: Action, type: string, label: string, card?: Card, target?: string) => {
      const binding: Partial<LegalAction>={};
      if(action instanceof PlayCardAction){binding.sourceRef={playerId:player.id-1,zone:'hand',index:action.handIndex};if(type!=='play-trainer')binding.targetRef=targetRef(state,player.id,action.target);}
      if(action instanceof UseAbilityAction)binding.sourceRef=targetRef(state,player.id,action.target);
      if(action instanceof AttackAction){binding.sourceRef={playerId:player.id-1,zone:'active',index:0};binding.targetRef={playerId:1-(player.id-1),zone:'active',index:0};}
      if(action instanceof RetreatAction){binding.sourceRef={playerId:player.id-1,zone:'active',index:0};binding.targetRef={playerId:player.id-1,zone:'bench',index:action.benchIndex};}
      raw.push({action,view:{id:'',type,label,...(card?{cardId:cardView(card).id}:{}),...(target?{target}:{}),...binding}});
    };
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
        if(candidate.action instanceof PlayCardAction){
          const actor=probe.state.players[probe.state.activePlayer];
          const card:any=actor.hand.cards[candidate.action.handIndex];
          if(typeof card?.canPlay==='function'&&!card.canPlay(probe,probe.state,actor))return false;
        }
        rng.scoped(() => probe.dispatch(candidate.action!));
        for (let n = 0; n < 100; n++) {
          const wait = probe.state.prompts.find(p => p.result === undefined && p.type === 'WaitPrompt' || (this.firstPlayer!==undefined&&String((p as any).message)==='GO_FIRST'));
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
      if(this.selectionPrompt!==pending.id){this.selectionPrompt=pending.id;this.selection=[];}
      const staged=STAGED_PROMPTS.has(pending.type);
      const resolved=staged?{choices:stagedChoices(pending,this.store.state,this.selection),warnings:[]}:promptChoices(pending, this.store.state);
      this.candidateWarnings = resolved.warnings;
      resolved.warnings.forEach(w => this.replayWarnings.add(w));
      candidates = resolved.choices.map(({raw, label,stage,finish}) => {
        const values=stage?.operation==='append'?[stage.selection.at(-1)]:!stage&&raw!==null?(Array.isArray(raw)?raw:[raw]):[];
        const refs=values.map(value=>choiceBinding(pending,this.store.state,value)).filter(value=>Object.keys(value).length);
        return {raw,stage,view:{id:'',type:staged?'choice':'prompt',label,...(refs.length===1?refs[0]:{}),...(refs.length?{choiceRefs:refs}:{}),...(staged?{choiceOperation:stage?.operation??'finish',selectionCount:this.selection.length}:{})}};
      });
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
    this.captureKnowledge();
    if (playerId === actor && this.status === 'running') {
      try { if(this.knowledgeRestricted[playerId])throw new Error('Search is unavailable: revealed-card or known-order history must be incorporated before sampling hidden states.'); if(this.hypotheticalOpponent)throw new Error('Nested sampling is unavailable for synthetic hypotheses.'); searchPosition = projectPublicPosition(state, playerId, this.decks[playerId]);
        if(this.ownPrizeKnowledge[playerId]) {
          if(this.ownPrizeKnowledge[playerId]!.length!==state.players[playerId].prizes.filter(p=>p.cards.length).length)throw new Error('Search is unavailable: known Prize identities need reconciliation after a Prize was taken.');
          searchPosition.ownPrizeCards=[...this.ownPrizeKnowledge[playerId]!];
        }
        searchPosition.ownDeckTop=this.knownTop[playerId].filter(c=>state.players[playerId].deck.cards.includes(c)).map(printedId);
        searchPosition.ownDeckBottom=this.knownBottom[playerId].filter(c=>state.players[playerId].deck.cards.includes(c)).map(printedId);
        const visible=this.cardCounts(this.ownVisibleCards(playerId));
        searchPosition.ownDeckKnown=[...this.knownNonPrizeCounts[playerId]].flatMap(([id,n])=>Array(Math.max(0,n-(visible.get(id)??0))).fill(id));
        searchPosition.knownOpponentHand=[...this.knownOpponentHand[playerId]].flatMap(([id,n])=>Array(n).fill(id));
        searchPosition.opponentRevealedCounts=Object.fromEntries(this.opponentRevealedCounts[playerId]);
        searchPosition.knowledgeHistory=structuredClone(this.knowledge[playerId]); }
      catch (error) { searchPosition = undefined; searchUnavailableReason = error instanceof Error ? error.message : String(error); }
    }
    return {
      ...(searchPosition ? {searchPosition} : {}), ...(searchUnavailableReason ? {searchUnavailableReason} : {}),
      schemaVersion: 1, playerId, decisionPlayer: actor, turn: state.turn, phase: GamePhase[state.phase], status: this.status,
      players: state.players.map((p, index) => ({
        id: index, name: p.name, active: pokemonView(p.active, index === playerId),
        bench: p.bench.map((b,slotIndex) => {const view=pokemonView(b,index===playerId);return view?{...view,slotIndex}:null;}).filter((b): b is PokemonView & {slotIndex:number} => !!b),
        hand: index === playerId ? p.hand.cards.map(cardView) : [], handCount: p.hand.cards.length,
        deckCount: p.deck.cards.length, prizesRemaining: p.prizes.filter(pr => pr.cards.length).length,
        discard: p.discard.cards.map(cardView),
      })),
      stadium: (()=>{const owner=state.players.findIndex(p=>p.stadium.cards.length);return owner<0?null:{owner,card:cardView(state.players[owner].stadium.cards[0])};})(),
      knowledge: this.knowledge[playerId].map(e=>({...e,cards:e.cards?.map((c:any)=>({...c}))})),
      ownDeck: (this.hypotheticalOpponent&&this.decks[playerId]===this.hypotheticalOpponent.id ? this.hypotheticalOpponent : getDeck(this.decks[playerId])).cards.map((c: any) => ({cardId: c.cardId, name: c.name, count: c.count})),
      legalActions: playerId === actor ? actions.map(c => ({...c.view})) : [],
      history: [...this.publicHistory],
      ...(prompt && actor === playerId ? {prompt: {type: prompt.type, message: String(prompt.message ?? ''), ...(STAGED_PROMPTS.has(prompt.type)?{selectionCount:this.selection.length,selection: this.selection.map((x:any)=>typeof x==='number'&&prompt.cards?.cards?{index:x,name:prompt.options?.isSecret?'hidden card':prompt.cards.cards[x]?.name}:x),selectedChoices:this.selection.map(x=>choiceBinding(prompt,state,x)),min:prompt.options?.min,max:prompt.options?.max,canFinish:actions.some(a=>a.view.choiceOperation==='finish'&&a.raw!==null),canUndo:actions.some(a=>a.view.choiceOperation==='undo')}:{}), ...(['Show cards', 'Confirm cards'].includes(prompt.type) ? {cards: prompt.cards.map(cardView)} : {}), ...(!prompt.options?.isSecret&&prompt.cards?.cards?{cards:prompt.cards.cards.map(cardView)}:{}), ...(prompt.type === 'Show mulligan' ? {hands: prompt.hands.map((h: Card[]) => h.map(cardView))} : {})}} : {}),
      warnings: [...this.warnings, ...(playerId === actor ? this.candidateWarnings : [])],
    };
  }
  private captureKnowledge(){
    for(const p of this.store.state.prompts.filter(p=>p.result===undefined)){
      if(this.seenKnowledge.has(p))continue;
      const prompt:any=p,viewer=p.playerId-1;
      let cards:Card[]|undefined;let type:string|undefined;
      if(['Show cards','Confirm cards'].includes(p.type)){cards=prompt.cards;type='revealed-cards';}
      else if(p.type==='Show mulligan'){cards=prompt.hands.flat();type='mulligan';}
      else if(p.type==='Order cards'){cards=prompt.cards.cards;type='order-choice';}
      else if(p.type==='Choose cards'&&!prompt.options?.isSecret&&this.store.state.players.some(pl=>pl.id!==p.playerId&&pl.hand===prompt.cards)){cards=prompt.cards.cards;type='opponent-hand-reveal';}
      else if(p.type==='Choose cards'&&!prompt.options?.isSecret&&this.store.state.players[viewer].deck===prompt.cards){cards=prompt.cards.cards;type='deck-search';}
      else if(p.type==='Choose cards'&&!prompt.options?.isSecret){
        const publicZones=this.store.state.players.flatMap(pl=>[pl.hand,pl.discard,pl.lostzone,pl.supporter,pl.stadium,pl.active,pl.active.energies,...pl.bench.flatMap(b=>[b,b.energies])]);
        if(!publicZones.includes(prompt.cards)){cards=prompt.cards.cards;type='temporary-zone-reveal';}
      }
      if(cards&&type){this.seenKnowledge.add(p);this.knowledge[viewer].push({decisionIndex:this.decisionIndex,type,cards:cards.map(cardView)});if(type==='deck-search') {
          const owner=this.store.state.players[viewer];
          const known=[...owner.hand.cards,...owner.deck.cards,...owner.discard.cards,...owner.lostzone.cards,...owner.supporter.cards,...owner.stadium.cards,...owner.active.cards,...owner.active.tools,...owner.active.energies.cards,...owner.bench.flatMap(b=>[...b.cards,...b.tools,...b.energies.cards])];
          const remaining=new Map((this.hypotheticalOpponent&&this.decks[viewer]===this.hypotheticalOpponent.id?this.hypotheticalOpponent:getDeck(this.decks[viewer])).cards.map(c=>[c.cardId,c.count]));
          for(const c of new Set(known))remaining.set(printedId(c),(remaining.get(printedId(c))??0)-1);
          const inferred=[...remaining].flatMap(([id,n])=>Array(Math.max(0,n)).fill(id));
          if([...remaining.values()].every(n=>n>=0)&&inferred.length===owner.prizes.filter(p=>p.cards.length).length)this.ownPrizeKnowledge[viewer]=inferred;
          else this.knowledgeRestricted[viewer]=true;
          this.rememberNonPrize(viewer,cards);
        } else if(type==='opponent-hand-reveal'){
          this.knownOpponentHand[viewer]=this.cardCounts(cards);
        } else if(type==='revealed-cards'){
          const opponent=this.store.state.players[1-viewer];
          // Ordinary search reveals occur with the selected cards already in hand.
          // Other show-card destinations need their own effect-specific contract.
          if(cards.every(c=>opponent.hand.cards.includes(c))){
            const gained=this.cardCounts(cards.filter(c=>this.effectStartHands&&!this.effectStartHands[1-viewer].has(c)&&!this.effectReveals[viewer].has(c)));
            const shown=this.cardCounts(cards);for(const[id,n]of shown)this.knownOpponentHand[viewer].set(id,Math.max(n,(this.knownOpponentHand[viewer].get(id)??0)+(gained.get(id)??0)));
            cards.forEach(c=>this.effectReveals[viewer].add(c));
          } else if(!cards.every(c=>this.publicCards(opponent).includes(c)))this.knowledgeRestricted[viewer]=true;
        } else if(type==='mulligan'){
          for(const hand of prompt.hands)for(const[id,n]of this.cardCounts(hand))this.opponentRevealedCounts[viewer].set(id,Math.max(n,this.opponentRevealedCounts[viewer].get(id)??0));
        } else if(type==='temporary-zone-reveal'&&this.activePeek?.viewer===viewer){
          this.rememberNonPrize(viewer,cards);
          this.knownTop[viewer]=this.knownTop[viewer].filter(c=>!cards.includes(c));
          this.knownBottom[viewer]=this.knownBottom[viewer].filter(c=>!cards.includes(c));
        } else if(type!=='order-choice')this.knowledgeRestricted[viewer]=true;
        if(type==='order-choice')this.rememberNonPrize(viewer,cards);
        this.rememberOpponentCounts(viewer);
      }
    }
  }
  private cardCounts(cards:Card[]){const counts=new Map<string,number>();for(const c of cards){const id=printedId(c);counts.set(id,(counts.get(id)??0)+1);}return counts;}
  private publicCards(p:State['players'][number]):Card[]{return [...new Set([...p.active.cards,...p.active.tools,...p.active.energies.cards,...p.bench.flatMap(b=>[...b.cards,...b.tools,...b.energies.cards]),...p.discard.cards,...p.lostzone.cards,...p.supporter.cards,...p.stadium.cards])];}
  private ownVisibleCards(viewer:number):Card[]{const p=this.store.state.players[viewer];return [...new Set([...p.hand.cards,...this.publicCards(p)])];}
  private rememberNonPrize(viewer:number,extra:Card[]=[]){
    // Lower bounds are by printed identity, never by tracking which indistinguishable
    // physical copy happened to be drawn from a hidden randomized deck.
    const seen=this.cardCounts([...new Set([...this.ownVisibleCards(viewer),...this.knownTop[viewer],...this.knownBottom[viewer],...extra])]);
    for(const[id,n]of seen)this.knownNonPrizeCounts[viewer].set(id,Math.max(n,this.knownNonPrizeCounts[viewer].get(id)??0));
  }
  private rememberOpponentCounts(viewer:number){
    const counts=this.cardCounts(this.publicCards(this.store.state.players[1-viewer]));
    for(const[id,n]of this.knownOpponentHand[viewer])counts.set(id,(counts.get(id)??0)+n);
    for(const[id,n]of counts)this.opponentRevealedCounts[viewer].set(id,Math.max(n,this.opponentRevealedCounts[viewer].get(id)??0));
  }
  private reconcilePublicKnowledge(before:Map<string,number>[]){
    for(const viewer of[0,1]){
      const owner=this.store.state.players[1-viewer],current=this.cardCounts(this.publicCards(owner)),hand=this.knownOpponentHand[viewer];
      for(const[id,n]of current){const newlyPublic=Math.max(0,n-(before[1-viewer].get(id)??0));if(newlyPublic)hand.set(id,Math.max(0,(hand.get(id)??0)-newlyPublic));}
      if(owner.hand.cards.length===0)hand.clear();
      if([...hand.values()].reduce((a,b)=>a+b,0)>owner.hand.cards.length)this.knowledgeRestricted[viewer]=true;
      this.rememberOpponentCounts(viewer);
    }
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
    const previousHands=this.store.state.players.map(p=>new Set(p.hand.cards));
    const previousPrizeCounts=this.store.state.players.map(p=>p.prizes.filter(c=>c.cards.length).length);
    const previousPublic=this.store.state.players.map(p=>this.cardCounts(this.publicCards(p)));
    try {
      const prompt = this.pending();
      let bottom:Card[]|undefined;
      if(!prompt){
        this.effectStartHands=this.store.state.players.map(p=>new Set(p.hand.cards));this.effectReveals=[new Set(),new Set()];
        this.activeEffectCardId=chosen.view.cardId;
        this.store.state.players.forEach((_,index)=>this.rememberNonPrize(index));
        this.activePeek=chosen.action instanceof UseAbilityAction&&chosen.view.cardId==='TWM-129'&&chosen.action.name==='Recon Directive'?{viewer:this.actor,kind:'recon'}:chosen.action instanceof PlayCardAction&&chosen.view.cardId==='BLK-84'?{viewer:this.actor,kind:'pokegear'}:undefined;
      }
      if(chosen.stage){this.selection=[...chosen.stage.selection];}
      else if (prompt) {
        if(prompt.type==='Order cards'&&chosen.raw!==null){const p:any=prompt;this.knowledge[this.actor].push({decisionIndex:this.decisionIndex,type:'ordered-cards',cards:chosen.raw.map((i:number)=>cardView(p.cards.cards[i]))});this.knownTop[this.actor]=chosen.raw.map((i:number)=>p.cards.cards[i]);this.knownBottom[this.actor]=this.knownBottom[this.actor].filter(c=>!this.knownTop[this.actor].includes(c));}
        if(prompt.type==='Choose cards'&&this.activePeek?.kind==='recon'&&Array.isArray(chosen.raw)){const p:any=prompt;bottom=p.cards.cards.filter((_:Card,i:number)=>!chosen.raw.includes(i));}
        const decoded = prompt.decode(chosen.raw, this.store.state);
        if (!prompt.validate(decoded, this.store.state)) throw new Error('Prompt candidate failed validation.');
        this.random.scoped(() => this.store.dispatch(new ResolvePromptAction(prompt.id, decoded)));
      } else this.random.scoped(() => this.store.dispatch(chosen.action!));
      if(!chosen.stage){this.selection=[];this.selectionPrompt=undefined;}
      if(!chosen.stage)this.publicHistory.push(`P${before.actor + 1}: ${prompt ? `resolved ${prompt.type}` : chosen.view.label}`);
      this.recorded.push(before); this.actionIds.push(actionId); this.decisionIndex++; this.candidates = null;
      this.resolveChance();
      if(bottom&&this.activePeek){const viewer=this.activePeek.viewer;this.knownBottom[viewer]=[...this.knownBottom[viewer].filter(c=>this.store.state.players[viewer].deck.cards.includes(c)&&!bottom!.includes(c)),...bottom];this.knowledge[viewer].push({decisionIndex:this.decisionIndex,type:'known-bottom',cards:bottom.map(cardView)});}
      this.reconcilePublicKnowledge(previousPublic);
      this.store.state.players.forEach((player,index)=>{
        const prizeCount=player.prizes.filter(p=>p.cards.length).length,prizesTaken=previousPrizeCounts[index]-prizeCount;
        if(prizesTaken>0){const gained=player.hand.cards.filter(c=>!previousHands[index].has(c));if(gained.length===prizesTaken)for(const card of gained){const id=printedId(card);this.knownNonPrizeCounts[index].set(id,(this.knownNonPrizeCounts[index].get(id)??0)+1);}}
        const inferred=this.ownPrizeKnowledge[index];if(!inferred)return;
        const taken=inferred.length-player.prizes.filter(p=>p.cards.length).length;
        if(taken<=0)return;
        const additions=player.hand.cards.filter(c=>!previousHands[index].has(c));
        if(additions.length!==taken)return; // Keep stale count: search explicitly refuses.
        const remaining=[...inferred];
        for(const c of additions){const at=remaining.indexOf(printedId(c));if(at<0)return;remaining.splice(at,1);}
        this.ownPrizeKnowledge[index]=remaining;
      });
      return this.result();
    } catch (error) {
      // Dispatch may mutate its state before throwing. Reconstruct all callback closures as well.
      if (this.hypothetical) { this.fail('Hypothetical continuation failed'); throw error; }
      this.reset(seed, decks, this.firstPlayer);
      for (const id of oldIds) this.step(id);
      throw error;
    }
  }
  static fromPublicPosition(position: PublicPosition, seed: number, opponentDeckId: DeckHypothesis, decisionIndex = 0): Environment {
    const copy = new Environment(); copy.seed = seed; copy.random = new SeededRandom(seed); copy.hypothetical = true;
    const opponentId=typeof opponentDeckId==='string'?opponentDeckId:opponentDeckId.id;
    if(typeof opponentDeckId!=='string')copy.hypotheticalOpponent=opponentDeckId;
    copy.decks = position.observer === 0 ? [position.ownDeckId, opponentId] : [opponentId, position.ownDeckId];
    copy.store = new Store({onStateChange: () => {}});
    copy.store.state = samplePublicPosition(position, opponentDeckId, copy.random);
    const viewer=position.observer,own=copy.store.state.players[viewer];
    copy.knowledge[viewer]=structuredClone(position.knowledgeHistory??[]);
    copy.ownPrizeKnowledge[viewer]=position.ownPrizeCards?[...position.ownPrizeCards]:undefined;
    copy.knownTop[viewer]=own.deck.cards.slice(0,position.ownDeckTop?.length??0);
    copy.knownBottom[viewer]=position.ownDeckBottom?.length?own.deck.cards.slice(-position.ownDeckBottom.length):[];
    copy.knownNonPrizeCounts[viewer]=copy.cardCounts(copy.ownVisibleCards(viewer));
    for(const id of position.ownDeckKnown??[])copy.knownNonPrizeCounts[viewer].set(id,(copy.knownNonPrizeCounts[viewer].get(id)??0)+1);
    copy.rememberNonPrize(viewer);
    for(const id of position.knownOpponentHand??[])copy.knownOpponentHand[viewer].set(id,(copy.knownOpponentHand[viewer].get(id)??0)+1);
    copy.opponentRevealedCounts[viewer]=new Map(Object.entries(position.opponentRevealedCounts??{}));
    copy.decisionIndex = decisionIndex; return copy;
  }
  /** Curated research positions use actual registered cards and an explicit recipe. */
  static fromFixtureState(state:State,seed:number,decks:[string,string],reconstruct:()=>Environment):Environment {
    decks.forEach(id=>getDeck(id));const e=new Environment();e.seed=seed;e.decks=[...decks];
    e.random=new SeededRandom(seed,value=>e.chance.push({decisionIndex:e.decisionIndex,type:'upstream-random',result:value}));
    e.store=new Store({onStateChange:()=>{}});e.store.state=state;e.reconstructFixture=reconstruct;return e;
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
    const copy = this.reconstructFixture?this.reconstructFixture():new Environment(); if(!this.reconstructFixture)copy.reset(this.seed, this.decks, this.firstPlayer);
    for (const id of this.actionIds) copy.step(id);
    return copy;
  }
  replay(status?: Replay['status']): Replay {
    const finished = this.status === 'finished';
    const winner = this.store.state.winner;
    const actualStatus = status ?? (finished ? 'finished' : this.status === 'error' ? 'error' : 'truncated');
    if (actualStatus === 'finished' && !finished) throw new Error('Cannot mark a nonterminal game finished.');
    const id = createHash('sha256').update(JSON.stringify({seed: this.seed, firstPlayer:this.firstPlayer, decks: this.decks, actions: this.actionIds, engine: ENGINE_VERSION})).digest('hex').slice(0, 20);
    let finalFrame: ReplayFrame;
    try { finalFrame = this.frame(null); } catch { this.failure ??= 'Unable to enumerate current decision'; finalFrame = {decisionIndex: this.decisionIndex, actor: this.actor, action: null, observations: this.recorded.at(-1)?.observations ?? [] as any}; }
    return {
      schemaVersion: 1, id, seed: this.seed, ...(this.firstPlayer!==undefined?{firstPlayer:this.firstPlayer}:{}), decks: [...this.decks], engineVersion: ENGINE_VERSION,
      status: actualStatus, outcome: finished ? {winner: winner === GameWinner.PLAYER_1 ? 0 : winner === GameWinner.PLAYER_2 ? 1 : null, reason: winner === GameWinner.DRAW ? 'rules-draw' : winner === GameWinner.NONE ? 'engine-ended-without-winner' : 'rules-terminal'} : null,
      frames: [...this.recorded, finalFrame], warnings: [...this.warnings, ...this.replayWarnings, ...(this.status === 'error' && !this.failure ? ['Engine ended without a valid winner; this is not a draw.'] : []), ...(this.failure ? [this.failure] : [])], visibility: 'private-research', chance: [...this.chance],
    };
  }
  fail(message: string) { this.failure = message; this.candidates = []; }
}
export { getDecks };
