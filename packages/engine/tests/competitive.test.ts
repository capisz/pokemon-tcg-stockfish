import test from 'node:test';
import assert from 'node:assert/strict';
import { State, GamePhase } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { Store } from '../../../vendor/twinleaf/ptcg-server/src/game/store/store';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { PowerEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/game-effects';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { ChooseCardsPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/choose-cards-prompt';
import { ChoosePrizePrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/choose-prize-prompt';
import { AttachEnergyPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/attach-energy-prompt';
import { OrderCardsPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/order-cards-prompt';
import { GameMessage } from '../../../vendor/twinleaf/ptcg-server/src/game/game-message';
import { PlayerType, SlotType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { PlayCardAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { ResolvePromptAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/resolve-prompt-action';
import { EndTurnEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/game-phase-effects';
import { AttackEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/game-effects';
import { DealDamageEffect, PutCountersEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/attack-effects';
import { CARD_FACTORIES as F, registerCards, getDecks, validateDeck } from '../src/catalog';
import { promptChoices, stagedChoices } from '../src/choices';
import { Environment } from '../src/environment';
import { chooseAction } from '../src/policies';
import { SeededRandom } from '../src/random';
import { opponentHypotheses } from '../src/hypotheses';
import { search } from '../src/search';
registerCards();
function board(source='TWM-107') {
  const state=new State(), store=new Store({onStateChange:()=>{}});
  const a=new Player(), b=new Player();a.id=1;b.id=2;
  a.active.cards=[F[source]()];b.active.cards=[F['DRI-12']()];
  for(const p of [a,b]){p.bench=Array.from({length:5},()=>new PokemonCardList());p.prizes=Array.from({length:6},()=>{const c=new CardList();c.cards=[F['MEE-1']()];return c;});p.deck.cards=[F['MEE-1']()];}
  state.players=[a,b];state.phase=GamePhase.PLAYER_TURN;state.turn=3;store.state=state;
  return {state,store,a,b};
}
function ready(){const e=new Environment();e.reset(5,['crustle','mega-lucario'],0);const r=new SeededRandom(5);for(let i=0;i<100&&!e.observe().searchPosition;i++)e.step(chooseAction(e.observe(),'heuristic',r).id);assert.ok(e.observe().searchPosition);return e;}

test('staged selections reach a combination beyond the former cap, with reversible prefixes',()=>{
  const {state,a}=board();a.hand.cards=Array.from({length:24},()=>F['MEE-1']());
  const p=new ChooseCardsPrompt(a,GameMessage.CHOOSE_CARDS,a.hand,{}, {min:12,max:12,allowCancel:false});
  let selected:any[]=[];
  for(const i of [23,21,19,17,15,13,11,9,7,5,3,1]) {
    const options=stagedChoices(p,state,selected);
    assert.ok(!options.some(c=>c.finish));
    selected=options.find(c=>c.stage?.selection.at(-1)===i&&c.stage.operation==='append')!.stage!.selection;
  }
  const choices=stagedChoices(p,state,selected);assert.ok(choices.some(c=>c.finish));
  assert.deepEqual(choices.find(c=>c.stage?.operation==='undo')!.stage!.selection,selected.slice(0,-1));
  assert.ok(p.validate(p.decode(choices.find(c=>c.finish)!.raw)!));
});

test('staged attachment covers distinct targets and respects uniqueness',()=>{
  const {state,a}=board();a.bench[0].cards=[F['MEG-104']()];a.bench[1].cards=[F['POR-62']()];a.discard.cards=[F['MEE-1'](),F['MEE-2']()];
  const p=new AttachEnergyPrompt(1,GameMessage.CHOOSE_CARDS,a.discard,PlayerType.BOTTOM_PLAYER,[SlotType.BENCH],{}, {min:1,max:2,differentTargets:true,allowCancel:false});
  const first=stagedChoices(p,state,[]).find(c=>c.stage?.selection[0].index===0)!;
  const next=stagedChoices(p,state,first.stage!.selection);
  assert.ok(next.some(c=>c.finish));
  assert.ok(next.filter(c=>c.stage?.operation==='append').every(c=>c.stage!.selection[1].index===1&&c.stage!.selection[1].to.index!==first.stage!.selection[0].to.index));
});

test('prize choice uses compressed indexes after prior prizes were taken',()=>{
  const {state,a}=board();a.prizes[0].cards=[];a.prizes[2].cards=[];
  const p=new ChoosePrizePrompt(1,GameMessage.CHOOSE_PRIZE_CARD,{count:3});
  const options=promptChoices(p,state).choices;
  assert.equal(options.length,4);assert.ok(options.every(c=>p.validate(p.decode(c.raw,state),state)));
});

test('Crustle protection suppresses Spiky retaliation; real damage triggers each Spiky and ignores attacker Mist',()=>{
  for(const [source,retaliation]of [['TEF-123',0],['TWM-107',40]] as const){
    const {state,store,a,b}=board(source);state.phase=GamePhase.ATTACK;
    b.active.cards.push(F['TEF-161'](),F['JTG-159'](),F['JTG-159']());a.active.cards.push(F['TEF-161']());
    const attack=new AttackEffect(a,b,a.active.getPokemonCard()!.attacks[0]);store.reduceEffect(state,new DealDamageEffect(attack,40));
    assert.equal(a.active.damage,retaliation);assert.equal(b.active.damage,retaliation?40:0);
  }
});

test('Mist prevents opposing attack counters but not an ability counter transfer',()=>{
  const {state,store,a,b}=board('TWM-95');b.active.cards.push(F['TEF-161']());state.phase=GamePhase.ATTACK;
  const attack=new AttackEffect(a,b,a.active.getPokemonCard()!.attacks[0]);
  store.reduceEffect(state,new PutCountersEffect(attack,30));assert.equal(b.active.damage,0);
  state.phase=GamePhase.PLAYER_TURN;a.active.cards.push(F['MEE-7']());a.active.energies.cards=[a.active.cards[1]];a.active.damage=30;
  const munki=a.active.getPokemonCard()!;store.reduceEffect(state,new PowerEffect(a,munki.powers[0],munki));
  const p=state.prompts.find(p=>p.result===undefined)!;assert.equal(p.type,'Remove damage');
  const choices=promptChoices(p,state).choices;store.dispatch(new ResolvePromptAction(p.id,p.decode(choices[0].raw,state)));
  assert.equal(b.active.damage,30);assert.equal(a.active.damage,0);
});

test('Crushing Hammer is discarded on tails and only heads removes an opposing energy',()=>{
  for(const heads of [false,true]){
    const {state,store,a,b}=board();const hammer=F['POR-71']();a.hand.cards=[hammer];const energy=F['MEE-1']();b.active.cards.push(energy);b.active.energies.cards=[energy];
    const previous=Math.random;
    try {Math.random=()=>heads?0.1:0.9;store.dispatch(new PlayCardAction(1,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));}
    finally {Math.random=previous;}
    let pending=state.prompts.find(p=>p.result===undefined)!;assert.equal(pending.type,'WaitPrompt');
    while((pending=state.prompts.find(p=>p.result===undefined)!)){const c=pending.type==='WaitPrompt'?{raw:true}:promptChoices(pending,state).choices[0];store.dispatch(new ResolvePromptAction(pending.id,pending.decode(c.raw,state)));}
    assert.ok(a.discard.cards.includes(hammer));assert.equal(b.discard.cards.includes(energy),heads);
  }
});

test('own deck search infers Prize multiset; known top order constrains independent samples',()=>{
  const e=ready(), actor=e.actor, player=e.store.state.players[actor];
  const toolIndex=player.deck.cards.findIndex(c=>c.name==="Hero's Cape");assert.ok(toolIndex>=0);player.active.tools.push(player.deck.cards.splice(toolIndex,1)[0]);
  const p=new ChooseCardsPrompt(player,GameMessage.CHOOSE_CARDS,player.deck,{}, {min:0,max:0,allowCancel:false});
  e.store.prompt(e.store.state,p,()=>{});(e as any).candidates=null;
  const finish=e.observe().legalActions.find(a=>a.choiceOperation==='finish')!;e.step(finish.id);
  let observation=e.observe();assert.ok(observation.searchPosition);assert.equal(observation.searchPosition.ownPrizeCards?.length,6);
  const actual=player.prizes.pop()!;const stale=e.observe();assert.equal(stale.searchPosition,undefined);assert.match(stale.searchUnavailableReason!,/Prize identities/);player.prizes.push(actual);
  const top=new CardList();top.cards=player.deck.cards.slice(0,2);
  e.store.prompt(e.store.state,new OrderCardsPrompt(player.id,GameMessage.CHOOSE_CARDS_ORDER,top,{allowCancel:false}),order=>{top.applyOrder(order!);const rest=player.deck.cards.filter(c=>!top.cards.includes(c));player.deck.cards=[...top.cards,...rest];});(e as any).candidates=null;
  while(e.observe().prompt){const o=e.observe();e.step((o.legalActions.find(a=>a.choiceOperation==='append')??o.legalActions.find(a=>a.choiceOperation==='finish'))!.id);}
  observation=e.observe();assert.ok(observation.searchPosition);assert.equal(observation.searchPosition.ownDeckTop?.length,2);
  for(const seed of [1,2]){const sample=Environment.fromPublicPosition(observation.searchPosition,seed,'mega-lucario');assert.deepEqual(sample.store.state.players[actor].deck.cards.slice(0,2).map(c=>`${c.set}-${c.setNumber}`),observation.searchPosition.ownDeckTop);}
});

test('heldout exact lists stay excluded; unknown variants and explicit lab lists are distinct',()=>{
  const o=ready().observe();const hs=opponentHypotheses(o.searchPosition!,['TWM-148']);
  assert.ok(hs.every(h=>!h.id.includes('heldout')));assert.ok(hs.some(h=>h.kind==='unknown-variant'));
  assert.ok(hs.filter(h=>h.kind==='unknown-variant').every(h=>validateDeck(h.deck).length===0));
  assert.ok(Math.abs(hs.reduce((n,h)=>n+h.weight,0)-1)<1e-9);
  const known=opponentHypotheses(o.searchPosition!,[],'mega-lucario-heldout');assert.equal(known[0].kind,'known-list');
});

test('Special Red Card shuffles the opponent hand before bottom-decking and draws the existing top three',()=>{
  const {state,store,a,b}=board();b.prizes=b.prizes.slice(0,3);const red=F['CRI-82']();a.hand.cards=[red];
  b.hand.cards=[F['MEE-1'](),F['MEE-2'](),F['MEE-7']()];const oldHand=[...b.hand.cards];
  b.deck.cards=[F['TWM-128'](),F['TWM-129'](),F['TWM-130']()];const oldTop=[...b.deck.cards];
  store.dispatch(new PlayCardAction(1,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));
  const p=state.prompts.find(p=>p.result===undefined)!;assert.equal(p.type,'Shuffle hand');assert.equal(p.playerId,2);
  store.dispatch(new ResolvePromptAction(p.id,[2,0,1]));assert.deepEqual(b.hand.cards,oldTop);assert.deepEqual(b.deck.cards,[oldHand[2],oldHand[0],oldHand[1]]);
});

test('opponent hand revelation refuses search and never becomes the other player\'s knowledge',()=>{
  const e=ready(), actor=e.actor,player=e.store.state.players[actor],other=e.store.state.players[1-actor];
  const p=new ChooseCardsPrompt(player,GameMessage.CHOOSE_CARDS,other.hand,{}, {min:0,max:0,allowCancel:false});
  e.store.prompt(e.store.state,p,()=>{});(e as any).candidates=null;
  assert.ok(e.observe().knowledge?.some(k=>k.type==='opponent-hand-reveal'));
  assert.ok(!e.observe(1-actor).knowledge?.some(k=>k.type==='opponent-hand-reveal'));
  e.step(e.observe().legalActions.find(a=>a.choiceOperation==='finish')!.id);
  assert.equal(e.observe().searchPosition,undefined);assert.match(e.observe().searchUnavailableReason!,/revealed-card/);
});

test('learned root priors guide information-set search without claiming learned leaf values',()=>{
  const o=ready().observe(),target=o.legalActions.at(-1)!;
  const result=search({observation:o,method:'ismcts',iterations:1,budgetMs:5000,maxRolloutDecisions:1,rootPriors:[{actionId:target.id,probability:1}]});
  assert.equal(result.alternatives.find(a=>a.actionId===target.id)?.visits,1);assert.ok(result.warnings.some(w=>w.includes('untrained')));
  assert.throws(()=>search({observation:o,rootPriors:[{actionId:'stale',probability:1}]}),/Invalid root/);
});

test('Meowth Last-Ditch Catch is shared across copies and resets at turn end',()=>{
  const {state,store,a}=board();a.deck.cards=[F['MEG-114'](),F['MEE-1']()];
  const first=F['POR-62'](), second=F['POR-62']();a.hand.cards=[first,second];
  store.dispatch(new PlayCardAction(1,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.BENCH,index:0}));
  let p=state.prompts.find(p=>p.result===undefined)!;assert.equal(p.type,'Confirm');store.dispatch(new ResolvePromptAction(p.id,true));
  while((p=state.prompts.find(p=>p.result===undefined)!)){
    let raw:any;
    if(p.type==='Shuffle deck')raw=a.deck.cards.map((_,i)=>i);
    else if(p.type==='WaitPrompt')raw=true;
    else raw=promptChoices(p,state).choices[0].raw;
    store.dispatch(new ResolvePromptAction(p.id,p.decode(raw,state)));
  }
  assert.ok(a.marker.hasMarker('TRUMP_CARD_MARKER'));
  store.dispatch(new PlayCardAction(1,a.hand.cards.indexOf(second),{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.BENCH,index:1}));
  assert.ok(!state.prompts.some(p=>p.result===undefined));
  store.reduceEffect(state,new EndTurnEffect(a));assert.ok(!a.marker.hasMarker('TRUMP_CARD_MARKER'));
});
