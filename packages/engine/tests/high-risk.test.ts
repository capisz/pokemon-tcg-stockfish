import test from 'node:test';
import assert from 'node:assert/strict';
import { State, GamePhase } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { Store } from '../../../vendor/twinleaf/ptcg-server/src/game/store/store';
import { PassTurnAction,AttackAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/game-actions';
import { PlayCardAction,PlayerType,SlotType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { ResolvePromptAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/resolve-prompt-action';
import { CheckTableStateEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/check-effects';
import { checkState } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effect-reducers/check-effect';
import { CARD_FACTORIES as F,registerCards } from '../src/catalog';
import { promptChoices,stagedChoices,STAGED_PROMPTS } from '../src/choices';
import { deepClone } from '../../../vendor/twinleaf/ptcg-server/src/utils/utils';
registerCards();
function board(){
  const state=new State(),store=new Store({onStateChange:()=>{}}),a=new Player(),b=new Player();a.id=1;b.id=2;let id=0;
  for(const p of[a,b]){p.canEvolve=true;p.active.cards=[F['DRI-12']()];p.active.isSecret=false;p.active.pokemonPlayedTurn=1;p.bench=Array.from({length:5},()=>new PokemonCardList());p.deck.cards=Array.from({length:12},()=>F['MEE-1']());p.prizes=Array.from({length:6},()=>{const c=new CardList();c.cards=[F['MEE-1']()];return c;});for(const c of[...p.active.cards,...p.deck.cards,...p.prizes.flatMap(q=>q.cards)]){c.id=id++;state.cardNames[c.id]=c.fullName;}}
  state.players=[a,b];state.phase=GamePhase.PLAYER_TURN;state.turn=5;store.state=state;return{state,store,a,b};
}
function resolveAll(store:Store,state:State){
  for(let i=0;i<100;i++){const p:any=state.prompts.find(p=>p.result===undefined);if(!p)return;
    let raw:any;
    if(p.type==='WaitPrompt')raw=true;
    else if(STAGED_PROMPTS.has(p.type)){let selected:any[]=[];for(let n=0;n<60;n++){const choices=stagedChoices(p,state,selected),append=choices.find(c=>c.stage?.operation==='append'),finish=choices.find(c=>c.finish&&c.raw!==null);if(finish){raw=finish.raw;break;}if(!append)throw new Error('No fixture continuation');selected=append.stage!.selection;}}
    else raw=promptChoices(p,state).choices[0].raw;
    store.dispatch(new ResolvePromptAction(p.id,p.decode(raw,state)));
  }throw new Error('Fixture prompt loop');
}
function attach(target:PokemonCardList,...ids:string[]){const cards=ids.map(id=>F[id]());target.cards.push(...cards);target.energies.cards.push(...cards);}
test('Froslass checkup counts both players sources, excludes Froslass and Pokemon without Abilities',()=>{
  const {state,store,a,b}=board();a.bench[0].cards=[F['TWM-53']()];a.bench[1].cards=[F['TWM-53']()];a.bench[2].cards=[F['MEG-76']()];b.bench[0].cards=[F['TWM-53']()];
  store.dispatch(new PassTurnAction(1));resolveAll(store,state);
  assert.equal(a.active.damage,30);assert.equal(b.active.damage,30);assert.equal(a.bench[0].damage,0);assert.equal(a.bench[1].damage,0);assert.equal(a.bench[2].damage,0);assert.equal(b.bench[0].damage,0);
});
test('Froslass simultaneous checkup knockouts award each player a Prize before next turn',()=>{
  const {state,store,a,b}=board();a.active.damage=140;b.active.damage=140;a.bench[0].cards=[F['TWM-53']()];b.bench[0].cards=[F['MEG-76']()];
  store.dispatch(new PassTurnAction(1));resolveAll(store,state);
  assert.ok(a.discard.cards.some(c=>c.name==='Crustle'));assert.ok(b.discard.cards.some(c=>c.name==='Crustle'));assert.equal(a.prizes.filter(p=>p.cards.length).length,5);assert.equal(b.prizes.filter(p=>p.cards.length).length,5);
});
test('Area Zero expands only Tera players and shrinking preserves attachments in discard',()=>{
  const {state,store,a,b}=board();a.active.cards=[F['TWM-25']()];b.active.cards=[F['TEF-123']()];a.stadium.cards=[F['SCR-131']()];
  const sizes=new CheckTableStateEffect([5,5]);store.reduceEffect(state,sizes);assert.deepEqual(sizes.benchSizes,[8,5]);checkState(store,state);resolveAll(store,state);assert.equal(a.bench.length,8);
  a.bench.forEach((slot,i)=>{slot.cards=[F['TWM-128']()];slot.isSecret=false;if(i<3){attach(slot,'MEE-1');slot.tools=[F['ASC-181']()];}});
  a.stadium.moveTo(a.discard);state.benchSizeChangeHandled=false;checkState(store,state);resolveAll(store,state);
  assert.equal(a.bench.length,5);assert.equal(a.discard.cards.filter(c=>c.name==='Dreepy').length,3);assert.equal(a.discard.cards.filter(c=>c.name==='Grass Energy').length,3);assert.equal(a.discard.cards.filter(c=>c.name==='Air Balloon').length,3);
});
test('Area Zero replacement asks its original owner to shrink first',()=>{
  const {state,store,a,b}=board();b.stadium.cards=[F['SCR-131']()];
  for(const p of[a,b]){p.active.cards=[F['TWM-25']()];p.bench=Array.from({length:8},()=>{const c=new PokemonCardList();c.cards=[F['TWM-128']()];c.isSecret=false;return c;});}
  a.hand.cards=[F['SSP-177']()];store.dispatch(new PlayCardAction(1,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));
  let p:any=state.prompts.find(p=>p.result===undefined);while(p?.type==='WaitPrompt'){store.dispatch(new ResolvePromptAction(p.id,true));p=state.prompts.find(p=>p.result===undefined);}
  assert.equal(p.type,'Choose pokemon');assert.equal(p.playerId,2);resolveAll(store,state);assert.equal(a.bench.length,5);assert.equal(b.bench.length,5);
});
test('Aura Jab permits zero recovery and separate Fighting Energy targets',()=>{
  for(const count of[0,3]){
    const {state,store,a,b}=board();a.active.cards=[F['MEG-76'](),F['MEG-77']()];attach(a.active,'MEE-6');b.active.cards=[F['MEG-77']()];a.bench[0].cards=[F['MEG-76']()];a.bench[1].cards=[F['MEG-75']()];a.discard.cards=[F['MEE-6'](),F['MEE-6'](),F['MEE-6'](),F['MEE-1']()];
    store.dispatch(new AttackAction(1,'Aura Jab'));let p:any=state.prompts.find(p=>p.result===undefined);while(p?.type==='WaitPrompt'){store.dispatch(new ResolvePromptAction(p.id,true));p=state.prompts.find(p=>p.result===undefined);}
    assert.equal(p.type,'Attach energy');const raw=Array.from({length:count},(_,index)=>({index,to:{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.BENCH,index:index%2}}));assert.ok(p.validate(p.decode(raw,state),state));store.dispatch(new ResolvePromptAction(p.id,p.decode(raw,state)));resolveAll(store,state);
    assert.equal(a.bench.reduce((n,b)=>n+b.energies.cards.length,0),count);assert.equal(a.discard.cards.filter(c=>c.name==='Fighting Energy').length,3-count);assert.equal(b.active.damage,130);
  }
});
test('Premium Power Pro copies stack for Lucario and expire after its turn',()=>{
  const {state,store,a,b}=board();a.active.cards=[F['MEG-77']()];attach(a.active,'MEE-6');b.active.cards=[F['MEG-77']()];a.hand.cards=[F['MEG-124'](),F['MEG-124']()];
  for(let i=0;i<2;i++){store.dispatch(new PlayCardAction(1,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));resolveAll(store,state);}
  store.dispatch(new AttackAction(1,'Aura Jab'));resolveAll(store,state);assert.equal(b.active.damage,190);assert.ok(!a.marker.markers.some(m=>m.name==='POWER_PROTEIN_MARKER'));
});
test('Hariyama hand evolution may gust an opposing Benched Pokemon',()=>{
  const {state,store,a,b}=board();a.active.cards=[F['MEG-72']()];b.bench[0].cards=[F['MEG-76']()];a.hand.cards=[F['MEG-73']()];const old=b.active;
  store.dispatch(new PlayCardAction(1,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));resolveAll(store,state);assert.equal(a.active.getPokemonCard()!.name,'Hariyama');assert.equal(b.active.getPokemonCard()!.name,'Riolu');assert.ok(b.bench.includes(old));
});
test('Mega Brave cannot be reused next turn and becomes available after that turn ends',()=>{
  const {state,store,a,b}=board();a.active.cards=[F['MEG-77']()];attach(a.active,'MEE-6','MEE-6');b.active.cards=[F['MEG-77']()];b.bench[0].cards=[F['MEG-76']()];
  store.dispatch(new AttackAction(1,'Mega Brave'));resolveAll(store,state);store.dispatch(new PassTurnAction(2));resolveAll(store,state);
  const probe=new Store({onStateChange:()=>{}});probe.state=deepClone(state);assert.throws(()=>probe.dispatch(new AttackAction(1,'Mega Brave')));
  store.dispatch(new PassTurnAction(1));resolveAll(store,state);store.dispatch(new PassTurnAction(2));resolveAll(store,state);
  assert.doesNotThrow(()=>store.dispatch(new AttackAction(1,'Mega Brave')));resolveAll(store,state);
});
