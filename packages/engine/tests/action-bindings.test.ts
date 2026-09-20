import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { State, GamePhase } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { ChooseCardsPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/choose-cards-prompt';
import { AttachEnergyPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/attach-energy-prompt';
import { GameMessage } from '../../../vendor/twinleaf/ptcg-server/src/game/game-message';
import { PlayerType, SlotType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { CARD_FACTORIES as F } from '../src/catalog';
import { Environment } from '../src/environment';
import { choiceBinding } from '../src/action-bindings';
import { stagedChoices } from '../src/choices';

function fixture(){
  const e=new Environment();e.reset(5,['crustle','mega-lucario'],0);
  const state=new State(),a=new Player(),b=new Player();a.id=1;b.id=2;
  for(const p of[a,b]){p.active.cards=[F['DRI-12']()];p.active.isSecret=false;p.bench=Array.from({length:8},()=>new PokemonCardList());p.deck.cards=[F['MEE-1']()];p.prizes=Array.from({length:6},()=>{const c=new CardList();c.cards=[F['MEE-1']()];return c;});}
  state.players=[a,b];state.phase=GamePhase.PLAYER_TURN;state.turn=4;e.store.state=state;(e as any).candidates=null;
  return {e,state,a,b};
}
test('card actions bind exact hand and sparse expanded Bench slots, retaining inspectable attachments',()=>{
  const {e,a}=fixture();a.bench[6].cards=[F['MEG-104']()];a.hand.cards=[F['MEE-1'](),F['MEE-1']()];
  const energy=F['MEE-2'](),tool=F['TEF-152']();a.active.cards.push(energy);a.active.energies.cards=[energy];a.active.tools=[tool];
  const o=e.observe(),action=o.legalActions.find(x=>x.type==='attach-energy'&&x.targetRef?.index===6)!;
  assert.ok(action);assert.deepEqual(action.sourceRef,{playerId:0,zone:'hand',index:0});assert.deepEqual(action.targetRef,{playerId:0,zone:'bench',index:6});
  assert.equal(o.players[0].bench[0].slotIndex,6);assert.equal(o.players[0].active!.attachments!.length,2);assert.deepEqual(o.players[0].active!.energy,[energy.name]);
  const target=a.bench[6];e.step(action.id);assert.equal(target.cards.filter(c=>c.name==='Grass Energy').length,1);assert.equal(a.hand.cards.length,1);
});
test('staged choice bindings are derived from source lists and target objects, not text',()=>{
  const {state,a}=fixture();a.bench[7].cards=[F['MEG-104']()];a.discard.cards=[F['MEE-1'](),F['MEE-2']()];
  const p=new AttachEnergyPrompt(1,GameMessage.CHOOSE_CARDS,a.discard,PlayerType.BOTTOM_PLAYER,[SlotType.BENCH],{}, {min:1,max:1,allowCancel:false});
  const choice=stagedChoices(p,state,[]).find(c=>c.stage?.selection[0].index===1)!;choice.label='arbitrary translated label';
  assert.deepEqual(choiceBinding(p,state,choice.stage!.selection[0]),{sourceRef:{playerId:0,zone:'discard',index:1},targetRef:{playerId:0,zone:'bench',index:7},cardId:'MEE-2'});
});
test('private prompt metadata excludes secret card identities and is restricted to decision player',()=>{
  const {e,a}=fixture();const hidden=new CardList();hidden.cards=[F['POR-62']()];
  const p=new ChooseCardsPrompt(a,GameMessage.CHOOSE_CARDS,hidden,{}, {min:1,max:1,isSecret:true,allowCancel:false});
  e.store.prompt(e.store.state,p,()=>{});(e as any).candidates=null;
  const o=e.observe(0),action=o.legalActions.find(x=>x.choiceOperation==='append')!;
  assert.equal(action.cardId,undefined);assert.deepEqual(action.sourceRef,{playerId:0,zone:'prompt',index:0});assert.equal(o.prompt?.cards,undefined);assert.equal(e.observe(1).prompt,undefined);
  e.step(action.id);const selected=e.observe();assert.equal(selected.prompt?.selectionCount,1);assert.equal(selected.prompt?.canFinish,true);assert.equal(selected.prompt?.canUndo,true);assert.equal(selected.prompt?.selectedChoices?.[0].cardId,undefined);
});
test('worker choose returns a reproducible current action without advancing game or chance',()=>{
  const messages=[{id:1,method:'reset',params:{seed:5,decks:['crustle','mega-lucario'],firstPlayer:0}},{id:2,method:'replay'},{id:3,method:'choose',params:{policy:'random',seed:123}},{id:4,method:'choose',params:{policy:'random',seed:123}},{id:5,method:'replay'},{id:6,method:'observe'},{id:7,method:'choose',params:{policy:'invalid'}}];
  const child=spawnSync(process.execPath,['packages/engine/dist/worker.cjs'],{input:messages.map(m=>JSON.stringify(m)).join('\n')+'\n',encoding:'utf8'});
  assert.equal(child.status,0,child.stderr);const output=child.stdout.trim().split('\n').map(line=>JSON.parse(line));
  assert.deepEqual(output[2].result,output[3].result);assert.deepEqual(output[1].result,output[4].result);assert.ok(output[5].result.legalActions.some((a:any)=>a.id===output[2].result.id));assert.match(output[6].error.message,/Unknown policy/);
});
