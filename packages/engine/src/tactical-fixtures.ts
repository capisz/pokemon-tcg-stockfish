import { createHash } from 'node:crypto';
import { State, GamePhase } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { CheckHpEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/check-effects';
import { Environment, ENGINE_VERSION } from './environment';
import { instantiateDeck, getDeck, printedId } from './catalog';
import { SeededRandom } from './random';
import { chooseAction } from './policies';
import type { LegalAction } from './types';

export const FIXTURE_FAMILIES=['delay-prize','energy-function','information-order','hammer-target'] as const;
type Family=typeof FIXTURE_FAMILIES[number];
const assert=(condition:unknown,message:string)=>{if(!condition)throw new Error('Fixture assertion: '+message);};
export function assertConservation(e:Environment){
  for(const[pIndex,p]of e.store.state.players.entries()){
    const all=[...p.hand.cards,...p.deck.cards,...p.discard.cards,...p.lostzone.cards,...p.stadium.cards,...p.supporter.cards,...p.prizes.flatMap(c=>c.cards),...[p.active,...p.bench].flatMap(b=>[...new Set([...b.cards,...b.tools,...b.energies.cards])])],cards=new Set(all);
    assert(all.length===60&&cards.size===60,'exactly sixty physical cards in distinct zones per player');const counts=new Map<string,number>();for(const c of cards)counts.set(printedId(c),(counts.get(printedId(c))??0)+1);
    assert(getDeck(e.decks[pIndex]).cards.every(c=>counts.get(c.cardId)===c.count),'manifest copy conservation');
  }
}
function buildBase(family:Family,variation:number,chanceSeed?:number):Environment {
  const decks:[string,string]=[family==='information-order'||family==='hammer-target'?'dragapult':'crustle',family==='hammer-target'&&variation===2?'raging-bolt':'dragapult'];
  const seed=17+variation,state=new State(),rng=new SeededRandom(seed);state.phase=GamePhase.PLAYER_TURN;state.turn=8;state.activePlayer=0;
  let cardId=0;state.players=decks.map((id,index)=>{const p=new Player();p.id=index+1;p.name=`Player ${index+1}`;p.canEvolve=true;p.deck.cards=instantiateDeck(id);for(const c of p.deck.cards){c.id=cardId++;state.cardNames[c.id]=c.fullName;}p.deck.cards=rng.shuffle(60).map(i=>p.deck.cards[i]);p.bench=Array.from({length:5},()=>new PokemonCardList());p.active.isSecret=false;return p;});
  const[a,b]=state.players;
  const take=(p:Player,id:string)=>{const index=p.deck.cards.findIndex(c=>printedId(c)===id);if(index<0)throw new Error('Fixture exceeds manifest copies for '+id);return p.deck.cards.splice(index,1)[0];};
  const hand=(p:Player,...ids:string[])=>p.hand.cards.push(...ids.map(id=>take(p,id)));
  const stack=(p:Player,target:PokemonCardList,ids:string[],energies:string[]=[])=>{target.cards=ids.map(id=>take(p,id));target.isSecret=false;target.pokemonPlayedTurn=2;const attached=energies.map(id=>take(p,id));target.cards.push(...attached);target.energies.cards=attached;};
  if(family==='delay-prize'||family==='energy-function'){
    const energy=family==='delay-prize'?['MEE-1','TEF-161','JTG-159']:['MEE-1','JTG-159',...(variation===3?['TEF-161']:[])];
    stack(a,a.active,['DRI-11','DRI-12'],energy);stack(a,a.bench[0],['MEG-104']);
    hand(a,'TEF-161','JTG-159','POR-86');
    if(family==='delay-prize'){
      stack(b,b.active,['TWM-128','TWM-129','TWM-130'],['MEE-2','MEE-5']);b.active.damage=200;
      stack(b,b.bench[0],variation===2?['TWM-128','TWM-129','TWM-130']:['TWM-128'],variation===2?['MEE-2','MEE-5']:[]);
      if(variation===2){for(const zone of[a.hand,a.deck]){const exhausted=zone.cards.filter(c=>printedId(c)==='TEF-161');zone.cards=zone.cards.filter(c=>printedId(c)!=='TEF-161');a.discard.cards.push(...exhausted);}}
    }else{
      stack(b,b.active,variation===1?['TWM-128','TWM-129','TWM-130']:['TWM-128','TWM-129'],['MEE-2','MEE-5']);stack(b,b.bench[0],['TWM-128']);
      if(variation>1){a.active.damage=80;b.active.damage=variation===2?70:0;}
    }
  }else if(family==='information-order'){
    stack(a,a.active,['TWM-128','TWM-129']);stack(a,a.bench[0],['TWM-128']);hand(a,'POR-81');
    if(variation===1)stack(a,a.bench[1],['TWM-128']);
    if(variation===2){while(a.deck.cards.filter(c=>printedId(c)==='TWM-129').length>1)a.discard.cards.push(take(a,'TWM-129'));}
    if(variation===3)stack(a,a.bench[1],['JTG-120','TEF-129']);
    stack(b,b.active,['TWM-128','TWM-129','TWM-130'],['MEE-2','MEE-5']);stack(b,b.bench[0],['TWM-128']);
  }else{
    stack(a,a.active,['TWM-128','TWM-129','TWM-130'],['MEE-2','MEE-5']);stack(a,a.bench[0],['TWM-128']);hand(a,'POR-71','MEG-114','MEG-119');
    if(variation===2){stack(b,b.active,['TEF-123'],['MEE-4','MEE-6']);b.active.damage=150;stack(b,b.bench[0],['TWM-25'],['MEE-1']);stack(b,b.bench[1],['SSP-76']);}
    else{stack(b,b.active,['TWM-128','TWM-129','TWM-130'],['MEE-2','MEE-5']);b.active.damage=200;stack(b,b.bench[0],['TWM-128','TWM-129','TWM-130'],['MEE-2','MEE-5']);}
    if(variation===3){a.supporterTurn=state.turn;a.discard.cards.push(take(a,'SCR-133'));}
  }
  for(const[pIndex,p]of state.players.entries()){
    const prizes=family==='delay-prize'&&variation===3&&pIndex===0?2:6;p.prizesTaken=6-prizes;p.prizes=Array.from({length:prizes},()=>{const c=new CardList();c.isSecret=true;c.cards=[p.deck.cards.pop()!];return c;});
    if(pIndex===1)p.hand.cards.push(...p.deck.cards.splice(0,5));
  }
  const e=Environment.fromFixtureState(state,chanceSeed??seed,decks,()=>buildBase(family,variation,chanceSeed));assertConservation(e);return e;
}
export function fixtureEnvironment(family:Family,variation:number):Environment {
  const e=buildBase(family,variation);
  if(family==='hammer-target'){
    // A conditional heads position: find and record a seed whose real card flip
    // reaches its discard prompt. No chance outcome is rewritten after dispatch.
    for(let seed=1;seed<=100;seed++){
      const candidate=buildBase(family,variation,seed);const action=candidate.observe().legalActions.find(a=>a.cardId==='POR-71'&&a.type==='play-trainer')!;candidate.step(action.id);
      if(candidate.observe().prompt?.type==='Choose pokemon')return candidate;
    }
    throw new Error('Could not reconstruct a conditional Hammer heads fixture.');
  }
  return e;
}
function settle(e:Environment){const rng=new SeededRandom(7);for(let i=0;i<100&&e.status==='running'&&e.observe().prompt;i++)e.step(chooseAction(e.observe(),'heuristic',rng).id);assert(!e.observe().prompt||e.status==='finished','bounded fixture prompt resolution');}
const eligible=(family:Family,a:LegalAction)=>family==='delay-prize'?a.type==='attack'||a.type==='pass':family==='energy-function'?a.type==='attach-energy'&&a.targetRef?.zone==='active':family==='information-order'?a.type==='ability'||a.cardId==='POR-81':a.type==='prompt'&&!!a.targetRef;
export function tacticalFixture(params:{fixtureId:string;variationId:string|number}){
  const family=params.fixtureId.replace(/^(crustle|dragapult)-/,'') as Family,variation=Number(String(params.variationId).split('-').at(-1));
  if(!FIXTURE_FAMILIES.includes(family)||!Number.isInteger(variation)||variation<1||variation>3)throw new Error('Unknown tactical fixture/variation.');
  const fixtureId=`${family==='delay-prize'||family==='energy-function'?'crustle':'dragapult'}-${family}`,variationId=`${fixtureId}-${variation}`;
  const e=fixtureEnvironment(family,variation),observation=e.observe(),validatedActionIds:string[]=[],transitionEvidence:any[]=[];
  for(const action of observation.legalActions.filter(a=>eligible(family,a))){
    // Materialization repeats the recipe; it never transplants pending closures.
    const trial=fixtureEnvironment(family,variation),before=trial.store.state.players[0],handCount=before.hand.cards.length;
    const oldEnergy=before.active.energies.cards.length,oldPrizes=before.prizes.filter(p=>p.cards.length).length;
    const target=action.targetRef?.zone==='active'?trial.store.state.players[action.targetRef.playerId].active:action.targetRef?.zone==='bench'?trial.store.state.players[action.targetRef.playerId].bench[action.targetRef.index!]:undefined;
    const opposingDiscard=trial.store.state.players[1].discard.cards.length,targetEnergy=target?.energies.cards.length;
    const hpBefore=new CheckHpEffect(before,before.active);trial.store.reduceEffect(trial.store.state,hpBefore);const startingHp=hpBefore.hp;
    trial.step(action.id);settle(trial);assertConservation(trial);
    if(family==='energy-function'){
      const owner=trial.store.state.players[0];assert(owner.hand.cards.length===handCount-1,'attachment moves one hand card');assert(owner.active.energies.cards.length===oldEnergy+1,'attachment reaches selected Active');
      const hpAfter=new CheckHpEffect(owner,owner.active);trial.store.reduceEffect(trial.store.state,hpAfter);assert(hpAfter.hp===startingHp+(action.cardId==='POR-86'?20:0),`Growing Grass HP transition: ${startingHp} -> ${hpAfter.hp} after ${action.cardId}`);
    }
    if(family==='delay-prize'&&action.type==='attack')assert(trial.store.state.players[0].prizes.filter(p=>p.cards.length).length===Math.max(0,oldPrizes-2),'Crustle attack takes two Prizes from damaged Dragapult');
    if(family==='hammer-target'){assert(trial.store.state.players[1].discard.cards.length===opposingDiscard+1,'one opposing energy reaches discard');assert(target!.energies.cards.length===targetEnergy!-1,'energy removed from chosen target');}
    validatedActionIds.push(action.id);transitionEvidence.push({actionId:action.id,status:trial.status,decisionCount:trial.decisionIndex-e.decisionIndex});
  }
  assert(validatedActionIds.length>0,'at least one audited root');
  if(family==='hammer-target'&&variation===3)assert(!buildBase(family,variation).observe().legalActions.some(a=>a.type==='play-trainer'&&['MEG-114','MEG-119'].includes(a.cardId??'')),'second Supporter unavailable');
  const fixtureHash=createHash('sha256').update(JSON.stringify({recipeVersion:1,family,variation,engineVersion:ENGINE_VERSION,decks:e.decks.map(id=>getDeck(id).listHash),observation})).digest('hex');
  return {fixtureId,variationId,observation,engineVersion:ENGINE_VERSION,fixtureHash,mechanicsAudit:{status:'verified',validatedActionIds,tests:['tactical-fixtures.test.ts','runtime recipe transition assertions'],scope:['exact registered-card conservation','legal roots and prompt continuations','selected one-step resource/Prize transitions'],limitations:['Constructed tactical analogue, not a historical game replay.','No strategic answer or game-win probability is certified.','Only listed root actions are admitted; downstream policies and complete decks remain experimental.',...(family==='hammer-target'?['Conditional heads position; does not establish the value of playing Hammer before the flip.']:[])]},decisionBindings:Object.fromEntries(observation.legalActions.filter(a=>validatedActionIds.includes(a.id)).map(a=>[a.id,{type:a.type,cardId:a.cardId,sourceRef:a.sourceRef,targetRef:a.targetRef}])),transitionEvidence};
}
