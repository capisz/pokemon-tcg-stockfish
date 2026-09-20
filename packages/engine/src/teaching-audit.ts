import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { CheckHpEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/check-effects';
import { ENGINE_VERSION, Environment } from './environment';
import { printedId } from './catalog';
import { fixtureEnvironment, assertConservation, tacticalFixture } from './tactical-fixtures';
import type { LegalAction, Observation } from './types';

const check=(condition:unknown,message:string)=>{if(!condition)throw new Error('Teaching audit: '+message);};
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
function stateSignature(e:Environment) {
  const s=e.store.state,ids=(cards:any[])=>cards.map(c=>({id:c.id,printedId:printedId(c)}));
  return {turn:s.turn,phase:s.phase,activePlayer:s.activePlayer,players:s.players.map(p=>({
    hand:ids(p.hand.cards),deck:ids(p.deck.cards),discard:ids(p.discard.cards),lostzone:ids(p.lostzone.cards),
    prizes:p.prizes.map(pr=>ids(pr.cards)),stadium:ids(p.stadium.cards),supporter:ids(p.supporter.cards),
    energyPlayedTurn:p.energyPlayedTurn,board:[p.active,...p.bench].map(b=>({cards:ids(b.cards),energies:ids(b.energies.cards),tools:ids(b.tools),damage:b.damage,specialConditions:b.specialConditions}))})),
    observations:[e.observe(0),e.observe(1)]};
}

/** A mechanical extension of an existing review, never a new strategic label. */
export function auditTeachingFixture(params:{fixtureId:string;variationId:string;observation:Observation;acceptedActionIds:string[];sourceEngineVersion:string;sourceFixtureHash:string}) {
  check(params.fixtureId==='crustle-delay-prize'&&params.variationId==='crustle-delay-prize-2','only the scoped delay-prize-2 attachment audit is supported');
  check(typeof params.sourceEngineVersion==='string'&&params.sourceEngineVersion.length>0&&typeof params.sourceFixtureHash==='string'&&params.sourceFixtureHash.length>0,'source provenance is required');
  const fixture=tacticalFixture({fixtureId:params.fixtureId,variationId:params.variationId});
  // Compare the complete JSON-lines wire representation. Optional undefined
  // properties are absent in saved JSON; no observable field is excluded.
  const wire=(value:unknown)=>JSON.parse(JSON.stringify(value));
  check(isDeepStrictEqual(wire(params.observation),wire(fixture.observation)),'saved observation or action bindings differ from the current recipe; strategic review cannot be carried over');
  const accepted=params.acceptedActionIds;
  check(Array.isArray(accepted)&&accepted.length>0&&new Set(accepted).size===accepted.length,'nonempty distinct accepted actions are required');
  const actions=accepted.map(id=>fixture.observation.legalActions.find(a=>a.id===id));
  check(actions.every(a=>a?.type==='attach-energy'&&a.sourceRef?.zone==='hand'&&a.sourceRef.playerId===0&&a.targetRef?.playerId===0&&
    (a.cardId==='POR-86'&&(a.targetRef.zone==='active'||a.targetRef.zone==='bench'&&a.targetRef.index===0)||a.cardId==='JTG-159'&&a.targetRef.zone==='bench'&&a.targetRef.index===0)),
  'accepted action is outside the three scoped attachments');
  const transitions=actions.map(raw=>{
    const action=raw as LegalAction,e=fixtureEnvironment('delay-prize',2),branch=e.branch(),s=e.store.state,p=s.players[0];
    const target=action.targetRef!.zone==='active'?p.active:p.bench[action.targetRef!.index!];
    const source=p.hand.cards[action.sourceRef!.index!];
    check(source&&printedId(source)===action.cardId,'source binding identifies the exact hand card');
    const before=stateSignature(e),handBefore=[...p.hand.cards],energyBefore=[...target.energies.cards],targetCards=[...target.cards];
    const hpBefore=new CheckHpEffect(p,target);e.store.reduceEffect(s,hpBefore);const startingHp=hpBefore.hp;
    e.step(action.id);branch.step(action.id);
    check(!e.observe().prompt,'attachment must finish without an unresolved prompt');
    assertConservation(e);assertConservation(branch);
    const hpAfter=new CheckHpEffect(p,target);e.store.reduceEffect(s,hpAfter);
    const expectedGain=action.cardId==='POR-86'&&action.targetRef!.zone==='active'?20:0;
    check(hpAfter.hp===startingHp+expectedGain,`Growing Grass HP applies only to the Grass-type Crustle, not Colorless Kangaskhan (${action.id}: ${startingHp} -> ${hpAfter.hp}, expected +${expectedGain})`);
    check(isDeepStrictEqual(p.hand.cards,handBefore.filter(c=>c!==source)),'exactly the selected hand card moves');
    check(isDeepStrictEqual(target.energies.cards,[...energyBefore,source])&&isDeepStrictEqual(target.cards,[...targetCards,source]),'the selected Energy reaches only its bound Pokemon');
    check(p.energyPlayedTurn===s.turn&&!e.observe().legalActions.some(a=>a.type==='attach-energy'),'manual Energy attachment is once per turn');
    const after=stateSignature(e),expected=structuredClone(before);
    expected.players[0].hand=after.players[0].hand;expected.players[0].energyPlayedTurn=s.turn;
    const slot=action.targetRef!.zone==='active'?0:action.targetRef!.index!+1;
    expected.players[0].board[slot].cards=after.players[0].board[slot].cards;
    expected.players[0].board[slot].energies=after.players[0].board[slot].energies;
    // Legal actions and visible attachments change. All private zones, all damage,
    // conditions, Prizes, turn and the other Pokemon must otherwise stay identical.
    expected.observations=after.observations;
    check(isDeepStrictEqual(after,expected),'attachment changed unrelated state or caused immediate Spiky damage');
    check(isDeepStrictEqual(after,stateSignature(branch)),'independent deterministic branch differs');
    return {actionId:action.id,binding:{type:action.type,cardId:action.cardId,sourceRef:action.sourceRef,targetRef:action.targetRef},
      hpBefore:startingHp,hpAfter:hpAfter.hp,expectedHpGain:expectedGain,transitionHash:hash(after),deterministic:true,conserved:true,oncePerTurn:true,noImmediateDamage:true};
  });
  return {schemaVersion:1,status:'verified',fixtureId:params.fixtureId,variationId:params.variationId,
    sourceEngineVersion:params.sourceEngineVersion,sourceFixtureHash:params.sourceFixtureHash,engineVersion:ENGINE_VERSION,fixtureHash:fixture.fixtureHash,
    compatibility:{mode:'exact-observation-and-action-bindings-v1',observationEqual:true,actionBindingsEqual:true},
    validatedActionIds:accepted,transitions,tests:['teaching-audit.test.ts','runtime attachment transition assertions'],
    scope:['selected hand card and target','Grass-only HP modifier','no immediate retaliation damage','once-per-turn Energy attachment','card conservation','independent deterministic reconstruction'],
    limitations:['Only these three attachment roots in this exact fixture are covered.','No strategic preference, downstream game result, or whole-deck eligibility is certified.']};
}
