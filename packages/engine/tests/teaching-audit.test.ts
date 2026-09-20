import test from 'node:test';
import assert from 'node:assert/strict';
import { tacticalFixture } from '../src/tactical-fixtures';
import { auditTeachingFixture } from '../src/teaching-audit';

function input() {
  const fixture=tacticalFixture({fixtureId:'crustle-delay-prize',variationId:2});
  return {fixtureId:fixture.fixtureId,variationId:fixture.variationId,observation:fixture.observation,
    acceptedActionIds:fixture.observation.legalActions.filter(a=>a.type==='attach-energy'&&(a.cardId==='POR-86'||a.cardId==='JTG-159'&&a.targetRef?.zone==='bench')).map(a=>a.id),
    sourceEngineVersion:'original-saved-version',sourceFixtureHash:'original-saved-fixture-hash'};
}
test('the three reviewed attachments pass narrow HP, conservation, energy-turn and deterministic checks',()=>{
  const params=input(),before=structuredClone(params),result=auditTeachingFixture(params);
  assert.deepEqual(params,before);assert.equal(result.status,'verified');assert.equal(result.transitions.length,3);
  assert.deepEqual(result.validatedActionIds,params.acceptedActionIds);
  assert.equal(result.transitions.find(t=>t.binding.cardId==='POR-86'&&t.binding.targetRef?.zone==='active')?.expectedHpGain,20);
  assert.equal(result.transitions.find(t=>t.binding.cardId==='POR-86'&&t.binding.targetRef?.zone==='bench')?.expectedHpGain,0);
  assert.equal(result.transitions.find(t=>t.binding.cardId==='JTG-159')?.noImmediateDamage,true);
  assert.deepEqual(auditTeachingFixture(params),result);
});
test('audit refuses changed information, action bindings, missing roots and unsupported transitions',()=>{
  for(const mutate of[(p:ReturnType<typeof input>)=>p.observation.players[0].active!.damage++,
    (p:ReturnType<typeof input>)=>p.observation.legalActions[0].sourceRef!.index!++,
    (p:ReturnType<typeof input>)=>{p.acceptedActionIds=['missing'];},
    (p:ReturnType<typeof input>)=>{p.acceptedActionIds=[p.observation.legalActions.find(a=>a.type==='attack')!.id];},
    (p:ReturnType<typeof input>)=>{p.variationId='crustle-delay-prize-1';}]){
    const params=input();mutate(params);assert.throws(()=>auditTeachingFixture(params),/Teaching audit:/);
  }
});
