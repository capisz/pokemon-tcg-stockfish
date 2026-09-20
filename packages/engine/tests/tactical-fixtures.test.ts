import test from 'node:test';
import assert from 'node:assert/strict';
import { tacticalFixture,fixtureEnvironment,FIXTURE_FAMILIES,assertConservation } from '../src/tactical-fixtures';
test('twelve guide-review analogues materialize legal, conserved, independently reconstructed transitions',()=>{
  for(const family of FIXTURE_FAMILIES)for(const variation of[1,2,3]){
    const fixture=tacticalFixture({fixtureId:family,variationId:variation});
    assert.equal(fixture.mechanicsAudit.status,'verified');assert.ok(fixture.mechanicsAudit.validatedActionIds.length>0);
    assert.ok(fixture.mechanicsAudit.validatedActionIds.every(id=>fixture.observation.legalActions.some(a=>a.id===id)));
    assert.ok(fixture.mechanicsAudit.limitations.some(l=>l.includes('No strategic answer')));
    const e=fixtureEnvironment(family,variation);assertConservation(e);assert.deepEqual(e.branch().observe(),e.observe());
    assert.equal(fixture.observation.players[1].hand.length,0);
  }
});
test('fixtures have immutable content hashes and no unknown deck order in learner observations',()=>{
  const params={fixtureId:'information-order',variationId:'1'},one=tacticalFixture(params),two=tacticalFixture(params);
  assert.equal(one.fixtureHash,two.fixtureHash);assert.deepEqual(one.observation.searchPosition?.ownDeckTop,[]);
  assert.deepEqual(one.observation.searchPosition?.ownDeckBottom,[]);assert.deepEqual(one.observation.searchPosition?.ownDeckKnown,[]);
  assert.notEqual(one.fixtureHash,tacticalFixture({...params,variationId:'2'}).fixtureHash);
  assert.throws(()=>tacticalFixture({fixtureId:'not-registered',variationId:'1'}),/Unknown/);
});
