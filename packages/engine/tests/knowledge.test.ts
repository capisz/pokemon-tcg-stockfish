import test from 'node:test';
import assert from 'node:assert/strict';
import { Environment } from '../src/environment';
import { fixtureEnvironment,assertConservation } from '../src/tactical-fixtures';
import { printedId } from '../src/catalog';
import { chooseAction } from '../src/policies';
import { SeededRandom } from '../src/random';
function hand(e:Environment,player:number,id:string){const p=e.store.state.players[player],i=p.deck.cards.findIndex(c=>printedId(c)===id);assert.ok(i>=0,id);p.hand.cards.push(p.deck.cards.splice(i,1)[0]);(e as any).candidates=null;}
function settle(e:Environment,preferred?:string){const rng=new SeededRandom(8);for(let i=0;i<100&&e.observe().prompt;i++){const o=e.observe(),action=o.legalActions.find(a=>a.choiceOperation==='append'&&a.cardId===preferred)??chooseAction(o,'heuristic',rng);e.step(action.id);}assert.ok(!e.observe().prompt);}
function counts(ids:string[]){const out=new Map<string,number>();for(const id of ids)out.set(id,(out.get(id)??0)+1);return out;}
test('actual Pokegear peek survives its shuffle as deck membership, without retaining unknown order',()=>{
  const e=fixtureEnvironment('delay-prize',1);hand(e,0,'BLK-84');e.step(e.observe().legalActions.find(a=>a.cardId==='BLK-84'&&a.type==='play-trainer')!.id);assert.equal(e.observe().prompt?.cards?.length,7);settle(e);
  const p=e.observe().searchPosition!;assert.ok(p);assert.ok(p.ownDeckKnown!.length>=6);assert.deepEqual(p.ownDeckTop,[]);assert.deepEqual(p.ownDeckBottom,[]);assertConservation(e);
  for(const seed of[1,2,42,731]){const sample=Environment.fromPublicPosition(p,seed,'dragapult'),deck=counts(sample.store.state.players[0].deck.cards.map(printedId));for(const[id,n]of counts(p.ownDeckKnown!))assert.ok((deck.get(id)??0)>=n);assertConservation(sample);}
});
test('Ultra Ball reveal preserves surviving actor-deck identities as known non-Prizes',()=>{
  const e=fixtureEnvironment('delay-prize',1);hand(e,0,'MEG-131');
  e.step(e.observe().legalActions.find(a=>a.cardId==='MEG-131'&&a.type==='play-trainer')!.id);settle(e);
  const observation=e.observe(),position=observation.searchPosition!;assert.ok(position);assertConservation(e);
  const reveal=(observation.knowledge??[]).filter((event:any)=>event.type==='temporary-zone-reveal').at(-1);
  assert.ok(reveal);const deck=e.store.state.players[0].deck.cards.map(printedId),known=counts(position.ownDeckKnown??[]);
  for(const[id,n]of counts((reveal.cards??[]).map((card:any)=>card.id)))
    assert.ok((known.get(id)??0)>=Math.min(n,deck.filter(cardId=>cardId===id).length),`${id} must not be sampled into Prizes`);
  for(const seed of[17,83,411]){const sampled=Environment.fromPublicPosition(position,seed,'dragapult'),prizes=sampled.store.state.players[0].prizes.flatMap(prize=>prize.cards.map(printedId));
    for(const[id,n]of known)assert.ok((sampled.store.state.players[0].deck.cards.map(printedId).filter(cardId=>cardId===id).length)>=n,`${id} remains in sampled deck`);
    assertConservation(sampled);assert.ok(prizes.length>0);}
});
test('Crispin reveal reconstructs opponent-known hand identities after its prompts resolve',()=>{
  const e=fixtureEnvironment('information-order',1),viewer=e.actor;hand(e,viewer,'SCR-133');
  e.step(e.observe().legalActions.find(a=>a.cardId==='SCR-133'&&a.type==='play-trainer')!.id);settle(e);
  const revealView=e.observe(1-viewer),revealed=(revealView.knowledge??[]).filter((event:any)=>event.type==='revealed-cards').at(-1);
  assert.ok(revealed);const opponentHandIds=e.store.state.players[viewer].hand.cards.map(printedId);
  const knownHand=(revealed.cards??[]).map((card:any)=>card.id).filter((id:string)=>opponentHandIds.includes(id));
  const endTurn=e.observe().legalActions.find(action=>action.type==='pass');assert.ok(endTurn);e.step(endTurn.id);settle(e);
  const observation=e.observe(),position=observation.searchPosition!;assert.ok(position);assertConservation(e);
  for(const id of knownHand)assert.ok(position.knownOpponentHand?.includes(id),`${id} remains known in the opponent's hand`);
  for(const seed of[29,311,907]){const sampled=Environment.fromPublicPosition(position,seed,'dragapult');assertConservation(sampled);
    const sampledHand=sampled.store.state.players[viewer].hand.cards.map(printedId);
    for(const id of position.knownOpponentHand??[])assert.ok(sampledHand.includes(id),`${id} must remain in the sampled known hand`);}
});
test('actual Eri preserves the surviving revealed hand after Item discards',()=>{
  const e=fixtureEnvironment('delay-prize',1);hand(e,0,'TEF-146');e.step(e.observe().legalActions.find(a=>a.cardId==='TEF-146'&&a.type==='play-trainer')!.id);settle(e);
  const p=e.observe().searchPosition!;assert.ok(p);assert.deepEqual(p.knownOpponentHand?.sort(),e.store.state.players[1].hand.cards.map(printedId).sort());assertConservation(e);
  assert.deepEqual(Environment.fromPublicPosition(p,42,'dragapult').store.state.players[1].hand.cards.map(printedId).sort(),p.knownOpponentHand);
});
test('separate revealed searches accumulate same-name copies without seeing hidden hand identities',()=>{
  const e=fixtureEnvironment('information-order',1);hand(e,0,'POR-81');
  assert.ok(e.store.state.players[0].deck.cards.filter(c=>printedId(c)==='TWM-129').length>=2);
  for(let i=0;i<2;i++){e.step(e.observe().legalActions.find(a=>a.cardId==='POR-81'&&a.type==='play-trainer')!.id);settle(e,'TWM-129');}
  e.step(e.observe().legalActions.find(a=>a.type==='pass')!.id);settle(e);const p=e.observe().searchPosition!;assert.ok(p);assert.equal(p.observer,1);assert.equal(p.knownOpponentHand!.filter(id=>id==='TWM-129').length,2);assertConservation(e);
});
test('actual Ciphermaniac ordering survives shuffle and constrains independently reconstructed decks',()=>{
  const e=fixtureEnvironment('hammer-target',2);settle(e);e.store.state.activePlayer=1;(e as any).candidates=null;hand(e,1,'TEF-145');
  e.step(e.observe().legalActions.find(a=>a.cardId==='TEF-145'&&a.type==='play-trainer')!.id);settle(e);const p=e.observe().searchPosition!;assert.ok(p);assert.equal(p.ownDeckTop!.length,2);assert.equal(p.ownPrizeCards!.length,6);assertConservation(e);
  for(const seed of[4,81]){const sample=Environment.fromPublicPosition(p,seed,'dragapult');assert.deepEqual(sample.store.state.players[1].deck.cards.slice(0,2).map(printedId),p.ownDeckTop);assertConservation(sample);}
});
test('Judge preserves known former hand cards as deck membership instead of sampling them into Prizes',()=>{
  const e=fixtureEnvironment('information-order',1);hand(e,0,'POR-76');hand(e,0,'TWM-129');hand(e,0,'MEG-131');
  e.step(e.observe().legalActions.find(a=>a.cardId==='POR-76'&&a.type==='play-trainer')!.id);settle(e);const p=e.observe().searchPosition!;assert.ok(p);assert.ok(p.ownDeckKnown!.length>0);
  for(const seed of[3,8,31]){const sample=Environment.fromPublicPosition(p,seed,'dragapult'),deck=counts(sample.store.state.players[0].deck.cards.map(printedId));for(const[id,n]of counts(p.ownDeckKnown!))assert.ok((deck.get(id)??0)>=n);assertConservation(sample);}
});
test('Special Red Card refuses search when an unmodeled known bottom segment would otherwise be forgotten',()=>{
  const e=fixtureEnvironment('delay-prize',1);hand(e,0,'TEF-146');hand(e,0,'CRI-82');const other=e.store.state.players[1];for(const prize of other.prizes.splice(3))other.hand.cards.push(...prize.cards);
  e.step(e.observe().legalActions.find(a=>a.cardId==='TEF-146'&&a.type==='play-trainer')!.id);settle(e);
  e.step(e.observe().legalActions.find(a=>a.cardId==='CRI-82'&&a.type==='play-trainer')!.id);settle(e);assert.equal(e.observe().searchPosition,undefined);assert.match(e.observe().searchUnavailableReason!,/history/);assertConservation(e);
});
test('drawing indistinguishable copies cannot reveal which remembered physical copy left the randomized deck',()=>{
  const afterDraw=(drawRemembered:boolean)=>{
    const e=fixtureEnvironment('delay-prize',1);hand(e,0,'BLK-84');const p=e.store.state.players[0];
    // Construct the same private Prize allocation in both fixtures, ensuring two
    // indistinguishable deck copies. Neither allocation is supplied to the agent.
    for(const prize of p.prizes)if(printedId(prize.cards[0])==='JTG-159'){const index=p.deck.cards.findIndex(c=>printedId(c)!=='JTG-159');[prize.cards[0],p.deck.cards[index]]=[p.deck.cards[index],prize.cards[0]];}
    const copies=p.deck.cards.filter(c=>printedId(c)==='JTG-159');assert.equal(copies.length,2);
    const peek=[copies[0],...p.deck.cards.filter(c=>printedId(c)!=='JTG-159').slice(0,6)];p.deck.cards=[...peek,...p.deck.cards.filter(c=>!peek.includes(c))];
    e.step(e.observe().legalActions.find(a=>a.cardId==='BLK-84'&&a.type==='play-trainer')!.id);settle(e);
    const choice=copies[drawRemembered?0:1],index=p.deck.cards.indexOf(choice);assert.ok(index>=0);p.hand.cards.push(p.deck.cards.splice(index,1)[0]);(e as any).candidates=null;
    assertConservation(e);return e.observe();
  };
  const one=afterDraw(true),two=afterDraw(false);assert.deepEqual(one,two);assert.equal(one.searchPosition!.ownDeckKnown!.filter(id=>id==='JTG-159').length,0);
});
