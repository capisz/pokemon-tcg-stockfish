import {chromium,expect} from '@playwright/test';
import assert from 'node:assert/strict';

// Explicit UI-only fixtures exercise structured references, never invented simulator results.
const base=process.env.PTCG_BROWSER_URL||'http://127.0.0.1:5173';
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900},reducedMotion:'reduce'});
 await page.goto(base);
 await page.evaluate(async()=>{
  const React=(await import('/node_modules/.vite/deps/react.js')).default;
  const ReactDOM=(await import('/node_modules/.vite/deps/react-dom_client.js')).default;
  const {CardTable}=await import('/src/Board.tsx');
  const host=document.createElement('div');host.id='board-binding-fixture';Object.assign(host.style,{position:'fixed',inset:'0',padding:'16px',background:'white',overflow:'auto',zIndex:'1000'});document.body.append(host);
  const card=(id,name,kind='pokemon')=>({id,name,kind,hp:kind==='pokemon'?150:undefined});
  const pokemon=(index)=>({card:card(`fixture-${index}`,`Visible Pokémon ${index}`),damage:index===3?40:0,energy:['Fighting Energy'],tools:[],conditions:[],slotIndex:index,attachments:[card('energy-1','Fighting Energy','energy')]});
  const observation={schemaVersion:1,playerId:0,decisionPlayer:0,turn:4,phase:'PLAYER_TURN',status:'running',ownDeck:[],legalActions:[],history:[],warnings:[],players:[{id:0,name:'You',active:pokemon(0),bench:[pokemon(1),pokemon(3)],hand:[card('hand-1','Selected card')],handCount:1,deckCount:35,prizesRemaining:4,discard:[]},{id:1,name:'Opponent',active:pokemon(9),bench:Array.from({length:8},(_,i)=>pokemon(i)),hand:[],handCount:6,deckCount:29,prizesRemaining:3,discard:[]}]};
  window.bindingClicks=[];
  ReactDOM.createRoot(host).render(React.createElement(CardTable,{observation,inspect:c=>window.bindingClicks.push({inspect:c.id}),select:s=>window.bindingClicks.push(s.ref),selectTarget:r=>window.bindingClicks.push(r),selected:{playerId:0,zone:'hand',index:0},targets:[{playerId:0,zone:'bench',index:3},{playerId:0,zone:'bench',index:2}]}));
 });
 const fixture=page.locator('#board-binding-fixture');
 await expect(fixture.locator('.selected-card')).toHaveCount(1);
 await expect(fixture.locator('.their-side .table-bench .in-play')).toHaveCount(8);
 await expect(fixture.getByLabel('6 hidden cards')).toBeVisible();
 await fixture.locator('.our-side .table-bench .legal-target').click();
 await fixture.getByRole('button',{name:'Choose empty Bench slot 3'}).click();
 await fixture.getByRole('button',{name:'Inspect attached Fighting Energy'}).first().click();
 const clicks=await page.evaluate(()=>window.bindingClicks);
 assert.deepEqual(clicks[0],{playerId:0,zone:'bench',index:3},'Sparse bench uses simulator slotIndex, not rendered array index');
 assert.deepEqual(clicks[1],{playerId:0,zone:'bench',index:2});
 assert.deepEqual(clicks[2],{inspect:'energy-1'});
 assert.equal(await fixture.locator('.table-card').first().evaluate(el=>getComputedStyle(el).animationName),'none');
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 console.log(JSON.stringify({passed:true,scope:'explicit-ui-fixtures',sparseBench:true,expandedBench:true,emptyBenchTarget:true,inspectAttachments:true,hiddenHand:true,reducedMotion:true,mobileOverflow:false}));
}finally{await browser.close();}
