import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';

// Uses a real, isolated API data directory. No responses or game states are mocked.
// Start the API and Vite first; PTCG_BROWSER_URL may select a dedicated test port.
const base = process.env.PTCG_BROWSER_URL || 'http://127.0.0.1:5173';
const browser = await chromium.launch({headless:true});
try {
  const page = await browser.newPage({viewport:{width:1440,height:900},reducedMotion:'reduce'});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  // Only shorten the real engine's compute budget for this integration test.
  await page.route('**/api/matches',async route=>{
    if(route.request().method()==='POST') {
      await route.continue({postData:JSON.stringify({...route.request().postDataJSON(),budgetMs:25})});
    } else await route.continue();
  });
  await page.goto(`${base}/#play`);
  await page.getByRole('heading',{name:'Play a best-of-three'}).waitFor();
  await page.getByLabel('Your deck',{exact:true}).locator('option').first().waitFor();
  const created = page.waitForResponse(r=>r.url().endsWith('/api/matches')&&r.request().method()==='POST');
  await page.getByRole('button',{name:'Start match',exact:true}).click();
  const match=await (await created).json();
  assert.ok(match.id);
  assert.equal(match.engineTurnBudgetMs,25);
  const url=path=>`${base}/api${path}`;
  async function idleHuman(){
    for(let attempt=0;attempt<120;attempt++){
      const state=await (await page.request.get(url(`/matches/${match.id}`))).json();
      assert.notEqual(state.status,'paused',state.error);
      if(!state.thinking&&state.observation.decisionPlayer===0)return state;
      await page.waitForTimeout(250);
    }
    throw new Error('The engine did not return a human choice');
  }
  await idleHuman();
  await page.locator('.legal-moves button').first().waitFor();
  const move=page.waitForResponse(r=>r.url().endsWith(`/matches/${match.id}/actions`)&&r.request().method()==='POST');
  await page.locator('.legal-moves button').first().click();
  assert.equal((await move).status(),200);
  await idleHuman();
  await page.getByRole('button',{name:'Pause',exact:true}).waitFor({state:'visible'});
  const bookmarked=page.waitForResponse(r=>r.url().endsWith('/bookmarks'));
  await page.getByRole('button',{name:'Bookmark decision',exact:true}).click();
  const bookmark=await (await bookmarked).json();
  const pausedResponse=page.waitForResponse(r=>r.url().endsWith('/control/pause'));
  await page.getByRole('button',{name:'Pause',exact:true}).click();
  const paused=await (await pausedResponse).json();
  assert.equal(paused.status,'paused');
  assert.equal(paused.observation.players[1].hand.length,0);
  assert.equal(paused.decks,undefined);
  assert.equal(paused.seed,undefined);
  assert.equal(paused.actions,undefined);
  await page.reload();
  await page.getByRole('button',{name:'Resume',exact:true}).waitFor();
  const resumedResponse=page.waitForResponse(r=>r.url().endsWith('/control/resume'));
  await page.getByRole('button',{name:'Resume',exact:true}).click();
  const resumed=await (await resumedResponse).json();
  assert.deepEqual(resumed.observation,paused.observation);
  await idleHuman();
  const visibleCard=page.getByRole('button',{name:/^Inspect /}).first();
  if(await visibleCard.count()){
    await visibleCard.click();
    await page.getByRole('region',{name:'Card inspector',exact:true}).waitFor();
    await page.getByRole('button',{name:'Close card inspector'}).click();
  }
  await mkdir('artifacts/browser',{recursive:true});
  await page.screenshot({path:'artifacts/browser/play-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:'artifacts/browser/play-mobile.png',fullPage:true});
  await page.setViewportSize({width:1440,height:900});
  for(let game=0;game<2;game++){
    await idleHuman();
    await page.locator('.concede-control summary').click();
    const concession=page.waitForResponse(r=>r.url().endsWith('/control/concede'));
    await page.getByRole('button',{name:'Concede this game',exact:true}).click();
    const state=await (await concession).json();
    assert.deepEqual(state.score,[0,game+1]);
    if(!game){
      assert.equal(state.status,'between-games');
      await page.getByLabel('Starting player',{exact:true}).selectOption('0');
      await page.getByRole('button',{name:'Start next game',exact:true}).click();
    }else assert.equal(state.status,'completed');
  }
  const publication=page.waitForResponse(r=>r.url().endsWith(`/matches/${match.id}/replays`));
  await page.getByRole('button',{name:'Make replays available',exact:true}).click();
  const published=await (await publication).json();
  assert.equal(published.replayIds.length,2);
  await page.getByRole('button',{name:'Teaching review',exact:true}).click();
  await page.locator('.review-item').filter({hasText:bookmark.title}).click();
  await page.getByRole('region',{name:'Pokémon card table',exact:true}).waitFor();
  await page.getByRole('checkbox').first().check();
  await page.getByLabel('Conditional reasoning',{exact:true}).fill('Smoke-test annotation only; this human match remains reserved for evaluation.');
  const reviewedResponse=page.waitForResponse(r=>r.url().endsWith(`/teaching/${bookmark.id}/review`));
  await page.getByRole('button',{name:'Save reviewed annotation',exact:true}).click();
  const reviewed=await (await reviewedResponse).json();
  assert.equal(reviewed.reviewStatus,'reviewed');
  assert.equal(reviewed.trainingEligible,false);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,matchId:match.id,bookmarkId:bookmark.id,replays:published.replayIds,privateView:true,mobileOverflow:false,pageErrors:errors}));
} finally {await browser.close();}
