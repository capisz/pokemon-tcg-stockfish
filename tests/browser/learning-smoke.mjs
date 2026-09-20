import {chromium,expect} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';

// Real bounded UI run. Explicit opt-in and an isolated backend are mandatory.
if(process.env.PTCG_LEARNING_SMOKE!=='1'||!process.env.PTCG_BROWSER_URL)throw new Error('Set PTCG_LEARNING_SMOKE=1 and PTCG_BROWSER_URL for an isolated test backend.');
const base=process.env.PTCG_BROWSER_URL;
const browser=await chromium.launch({headless:true});
let page,runId;
try{
 page=await browser.newPage({viewport:{width:1440,height:900},reducedMotion:'reduce'});
 const errors=[],controls=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',request=>{if(request.method()==='POST'&&request.url().includes('/control/'))controls.push(request.url());});
 const initial=await (await page.request.get(`${base}/api/learning/runs`)).json();
 assert.ok(initial.runs.every(r=>['stopped','failed'].includes(r.status)),'Isolated smoke refuses to disturb an existing active run');
 await page.goto(`${base}/#replays`);
 await page.getByRole('button',{name:'Run settings',exact:true}).click();
 await page.getByRole('checkbox',{name:'Keep this computer awake during learning',exact:true}).uncheck();
 const started=page.waitForResponse(r=>r.url().endsWith('/api/learning/runs')&&r.request().method()==='POST');
 await page.getByRole('button',{name:'Start learning',exact:true}).click();
 const created=await (await started).json();runId=created.id;assert.ok(runId);
 await page.getByRole('region',{name:'Pokémon card table',exact:true}).waitFor({timeout:90000});
 await expect(page.getByRole('button',{name:/Watch worker 2/})).toBeEnabled({timeout:90000});
 await page.getByRole('button',{name:/Watch worker 2/}).click();
 await page.getByRole('button',{name:'Pause playback',exact:true}).click();
 const index=await page.getByRole('slider',{name:'Replay position'}).inputValue();
 const before=controls.length;
 await page.waitForTimeout(1700);
 assert.equal(controls.length,before,'Presentation pause must not pause learning');
 assert.equal(await page.getByRole('slider',{name:'Replay position'}).inputValue(),index);
 const state=await (await page.request.get(`${base}/api/learning/runs/${runId}`)).json();
 assert.ok(['running','waiting-for-play'].includes(state.status),state.error??state.status);
 const journal=await (await page.request.get(`${base}/api/learning/runs/${runId}/games`)).json();
 assert.ok(journal.games.length>0);
 const game=journal.games.find(g=>g.frameCursor>=0);assert.ok(game);
 for(const playerId of [0,1]){
  const feed=await (await page.request.get(`${base}/api/learning/games/${game.id}/frames?playerId=${playerId}&after=-1&limit=100`)).json();
  assert.ok(feed.frames.length>0);
  assert.ok(feed.frames.every(frame=>frame.observations===undefined&&frame.observation.playerId===playerId&&frame.observation.players[1-playerId].hand.length===0));
 }
 await page.getByRole('button',{name:'Pause learning',exact:true}).click();
 await expect(page.getByRole('button',{name:'Resume learning',exact:true})).toBeVisible({timeout:90000});
 await page.reload();
 await expect(page.getByRole('button',{name:'Resume learning',exact:true})).toBeVisible({timeout:30000});
 await expect(page.getByRole('button',{name:'Play playback',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Resume learning',exact:true}).click();
 await expect(page.getByRole('button',{name:'Pause learning',exact:true})).toBeVisible({timeout:30000});
 await mkdir('artifacts/browser',{recursive:true});
 await page.screenshot({path:'artifacts/browser/learning-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:'artifacts/browser/learning-mobile.png',fullPage:true});
 await page.getByRole('button',{name:'Stop learning',exact:true}).click();
 await expect(page.getByRole('button',{name:'Start learning',exact:true})).toBeVisible({timeout:90000});
 const stopped=await (await page.request.get(`${base}/api/learning/runs/${runId}`)).json();
 assert.equal(stopped.status,'stopped');
 assert.deepEqual(errors,[]);
 console.log(JSON.stringify({passed:true,runId,manualStart:true,bothWorkers:true,projectedFrames:true,pauseResumeReload:true,stopped:true,completedGames:stopped.metrics.completedGames,pageErrors:errors,mobileOverflow:false}));
}finally{
 // Test runs never continue unintentionally after an assertion or browser failure.
 if(page&&runId){const response=await page.request.get(`${base}/api/learning/runs/${runId}`).catch(()=>null);if(response?.ok()){const current=await response.json();if(!['stopped','failed'].includes(current.status))await page.request.post(`${base}/api/learning/runs/${runId}/control/stop`,{data:{revision:current.revision,requestId:crypto.randomUUID()}}).catch(()=>{});}}
 await browser.close();
}
