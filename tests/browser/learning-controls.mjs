import {chromium,expect} from '@playwright/test';
import assert from 'node:assert/strict';

// Explicit API fixtures: UI behavior only. This does not start or train a real run.
const base=process.env.PTCG_BROWSER_URL||'http://127.0.0.1:5174';
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900},reducedMotion:'reduce'});
 const pageErrors=[];page.on('pageerror',e=>pageErrors.push(e.message));
 const posts=[],frameRequests=[];let run=null,failedPause=false;
 const card={id:'fixture-1',name:'UI fixture Pokémon',kind:'pokemon',hp:100};
 const pokemon={card,damage:0,energy:[],tools:[],conditions:[]};
 const observation=playerId=>({schemaVersion:1,playerId,decisionPlayer:playerId,turn:1,phase:'PLAYER_TURN',status:'running',players:[0,1].map(id=>({id,name:`Player ${id+1}`,active:pokemon,bench:[],hand:id===playerId?[card]:[],handCount:1,deckCount:40,prizesRemaining:6,discard:[]})),ownDeck:[],legalActions:[],history:[],warnings:[]});
 const game=(id,workerIndex)=>({id,workerIndex,status:'running',purpose:'collection',archetypes:['crustle','dragapult'],decisionIndex:1,turn:1,frameCursor:1,replayAvailable:false});
 let games=[game('g0',0),game('g1',1)];
 const summary=()=>({schemaVersion:1,id:'run-fixture',revision:0,status:'running',phase:'collecting',cycle:1,incumbent:{name:'Fixture experimental policy',version:'fixture-v1'},configuration:{seed:42,keepAwake:false},activeGames:games,metrics:{completedGames:4,errors:0,truncatedGames:1,searchDecisions:12,searchTargets:9,peakMemoryBytes:1024**3,managedBytes:1024**2,loss:0.375,coverage:{scheduledPairs:4,totalPairs:100,completedPairs:3,archetypes:5},comparison:{status:'unavailable',completedGames:0,totalGames:20,mean:null,lower:null,upper:null,adopted:false,notes:['Fixture comparison is not measured.']},guideAgreement:{status:'measured',positions:10,acceptableActionAccuracy:.8}}});
 await page.route('**/api/**',async route=>{
  const request=route.request(),url=new URL(request.url()),path=url.pathname;
  const json=body=>route.fulfill({contentType:'application/json',body:JSON.stringify(body)});
  if(request.method()==='POST'){
   const body=request.postDataJSON();posts.push({path,body});
   if(path==='/api/learning/runs'){run=summary();return json(run);}
   if(path.endsWith('/control/pause')){if(!failedPause){failedPause=true;return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Fixture connection failure'})});}run={...run,revision:run.revision+1,status:'paused'};return json(run);}
   if(path.endsWith('/control/resume')){run={...run,revision:run.revision+1,status:'running'};return json(run);}
   if(path.endsWith('/control/stop')){run={...run,revision:run.revision+1,status:'stopped',activeGames:[]};games=games.map(g=>({...g,status:'interrupted'}));return json(run);}
   return route.fulfill({status:400,contentType:'application/json',body:'{"detail":"Unexpected fixture mutation"}'});
  }
  if(path==='/api/decks')return json({decks:[]});
  if(path==='/api/models')return json({models:[]});
  if(path==='/api/replays')return json({replays:[]});
  if(path==='/api/positions')return json({positions:[]});
  if(path==='/api/learning/runs')return json({runs:run?[run]:[]});
  if(path==='/api/learning/runs/run-fixture')return json(run);
  if(path==='/api/learning/runs/run-fixture/games')return json({games});
  const match=path.match(/^\/api\/learning\/games\/([^/]+)\/frames$/);
  if(match){const id=match[1],after=Number(url.searchParams.get('after')),playerId=Number(url.searchParams.get('playerId'));frameRequests.push(id);const status=games.find(g=>g.id===id)?.status??'finished';return json({schemaVersion:1,runId:'run-fixture',gameId:id,frames:[0,1].filter(i=>i>after).map(i=>({cursor:i,decisionIndex:i,actor:playerId,priorAction:i?{id:'fixture-action',label:'Fixture action',type:'pass'}:null,observation:observation(playerId)})),nextCursor:1,status,hasMore:false,policyContext:{policy:'model',trainingStatus:'experimental',modelVersion:'fixture-v1',learnsDuringRun:false}});}
  return route.fulfill({status:404,contentType:'application/json',body:'{"detail":"Unknown fixture endpoint"}'});
 });
 await page.goto(`${base}/#replays`);
 await expect(page.getByRole('button',{name:'Start learning',exact:true})).toBeEnabled();
 assert.equal(posts.length,0,'Loading the app must never start learning');
 await page.getByRole('button',{name:'Run settings',exact:true}).click();
 await page.getByLabel('Learning seed',{exact:true}).fill('-1');
 await expect(page.getByRole('button',{name:'Start learning',exact:true})).toBeDisabled();
 await page.getByLabel('Learning seed',{exact:true}).fill('42');
 await page.getByRole('checkbox',{name:'Keep this computer awake during learning',exact:true}).uncheck();
 await page.getByRole('button',{name:'Start learning',exact:true}).click();
 await expect(page.getByRole('region',{name:'Pokémon card table',exact:true})).toBeVisible();
 assert.deepEqual(posts[0].body,{seed:42,keepAwake:false});
 await expect(page.getByRole('region',{name:'Learning progress'})).toContainText('3 of 100 pairings completed');
 await expect(page.getByRole('region',{name:'Learning progress'})).toContainText('0.3750');
 await page.getByRole('button',{name:'Pause playback',exact:true}).click();
 const pausedIndex=await page.getByRole('slider',{name:'Replay position'}).inputValue();
 const beforePosts=posts.length;
 games=[game('g0-next',0),games[1],{...games[0],status:'finished'}];run={...run,revision:run.revision+1,activeGames:games.slice(0,2)};
 await page.waitForTimeout(1900);
 assert.equal(posts.length,beforePosts,'Pausing presentation cannot mutate a learning run');
 assert.equal(frameRequests.includes('g0-next'),false,'A paused viewer must remain in the selected game');
 assert.equal(await page.getByRole('slider',{name:'Replay position'}).inputValue(),pausedIndex);
 await page.getByRole('button',{name:/^(At latest position|Return to live position)$/}).click();
 await expect.poll(()=>frameRequests.includes('g0-next')).toBe(true);
 await page.getByRole('button',{name:'Watch worker 2 · collection',exact:true}).click();
 await expect.poll(()=>frameRequests.includes('g1')).toBe(true);
 await page.getByRole('button',{name:'Pause playback',exact:true}).click();
 await page.getByRole('button',{name:'Pause learning',exact:true}).click();
 await expect(page.getByText('Fixture connection failure')).toBeVisible();
 await page.getByRole('button',{name:'Retry learning request',exact:true}).click();
 await expect(page.getByRole('button',{name:'Resume learning',exact:true})).toBeVisible();
 const pauses=posts.filter(p=>p.path.endsWith('/control/pause'));assert.equal(pauses.length,2);assert.deepEqual(pauses[0].body,pauses[1].body,'Control retries reuse the exact revision and idempotency key');
 await page.reload();
 await expect(page.getByRole('button',{name:'Resume learning',exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'Play playback',exact:true})).toBeVisible();
 await expect(page.getByRole('button',{name:'Watch worker 2 · collection',exact:true})).toHaveClass(/selected-worker/);
 assert.equal(posts.filter(p=>p.path==='/api/learning/runs').length,1,'Reload cannot start another run');
 await page.getByRole('button',{name:'Resume learning',exact:true}).click();
 await expect(page.getByRole('button',{name:'Pause learning',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Stop learning',exact:true}).click();
 await expect(page.getByRole('button',{name:'Start learning',exact:true})).toBeVisible();
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 assert.deepEqual(pageErrors,[]);
 console.log(JSON.stringify({passed:true,scope:'explicit-continuous-learning-ui-fixtures',manualStart:true,bothWorkers:true,pausedPresentationDoesNotFollow:true,automaticLiveFollow:true,idempotentControlRetry:true,reload:true,metricsSeparated:true,mobileOverflow:false}));
}finally{await browser.close();}
