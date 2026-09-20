import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';

// Integration test against the actual local API/worker. Start npm run dev first.
const browser = await chromium.launch({headless:true});
try {
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/#replays');
  await page.getByText(/decks in the research pool/).waitFor();
  await page.getByRole('combobox',{name:/^Player 1/}).selectOption('crustle');
  await page.getByRole('combobox',{name:/^Player 2/}).selectOption('mega-lucario');
  await page.getByLabel('Seed',{exact:true}).fill('17');
  await page.getByRole('button',{name:'Run game',exact:true}).click();
  await page.getByText(/won · rules-terminal/).waitFor({timeout:120000});
  const replayId = await page.getByRole('combobox',{name:/^Saved replay/}).inputValue();
  const replay = await (await page.request.get(`http://127.0.0.1:8765/api/replays/${replayId}`)).json();
  assert.equal(replay.status,'finished');
  let index = replay.frames.findIndex(frame => frame.observations[frame.actor]?.searchPosition && frame.observations[frame.actor].turn >= 2);
  if(index < 0) index = replay.frames.findIndex(frame => frame.action && frame.observations[frame.actor]?.turn >= 2);
  assert.ok(index >= 0, 'Expected a real pre-decision position');
  const searchable = Boolean(replay.frames[index].observations[replay.frames[index].actor].searchPosition);
  if(!searchable) assert.ok(replay.frames[index].observations[replay.frames[index].actor].searchUnavailableReason,
    'Unsupported knowledge must explain why search is unavailable');
  await page.getByRole('combobox',{name:/^Perspective/}).selectOption(String(replay.frames[index].actor));
  const slider=page.getByRole('slider',{name:'Replay position'});
  await slider.focus(); await slider.press('Home');
  for(let i=0;i<index;i++) await slider.press('ArrowRight');
  assert.equal(await slider.inputValue(),String(index));
  const saveResponse=page.waitForResponse(response => response.url().endsWith('/api/positions') && response.request().method()==='POST');
  await page.getByRole('button',{name:'Save position',exact:true}).click();
  const saved=await (await saveResponse).json();
  await page.getByRole('combobox',{name:/^Saved position/}).selectOption(saved.id);
  await page.getByRole('button',{name:'Open source replay'}).waitFor();
  assert.equal(await page.getByRole('combobox',{name:/^Perspective/}).isDisabled(),true);
  const position=await (await page.request.get(`http://127.0.0.1:8765/api/positions/${saved.id}`)).json();
  assert.equal(position.frames,undefined);
  assert.equal(position.observation.players[1-position.playerId].hand.length,0);
  const analysis=await (await page.request.post('http://127.0.0.1:8765/api/analyze',{data:{positionId:saved.id,budgetMs:500}})).json();
  assert.ok(analysis.evaluation);
  if(!searchable) assert.equal(analysis.search.status,'unavailable');
  if(!analysis.evaluation.calibrated) assert.equal(analysis.evaluation.winProbability,null);
  await page.waitForTimeout(800);
  const meter=page.getByRole('meter');
  await meter.waitFor();
  assert.ok(Number(await meter.getAttribute('aria-valuenow')) >= -6);
  assert.ok(Number(await meter.getAttribute('aria-valuenow')) <= 6);
  await mkdir('artifacts/browser',{recursive:true});
  await page.screenshot({path:'artifacts/browser/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:'artifacts/browser/mobile.png',fullPage:true});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,replayId,positionId:saved.id,frames:replay.frames.length,evaluation:analysis.evaluation.status,search:analysis.search,pageErrors:errors,mobileOverflow:false}));
} finally {await browser.close();}
