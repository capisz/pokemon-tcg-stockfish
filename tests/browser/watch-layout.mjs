import {chromium,expect} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';

// Read-only visual verification of an existing real replay. Run smoke.mjs first.
const base=process.env.PTCG_BROWSER_URL||'http://127.0.0.1:5173';
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900},reducedMotion:'reduce'});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const library=await (await page.request.get(`${base}/api/replays`)).json();
 const record=library.replays.find(r=>r.status==='finished'&&r.frames>30);
 assert.ok(record,'Run real replay smoke first');
 const replay=await (await page.request.get(`${base}/api/replays/${record.id}?playerId=0`)).json();
 const index=replay.frames.findIndex(f=>f.observation.turn>=2&&f.actor===0);
 assert.ok(index>=0);
 await page.goto(`${base}/#replays`);
 const picker=page.getByRole('combobox',{name:/^Saved replay/});
 await expect(picker.locator(`option[value="${record.id}"]`)).toHaveCount(1);
 await picker.selectOption(record.id);
 const slider=page.getByRole('slider',{name:'Replay position'});
 await expect(slider).toHaveAttribute('max',String(replay.frames.length-1));
 await slider.fill(String(index));
 await expect(page.locator('.table-active .table-card')).toHaveCount(2);
 await expect(page.getByRole('meter')).toBeVisible();
 const table=await page.getByRole('region',{name:'Pokémon card table',exact:true}).boundingBox();
 const transport=await page.getByRole('region',{name:'Playback controls',exact:true}).boundingBox();
 assert.ok(table&&table.y+table.height<=900,`Table should fit900px desktop viewport: ${JSON.stringify(table)}`);
 assert.ok(transport&&transport.y+transport.height<=900,`Playback should fit900px desktop viewport: ${JSON.stringify(transport)}`);
 await mkdir('artifacts/browser',{recursive:true});
 await page.screenshot({path:'artifacts/browser/watch-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:'artifacts/browser/watch-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);
 console.log(JSON.stringify({passed:true,replayId:record.id,decision:index,desktopBoardAndControlsFit:true,mobileOverflow:false,reducedMotion:true,pageErrors:errors}));
}finally{await browser.close();}
