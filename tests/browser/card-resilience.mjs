import {chromium, expect} from '@playwright/test';
import assert from 'node:assert/strict';

// Focused UI fixture test against the Vite-served production components.
// Only card image requests are simulated; these are not gameplay screenshots.
const base=process.env.PTCG_BROWSER_URL||'http://127.0.0.1:5173';
const browser=await chromium.launch({headless:true});
try{
  const page=await browser.newPage({viewport:{width:1000,height:800}});
  const held=new Map();
  await page.route('**/__card-art-fixture/**',route=>{held.set(route.request().url().split('/').pop(),route);});
  await page.goto(`${base}/#play`);
  const messages=await page.evaluate(async()=>{
    const React=(await import('/node_modules/.vite/deps/react.js')).default;
    const ReactDOM=(await import('/node_modules/.vite/deps/react-dom_client.js')).default;
    const {CardFace,promptInstruction}=await import('/src/Play.tsx');
    const host=document.createElement('div');
    host.id='card-resilience-fixture';
    Object.assign(host.style,{position:'fixed',inset:'0',padding:'32px',background:'white',zIndex:'1000'});
    document.body.append(host);
    const root=ReactDOM.createRoot(host);
    window.renderFixture=(file)=>root.render(React.createElement(CardFace,{card:{id:'fixture-1',name:'Readable fixture card',kind:'pokemon',hp:150,imageUrl:`${location.origin}/__card-art-fixture/${file}`},inspect:()=>{}}));
    window.renderFixture('slow.png');
    return {
      setup:promptInstruction('CHOOSE_STARTING_POKEMONS'),
      first:promptInstruction('GO_FIRST'),
      unknown:promptInstruction('CHOOSE_POKEMON_TO_HEAL'),
      prose:promptInstruction('Choose exactly two cards.'),
    };
  });
  const card=page.locator('#card-resilience-fixture .table-card');
  await expect(card).toHaveAttribute('data-art-state','loading');
  await expect(card.locator('.printed-face strong')).toHaveText('Readable fixture card');
  await expect(card.locator('.printed-id')).toHaveText('fixture-1');
  await expect(card.locator('.printed-face')).toBeVisible();
  await expect.poll(()=>held.has('slow.png')).toBe(true);
  const dimensions=await card.locator('img').evaluate(img=>({width:img.getBoundingClientRect().width,height:img.getBoundingClientRect().height,display:getComputedStyle(img).display}));
  assert.ok(dimensions.width>0&&dimensions.height>0);
  assert.notEqual(dimensions.display,'none');
  await held.get('slow.png').abort('failed');
  await expect(card).toHaveAttribute('data-art-state','unavailable');
  await expect(card.locator('.printed-face')).toBeVisible();
  // Reusing a hand-slot component for a new image must clear the old failure.
  await page.evaluate(()=>window.renderFixture('loaded.png'));
  await expect(card).toHaveAttribute('data-art-state','loading');
  await expect(card.locator('.printed-face')).toBeVisible();
  await expect.poll(()=>held.has('loaded.png')).toBe(true);
  await held.get('loaded.png').fulfill({contentType:'image/png',body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aKfkAAAAASUVORK5CYII=','base64')});
  await expect(card).toHaveAttribute('data-art-state','loaded');
  await expect(card.locator('.printed-face')).toHaveCount(0);
  await expect(card.locator('img')).toHaveCSS('opacity','1');
  assert.match(messages.setup,/first selected becomes Active/);
  assert.equal(messages.first,'Do you want to go first?');
  assert.equal(messages.unknown,'Choose Pokémon to heal.');
  assert.equal(messages.prose,'Choose exactly two cards.');
  console.log(JSON.stringify({passed:true,scope:'card-image-network-resilience',slowFallback:true,failedFallback:true,loadedArt:true,promptInstructions:true}));
}finally{await browser.close();}
