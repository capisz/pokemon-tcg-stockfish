import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

async function sourceFiles(directory) {
  const paths = [];
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) paths.push(...await sourceFiles(path));
    else if (/\.(ts|json)$/.test(path)) paths.push(path);
  }
  return paths;
}
const hash = createHash('sha256');
const paths = (await Promise.all(['packages/engine/src', 'vendor/twinleaf/ptcg-server/src', 'decks', 'formats'].map(sourceFiles))).flat().sort();
for (const path of paths) hash.update(path).update('\0').update(await readFile(path));
const fingerprint = hash.digest('hex').slice(0, 16);
await build({entryPoints: ['packages/engine/src/worker.ts'], outfile: 'packages/engine/dist/worker.cjs', bundle: true, platform: 'node', format: 'cjs', target: 'node22', sourcemap: true, logLevel: 'info', tsconfig: 'tsconfig.json', define: {__ENGINE_BUILD__: JSON.stringify(fingerprint)}});
await build({entryPoints: ['research/learning_mind/planner_worker.ts'], outfile: 'packages/engine/dist/learning-mind-planner.cjs', bundle: true, platform: 'node', format: 'cjs', target: 'node22', sourcemap: true, logLevel: 'info', tsconfig: 'tsconfig.json'});
console.log(`Engine source fingerprint: ${fingerprint}`);
