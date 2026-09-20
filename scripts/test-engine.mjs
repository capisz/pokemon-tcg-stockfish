import { build } from 'esbuild';
import { readdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

// Match the production bundle. Mixing ESM test imports with the vendored CommonJS
// package can load duplicate effect classes and CardManager singletons under tsx.
const directory = await mkdtemp(join(tmpdir(), 'ptcg-engine-tests-'));
try {
  const entries = (await readdir('packages/engine/tests')).filter(n => n.endsWith('.test.ts')).sort();
  await build({entryPoints: entries.map(n => `packages/engine/tests/${n}`), outdir: directory,
    outExtension: {'.js': '.cjs'}, bundle: true, platform: 'node', format: 'cjs', target: 'node22', tsconfig: 'tsconfig.json'});
  const child = spawn(process.execPath, ['--test', ...entries.map(n => join(directory, n.replace(/\.ts$/, '.cjs')))], {stdio: 'inherit'});
  process.exitCode = await new Promise((resolve, reject) => {child.once('error', reject); child.once('exit', code => resolve(code ?? 1));});
} finally { await rm(directory, {recursive: true, force: true}); }
