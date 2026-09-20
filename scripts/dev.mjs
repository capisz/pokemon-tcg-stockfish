import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve, delimiter } from 'node:path';

const windows = process.platform === 'win32';
const python = process.env.PTCG_PYTHON || resolve('.venv', windows ? 'Scripts/python.exe' : 'bin/python');
const vite = resolve('web/node_modules/vite/bin/vite.js');
if (!existsSync(python) || !existsSync(vite) || !existsSync('packages/engine/dist/worker.cjs')) {
  console.error('Run the README installation steps and npm run engine:build first.');
  process.exit(1);
}
const children = [
  spawn(python, ['-m', 'ptcg_lab.cli', 'serve', '--port', '8765'], {
    stdio: 'inherit',
    env: {...process.env, PYTHONPATH: [resolve('src'), process.env.PYTHONPATH].filter(Boolean).join(delimiter)},
  }),
  spawn(process.execPath, [vite], {cwd: resolve('web'), stdio: 'inherit'}),
];
let stopping = false;
function stop(code = 0) {
  if(stopping) return;
  stopping = true;
  children.forEach(child => {
    if (!child.pid || child.exitCode !== null) return;
    if (windows) {
      // Windows does not propagate SIGTERM to descendants. Durable journals
      // recover an interrupted game; kill the complete local service tree.
      const killer = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], {stdio: 'ignore'});
      killer.on('error', () => child.kill());
    } else {
      child.kill('SIGTERM');
      const timeout = setTimeout(() => { if (child.exitCode === null) child.kill('SIGKILL'); }, 8000);
      timeout.unref();
      child.once('exit', () => clearTimeout(timeout));
    }
  });
  process.exitCode = code;
}
children.forEach(child => {
  child.on('error', error => {console.error(error.message); stop(1);});
  child.on('exit', code => stop(code ?? 0));
});
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
