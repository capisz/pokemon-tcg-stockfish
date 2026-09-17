import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
if (!existsSync('.venv/bin/python') || !existsSync('packages/engine/dist/worker.cjs')) {
  console.error('Run the README installation steps and npm run engine:build first.');
  process.exit(1);
}
const children = [
  spawn('.venv/bin/python', ['-m', 'ptcg_lab.cli', 'serve', '--port', '8765'], {stdio:'inherit'}),
  spawn('npm', ['--prefix','web','run','dev'], {stdio:'inherit'}),
];
let stopping = false;
function stop(code = 0) {
  if(stopping) return;
  stopping = true;
  children.forEach(child => child.kill('SIGTERM'));
  process.exitCode = code;
}
children.forEach(child => {
  child.on('error', error => {console.error(error.message); stop(1);});
  child.on('exit', code => stop(code ?? 0));
});
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
