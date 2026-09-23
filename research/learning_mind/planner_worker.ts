import {generateTransitionMacroPlans} from './transition_macro_planner';

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk: string) => input += chunk);
process.stdin.on('end', () => {
  try {
    if (Buffer.byteLength(input, 'utf8') > 32 * 1024 * 1024) throw new Error('Request exceeds 32 MiB UTF-8 limit.');
    const request = JSON.parse(input);
    const result = generateTransitionMacroPlans(request.observation, request.seed);
    process.stdout.write(JSON.stringify(result) + '\n');
  } catch (error) {
    process.stderr.write((error instanceof Error ? error.message : String(error)) + '\n');
    process.exitCode = 1;
  }
});
