import { createInterface } from 'node:readline';
import { Environment, ENGINE_VERSION, getDecks } from './environment';
import { chooseAction } from './policies';
import { SeededRandom } from './random';
import { search } from './search';
import { tacticalFixture } from './tactical-fixtures';

// stdout is protocol only, even when upstream logs diagnostics.
console.log = (...args: unknown[]) => console.error(...args);
let environment: Environment | null = null;
function current(): Environment { if (!environment) throw new Error('Call reset first.'); return environment; }
export function request(method: string, params: any = {}): any {
  switch (method) {
    case 'health': return {ok: true, protocolVersion: 1, engineVersion: ENGINE_VERSION, firstPlayerControl: true, warnings: ['Experimental rules adapter; learned strength is not established.']};
    case 'decks': return getDecks(params);
    case 'reset': environment = new Environment(); return environment.reset(params.seed, params.decks, params.firstPlayer);
    case 'observe': return current().observe(params.playerId);
    case 'choose': {
      const policy=params.policy??'heuristic';
      if(!['random','heuristic'].includes(policy))throw new Error('Unknown policy.');
      const seed=params.seed??((current().seed^current().decisionIndex^0x13579bdf)>>>0);
      if(!Number.isSafeInteger(seed)||seed<0||seed>0xffffffff)throw new Error('Seed must be a uint32 integer.');
      return chooseAction(current().observe(),policy,new SeededRandom(seed));
    }
    case 'step': return current().step(params.actionId);
    case 'replay': return current().replay();
    case 'branch': environment = current().branch(); return {observation: environment.observe(), status: environment.status, decisionIndex: environment.decisionIndex};
    case 'search': return search(params);
    case 'fixture': return tacticalFixture(params);
    case 'run': {
      const max = params.maxDecisions ?? 500;
      if (!Number.isSafeInteger(max) || max < 1 || max > 10000) throw new Error('maxDecisions must be 1..10000.');
      if (!['random', 'heuristic'].includes(params.policy ?? 'heuristic')) throw new Error('Unknown policy.');
      environment = new Environment(); environment.reset(params.seed, params.decks, params.firstPlayer);
      const rng = new SeededRandom((params.seed ^ 0x13579bdf) >>> 0);
      try {
        for (let i = 0; i < max && environment.status === 'running'; i++) {
          const action = chooseAction(environment.observe(), params.policy ?? 'heuristic', rng);
          environment.step(action.id);
        }
      } catch (error) { environment.fail(error instanceof Error ? error.message : JSON.stringify(error)); }
      return environment.replay();
    }
    default: throw new Error(`Unknown method: ${method}`);
  }
}
const lines = createInterface({input: process.stdin, crlfDelay: Infinity});
lines.on('line', line => {
  let id: any = null;
  try {
    if (line.length > 1024 * 1024) throw new Error('Request exceeds 1 MB.');
    const parsed = JSON.parse(line); id = parsed.id;
    const result = request(parsed.method, parsed.params);
    process.stdout.write(JSON.stringify({id, result}) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({id, error: {code: 'ENGINE_ERROR', message: String(error)}}) + '\n');
  }
});
