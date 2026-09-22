/** Research-only evaluator for the frozen TypeScript heuristic on saved observations. */
import { readFileSync } from 'node:fs';
import { chooseAction } from '../../packages/engine/src/policies';
import { SeededRandom } from '../../packages/engine/src/random';
import type { Observation } from '../../packages/engine/src/types';

interface Request { observation: Observation; seed: number }

const requests = JSON.parse(readFileSync(0, 'utf8')) as Request[];
const results = requests.map(({ observation, seed }) => ({
  actionId: chooseAction(observation, 'heuristic', new SeededRandom(seed)).id,
}));
process.stdout.write(`${JSON.stringify(results)}\n`);
