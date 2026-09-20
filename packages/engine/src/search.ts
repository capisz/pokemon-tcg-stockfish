import { Environment } from './environment';
import { opponentHypotheses } from './hypotheses';
import { chooseAction } from './policies';
import { SeededRandom } from './random';
import type { Observation, LegalAction, Replay } from './types';
import {loadLeafModel, type LeafEnvelope} from './learned-value';

interface ContinuationStep { playerId: number; label: string; type: string }
interface Continuation {
  conditional: true; representative: true; description: string; opponentArchetype: string;
  steps: ContinuationStep[]; end: 'terminal' | 'cutoff'; outcome?: Replay['outcome'];
}
interface Edge { prior?:number; visits: number; total: number; squares: number; action: LegalAction; continuation?: Continuation }
/** A sampled opponent's private prompt choice is never presented as a public line. */
export function continuationStep(observation: Observation, action: LegalAction, rootPlayer: number): ContinuationStep {
  return {playerId: observation.playerId, type: action.type,
    label: observation.playerId !== rootPlayer && (action.type === 'prompt'||action.type==='choice')
      ? `Resolve ${observation.prompt?.type ?? 'private choice'} (private choice omitted)` : action.label};
}
interface Node { visits: number; edges: Map<string, Edge> }
const actionKey = (a: LegalAction) => JSON.stringify([a.type, a.cardId, a.target, a.label]);
function infoKey(o: Observation): string {
  return JSON.stringify([o.playerId, o.turn, o.phase, o.players, o.prompt, o.knowledge, o.legalActions.map(actionKey)]);
}
function leafResult(o: Observation, root: number): number {
  const resource = (p: Observation['players'][number]) => (6 - p.prizesRemaining) * 1.5
    + [p.active, ...p.bench].filter(x => x).reduce((n, x) => n + ((x!.card.hp ?? 0) - x!.damage) / 200 + x!.energy.length * 0.2, 0)
    + p.handCount * 0.08;
  const advantage = resource(o.players[root]) - resource(o.players[1 - root]);
  return 1 / (1 + Math.exp(-advantage / 3));
}

/** Executable imperfect-information research search. No live Environment or true hidden state input. */
export function search(params: {observation: Observation; method?: 'rollout' | 'ismcts'; budgetMs?: number; seed?: number; iterations?: number; maxRolloutDecisions?: number; knownOpponentDeckId?: string; priorRevealedCards?: string[]; rootPriors?: {actionId:string; probability:number}[]; leafModel?:LeafEnvelope}) {
  const start = performance.now();
  const o = params.observation;
  const method = params.method ?? 'rollout';
  const learned = params.leafModel ? loadLeafModel(params.leafModel) : null;
  if (method !== 'rollout' && method !== 'ismcts') throw new Error('Unknown search method.');
  const budget = Math.max(1, Math.min(120000, params.budgetMs ?? 1000));
  const limit = Math.max(1, Math.min(1000, params.iterations ?? 100));
  const horizon = Math.max(1, Math.min(80, params.maxRolloutDecisions ?? 16));
  const warnings = new Set<string>([learned?'Experimental search: cutoff values use a frozen outcome model; scores are not calibrated probabilities.':'Experimental search: rollout cutoffs use an untrained resource heuristic; scores are not calibrated probabilities.',
    'Beliefs include main/training lists and observation-constrained unknown variants; heldout exact lists require explicit known-list mode. Unsupported knowledge histories refuse search.',
    'Determinized continuations can suffer strategy fusion; no optimal-play claim.']);
  const unavailable = (reason: string) => ({status: 'unavailable' as const, method, alternatives: [], iterations: 0, elapsedMs: performance.now() - start, warnings: [reason]});
  if (!o?.searchPosition) return unavailable(o?.searchUnavailableReason ?? 'A stable public search position is required.');
  if (o.playerId !== o.decisionPlayer || !o.legalActions.length) return unavailable('Observation must belong to the current decision player.');
  const rng = new SeededRandom(params.seed ?? 42);
  const hypotheses=opponentHypotheses(o.searchPosition,params.priorRevealedCards,params.knownOpponentDeckId).filter(h=>{
    try {Environment.fromPublicPosition(o.searchPosition!,0,h.kind==='unknown-variant'?h.deck:h.id);return true;}catch{return false;}
  });
  if (!hypotheses.length) return unavailable('No supported deck hypothesis matches the visible cards and public state.');
  const totalWeight=hypotheses.reduce((n,h)=>n+h.weight,0);
  hypotheses.forEach(h=>h.weight/=totalWeight);
  const roots = new Map<string, Edge>(o.legalActions.map(action => [actionKey(action), {visits: 0, total: 0, squares: 0, action}]));
  if(params.rootPriors){
    const values=new Map(params.rootPriors.map(p=>[p.actionId,p.probability]));
    if([...values].some(([id,p])=>!o.legalActions.some(a=>a.id===id)||!Number.isFinite(p)||p<0))throw new Error('Invalid root policy priors.');
    const total=[...roots.values()].reduce((n,e)=>n+(values.get(e.action.id)??0),0);
    if(total<=0)throw new Error('Root policy priors require positive mass.');
    for(const edge of roots.values())edge.prior=(values.get(edge.action.id)??0)/total;
  }
  const tree = new Map<string, Node>();
  let iterations = 0; let aborted = 0; let cutoffCount = 0;
  const select = (edges: Edge[], visits: number, actor: number) => {
    const unvisited = edges.filter(e => e.visits === 0);
    if (unvisited.length) return unvisited.reduce((a,b)=>(b.prior??0)>(a.prior??0)?b:a,unvisited[rng.int(unvisited.length)]);
    const value = (e: Edge) => (actor === o.playerId ? e.total / e.visits : 1 - e.total / e.visits) + (e.prior===undefined?Math.sqrt(2 * Math.log(visits + 1) / e.visits):1.4*e.prior*Math.sqrt(visits+1)/(e.visits+1));
    return edges.reduce((a, b) => value(a) > value(b) ? a : b);
  };
  while (iterations + aborted < limit && performance.now() - start < budget) {
    let sample=rng.uint32()/0x100000000;
    const hypothesis=hypotheses.find(h=>(sample-=h.weight)<0)??hypotheses[hypotheses.length-1];
    const env = Environment.fromPublicPosition(o.searchPosition, rng.uint32(), hypothesis.kind==='unknown-variant'?hypothesis.deck:hypothesis.id, Number(o.legalActions[0].id.split(':')[0]));
    const rootObservation = env.observe();
    const rootChoices = rootObservation.legalActions;
    const available = rootChoices.map(a => roots.get(actionKey(a))).filter((e): e is Edge => !!e);
    if (!available.length) {aborted++; continue;}
    let root: Edge;
    if (method === 'rollout') {const min = Math.min(...available.map(e => e.visits)); const least = available.filter(e => e.visits === min); root = least[rng.int(least.length)];}
    else root = select(available, iterations, o.playerId);
    const rootAction = rootChoices.find(a => actionKey(a) === actionKey(root.action))!;
    const path: {edge: Edge; node?: Node}[] = [{edge: root}];
    const steps = [continuationStep(rootObservation, rootAction, o.playerId)];
    try {
      env.step(rootAction.id);
      let expanded = false;
      for (let depth = 1; depth < horizon && env.status === 'running'; depth++) {
        if (performance.now() - start >= budget) break;
        const obs = env.observe();
        let chosen: LegalAction;
        if (method === 'ismcts' && !expanded) {
          const key = infoKey(obs);
          let node = tree.get(key);
          if (!node) {node = {visits: 0, edges: new Map()}; tree.set(key, node);}
          for (const action of obs.legalActions) if (!node.edges.has(actionKey(action))) node.edges.set(actionKey(action), {visits: 0, total: 0, squares: 0, action});
          const edges = obs.legalActions.map(a => node!.edges.get(actionKey(a))!);
          const edge = select(edges, node.visits, obs.playerId);
          expanded = edge.visits === 0;
          chosen = obs.legalActions.find(a => actionKey(a) === actionKey(edge.action))!;
          path.push({edge, node});
        } else chosen = chooseAction(obs, 'heuristic', rng);
        if (steps.length < 8) steps.push(continuationStep(obs, chosen, o.playerId));
        env.step(chosen.id);
      }
      let value: number;
      let outcome: Replay['outcome'] = null;
      if (env.status === 'finished') {
        outcome = env.replay().outcome;
        const winner = outcome!.winner; value = winner === null ? 0.5 : winner === o.playerId ? 1 : 0;
      } else if (env.status === 'error') {aborted++; continue;}
      else {value = learned ? learned.evaluate(env.observe(o.playerId)) : leafResult(env.observe(), o.playerId); cutoffCount++;}
      root.continuation ??= {
        conditional: true, representative: true,
        description: 'Illustrative sampled continuation; not a forced or proven best line. Subsequent choices depend on sampled draws and hidden information.',
        opponentArchetype: hypothesis.archetype, steps, end: env.status === 'finished' ? 'terminal' : 'cutoff',
        ...(outcome ? {outcome} : {}),
      };
      for (const {edge, node} of path) {edge.visits++; edge.total += value; edge.squares += value * value; if (node) node.visits++;}
      iterations++;
    } catch (error) {aborted++; warnings.add(`Some sampled continuations failed and were excluded: ${error instanceof Error ? error.message : JSON.stringify(error)}`);}
  }
  if (cutoffCount) warnings.add(`${cutoffCount} of ${iterations} completed samples used ${learned?'learned':'heuristic'} horizon/budget evaluation, not terminal outcomes.`);
  if (aborted) warnings.add(`${aborted} continuations were excluded due to unsupported or failed transitions.`);
  if ([...roots.values()].some(e => !e.visits)) warnings.add('Budget did not visit every legal candidate; unvisited moves have null scores.');
  return {
    status: iterations ? 'complete' as const : 'unavailable' as const, method, iterations, elapsedMs: performance.now() - start,
    valueContext: learned?{kind:'learned',dataTier:learned.dataTier,label:learned.dataTier==='experimental'?'Experimental learned':'Learned',modelVersion:learned.modelVersion,checkpointHash:learned.checkpointHash,calibrated:false}:{kind:'heuristic',calibrated:false},
    hypotheses: hypotheses.map(h=>h.id), hypothesisWeights:hypotheses.map(({id,archetype,kind,weight})=>({id,archetype,kind,weight})), treeNodes: tree.size, warnings: [...warnings],
    alternatives: [...roots.values()].map(e => {
      const mean = e.visits ? e.total / e.visits : null;
      const stderr = e.visits > 1 ? Math.sqrt(Math.max(0, e.squares / e.visits - mean! * mean!) / (e.visits - 1)) : null;
      return {actionId: e.action.id, label: e.action.label, score: mean, expectedResult: mean, visits: e.visits, uncertainty: stderr, ...(e.continuation ? {continuation: e.continuation} : {}),
        description: e.visits ? `${method === 'ismcts' ? 'Information-set UCB' : 'Balanced rollout'}: ${e.visits} sampled continuations; approximate ${learned?'learned':'heuristic'} leaves.` : 'Not visited within the compute budget.'};
    }).sort((a, b) => (b.score ?? -1) - (a.score ?? -1)),
  };
}
