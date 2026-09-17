"""Compare search methods under matched budgets; this does not measure strength."""
import argparse
import json
from pathlib import Path
from ptcg_lab.config import Settings
from ptcg_lab.engine import EngineClient
from ptcg_lab.storage import Store

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('data/current-smoke'))
parser.add_argument('--budget-ms', type=int, default=1000)
args = parser.parse_args()
if not 50 <= args.budget_ms <= 5000:
    parser.error('budget-ms must be 50..5000')
store = Store(args.data)
settings = Settings.from_env()
records = []
with EngineClient(settings.root) as engine:
    for index in store.list('replay-index')[:3]:
        replay = store.get('replays', index['id'])
        frame = next((frame for frame in replay['frames'] if frame['observations'][frame['actor']].get('searchPosition') and len(frame['observations'][frame['actor']]['legalActions']) > 1), None)
        if not frame:
            continue
        observation = frame['observations'][frame['actor']]
        for method in ('rollout', 'ismcts'):
            result = engine.request('search', {'observation': observation, 'method': method, 'budgetMs': args.budget_ms, 'iterations': 1000, 'seed': 42, 'maxRolloutDecisions': 16})
            records.append({'replayId': index['id'], 'decisionIndex': frame['decisionIndex'], 'method': method, 'budgetMs': args.budget_ms, 'iterations': result['iterations'], 'elapsedMs': result['elapsedMs'], 'visitedCandidates': sum(a['visits'] > 0 for a in result['alternatives']), 'legalCandidates': len(observation['legalActions']), 'status': result['status'], 'warnings': result['warnings']})
    print(json.dumps({'engine': engine.request('health'), 'records': records, 'interpretation': 'Matched wall-clock budgets and seeds on identical player observations. Heuristic leaves, three positions only, no playing-strength or preferred-method conclusion.'}, indent=2))
