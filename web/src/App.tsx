import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api, errorMessage } from './api';
import { Play } from './Play';
import { TeachingReview } from './TeachingReview';
import type { Analysis, Card, Deck, Job, Observation, Player, Pokemon, PositionSummary, Replay, ReplaySummary, SavedPosition, SampledContinuation } from './types';

const signed = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const humanize = (value: string) => value.replace(/[_-]+/g, ' ').replace(/^./, (letter) => letter.toUpperCase());

function Arrow({ direction, end = false }: { direction: 'left' | 'right'; end?: boolean }) {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={direction === 'left' ? 'm14 6-6 6 6 6' : 'm10 6 6 6-6 6'} />
    {end && <path d={direction === 'left' ? 'M5 5v14' : 'M19 5v14'} />}
  </svg>;
}

function WarningList({ warnings, label = 'Research notes' }: { warnings: string[]; label?: string }) {
  const unique = [...new Set(warnings.filter(Boolean))];
  if (!unique.length) return null;
  return <details className="warning-details"><summary>{label} <span className="count">{unique.length}</span></summary>
    <ul>{unique.map((warning) => <li key={warning}>{warning}</li>)}</ul>
  </details>;
}

function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="error-notice" role="alert"><p>{message}</p>{retry && <button type="button" className="secondary small" onClick={retry}>Retry</button>}</div>;
}

function CardList({ cards, empty = 'None' }: { cards: Card[]; empty?: string }) {
  if (!cards.length) return <p className="muted empty-inline">{empty}</p>;
  const grouped = new Map<string, { card: Card; count: number }>();
  for (const card of cards) {
    const existing = grouped.get(card.id);
    grouped.set(card.id, { card, count: (existing?.count ?? 0) + 1 });
  }
  return <ul className="card-list">{[...grouped.values()].map(({ card, count }) => <li key={card.id}>
    <span className={`card-kind kind-${card.kind}`}>{card.kind === 'pokemon' ? 'P' : card.kind === 'energy' ? 'E' : 'T'}</span>
    <span title={card.id}>{card.name}</span><span className="card-count">{count > 1 ? `×${count}` : ''}</span>
  </li>)}</ul>;
}

function PokemonTile({ pokemon, active = false }: { pokemon: Pokemon; active?: boolean }) {
  const { card } = pokemon;
  const remaining = typeof card.hp === 'number' ? Math.max(0, card.hp - pokemon.damage) : null;
  const energy = [...new Set(pokemon.energy)].map((name) => ({ name, count: pokemon.energy.filter((item) => item === name).length }));
  return <article className={`pokemon-tile ${active ? 'active-pokemon' : ''}`} aria-label={`${card.name}, ${active ? 'active' : 'bench'} Pokémon`}>
    <div className="pokemon-card-top"><span className="pokemon-stage">{card.stage || 'Pokémon'}</span>{card.prizeValue && card.prizeValue > 1 ? <span>{card.prizeValue} prizes</span> : null}</div>
    <h4 title={card.id}>{card.name}</h4>
    <div className="pokemon-hp"><span>{remaining !== null ? <><strong>{remaining}</strong> / {card.hp} HP</> : 'HP unavailable'}</span>{pokemon.damage > 0 && <span className="damage-label">{pokemon.damage} damage</span>}</div>
    {typeof card.hp === 'number' && card.hp > 0 && <div className="hp-track" aria-hidden="true"><span style={{ width: `${100 * (remaining ?? 0) / card.hp}%` }} /></div>}
    <p className="pokemon-energy">{energy.length ? energy.map(({ name, count }) => `${count} ${humanize(name)}`).join(' · ') : 'No energy attached'}</p>
    {pokemon.tools.length > 0 && <p className="pokemon-extra">{pokemon.tools.join(', ')}</p>}
    {pokemon.conditions.length > 0 && <p className="pokemon-condition">{pokemon.conditions.map(humanize).join(', ')}</p>}
    {active && card.attacks && card.attacks.length > 0 && <details className="attacks"><summary>Attacks</summary><ul>{card.attacks.map((attack, index) => <li key={`${attack.name}-${index}`}><span>{attack.name}<small>{attack.cost.map(humanize).join(' · ') || 'No energy cost'}</small></span><strong>{attack.damage || '—'}</strong></li>)}</ul></details>}
  </article>;
}

function PlayerBoard({ player, own, deciding, deckName }: { player: Player; own: boolean; deciding: boolean; deckName: string }) {
  return <section className={`player-board ${own ? 'own-board' : 'opponent-board'}`} aria-label={`Player ${player.id + 1} board`}>
    <div className="player-heading"><div><h3><span className="player-number">P{player.id + 1}</span> {deckName}</h3><span className="player-view-label">{own ? 'Selected player' : 'Opponent'}{deciding ? ' · To act' : ''}</span></div>
      <dl className="zone-counts"><div><dt>Prizes</dt><dd>{player.prizesRemaining}</dd></div><div><dt>Deck</dt><dd>{player.deckCount}</dd></div><div><dt>Hand</dt><dd>{player.handCount}</dd></div></dl>
    </div>
    <div className="play-area"><div className="active-slot"><span className="zone-label">Active</span>{player.active ? <PokemonTile pokemon={player.active} active /> : <div className="empty-slot">No active Pokémon</div>}</div>
      <div className="bench-area"><span className="zone-label">Bench <span className="muted">{player.bench.length}</span></span><div className="bench-row">{player.bench.length ? player.bench.map((pokemon, index) => <PokemonTile key={`${pokemon.card.id}-${index}`} pokemon={pokemon} />) : <p className="empty-bench">No Pokémon on the bench</p>}</div></div>
    </div>
    <div className="player-zones"><details className="zone-details" open={own}><summary>{own ? `Hand · ${player.handCount}` : `Hidden hand · ${player.handCount}`}</summary>{own ? <CardList cards={player.hand} empty="The hand is empty." /> : <p className="muted empty-inline">Card identities are hidden from this perspective.</p>}</details>
      <details className="zone-details"><summary>Discard · {player.discard.length}</summary><CardList cards={player.discard} /></details></div>
  </section>;
}

function Continuation({ line }: { line: SampledContinuation }) {
  const result = line.end === 'cutoff' ? 'Sample stopped at a rollout cutoff.' : line.outcome?.winner !== null && line.outcome?.winner !== undefined ? `Sample ended in a Player ${line.outcome.winner + 1} win.` : line.outcome?.reason === 'rules-draw' ? 'Sample ended in a rules draw.' : 'Sample reached a terminal state.';
  return <details className="sampled-line"><summary>Illustrative sampled line</summary>
    <p>{line.description}</p><p>Sampled opponent assumption: {humanize(line.opponentArchetype)}.</p>
    <ol>{line.steps.map((step, index) => <li key={`${index}-${step.playerId}`}><span>P{step.playerId + 1}</span><span>{step.label}</span></li>)}</ol>
    <p>{result} The displayed steps are an excerpt, not a forced continuation.</p>
  </details>;
}

export function EvaluationPanel({ analysis, loading, error, retry, observation, playerId }: { analysis: Analysis | null; loading: boolean; error: string; retry: () => void; observation: Observation | null; playerId: number }) {
  const evaluation = analysis?.evaluation;
  const hasWdl = evaluation && evaluation.winProbability !== null && evaluation.lossProbability !== null && evaluation.drawProbability !== null;
  const hasRollouts = analysis?.alternatives.some((alternative) => alternative.visits > 0) ?? false;
  const expectedResult = evaluation?.expectedResult;
  const score = evaluation?.score;
  const meterScore = score !== null && score !== undefined ? Math.max(-6, Math.min(6, score)) : null;
  const resourceUnitLabel = evaluation?.status === 'trained' ? 'Learned advantage units' : 'Untrained resource index';
  return <aside className="analysis-panel" aria-label="Position analysis" aria-busy={loading}>
    <div className="panel-heading"><h2>Analysis</h2>{evaluation && <span className={`status-badge ${evaluation.status === 'trained' ? 'trained' : ''}`}>{evaluation.status === 'heuristic' ? 'Heuristic baseline' : humanize(evaluation.status)}</span>}</div>
    {!observation ? <div className="analysis-empty"><p>Position analysis appears here.</p><p className="muted">Run a game or open a saved replay to inspect resource balance, possible actions, and opponent beliefs.</p></div> : <>
      <p className="analysis-perspective">Player {playerId + 1} perspective · {observation.decisionPlayer === playerId ? 'To act' : 'Waiting for opponent'}</p>
      {error && <ErrorNotice message={error} retry={retry} />}
      {loading && !analysis && <div className="analysis-loading" role="status"><span className="skeleton" /><span className="skeleton short" /><p>Evaluating this position…</p></div>}
      {evaluation && <>
        <section className="evaluation-block"><div className="score-line"><h3>Resource advantage</h3><strong className="resource-score">{evaluation.score === null ? '—' : signed(evaluation.score)}</strong></div>
          {meterScore !== null && score !== null && score !== undefined && <div className="resource-meter-block"><p className="resource-meter-label">{resourceUnitLabel}</p><div className="resource-meter" role="meter" aria-label={resourceUnitLabel} aria-valuemin={-6} aria-valuemax={6} aria-valuenow={meterScore} aria-valuetext={`${signed(score)} ${resourceUnitLabel.toLowerCase()}${Math.abs(score) > 6 ? '; visual meter capped at the displayed range' : ''}`}><span className={`resource-meter-fill ${meterScore < 0 ? 'resource-meter-negative' : ''}`} style={{ left: `${meterScore < 0 ? 50 + meterScore / 6 * 50 : 50}%`, width: `${Math.abs(meterScore) / 6 * 50}%` }} /><span className="resource-meter-zero" /></div><div className="resource-meter-ticks" aria-hidden="true"><span>−6</span><span>0</span><span>+6</span></div>{Math.abs(score) > 6 && <p className="resource-meter-cap">Score exceeds the displayed range.</p>}</div>}
          <p className="small-copy">{evaluation.description}</p>
          <p className="probability-label">Winning odds</p>
          <div className="odds-bar" role="img" aria-label={hasWdl ? `Win ${percent(evaluation.winProbability!)}, draw ${percent(evaluation.drawProbability!)}, loss ${percent(evaluation.lossProbability!)}` : 'Win probability is unavailable; the resource score is not a calibrated probability.'}>
            {hasWdl ? <><span className="odds-win" style={{ width: `${evaluation.winProbability! * 100}%` }} /><span className="odds-draw" style={{ width: `${evaluation.drawProbability! * 100}%` }} /><span className="odds-loss" style={{ width: `${evaluation.lossProbability! * 100}%` }} /></> : <span className="odds-unavailable" />}
          </div>
          {hasWdl ? <div className="odds-labels"><span>Win <strong>{percent(evaluation.winProbability!)}</strong></span><span>Draw <strong>{percent(evaluation.drawProbability!)}</strong></span><span>Loss <strong>{percent(evaluation.lossProbability!)}</strong></span></div> : <p className="unavailable-note">Win probability unavailable</p>}
          {!hasWdl && expectedResult !== null && expectedResult !== undefined && <p className="small-copy">Expected result: {percent(expectedResult)} (a draw counts as half a win)</p>}
          <p className="calibration-note">{evaluation.calibrated ? 'Calibrated model estimate.' : 'Uncalibrated. This score is not a winning percentage.'}</p>
        </section>
        <section className="analysis-section"><h3>Resource balance</h3>{evaluation.components.length ? <dl className="resource-terms">{evaluation.components.map((component, index) => <div key={`${component.name}-${index}`}><dt>{humanize(component.name)}</dt><dd className={component.value > 0 ? 'positive' : component.value < 0 ? 'negative' : ''}>{signed(component.value)}</dd></div>)}</dl> : <p className="small-copy">No resource breakdown is available for this position.</p>}<p className="small-copy">Positive values favor the selected player. Terms describe this evaluator, not fixed card values.</p></section>
        <section className="analysis-section"><div className="section-heading"><h3>Decision review</h3>{analysis.decisionReview && <span className="status-badge">{humanize(analysis.decisionReview.status)}</span>}</div>{analysis.decisionReview ? <><div className="decision-review-value"><span>Estimated opportunity loss</span><strong>{analysis.decisionReview.opportunityLoss === null ? 'Unavailable' : `${(analysis.decisionReview.opportunityLoss * 100).toFixed(1)} pp`}</strong></div><p className="small-copy">{analysis.decisionReview.description}</p>{analysis.decisionReview.opportunityLoss !== null && <><p className="small-copy">Percentage points of expected result, relative to the best tested action. This is not a calibrated mistake grade or a penalty for a later random draw.</p>{analysis.decisionReview.differenceStandardError !== null && analysis.decisionReview.differenceStandardError !== undefined && <p className="small-copy">Estimated difference SE: ±{(analysis.decisionReview.differenceStandardError * 100).toFixed(1)} pp.</p>}</>}<WarningList warnings={analysis.decisionReview.warnings} label="Review caveats" /></> : <p className="small-copy">Unavailable. Decision review needs comparable estimates for the played action and tested alternatives.</p>}</section>
        <section className="analysis-section"><div className="section-heading"><h3>Candidate actions</h3><span className="subtle-count">{analysis.alternatives.length}</span></div>{hasRollouts && <p className="small-copy candidate-method">Approximate expected result: win = 1, draw = ½, loss = 0. Cutoff positions may use heuristic estimates; this is not calibrated winning probability.</p>}{analysis.search && <p className="small-copy candidate-method">Search: {humanize(analysis.search.status)}{analysis.search.description ? ` · ${analysis.search.description}` : ''}{analysis.search.iterations !== undefined ? ` · ${analysis.search.iterations} iterations` : ''}</p>}{analysis.alternatives.length ? <ol className="alternatives">{analysis.alternatives.map((alternative, index) => <li key={`${alternative.actionId}-${index}`}><div><span>{alternative.label}</span><strong>{alternative.visits > 0 ? (alternative.expectedResult !== null && alternative.expectedResult !== undefined ? percent(alternative.expectedResult) : '—') : alternative.score === null ? '—' : signed(alternative.score)}</strong></div><p>{alternative.description}</p>{alternative.visits > 0 && <small>{alternative.visits.toLocaleString()} sampled continuations · expected result{alternative.uncertainty !== null && alternative.uncertainty !== undefined ? ` · SE ±${percent(alternative.uncertainty)}` : ' · uncertainty unavailable'}</small>}{alternative.continuation && <Continuation line={alternative.continuation} />}</li>)}</ol> : <p className="small-copy">{observation.decisionPlayer !== playerId ? 'Switch to the acting player to inspect their legal actions.' : 'No candidate analysis is available for this decision.'}</p>}</section>
        <section className="analysis-section"><h3>Opponent beliefs</h3><p className="small-copy">Limited to the supported deck pool; not certainty about an opponent’s intent.</p>{analysis.beliefs.length ? <dl className="beliefs">{analysis.beliefs.map((belief) => <div key={belief.archetype}><dt>{belief.archetype}</dt><dd>{percent(belief.probability)}</dd></div>)}</dl> : <p className="small-copy">No opponent belief estimate is available.</p>}</section>
        <WarningList warnings={analysis.warnings} label="Analysis limitations" />
        <p className="model-version">Evaluator: {evaluation.modelVersion}</p>
      </>}
    </>}
  </aside>;
}

function ReplayApp() {
  const [decks, setDecks] = useState<Deck[]>([]);
  const [replays, setReplays] = useState<ReplaySummary[]>([]);
  const [positions, setPositions] = useState<PositionSummary[]>([]);
  const [activePosition, setActivePosition] = useState<SavedPosition | null>(null);
  const [selectedPositionId, setSelectedPositionId] = useState('');
  const [positionLibraryError, setPositionLibraryError] = useState('');
  const [saveError, setSaveError] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveNotice, setSaveNotice] = useState('');
  const [deckIds, setDeckIds] = useState<[string, string]>(['', '']);
  const [seed, setSeed] = useState('42');
  const [maxDecisions, setMaxDecisions] = useState('2000');
  const [policy, setPolicy] = useState<'heuristic' | 'random'>('heuristic');
  const [initialLoading, setInitialLoading] = useState(true);
  const [connectionError, setConnectionError] = useState('');
  const [libraryError, setLibraryError] = useState('');
  const [reload, setReload] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [jobError, setJobError] = useState('');
  const [jobRetry, setJobRetry] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [selectedReplayId, setSelectedReplayId] = useState('');
  const [replay, setReplay] = useState<Replay | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayError, setReplayError] = useState('');
  const [frameIndex, setFrameIndex] = useState(0);
  const [playerId, setPlayerId] = useState(0);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysisSourceKey, setAnalysisSourceKey] = useState('');
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState('');
  const [analysisRetry, setAnalysisRetry] = useState(0);
  const [lastAnnouncement, setLastAnnouncement] = useState('');
  const replayRequest = useRef<AbortController | null>(null);
  const cache = useRef(new Map<string, Analysis>());
  const selectedAction = useRef<HTMLButtonElement | null>(null);

  const loadReplay = useCallback(async (id: string) => {
    replayRequest.current?.abort();
    const controller = new AbortController();
    replayRequest.current = controller;
    setSelectedReplayId(id);
    setSelectedPositionId('');
    setActivePosition(null);
    setSaveNotice('');
    setReplay(null);
    setAnalysis(null);
    setReplayError('');
    setFrameIndex(0);
    if (!id) { setReplayLoading(false); return; }
    setReplayLoading(true);
    try {
      const result = await api<Replay>(`/replays/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (!result.frames.length) throw new Error('This replay contains no decision frames. Run another game to create a replay.');
      setReplay(result);
      setLastAnnouncement(`Replay loaded. ${result.frames.length} positions.`);
      return result;
    } catch (error) {
      if (!controller.signal.aborted) setReplayError(errorMessage(error));
    } finally {
      if (!controller.signal.aborted) setReplayLoading(false);
    }
  }, []);

  const loadPosition = useCallback(async (id: string) => {
    replayRequest.current?.abort();
    const controller = new AbortController();
    replayRequest.current = controller;
    setSelectedPositionId(id);
    setSelectedReplayId('');
    setReplay(null);
    setActivePosition(null);
    setAnalysis(null);
    setReplayError('');
    setSaveNotice('');
    setFrameIndex(0);
    if (!id) { setReplayLoading(false); return; }
    setReplayLoading(true);
    try {
      const position = await api<SavedPosition>(`/positions/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (position.observation.playerId !== position.playerId) throw new Error('The saved position has an inconsistent player perspective. Open the source replay instead.');
      setActivePosition(position);
      setPlayerId(position.playerId);
      setLastAnnouncement(`Saved position loaded: ${position.title}.`);
    } catch (error) {
      if (!controller.signal.aborted) setReplayError(errorMessage(error));
    } finally {
      if (!controller.signal.aborted) setReplayLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setInitialLoading(true);
    setConnectionError('');
    setLibraryError('');
    setPositionLibraryError('');
    Promise.allSettled([
      api<{ decks: Deck[] }>('/decks', { signal: controller.signal }),
      api<{ replays: ReplaySummary[] }>('/replays', { signal: controller.signal }),
      api<{ positions: PositionSummary[] }>('/positions', { signal: controller.signal }),
    ]).then(([deckResult, replayResult, positionResult]) => {
      if (controller.signal.aborted) return;
      if (deckResult.status === 'fulfilled') {
        setDecks(deckResult.value.decks);
        const available = deckResult.value.decks.filter((deck) => deck.playable !== false && deck.support?.playable !== false);
        setDeckIds((current) => [current[0] || available[0]?.id || '', current[1] || available[1]?.id || available[0]?.id || '']);
      } else setConnectionError(errorMessage(deckResult.reason));
      if (replayResult.status === 'fulfilled') setReplays(replayResult.value.replays);
      else setLibraryError(errorMessage(replayResult.reason));
      if (positionResult.status === 'fulfilled') setPositions(positionResult.value.positions);
      else setPositionLibraryError(errorMessage(positionResult.reason));
      setInitialLoading(false);
    });
    return () => controller.abort();
  }, [reload]);

  useEffect(() => () => replayRequest.current?.abort(), []);

  const jobTerminal = job ? ['completed', 'complete', 'finished', 'failed', 'error', 'cancelled', 'interrupted'].includes(job.status) || Boolean(job.replayId) : true;
  const running = submitting || Boolean(job && !jobTerminal);

  useEffect(() => {
    if (!job || jobTerminal) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setJobError('');
    const poll = async () => {
      try {
        const next = await api<Job>(`/jobs/${encodeURIComponent(job.id)}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        if (next.replayId) {
          await loadReplay(next.replayId);
          try {
            const saved = await api<{ replays: ReplaySummary[] }>('/replays', { signal: controller.signal });
            setReplays(saved.replays);
            setLibraryError('');
          } catch (error) {
            if (!controller.signal.aborted) setLibraryError(errorMessage(error));
          }
          if (controller.signal.aborted) return;
          setJob(next);
          setLastAnnouncement('Game run complete. Replay is ready.');
          return;
        }
        setJob(next);
        if (['failed', 'error', 'cancelled', 'interrupted'].includes(next.status)) {
          const message = typeof next.error === 'string' ? next.error : next.error?.message;
          setJobError(message || 'The game could not finish. Check the engine, then run another game.');
          return;
        }
        if (['completed', 'complete', 'finished'].includes(next.status)) {
          setJobError('The run finished without a replay. Check the local engine logs, then try another run.');
          return;
        }
        timer = setTimeout(poll, 1000);
      } catch (error) {
        if (!controller.signal.aborted) setJobError(errorMessage(error));
      }
    };
    timer = setTimeout(poll, 300);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [job?.id, jobTerminal, jobRetry, loadReplay]);

  const frame = replay?.frames[frameIndex];
  const observation = activePosition?.observation ?? frame?.observations[playerId] ?? null;
  const analysisKey = activePosition ? `position:${activePosition.id}` : replay && frame ? `${replay.id}:${frame.decisionIndex}:${playerId}` : '';

  useEffect(() => {
    setAnalysisError('');
    if (!activePosition && (!replay || !frame)) { setAnalysis(null); setAnalysisLoading(false); return; }
    const cached = cache.current.get(analysisKey);
    if (cached) { setAnalysis(cached); setAnalysisSourceKey(analysisKey); setAnalysisLoading(false); return; }
    const controller = new AbortController();
    setAnalysis(null);
    setAnalysisLoading(true);
    const timer = setTimeout(async () => {
      try {
        const result = await api<Analysis>('/analyze', {
          method: 'POST', signal: controller.signal,
          body: JSON.stringify(activePosition ? { positionId: activePosition.id, budgetMs: 300 } : { replayId: replay!.id, decisionIndex: frame!.decisionIndex, playerId, budgetMs: 300 }),
        });
        if (controller.signal.aborted) return;
        cache.current.set(analysisKey, result);
        setAnalysis(result);
        setAnalysisSourceKey(analysisKey);
      } catch (error) {
        if (!controller.signal.aborted) setAnalysisError(errorMessage(error));
      } finally {
        if (!controller.signal.aborted) setAnalysisLoading(false);
      }
    }, 220);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [analysisKey, analysisRetry]);

  const navigate = useCallback((next: number) => {
    if (!replay) return;
    setFrameIndex(Math.max(0, Math.min(replay.frames.length - 1, next)));
  }, [replay]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (!replay || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target as HTMLElement;
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName) || target.isContentEditable) return;
      const targetIndex = event.key === 'ArrowLeft' ? frameIndex - 1 : event.key === 'ArrowRight' ? frameIndex + 1 : event.key === 'Home' ? 0 : event.key === 'End' ? replay.frames.length - 1 : null;
      if (targetIndex !== null) { event.preventDefault(); navigate(targetIndex); }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [frameIndex, navigate, replay]);

  useEffect(() => {
    const button = selectedAction.current;
    const list = button?.parentElement;
    if (button && list) {
      if (button.offsetTop < list.scrollTop) list.scrollTop = button.offsetTop;
      else if (button.offsetTop + button.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = button.offsetTop + button.offsetHeight - list.clientHeight;
    }
  }, [frameIndex]);

  async function runGame(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setJobError('');
    setJob(null);
    try {
      const next = await api<Job>('/games', { method: 'POST', body: JSON.stringify({ decks: deckIds, seed: Number(seed), maxDecisions: Number(maxDecisions), policy }) });
      setJob(next);
      if (next.replayId) void loadReplay(next.replayId);
    } catch (error) { setJobError(errorMessage(error)); }
    finally { setSubmitting(false); }
  }

  async function savePosition() {
    if (!replay || !frame || !observation) return;
    setSaving(true);
    setSaveError('');
    setSaveNotice('');
    try {
      await api<PositionSummary>('/positions', { method: 'POST', body: JSON.stringify({ replayId: replay.id, decisionIndex: frame.decisionIndex, playerId, title: `P${playerId + 1} · Turn ${observation.turn} · Decision ${frame.decisionIndex}` }) });
      const saved = await api<{ positions: PositionSummary[] }>('/positions');
      setPositions(saved.positions);
      setSaveNotice('Position saved with this player’s information only.');
      setLastAnnouncement('Position saved. Open it from the saved position selector.');
    } catch (error) { setSaveError(errorMessage(error)); }
    finally { setSaving(false); }
  }

  const deckName = (id: string) => decks.find((deck) => deck.id === id)?.name ?? humanize(id);
  const selectedDecks = deckIds.map((id) => decks.find((deck) => deck.id === id)).filter((deck): deck is Deck => Boolean(deck));
  const deckWarnings = selectedDecks.flatMap((deck) => [
    ...(deck.validation?.status !== 'validated' ? [`${deck.name}: ${deck.validation?.status || 'unverified'} deck. ${deck.validation?.notes?.join(' ') || ''}`] : []),
    ...(deck.warnings ?? []), ...(deck.support?.warnings ?? []), ...(deck.support?.errors ?? []),
  ]);
  const ownPlayer = observation?.players.find((player) => player.id === playerId);
  const opponent = observation?.players.find((player) => player.id !== playerId);
  const actionLabel = (item: Replay['frames'][number]) => !item.action ? 'Final recorded position' : item.actor !== playerId && ['prompt', 'choice'].includes(item.action.type) ? 'Opponent choice · private details hidden' : item.action.label;
  const resultLabel = !replay ? '' : replay.status === 'truncated' ? 'Decision limit reached · outcome unknown' : replay.status === 'error' ? 'Simulation error · inspect research notes' : replay.outcome?.winner !== null && replay.outcome?.winner !== undefined ? `Player ${replay.outcome.winner + 1} won · ${replay.outcome.reason}` : replay.outcome?.reason === 'rules-draw' ? 'Draw · rules terminal' : 'Finished · outcome unavailable';

  return <>
    <a className="skip-link" href="#replay-workspace">Skip to replay</a>
    <header className="app-header"><div className="app-identity"><svg className="app-mark" width="26" height="26" viewBox="0 0 26 26" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><rect x="3" y="5" width="14" height="18" rx="2" /><path d="M8 5V3h14v18h-5M7 11h6M7 15h4" /></svg><h1>TCG Engine Lab</h1></div><p>Local research · Experimental</p></header>
    <main>
      <section className="run-panel" aria-labelledby="run-heading">
        <div className="run-heading"><h2 id="run-heading">Run a game</h2><span className="small-copy">{initialLoading ? 'Connecting to local engine…' : decks.length ? `${decks.length} decks in the research pool` : 'Connect the local engine to begin'}</span></div>
        {connectionError && <ErrorNotice message={connectionError} retry={() => setReload((value) => value + 1)} />}
        <form className="run-form" onSubmit={runGame}>
          {[0, 1].map((index) => <label className="deck-field" key={index}>Player {index + 1}<select value={deckIds[index]} required disabled={initialLoading || running || !decks.length} onChange={(event) => setDeckIds((current) => index === 0 ? [event.target.value, current[1]] : [current[0], event.target.value])}>
            {!decks.length && <option value="">{initialLoading ? 'Loading decks…' : 'No decks available'}</option>}
            {decks.map((deck) => <option key={deck.id} value={deck.id} disabled={deck.playable === false || deck.support?.playable === false}>{deck.name}{deck.playable === false || deck.support?.playable === false ? ' · unsupported' : ''}</option>)}
          </select></label>)}
          <label className="number-field">Seed<input type="number" required min="0" max="4294967295" step="1" value={seed} disabled={running} onChange={(event) => setSeed(event.target.value)} /></label>
          <label className="number-field limit-field">Decision limit<input type="number" required min="1" max="3000" step="1" value={maxDecisions} disabled={running} onChange={(event) => setMaxDecisions(event.target.value)} /></label>
          <label className="policy-field">Policy<select value={policy} disabled={running} onChange={(event) => setPolicy(event.target.value as 'heuristic' | 'random')}><option value="heuristic">Heuristic</option><option value="random">Random legal</option></select></label>
          <button className="primary run-button" type="submit" disabled={initialLoading || running || !deckIds[0] || !deckIds[1]}>{running ? 'Running game…' : 'Run game'}</button>
        </form>
        <div className="run-footer"><WarningList warnings={deckWarnings} label="Deck support notes" />{running && <p className="run-status" role="status">{jobError ? 'Run status unavailable.' : 'Simulating legal decisions. The replay will open when ready.'}{job?.progress !== undefined ? ` ${typeof job.progress === 'number' ? job.progress : job.progress}` : ''}</p>}</div>
        {jobError && <ErrorNotice message={jobError} retry={job && !jobTerminal ? () => setJobRetry((value) => value + 1) : undefined} />}
      </section>

      <div className="workspace-toolbar"><label className="replay-picker">Saved replay<select value={selectedReplayId} disabled={initialLoading || replayLoading} onChange={(event) => void loadReplay(event.target.value)}><option value="">{replays.length ? 'Choose a replay' : 'No saved replays yet'}</option>{replays.map((saved) => <option key={saved.id} value={saved.id}>{saved.decks.map(deckName).join(' vs ')} · {saved.status} · {saved.id.slice(-8)}</option>)}</select></label>
        <label className="position-picker">Saved position<select value={selectedPositionId} disabled={initialLoading || replayLoading} onChange={(event) => void loadPosition(event.target.value)}><option value="">{positions.length ? 'Choose a position' : 'No saved positions yet'}</option>{positions.map((saved) => <option key={saved.id} value={saved.id}>{saved.title}</option>)}</select></label>
        <button type="button" className="secondary refresh-button" disabled={initialLoading} onClick={() => { cache.current.clear(); setAnalysis(null); setReload((value) => value + 1); setAnalysisRetry((value) => value + 1); }}>Refresh</button>
        <label className="perspective-picker">Perspective<select value={playerId} disabled={Boolean(activePosition) || Boolean(selectedPositionId && replayLoading)} onChange={(event) => setPlayerId(Number(event.target.value))}><option value="0">Player 1</option><option value="1">Player 2</option></select></label>
      </div>
      {libraryError && <ErrorNotice message={`Could not load saved replays. ${libraryError}`} retry={() => setReload((value) => value + 1)} />}
      {positionLibraryError && <ErrorNotice message={`Could not load saved positions. ${positionLibraryError}`} retry={() => setReload((value) => value + 1)} />}
      {replayError && <ErrorNotice message={replayError} retry={() => selectedPositionId ? void loadPosition(selectedPositionId) : void loadReplay(selectedReplayId)} />}

      <div className="workspace" id="replay-workspace" tabIndex={-1}>
        <div className="replay-column">
          <section className="board-panel" aria-labelledby="position-heading" aria-busy={replayLoading}>
            <div className="panel-heading"><h2 id="position-heading">{observation ? `Turn ${observation.turn}` : 'Replay board'}</h2><span className="position-phase">{observation ? humanize(observation.phase) : 'Player information view'}</span></div>
            {replayLoading ? <div className="board-loading" role="status"><div className="skeleton" /><div className="skeleton short" /><p>Loading {selectedPositionId ? 'the saved position' : 'the replay'}…</p></div> : observation && ownPlayer && opponent ? <>
              <div className="view-notice">{activePosition ? 'Saved position · Contains only the selected player’s information. The other private view and future decisions are not included.' : 'Research replay · Each perspective reveals only that player’s hand. Both private views are stored locally.'}</div>
              <PlayerBoard player={opponent} own={false} deciding={observation.decisionPlayer === opponent.id} deckName="Opponent" />
              <div className="board-divider"><span>{observation.prompt?.message || (activePosition ? `Player ${observation.decisionPlayer + 1} to act` : frame?.action ? `Player ${(frame.actor ?? 0) + 1} to act` : 'Final recorded position')}</span></div>
              <PlayerBoard player={ownPlayer} own deciding={observation.decisionPlayer === ownPlayer.id} deckName={replay ? deckName(replay.decks[ownPlayer.id]) : 'Selected player'} />
            </> : <div className="board-empty"><div className="empty-board-diagram" aria-hidden="true"><span /><span /><span /></div><h3>A position worth understanding.</h3><p>Choose two decks and run a game. Step through the real decisions to see what each player knew and how the engine evaluates the position.</p><p className="small-copy">Games and analysis run on your computer.</p></div>}
          </section>

          <section className="timeline-panel" aria-labelledby="timeline-heading"><div className="panel-heading"><h2 id="timeline-heading">{activePosition ? 'Saved position' : 'Decisions'}</h2><div className="timeline-heading-actions"><span className="timeline-position">{replay ? `${frameIndex + 1} / ${replay.frames.length}` : activePosition ? `Decision ${activePosition.decisionIndex}` : 'No replay loaded'}</span>{replay && <button type="button" className="secondary small" disabled={saving || !frame} onClick={() => void savePosition()}>{saving ? 'Saving…' : 'Save position'}</button>}</div></div>
            {!activePosition && <div className="timeline-controls"><div className="step-buttons"><button type="button" className="icon-button" aria-label="First position" title="First position (Home)" disabled={!replay || frameIndex === 0} onClick={() => navigate(0)}><Arrow direction="left" end /></button><button type="button" className="icon-button" aria-label="Previous decision" title="Previous decision (Left arrow)" disabled={!replay || frameIndex === 0} onClick={() => navigate(frameIndex - 1)}><Arrow direction="left" /></button><button type="button" className="icon-button" aria-label="Next decision" title="Next decision (Right arrow)" disabled={!replay || frameIndex === replay.frames.length - 1} onClick={() => navigate(frameIndex + 1)}><Arrow direction="right" /></button><button type="button" className="icon-button" aria-label="Last position" title="Last position (End)" disabled={!replay || frameIndex === replay.frames.length - 1} onClick={() => navigate((replay?.frames.length ?? 1) - 1)}><Arrow direction="right" end /></button></div>
              <label className="timeline-range"><span className="sr-only">Replay position</span><input type="range" min="0" max={Math.max(1, (replay?.frames.length ?? 1) - 1)} value={frameIndex} disabled={!replay || replay.frames.length < 2} aria-valuetext={frame ? `Position ${frameIndex + 1}: ${actionLabel(frame)}` : 'No replay loaded'} onChange={(event) => navigate(Number(event.target.value))} /></label>
            </div>}
            {replay ? <><p className="replay-result">{resultLabel}</p><div className="action-list" aria-label="Replay decisions">{replay.frames.map((item, index) => <button type="button" key={`${item.decisionIndex}-${index}`} ref={index === frameIndex ? selectedAction : null} className={`action-row ${index === frameIndex ? 'selected' : ''}`} aria-current={index === frameIndex ? 'step' : undefined} onClick={() => navigate(index)}><span className="action-index">{index + 1}</span><span className="action-player">{item.action ? `P${item.actor + 1}` : 'End'}</span><span>{actionLabel(item)}</span></button>)}</div>
              <p className="timeline-help">Positions are shown before the selected action. Opponent choice details are hidden. Use Left / Right arrows to step.</p>
              <WarningList warnings={[...replay.warnings, ...(observation?.warnings ?? [])]} label="Replay limitations" />
              <details className="replay-metadata"><summary>Replay details</summary><dl><div><dt>Seed</dt><dd>{replay.seed}</dd></div><div><dt>Engine</dt><dd>{replay.engineVersion}</dd></div><div><dt>Replay</dt><dd>{replay.id}</dd></div><div><dt>Status</dt><dd>{replay.status}</dd></div></dl></details>
            </> : activePosition ? <div className="saved-position-details"><h3>{activePosition.title}</h3><p>This saved observation has no timeline or second private view.</p><button type="button" className="secondary small" onClick={async () => { const saved = activePosition; const source = await loadReplay(saved.sourceReplayId); if (source) setFrameIndex(Math.max(0, source.frames.findIndex((item) => item.decisionIndex === saved.decisionIndex))); }}>Open source replay</button></div> : <p className="timeline-empty">Every decision will appear here, including the final recorded position.</p>}
            {saveNotice && <p className="save-notice" role="status">{saveNotice}</p>}
            {saveError && <ErrorNotice message={saveError} retry={() => void savePosition()} />}
          </section>
        </div>
        <EvaluationPanel analysis={analysisSourceKey === analysisKey ? analysis : null} loading={analysisLoading} error={analysisError} retry={() => setAnalysisRetry((value) => value + 1)} observation={observation} playerId={playerId} />
      </div>
    </main>
    <footer className="app-footer">Experimental engine analysis. Hidden information stays hidden within each player view; uncertainty and unsupported behavior are reported.</footer>
    <div className="sr-only" role="status" aria-live="polite">{lastAnnouncement}</div>
  </>;
}

export default function App() {
  const [view, setView] = useState(() => window.location.hash === '#play' ? 'play' : window.location.hash === '#review' ? 'review' : 'replays');
  return <><nav className="surface-tabs" aria-label="Research tools">{[['play','Play'],['replays','Replay analysis'],['review','Teaching review']].map(([id,label]) => <button key={id} type="button" aria-current={view === id ? 'page' : undefined} onClick={() => {setView(id); window.location.hash=id;}}>{label}</button>)}</nav>{view === 'play' ? <Play /> : view === 'review' ? <TeachingReview /> : <ReplayApp />}</>;
}
