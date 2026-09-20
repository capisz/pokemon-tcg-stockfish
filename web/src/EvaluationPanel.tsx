import type { Analysis, Observation, SampledContinuation } from './types';
const signed = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const humanize = (value: string) => value.replace(/[_-]+/g, ' ').replace(/^./, (letter) => letter.toUpperCase());

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
