import { useCallback, useEffect, useRef, useState } from 'react';
import { api, errorMessage } from './api';
import type { LearningGame, LearningRun } from './types';

const active=(run:LearningRun)=>run.status!=='stopped';
const label=(value:string)=>value.replaceAll('-',' ');
const bytes=(value:number)=>`${(value/1024**3).toFixed(1)} GiB`;
const percent=(value:number|null|undefined)=>typeof value==='number'?`${(value*100).toFixed(1)}%`:'Unavailable';
export function useLearningRuns(){
 const [runs,setRuns]=useState<LearningRun[]>([]),[run,setRun]=useState<LearningRun|null>(null),[games,setGames]=useState<LearningGame[]>([]);
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),[reload,setReload]=useState(0),[ready,setReady]=useState(false);
 const [selectedId,setSelectedId]=useState(()=>{try{return sessionStorage.getItem('ptcg-learning-run')??'';}catch{return '';}});
 const pending=useRef<{path:string;body:unknown}|null>(null),mounted=useRef(true);
 const accept=useCallback((value:LearningRun)=>{setRun(current=>current?.id===value.id&&current.revision>value.revision?current:value);setRuns(current=>{const newer=current.find(r=>r.id===value.id&&r.revision>value.revision);return [newer??value,...current.filter(r=>r.id!==value.id)];});},[]);
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
 useEffect(()=>{let alive=true;api<{runs:LearningRun[]}>('/learning/runs').then(result=>{if(!alive)return;setRuns(result.runs);setReady(true);setError('');setSelectedId(current=>result.runs.some(r=>r.id===current)?current:result.runs.find(active)?.id??result.runs[0]?.id??'');}).catch(e=>{if(alive){setError(errorMessage(e));setReady(true);}});return()=>{alive=false;};},[reload]);
 useEffect(()=>{
  if(!selectedId){setRun(null);setGames([]);return;}
  try{sessionStorage.setItem('ptcg-learning-run',selectedId);}catch{/* Read-only browsers can still watch runs. */}
  let alive=true,timer:number|undefined;const controller=new AbortController();
  setRun(current=>current?.id===selectedId?current:null);setGames([]);
  async function poll(){
   try{const [value,history]=await Promise.all([api<LearningRun>(`/learning/runs/${selectedId}`,{signal:controller.signal}),api<{games:LearningGame[]}>(`/learning/runs/${selectedId}/games`,{signal:controller.signal})]);if(!alive)return;accept(value);setGames(history.games);if(!pending.current)setError('');}
   catch(e){if(alive)setError(errorMessage(e));}
   finally{if(alive)timer=window.setTimeout(()=>void poll(),1500);}
  }
  void poll();return()=>{alive=false;controller.abort();if(timer!==undefined)clearTimeout(timer);};
 },[selectedId,reload,accept]);
 async function send(path:string,body:unknown){
  setBusy(true);setError('');pending.current={path,body};
  try{const value=await api<LearningRun>(path,{method:'POST',body:JSON.stringify(body)});if(mounted.current){accept(value);setSelectedId(value.id);pending.current=null;}return value;}
  catch(e){if(mounted.current)setError(errorMessage(e));return null;}
  finally{if(mounted.current)setBusy(false);}
 }
 async function start(seed:number,keepAwake:boolean){return send('/learning/runs',{seed,keepAwake});}
 async function control(operation:'pause'|'resume'|'stop'){if(!run)return null;return send(`/learning/runs/${run.id}/control/${operation}`,{revision:run.revision,requestId:crypto.randomUUID()});}
 function retry(){if(pending.current)void send(pending.current.path,pending.current.body);else setReload(v=>v+1);}
 return {runs,run,games,error,busy,ready,start,control,retry,select:setSelectedId,refresh:()=>{pending.current=null;setReload(v=>v+1);},hasActive:runs.some(active)};
}
export type LearningState=ReturnType<typeof useLearningRuns>;
export function LearningControls({state,onStart,onWatch,selectedGameId}:{state:LearningState;onStart:(run:LearningRun)=>void;onWatch:(game:LearningGame,follow:boolean)=>void;selectedGameId?:string}){
 const [seed,setSeed]=useState('42'),[keepAwake,setKeepAwake]=useState(true),[settings,setSettings]=useState(false);
 const {run,busy}=state;
 const liveRun=run&&active(run);
 return <section className="learning-controls" aria-label="Continuous learning">
  <div className="learning-control-row"><h2>Continuous learning</h2><span className="status-badge">Experimental</span>{run&&<span className="learning-state" role="status">{label(run.status)} · {label(run.phase)} · cycle {run.cycle}</span>}
   <div className="learning-buttons">{!state.hasActive&&<button className="primary" disabled={busy||!state.ready||Boolean(state.error)||!Number.isInteger(Number(seed))||Number(seed)<0||Number(seed)>4294967295||!seed.trim()} onClick={async()=>{const value=await state.start(Number(seed),keepAwake);if(value){setSettings(false);onStart(value);}}}>{busy?'Starting…':'Start learning'}</button>}
   {state.hasActive&&!liveRun&&<button onClick={()=>{const current=state.runs.find(active);if(current)state.select(current.id);}}>View active run</button>}
   {liveRun&&['running','waiting-for-play'].includes(run.status)&&<button disabled={busy} onClick={()=>void state.control('pause')}>Pause learning</button>}
   {run&&run.status==='paused'&&<button className="primary" disabled={busy} onClick={()=>void state.control('resume')}>Resume learning</button>}
   {liveRun&&<button disabled={busy} onClick={()=>void state.control('stop')}>Stop learning</button>}
   <button aria-expanded={settings} onClick={()=>setSettings(v=>!v)}>Run settings</button></div>
  </div>
  {settings&&<div className="learning-settings">{!liveRun&&<label>Learning seed<input type="number" min={0} max={4294967295} required value={seed} disabled={Boolean(liveRun)} onChange={e=>setSeed(e.target.value)}/></label>}<label className="checkbox-label"><input type="checkbox" checked={liveRun?Boolean(run.configuration.keepAwake):keepAwake} disabled={Boolean(liveRun)} onChange={e=>setKeepAwake(e.target.checked)}/>Keep this computer awake during learning</label><p>Runs continue with the browser closed. Pause or stop learning to end background work.</p>{state.runs.length>0&&<label>Learning run<select value={run?.id??''} onChange={e=>state.select(e.target.value)}>{state.runs.map(r=><option value={r.id} key={r.id}>{r.id.slice(0,8)} · {label(r.status)} · cycle {r.cycle}</option>)}</select></label>}</div>}
  {state.error&&<div className="error-notice" role="alert"><p>{state.error}</p><button disabled={busy} onClick={state.retry}>Retry learning request</button><button disabled={busy} onClick={state.refresh}>Reload run status</button></div>}
  {run?.error&&<p className="learning-problem" role="alert">{run.error}</p>}{run?.pauseReason&&<p className="learning-problem">{run.pauseReason}</p>}
  {run&&<div className="learning-watch-row"><span>{run.metrics.completedGames.toLocaleString()} completed games</span>{[0,1].map(worker=>{const game=run.activeGames.find(g=>g.workerIndex===worker);return <button className={game?.id===selectedGameId?'selected-worker':''} key={worker} disabled={!game} onClick={()=>game&&onWatch(game,true)}>{game?`Watch worker ${worker+1} · ${game.purpose}`:`Worker ${worker+1} · waiting`}</button>;})}<label>Recorded learning game<select value={state.games.some(g=>g.id===selectedGameId)?selectedGameId:''} onChange={e=>{const game=state.games.find(g=>g.id===e.target.value);if(game)onWatch(game,false);}}><option value="">Choose a game</option>{state.games.map(g=><option key={g.id} value={g.id}>{g.id.slice(0,8)} · {g.archetypes.join(' / ')} · {g.purpose} · {g.status}</option>)}</select></label></div>}
 </section>;
}
export function LearningProgress({run}:{run:LearningRun}){
 const {metrics}=run,comparison=metrics.comparison,agreement=metrics.guideAgreement;
 return <section className="learning-progress" aria-label="Learning progress"><div className="panel-heading"><h2>Learning progress</h2><span className="status-badge">Experimental</span></div>
  <div className="learning-progress-body"><p>Weights stay fixed during each game and update between batches. This run is not evidence of verified competitive strength.</p>
   <dl className="learning-metrics"><div><dt>Completed games</dt><dd>{metrics.completedGames}</dd></div><div><dt>Incomplete / errors</dt><dd>{metrics.truncatedGames} / {metrics.errors}</dd></div><div><dt>Search decisions / targets</dt><dd>{metrics.searchDecisions} / {metrics.searchTargets}</dd></div><div><dt>Training loss</dt><dd>{typeof metrics.loss==='number'?metrics.loss.toFixed(4):'Unavailable'}</dd></div></dl>
   <details open><summary>Matchup coverage</summary>{metrics.coverage?<p>{metrics.coverage.completedPairs} of {metrics.coverage.totalPairs} pairings completed; {metrics.coverage.scheduledPairs} scheduled across {metrics.coverage.archetypes} archetypes.</p>:<p>Coverage is not available yet.</p>}</details>
   <details open><summary>Candidate comparison</summary>{comparison?<><p>{label(comparison.status)} · {comparison.completedGames} / {comparison.totalGames} games</p><p>Mean match result: {percent(comparison.mean)}{typeof comparison.lower==='number'&&typeof comparison.upper==='number'?` · 95% interval ${percent(comparison.lower)} to ${percent(comparison.upper)}`:''}</p><p>Win = 1, draw = 0.5, loss = 0. A 50% mean is even.</p><p>{comparison.adopted?'Candidate adopted as the experimental incumbent.':'No candidate adoption reported.'}</p>{comparison.notes.map((note,i)=><p key={i}>{note}</p>)}</>:<p>No comparison result yet.</p>}</details>
   <details><summary>Guide agreement</summary><p>{agreement?.status==='measured'?`${percent(agreement.acceptableActionAccuracy)} acceptable-action agreement across ${agreement.positions} positions.`:'Guide agreement is unavailable.'}</p><p>Guide agreement measures imitation separately from playing strength.</p></details>
   <details><summary>Checkpoints and resources</summary><p>Incumbent: {run.incumbent.name} · {run.incumbent.version}</p>{run.candidate&&<p>Candidate: {run.candidate.name} · {run.candidate.version}</p>}{run.guidePolicy&&<p>Guide initialization: {run.guidePolicy.name} · {run.guidePolicy.version}</p>}<p>Peak memory {bytes(metrics.peakMemoryBytes)} · managed data {bytes(metrics.managedBytes)}{typeof metrics.freeBytes==='number'?` · free space ${bytes(metrics.freeBytes)}`:''}</p></details>
  </div>
 </section>;
}
