import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api, errorMessage } from './api';
import { LearningControls, LearningProgress, useLearningRuns } from './Learning';
import { CardInspector, CardTable } from './Board';
import { EvaluationPanel } from './EvaluationPanel';
import { PlaybackControls, useFrameFeed, usePlayback, usePlaybackKeys } from './Playback';
import type { Analysis, Card, Deck, Job, LearningGame, ModelOption, PositionSummary, Replay, ReplaySummary, SavedPosition, ViewFrame } from './types';

type Source = {kind:'job'|'replay'|'position'|'learning';id:string;runId?:string;workerIndex?:number;playbackPaused?:boolean};
function rememberedSource():Source|null{try{const value=JSON.parse(sessionStorage.getItem('ptcg-watch-source')??'null');return value&&['job','replay','position','learning'].includes(value.kind)&&typeof value.id==='string'&&/^[a-zA-Z0-9_-]+$/.test(value.id)?value:null;}catch{return null;}}
function rememberedPerspective(){try{return sessionStorage.getItem('ptcg-watch-perspective')==='1'?1:0;}catch{return 0;}}
function normalize(replay:Replay):ViewFrame[]{
 return replay.frames.map((frame,index)=>({...frame,cursor:index,priorAction:frame.priorAction??(index?replay.frames[index-1].action:null),actor:index?replay.frames[index-1].actor:frame.actor}));
}
export function Watch(){
 const [decks,setDecks]=useState<Deck[]>([]),[models,setModels]=useState<ModelOption[]>([]),[replays,setReplays]=useState<ReplaySummary[]>([]),[positions,setPositions]=useState<PositionSummary[]>([]);
 const [deckIds,setDeckIds]=useState<[string,string]>(['','']),[policy,setPolicy]=useState('heuristic'),[seed,setSeed]=useState('42'),[limit,setLimit]=useState('2000'),[laboratory,setLaboratory]=useState(false);
 const [source,setSource]=useState<Source|null>(rememberedSource),[perspective,setPerspective]=useState(rememberedPerspective),[replay,setReplay]=useState<Replay|null>(null),[position,setPosition]=useState<SavedPosition|null>(null),[savedFrames,setSavedFrames]=useState<ViewFrame[]>([]);
 const [error,setError]=useState(''),[loading,setLoading]=useState(false),[submitting,setSubmitting]=useState(false),[inspect,setInspect]=useState<Card|null>(null),[notice,setNotice]=useState('');
 const [analysis,setAnalysis]=useState<Analysis|null>(null),[analysisError,setAnalysisError]=useState(''),[analyzing,setAnalyzing]=useState(false),[analysisRetry,setAnalysisRetry]=useState(0),[reload,setReload]=useState(0);
 const learning=useLearningRuns();
 const [pendingLearning,setPendingLearning]=useState<string|null>(null);
 const streamed=source?.kind==='job'||source?.kind==='learning';
 const feed=useFrameFeed(source?.kind==='job'?`/jobs/${source.id}/frames?playerId=${perspective}`:source?.kind==='learning'?`/learning/games/${source.id}/frames?playerId=${perspective}`:null);
 const frames=streamed?feed.frames:savedFrames;
 const followingLearning=source?.kind==='learning'&&source.workerIndex!==undefined&&learning.run?.id===source.runId&&!['stopped','failed'].includes(learning.run?.status??'stopped');
 const live=Boolean(followingLearning||(streamed&&!['finished','truncated','completed','error','failed','interrupted','stopped'].includes(feed.status)));
 const basePlayback=usePlayback(frames,`${source?.kind}:${source?.id}:${perspective}`,live,source?.kind==='learning'?!source.playbackPaused&&source.workerIndex!==undefined:undefined);
 const playback=source?.kind==='learning'?{...basePlayback,
  seek:(index:number)=>{setSource({...source,playbackPaused:true});basePlayback.seek(index);},
  turn:(direction:number)=>{setSource({...source,playbackPaused:true});basePlayback.turn(direction);},
  toggle:()=>{setSource({...source,playbackPaused:basePlayback.playing});basePlayback.toggle();},
  latest:()=>{setSource({...source,playbackPaused:false});basePlayback.latest();},
 }:basePlayback;
 usePlaybackKeys(playback);
 const observation=playback.frame?.observation??null;
 const replayId=source?.kind==='replay'?source.id:streamed?(replay?.id??feed.replayId):undefined;
 const selectedModel=models.find(m=>m.id===policy);
 const policyContext=streamed?feed.policyContext:replay?.policyContext;
 const watchedLearningGame=source?.kind==='learning'?learning.games.find(g=>g.id===source.id)??learning.run?.activeGames.find(g=>g.id===source.id):undefined;
 const learningReplayReady=source?.kind==='learning'&&(watchedLearningGame?.replayAvailable||Boolean(feed.replayId));
 function watchLearning(game:LearningGame,follow:boolean){
  setPendingLearning(null);setNotice('');
  if(source?.kind==='learning'&&source.id===game.id&&follow){playback.latest();setSource({...source,runId:learning.run?.id,workerIndex:game.workerIndex,playbackPaused:false});}
  else setSource({kind:'learning',id:game.id,runId:learning.run?.id,playbackPaused:!follow,...(follow?{workerIndex:game.workerIndex}:{})});
 }
 useEffect(()=>{
  const run=learning.run;if(!run)return;
  if(pendingLearning===run.id&&run.activeGames.length){watchLearning(run.activeGames[0],true);return;}
  if(source?.kind!=='learning'||source.runId!==run.id||source.workerIndex===undefined||!playback.playing||!playback.atLatest)return;
  const next=run.activeGames.find(game=>game.workerIndex===source.workerIndex);
  if(next&&next.id!==source.id)watchLearning(next,true);
 },[learning.run?.activeGames,source?.id,source?.workerIndex,pendingLearning,playback.playing,playback.atLatest]);
 useEffect(()=>{try{if(source)sessionStorage.setItem('ptcg-watch-source',JSON.stringify(source));else sessionStorage.removeItem('ptcg-watch-source');sessionStorage.setItem('ptcg-watch-perspective',String(perspective));}catch{/* Playback remains usable when storage is unavailable. */}},[source,perspective]);
 const loadLibraries=useCallback(async()=>{
  const [r,p]=await Promise.all([api<{replays:ReplaySummary[]}>('/replays'),api<{positions:PositionSummary[]}>('/positions')]);setReplays(r.replays);setPositions(p.positions);
 },[]);
 useEffect(()=>{let active=true;Promise.all([api<{decks:Deck[]}>('/decks'),api<{models:ModelOption[]}>('/models'),loadLibraries()]).then(([d,m])=>{if(!active)return;setDecks(d.decks);setModels(m.models);setDeckIds(current=>current[0]?current:[d.decks.find(d=>d.playable!==false&&d.role!=='heldout'&&d.role!=='historical')?.id??'',d.decks.find(d=>d.playable!==false&&d.role!=='heldout'&&d.role!=='historical')?.id??'']);setError('');}).catch(e=>{if(active)setError(errorMessage(e));});return()=>{active=false;};},[reload,loadLibraries]);
 useEffect(()=>{
  if(source?.kind==='learning'?!learningReplayReady:!feed.replayId)return;
  const controller=new AbortController();
  if(source?.kind!=='learning')void loadLibraries().catch(e=>setError(errorMessage(e)));
  const path=source?.kind==='learning'?`/learning/games/${source.id}/replay`:`/replays/${feed.replayId}`;
  api<Replay>(`${path}?playerId=${perspective}`,{signal:controller.signal}).then(setReplay).catch(e=>{if(!controller.signal.aborted)setError(errorMessage(e));});return()=>controller.abort();
 },[feed.replayId,learningReplayReady,source?.id,source?.kind,perspective,loadLibraries]);
 useEffect(()=>{
  setInspect(null);setAnalysis(null);setAnalysisError('');setPosition(null);setReplay(null);setSavedFrames([]);
  if(!source||source.kind==='job'||source.kind==='learning')return;
  const controller=new AbortController();setLoading(true);setError('');
  const request=source.kind==='replay'?api<Replay>(`/replays/${source.id}?playerId=${perspective}`,{signal:controller.signal}).then(value=>{setReplay(value);setSavedFrames(normalize(value));}):api<SavedPosition>(`/positions/${source.id}`,{signal:controller.signal}).then(value=>{setPosition(value);setPerspective(value.playerId);setSavedFrames([{decisionIndex:value.decisionIndex,actor:value.observation.decisionPlayer,observation:value.observation,priorAction:null}]);});
  request.catch(e=>{if(!controller.signal.aborted)setError(errorMessage(e));}).finally(()=>{if(!controller.signal.aborted)setLoading(false);});return()=>controller.abort();
 },[source?.kind,source?.id,perspective,reload]);
 useEffect(()=>{setInspect(null);},[playback.index,perspective,source?.id]);
 useEffect(()=>{
  setAnalysis(null);setAnalysisError('');setAnalyzing(false);
  if(!observation||playback.playing||(!replayId&&!position))return;
  const controller=new AbortController();
  const timer=window.setTimeout(()=>{
   setAnalyzing(true);
   api<Analysis>('/analyze',{method:'POST',signal:controller.signal,body:JSON.stringify(position?{positionId:position.id,budgetMs:300}:{replayId,decisionIndex:playback.frame!.decisionIndex,playerId:perspective,budgetMs:300})})
    .then(setAnalysis).catch(e=>{if(!controller.signal.aborted)setAnalysisError(errorMessage(e));}).finally(()=>{if(!controller.signal.aborted)setAnalyzing(false);});
  },250);
  return()=>{clearTimeout(timer);controller.abort();};
 },[observation,playback.playing,replayId,position?.id,perspective,analysisRetry]);
 async function run(event:FormEvent){
  event.preventDefault();setPendingLearning(null);setSubmitting(true);setError('');setNotice('');
  try{const job=await api<Job>('/games',{method:'POST',body:JSON.stringify({decks:deckIds,seed:Number(seed),maxDecisions:Number(limit),policy:policy==='heuristic'||policy==='random'?policy:'model',modelId:selectedModel?.id,laboratory})});setSource({kind:'job',id:job.id});setPerspective(0);}
  catch(e){setError(errorMessage(e));}finally{setSubmitting(false);}
 }
 async function save(){
  if(!replayId||!playback.frame)return;setNotice('');setError('');
  try{await api('/positions',{method:'POST',body:JSON.stringify({replayId,decisionIndex:playback.frame.decisionIndex,playerId:perspective,title:`Turn ${observation?.turn} · decision ${playback.frame.decisionIndex}`})});await loadLibraries();setNotice('Position saved.');}catch(e){setError(errorMessage(e));}
 }
 const available=decks.filter(d=>laboratory||!['heldout','historical'].includes(d.role??''));
 const outcome=replay?.outcome;
 const result=outcome?(outcome.winner===null?`Rules draw · ${outcome.reason}`:`Player ${outcome.winner+1} won · ${outcome.reason}`):replay?.status==='truncated'||feed.status==='truncated'?'Computation limit reached · outcome unknown':feed.status==='interrupted'?'Run interrupted · outcome unknown':live?(source?.kind==='learning'?'Learning game · experimental':'Simulation running'):streamed?`${feed.status} · replay saved when available`:replay?.status;
 return <><header className="app-header"><h1>TCG Engine Lab</h1><p>Watch games and study decisions</p></header><main className="watch-main">
  {error&&<div className="error-notice" role="alert"><p>{error}</p><button onClick={()=>setReload(v=>v+1)}>Retry</button></div>}
  <div className="run-launchers"><LearningControls state={learning} selectedGameId={source?.kind==='learning'?source.id:undefined} onStart={run=>setPendingLearning(run.id)} onWatch={watchLearning}/>
  <details className="run-setup" open={!source}><summary>Run a simulation <span>{decks.length} decks in the research pool</span></summary><form onSubmit={run} className="watch-run-form">
   {[0,1].map(i=><label key={i}>Player {i+1}<select value={deckIds[i]} onChange={e=>setDeckIds(current=>i===0?[e.target.value,current[1]]:[current[0],e.target.value])}>{available.map(d=><option key={d.id} value={d.id} disabled={d.playable===false}>{d.name}{d.role==='heldout'?' · held-out laboratory':''}{d.playable===false?' · unavailable':''}</option>)}</select></label>)}
   <label>Policy<select value={policy} onChange={e=>setPolicy(e.target.value)}><option value="heuristic">Heuristic baseline</option><option value="random">Random legal baseline</option>{models.map(m=><option key={m.id} value={m.id}>{m.name} · {m.status}</option>)}</select></label><label>Seed<input type="number" min={0} max={4294967295} value={seed} onChange={e=>setSeed(e.target.value)} required/></label><label>Decision limit<input type="number" min={1} max={3000} value={limit} onChange={e=>setLimit(e.target.value)} required/></label><button className="primary" type="submit" disabled={submitting||live||!deckIds.every(Boolean)}>{submitting?'Starting…':'Run game'}</button>
   <label className="checkbox-label laboratory-toggle"><input type="checkbox" checked={laboratory} onChange={e=>{setLaboratory(e.target.checked);if(!e.target.checked)setDeckIds(current=>current.map(id=>['heldout','historical'].includes(decks.find(d=>d.id===id)?.role??'')?decks.find(d=>d.role==='main')?.id??'':id) as [string,string]);}}/>Laboratory · allow held-out lists</label>
   <p className="run-policy-note">{selectedModel?`${selectedModel.name} · ${selectedModel.status} · ${selectedModel.modelHash}`:'Untrained baseline. Playing strength has not been established.'}</p>
  </form><details className="support-notes"><summary>Selected deck support</summary>{deckIds.map((id,i)=>{const d=decks.find(d=>d.id===id);return d&&<div key={i}><strong>Player {i+1}: {d.validation.status}</strong>{[...d.validation.notes,...(d.support?.warnings??[])].map((n,j)=><p key={j}>{n}</p>)}</div>;})}</details></details></div>
  <div className="watch-library"><label>Saved replay<select value={source?.kind==='learning'?'':replayId??''} onChange={e=>{if(e.target.value){setPendingLearning(null);setSource({kind:'replay',id:e.target.value});setNotice('');}}}><option value="">Choose a replay</option>{replays.map(r=><option key={r.id} value={r.id}>{r.id} · {r.status} · {r.frames} positions</option>)}</select></label><label>Saved position<select value={position?.id??''} onChange={e=>{if(e.target.value){setPendingLearning(null);setSource({kind:'position',id:e.target.value});}}}><option value="">Choose a position</option>{positions.map(p=><option key={p.id} value={p.id}>{p.title}</option>)}</select></label><label>Perspective<select value={perspective} disabled={source?.kind==='position'} onChange={e=>setPerspective(Number(e.target.value))}><option value={0}>Player 1</option><option value={1}>Player 2</option></select></label></div>
  {feed.error&&<div className="error-notice" role="alert"><p>{feed.error}</p><button onClick={feed.retry}>Reconnect playback</button></div>}
  <div className="arena-layout"><div className="arena-stage">
   <div className="position-strip"><strong>{position?position.title:result||'Simulation table'}</strong>{observation&&<span>{observation.phase.replaceAll('_',' ')} · {observation.decisionPlayer===perspective?'Selected player to act':'Opponent to act'}</span>}</div>
   {observation?<CardTable observation={observation} previous={playback.previous} inspect={setInspect}/>:<div className="empty-playmat" role="status"><h2>{pendingLearning?'Preparing the first learning games…':loading?'Opening position…':source?'Waiting for the first position…':'Watch a game unfold'}</h2><p>{source?'The board updates from the simulator’s recorded decisions.':'Choose two decks above, or open a saved replay. Only the selected player’s hand is visible.'}</p></div>}
   {source?.kind!=='position'&&<PlaybackControls playback={playback} live={Boolean(streamed)} status={feed.status}/>}
   {position?<div className="position-actions"><button onClick={()=>{setPendingLearning(null);setPerspective(position.playerId);setSource(position.sourceReplayId.startsWith('experimental-')?{kind:'learning',id:position.sourceReplayId.slice('experimental-'.length),playbackPaused:true}:{kind:'replay',id:position.sourceReplayId});}}>Open source replay</button><span>Saved Player {position.playerId+1} information only</span></div>:<div className="position-actions"><button disabled={!replayId||!observation} onClick={()=>void save()}>Save position</button><span>{notice||(!replayId&&source?'Saving positions becomes available when the run is persisted.':'Left / Right to step. Playback pause leaves the simulation running.')}</span></div>}
  </div><aside className="arena-aside" aria-label="Study panel">{learning.run&&<LearningProgress run={learning.run}/>} {policyContext&&<section className="policy-context" aria-label="Simulation policy"><h3>{policyContext.policy==='model'?'Frozen local checkpoint':policyContext.policy==='random'?'Random legal baseline':'Heuristic baseline'}</h3><p>{policyContext.trainingStatus?.replaceAll('-',' ')}</p><details><summary>Policy and compute</summary><p>{policyContext.modelVersion}</p><p>{policyContext.opponentPopulation}</p><p>{policyContext.computeBudget}</p></details><p>{source?.kind==='learning'?'This game uses frozen weights. Learning updates the next batch.':'Simulation does not train a model.'}</p></section>}{inspect&&<CardInspector card={inspect} close={()=>setInspect(null)}/>}<EvaluationPanel analysis={analysis} loading={analyzing} error={analysisError} retry={()=>setAnalysisRetry(v=>v+1)} observation={observation} playerId={perspective}/>{observation&&!analysis&&!analyzing&&!analysisError&&<p className="small-copy aside-note">{playback.playing?'Pause playback to analyze this position.':!replayId&&!position?'Analysis becomes available after this run is saved.':'Select a decision to study its alternatives.'}</p>}{replay&&<details className="aside-note"><summary>Replay provenance</summary><p>{replay.engineVersion}</p><p>{replay.policyContext?.policy??replay.policy??'Policy recorded in experiment metadata'}{replay.policyContext?.modelVersion?` · ${replay.policyContext.modelVersion}`:''}</p><p>{source?.kind==='learning'?'This game uses frozen weights. Learning updates the next batch.':'Simulation does not train a model.'}</p>{replay.warnings.map((warning,i)=><p key={i}>{warning}</p>)}</details>}</aside></div>
 </main><footer className="app-footer">Closed player views · local simulation · experimental policies</footer></>;
}
