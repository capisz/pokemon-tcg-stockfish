import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api, errorMessage } from './api';
import { CardFace, CardInspector, CardTable, promptInstruction, sameRef, type CardSelection } from './Board';
import { EvaluationPanel } from './EvaluationPanel';
import { PlaybackControls, useFrameFeed, usePlayback, usePlaybackKeys } from './Playback';
import type { Action, Analysis, Card, CardRef, Deck, Match, ModelOption, ViewFrame } from './types';
export { CardFace, CardInspector, CardTable, promptInstruction } from './Board';
const bindings=(action:Action)=>action.choiceRefs?.length?action.choiceRefs:[action];

export function Play() {
 const [decks,setDecks]=useState<Deck[]>([]),[deckId,setDeckId]=useState(''),[opponent,setOpponent]=useState('');
 const [models,setModels]=useState<ModelOption[]>([]),[modelId,setModelId]=useState('heuristic');
 const [mode,setMode]=useState<'practice'|'benchmark'>('practice'),[knownList,setKnownList]=useState(false);
 const [match,setMatch]=useState<Match|null>(null),[matches,setMatches]=useState<Match[]>([]),[error,setError]=useState(''),[busy,setBusy]=useState(false);
 const [inspect,setInspect]=useState<Card|null>(null),[selected,setSelected]=useState<CardSelection|null>(null),[target,setTarget]=useState<CardRef|null>(null),[filter,setFilter]=useState(''),[notice,setNotice]=useState(''),[analysis,setAnalysis]=useState<Analysis|null>(null);
 const [firstPlayer,setFirstPlayer]=useState('0');
 const mounted=useRef(true),pending=useRef<{path:string;body:unknown}|null>(null),advancing=useRef(false),followRevision=useRef<number|null>(null),opened=useRef<string|null>(null);
 const acceptMatch=useCallback((value:Match)=>setMatch(current=>current?.id===value.id&&current.revision>value.revision?current:value),[]);
 const feed=useFrameFeed(match?`/matches/${match.id}/frames`:null);
 const fallback:ViewFrame[]=match?[{cursor:match.frameCursor,decisionIndex:match.observation.decisionIndex??0,actor:match.observation.decisionPlayer,observation:match.observation,revision:match.revision,gameNumber:match.gameNumber}]:[];
 const frames=feed.frames.length?feed.frames:fallback;
 const playback=usePlayback(frames,match?.id??'',true);usePlaybackKeys(playback);
 const observation=playback.frame?.observation??match?.observation;
 const current=Boolean(match&&playback.frame&&(match.frameCursor!==undefined?playback.frame.cursor===match.frameCursor:playback.frame.revision===match.revision));
 const catchingUp=Boolean(match&&!current&&playback.atLatest);
 const locked=busy||Boolean(match?.thinking),canAct=match?.status==='active'&&current&&observation?.decisionPlayer===0&&!locked;
 const actions=observation?.legalActions??[];
 const sourceActions=selected?.ref?actions.filter(a=>bindings(a).some(b=>sameRef(b.sourceRef,selected.ref)||(!b.sourceRef&&sameRef(b.targetRef,selected.ref)))):actions;
 const selectedHasActions=Boolean(selected?.ref&&sourceActions.length);
 const targetRefs=selectedHasActions?sourceActions.flatMap(a=>bindings(a).flatMap(b=>b.targetRef?[b.targetRef]:[])):[];
 const sources=canAct?actions.flatMap(a=>bindings(a).flatMap(b=>b.sourceRef?[b.sourceRef]:b.targetRef?[b.targetRef]:[])):[];
 const filtered=selectedHasActions?sourceActions:actions;
 const legal=filtered.filter(a=>(!target||bindings(a).some(b=>sameRef(b.targetRef,target)))&&a.label.toLowerCase().includes(filter.toLowerCase()));
 const staged=actions.filter(a=>a.choiceOperation==='finish'||a.choiceOperation==='undo');
 function selectCard(value:CardSelection){
  setInspect(value.card);
  if(canAct&&selectedHasActions&&value.ref&&targetRefs.some(r=>sameRef(r,value.ref))){setTarget(value.ref);return;}
  setSelected(value);setTarget(null);setFilter('');
 }
 useEffect(()=>{mounted.current=true;Promise.all([api<{decks:Deck[]}>('/decks'),api<{matches:Match[]}>('/matches'),api<{models:ModelOption[]}>('/models')]).then(([catalog,history,available])=>{
  const eligible=catalog.decks.filter(d=>d.role!=='heldout'&&d.role!=='historical'&&d.playable!==false);setDecks(eligible);setDeckId(eligible[0]?.id||'');setOpponent(eligible[0]?.archetype||'');setMatches(history.matches);setModels(available.models);
  const ongoing=history.matches.find(m=>m.status!=='completed'&&m.status!=='abandoned');if(ongoing)return api<Match>(`/matches/${ongoing.id}`).then(acceptMatch);
 }).catch(e=>setError(errorMessage(e)));return()=>{mounted.current=false;};},[]);
 const load=useCallback(async()=>{if(!match)return;try{const value=await api<Match>(`/matches/${match.id}`);if(mounted.current)acceptMatch(value);}catch(e){setError(errorMessage(e));}},[match?.id]);
 useEffect(()=>{setAnalysis(null);setSelected(null);setTarget(null);setInspect(null);setFilter('');},[match?.id,playback.frame?.cursor,playback.frame?.revision]);
 useEffect(()=>{
  if(!match||!feed.frames.length)return;
  const last=feed.frames.at(-1)!;
  if(opened.current!==match.id||(followRevision.current!==null&&(last.revision??-1)>=followRevision.current)){
   opened.current=match.id;followRevision.current=null;playback.latest();
  }
 },[match?.id,feed.frames]);
 useEffect(()=>{if(!match||!['active','paused','between-games'].includes(match.status))return;const id=window.setInterval(()=>void load(),1000);return()=>clearInterval(id);},[match?.id,match?.status,load]);
 useEffect(()=>{if(!match||match.status!=='active'||match.observation.decisionPlayer!==1||match.thinking||advancing.current)return;
  advancing.current=true;api<Match>(`/matches/${match.id}/advance`,{method:'POST'}).then(value=>{if(mounted.current)acceptMatch(value);}).catch(e=>setError(errorMessage(e))).finally(()=>{advancing.current=false;});
 },[match?.id,match?.revision,match?.thinking,match?.status]);
 async function send(path:string,body:unknown,retain=true){setBusy(true);setError('');if(retain)pending.current={path,body};try{const value=await api<Match>(path,{method:'POST',body:JSON.stringify(body)});acceptMatch(value);followRevision.current=value.revision;pending.current=null;setFilter('');setAnalysis(null);}catch(e){setError(errorMessage(e));}finally{setBusy(false);}}
 async function create(event:FormEvent){event.preventDefault();await send('/matches',{deckId,opponentArchetype:opponent,mode,knownList,...(modelId!=='heuristic'?{modelId}:{})},false);}
 async function command(operation:string,extra:Record<string,unknown>={}){if(!match)return;if(operation==='action'&&!canAct)return;const path=operation==='action'?`/matches/${match.id}/actions`:`/matches/${match.id}/control/${operation}`;await send(path,{revision:match.revision,requestId:crypto.randomUUID(),...extra});}
 return <><header className="app-header"><h1>TCG Engine Lab</h1><p>Local competitive play</p></header><main className="live-main">
  {error&&<div className="error-notice" role="alert"><p>{error}</p><button type="button" onClick={()=>pending.current?void send(pending.current.path,pending.current.body):void load()}>Retry</button>{match&&<button type="button" onClick={()=>{pending.current=null;setError('');void load();}}>Reload current position</button>}</div>}
  {!match?<section className="match-create"><h2>Play a best-of-three</h2><p>Untimed for you. The engine shares up to two minutes across each turn.</p><form onSubmit={create}>
   <label>Your deck<select value={deckId} onChange={e=>setDeckId(e.target.value)}>{decks.map(d=><option key={d.id} value={d.id}>{d.name}</option>)}</select></label><label>Opponent archetype<select value={opponent} onChange={e=>setOpponent(e.target.value)}>{[...new Set(decks.map(d=>d.archetype))].map(a=><option key={a}>{a}</option>)}</select></label>
   <label>Engine policy<select value={modelId} onChange={e=>setModelId(e.target.value)}><option value="heuristic">Heuristic baseline · untrained</option>{models.map(m=><option key={m.id} value={m.id} disabled={mode==='benchmark'&&m.dataTier==='experimental'}>{m.name} · {m.status}</option>)}</select></label><label>Match mode<select value={mode} onChange={e=>{setMode(e.target.value as 'practice'|'benchmark');if(e.target.value==='benchmark'&&modelId.startsWith('experimental-'))setModelId('heuristic');}}><option value="practice">Practice · hints available</option><option value="benchmark">Benchmark · review after match</option></select></label><label className="checkbox-label"><input type="checkbox" checked={knownList} onChange={e=>setKnownList(e.target.checked)}/>Known-list laboratory</label><button className="primary" disabled={busy||!deckId} type="submit">Start match</button>
  </form><p className="small-copy">The selected policy is frozen for the whole match. Local checkpoints are experimental until evaluated.</p>{matches.length>0&&<details><summary>Previous matches</summary>{matches.map(m=><button key={m.id} onClick={()=>api<Match>(`/matches/${m.id}`).then(setMatch).catch(e=>setError(errorMessage(e)))}>Game {m.gameNumber} · {m.score.join(' – ')} · {m.status}</button>)}</details>}</section>:<>
   <div className="match-strip"><div><h2>Game {match.gameNumber}<span className="match-score">You {match.score[0]} – {match.score[1]} Engine</span></h2><p>{match.mode==='benchmark'?'Benchmark · review locked until match ends':'Practice'} · {match.knownList?'Known-list laboratory':'Closed lists'}</p></div><div className="match-controls"><span role="status">{match.thinking?'Engine thinking…':match.status==='active'?(match.observation.decisionPlayer===0?'Your decision':'Engine to act'):match.status.replace('-',' ')}</span>{match.status==='active'&&<button disabled={locked} onClick={()=>void command('pause')}>Pause</button>}{match.status==='paused'&&<button disabled={busy} onClick={()=>void command('resume')}>Resume</button>}{(match.status==='paused'||match.status==='between-games')&&<details><summary>End incomplete match</summary><p>The journal stays saved. No match outcome is assigned.</p><button disabled={locked} onClick={()=>void command('abandon')}>Keep journal and end match</button></details>}{match.status==='active'&&<details className="concede-control"><summary>Concede</summary><button disabled={locked} onClick={()=>void command('concede')}>Concede this game</button></details>}</div></div>
   {match.error&&<p className="error-notice" role="alert">{match.error}</p>}{feed.error&&<p className="error-notice" role="alert">{feed.error}<button onClick={feed.retry}>Reconnect playback</button></p>}
   <div className="arena-layout"><div className="arena-stage"><div className="position-strip"><strong>{current?'Current position':catchingUp?'Updating board…':'Reviewing earlier position'}</strong><span>{observation?.phase.replaceAll('_',' ')} · Turn {observation?.turn}</span></div>{observation&&<CardTable observation={observation} previous={playback.previous} inspect={setInspect} select={selectCard} selectTarget={setTarget} selected={selected?.ref} sources={sources} targets={canAct?targetRefs:[]} labels={['You','Engine']}/>}<PlaybackControls playback={playback} live status={match.status} catchingUp={catchingUp}/></div>
   <aside className="arena-aside"><section className="move-panel" aria-label="Match decisions"><div className="panel-heading"><h2>{observation?.prompt?'Resolve choice':'Your moves'}</h2><span>Turn {observation?.turn}</span></div>
    {!current&&<div className="history-notice"><p>{catchingUp?'Receiving the latest accepted decision. Controls return when the board is current.':'You are reviewing the journal. Moves and annotations use the current position.'}</p>{!catchingUp&&<button onClick={playback.latest}>Return to live position</button>}</div>}
    {match.status==='between-games'?<div className="move-body"><h3>{match.gameResult?.winner===0?'You won the game':match.gameResult?.winner===1?'The engine won the game':'Rules draw'}</h3><p>{match.nextStarterChooser===0?'You choose who starts next.':match.nextStarterChooser===1?'The engine chooses to start next.':'The next game uses a fresh starting-player selection.'}</p>{match.nextStarterChooser===0&&<label>Starting player<select value={firstPlayer} onChange={e=>setFirstPlayer(e.target.value)}><option value="0">You</option><option value="1">Engine</option></select></label>}<button className="primary" disabled={busy} onClick={()=>void command('next-game',{firstPlayer:match.nextStarterChooser===0?Number(firstPlayer):null})}>Start next game</button></div>:(match.status==='completed'||match.status==='abandoned')?<div className="move-body"><h3>{match.status==='completed'?'Match complete':'Incomplete match ended'}</h3>{match.status==='completed'?<p>{match.score[0]>match.score[1]?'You won':'The engine won'} {match.score.join(' – ')}.</p>:<p>The private journal is preserved. This match has no final result.</p>}{match.status==='completed'&&<button onClick={()=>api<{replayIds:string[]}>(`/matches/${match.id}/replays`,{method:'POST'}).then(result=>setNotice(`${result.replayIds.length} games are now available in Replay analysis.`)).catch(e=>setError(errorMessage(e)))}>Make replays available</button>}<button onClick={()=>{setMatch(null);setNotice('');}}>New match</button></div>:<>
    <div className="move-body"><p>{observation?.prompt?.message?promptInstruction(observation.prompt.message):canAct?'Select a card to see its moves and legal targets, or choose an action below.':match.status==='paused'?'This match is paused. Its accepted decisions are saved.':!current?'Return to the current position to play.':'Waiting for the engine’s decision.'}</p>
     {observation?.prompt?.cards&&<div className="prompt-card-grid">{observation.prompt.cards.map((card,i)=><CardFace key={i} card={card} inspect={()=>selectCard({card,ref:{playerId:0,zone:'prompt',index:i}})} small selected={sameRef(selected?.ref,{playerId:0,zone:'prompt',index:i})}/>)}</div>}
     {observation?.prompt?.hands?.map((hand,i)=><div key={i}>Revealed hand {i+1}: {hand.map(c=>c.name).join(', ')}</div>)}
     {observation?.prompt?.selectionCount!==undefined&&<div className="selection-tray" aria-label="Selected choices"><strong>{observation.prompt.selectionCount} selected{observation.prompt.min!==undefined?` · minimum ${observation.prompt.min}`:''}{observation.prompt.max!==undefined?` · maximum ${observation.prompt.max}`:''}</strong>{observation.prompt.selection?.map((choice,i)=><span key={i}>{typeof choice?.name==='string'?choice.name:`Choice ${i+1}`}</span>)}{observation.prompt.selectedChoices?.length&&!observation.prompt.selection?.length?<p>{observation.prompt.selectedChoices.length} choices staged</p>:null}</div>}
     {selectedHasActions&&canAct&&<div className="selected-action-source"><strong>{selected?.card.name}</strong><p>{target?'Target selected. Confirm a move below.':targetRefs.length?'Legal targets are highlighted on the table.':'Choose a move below.'}</p><button onClick={()=>{setSelected(null);setTarget(null);}}>Show all moves</button></div>}
     {canAct&&actions.length>8&&<label>Find a move<input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Card or action name"/></label>}
    </div><div className="legal-moves">{canAct&&legal.filter(a=>!['finish','undo'].includes(a.choiceOperation??'')).map(a=><button key={a.id} type="button" onClick={()=>void command('action',{actionId:a.id})}>{a.label}</button>)}</div>{canAct&&staged.length>0&&<div className="staged-controls">{staged.map(a=><button className={a.choiceOperation==='finish'?'primary':''} key={a.id} onClick={()=>void command('action',{actionId:a.id})}>{a.choiceOperation==='finish'?'Finish selection':'Undo last choice'}</button>)}</div>}
    </>}
    <div className="move-body"><p className="small-copy">Engine turn budget remaining: {(match.engineTurnRemainingMs/1000).toFixed(1)}s</p><button disabled={busy||!current} onClick={()=>api(`/matches/${match.id}/bookmarks`,{method:'POST',body:JSON.stringify({title:`Game ${match.gameNumber}, turn ${observation?.turn}, decision ${match.revision}`})}).then(()=>setNotice('Position bookmarked for teaching review.')).catch(e=>setError(errorMessage(e)))}>Bookmark decision</button>{match.mode==='practice'&&<button disabled={locked||!current} onClick={()=>api<Analysis>(`/matches/${match.id}/analyze`,{method:'POST'}).then(setAnalysis).catch(e=>setError(errorMessage(e)))}>Evaluate position</button>}{notice&&<p role="status">{notice}</p>}
     <details><summary>Match details</summary><p className="small-copy">Frozen policy: {match.modelVersion}</p>{match.warnings.map((w,i)=><p className="small-copy" key={i}>{w}</p>)}{match.opponentList&&<ul>{match.opponentList.map(c=><li key={c.cardId}>{c.count} {c.name} · {c.cardId}</li>)}</ul>}</details>
    </div></section>{inspect&&<CardInspector card={inspect} close={()=>setInspect(null)}/>} {analysis&&observation&&current&&<EvaluationPanel analysis={analysis} loading={false} error="" retry={()=>{}} observation={observation} playerId={0}/>}</aside></div>
  </>}
 </main><footer className="app-footer">The engine is experimental. Benchmark results use the frozen policy shown in match details.</footer></>;
}
