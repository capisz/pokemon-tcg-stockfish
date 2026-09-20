import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api, errorMessage } from './api';
import { EvaluationPanel } from './App';
import type { Analysis, Card, Deck, Match, Observation, Player, Pokemon } from './types';

export function CardFace({card, inspect, small=false}: {card: Card; inspect:(card:Card)=>void; small?:boolean}) {
  const [loadedUrl,setLoadedUrl]=useState<string|null>(null),[failedUrl,setFailedUrl]=useState<string|null>(null);
  const failed=failedUrl===card.imageUrl,ready=Boolean(card.imageUrl&&loadedUrl===card.imageUrl&&!failed);
  return <button type="button" className={`table-card kind-${card.kind} ${small?'compact-card':''}`} data-art-state={ready?'loaded':card.imageUrl&&!failed?'loading':'unavailable'} onClick={()=>inspect(card)} aria-label={`Inspect ${card.name}`}>
    {!ready&&<span className="printed-face"><span className="printed-kind">{card.stage || card.kind}</span><strong>{card.name}</strong>{card.hp && <span>{card.hp} HP</span>}<span className="printed-id">{card.id}</span></span>}
    {card.imageUrl&&!failed&&<img key={card.imageUrl} className={`card-art ${ready?'is-loaded':''}`} src={card.imageUrl} alt="" aria-hidden="true" loading="lazy" onLoad={()=>setLoadedUrl(card.imageUrl??null)} onError={()=>setFailedUrl(card.imageUrl??null)} />}
  </button>;
}

const promptInstructions:Record<string,string>={
  CHOOSE_STARTING_POKEMONS:'Choose your starting Basic Pokémon. The first selected becomes Active; any others go on your Bench.',
  CHOOSE_NEW_ACTIVE_POKEMON:'Choose a Benched Pokémon to become your Active Pokémon.',
  CHOOSE_PRIZE_CARD:'Choose a Prize card to take.',
  CHOOSE_PRIZES_SETUP:'Choose the cards to set aside as your face-down Prizes.',
  CHOOSE_CARD_FROM_DECK:'Choose a card from your deck.',
  CHOOSE_CARD_FROM_DISCARD:'Choose a card from your discard pile.',
  CHOOSE_CARD_TO_DISCARD:'Choose a card to discard.',
  CHOOSE_CARD_TO_HAND:'Choose a card to put into your hand.',
  CHOOSE_CARDS_ORDER:'Choose the order of these cards.',
  CHOOSE_ENERGY_TO_PAY_RETREAT_COST:'Choose the Energy to discard for the retreat cost.',
  CHOOSE_POKEMON_TO_SWITCH:'Choose the Pokémon to switch.',
  ATTACH_ENERGY_CARDS:'Choose which Pokémon receive these Energy cards.',
  GO_FIRST:'Do you want to go first?',
  WANT_TO_DRAW_CARDS:'Do you want to draw cards?',
  WANT_TO_USE_ABILITY:'Do you want to use this Ability?',
  CARDS_SHOWED_BY_THE_OPPONENT:'Review the cards revealed by your opponent.',
  SETUP_OPPONENT_NO_BASIC:'Your opponent has no Basic Pokémon. Review their mulligan hand.',
};
export function promptInstruction(message:string):string {
  if(promptInstructions[message])return promptInstructions[message];
  if(!/^[A-Z][A-Z0-9_]+$/.test(message))return message;
  const words=message.toLowerCase().replaceAll('_',' ').replace(/\bpokemons?\b/g,'Pokémon');
  return `${words.charAt(0).toUpperCase()}${words.slice(1)}.`;
}
function InPlay({pokemon,inspect}: {pokemon:Pokemon;inspect:(card:Card)=>void}) {
  return <div className="in-play"><CardFace card={pokemon.card} inspect={inspect}/>{pokemon.damage>0 && <span className="damage-counter" aria-label={`${pokemon.damage} damage`}>{pokemon.damage}</span>}<div className="attached-cards"><span>{pokemon.energy.length?pokemon.energy.join(' · '):'No Energy'}</span>{pokemon.tools.map((tool,index)=><span key={index}>{tool}</span>)}{pokemon.conditions.length>0 && <strong>{pokemon.conditions.join(' · ')}</strong>}</div></div>;
}
function Side({player,own,inspect}: {player:Player;own:boolean;inspect:(card:Card)=>void}) {
  return <section className={`table-side ${own?'our-side':'their-side'}`} aria-label={own?'Your board':'Opponent board'}>
    <div className="table-counts"><strong>{own?'You':'Engine'}</strong><span>{player.prizesRemaining} Prizes</span><span>{player.deckCount} in deck</span><span>{player.handCount} in hand</span></div>
    {!own && <div className="hidden-hand" aria-label={`${player.handCount} hidden cards`}>{Array.from({length:Math.min(12,player.handCount)},(_,i)=><span key={i}/>)}</div>}
    <div className="table-zones"><div className="prize-zone" aria-label={`${player.prizesRemaining} Prize cards`}>{Array.from({length:player.prizesRemaining},(_,i)=><span className="card-back" key={i}/>)}</div>
      <div className="table-active"><span className="zone-label">Active</span>{player.active?<InPlay pokemon={player.active} inspect={inspect}/>:<div className="vacant-card">Choose an Active Pokémon</div>}</div>
      <details className="table-discard"><summary>Discard · {player.discard.length}</summary><div>{player.discard.map((card,i)=><button type="button" key={i} onClick={()=>inspect(card)}>{card.name}</button>)}</div></details>
    </div>
    <div className="table-bench"><span className="zone-label">Bench · {player.bench.length}</span><div>{player.bench.length?player.bench.map((pokemon,i)=><InPlay key={i} pokemon={pokemon} inspect={inspect}/>):<span className="bench-empty">No Benched Pokémon</span>}</div></div>
    {own && <div className="table-hand"><span className="zone-label">Your hand · {player.handCount}</span><div>{player.hand.map((card,i)=><CardFace key={i} card={card} inspect={inspect} small/>)}</div></div>}
  </section>;
}
export function CardTable({observation,inspect}: {observation:Observation;inspect:(card:Card)=>void}) {
  const own=observation.players.find(p=>p.id===observation.playerId),other=observation.players.find(p=>p.id!==observation.playerId);
  return <section className="card-table" aria-label="Pokémon card table">{other&&<Side player={other} own={false} inspect={inspect}/>}<div className="stadium-line">{observation.stadium?<button onClick={()=>inspect(observation.stadium!.card)}>{observation.stadium.card.name}</button>:<span>No Stadium in play</span>}</div>{own&&<Side player={own} own inspect={inspect}/>}</section>;
}
export function CardInspector({card,close}: {card:Card;close:()=>void}) {
  return <section className="card-inspector" aria-label="Card inspector"><div className="section-heading"><h3>{card.name}</h3><button aria-label="Close card inspector" onClick={close}>Close</button></div><p className="small-copy">{card.id}{card.hp?` · ${card.hp} HP`:''}</p>{card.text&&<p>{card.text}</p>}{card.powers?.map((p,i)=><p key={i}><strong>{p.name}</strong> {p.text}</p>)}{card.attacks?.map((a,i)=><p key={i}><strong>{a.name} · {a.damage}</strong><br/>{a.cost.join(' · ')}<br/>{a.text}</p>)}{!card.text&&!card.attacks?.length&&!card.powers?.length&&<p>Full card text is unavailable in this catalog.</p>}</section>;
}
export function Play() {
  const [decks,setDecks]=useState<Deck[]>([]),[deckId,setDeckId]=useState(''),[opponent,setOpponent]=useState('');
  const [mode,setMode]=useState<'practice'|'benchmark'>('practice'),[knownList,setKnownList]=useState(false);
  const [match,setMatch]=useState<Match|null>(null),[matches,setMatches]=useState<Match[]>([]),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const [inspect,setInspect]=useState<Card|null>(null),[filter,setFilter]=useState(''),[notice,setNotice]=useState(''),[analysis,setAnalysis]=useState<Analysis|null>(null);
  const [firstPlayer,setFirstPlayer]=useState('0');
  const mounted=useRef(true),pending=useRef<{path:string;body:unknown}|null>(null),advancing=useRef(false);
  const acceptMatch=useCallback((value:Match)=>setMatch(current=>current?.id===value.id&&current.revision>value.revision?current:value),[]);
  useEffect(()=>{mounted.current=true; Promise.all([api<{decks:Deck[]}>('/decks'),api<{matches:Match[]}>('/matches')]).then(([catalog,history])=>{
    const eligible=catalog.decks.filter(d=>d.role!=='heldout'&&d.role!=='historical'&&d.playable!==false);setDecks(eligible);setDeckId(eligible[0]?.id||'');setOpponent(eligible[0]?.archetype||'');setMatches(history.matches);
    const ongoing=history.matches.find(m=>m.status!=='completed'&&m.status!=='abandoned'); if(ongoing) return api<Match>(`/matches/${ongoing.id}`).then(acceptMatch);
  }).catch(e=>setError(errorMessage(e)));return()=>{mounted.current=false;};},[]);
  const load=useCallback(async()=>{if(!match)return;try{const value=await api<Match>(`/matches/${match.id}`);if(mounted.current)acceptMatch(value);}catch(e){setError(errorMessage(e));}},[match?.id]);
  useEffect(()=>setAnalysis(null),[match?.id,match?.revision]);
  useEffect(()=>{if(!match||match.status!=='active')return;const id=window.setInterval(()=>void load(),1000);return()=>clearInterval(id);},[match?.id,match?.status,load]);
  useEffect(()=>{if(!match||match.status!=='active'||match.observation.decisionPlayer!==1||match.thinking||advancing.current)return;
    advancing.current=true;api<Match>(`/matches/${match.id}/advance`,{method:'POST'}).then(value=>{if(mounted.current)acceptMatch(value);}).catch(e=>setError(errorMessage(e))).finally(()=>{advancing.current=false;});
  },[match?.id,match?.revision,match?.thinking,match?.status]);
  async function send(path:string,body:unknown,retain=true){setBusy(true);setError('');if(retain)pending.current={path,body};try{const value=await api<Match>(path,{method:'POST',body:JSON.stringify(body)});acceptMatch(value);pending.current=null;setFilter('');setAnalysis(null);}catch(e){setError(errorMessage(e));}finally{setBusy(false);}}
  async function create(event:FormEvent){event.preventDefault();await send('/matches',{deckId,opponentArchetype:opponent,mode,knownList},false);}
  async function command(operation:string,extra:Record<string,unknown>={}){if(!match)return;const path=operation==='action'?`/matches/${match.id}/actions`:`/matches/${match.id}/control/${operation}`;await send(path,{revision:match.revision,requestId:crypto.randomUUID(),...extra});}
  const observation=match?.observation;
  const legal=observation?.legalActions.filter(a=>a.label.toLowerCase().includes(filter.toLowerCase()))??[];
  const locked=busy||Boolean(match?.thinking),canAct=match?.status==='active'&&observation?.decisionPlayer===0&&!locked;
  return <><header className="app-header"><h1>TCG Engine Lab</h1><p>Local competitive play</p></header><main className="live-main">
    {error&&<div className="error-notice" role="alert"><p>{error}</p><button type="button" onClick={()=>pending.current?void send(pending.current.path,pending.current.body):void load()}>Retry</button>{match&&<button type="button" onClick={()=>{pending.current=null;setError('');void load();}}>Reload current position</button>}</div>}
    {!match?<section className="match-create"><h2>Play a best-of-three</h2><p>Untimed for you. The engine shares up to two minutes across each turn.</p><form onSubmit={create}><label>Your deck<select value={deckId} onChange={e=>setDeckId(e.target.value)}>{decks.map(d=><option key={d.id} value={d.id}>{d.name}</option>)}</select></label><label>Opponent archetype<select value={opponent} onChange={e=>setOpponent(e.target.value)}>{[...new Set(decks.map(d=>d.archetype))].map(a=><option key={a}>{a}</option>)}</select></label><label>Match mode<select value={mode} onChange={e=>setMode(e.target.value as 'practice'|'benchmark')}><option value="practice">Practice · hints available</option><option value="benchmark">Benchmark · review after match</option></select></label><label className="checkbox-label"><input type="checkbox" checked={knownList} onChange={e=>setKnownList(e.target.checked)}/>Known-list laboratory</label><button className="primary" disabled={busy||!deckId} type="submit">Start match</button></form>{matches.length>0&&<details><summary>Previous matches</summary>{matches.map(m=><button key={m.id} onClick={()=>api<Match>(`/matches/${m.id}`).then(setMatch).catch(e=>setError(errorMessage(e)))}>Game {m.gameNumber} · {m.score.join(' – ')} · {m.status}</button>)}</details>}</section>:<>
    <div className="match-strip"><div><h2>Game {match.gameNumber} <span className="match-score">You {match.score[0]} – {match.score[1]} Engine</span></h2><p>{match.mode==='benchmark'?'Benchmark · review locked until match ends':'Practice'} · {match.knownList?'Known-list laboratory':'Closed lists'}</p></div><div className="match-controls"><span role="status">{match.thinking?'Engine thinking…':match.status==='active'?(observation?.decisionPlayer===0?'Your decision':'Engine to act'):match.status.replace('-',' ')}</span>{match.status==='active'&&<button disabled={locked} onClick={()=>void command('pause')}>Pause</button>}{match.status==='paused'&&<button disabled={busy} onClick={()=>void command('resume')}>Resume</button>}{(match.status==='paused'||match.status==='between-games')&&<details><summary>End incomplete match</summary><p>The journal stays saved. No match outcome is assigned.</p><button disabled={locked} onClick={()=>void command('abandon')}>Keep journal and end match</button></details>}{match.status==='active'&&<details className="concede-control"><summary>Concede</summary><button disabled={locked} onClick={()=>void command('concede')}>Concede this game</button></details>}</div></div>
    {match.error&&<p className="error-notice" role="alert">{match.error}</p>}
    <div className="play-layout">{observation&&<CardTable observation={observation} inspect={setInspect}/>}
    <aside className="move-panel" aria-label="Match decisions"><div className="panel-heading"><h2>{observation?.prompt?'Resolve choice':'Your moves'}</h2><span>Turn {observation?.turn}</span></div>
      {match.status==='between-games'?<div className="move-body"><h3>{match.gameResult?.winner===0?'You won the game':match.gameResult?.winner===1?'The engine won the game':'Rules draw'}</h3><p>{match.nextStarterChooser===0?'You choose who starts next.':match.nextStarterChooser===1?'The engine chooses to start next.':'The next game uses a fresh starting-player selection.'}</p>{match.nextStarterChooser===0&&<label>Starting player<select value={firstPlayer} onChange={e=>setFirstPlayer(e.target.value)}><option value="0">You</option><option value="1">Engine</option></select></label>}<button className="primary" disabled={busy} onClick={()=>void command('next-game',{firstPlayer:match.nextStarterChooser===0?Number(firstPlayer):null})}>Start next game</button></div>:(match.status==='completed'||match.status==='abandoned')?<div className="move-body"><h3>{match.status==='completed'?'Match complete':'Incomplete match ended'}</h3>{match.status==='completed'?<p>{match.score[0]>match.score[1]?'You won':'The engine won'} {match.score.join(' – ')}.</p>:<p>The private journal is preserved. This match has no final result.</p>}{match.status==='completed'&&<button onClick={()=>api<{replayIds:string[]}>(`/matches/${match.id}/replays`,{method:'POST'}).then(result=>setNotice(`${result.replayIds.length} games are now available in Replay analysis.`)).catch(e=>setError(errorMessage(e)))}>Make replays available</button>}<button onClick={()=>{setMatch(null);setNotice('');}}>New match</button></div>:<><div className="move-body"><p>{(observation?.prompt?.message?promptInstruction(observation.prompt.message):'')|| (canAct?'Choose a legal action. Inspect any visible card for its text.':match.status==='paused'?'This match is paused. Its accepted decisions are saved.':'Waiting for the engine’s decision.')}</p>{observation?.prompt?.cards?.map((card,i)=><button key={i} onClick={()=>setInspect(card)}>{card.name}</button>)}{observation?.prompt?.hands?.map((hand,i)=><div key={i}>Revealed hand {i+1}: {hand.map(c=>c.name).join(', ')}</div>)}{canAct&&observation&&observation.legalActions.length>8&&<label>Find a move<input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Card or action name"/></label>}</div><div className="legal-moves">{canAct&&legal.map(a=><button key={a.id} type="button" onClick={()=>void command('action',{actionId:a.id})}>{a.label}</button>)}</div></>}
      <div className="move-body"><p className="small-copy">Engine turn budget remaining: {(match.engineTurnRemainingMs/1000).toFixed(1)}s</p><button disabled={busy} onClick={()=>api(`/matches/${match.id}/bookmarks`,{method:'POST',body:JSON.stringify({title:`Game ${match.gameNumber}, turn ${observation?.turn}, decision ${match.revision}`})}).then(()=>setNotice('Position bookmarked for teaching review.')).catch(e=>setError(errorMessage(e)))}>Bookmark decision</button>{match.mode==='practice'&&<button disabled={locked} onClick={()=>api<Analysis>(`/matches/${match.id}/analyze`,{method:'POST'}).then(setAnalysis).catch(e=>setError(errorMessage(e)))}>Evaluate position</button>}{notice&&<p role="status">{notice}</p>}
      <details><summary>Match details</summary><p className="small-copy">Frozen policy: {match.modelVersion}</p>{match.warnings.map((w,i)=><p className="small-copy" key={i}>{w}</p>)}{match.opponentList&&<ul>{match.opponentList.map(c=><li key={c.cardId}>{c.count} {c.name} · {c.cardId}</li>)}</ul>}</details></div>
      {inspect&&<CardInspector card={inspect} close={()=>setInspect(null)}/>}
    </aside></div>{analysis&&observation&&<EvaluationPanel analysis={analysis} loading={false} error="" retry={()=>{}} observation={observation} playerId={0}/>}</>}
  </main><footer className="app-footer">The engine is experimental. Benchmark results use the frozen policy shown in match details.</footer></>;
}
