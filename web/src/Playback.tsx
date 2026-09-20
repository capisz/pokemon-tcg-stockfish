import { useCallback, useEffect, useRef, useState } from 'react';
import { api, errorMessage } from './api';
import type { FrameFeed, PolicyContext, ViewFrame } from './types';

const ended=(status:string)=>['finished','completed','truncated','error','failed','abandoned','interrupted','stopped'].includes(status);
/** Feeds carry immutable snapshots: priorAction produced this frame. Cursor is the resume token. */
export function useFrameFeed(path:string|null) {
 const [frames,setFrames]=useState<ViewFrame[]>([]),[status,setStatus]=useState('queued'),[error,setError]=useState(''),[replayId,setReplayId]=useState<string|undefined>();
 const [retry,setRetry]=useState(0),[policyContext,setPolicyContext]=useState<PolicyContext|undefined>();
 useEffect(()=>{
  setFrames([]);setStatus('queued');setError('');setReplayId(undefined);setPolicyContext(undefined);
  if(!path)return;
  let alive=true,cursor=-1,timer:number|undefined;
  const controller=new AbortController();
  async function poll(){
   try{
    let more=true;
    while(alive&&more){
     const result=await api<FrameFeed>(`${path}${path!.includes('?')?'&':'?'}after=${cursor}&limit=100`,{signal:controller.signal});
     if(!alive)return;
     setFrames(current=>{
      const last=current.at(-1)?.cursor??-1;
      return [...current,...result.frames.filter(f=>(f.cursor??-1)>last).sort((a,b)=>(a.cursor??0)-(b.cursor??0))];
     });
     const next=result.nextCursor;
     if(result.hasMore&&next<=cursor)throw new Error('Playback feed did not advance. Retry loading the journal.');
     cursor=Math.max(cursor,next);setStatus(result.status);setReplayId(result.replayId);setPolicyContext(result.policyContext);setError(typeof result.error==='string'?result.error:result.error?.message??'');
     more=result.hasMore;
     if(!more&&!ended(result.status))timer=window.setTimeout(()=>void poll(),1000);
    }
   }catch(e){if(alive){setError(errorMessage(e));timer=window.setTimeout(()=>void poll(),2500);}}
  }
  void poll();return()=>{alive=false;controller.abort();if(timer!==undefined)clearTimeout(timer);};
 },[path,retry]);
 return {frames,status,error,replayId,policyContext,retry:()=>setRetry(v=>v+1)};
}
export function usePlayback(frames:ViewFrame[],key:string,live:boolean,autoplay?:boolean) {
 const [index,setIndex]=useState(0),[playing,setPlaying]=useState(live),[speed,setSpeed]=useState(1),[animate,setAnimate]=useState(false);
 const count=frames.length;
 useEffect(()=>{setIndex(0);setPlaying(autoplay??live);setAnimate(false);},[key]);
 useEffect(()=>{if(index>=count&&count)setIndex(count-1);},[count,index]);
 useEffect(()=>{
  if(!playing)return;
  if(index>=count-1){if(!live&&count)setPlaying(false);return;}
  const timer=window.setTimeout(()=>{setAnimate(true);setIndex(i=>Math.min(i+1,count-1));},700/speed);
  return()=>clearTimeout(timer);
 },[index,count,playing,speed,live]);
 const seek=useCallback((next:number)=>{setPlaying(false);setAnimate(false);setIndex(Math.max(0,Math.min(count-1,next)));},[count]);
 const turn=useCallback((direction:number)=>{
  const now=frames[index];if(!now)return;
  let next=index+direction;
  while(next>=0&&next<count&&frames[next].observation.turn===now.observation.turn&&frames[next].gameNumber===now.gameNumber)next+=direction;
  seek(next);
 },[frames,index,count,seek]);
 const latest=()=>{setAnimate(false);setIndex(Math.max(0,count-1));setPlaying(live);};
 const frame=frames[index],previous=animate&&index>0?frames[index-1]?.observation:undefined;
 return {index,frame,previous,playing,speed,seek,turn,latest,setSpeed,toggle:()=>setPlaying(v=>!v),atLatest:index>=count-1,count};
}
export type Playback = ReturnType<typeof usePlayback>;
export function PlaybackControls({playback,live=false,status,catchingUp=false}:{playback:Playback;live?:boolean;status?:string;catchingUp?:boolean}) {
 const {index,count,frame}=playback;
 return <section className="playback-controls" aria-label="Playback controls">
  <div className="transport-row"><button title="Previous turn" aria-label="Previous turn" disabled={!index} onClick={()=>playback.turn(-1)}>Previous turn</button><button aria-label="Previous decision" disabled={!index} onClick={()=>playback.seek(index-1)}>Back</button><button className="transport-play" disabled={!count} onClick={playback.toggle}>{playback.playing?'Pause playback':'Play playback'}</button><button aria-label="Next decision" disabled={index>=count-1} onClick={()=>playback.seek(index+1)}>Next</button><button aria-label="Next turn" disabled={index>=count-1} onClick={()=>playback.turn(1)}>Next turn</button>
  <label className="speed-control">Speed<select aria-label="Playback speed" value={playback.speed} onChange={e=>playback.setSpeed(Number(e.target.value))}>{[0.5,1,2,4].map(v=><option key={v} value={v}>{v}×</option>)}</select></label>{live&&<button className={playback.atLatest?'is-current':''} onClick={playback.latest}>{catchingUp?'Updating board…':playback.atLatest?'At latest position':'Return to live position'}</button>}</div>
  <label className="timeline-range"><span>{frame?.gameNumber?`Game ${frame.gameNumber} · `:''}Turn {frame?.observation.turn??0} · Decision {frame?.decisionIndex??0}<span>{count?`${index+1} / ${count}`:'Waiting for first position'}{live&&status?` · ${status}`:''}</span></span><input type="range" aria-label="Replay position" min={0} max={Math.max(0,count-1)} value={index} disabled={!count} onChange={e=>playback.seek(Number(e.target.value))}/></label>
  <p className="played-event" aria-live="off">{frame?.priorAction?<><span>Last action · </span>{frame.priorAction.label}</>:frame?'Position ready':'The board will appear as soon as the simulator publishes its first position.'}</p>
 </section>;
}
export function usePlaybackKeys(playback:Playback) {
 const ref=useRef(playback);ref.current=playback;
 useEffect(()=>{
  const handler=(event:KeyboardEvent)=>{
   if(event.altKey||event.ctrlKey||event.metaKey)return;
   if(event.target instanceof HTMLElement&&(event.target.closest('input,select,textarea,button,summary,[contenteditable]')))return;
   const p=ref.current;
   if(event.key==='ArrowLeft'){event.preventDefault();p.seek(p.index-1);}
   if(event.key==='ArrowRight'){event.preventDefault();p.seek(p.index+1);}
   if(event.key==='Home'){event.preventDefault();p.seek(0);}
   if(event.key==='End'){event.preventDefault();p.seek(p.count-1);}
  };window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler);
 },[]);
}
