import { useEffect, useState } from 'react';
import { Play } from './Play';
import { Watch } from './Watch';
import { TeachingReview } from './TeachingReview';
export { EvaluationPanel } from './EvaluationPanel';
const currentView=()=>location.hash==='#play'?'play':location.hash==='#review'?'review':'replays';
export default function App(){
 const [view,setView]=useState(currentView);
 useEffect(()=>{const changed=()=>setView(currentView());window.addEventListener('hashchange',changed);return()=>window.removeEventListener('hashchange',changed);},[]);
 return <><nav className="app-tabs" aria-label="Workspace">{[['play','Play'],['replays','Replay analysis'],['review','Teaching review']].map(([id,label])=><button key={id} className={view===id?'selected':''} aria-current={view===id?'page':undefined} onClick={()=>{location.hash=id;setView(id);}}>{label}</button>)}</nav>{view==='play'?<Play/>:view==='review'?<TeachingReview/>:<Watch/>}</>;
}
