import { useEffect, useState, type FormEvent } from 'react';
import { api, errorMessage } from './api';
import { CardInspector, CardTable, promptInstruction } from './Board';
import type { Card, PositionSummary, Teaching } from './types';

type Family = {id?:string; familyId?:string; title?:string; partition?:string; hypothesis?:string};
type Passage = {chunkId:string; title:string; author:string; text:string; page?:number; timestamp?:string; formatDate:string; deckVersion?:string};

export function TeachingReview(){
 const [items,setItems]=useState<Teaching[]>([]),[selected,setSelected]=useState<Teaching|null>(null);
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),[accepted,setAccepted]=useState<string[]>([]);
 const [reason,setReason]=useState(''),[resources,setResources]=useState(''),[notice,setNotice]=useState('');
 const [families,setFamilies]=useState<Family[]>([]),[positions,setPositions]=useState<PositionSummary[]>([]);
 const [positionId,setPositionId]=useState(''),[familyId,setFamilyId]=useState(''),[inspect,setInspect]=useState<Card|null>(null);
 const [query,setQuery]=useState(''),[passages,setPassages]=useState<Passage[]>([]);
 function reload(){
  Promise.all([api<{items:Teaching[]}>('/teaching/queue'),api<{families:Family[]}>('/teaching/curriculum'),api<{positions:PositionSummary[]}>('/positions')])
   .then(([q,c,p])=>{setItems(q.items);setFamilies(c.families);setPositions(p.positions);})
   .catch(e=>setError(errorMessage(e)));
 }
 function select(value:Teaching){setSelected(value);setAccepted(value.acceptableActionIds??[]);setReason(value.conditionalReasoning??'');setResources(value.criticalResources??'');setInspect(null);setNotice('');}
 useEffect(reload,[]);
 async function bindPosition(event:FormEvent){
  event.preventDefault();setBusy(true);setError('');
  try{select(await api<Teaching>('/teaching',{method:'POST',body:JSON.stringify({positionId,familyId})}));reload();}
  catch(e){setError(errorMessage(e));}finally{setBusy(false);}
 }
 async function review(event:FormEvent){
  event.preventDefault();if(!selected)return;setBusy(true);setError('');
  try{const value=await api<Teaching>(`/teaching/${selected.id}/review`,{method:'POST',body:JSON.stringify({reviewStatus:'reviewed',acceptableActionIds:accepted,conditionalReasoning:reason,criticalResources:resources,confidence:'likely'})});
   setSelected(value);setNotice(value.trainingEligible?'Reviewed and eligible for training.':value.partition!=='train'?'Reviewed; reserved for evaluation.':'Reviewed; rules audit is required before training.');reload();
  }catch(e){setError(errorMessage(e));}finally{setBusy(false);}
 }
 async function retrieve(event:FormEvent){
  event.preventDefault();setBusy(true);setError('');
  try{const result=await api<{passages:Passage[]}>('/guides/retrieve',{method:'POST',body:JSON.stringify({query,limit:5})});setPassages(result.passages);if(!result.passages.length)setNotice('No local guide passage matches this query.');}
  catch(e){setError(errorMessage(e));}finally{setBusy(false);}
 }
 return <><header className="app-header"><h1>Teaching review</h1><p>Up to ten positions per batch</p></header><main className="teaching-main">
  {error&&<div className="error-notice" role="alert">{error}<button onClick={()=>{setError('');reload();}}>Retry loading review</button></div>}
  <div className="teaching-layout"><section><h2>Review queue</h2>
   {!items.length&&<p className="small-copy">Bookmark a real decision during a match. It will appear here after benchmark play ends.</p>}
   {items.map(item=><button className="review-item" key={item.id} onClick={()=>api<Teaching>(`/teaching/${item.id}`).then(select).catch(e=>setError(errorMessage(e)))}>{item.title}<small>{item.partition} · {item.reviewStatus}</small></button>)}
   <details><summary>Guide curriculum · {families.length} families</summary><p className="small-copy">Draft guide lessons need concrete, rules-validated positions and review before they can teach an action.</p>
    {families.map((f,i)=><p key={i}><strong>{f.title??f.id??f.familyId} · {f.partition??'draft'}</strong>{f.hypothesis&&<><br/>{f.hypothesis}</>}</p>)}
   </details>
   <details><summary>Connect a saved position</summary><form onSubmit={bindPosition}>
    <label>Saved position<select required value={positionId} onChange={e=>setPositionId(e.target.value)}><option value="">Choose a position</option>{positions.map(p=><option key={p.id} value={p.id}>{p.title}</option>)}</select></label>
    <label>Guide family<select required value={familyId} onChange={e=>setFamilyId(e.target.value)}><option value="">Choose a family</option>{families.map(f=><option key={f.id??f.familyId} value={f.id??f.familyId}>{f.title??f.id??f.familyId} · {f.partition}</option>)}</select></label>
    <button disabled={busy||!positionId||!familyId}>Create teaching example</button>
   </form></details>
   <details><summary>Search local guides</summary><form onSubmit={retrieve}><label>Strategy or matchup<input required value={query} onChange={e=>setQuery(e.target.value)} placeholder="Preserving Crustle answers"/></label><button disabled={busy||!query.trim()}>Find cited passages</button></form>
    {passages.map(p=><article className="guide-passage" key={p.chunkId}><strong>{p.title}</strong><p className="small-copy">{p.author} · {p.page?`Page ${p.page}`:p.timestamp||'Location unspecified'} · {p.formatDate}{p.deckVersion?` · ${p.deckVersion}`:''}</p><p>{p.text}</p><small>Citation: {p.chunkId}</small></article>)}
   </details>
  </section><section>{selected?<>
   <h2>{selected.title}</h2><p className="small-copy">{selected.partition} · {selected.reviewStatus} · Player {(selected.playerId??0)+1} information only</p>
   {selected.source?.author&&<p className="small-copy">{selected.source.title} · {selected.source.author}{selected.source.pages?.length?` · Pages ${selected.source.pages.join(', ')}`:selected.source.page?` · Page ${selected.source.page}`:''}</p>}
   {selected.observation&&<CardTable observation={selected.observation} inspect={setInspect}/>}
   {selected.observation?.prompt&&<div className="move-body"><p>{promptInstruction(selected.observation.prompt.message)}</p>{selected.observation.prompt.cards?.map((card,i)=><button type="button" key={i} onClick={()=>setInspect(card)}>{card.name}</button>)}</div>}
   {inspect&&<CardInspector card={inspect} close={()=>setInspect(null)}/>}
   <form onSubmit={review}>{selected.mechanicsAudit&&<p className="small-copy">Only {selected.mechanicsAudit.validatedActionIds?.length??0} of {selected.observation?.legalActions.length??0} choices have checked continuations; selecting another keeps this example out of training.</p>}<p className="small-copy">Accept all sound alternatives. Explain the condition that changes your choice.</p>
    <fieldset><legend>Acceptable legal actions</legend>{selected.observation?.legalActions.map(action=><label className="checkbox-label" key={action.id}><input type="checkbox" checked={accepted.includes(action.id)} onChange={e=>setAccepted(e.target.checked?[...accepted,action.id]:accepted.filter(a=>a!==action.id))}/>{action.label}</label>)}</fieldset>
    {!selected.observation?.legalActions.length&&<p>This bookmark was taken while the other player was deciding. Bookmark your own decision to annotate legal alternatives.</p>}
    <label>Conditional reasoning<textarea required value={reason} onChange={e=>setReason(e.target.value)} rows={5}/></label>
    <label>Critical resources<textarea value={resources} onChange={e=>setResources(e.target.value)} rows={3}/></label>
    <button className="primary" disabled={busy||!accepted.length||!reason.trim()}>Save reviewed annotation</button>
   </form>
  </>:<p className="small-copy">Select a bookmarked decision to review its board and legal choices.</p>}
  {notice&&<p role="status">{notice}</p>}</section></div>
 </main></>;
}
