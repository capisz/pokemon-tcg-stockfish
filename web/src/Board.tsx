import { useState } from 'react';
import type { Card, CardRef, Observation, Player, Pokemon } from './types';

export const sameRef = (a?:CardRef|null,b?:CardRef|null) => Boolean(a&&b&&a.playerId===b.playerId&&a.zone===b.zone&&(a.index??0)===(b.index??0));
export const refKey = (ref:CardRef) => `${ref.playerId}:${ref.zone}:${ref.index??0}`;
export type CardSelection = {card:Card;ref?:CardRef};

type CardFaceProps = {card:Card;inspect:(card:Card)=>void;small?:boolean;selected?:boolean;target?:boolean;actionable?:boolean};
export function CardFace({card,inspect,small=false,selected=false,target=false,actionable=false}:CardFaceProps) {
  const [loadedUrl,setLoadedUrl]=useState<string|null>(null),[failedUrl,setFailedUrl]=useState<string|null>(null);
  const failed=failedUrl===card.imageUrl,ready=Boolean(card.imageUrl&&loadedUrl===card.imageUrl&&!failed);
  return <button type="button" className={`table-card kind-${card.kind} ${small?'compact-card':''} ${selected?'selected-card':''} ${target?'legal-target':''} ${actionable?'actionable-card':''}`} data-art-state={ready?'loaded':card.imageUrl&&!failed?'loading':'unavailable'} onClick={()=>inspect(card)} aria-label={`Inspect ${card.name}${target?', legal target':''}`} aria-pressed={selected}>
    {!ready&&<span className="printed-face"><span className="printed-kind">{card.stage||card.kind}</span><strong>{card.name}</strong>{card.hp&&<span>{card.hp} HP</span>}<span className="printed-id">{card.id}</span></span>}
    {card.imageUrl&&!failed&&<img key={card.imageUrl} className={`card-art ${ready?'is-loaded':''}`} src={card.imageUrl} alt="" aria-hidden="true" loading="lazy" onLoad={()=>setLoadedUrl(card.imageUrl??null)} onError={()=>setFailedUrl(card.imageUrl??null)}/>}
    {target&&<span className="target-label">Target</span>}
  </button>;
}
const instructions:Record<string,string>={
 CHOOSE_STARTING_POKEMONS:'Choose your starting Basic Pokémon. The first selected becomes Active; any others go on your Bench.',
 CHOOSE_NEW_ACTIVE_POKEMON:'Choose a Benched Pokémon to become your Active Pokémon.',
 CHOOSE_PRIZE_CARD:'Choose a Prize card to take.',CHOOSE_PRIZES_SETUP:'Choose your face-down Prize cards.',
 CHOOSE_CARD_FROM_DECK:'Choose a card from your deck.',CHOOSE_CARD_FROM_DISCARD:'Choose a card from your discard pile.',
 CHOOSE_CARD_TO_DISCARD:'Choose a card to discard.',CHOOSE_CARD_TO_HAND:'Choose a card to put into your hand.',
 CHOOSE_CARDS_ORDER:'Choose the order of these cards.',CHOOSE_ENERGY_TO_PAY_RETREAT_COST:'Choose Energy to discard for the retreat cost.',
 CHOOSE_POKEMON_TO_SWITCH:'Choose the Pokémon to switch.',ATTACH_ENERGY_CARDS:'Choose which Pokémon receive these Energy cards.',
 GO_FIRST:'Do you want to go first?',WANT_TO_DRAW_CARDS:'Do you want to draw cards?',WANT_TO_USE_ABILITY:'Do you want to use this Ability?',
 CARDS_SHOWED_BY_THE_OPPONENT:'Review the cards revealed by your opponent.',SETUP_OPPONENT_NO_BASIC:'Your opponent has no Basic Pokémon. Review their mulligan hand.',
};
export function promptInstruction(message:string):string {
 if(instructions[message])return instructions[message];
 if(!/^[A-Z][A-Z0-9_]+$/.test(message))return message;
 const words=message.toLowerCase().replaceAll('_',' ').replace(/\bpokemons?\b/g,'Pokémon');
 return `${words.charAt(0).toUpperCase()}${words.slice(1)}.`;
}
type Interaction = {inspect:(card:Card)=>void; select?:(selection:CardSelection)=>void; selectTarget?:(ref:CardRef)=>void; selected?:CardRef|null; targets?:CardRef[]; sources?:CardRef[]};
function faceProps(card:Card,ref:CardRef,interaction:Interaction) {
 return {card,inspect:()=>interaction.select?interaction.select({card,ref}):interaction.inspect(card),selected:sameRef(ref,interaction.selected),target:interaction.targets?.some(r=>sameRef(r,ref)),actionable:interaction.sources?.some(r=>sameRef(r,ref))};
}
function InPlay({pokemon,reference,interaction,changed}: {pokemon:Pokemon;reference:CardRef;interaction:Interaction;changed:boolean}) {
 const grouped=new Map<string,{card:Card;count:number}>();
 for(const card of pokemon.attachments??[])grouped.set(card.id,{card,count:(grouped.get(card.id)?.count??0)+1});
 return <div className={`in-play ${changed?'zone-changed':''}`} data-zone={refKey(reference)}><CardFace {...faceProps(pokemon.card,reference,interaction)}/>{pokemon.damage>0&&<span className="damage-counter" aria-label={`${pokemon.damage} damage`}>{pokemon.damage}</span>}
  <div className="attached-cards" aria-label="Attachments and conditions">{pokemon.attachments?.length?[...grouped.values()].map(({card,count},index)=><button type="button" className={`attachment-chip kind-${card.kind}`} key={index} onClick={()=>interaction.inspect(card)} aria-label={`Inspect attached ${card.name}${count>1?`, ${count} copies`:""}`}>{card.name}{count>1?` ×${count}`:""}</button>):<><span>{pokemon.energy.length?pokemon.energy.join(' · '):'No Energy'}</span>{pokemon.tools.map((tool,index)=><span key={index}>{tool}</span>)}</>}{pokemon.conditions.length>0&&<strong>{pokemon.conditions.join(' · ')}</strong>}</div>
 </div>;
}
function Side({player,previous,own,interaction,labels}: {player:Player;previous?:Player;own:boolean;interaction:Interaction;labels?:[string,string]}) {
 const label=labels?.[own?0:1]??(own?`Player ${player.id+1}`:`Player ${player.id+1} · opponent`);
 const changed=(pokemon:Pokemon,old?:Pokemon|null)=>Boolean(previous&&JSON.stringify(pokemon)!==JSON.stringify(old));
 return <section className={`table-side ${own?'our-side':'their-side'}`} aria-label={own?'Your board':'Opponent board'}>
  <div className="table-counts"><strong>{label}</strong><span>{player.prizesRemaining} Prizes</span><span>{player.deckCount} in deck</span><span>{player.handCount} in hand</span></div>
  {!own&&<div className="hidden-hand" aria-label={`${player.handCount} hidden cards`}>{Array.from({length:Math.min(12,player.handCount)},(_,i)=><span key={i}/>)}</div>}
  <div className="table-zones"><div className={`prize-zone ${previous&&previous.prizesRemaining!==player.prizesRemaining?'zone-changed':''}`} aria-label={`${player.prizesRemaining} Prize cards`}>{Array.from({length:player.prizesRemaining},(_,i)=><span className="card-back" key={i}/>)}</div>
   <div className="table-active"><span className="zone-label">Active</span>{player.active?<InPlay pokemon={player.active} reference={{playerId:player.id,zone:'active'}} interaction={interaction} changed={changed(player.active,previous?.active)}/>:<div className="vacant-card">No Active Pokémon</div>}</div>
   <details className="table-discard"><summary>Discard · {player.discard.length}</summary><div>{player.discard.map((card,i)=><button type="button" key={i} onClick={()=>interaction.select?interaction.select({card,ref:{playerId:player.id,zone:'discard',index:i}}):interaction.inspect(card)}>{card.name}</button>)}</div></details>
  </div>
  <div className="table-bench"><span className="zone-label">Bench · {player.bench.length}</span><div>{player.bench.length?player.bench.map((pokemon,i)=><InPlay key={i} pokemon={pokemon} reference={{playerId:player.id,zone:'bench',index:pokemon.slotIndex??i}} interaction={interaction} changed={changed(pokemon,previous?.bench[i])}/>):<span className="bench-empty">No Benched Pokémon</span>}{interaction.targets?.filter((ref,index,all)=>ref.playerId===player.id&&ref.zone==='bench'&&!player.bench.some((p,i)=>(p.slotIndex??i)===(ref.index??0))&&all.findIndex(r=>sameRef(r,ref))===index).map(ref=><button key={refKey(ref)} className="empty-target-slot" onClick={()=>interaction.selectTarget?.(ref)} aria-label={`Choose empty Bench slot ${(ref.index??0)+1}`}>Bench slot {(ref.index??0)+1}<span>Legal target</span></button>)}</div></div>
  {own&&<div className={`table-hand ${previous&&previous.handCount!==player.handCount?'zone-changed':''}`}><span className="zone-label">Your hand · {player.handCount}</span><div>{player.hand.map((card,i)=><CardFace key={i} {...faceProps(card,{playerId:player.id,zone:'hand',index:i},interaction)} small/>)}</div></div>}
 </section>;
}
export function CardTable({observation,previous,labels,...interaction}:Interaction&{observation:Observation;previous?:Observation;labels?:[string,string]}) {
 const own=observation.players.find(p=>p.id===observation.playerId),other=observation.players.find(p=>p.id!==observation.playerId);
 return <section className="card-table arena-table" aria-label="Pokémon card table">{other&&<Side player={other} previous={previous?.players.find(p=>p.id===other.id)} own={false} interaction={interaction} labels={labels}/>}
 <div className="stadium-line">{observation.stadium?<button onClick={()=>interaction.inspect(observation.stadium!.card)}>{observation.stadium.card.name}</button>:<span>No Stadium in play</span>}</div>
 {own&&<Side player={own} previous={previous?.players.find(p=>p.id===own.id)} own interaction={interaction} labels={labels}/>}</section>;
}
export function CardInspector({card,close}: {card:Card;close:()=>void}) {
 return <section className="card-inspector" aria-label="Card inspector"><div className="section-heading"><h3>{card.name}</h3><button aria-label="Close card inspector" onClick={close}>Close</button></div><div className="inspector-card"><CardFace card={card} inspect={()=>{}}/></div><p className="small-copy">{card.id}{card.hp?` · ${card.hp} HP`:''}</p>{card.text&&<p>{card.text}</p>}{card.powers?.map((p,i)=><p key={i}><strong>{p.name}</strong> {p.text}</p>)}{card.attacks?.map((a,i)=><p key={i}><strong>{a.name} · {a.damage}</strong><br/>{a.cost.join(' · ')}<br/>{a.text}</p>)}{!card.text&&!card.attacks?.length&&!card.powers?.length&&<p>Full card text is unavailable in this catalog.</p>}</section>;
}
