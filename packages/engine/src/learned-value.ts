import {createHash} from 'node:crypto';
import type {Observation, PokemonView, PlayerView, CardView} from './types';

// Deliberately versioned mirror of Python features.py. Cross-runtime tests compare
// all inputs and the final value. These features are not a second rules engine.
export const VALUE_FEATURE_VERSION = 'visible-energy-coverage-v3';
const names = ['prize_race','attacker_health','bench_development','attached_energy','hand_access','deck_reserve',
  'evolution_development','energy_covered_attackers','recovery_access','draw_search_access','mobility_access',
  'disruption_access','exposed_prizes','ex_damage_protection_potential','non_ex_answers','resource_exhaustion'];
const kinds = new Set(['grass','fire','water','lightning','psychic','fighting','darkness','metal','fairy','colorless']);
const board = (p:PlayerView):PokemonView[] => [p.active,...p.bench].filter((v):v is PokemonView=>Boolean(v));
function energyKind(name:string):string|null {
  const value=name.trim().toLowerCase();
  if(kinds.has(value))return value;
  for(const kind of kinds)if(kind!=='colorless'&&[`${kind} energy`,`basic ${kind} energy`].includes(value))return kind;
  const symbol:Record<string,string>={g:'grass',r:'fire',w:'water',l:'lightning',p:'psychic',f:'fighting',d:'darkness',m:'metal',y:'fairy'};
  const match=value.match(/^basic \[([grwlpf dmy])\] energy$/);
  if(match)return symbol[match[1]]??null;
  if(['mist energy','spiky energy'].includes(value))return 'colorless';
  if(['growing [g] energy','grow [g] energy'].includes(value))return 'grass';
  return null;
}
function covered(p:PokemonView):boolean {
  const energy:Record<string,number>={};
  for(const name of p.energy??[]){const kind=energyKind(name);if(kind)energy[kind]=(energy[kind]??0)+1;}
  return (p.card.attacks??[]).some(attack=>{
    if(!Array.isArray(attack.cost))return false;
    const need:Record<string,number>={};
    for(const raw of attack.cost){const kind=String(raw).toLowerCase();need[kind]=(need[kind]??0)+1;}
    return Object.entries(need).every(([kind,count])=>kinds.has(kind)&&(kind==='colorless'||(energy[kind]??0)>=count))
      &&Object.values(energy).reduce((a,b)=>a+b,0)>=attack.cost.length;
  });
}
function resources(p:PlayerView, own:boolean, opponent:PlayerView):number[] {
  const all=board(p), hand=own?p.hand:[], handNames=hand.map(c=>c.name.toLowerCase());
  const count=(words:string[])=>handNames.filter(name=>words.some(word=>name.includes(word))).length/4;
  const opposingEx=Boolean(opponent.active&&/\bex$/.test(opponent.active.card.name.toLowerCase()));
  return [(6-p.prizesRemaining)/6,all.reduce((n,p)=>n+Math.max(0,(p.card.hp??0)-p.damage),0)/1000,all.length/6,
    all.reduce((n,p)=>n+p.energy.length,0)/12,p.handCount/10,p.deckCount/60,
    all.filter(p=>!['','basic','0'].includes(String(p.card.stage??'').toLowerCase())).length/6,
    all.filter(covered).length/6,count(['rod','recovery','recycler','stretcher']),
    count(['research','iono','ultra ball','nest ball','poffin','arven','lillie','petrel','poké pad','pokégear']),
    count(['switch','jet energy','rescue board']),count(['boss','iono','catcher','stamp']),
    all.reduce((n,p)=>n+Math.max(0,(p.card.prizeValue??1)-1),0)/6,
    all.filter(p=>p.card.name.toLowerCase()==='crustle'&&opposingEx).length/4,
    all.filter(p=>(p.card.prizeValue??1)===1&&covered(p)).length/6,p.discard.length/60].map(Math.fround);
}
export function valueFeatures(o:Observation):number[] {
  const own=o.players.find(p=>p.id===o.playerId), other=o.players.find(p=>p.id!==o.playerId);
  if(!own||!other)throw new Error('Value observation requires both players.');
  const a=resources(own,true,other),b=resources(other,false,own);
  return a.map((v,i)=>Math.max(-4,Math.min(4,Math.fround(v-b[i]))));
}
export function valueCardTokens(o:Observation):number[] {
  const result:number[]=[];
  for(const p of o.players){
    const owner=p.id===o.playerId?'own':'opponent';
    const zones:[string,CardView[]][]=[['active',p.active?[p.active.card]:[]],['bench',p.bench.map(v=>v.card)],['discard',p.discard]];
    if(owner==='own')zones.push(['hand',p.hand]);
    for(const [zone,cards]of zones)for(const card of cards){
      const hash=createHash('sha256').update(`${owner}:${zone}:${card.id??card.name??''}`).digest();
      result.push(1+hash.readUInt32BE(0)%2047);
    }
  }
  return result.slice(0,128).concat(Array(Math.max(0,128-result.length)).fill(0));
}
export interface LeafEnvelope {schemaVersion:1;payload:string;hash:string}
export interface LeafModel {modelVersion:string;checkpointHash:string;evaluate:(o:Observation)=>number}
export function loadLeafModel(envelope:LeafEnvelope):LeafModel {
  if(envelope?.schemaVersion!==1||typeof envelope.payload!=='string'||envelope.payload.length>950000
    ||createHash('sha256').update(envelope.payload).digest('hex')!==envelope.hash)throw new Error('Invalid learned-value checksum or envelope.');
  const payload=JSON.parse(envelope.payload);
  if(payload.featureVersion!==VALUE_FEATURE_VERSION||JSON.stringify(payload.featureNames)!==JSON.stringify(names)
    ||payload.cardBuckets!==2048||payload.maxVisibleCards!==128||payload.valueTrained!==true||!payload.checkpointHash||!payload.modelVersion)
    throw new Error('Incompatible learned-value features.');
  const weights=payload.weights;
  const vector=(name:string,length:number):number[]=>{
    const value=weights?.[name];
    if(!Array.isArray(value)||value.length!==length||!value.every((v:unknown)=>typeof v==='number'&&Number.isFinite(v)))throw new Error(`Invalid value weight ${name}.`);
    return value;
  };
  const matrix=(name:string,rows:number,cols:number):number[][]=>{
    const value=weights?.[name];
    if(!Array.isArray(value)||value.length!==rows||!value.every((r:unknown)=>Array.isArray(r)&&r.length===cols&&r.every(v=>typeof v==='number'&&Number.isFinite(v))))throw new Error(`Invalid value weight ${name}.`);
    return value;
  };
  const embedding=matrix('card_embedding.weight',2048,16), baseline=vector('baseline',1)[0];
  const terms=names.map((_,i)=>({w:matrix(`resource_terms.${i}.0.weight`,8,1),b:vector(`resource_terms.${i}.0.bias`,8),out:matrix(`resource_terms.${i}.2.weight`,1,8)[0]}));
  const w0=matrix('context.0.weight',64,32), b0=vector('context.0.bias',64);
  const w2=matrix('context.2.weight',32,64), b2=vector('context.2.bias',32), interaction=matrix('interaction.weight',1,32)[0];
  const dot=(a:number[],b:number[])=>a.reduce((n,v,i)=>n+v*b[i],0);
  return {modelVersion:payload.modelVersion,checkpointHash:payload.checkpointHash,evaluate(o){
    const features=valueFeatures(o), tokens=valueCardTokens(o).filter(Boolean), pool=Array(16).fill(0);
    for(const token of tokens)for(let i=0;i<16;i++)pool[i]+=embedding[token][i]/Math.max(1,tokens.length);
    const input=features.concat(pool), hidden=w0.map((w,i)=>Math.tanh(dot(w,input)+b0[i]));
    const context=w2.map((w,i)=>Math.tanh(dot(w,hidden)+b2[i]));
    const sum=terms.reduce((sum,term,i)=>sum+dot(term.out,term.w.map((row,j)=>Math.tanh(row[0]*features[i]+term.b[j]))),baseline)+dot(interaction,context);
    if(!Number.isFinite(sum))throw new Error('Non-finite learned value.');
    return 1/(1+Math.exp(-sum));
  }};
}
