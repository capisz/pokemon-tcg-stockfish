import { createHash } from 'node:crypto';
import { CardManager } from '../../../vendor/twinleaf/ptcg-server/src/game/cards/card-manager';
import { Card } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card';
import { SuperType, Stage, EnergyType, CardTag } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';
import { CARD_FACTORIES, CARD_METADATA, manifests } from './generated-catalog';
import format from '../../../formats/standard-2026-09-17.json';
export { CARD_FACTORIES, CARD_METADATA };
export interface DeckManifest {
  id:string; name:string; archetype:string; formatDate:string; version:number;
  role:'main'|'training-variant'|'heldout'|'historical'; source:unknown; listHash:string;
  cards:{cardId:string;name:string;engineName:string;regulationMark:string;count:number}[];
  validation:{status:string;trainingEligible:boolean;legalityVerified:boolean;notes:string[]};
}
const registry=manifests as unknown as DeckManifest[];
let registered=false;
export function registerCards() {
  if(registered)return;
  for(const make of Object.values(CARD_FACTORIES)) CardManager.getInstance().defineCard(make());
  registered=true;
}
export function listHash(deck:Pick<DeckManifest,'cards'>) {
  return createHash('sha256').update(JSON.stringify(deck.cards.map(c=>[c.cardId,c.count]).sort((a,b)=>String(a[0]).localeCompare(String(b[0]),'en')))).digest('hex');
}
export function validateDeck(deck:DeckManifest):string[] {
  const errors:string[]=[];const counts=new Map<string,number>();let basics=0;let aceSpecs=0;
  if(deck.formatDate!==format.date)errors.push('Unsupported format freeze: '+deck.formatDate);
  if(deck.cards.reduce((n,c)=>n+c.count,0)!==60)errors.push('Deck must contain exactly 60 cards.');
  const ids=new Set<string>();
  for(const entry of deck.cards) {
    if(ids.has(entry.cardId))errors.push('Duplicate manifest row: '+entry.cardId);ids.add(entry.cardId);
    if(!Number.isInteger(entry.count)||entry.count<1)errors.push('Invalid count: '+entry.cardId);
    const make=CARD_FACTORIES[entry.cardId];if(!make){errors.push('Unsupported card: '+entry.cardId);continue;}
    const card=make();const energy=card.superType===SuperType.ENERGY&&card.energyType===EnergyType.BASIC;
    if(card.fullName!==entry.engineName||card.name!==entry.name)errors.push('Engine mapping mismatch: '+entry.cardId);
    const expansion=(format.sets as Record<string, {released:string;tournamentLegal?:string}>)[entry.cardId.split('-')[0]];
    if(!expansion||expansion.released>format.date||(expansion.tournamentLegal??expansion.released)>format.date)errors.push('Unreleased or unverified set: '+entry.cardId);
    if(!energy&&!format.regulationMarks.includes(card.regulationMark))errors.push('Rotated card: '+entry.cardId);
    if(!energy)counts.set(card.name,(counts.get(card.name)??0)+entry.count);
    if(card.superType===SuperType.POKEMON&&(card as any).stage===Stage.BASIC)basics+=entry.count;
    if(card.tags.includes(CardTag.ACE_SPEC))aceSpecs+=entry.count;
  }
  for(const [name,n]of counts)if(n>4)errors.push('More than four copies: '+name);
  if(!basics)errors.push('Deck needs a Basic Pokemon.');if(aceSpecs>1)errors.push('Only one ACE SPEC is permitted.');
  const hash=createHash('sha256').update(JSON.stringify(deck.cards.map(c=>[c.cardId,c.count]).sort((a,b)=>String(a[0])<String(b[0])?-1:1))).digest('hex');
  if(deck.listHash!==hash)errors.push('Immutable list hash mismatch.');
  return errors;
}
export function getDecks(options:{includeHeldout?:boolean;includeHistorical?:boolean}={}) {
  return registry.filter(d=>(options.includeHistorical||d.role!=='historical')&&(options.includeHeldout!==false||d.role!=='heldout')).map(d=>({...d,cardCount:d.cards.reduce((n,c)=>n+c.count,0),support:{playable:validateDeck(d).length===0,errors:validateDeck(d),trainingEligible:d.validation.trainingEligible===true}}));
}
export function getAgentDecks(){return getDecks({includeHeldout:false}).filter(d=>d.role==='main'||d.role==='training-variant');}
export function getDeck(id:string):DeckManifest {
  const deck=registry.find(d=>d.id===id);if(!deck)throw new Error('Unknown deck: '+id);
  const errors=validateDeck(deck);if(errors.length)throw new Error(errors.join(' '));return deck;
}
export function deckNames(id:string){return getDeck(id).cards.flatMap(c=>Array(c.count).fill(c.engineName));}
export function instantiateDeck(id:string):Card[]{registerCards();return getDeck(id).cards.flatMap(c=>Array.from({length:c.count},()=>CARD_FACTORIES[c.cardId]()));}
export function printedId(card:Card){return `${card.set}-${card.setNumber}`;}
