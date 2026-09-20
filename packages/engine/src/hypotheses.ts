import { CARD_FACTORIES, CARD_METADATA, getAgentDecks, getDeck, listHash, validateDeck } from './catalog';
import type { DeckManifest } from './catalog';
import type { PublicPosition } from './belief-state';
import { SuperType, EnergyType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';

export interface OpponentHypothesis { id:string; archetype:string; kind:'listed'|'unknown-variant'|'known-list'; weight:number; deck:DeckManifest }
/** Uses the training registry and legally observed identities only. Never consults heldout lists. */
export function opponentHypotheses(position:PublicPosition, priorRevealedCards:string[]=[], knownOpponentDeckId?:string):OpponentHypothesis[] {
  const cards=new Map(position.cards.map(c=>[c.key,c.cardId]));
  const other=position.players[1-position.observer];
  const keys=new Set<number>([...other.active.cards,...other.active.tools,...other.active.energies,
    ...other.bench.flatMap((b:any)=>[...b.cards,...b.tools,...b.energies]),...other.discard,...other.lostzone,...other.stadium,...other.supporter]);
  const required=new Map<string,number>();
  for(const key of keys){const id=cards.get(key);if(!id)throw new Error('Unknown observed card reference.');required.set(id,(required.get(id)??0)+1);}
  // Cross-game reveals establish presence, not additive copy counts across games.
  for(const id of new Set(priorRevealedCards)){if(!CARD_FACTORIES[id])throw new Error('Unsupported prior revealed card.');required.set(id,Math.max(1,required.get(id)??0));}
  const fits=(d:DeckManifest)=>[...required].every(([id,n])=>(d.cards.find(c=>c.cardId===id)?.count??0)>=n);
  if(knownOpponentDeckId){const deck=getDeck(knownOpponentDeckId);return fits(deck)?[{id:deck.id,archetype:deck.archetype,kind:'known-list',weight:1,deck}]:[];}
  const training=getAgentDecks();
  const listed=training.filter(fits).map(deck=>({id:deck.id,archetype:deck.archetype,kind:'listed' as const,weight:0,deck}));
  const synthetic:OpponentHypothesis[]=[];
  for(const base of training.filter(d=>d.role==='main')){
    const deck=structuredClone(base);deck.id=`unknown-${base.archetype}`;deck.source={kind:'observation-constrained-hypothesis'};
    // Add only observed deficits, drawing replacements from surplus copies. This is
    // a conservative nearby-list model, not an assertion about the true list.
    for(const [id,n] of required){
      let row=deck.cards.find(c=>c.cardId===id);
      if(!row){row={...CARD_METADATA[id],count:0};deck.cards.push(row);}
      while(row.count<n){
        const wanted=CARD_FACTORIES[id]();
        const candidates=deck.cards.filter(c=>c.cardId!==id&&c.count>(required.get(c.cardId)??0));
        candidates.sort((a,b)=>{
          const priority=(c:typeof a)=>{const card=CARD_FACTORIES[c.cardId]();return (card.superType===wanted.superType?10:0)+(card.superType===SuperType.ENERGY&&card.energyType===EnergyType.BASIC?0:2)+c.count;};
          return priority(b)-priority(a)||a.cardId.localeCompare(b.cardId);
        });
        if(!candidates.length)break;
        candidates[0].count--;row.count++;
      }
    }
    deck.cards=deck.cards.filter(c=>c.count>0);deck.listHash=listHash(deck);
    if(fits(deck)&&validateDeck(deck).length===0)synthetic.push({id:deck.id,archetype:deck.archetype,kind:'unknown-variant',weight:0,deck});
  }
  const unknownMass=synthetic.length?(listed.length?0.15:1):0;
  listed.forEach(h=>h.weight=(1-unknownMass)/listed.length);
  synthetic.forEach(h=>h.weight=unknownMass/synthetic.length);
  return [...listed,...synthetic];
}
