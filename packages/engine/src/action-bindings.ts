import { State } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Card } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { CardTarget, PlayerType, SlotType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { StateUtils } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state-utils';
import { printedId } from './catalog';
import type { ActionRef, ChoiceBinding } from './types';

export function targetRef(state:State,perspective:number,target:CardTarget):ActionRef {
  const owner=state.players.find(p=>target.player===PlayerType.BOTTOM_PLAYER?p.id===perspective:p.id!==perspective);
  if(!owner || ![SlotType.ACTIVE,SlotType.BENCH].includes(target.slot))throw new Error('Unsupported board action reference.');
  return {playerId:owner.id-1,zone:target.slot===SlotType.ACTIVE?'active':'bench',index:target.slot===SlotType.ACTIVE?0:target.index};
}
function listRef(state:State,viewer:number,list:CardList,index:number):ActionRef {
  for(const p of state.players){
    if(list===p.hand)return {playerId:p.id-1,zone:'hand',index};
    if(list===p.discard)return {playerId:p.id-1,zone:'discard',index};
    if(list===p.active||list===p.active.energies)return {playerId:p.id-1,zone:'active',index:0};
    for(const[slot,board]of p.bench.entries())if(list===board||list===board.energies)return {playerId:p.id-1,zone:'bench',index:slot};
  }
  // Searches and temporary lists are scoped to the displayed prompt, not to a
  // hidden deck slot. The reference never reveals the card at that slot.
  return {playerId:viewer-1,zone:'prompt',index};
}
function cardRef(state:State,viewer:number,card:Card):ActionRef|undefined {
  for(const p of state.players){
    for(const [zone,list]of [['hand',p.hand],['discard',p.discard]] as const){const index=list.cards.indexOf(card);if(index>=0)return {playerId:p.id-1,zone,index};}
    for(const [index,list]of [p.active,...p.bench].entries())if([...list.cards,...list.tools,...list.energies.cards].includes(card))return {playerId:p.id-1,zone:index===0?'active':'bench',index:index===0?0:index-1};
  }
  return undefined;
}
/** Bind raw engine choices. Human-readable labels are deliberately not inputs. */
export function choiceBinding(prompt:any,state:State,value:any):ChoiceBinding {
  const viewer=prompt.getPerspectivePlayerId(),owner=state.players.find(p=>p.id===viewer)!;
  const visible=(card:Card|undefined)=>card&&!prompt.options?.isSecret?{cardId:printedId(card)}:{};
  const ref=(target:CardTarget)=>targetRef(state,viewer,target);
  switch(prompt.type){
    case 'Choose cards':case 'Order cards':return {sourceRef:listRef(state,viewer,prompt.cards,value),...visible(prompt.cards.cards[value])};
    case 'Choose energy':{const card=prompt.energy[value]?.card;return {sourceRef:card&&cardRef(state,viewer,card),...visible(card)};}
    case 'Choose pokemon':return {targetRef:ref(value)};
    case 'Choose prize':return {targetRef:{playerId:prompt.options?.useOpponentPrizes?1-(viewer-1):viewer-1,zone:'prompt',index:value}};
    case 'Attach energy':return {sourceRef:listRef(state,viewer,prompt.cardList,value.index),targetRef:ref(value.to),...visible(prompt.cardList.cards[value.index])};
    case 'Discard energy':case 'Move energy':return {sourceRef:ref(value.from),...(value.to?{targetRef:ref(value.to)}:{}),...visible(StateUtils.getTarget(state,owner,value.from).cards[value.index])};
    case 'Put damage':return {targetRef:ref(value.target),amount:value.damage};
    case 'Move damage':case 'Remove damage':return {sourceRef:ref(value.from),targetRef:ref(value.to),amount:prompt.options?.damageMultiple??10};
    case 'Choose attack':{const card=prompt.cards[value.index];return {sourceRef:cardRef(state,viewer,card),...visible(card)};}
    default:return {};
  }
}
