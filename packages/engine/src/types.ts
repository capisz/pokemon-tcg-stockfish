export interface CardView {
  id: string; name: string; text?:string; imageUrl?:string; powers?:{name:string;text:string}[]; kind: 'pokemon' | 'trainer' | 'energy'; types?: string[];
  hp?: number; stage?: string; prizeValue?: number;
  attacks?: {name: string; damage: number; text?:string; cost: string[]}[];
}
export interface PokemonView {
  card: CardView; damage: number; energy: string[]; tools: string[]; conditions: string[];
  attachments?: CardView[]; slotIndex?: number;
}
export interface PlayerView {
  id: number; name: string; active: PokemonView | null; bench: PokemonView[];
  hand: CardView[]; handCount: number; deckCount: number; prizesRemaining: number; discard: CardView[];
}
/** References resolve only within the observation's decision revision. */
export interface ActionRef { playerId:number; zone:'hand'|'active'|'bench'|'discard'|'prompt'; index?:number }
export interface ChoiceBinding { sourceRef?:ActionRef; targetRef?:ActionRef; cardId?:string; amount?:number }
export interface LegalAction extends ChoiceBinding { id: string; label: string; type: string; target?: string; choiceOperation?: 'append'|'finish'|'undo'; selectionCount?:number; choiceRefs?:ChoiceBinding[] }
export interface Observation {
  schemaVersion: 1; playerId: number; decisionPlayer: number; turn: number; phase: string;
  status: 'running' | 'finished' | 'error'; players: PlayerView[];
  stadium?: {owner:number;card:CardView}|null; knowledge?: any[];
  ownDeck: {cardId: string; name: string; count: number}[];
  searchPosition?: import('./belief-state').PublicPosition; searchUnavailableReason?: string;
  legalActions: LegalAction[]; history: string[]; prompt?: {type: string; message: string; cards?: CardView[]; hands?: CardView[][]; selectionCount?:number; selection?:any[]; selectedChoices?:ChoiceBinding[]; min?:number; max?:number; canFinish?:boolean; canUndo?:boolean}; warnings: string[];
}
export interface ReplayFrame { decisionIndex: number; actor: number; action: LegalAction | null; observations: [Observation, Observation] }
export interface Replay {
  schemaVersion: 1; id: string; seed: number; firstPlayer?:0|1; decks: string[]; engineVersion: string;
  status: 'finished' | 'truncated' | 'error'; outcome: {winner: 0 | 1 | null; reason: string} | null;
  frames: ReplayFrame[]; warnings: string[];
  /** Private research artifact: do not serve as a public game transcript. */
  visibility: 'private-research';
  chance: {decisionIndex: number; type: string; result: unknown}[];
}
