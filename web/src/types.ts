export type Card = {
  id: string;
  name: string;
  kind: 'pokemon' | 'trainer' | 'energy';
  types?: string[];
  text?: string;
  imageUrl?: string;
  powers?: {name: string; text: string}[];
  hp?: number;
  stage?: string;
  prizeValue?: number;
  attacks?: { name: string; damage: number | string; cost: string[]; text?: string }[];
};

export type Pokemon = {
  card: Card;
  damage: number;
  attachments?: Card[];
  slotIndex?: number;
  energy: string[];
  tools: string[];
  conditions: string[];
};

export type Player = {
  id: number;
  name: string;
  active: Pokemon | null;
  bench: Pokemon[];
  hand: Card[];
  handCount: number;
  deckCount: number;
  prizesRemaining: number;
  discard: Card[];
};

export type CardRef = {playerId:number; zone:'hand'|'active'|'bench'|'discard'|'prompt'; index?:number};
export type ChoiceBinding = {sourceRef?:CardRef;targetRef?:CardRef;cardId?:string;amount?:number};
export type Action = { id: string; label: string; type: string; cardId?: string; target?: string; sourceRef?:CardRef; targetRef?:CardRef; choiceOperation?:'append'|'finish'|'undo'; selectionCount?:number; choiceRefs?:ChoiceBinding[] };
export type Observation = {
  schemaVersion: number;
  decisionIndex?: number;
  playerId: number;
  decisionPlayer: number;
  turn: number;
  phase: string;
  status: string;
  players: Player[];
  ownDeck: { cardId: string; name: string; count: number }[];
  legalActions: Action[];
  history: string[];
  stadium?: {owner: number; card: Card} | null;
  prompt?: { type: string; message: string; cards?: Card[]; hands?: Card[][]; selection?:{index?:number;name?:string}[]; selectionCount?:number; selectedChoices?:ChoiceBinding[]; min?:number; max?:number; canFinish?:boolean; canUndo?:boolean };
  warnings: string[];
};

export type Replay = {
  schemaVersion: number;
  id: string;
  seed?: number;
  decks: string[];
  engineVersion: string;
  status: 'finished' | 'truncated' | 'error';
  outcome: { winner: 0 | 1 | null; reason: string } | null;
  frames: ViewFrame[];
  modelVersion?:string;
  policy?:string;
  policyContext?:PolicyContext;
  warnings: string[];
};

export type ReplaySummary = { id: string; decks: string[]; status: string; frames: number };
export type PositionSummary = { id: string; title: string; createdAt: string; playerId: number; decisionIndex: number; sourceReplayId: string; engineVersion: string };
export type SavedPosition = PositionSummary & { observation: Observation };
export type SampledContinuation = {
  conditional: true;
  representative: true;
  description: string;
  opponentArchetype: string;
  steps: { playerId: 0 | 1; label: string; type: string }[];
  end: 'terminal' | 'cutoff';
  outcome?: { winner: 0 | 1 | null; reason: string };
};
export type Deck = {
  id: string;
  name: string;
  archetype: string;
  formatDate: string;
  role?: string;
  cards: { cardId: string; name: string; count: number }[];
  validation: { status: string; notes: string[] };
  playable?: boolean;
  support?: { playable?: boolean; warnings?: string[]; errors?: string[]; missingCards?: string[] };
  warnings?: string[];
};

export type Analysis = {
  evaluation: {
    status: 'heuristic' | 'trained' | 'unavailable';
    score: number | null;
    expectedResult: number | null;
    winProbability: number | null;
    drawProbability: number | null;
    lossProbability: number | null;
    modelVersion: string;
    components: { name: string; value: number }[];
    calibrated: boolean;
    description: string;
  };
  alternatives: { actionId: string; label: string; score: number | null; expectedResult?: number | null; visits: number; description: string; uncertainty?: number | null; continuation?: SampledContinuation }[];
  search?: { status: string; method?: string; iterations?: number; elapsedMs?: number; description?: string; warnings?: string[] };
  decisionReview?: {
    status: 'experimental' | 'unavailable';
    opportunityLoss: number | null;
    bestTestedActionId?: string;
    playedActionId: string | null;
    playedExpectedResult?: number;
    bestTestedExpectedResult?: number;
    playedVisits?: number;
    bestTestedVisits?: number;
    differenceStandardError?: number | null;
    description: string;
    warnings: string[];
  };
  beliefs: { archetype: string; probability: number }[];
  warnings: string[];
};

export type Job = { id: string; status: string; progress?: number | string; error?: string | { message?: string }; replayId?: string };

export type Match = {
  id: string; schemaVersion: number; revision: number; mode: 'practice'|'benchmark';
  status: 'active'|'paused'|'between-games'|'completed'|'abandoned'; gameNumber: number; score: [number,number];
  frameCursor?: number; observation: Observation; thinking: boolean; knownList: boolean; ownDeckId: string; modelVersion: string;
  engineTurnRemainingMs: number; engineTurnBudgetMs: number; error?: string; warnings: string[];
  nextStarterChooser?: number|null; gameResult?: {winner:number|null;reason:string}; replayIds?: string[];
  opponentList?: {cardId:string; name:string; count:number}[];
};
export type Teaching = {
  id: string; title: string; familyId: string; partition: string; reviewStatus: string; trainingEligible: boolean;
  playerId?: number; rulesAuditStatus?: string; mechanicsAudit?:{validatedActionIds?:string[]};
  source?: {title?:string; author?:string; pages?:number[]; page?:number; contentHash?:string; kind?:string};
  observation?: Observation; acceptableActionIds?: string[]; conditionalReasoning?: string; criticalResources?: string;
};

export type ViewFrame = { cursor?:number; decisionIndex:number; actor:number; action?:Action|null; priorAction?:Action|null; observation:Observation; revision?:number; gameNumber?:number };
export type PolicyContext = {policy?:string;modelVersion?:string;trainingStatus?:string;learnsDuringRun?:boolean;opponentPopulation?:string;computeBudget?:string};
export type FrameFeed = { schemaVersion:number; frames:ViewFrame[]; nextCursor:number; status:string; hasMore:boolean; replayId?:string;runId?:string;gameId?:string;policyContext?:PolicyContext;error?:string|{message?:string} };
export type ModelOption = {dataTier?:string;id:string; name:string; modelKind:string; modelHash:string; status:string; description:string; valueTrained:boolean; calibrated:boolean; parameters?:number; featureVersion?:string};

export type LearningGame = {id:string; workerIndex:number; status:string; purpose:'collection'|'comparison'; archetypes:[string,string]; decisionIndex:number; turn:number; frameCursor:number; replayAvailable?:boolean; createdAt?:string};
export type LearningPolicy = {version:string; name:string};
export type LearningRun = {
 schemaVersion:1; id:string; revision:number; status:'running'|'pausing'|'paused'|'stopped'|'waiting-for-play'|'failed';
 phase:'initializing'|'collecting'|'investigating'|'training'|'comparing'|'checkpointed'; cycle:number;
 incumbent:LearningPolicy; candidate?:LearningPolicy; guidePolicy?:LearningPolicy;
 configuration:{seed?:number;keepAwake?:boolean;[key:string]:unknown}; activeGames:LearningGame[]; error?:string; pauseReason?:string;
 metrics:{completedGames:number;errors:number;truncatedGames:number;searchDecisions:number;searchTargets:number;peakMemoryBytes:number;managedBytes:number;freeBytes?:number;loss?:number|null;
 coverage?:{scheduledPairs:number;totalPairs:number;completedPairs:number;archetypes:number};
 comparison?:{status:'running'|'completed'|'unavailable';completedGames:number;totalGames:number;mean:number|null;lower:number|null;upper:number|null;adopted:boolean;notes:string[]};
 guideAgreement?:{status:'measured'|'unavailable';positions:number;acceptableActionAccuracy:number|null};
 };
};
