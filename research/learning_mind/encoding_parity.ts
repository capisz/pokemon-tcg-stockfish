import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';

function canonical(value:any):string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
}
function digest(value:any):string { return createHash('sha256').update(canonical(value)).digest('hex'); }
function cardId(card:any):string { return String(card?.id ?? card?.cardId ?? card?.name ?? 'unknown'); }
function normalizedRef(value:any):any {
  if (!value || typeof value !== 'object') return null;
  return {playerId:value.playerId ?? null, zone:value.zone ?? null,
          index:['hand','prompt'].includes(value.zone) ? null : (value.index ?? null)};
}
function semantics(action:any):any {
  return {type:action.type ?? null, label:String(action.label ?? '').trim().replace(/\s+/g, ' ').toLowerCase(),
    cardId:action.cardId ?? null, target:action.target ?? null, choiceOperation:action.choiceOperation ?? null,
    sourceRef:normalizedRef(action.sourceRef), targetRef:normalizedRef(action.targetRef),
    choiceRefs:(action.choiceRefs ?? []).map((item:any) => ({sourceRef:normalizedRef(item.sourceRef),
      targetRef:normalizedRef(item.targetRef), cardId:item.cardId ?? null, amount:item.amount ?? null}))};
}
function encodeStructure(observation:any, tracker:any):any {
  const actor=observation.playerId;
  const own=observation.players.find((player:any) => player.id === actor);
  const other=observation.players.find((player:any) => player.id !== actor);
  const keys:string[]=['global'];
  for (const player of [own, other]) {
    if (player.active) keys.push(`p${player.id}:active:0`);
    (player.bench ?? []).forEach((_item:any,index:number) => keys.push(`p${player.id}:bench:${index}`));
  }
  if (observation.stadium) keys.push('stadium');
  for (const [zone,cards] of [['own-hand',own.hand ?? []],['own-discard',own.discard ?? []],['opponent-discard',other.discard ?? []]] as any[]) {
    const counts=new Map<string,number>();
    for (const card of cards) counts.set(cardId(card),(counts.get(cardId(card)) ?? 0)+1);
    for (const identifier of [...counts.keys()].sort()) keys.push(`summary:${zone}:${identifier}`);
  }
  for (const owner of Object.keys(tracker.seats ?? {}).sort()) {
    const facts=tracker.seats[owner];
    for (const identifier of Object.keys(facts.revealedHand ?? {}).sort()) keys.push(`known-hand:${owner}:${identifier}`);
    (facts.knownDeckOrder ?? []).forEach((_id:string,index:number) => keys.push(`known-deck:${owner}:${index}`));
  }
  (tracker.lingeringEffects ?? []).forEach((_effect:string,index:number) => keys.push(`effect:${index}`));
  const actionKeys=[...new Set((observation.legalActions ?? []).map((action:any) => digest({version:'legal-action-equivalence-v1',...semantics(action)})))].sort();
  const record={schemaVersion:'learning-mind-visible-tokens-v1',trackerHash:tracker.hash ?? null,tokenKeys:keys,actionKeys};
  return {tokenKeys:keys,actionKeys,identity:digest(record)};
}

const input=JSON.parse(readFileSync(0,'utf8'));
process.stdout.write(`${JSON.stringify(encodeStructure(input.observation,input.tracker))}\n`);
