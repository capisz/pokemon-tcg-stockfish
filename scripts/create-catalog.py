"""Rebuild the explicit research card catalog and five authored starter lists.

No competition code/data is used. Card effects come from the pinned Twinleaf
source. Lists are engineering baselines, not attributed tournament lists.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / 'vendor/twinleaf/ptcg-server/src/sets'
SV = '10-scarlet-and-violet/'
ME = '11-mega-evolution/set-mega-evolution/'
entries = [
    ('TWM-128', SV+'set-twilight-masquerade/dreepy', 'Dreepy'),
    ('TWM-129', SV+'set-twilight-masquerade/drakloak', 'Drakloak'),
    ('TWM-130', SV+'set-twilight-masquerade/dragapult-ex', 'Dragapultex'),
    ('TWM-25', SV+'set-twilight-masquerade/teal-mask-ogerpon-ex', 'TealMaskOgerponex'),
    ('TWM-95', SV+'set-twilight-masquerade/munkidori', 'Munkidori'),
    ('TWM-107', SV+'set-twilight-masquerade/hawlucha', 'Hawlucha'),
    ('TWM-145', SV+'set-twilight-masquerade/carmine', 'Carmine'),
    ('TEF-123', SV+'set-temporal-forces/raging-bolt-ex', 'RagingBoltex'),
    ('TEF-144', SV+'set-temporal-forces/buddy-buddy-poffin', 'BuddyBuddyPoffin'),
    ('TEF-150', SV+'set-temporal-forces/hand-trimmer', 'HandTrimmer'),
    ('DRI-134', SV+'set-destined-rivals/marnies-impidimp', 'MarniesImpidimp'),
    ('DRI-135', SV+'set-destined-rivals/marnies-morgrem', 'MarniesMorgrem'),
    ('DRI-136', SV+'set-destined-rivals/marnies-grimmsnarl-ex', 'MarniesGrimmsnarlex'),
    ('DRI-11', SV+'set-destined-rivals/dwebble', 'Dwebble'),
    ('DRI-12', SV+'set-destined-rivals/crustle', 'Crustle'),
    ('DRI-10', SV+'set-destined-rivals/shaymin', 'Shaymin'),
    ('SFA-61', SV+'set-shrouded-fable/nightly-stretcher', 'NightlyStretcher'),
    ('SFA-38', SV+'set-shrouded-fable/fezandipiti-ex', 'Fezandipitiex'),
    ('SCR-133', SV+'set-stellar-crown/crispin', 'Crispin'),
    ('SCR-137', SV+'set-stellar-crown/gravity-gemstone', 'GravityGemstone'),
    ('MEG-76', ME+'riolu', 'Riolu'),
    ('MEG-77', ME+'mega-lucario-ex', 'MegaLucarioex'),
    ('MEG-119', ME+'lillies-determination', 'LilliesDetermination'),
    ('MEG-116', ME+'fighting-gong', 'FightingGong'),
    ('MEG-121', ME+'mega-signal', 'MegaSignal'),
    ('MEG-132', ME+'wallys-compassion', 'WallysCompassion'),
    # Exact MEG reprints use the same base effect classes as upstream other-prints.ts.
    ('MEG-114', SV+'set-paldea-evolved/boss-orders', 'BossOrders'),
    ('MEG-125', '03-ex-ruby-and-sapphire/set-ex-holon-phantoms/rare-candy', 'RareCandy'),
    ('MEG-130', 'set-base-set/switch', 'Switch'),
    ('MEG-131', SV+'set-scarlet-and-violet/ultra-ball', 'UltraBall'),
    ('MEG-115', SV+'set-scarlet-and-violet/energy-switch', 'EnergySwitch'),
]
for number, color in [(1,'grass'),(2,'fire'),(4,'lightning'),(5,'psychic'),(6,'fighting'),(7,'darkness')]:
    entries.append((f'SVE-{number}', SV+f'set-scarlet-and-violet-energy/{color}-energy', color.title()+'Energy'))

reprints = {'MEG-114':"Boss's Orders", 'MEG-125':'Rare Candy', 'MEG-130':'Switch', 'MEG-131':'Ultra Ball', 'MEG-115':'Energy Switch'}
metadata = {}
imports = []
factories = []
for i, (card_id, path, cls) in enumerate(entries):
    source = (SETS / (path+'.ts')).read_text()
    def field(name):
        match = re.search(r'public '+name+r'(?:\s*:\s*\w+)?\s*=\s*([\'"])((?:\\.|(?!\1).)*)\1', source)
        if not match:
            raise ValueError((path, name))
        return match.group(2).replace("\\'", "'").replace('\\"', '"')
    name = reprints.get(card_id) or field('name')
    engine_name = f'{name} MEG' if card_id in reprints else field('fullName')
    mark = 'I' if card_id in reprints else field('regulationMark')
    metadata[card_id] = {'cardId':card_id,'name':name,'engineName':engine_name,'regulationMark':mark}
    imports.append(f"import {{ {cls} as C{i} }} from '../../../vendor/twinleaf/ptcg-server/src/sets/{path}';")
    overrides = {'set':'MEG','setNumber':card_id.split('-')[1], 'regulationMark':'I','fullName':engine_name} if card_id in reprints else {}
    factories.append(f'  {json.dumps(card_id)}: () => Object.assign(new C{i}(), {json.dumps(overrides)}),')

recipes = [
 ('dragapult','Dragapult ex', [('TWM-128',4),('TWM-129',4),('TWM-130',3),('SFA-38',2),('TWM-95',1),('TEF-144',4),('MEG-131',4),('MEG-119',4),('SCR-133',3),('MEG-114',3),('SFA-61',4),('MEG-125',4),('MEG-130',3),('TWM-145',2),('SVE-2',6),('SVE-5',6),('SVE-7',3)]),
 ('raging-bolt','Raging Bolt ex / Teal Mask Ogerpon ex', [('TEF-123',4),('TWM-25',4),('SFA-38',1),('TWM-95',1),('MEG-131',4),('MEG-119',4),('SCR-133',4),('MEG-114',3),('SFA-61',4),('MEG-130',3),('TWM-145',4),('MEG-115',4),('MEG-116',2),('SVE-1',8),('SVE-4',4),('SVE-6',4),('SVE-7',2)]),
 ('grimmsnarl',"Marnie's Grimmsnarl ex", [('DRI-134',4),('DRI-135',4),('DRI-136',3),('TWM-95',2),('SFA-38',1),('TEF-144',4),('MEG-131',4),('MEG-119',4),('MEG-114',3),('SFA-61',4),('MEG-125',4),('MEG-130',3),('TWM-145',3),('SVE-7',17)]),
 ('mega-lucario','Mega Lucario ex', [('MEG-76',4),('MEG-77',4),('TWM-107',2),('SFA-38',2),('MEG-131',4),('MEG-119',4),('MEG-116',4),('MEG-114',3),('SFA-61',4),('MEG-130',3),('TWM-145',3),('MEG-121',2),('MEG-132',2),('SVE-6',19)]),
 ('crustle','Crustle', [('DRI-11',4),('DRI-12',4),('DRI-10',2),('TWM-95',2),('TEF-144',4),('MEG-131',4),('MEG-119',4),('MEG-114',4),('SFA-61',4),('MEG-130',3),('SCR-133',2),('TWM-145',2),('SCR-137',3),('TEF-150',4),('SVE-1',10),('SVE-7',4)]),
]
decks = ROOT/'decks'
decks.mkdir(exist_ok=True)
for id, name, cards in recipes:
    assert sum(n for _,n in cards)==60, (id,sum(n for _,n in cards))
    deck={'id':id,'name':name,'archetype':id,'formatDate':'2026-09-17','version':1,
          'source':'Authored engineering baseline; not a tournament list.',
          'cards':[{**metadata[k],'count':n} for k,n in cards],
          'validation':{'status':'experimental','notes':['60 cards; H/I regulation marks or basic Energy. Required effects are implemented but the entire interaction space is not certified.','Starter ratios prioritize simulation coverage. Expert matchup and tournament-list review remain pending.']}}
    (decks/f'{id}.json').write_text(json.dumps(deck,indent=2)+'\n')

out = ROOT/'packages/engine/src/catalog.ts'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text('// Generated by scripts/create-catalog.py. Explicit imports prevent loading every set.\n'+ '\n'.join(imports)+'''
import { CardManager } from '../../../vendor/twinleaf/ptcg-server/src/game/cards/card-manager';
import { Card } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card';
import { SuperType, Stage, EnergyType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/card/card-types';
import dragapult from '../../../decks/dragapult.json';
import ragingBolt from '../../../decks/raging-bolt.json';
import grimmsnarl from '../../../decks/grimmsnarl.json';
import megaLucario from '../../../decks/mega-lucario.json';
import crustle from '../../../decks/crustle.json';
import format from '../../../formats/standard-2026-09-17.json';
export const CARD_FACTORIES: Record<string, () => Card> = {
'''+ '\n'.join(factories)+'\n};\n'+ 'export const CARD_METADATA = '+json.dumps(metadata,indent=2)+''' as const;
export type DeckManifest = typeof dragapult;
const manifests = [dragapult, ragingBolt, grimmsnarl, megaLucario, crustle];
let registered = false;
export function registerCards(): void {
  if (registered) return;
  const manager = CardManager.getInstance();
  for (const factory of Object.values(CARD_FACTORIES)) manager.defineCard(factory());
  registered = true;
}
export function validateDeck(deck: DeckManifest): string[] {
  const errors: string[] = [];
  if(deck.formatDate!==format.date) errors.push('Unsupported format freeze: '+deck.formatDate);
  if (deck.cards.reduce((n,c)=>n+c.count,0)!==60) errors.push('Deck must contain exactly 60 cards.');
  const counts = new Map<string,number>(); let basic = false;
  for (const entry of deck.cards) {
    if (!Number.isInteger(entry.count) || entry.count < 1) errors.push('Invalid count: '+entry.cardId);
    const factory = CARD_FACTORIES[entry.cardId];
    if (!factory) { errors.push('Unsupported card: '+entry.cardId); continue; }
    const card = factory();
    const set = entry.cardId.split('-')[0] as keyof typeof format.sets;
    if(!format.sets[set] || format.sets[set].released>format.date) errors.push('Unreleased or unverified set: '+set);
    if (card.fullName !== entry.engineName) errors.push('Engine mapping mismatch: '+entry.cardId);
    const energy = card.superType===SuperType.ENERGY && card.energyType===EnergyType.BASIC;
    if (!energy && !['H','I','J'].includes(card.regulationMark)) errors.push('Rotated card: '+entry.cardId);
    if (!energy) counts.set(card.name,(counts.get(card.name)??0)+entry.count);
    if (card.superType===SuperType.POKEMON && (card as any).stage===Stage.BASIC) basic=true;
  }
  for (const [name,count] of counts) if(count>4) errors.push('More than four copies: '+name);
  if(!basic) errors.push('Deck needs a Basic Pokemon.');
  return errors;
}
export function getDecks() {
  return manifests.map(d=>({...d,cardCount:d.cards.reduce((n,c)=>n+c.count,0),support:{playable:validateDeck(d).length===0,errors:validateDeck(d)}}));
}
export function getDeck(id: string): DeckManifest {
  const deck = manifests.find(d=>d.id===id);
  if(!deck) throw new Error('Unknown deck: '+id);
  const errors = validateDeck(deck);
  if(errors.length) throw new Error(errors.join(' '));
  return deck;
}
export function deckNames(id: string): string[] {
  return getDeck(id).cards.flatMap(c=>Array(c.count).fill(c.engineName));
}
export function instantiateDeck(id: string): Card[] {
  registerCards();
  return getDeck(id).cards.flatMap(c=>Array.from({length:c.count},()=>CARD_FACTORIES[c.cardId]()));
}
export function printedId(card: Card): string {
  return Object.entries(CARD_METADATA).find(([,m])=>m.engineName===card.fullName)?.[0] ?? `${card.set}-${card.setNumber}`;
}
''')
print(f'Generated {len(entries)} card mappings and {len(recipes)} 60-card decks.')
