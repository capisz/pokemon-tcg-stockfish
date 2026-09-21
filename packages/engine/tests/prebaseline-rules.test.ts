import test from 'node:test';
import assert from 'node:assert/strict';
import { State, GamePhase } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/state';
import { Player } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/player';
import { Store } from '../../../vendor/twinleaf/ptcg-server/src/game/store/store';
import { PokemonCardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/pokemon-card-list';
import { CardList } from '../../../vendor/twinleaf/ptcg-server/src/game/store/state/card-list';
import { GameError } from '../../../vendor/twinleaf/ptcg-server/src/game/game-error';
import { AttackEffect, UseStadiumEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/game-effects';
import { AfterDamageEffect, DealDamageEffect, PutCountersEffect } from '../../../vendor/twinleaf/ptcg-server/src/game/store/effects/attack-effects';
import { AttachEnergyPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/attach-energy-prompt';
import { ChooseCardsPrompt } from '../../../vendor/twinleaf/ptcg-server/src/game/store/prompts/choose-cards-prompt';
import { PlayCardAction, PlayerType, SlotType } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/play-card-action';
import { UseAbilityAction, UseStadiumAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/game-actions';
import { ResolvePromptAction } from '../../../vendor/twinleaf/ptcg-server/src/game/store/actions/resolve-prompt-action';
import { CARD_FACTORIES as F, registerCards } from '../src/catalog';
import { promptChoices, stagedChoices, STAGED_PROMPTS } from '../src/choices';

registerCards();

function board(source='TEF-129', target='DRI-12') {
  const state=new State(), store=new Store({onStateChange:()=>{}});
  const a=new Player(), b=new Player();a.id=1;b.id=2;
  a.active.cards=[F[source]()];b.active.cards=[F[target]()];
  for(const p of[a,b]){
    p.bench=Array.from({length:5},()=>new PokemonCardList());
    p.prizes=Array.from({length:6},()=>{const c=new CardList();c.cards=[F['MEE-1']()];return c;});
    p.deck.cards=Array.from({length:12},()=>F['MEE-1']());
  }
  state.players=[a,b];state.activePlayer=0;state.phase=GamePhase.PLAYER_TURN;state.turn=5;store.state=state;
  return{state,store,a,b};
}

function attach(target:PokemonCardList,...ids:string[]){
  const cards=ids.map(id=>F[id]());target.cards.push(...cards);target.energies.cards.push(...cards);return cards;
}

function resolveAll(store:Store,state:State){
  for(let i=0;i<100;i++){
    const p:any=state.prompts.find(p=>p.result===undefined);if(!p)return;
    let raw:any;
    if(p.type==='WaitPrompt'||p.type==='Alert')raw=true;
    else if(p.type==='Shuffle deck')raw=state.players.find(player=>player.id===p.playerId)!.deck.cards.map((_:unknown,index:number)=>index);
    else if(p.type==='Shuffle hand')raw=state.players.find(player=>player.id===p.playerId)!.hand.cards.map((_:unknown,index:number)=>index);
    else if(p.type==='Order cards')raw=p.cards.cards.map((_:unknown,index:number)=>index);
    else if(STAGED_PROMPTS.has(p.type)){
      let selected:any[]=[];
      for(let n=0;n<80;n++){
        const choices=stagedChoices(p,state,selected),finish=choices.find(c=>c.finish&&c.raw!==null),append=choices.find(c=>c.stage?.operation==='append');
        if(finish){raw=finish.raw;break;}if(!append)throw new Error(`No continuation for ${p.type}`);selected=append.stage!.selection;
      }
    } else raw=promptChoices(p,state).choices[0].raw;
    store.dispatch(new ResolvePromptAction(p.id,p.decode(raw,state)));
  }
  throw new Error('Prompt loop did not settle');
}

test('Handheld Fan uses the attacker Bench and requires one move chosen by the Fan owner',()=>{
  const {state,store,a,b}=board();state.phase=GamePhase.ATTACK;
  const energy=attach(a.active,'MEE-1')[0];a.bench[0].cards=[F['MEG-104']()];b.active.tools=[F['TWM-150']()];
  const attack=new AttackEffect(a,b,a.active.getPokemonCard()!.attacks[0]);
  store.reduceEffect(state,new AfterDamageEffect(attack,30));
  const prompt=state.prompts.find(p=>p.result===undefined) as AttachEnergyPrompt;
  assert.ok(prompt);assert.equal(prompt.playerId,b.id);assert.equal(prompt.options.min,1);assert.equal(prompt.options.max,1);
  const raw=[{index:a.active.cards.indexOf(energy),to:{player:PlayerType.TOP_PLAYER,slot:SlotType.BENCH,index:0}}];
  assert.ok(prompt.validate(prompt.decode(raw,state)));
  store.dispatch(new ResolvePromptAction(prompt.id,prompt.decode(raw,state)));
  assert.ok(a.bench[0].energies.cards.includes(energy));assert.ok(!a.active.energies.cards.includes(energy));
});

test('Handheld Fan ignores placed counters and does not prompt without movable Energy',()=>{
  for(const mode of['counters','no-energy'] as const){
    const {state,store,a,b}=board();state.phase=GamePhase.ATTACK;a.bench[0].cards=[F['MEG-104']()];b.bench[0].cards=[F['MEG-104']()];b.active.tools=[F['TWM-150']()];
    const attack=new AttackEffect(a,b,a.active.getPokemonCard()!.attacks[0]);
    if(mode==='counters'){attach(a.active,'MEE-1');store.reduceEffect(state,new PutCountersEffect(attack,20));}
    else store.reduceEffect(state,new AfterDamageEffect(attack,30));
    assert.equal(state.prompts.filter(p=>p.result===undefined).length,0,mode);
  }
});

test('Lumiose City is usable through Item lock, replaces the Stadium, and ends the turn after a zero-card search',()=>{
  const {state,store,a}=board();a.cannotPlayItemCards=true;
  const risky=F['MEG-127'](),lumiose=F['POR-77']();a.stadium.cards=[risky];a.hand.cards=[lumiose];
  assert.doesNotThrow(()=>store.dispatch(new PlayCardAction(a.id,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0})));
  assert.equal(a.stadium.cards[0],lumiose);assert.ok(a.discard.cards.includes(risky));
  a.deck.cards=Array.from({length:3},()=>F['MEE-1']());const before=state.activePlayer;
  store.dispatch(new UseStadiumAction(a.id));
  let prompt:any=state.prompts.find(p=>p.result===undefined);assert.equal(prompt.type,'Choose cards');assert.equal(prompt.options.min,0);
  store.dispatch(new ResolvePromptAction(prompt.id,prompt.decode([],state)));
  resolveAll(store,state);assert.notEqual(state.activePlayer,before);
});

test("Lillie's Determination draws eight at six Prizes and six otherwise, including an empty-deck result",()=>{
  for(const [prizes,draw] of [[6,8],[5,6]] as const){
    const {state,store,a}=board();a.prizes=a.prizes.slice(0,prizes);a.hand.cards=[F['MEG-119'](),F['MEE-2'](),F['MEE-3']()];a.deck.cards=Array.from({length:draw-2},()=>F['MEE-1']());
    store.dispatch(new PlayCardAction(a.id,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));resolveAll(store,state);
    assert.equal(a.hand.cards.length,draw);assert.equal(a.deck.cards.length,0);
  }
});

test('Jumbo Ice Cream rejects two one-Energy cards and heals 80 with three',()=>{
  for(const count of[2,3]){
    const {state,store,a}=board();const jumbo=F['PFL-91']() as any;a.active.damage=100;attach(a.active,...Array(count).fill('MEE-1'));a.hand.cards=[jumbo];
    assert.equal(jumbo.canPlay(store,state,a),count===3);
    if(count===3){store.dispatch(new PlayCardAction(a.id,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));resolveAll(store,state);assert.equal(a.active.damage,20);}
  }
});

test('Special Red Card is rejected at four opposing Prizes and allowed at three',()=>{
  for(const prizes of[4,3]){const {state,store,a,b}=board();b.prizes=b.prizes.slice(0,prizes);assert.equal((F['CRI-82']() as any).canPlay(store,state,a),prizes===3);}
});

test('Spiky Energy puts two counters on Land Crash attacker and enables the exact 120 follow-up knockout',()=>{
  const {state,store,a,b}=board('TEF-129','DRI-12');state.phase=GamePhase.ATTACK;attach(b.active,'JTG-159');
  const landCrash=new AttackEffect(a,b,a.active.getPokemonCard()!.attacks[0]);store.reduceEffect(state,new DealDamageEffect(landCrash,90));
  assert.equal(a.active.damage,20);assert.equal(b.active.damage,90);
  const scissors=new AttackEffect(b,a,b.active.getPokemonCard()!.attacks[0]);store.reduceEffect(state,new DealDamageEffect(scissors,120));
  assert.equal(a.active.damage,140);assert.equal(a.active.getPokemonCard()!.hp,140);
});

test('Mist Energy blocks Phantom Dive Bench counters while attack damage still lands',()=>{
  const {state,store,a,b}=board('TWM-130','MEG-104');state.phase=GamePhase.ATTACK;b.bench[0].cards=[F['DRI-12']()];attach(b.bench[0],'TEF-161');
  const attack=new AttackEffect(a,b,a.active.getPokemonCard()!.attacks[1]);store.reduceEffect(state,new DealDamageEffect(attack,200));
  const counters=new PutCountersEffect(attack,60);counters.target=b.bench[0];store.reduceEffect(state,counters);
  assert.equal(b.active.damage,200);assert.equal(b.bench[0].damage,0);
});

test('Itchy Pollen blocks Items only and still permits Supporters, Stadiums, and Pokemon Tools',()=>{
  const candidates=[
    ['PFL-91',true],
    ['MEG-119',false],
    ['POR-77',false],
    ['TWM-150',false],
  ] as const;
  for(const [cardId,blocked] of candidates){
    const {state,store,a}=board();a.cannotPlayItemCards=true;const card=F[cardId]();a.hand.cards=[card];if(cardId==='PFL-91'){a.active.damage=10;attach(a.active,'MEE-1','MEE-1','MEE-1');}
    const play=()=>store.dispatch(new PlayCardAction(a.id,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));
    if(blocked)assert.throws(play,GameError);else assert.doesNotThrow(play,cardId);
  }
});

test('Run Away Draw draws, shuffles itself and attachments, promotes a new Active, and fails on an empty deck',()=>{
  {
    const {state,store,a}=board();a.bench[0].cards=[F['MEG-104']()];const attached=attach(a.active,'MEE-1','TWM-150');const dud=a.active.getPokemonCard()!;
    const before=a.hand.cards.length;store.dispatch(new UseAbilityAction(a.id,'Run Away Draw',{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));resolveAll(store,state);
    assert.equal(a.hand.cards.length,before+3);assert.equal(a.active.getPokemonCard()!.name,'Mega Kangaskhan ex');
    assert.ok(a.deck.cards.includes(dud));assert.ok(attached.every(card=>a.deck.cards.includes(card)));
  }
  {
    const {state,store,a}=board();a.deck.cards=[];
    store.dispatch(new UseAbilityAction(a.id,'Run Away Draw',{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));
    assert.throws(()=>resolveAll(store,state),GameError);
  }
});

test("Xerosic's Machinations is mandatory to three and unplayable at three or fewer",()=>{
  for(const count of[3,5]){
    const {state,store,a,b}=board();b.hand.cards=Array.from({length:count},()=>F['MEE-1']());const x=F['SFA-64']() as any;a.hand.cards=[x];assert.equal(x.canPlay(store,state,a),count>3);
    if(count>3){store.dispatch(new PlayCardAction(a.id,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));const p=state.prompts.find(q=>q.result===undefined) as ChooseCardsPrompt;assert.equal(p.options.min,2);assert.equal(p.options.max,2);store.dispatch(new ResolvePromptAction(p.id,p.decode([0,1])));assert.equal(b.hand.cards.length,3);}
  }
});

test('Eri selects Items only, allows zero, and caps the selection at two',()=>{
  const {state,store,a,b}=board();const eri=F['TEF-146']();a.hand.cards=[eri];b.hand.cards=[F['PFL-91'](),F['CRI-82'](),F['TWM-150'](),F['MEG-119']()];
  store.dispatch(new PlayCardAction(a.id,0,{player:PlayerType.BOTTOM_PLAYER,slot:SlotType.ACTIVE,index:0}));
  const p=state.prompts.find(q=>q.result===undefined) as ChooseCardsPrompt;assert.equal(p.options.min,0);assert.equal(p.options.max,2);
  assert.ok(p.validate(p.decode([])));assert.ok(p.validate(p.decode([0,1])));
  assert.equal(p.validate(p.decode([2])),false);assert.equal(p.validate(p.decode([0,1,2])),false);
  store.dispatch(new ResolvePromptAction(p.id,p.decode([])));assert.equal(b.discard.cards.length,0);
});
