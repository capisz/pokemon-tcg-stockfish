import copy
from ptcg_lab.decision_guard import forward_choices,record_choice


def choices(observation):
    obs=copy.deepcopy(observation)
    obs.update(turn=3,legalActions=[{'id':'1:0','type':'ability','label':'Optional draw'}, {'id':'1:1','type':'end','label':'End turn'}])
    return obs


def test_repeated_cancel_loop_retires_unchanged_optional_action(observation):
    obs=choices(observation);state={}
    for revision in range(2):
        permitted,key,filtered=forward_choices(obs,state)
        assert not filtered
        record_choice(state,key,obs['legalActions'][0])
        obs['legalActions'][0]['id']=f'{revision+2}:0'
        obs['legalActions'][1]['id']=f'{revision+2}:1'
        obs['history']=['An optional effect was cancelled']*(revision+1)
    permitted,key,filtered=forward_choices(obs,state)
    assert filtered and [a['type'] for a in permitted['legalActions']]==['end']
    # The environment's complete legal set is untouched.
    assert len(obs['legalActions'])==2
    reconstructed=copy.deepcopy(state)
    assert forward_choices(obs,reconstructed)==forward_choices(obs,state)


def test_new_visible_state_or_new_turn_can_reconsider_action(observation):
    obs=choices(observation);state={}
    _,key,_=forward_choices(obs,state)
    for _ in range(2):record_choice(state,key,obs['legalActions'][0])
    changed=copy.deepcopy(obs);changed['players'][0]['handCount']+=1
    assert len(forward_choices(changed,state)[0]['legalActions'])==2
    obs['turn']+=1
    assert len(forward_choices(obs,state)[0]['legalActions'])==2


def test_forced_action_is_never_removed(observation):
    obs=choices(observation);obs['legalActions']=obs['legalActions'][:1];state={}
    for _ in range(5):
        selected,key,_=forward_choices(obs,state)
        assert len(selected['legalActions'])==1
        record_choice(state,key,selected['legalActions'][0])
