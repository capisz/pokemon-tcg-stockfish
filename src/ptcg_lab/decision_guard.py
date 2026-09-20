"""Visible-information progress guard for experimental policies, not rule changes.

A new policy may reopen and cancel the same optional effect forever. At an
identical player-visible position within one turn, retire a twice-tried action
while an untried alternative exists. Never assign a search target to this fallback.
"""
from __future__ import annotations
import copy
from .storage import digest

VERSION = 'visible-repetition-v1'


def action_key(action):
    return digest({key:value for key,value in action.items() if key != 'id'})


def position_key(observation):
    # History/log length and action revision are not changes to this information
    # set. Legally available actions encode once-per-turn and cost restrictions.
    return digest({**{key:observation.get(key) for key in ('playerId','decisionPlayer','turn','phase','players','prompt','stadium','knowledge')},
                   'actions':sorted(action_key(a) for a in observation['legalActions'])})


def forward_choices(observation, state):
    if state.get('turn') != observation['turn']:
        state.clear(); state.update(turn=observation['turn'], positions={})
    key=position_key(observation)
    counts=state['positions'].get(key,{})
    forward=[a for a in observation['legalActions'] if a.get('choiceOperation') != 'undo']
    if not forward:
        forward=list(observation['legalActions'])
    unused=[a for a in forward if counts.get(action_key(a),0)<2]
    # If every action recurred, choose the least-tried forward alternative. A
    # genuinely mandatory cycle may still truncate; it never fabricates a result.
    choices=unused or [a for a in forward if counts.get(action_key(a),0)==min(counts.get(action_key(x),0) for x in forward)]
    allowed=copy.deepcopy(observation)
    allowed['legalActions']=choices
    return allowed,key,len(choices)!=len(observation['legalActions'])


def record_choice(state, key, action):
    counts=state['positions'].setdefault(key,{})
    token=action_key(action)
    counts[token]=counts.get(token,0)+1
