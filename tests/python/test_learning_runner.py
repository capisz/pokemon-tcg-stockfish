"""Supervisor recovery tests use real journals/projections with a bounded engine."""
from __future__ import annotations
import copy
import queue
from contextlib import contextmanager
from concurrent.futures import Future

import pytest

from ptcg_lab.config import Settings
from ptcg_lab.engine import EngineError
from ptcg_lab.learning import LearningService
from ptcg_lab.learning_games import LearningPaused, LearningYield, run_game
from ptcg_lab.learning_protocol import collection_assignment
from ptcg_lab.storage import Store


class Engine:
    def __init__(self, observation):
        self.base, self.timeout, self.index = observation, 10, 0
        self.history, self.failures = [], 0
        self.fail_after_step = False
        self.health = {"engineVersion": "engine-v1", "engineBuildHash": "build-v1"}

    def view(self, player):
        result = copy.deepcopy(self.base)
        result.update(playerId=player, decisionPlayer=self.index % 2, turn=self.index//2+1,
                      status="finished" if self.index == 6 else "running", phase="PLAYER_TURN")
        result["legalActions"] = [{"id": f"attack-{self.index}", "type": "attack", "label": "Attack"}] if self.index < 6 and player == self.index%2 else []
        result["searchPosition"] = {"neverPublic": True}
        for side in result["players"]:
            side["hand"] = [{"id": f"private-hand-{player}", "name": "Secret"}] if side["id"] == player else []
        return result

    def request(self, method, params=None):
        params = params or {}
        if method == "health": return self.health.copy()
        if method == "reset":
            self.index, self.history = 0, []
            return {}
        if method == "observe":
            if self.fail_after_step:
                self.fail_after_step = False
                raise EngineError("Crashed after applying unacknowledged step")
            return self.view(params["playerId"])
        if method == "step":
            assert params["actionId"] == f"attack-{self.index}"
            self.history.append({"decisionIndex": self.index, "actor": self.index%2,
                "action": self.view(self.index%2)["legalActions"][0], "observations": [self.view(0),self.view(1)]})
            self.index += 1
            if self.failures:
                self.failures -= 1
                self.fail_after_step = True
            return {}
        if method == "replay":
            return {"engineVersion": "engine-v1", "status": "finished" if self.index == 6 else "running",
                    "outcome": {"winner": 0, "reason": "rules-terminal"} if self.index == 6 else None,
                    "frames": self.history + [{"decisionIndex": self.index, "actor": self.index%2, "action": None,
                                               "observations": [self.view(0), self.view(1)]}]}
        raise AssertionError(method)

    def close(self): pass


class Pool:
    def __init__(self, engine): self.engine = engine
    def acquire(self, **kwargs): return self.engine
    def release(self, engine): pass
    @contextmanager
    def lease(self, **kwargs): yield self.engine


class ManualExecutor:
    def __init__(self): self.calls=[]
    def submit(self,*args):
        self.calls.append(args)
        future=Future(); future.set_result(None)
        return future
    def shutdown(self,**kwargs): pass


@pytest.fixture
def service(tmp_path, observation, monkeypatch):
    monkeypatch.setattr('ptcg_lab.resources.process_memory',lambda:1000)
    monkeypatch.setattr('ptcg_lab.learning.process_memory',lambda:1000)
    registry=[{"id":f"deck-{i}","archetype":f"type-{i//2}","role":"main" if i%2==0 else "training-variant","listHash":f"hash-{i}"} for i in range(10)]
    settings=Settings(root=tmp_path,data=tmp_path/'data',min_free_bytes=0)
    service=LearningService(settings,Store(settings.data),Pool(Engine(observation)),lambda:registry)
    service.executor.shutdown();service.executor=ManualExecutor()
    yield service
    service.close()


def game(service, max_decisions=3000):
    public=service.create({'keepAwake':False,'maxDecisions':max_decisions,'searchBudgetMs':0})
    run=service.update(public['id'],phase='collecting',guidePolicy={'path':'heuristic','hash':'heuristic','name':'heuristic'})
    assignment=collection_assignment(0,[d['id'] for d in run['decks']],[run['incumbent']])
    identifier=service._new_game(run,assignment,0,'collection',0)
    service.update(run['id'],pendingGames=[identifier],collectionCursor=1)
    return run['id'],identifier


def test_acknowledged_frames_restart_resume_and_projection(service):
    run_id,game_id=game(service)
    original=service.commit_frame
    def pause_at_boundary(record,*args):
        original(record,*args)
        if len(record['actions'])==2:
            service.update(run_id,desired='paused',status='pausing')
    service.commit_frame=pause_at_boundary
    with pytest.raises(LearningPaused):run_game(service,run_id,game_id,0,None)
    assert len(service.store.get('learning-games',game_id)['actions'])==2
    for player in (0,1):
        feed=service.frames(game_id,player)
        assert len(feed['frames'])==3
        for frame in feed['frames']:
            assert 'observations' not in frame and 'searchPosition' not in frame['observation']
            assert frame['observation']['players'][1-player]['hand']==[]
    # Recovery requires explicit resume and retains the original schedule cursor.
    service.recover();assert service.get(run_id)['status']=='paused'
    service.commit_frame=original
    state=service.get(run_id)
    service.control(run_id,'resume',state['revision'],'resume-1')
    complete=run_game(service,run_id,game_id,1,None)
    assert complete['status']=='finished' and len(complete['actions'])==6
    assert service.private(run_id)['collectionCursor']==1
    assert not service.stream._path(game_id).exists() # redundant stream only
    saved=service.replay(game_id,0)
    frames=service.frames(game_id,0)['frames']
    assert [f['observation'] for f in frames]==[f['observation'] for f in saved['frames']]
    assert service.frames(game_id,0,after=2)['frames'][0]['cursor']==3


def test_unacknowledged_step_recovered_once_and_repeated_failure_preserves_journal(service):
    run_id,game_id=game(service)
    service.pool.engine.failures=1
    result=run_game(service,run_id,game_id,0,None)
    assert result['workerFailures']==1 and len(result['actions'])==6
    assert [a['id'] for a in result['actions']]==[f'attack-{i}' for i in range(6)]


def test_second_worker_failure_never_becomes_outcome(service):
    run_id,game_id=game(service)
    service.pool.engine.failures=2
    with pytest.raises(EngineError,match='twice'):run_game(service,run_id,game_id,0,None)
    record=service.store.get('learning-games',game_id)
    assert record['workerFailures']==2 and not record['actions'] and not record['replayAvailable']
    assert record['workerError'] == {'message': 'Crashed after applying unacknowledged step',
                                      'attempt': 2, 'decisionIndex': 0}
    assert 'workerError' not in service.game_summary(record)
    assert not service.store.list('replays')


def test_truncation_never_fabricates_outcome(service):
    run_id,game_id=game(service,max_decisions=2)
    record=run_game(service,run_id,game_id,0,None)
    assert record['status']=='truncated' and record['outcome'] is None
    replay=service.store.get('replays',record['replayId'])
    assert replay['dataTier']=='experimental' and replay['trainingEligible'] is False
    assert replay['outcome'] is None


def test_request_size_failure_has_actionable_public_message_without_private_detail(service):
    run_id,_=game(service)
    service.update(run_id,error='Worker failed twice. Cause: Request exceeds 32 MiB. private sentinel')
    message=service.get(run_id)['error']
    assert 'Update the app before resuming' in message
    assert 'private sentinel' not in message


def test_controls_retries_and_recovery_are_manual(service):
    run_id,_=game(service)
    first=service.control(run_id,'pause',0,'pause-id')
    assert first['status']=='paused'
    assert service.control(run_id,'pause',0,'pause-id')['revision']==first['revision']
    with pytest.raises(ValueError,match='differently'):service.control(run_id,'stop',0,'pause-id')
    with pytest.raises(ValueError,match='changed'):service.control(run_id,'resume',0,'new-id')
    before=len(service.executor.calls)
    service.recover();assert len(service.executor.calls)==before
    service.control(run_id,'resume',first['revision'],'resume')
    assert len(service.executor.calls)==before+1
    with pytest.raises(ValueError,match='already enabled'):service.control(run_id,'resume',2,'again')


def test_foreground_priority_and_build_changes(service):
    run_id,game_id=game(service)
    with service.priority():
        with pytest.raises(LearningYield):service.check(run_id)
    service.check(run_id)
    service.pool.engine.health['engineVersion']='different'
    with pytest.raises(ValueError,match='Engine build changed'):run_game(service,run_id,game_id,0,None)
    assert not service.store.list('replays')


def test_uncommitted_stream_tail_not_published(service):
    run_id,game_id=game(service)
    record=service.store.get('learning-games',game_id)
    observations=[service.pool.engine.view(i) for i in (0,1)]
    service.commit_frame(record,observations,None,0)
    service.stream.append(game_id,{'observations':observations,'actor':0,'decisionIndex':1},0)
    assert len(service.frames(game_id)['frames'])==1


def test_resource_pause_is_durable_even_when_normal_metadata_writes_exceed_cap(service):
    from dataclasses import replace
    run_id,_=game(service)
    service.settings=replace(service.settings,max_disk_bytes=1)
    service.store.max_bytes=1
    service._run(run_id)
    saved=service.private(run_id)
    assert saved['status']=='paused' and saved['desired']=='paused'
    assert 'disk cap' in saved['error']
    assert saved['collectionCursor']==1 and len(saved['pendingGames'])==1


def test_comparison_seed_ledger_survives_collisions_and_seat_retries(service,monkeypatch):
    run_id,_=game(service)
    run=service.private(run_id)
    monkeypatch.setattr('ptcg_lab.learning.game_seed',lambda *args:1_000_000_005)
    seed=service.reserve_comparison_seed(run,'cycle-0-pair-0')
    assert service.reserve_comparison_seed(run,'cycle-0-pair-0')==seed
    assert service.reserve_comparison_seed(run,'cycle-1-pair-0')==seed+1
    entries=service.store.list('comparison-seeds')
    assert len(entries)==2 and all(e['excludedFromTraining'] for e in entries)


def test_late_failure_cannot_reverse_an_acknowledged_stop(service):
    run_id,_=game(service)
    service.control(run_id,'stop',0,'stop')
    service.update(run_id,status='paused',desired='paused',error='Late failure')
    assert service.private(run_id)['desired']=='stopped'
    assert service.get(run_id)['status']=='stopped'
