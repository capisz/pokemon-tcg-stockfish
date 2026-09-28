from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind.supervisor import (MindSupervisor, continuous_operation_enablement,
    cpu_worker_count, VerifiedContinuousOperationRecord)


def test_supervisor_always_restarts_paused_and_three_strikes_pause(tmp_path):
    events = []; clock = [100.]
    supervisor = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=10_000_000,
                                notifier=events.append, clock=lambda: clock[0])
    supervisor.state.status = "RUNNING"
    supervisor.persist()
    restarted = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=10_000_000)
    assert restarted.state.status == "PAUSED" and restarted.state.pause_reason == "reboot-safe default"
    restarted.state.status = "RUNNING"  # Exercise failure handling independently of the closed start gate.
    restarted.record_failure("worker-restart"); restarted.record_failure("rejected-update")
    assert restarted.state.status == "RUNNING"
    restarted.record_failure("worker-restart")
    assert restarted.state.status == "PAUSED"


def test_immediate_failure_and_explicit_start(tmp_path):
    supervisor = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=1000)
    with pytest.raises(PermissionError): supervisor.start(human_enabled=False)
    with pytest.raises(PermissionError, match="not enabled by the accepted stage record"):
        supervisor.start(human_enabled=True, stage_record={"continuousOperationEnabled": True})
    with pytest.raises(TypeError, match="must come from the evidence verifier"):
        VerifiedContinuousOperationRecord({"continuousOperationEnabled": True}, _verification_token=object())
    supervisor.state.status = "RUNNING"
    supervisor.record_failure("private-view-leakage")
    assert supervisor.state.status == "PAUSED"


@pytest.mark.parametrize("field,value", [
    ("ppoEnabled", False), ("specialistCurriculumPassed", False),
    ("continuousOperationEnabled", False), ("humanEnableContinuousOperation", False),
    ("humanEnableContinuousOperation", 1),
])
def test_continuous_operation_requires_every_exact_gate(field, value):
    record = {"ppoEnabled": True, "specialistCurriculumPassed": True,
              "continuousOperationEnabled": True, "humanEnableContinuousOperation": True}
    record[field] = value
    assert continuous_operation_enablement(record)["enabled"] is False


def test_phase_cursor_survives_pause_and_restart_and_transitions_are_ordered(tmp_path):
    supervisor = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=10_000_000)
    supervisor.state.status = "RUNNING"  # Start remains impossible without verified evidence.
    supervisor.record_progress({"positionIndex": 7, "optimizerBatch": 2})
    supervisor.pause("planned checkpoint", notify=False)

    restarted = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=10_000_000)
    assert restarted.state.status == "PAUSED"
    assert restarted.state.phase == "collection"
    assert restarted.state.cursor == {"positionIndex": 7, "optimizerBatch": 2}
    with pytest.raises(PermissionError, match="only while the supervisor is running"):
        restarted.record_progress({"positionIndex": 8})
    restarted.state.status = "RUNNING"
    with pytest.raises(ValueError, match="must advance"):
        restarted.advance_phase("evaluation")
    restarted.advance_phase("training", next_cursor={"epoch": 0, "batch": 0})
    assert restarted.state.phase == "training"
    assert restarted.state.cursor == {"epoch": 0, "batch": 0}
    with pytest.raises(ValueError, match="finite JSON values"):
        restarted.record_progress({"unsafe": float("nan")})
    assert restarted.state.cursor == {"epoch": 0, "batch": 0}


def test_cpu_worker_profile():
    assert cpu_worker_count(16) == 12
    assert cpu_worker_count(8) == 6
    assert cpu_worker_count(2) == 1
