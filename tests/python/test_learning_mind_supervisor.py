from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind.supervisor import (MindSupervisor, continuous_operation_enablement,
                                               cpu_worker_count)


def approved_continuous_record():
    return {"ppoEnabled": True, "specialistCurriculumPassed": True,
            "continuousOperationEnabled": True, "humanEnableContinuousOperation": True}


def test_supervisor_always_restarts_paused_and_three_strikes_pause(tmp_path):
    events = []; clock = [100.]
    supervisor = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=10_000_000,
                                notifier=events.append, clock=lambda: clock[0])
    record = approved_continuous_record()
    supervisor.start(human_enabled=True, stage_record=record); assert supervisor.state.status == "RUNNING"
    restarted = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=10_000_000)
    assert restarted.state.status == "PAUSED" and restarted.state.pause_reason == "reboot-safe default"
    restarted.start(human_enabled=True, stage_record=record)
    restarted.record_failure("worker-restart"); restarted.record_failure("rejected-update")
    assert restarted.state.status == "RUNNING"
    restarted.record_failure("worker-restart")
    assert restarted.state.status == "PAUSED"


def test_immediate_failure_and_explicit_start(tmp_path):
    supervisor = MindSupervisor(tmp_path, reserve_bytes=0, data_cap_bytes=1000)
    with pytest.raises(PermissionError): supervisor.start(human_enabled=False)
    with pytest.raises(PermissionError, match="not enabled by the accepted stage record"):
        supervisor.start(human_enabled=True, stage_record={"continuousOperationEnabled": True})
    supervisor.start(human_enabled=True, stage_record=approved_continuous_record())
    supervisor.record_failure("private-view-leakage")
    assert supervisor.state.status == "PAUSED"


@pytest.mark.parametrize("field,value", [
    ("ppoEnabled", False), ("specialistCurriculumPassed", False),
    ("continuousOperationEnabled", False), ("humanEnableContinuousOperation", False),
    ("humanEnableContinuousOperation", 1),
])
def test_continuous_operation_requires_every_exact_gate(field, value):
    record = {**approved_continuous_record(), field: value}
    assert continuous_operation_enablement(record)["enabled"] is False


def test_cpu_worker_profile():
    assert cpu_worker_count(16) == 12
    assert cpu_worker_count(8) == 6
    assert cpu_worker_count(2) == 1
