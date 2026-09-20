from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ptcg_lab.resources import ResourceLimit
from ptcg_lab.storage import Store


def running(store):
    record = {"id": "run", "status": "running", "desired": "running", "revision": 4,
              "phase": "training", "metrics": {"completedGames": 7},
              "pendingGames": ["unchanged-game"], "datasetId": "pinned", "modelHash": "frozen"}
    store.put("learning-runs", "run", record)
    return record


def pause(record):
    return {**record, "status": "paused", "desired": "paused", "pauseReason": "Storage limit reached; archive completed bundles.",
            "error": "Disk reserve reached", "updatedAt": "2026-09-20T18:00:00+00:00"}


def test_emergency_pause_bypasses_only_configured_cap_and_preserves_progress(tmp_path):
    store = Store(tmp_path)
    original = running(store)
    path = store.location("learning-runs", "run")
    original_size = path.stat().st_size
    store.max_bytes = 1
    proposed = pause(original)
    proposed.update(datasetId="not-allowed", metrics={"completedGames": 999}, newArtifact="not-allowed")
    with pytest.raises(RuntimeError, match="disk cap"):
        store.put("learning-runs", "run", proposed)
    result = store.emergency_control_pause("run", proposed)
    assert result == store.get("learning-runs", "run")
    assert result["status"] == result["desired"] == "paused" and result["revision"] == 5
    assert result["datasetId"] == "pinned" and result["metrics"] == original["metrics"]
    assert result["modelHash"] == original["modelHash"] and "newArtifact" not in result
    assert path.stat().st_size <= original_size + 2048
    assert list(path.parent.glob("*.tmp")) == []
    with pytest.raises(RuntimeError, match="disk cap"):
        store.put("models", "ordinary-write", {"not": "permitted"})


def test_emergency_pause_uses_physical_space_below_configured_reserve(tmp_path, monkeypatch):
    store = Store(tmp_path)
    record = running(store)
    store.min_free_bytes = 10_000_000
    monkeypatch.setattr("ptcg_lab.resources.volume_free", lambda _: 100_000)
    with pytest.raises(ResourceLimit, match="reserve"):
        store.put("learning-runs", "run", pause(record))
    assert store.emergency_control_pause("run", pause(record))["status"] == "paused"
    before = store.location("learning-runs", "run").read_bytes()
    monkeypatch.setattr("ptcg_lab.resources.volume_free", lambda _: 1)
    with pytest.raises(ResourceLimit, match="physical space"):
        store.emergency_control_pause("run", pause(record))
    assert store.location("learning-runs", "run").read_bytes() == before


def test_emergency_control_refuses_new_runs_running_requests_and_acknowledged_stops(tmp_path):
    store = Store(tmp_path)
    record = running(store)
    before = store.location("learning-runs", "run").read_bytes()
    with pytest.raises(ValueError, match="create"):
        store.emergency_control_pause("new", {**pause(record), "id": "new"})
    with pytest.raises(ValueError, match="only pause"):
        store.emergency_control_pause("run", record)
    assert store.location("learning-runs", "run").read_bytes() == before
    store.put("learning-runs", "run", {**record, "desired": "stopped"})
    with pytest.raises(ValueError, match="acknowledged stop"):
        store.emergency_control_pause("run", pause(record))


def test_failed_atomic_emergency_replace_keeps_old_acknowledged_record(tmp_path, monkeypatch):
    store = Store(tmp_path)
    record = running(store)
    path = store.location("learning-runs", "run")
    before = path.read_bytes()
    def disconnected(*args):
        raise OSError("Synthetic disconnected drive")
    monkeypatch.setattr(Path, "replace", disconnected)
    with pytest.raises(OSError, match="disconnected"):
        store.emergency_control_pause("run", pause(record))
    assert path.read_bytes() == before
    assert list(path.parent.glob("*.tmp")) == []


def test_emergency_details_are_bounded_and_do_not_erase_unrelated_fields(tmp_path):
    store = Store(tmp_path)
    record = running(store)
    original = copy.deepcopy(record)
    result = store.emergency_control_pause("run", {**pause(record), "error": "\n💾" * 10000, "pauseReason": "x" * 10000})
    assert len(result["error"].encode()) <= 1000 and len(result["pauseReason"].encode()) <= 500
    assert record == original
    assert result["pendingGames"] == ["unchanged-game"]
