from __future__ import annotations

import copy
from pathlib import Path

import pytest

from test_experimental_learning import stores, pin_teaching, experimental_game, training_seed
from ptcg_lab import experimental as exp
from ptcg_lab.checkpoint_files import freeze_checkpoint
from ptcg_lab.recovery import fork_paused_comparison
from ptcg_lab.storage import Store, digest, file_digest


@pytest.fixture
def recovery_source(tmp_path, observation):
    pytest.importorskip("torch")
    main, store = stores(tmp_path)
    _, snapshot = pin_teaching(tmp_path, main, store, observation)
    guide = exp.bootstrap_policy(store, snapshot_id=snapshot["id"], run_id="guide")
    game = experimental_game(observation, training_seed(observation))
    store.save_replay(game)
    dataset = exp.prepare_dataset(store, [game["id"]], teaching_snapshot_id=snapshot["id"])
    candidate = exp.train_experimental(store, dataset_id=dataset["id"], run_id="candidate", checkpoint=Path(guide["checkpoint"]))
    def frozen(report):
        path = freeze_checkpoint(store, Path(report["checkpoint"]))
        return {"path": str(path), "hash": file_digest(path), "name": report["id"]}
    decks = [{"id": f"{archetype}-{role}", "archetype": archetype, "role": role}
             for archetype in ("crustle", "dragapult", "raging-bolt", "grimmsnarl", "mega-lucario")
             for role in ("main", "training-variant")]
    parent = {"schemaVersion": 1, "id": "parent", "status": "paused", "desired": "paused", "phase": "comparing", "revision": 7,
              "engine": {"engineVersion": "old-engine", "engineBuildHash": "1" * 64, "protocolVersion": 1},
              "implementationHash": "2" * 64, "decks": decks, "deckHash": digest(decks), "datasetId": dataset["id"],
              "teachingSnapshotId": snapshot["id"], "trainingId": "candidate", "candidate": frozen(candidate),
              "guidePolicy": frozen(guide), "incumbent": {"path": "heuristic", "hash": "heuristic", "name": "heuristic"},
              "collectedReplayIds": [game["id"]], "collectionCursor": 50, "batchReplayIds": [game["id"]],
              "pendingGames": ["pending"], "comparisonRecords": [{"gameId": "old-evidence", "score": 0}], "comparisonCursor": 52,
              "cycle": 0, "controls": {}, "history": [frozen(guide), {"path": "heuristic", "hash": "heuristic", "name": "heuristic"}], "configuration": {"seed": 42},
              "metrics": {"completedGames": 50, "comparison": {"completedGames": 50}}, "error": "Original transport failure"}
    pending = {"id": "pending", "runId": "parent", "purpose": "comparison", "status": "running", "workerFailures": 2,
               "actions": [{"id": "accepted"}], "observationHash": "old-position", "frameCursor": 1}
    store.put("learning-runs", parent["id"], parent)
    store.put("learning-games", "pending", pending)
    kwargs = {"parent_id": "parent", "expected_revision": 7, "expected_parent_hash": digest(parent),
              "current_engine": {"engineVersion": "new-transport", "engineBuildHash": "3" * 64, "protocolVersion": 1},
              "implementation_hash": "4" * 64, "decks": decks}
    return main, store, parent, pending, kwargs


def test_fork_preserves_source_bytes_and_starts_fresh_paused_comparison(recovery_source):
    _, store, parent, pending, kwargs = recovery_source
    files = [store.location("learning-games", "pending"), Path(parent["candidate"]["path"]), Path(parent["guidePolicy"]["path"])]
    before = {path: path.read_bytes() for path in files}
    result = fork_paused_comparison(store, **kwargs)
    child = store.get("learning-runs", result["successorId"])
    archived = store.get("learning-runs", "parent")
    receipt = store.get("learning-recoveries", result["recoveryId"])
    assert child["status"] == child["desired"] == "paused" and child["revision"] == 0
    assert child["engine"] == kwargs["current_engine"] and child["implementationHash"] == kwargs["implementation_hash"]
    assert child["comparisonRecords"] == child["pendingGames"] == [] and child["comparisonCursor"] == 0
    assert child["collectionCursor"] == 50 and child["collectedReplayIds"] == parent["collectedReplayIds"]
    assert child["datasetId"] == parent["datasetId"] and child["candidate"] == parent["candidate"]
    assert child["history"] == parent["history"]
    assert receipt["parentSnapshot"] == parent and receipt["pendingGameSnapshots"]["pending"] == pending
    assert archived["status"] == archived["desired"] == "stopped" and archived["successorId"] == child["id"]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert fork_paused_comparison(store, **kwargs) == result
    # A fresh run ID is part of both game identities and comparison seed keys.
    from ptcg_lab.learning_protocol import game_seed
    assert game_seed(42, "comparison", f"{child['id']}:0:0") != game_seed(42, "comparison", "parent:0:0")


def test_fork_is_restart_safe_after_successor_written_before_parent_archive(recovery_source, monkeypatch):
    _, store, parent, _, kwargs = recovery_source
    original_put = store.put
    def interrupted(category, identifier, value):
        if category == "learning-runs" and identifier == "parent" and value["status"] == "stopped":
            raise OSError("Synthetic interruption before archival")
        return original_put(category, identifier, value)
    monkeypatch.setattr(store, "put", interrupted)
    with pytest.raises(OSError, match="interruption"):
        fork_paused_comparison(store, **kwargs)
    assert store.get("learning-runs", "parent") == parent
    children = [r for r in store.list("learning-runs") if r["id"] != "parent"]
    assert len(children) == 1 and children[0]["status"] == "paused"
    monkeypatch.setattr(store, "put", original_put)
    result = fork_paused_comparison(store, **kwargs)
    assert result["successorId"] == children[0]["id"] and len(store.list("learning-runs")) == 2


def test_fork_rejects_parent_changes_and_other_phases_without_creating_receipt(recovery_source):
    _, store, parent, _, kwargs = recovery_source
    with pytest.raises(ValueError, match="revision"):
        fork_paused_comparison(store, **{**kwargs, "expected_revision": 6})
    modified = {**parent, "phase": "training"}
    store.put("learning-runs", "parent", modified)
    with pytest.raises(ValueError, match="paused experimental comparison"):
        fork_paused_comparison(store, **{**kwargs, "expected_parent_hash": digest(modified)})
    assert not store.list("learning-recoveries")


def test_fork_rejects_changed_pinned_data_and_decks(recovery_source):
    _, store, parent, _, kwargs = recovery_source
    with pytest.raises(ValueError, match="deck registry"):
        fork_paused_comparison(store, **{**kwargs, "decks": kwargs["decks"][::-1]})
    rows = store.get("dataset-rows", parent["datasetId"])
    rows[0]["expectedResult"] = .123
    store.put("dataset-rows", parent["datasetId"], rows)
    with pytest.raises(ValueError, match="dataset rows"):
        fork_paused_comparison(store, **kwargs)
    assert not store.list("learning-recoveries")


def test_fork_rejects_mutated_frozen_model_and_imported_roots(recovery_source):
    _, store, parent, _, kwargs = recovery_source
    with pytest.raises(ValueError, match="local experimental"):
        fork_paused_comparison(Store(store.path / "imports/x/experimental"), **kwargs)
    Path(parent["candidate"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="model path or bytes"):
        fork_paused_comparison(store, **kwargs)
    assert store.get("learning-runs", "parent")["status"] == "paused"


def test_recovery_receipt_survives_bundle_without_starting_imported_child(recovery_source, tmp_path):
    from ptcg_lab.bundles import export_bundle, import_bundle
    from ptcg_lab.features import FEATURE_VERSION
    main, store, _, _, kwargs = recovery_source
    result = fork_paused_comparison(store, **kwargs)
    identities = {"engineBuildHash": "synthetic", "deckManifestHash": "synthetic", "featureVersion": FEATURE_VERSION}
    bundle = export_bundle(main, tmp_path / "bundle", identities, min_free=0)
    imported = import_bundle(Store(tmp_path / "destination"), Path(bundle["path"]), identities, min_free=0)
    restored = Store(Path(imported["dataRoot"]) / "experimental")
    assert restored.get("learning-recoveries", result["recoveryId"]) == store.get("learning-recoveries", result["recoveryId"])
    assert restored.get("learning-runs", result["successorId"])["status"] == "paused"


def test_fork_rejects_unrelated_active_run_and_changed_historical_opponent(recovery_source):
    _, store, parent, _, kwargs = recovery_source
    store.put("learning-runs", "unrelated", {"id": "unrelated", "status": "paused", "desired": "paused"})
    with pytest.raises(ValueError, match="Another learning run"):
        fork_paused_comparison(store, **kwargs)
    store.put("learning-runs", "unrelated", {"id": "unrelated", "status": "stopped", "desired": "stopped"})
    changed = copy.deepcopy(parent)
    changed["history"][0]["hash"] = "f" * 64
    store.put("learning-runs", "parent", changed)
    with pytest.raises(ValueError, match="model path or bytes"):
        fork_paused_comparison(store, **{**kwargs, "expected_parent_hash": digest(changed)})
    assert not store.list("learning-recoveries")
