from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import make_replay
from ptcg_lab.bundles import export_bundle, import_bundle
from ptcg_lab.config import Settings
from ptcg_lab.dataset import complete_games, examples, export_parquet, family_partition, training_eligible
from ptcg_lab.resources import ResourceGuard, ResourceLimit, check_storage
from ptcg_lab.selfplay import Agent, admitted_decks, population, search_choice
from ptcg_lab.storage import Store, digest

IDENTITIES = {"engineBuildHash": "build", "deckManifestHash": "decks", "featureVersion": "owned-zone-semantic-actions-v2"}


def reseal(bundle):
    path = bundle / "manifest.json"
    data = json.loads(path.read_text())
    data.pop("id")
    data["id"] = digest(data)
    path.write_text(json.dumps(data))


def test_profiles_and_invalid_limits(monkeypatch, tmp_path):
    monkeypatch.setenv("PTCG_LAB_ROOT", str(tmp_path))
    monkeypatch.setenv("PTCG_PROFILE", "windows")
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    settings = Settings.from_env()
    assert (settings.workers, settings.max_memory_bytes, settings.max_disk_bytes) == (2, 40 * 1024**3, 200 * 1024**3)
    assert settings.worker_tuning_limit == 8
    monkeypatch.setenv("PTCG_PROFILE", "mac")
    assert Settings.from_env().max_memory_bytes == 8 * 1024**3
    with pytest.raises(ValueError):
        Settings(tmp_path, tmp_path, workers=0)
    with pytest.raises(ValueError):
        Settings(tmp_path, tmp_path, max_disk_bytes=-1)


def test_memory_accounts_for_child_processes(monkeypatch):
    import psutil
    from types import SimpleNamespace
    from ptcg_lab.resources import process_memory
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=300))
    parent = SimpleNamespace(children=lambda recursive: [child], memory_info=lambda: SimpleNamespace(rss=200))
    monkeypatch.setattr(psutil, "Process", lambda pid: parent)
    assert process_memory() == 500


def test_resource_pause_never_becomes_outcome(tmp_path, monkeypatch):
    settings = Settings(tmp_path, tmp_path, max_memory_bytes=100, min_free_bytes=0)
    monkeypatch.setattr("ptcg_lab.resources.process_memory", lambda: 101)
    with pytest.raises(ResourceLimit, match="memory target"):
        with ResourceGuard(settings):
            pytest.fail("Over-budget work must not start")
    (tmp_path / "artifact").write_bytes(b"x" * 10)
    with pytest.raises(ResourceLimit, match="disk cap"):
        check_storage(tmp_path, 9, 0)
    monkeypatch.setattr("ptcg_lab.resources.volume_free", lambda path: 10)
    with pytest.raises(ResourceLimit, match="free-space"):
        check_storage(tmp_path, 100, 20)


def test_bundles_roundtrip_and_private_sources_excluded(tmp_path, observation):
    source = Store(tmp_path / "source")
    source.save_replay(make_replay(observation))
    source.put("guides", "private", {"text": "private source passage"})
    source.put("evaluation-protocols", "protocol", {"hash": "registered-before-play"})
    result = export_bundle(source, tmp_path / "bundle", IDENTITIES, min_free=0)
    destination = Store(tmp_path / "destination")
    imported = import_bundle(destination, tmp_path / "bundle", IDENTITIES, min_free=0)
    root = Path(imported["dataRoot"])
    assert result["id"] == imported["id"]
    assert Store(root).get("replays", "test-0") == source.get("replays", "test-0")
    assert not (root / "guides").exists()
    assert (root / "evaluation-protocols/protocol.json").exists()
    assert import_bundle(destination, tmp_path / "bundle", IDENTITIES, min_free=0)["alreadyImported"]


@pytest.mark.parametrize("bad_path", ["../escape.json", "models/../../escape.pt", "C:/models/x.pt", "models\\x.pt", "/models/x.pt"])
def test_bundle_rejects_traversal_before_copy(tmp_path, bad_path):
    export_bundle(Store(tmp_path / "source"), tmp_path / "bundle", IDENTITIES, min_free=0)
    manifest = tmp_path / "bundle/manifest.json"
    data = json.loads(manifest.read_text())
    data["files"] = [{"path": bad_path, "bytes": 0, "sha256": "invalid"}]
    manifest.write_text(json.dumps(data))
    reseal(tmp_path / "bundle")
    with pytest.raises(ValueError):
        import_bundle(Store(tmp_path / "dest"), tmp_path / "bundle", IDENTITIES, min_free=0)
    assert not (tmp_path / "dest/imports").exists()


def test_bundle_corruption_and_incompatible_versions(tmp_path, observation):
    source = Store(tmp_path / "source")
    source.save_replay(make_replay(observation))
    export_bundle(source, tmp_path / "bundle", IDENTITIES, min_free=0)
    with pytest.raises(ValueError, match="incompatible"):
        import_bundle(Store(tmp_path / "dest"), tmp_path / "bundle", {**IDENTITIES, "engineBuildHash": "other"}, min_free=0)
    replay_path = tmp_path / "bundle/replays/test-0.json"
    replay_path.write_text(replay_path.read_text().replace("test_fixture", "evil_fixture"))
    with pytest.raises(ValueError, match="checksum"):
        import_bundle(Store(tmp_path / "dest"), tmp_path / "bundle", IDENTITIES, min_free=0)


def test_bundle_rejects_conflicting_family_partitions(tmp_path, observation):
    source = Store(tmp_path / "source")
    first = make_replay(observation)
    second = copy.deepcopy(first)
    second["id"], second["benchmarkMatch"] = "benchmark-copy", "match"
    source.save_replay(first)
    source.save_replay(second)
    with pytest.raises(ValueError, match="Conflicting dataset partitions"):
        export_bundle(source, tmp_path / "bundle", IDENTITIES, min_free=0)


def test_interrupted_export_and_import_never_publish(tmp_path, monkeypatch):
    import ptcg_lab.bundles as bundles
    source = Store(tmp_path / "source")
    source.put("experiments", "experiment", {"status": "completed"})
    real_copy = bundles.shutil.copyfile

    def interrupted(source, target):
        real_copy(source, target)
        raise OSError("external drive disconnected")

    monkeypatch.setattr(bundles.shutil, "copyfile", interrupted)
    with pytest.raises(OSError):
        export_bundle(source, tmp_path / "bundle", IDENTITIES, min_free=0)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".*.partial"))
    monkeypatch.setattr(bundles.shutil, "copyfile", real_copy)
    export_bundle(source, tmp_path / "bundle", IDENTITIES, min_free=0)
    monkeypatch.setattr(bundles.shutil, "copyfile", interrupted)
    with pytest.raises(OSError):
        import_bundle(Store(tmp_path / "dest"), tmp_path / "bundle", IDENTITIES, min_free=0)
    assert not list((tmp_path / "dest/imports").iterdir())


def test_heldout_and_unverified_decks_cannot_train(observation):
    registry = [{"id": "main", "role": "main", "validation": {"trainingEligible": True}},
                {"id": "reserved", "role": "heldout", "validation": {"trainingEligible": True}},
                {"id": "unverified", "role": "training-variant", "validation": {"trainingEligible": False}}]
    assert [deck["id"] for deck in admitted_decks(registry)] == ["main"]
    assert [deck["id"] for deck in admitted_decks(registry, evaluation=True)] == ["main", "reserved"]
    for field, value in [("deckRoles", ["main", "heldout"]), ("trainingEligible", False), ("benchmarkMatch", "match")]:
        replay = make_replay(observation)
        replay[field] = value
        assert not training_eligible(replay)
        assert examples([replay]) == []


def test_search_targets_require_complete_legal_coverage_and_same_information(observation):
    class Engine:
        alternatives = [{"actionId": "action-0", "score": .4, "visits": 2}, {"actionId": "action-1", "score": .8, "visits": 3}]
        def request(self, method, params):
            return {"status": "complete", "alternatives": self.alternatives, "iterations": 5}
    engine = Engine()
    choice, target = search_choice(engine, observation, Agent("heuristic", 1), seed=2, budget_ms=10)
    assert choice == "action-1"
    assert sum(target["probabilities"].values()) == pytest.approx(1)
    replay = make_replay(observation)
    replay["searchTargets"] = [{**target, "decisionIndex": 0}]
    assert examples([replay])[0]["policyTargetSource"] == "search-distillation"
    replay["searchTargets"][0]["observationHash"] = "future-information"
    assert examples([replay])[0]["policyTargetSource"] == "behavior-cloning"
    engine.alternatives = [engine.alternatives[0], engine.alternatives[0]]
    assert search_choice(engine, observation, Agent("heuristic", 1), seed=2, budget_ms=10)[1] is None


def test_streamed_parquet_excludes_heldout_games_and_keeps_splits(tmp_path, observation, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    store = Store(tmp_path / "data")
    for index in range(3):
        replay = make_replay(observation, index)
        if index == 2:
            replay["deckRoles"] = ["heldout", "main"]
        store.save_replay(replay)
    monkeypatch.setattr(store, "list", lambda *args: pytest.fail("Export must stream records"))
    result = export_parquet(store, tmp_path / "rows.parquet")
    rows = pq.read_table(tmp_path / "rows.parquet").to_pylist()
    assert result["rows"] == len(rows) == 2
    assert {row["gameId"] for row in rows} == {"test-0", "test-1"}
    assert all(row["split"] == family_partition(make_replay(observation, int(row["gameId"][-1]))) for row in rows)
    assert (tmp_path / "rows.manifest.json").exists()


def test_linked_resume_after_real_bundle_transfer_preserves_parent_and_optimizer(tmp_path, observation):
    torch = pytest.importorskip("torch")
    from ptcg_lab.training import train
    store = Store(tmp_path / "source")
    for index in range(12):
        store.save_replay(make_replay(observation, index))
    first = train(store, epochs=1, max_positions=100)
    export_bundle(store, tmp_path / "bundle", IDENTITIES, min_free=0)
    target = import_bundle(Store(tmp_path / "other-device"), tmp_path / "bundle", IDENTITIES, min_free=0)
    imported = Store(Path(target["dataRoot"]))
    checkpoint = imported.path / "models" / Path(first["checkpoint"]).name
    child = train(imported, epochs=2, max_positions=100, resume=checkpoint, linked_resume=True)
    assert child["id"] != first["id"]
    assert child["parent"]["experimentId"] == first["id"]
    assert child["config"]["datasetHash"] == first["config"]["datasetHash"]
    original = torch.load(checkpoint, weights_only=True)
    continued = torch.load(child["checkpoint"], weights_only=True)
    assert continued["epoch"] == 2 and original["epoch"] == 1
    state_key = next(iter(original["optimizer"]["state"]))
    assert continued["optimizer"]["state"][state_key]["step"] > original["optimizer"]["state"][state_key]["step"]
    # A checkpoint at the requested final epoch can be reopened without an unset local variable.
    reopened = train(imported, epochs=2, max_positions=100, resume=Path(child["checkpoint"]))
    assert reopened["status"] == "completed"
    assert reopened["parent"] == child["parent"]
    assert len(population(imported)) == 4


def test_continuous_pause_resume_preserves_batch_and_frozen_opponents(tmp_path, monkeypatch):
    cycles = importlib.import_module("ptcg_lab.cycles")
    training = importlib.import_module("ptcg_lab.training")
    settings = Settings(tmp_path, tmp_path / "data", min_free_bytes=0)
    monkeypatch.setattr(cycles, "population", lambda store: ["random", "heuristic"])
    monkeypatch.setattr("ptcg_lab.resources.process_memory", lambda: 100)
    calls = []
    def selfplay(settings, **kwargs):
        calls.append(kwargs)
        kwargs["on_created"]("frozen-selfplay")
        return {"status": "paused" if len(calls) == 1 else "completed", "pauseReason": "test interruption"}
    monkeypatch.setattr(cycles, "selfplay", selfplay)
    def train(store, **kwargs):
        path = store.path / "models" / f"{kwargs['run_id']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-checkpoint")
        return {"modelHash": "fixture"}
    monkeypatch.setattr(training, "train", train)
    first = cycles.continuous(settings, batches=1, games=10, evaluate_every=2)
    assert first["status"] == "paused" and first["nextBatch"] == 0
    second = cycles.continuous(settings, batches=1, games=10, evaluate_every=2, resume=first["id"])
    assert second["status"] == "completed" and second["nextBatch"] == 1, second
    assert calls[1]["resume"] == "frozen-selfplay"
    assert calls[0]["opponents"] == calls[1]["opponents"]
    assert len(second["history"]) == 1


def test_replay_buffer_is_bounded_by_source_bytes(tmp_path, observation):
    store = Store(tmp_path)
    store.save_replay(make_replay(observation))
    size = store.location("replays", "test-0").stat().st_size
    assert complete_games(store, max_source_bytes=size - 1) == []
    assert len(complete_games(store, max_source_bytes=size)) == 1


def test_learned_search_sends_only_normalized_legal_root_priors(observation, monkeypatch):
    import ptcg_lab.training as training
    fallback = Agent("heuristic", 1)
    fallback.loaded, fallback.policy = object(), "model.pt"
    monkeypatch.setattr(training, "predict", lambda *args: ({}, [1., 2.]))
    received = {}
    class Engine:
        def request(self, method, params):
            received.update(params)
            return {"alternatives": []}
    search_choice(Engine(), observation, fallback, seed=2, budget_ms=10)
    assert {item["actionId"] for item in received["rootPriors"]} == {"action-0", "action-1"}
    assert sum(item["probability"] for item in received["rootPriors"]) == pytest.approx(1)
    assert received["rootPriors"][1]["probability"] > received["rootPriors"][0]["probability"]
    assert received["observation"] == observation


def test_disk_accounting_does_not_follow_directory_links(tmp_path):
    from ptcg_lab.resources import directory_bytes
    managed = tmp_path / "managed"
    outside = tmp_path / "outside"
    managed.mkdir()
    outside.mkdir()
    (managed / "small").write_bytes(b"abc")
    (outside / "large").write_bytes(b"x" * 1000)
    try:
        (managed / "external").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlink creation requires Windows developer permission")
    assert directory_bytes(managed) == 3


def test_legacy_replays_and_checkpoints_cannot_supply_trusted_learning(tmp_path, observation):
    from ptcg_lab.training import train, load_model
    torch = pytest.importorskip("torch")
    store = Store(tmp_path)
    old = make_replay(observation, 99)
    old.pop("trainingEligible")
    old.pop("deckRoles")
    store.save_replay(old)
    assert not training_eligible(old)
    assert complete_games(store) == []
    with pytest.raises(ValueError, match="No rules-validated"):
        admitted_decks([{"id": "legacy"}])
    for index in range(12):
        store.save_replay(make_replay(observation, index))
    result = train(store, epochs=1, max_positions=100)
    checkpoint = torch.load(result["checkpoint"], weights_only=True)
    checkpoint["config"].pop("eligibilityPolicy")
    legacy = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy)
    with pytest.raises(ValueError, match="Historical checkpoint"):
        load_model(legacy)
    with pytest.raises(ValueError, match="Historical checkpoint"):
        Agent(str(legacy), 1)


def test_family_quarantine_persists_after_benchmark_copy_is_archived(tmp_path, observation):
    store = Store(tmp_path)
    trusted = make_replay(observation, 5)
    store.save_replay(trusted)
    benchmark = copy.deepcopy(trusted)
    benchmark["id"], benchmark["benchmarkMatch"] = "benchmark-copy", "match"
    store.save_replay(benchmark)
    assert complete_games(store) == []
    store.location("replays", "benchmark-copy").unlink()
    store.location("replay-index", "benchmark-copy").unlink()
    assert complete_games(store) == []
    newer_copy = copy.deepcopy(trusted)
    newer_copy["id"] = "trusted-looking-copy"
    store.save_replay(newer_copy)
    assert complete_games(store) == []
    bundle = tmp_path.parent / f"bundle-{tmp_path.name}"
    export_bundle(store, bundle, IDENTITIES, min_free=0)
    imported = import_bundle(Store(tmp_path.parent / f"import-{tmp_path.name}"), bundle, IDENTITIES, min_free=0)
    assert complete_games(Store(Path(imported["dataRoot"]))) == []


def test_promotion_requires_all_registered_lists_and_five_archetypes():
    from ptcg_lab.evaluation import coverage_failures, REQUIRED_ARCHETYPES, REQUIRED_ROLES
    registry = [{"id": f"{archetype}-{role}", "archetype": archetype, "role": role}
                for archetype in REQUIRED_ARCHETYPES for role in REQUIRED_ROLES]
    assert coverage_failures(registry, registry) == []
    subset = [deck for deck in registry if deck["archetype"] == "crustle"]
    assert any("every registered" in reason for reason in coverage_failures(registry, subset))
    assert any("all five" in reason for reason in coverage_failures(subset, subset))
    assert coverage_failures(registry, registry[:-1])


def test_search_forwards_lab_knowledge_and_rejects_unfinished_scores(observation):
    received = {}
    class Engine:
        def request(self, method, params):
            received.update(params)
            return {"status": "unavailable", "alternatives": [
                {"actionId": "action-1", "score": .99, "visits": 20}]}
    fallback = Agent("heuristic", 1)
    action, target = search_choice(Engine(), observation, fallback, seed=1, budget_ms=20,
                                  known_opponent_deck_id="known-lab-list", prior_revealed_cards=["DRI-007"])
    assert received["knownOpponentDeckId"] == "known-lab-list"
    assert received["priorRevealedCards"] == ["DRI-007"]
    assert action == "action-0" and target is None


def test_warm_start_keeps_ancestor_families_out_of_heldout_evaluation(tmp_path, observation, monkeypatch):
    torch = pytest.importorskip("torch")
    from ptcg_lab.training import train, load_model
    import ptcg_lab.evaluation as evaluation
    source = Store(tmp_path / "ancestor")
    for index in range(12):
        source.save_replay(make_replay(observation, index))
    ancestor = train(source, epochs=1, max_positions=100)
    newer = Store(tmp_path / "new-corpus")
    for index in range(100, 150):
        newer.save_replay(make_replay(observation, index))
    child = train(newer, epochs=1, max_positions=100, warm_start=Path(ancestor["checkpoint"]))
    _, checkpoint = load_model(Path(child["checkpoint"]))
    assert set(range(12)) <= set(checkpoint["dataLineage"]["seenSeeds"])
    assert all(game["id"] != "test-0" for group in checkpoint["manifest"].values() for game in group)
    class Engine:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, method):
            if method == "decks":
                return [{"id": "crustle", "archetype": "crustle", "role": "main", "validation": {"trainingEligible": True}}]
            return {"firstPlayerControl": True}
    monkeypatch.setattr(evaluation, "EngineClient", Engine)
    with pytest.raises(ValueError, match="overlaps"):
        evaluation.evaluate(Settings(tmp_path, newer.path, min_free_bytes=0), candidate=child["checkpoint"], opponent="heuristic", seed_start=0)
    checkpoint.pop("dataLineage")
    unsupported = tmp_path / "unsupported.pt"
    torch.save(checkpoint, unsupported)
    with pytest.raises(ValueError, match="cumulative data lineage"):
        load_model(unsupported)


def test_evaluation_freezes_source_before_games_and_promotes_only_frozen_bytes(tmp_path, observation, monkeypatch):
    pytest.importorskip("torch")
    from ptcg_lab.training import train
    from ptcg_lab.storage import file_digest
    import ptcg_lab.evaluation as evaluation
    store = Store(tmp_path / "data")
    for index in range(12):
        store.save_replay(make_replay(observation, index))
    trained = train(store, epochs=1, max_positions=100)
    source = Path(trained["checkpoint"])
    expected_hash = file_digest(source)
    calls = []
    class Engine:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, method):
            if method == "decks":
                return [{"id": "crustle", "archetype": "crustle", "role": "heldout", "validation": {"trainingEligible": True}}]
            return {"firstPlayerControl": True}
    def play(engine, *, decks, seed, policies, max_decisions, **kwargs):
        frozen = next(policy for policy in policies if policy != "heuristic")
        assert Path(frozen) != source
        assert file_digest(Path(frozen)) == expected_hash
        # The actual policy loader still sees valid original weights after the
        # public source checkpoint has been replaced by another process.
        assert Agent(frozen, seed).choose(observation) in {"action-0", "action-1"}
        calls.append(frozen)
        source.write_bytes(b"concurrent training replaced the source checkpoint")
        replay = make_replay(observation, len(calls))
        replay["seed"], replay["decks"], replay["startingPlayer"] = seed, decks, 0
        replay["outcome"]["winner"] = policies.index(frozen)
        return replay
    monkeypatch.setattr(evaluation, "EngineClient", Engine)
    monkeypatch.setattr(evaluation, "play_game", play)
    # Isolate artifact freezing/publication from the separately tested sample
    # size and complete-registry gates; two fake games do not prove strength.
    monkeypatch.setattr(evaluation, "coverage_failures", lambda *args: [])
    monkeypatch.setattr(evaluation, "promotion_decision", lambda *args: (True, []))
    report = evaluation.evaluate(Settings(tmp_path, store.path, min_free_bytes=0), candidate=str(source),
                                 opponent="heuristic", seeds=1, seed_start=2000, promote=True)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert report["promoted"] is True
    assert report["candidate"] == str(source)
    assert report["models"]["candidate"] == expected_hash
    assert file_digest(store.path / "models/champion.pt") == expected_hash
    assert file_digest(source) != expected_hash
    assert (store.path / report["checkpointSnapshots"]["candidate"]["path"]).exists()


def test_evaluation_copy_rejects_changed_bytes_before_publication(tmp_path, monkeypatch):
    from ptcg_lab.storage import file_digest
    import ptcg_lab.evaluation as evaluation
    store = Store(tmp_path / "data")
    source = tmp_path / "candidate.pt"
    source.write_bytes(b"original checkpoint")
    target = store.path / "evaluation-models/snapshot.pt"
    real_copy = evaluation.shutil.copyfile
    def corrupted_copy(source, destination):
        real_copy(source, destination)
        Path(destination).write_bytes(b"different checkpoint")
    monkeypatch.setattr(evaluation.shutil, "copyfile", corrupted_copy)
    with pytest.raises(ValueError, match="changed during evaluation copy"):
        evaluation._verified_checkpoint_copy(store, source, target, file_digest(source))
    assert not target.exists()
    assert not list(target.parent.glob("*.tmp"))
