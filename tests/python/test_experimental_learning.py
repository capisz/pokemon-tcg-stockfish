from __future__ import annotations

import copy
from pathlib import Path

import pytest

from conftest import make_replay
from test_guide_policy import fixture_record, review
from ptcg_lab import experimental as exp
from ptcg_lab.dataset import training_eligible
from ptcg_lab.storage import Store, digest
from ptcg_lab.training import load_model, predict, export_portable_value


def stores(tmp_path):
    main = Store(tmp_path / "data")
    return main, Store(main.path / "experimental")


def experimental_game(observation, index=0, frames=40):
    game = make_replay(observation, index)
    game.update(dataTier="experimental", experimentalLearning=True, trainingEligible=False,
                deckHashes=["frozen-crustle", "frozen-dragapult"], learningRun="test-run")
    game["outcome"]["reason"] = "rules-terminal"
    base = game["frames"][0]
    game["frames"] = []
    for decision in range(frames):
        frame = copy.deepcopy(base)
        frame["decisionIndex"] = decision
        frame["observations"][0]["turn"] = decision * 4 + 3
        game["frames"].append(frame)
    return game


def training_seed(observation):
    return next(seed for seed in range(100) if exp.experimental_partition(experimental_game(observation, seed)) == "train")


def pin_teaching(tmp_path, main, experimental, observation):
    record = review(main, fixture_record(tmp_path, main, observation))
    return record, exp.snapshot_teaching(main, experimental, "teaching-v1")


def test_bootstrap_uses_pinned_reviews_and_experimental_opt_in(tmp_path, observation):
    pytest.importorskip("torch")
    main, experimental = stores(tmp_path)
    record, snapshot = pin_teaching(tmp_path, main, experimental, observation)
    review(main, record, ["action-1"])
    result = exp.bootstrap_policy(experimental, snapshot_id=snapshot["id"], run_id="bootstrap")
    path = Path(result["checkpoint"])
    with pytest.raises(ValueError, match="quarantined"):
        load_model(path)
    loaded = load_model(path, allow_experimental=True)
    assert loaded[1]["dataTier"] == "experimental" and loaded[1]["dataLineage"]["experimentalAncestry"]
    assert loaded[1]["teachingManifest"][0]["reviewHash"] == record["reviewHash"]
    with pytest.raises(ValueError, match="quarantined"):
        predict(path, observation, loaded)
    assert predict(path, observation, loaded, allow_experimental=True)[0]["score"] is None
    assert export_portable_value(path, loaded, allow_experimental=True) is None
    resumed = exp.bootstrap_policy(experimental, snapshot_id=snapshot["id"], run_id="bootstrap")
    assert resumed["teachingDiagnostic"]["baselineHash"] == result["teachingDiagnostic"]["baselineHash"]
    assert list((main.path / "models").glob("*.pt")) == []


def test_dataset_quarantines_outcomes_and_caps_teaching_anchors(tmp_path, observation):
    main, experimental = stores(tmp_path)
    _, snapshot = pin_teaching(tmp_path, main, experimental, observation)
    game = experimental_game(observation, training_seed(observation), frames=60)
    experimental.save_replay(game)
    assert not training_eligible(game)
    manifest = exp.prepare_dataset(experimental, [game["id"]], teaching_snapshot_id=snapshot["id"], max_rows=50)
    rows = experimental.get("dataset-rows", manifest["id"])
    assert len(rows) <= 50 and 0 < manifest["demonstrationRows"] <= len(rows) * .1
    assert manifest["outcomeRows"] + manifest["demonstrationRows"] == len(rows)
    assert all(row["policyTargetSource"] == "none" for row in rows if "expectedResult" in row)
    assert all("expectedResult" not in row for row in rows if row["policyTargetSource"] == "reviewed-demonstration")
    repeat = exp.prepare_dataset(experimental, [game["id"]], teaching_snapshot_id=snapshot["id"], max_rows=50)
    assert repeat == manifest


def test_outcome_only_training_does_not_clone_played_moves_and_labels_values_experimental(tmp_path, observation):
    torch = pytest.importorskip("torch")
    from ptcg_lab.model import PolicyResourceModel
    main, experimental = stores(tmp_path)
    game = experimental_game(observation, training_seed(observation))
    experimental.save_replay(game)
    dataset = exp.prepare_dataset(experimental, [game["id"]])
    result = exp.train_experimental(experimental, dataset_id=dataset["id"], run_id="candidate")
    path = Path(result["checkpoint"])
    loaded = load_model(path, allow_experimental=True)
    torch.manual_seed(42)
    initial = PolicyResourceModel().state_dict()
    assert all(torch.equal(value, initial[name]) for name, value in loaded[1]["model"].items() if name.startswith("action_encoder."))
    evaluation, scores = predict(path, observation, loaded, allow_experimental=True)
    assert evaluation["label"] == "Experimental learned" and evaluation["score"] is not None
    assert not evaluation["calibrated"] and evaluation["expectedResult"] is None and evaluation["winProbability"] is None
    assert len(scores) == len(observation["legalActions"])
    with pytest.raises(ValueError, match="quarantined"):
        export_portable_value(path, loaded)
    assert export_portable_value(path, loaded, allow_experimental=True)
    from ptcg_lab.model_registry import list_models, resolve_model
    assert list_models(main)[0]["id"] == "experimental-candidate"
    assert resolve_model(main, "experimental-candidate") == path
    with pytest.raises(ValueError):
        resolve_model(main, "candidate")
    assert not (main.path / "models/champion.pt").exists()


def test_reanalysis_target_is_pinned_and_preferred_over_collection_policy(tmp_path, observation):
    _, store = stores(tmp_path)
    game = experimental_game(observation, training_seed(observation), frames=1)
    frame = game["frames"][0]
    target = {"id": "investigation", "replayId": game["id"], "decisionIndex": 0, "actor": 0,
              "observationHash": digest(frame["observations"][0]), "probabilities": {"action-0": .1, "action-1": .9},
              "dataTier": "experimental", "source": "reanalysis", "status": "supported", "independentSeeds": 2}
    game["searchTargets"] = [{**target, "probabilities": {"action-0": .8, "action-1": .2}}]
    store.save_replay(game)
    store.put("search-targets", target["id"], target)
    manifest = exp.prepare_dataset(store, [game["id"]])
    row = store.get("dataset-rows", manifest["id"])[0]
    assert row["policyDistribution"] == [.1, .9]
    assert manifest["searchTargets"] == [{"id": target["id"], "hash": digest(target)}]
    target["probabilities"] = {"action-0": 1., "action-1": 0.}
    store.put("search-targets", target["id"], target)
    assert store.get("dataset-rows", manifest["id"])[0]["policyDistribution"] == [.1, .9]


def test_incomplete_reserved_and_nontraining_partitions_never_supply_targets(tmp_path, observation):
    _, store = stores(tmp_path)
    training = experimental_game(observation, training_seed(observation))
    heldout_seed = next(seed for seed in range(100) if exp.experimental_partition(experimental_game(observation, seed)) == "test")
    heldout = experimental_game(observation, heldout_seed)
    incomplete = experimental_game(observation, 1001)
    incomplete.update(status="truncated", outcome=None)
    for game in (training, heldout, incomplete):
        store.save_replay(game)
    result = exp.prepare_dataset(store, [game["id"] for game in (training, heldout, incomplete)])
    assert [item["id"] for item in result["replays"]] == [training["id"]]
    assert result["skippedIncomplete"] == [incomplete["id"]]
    assert result["excludedPartitions"] == [{"id": heldout["id"], "partition": "test"}]
    for updates in ({"deckRoles": ["main", "heldout"]}, {"experimentalComparison": "comparison"},
                    {"experimentalLearning": False}, {"trainingEligible": True}):
        changed = {**training, **updates, "id": "invalid"}
        store.put("replays", "invalid", changed)
        with pytest.raises(ValueError, match="explicitly admitted"):
            exp.prepare_dataset(store, ["invalid"])


def test_optimizer_resume_is_pinned_despite_new_data_and_preserves_ancestry(tmp_path, observation):
    torch = pytest.importorskip("torch")
    _, store = stores(tmp_path)
    game = experimental_game(observation, training_seed(observation), frames=80)
    store.save_replay(game)
    manifest = exp.prepare_dataset(store, [game["id"]])
    class Pause:
        calls = 0
        def check(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Synthetic pause")
    with pytest.raises(RuntimeError, match="Synthetic pause"):
        exp.train_experimental(store, dataset_id=manifest["id"], run_id="resumed", guard=Pause())
    path = store.path / "models/resumed.pt"
    assert torch.load(path, weights_only=True)["nextBatch"] == 32
    store.save_replay(experimental_game(observation, 1002))
    resumed = exp.train_experimental(store, dataset_id=manifest["id"], run_id="resumed", checkpoint=path, resume=True)
    uninterrupted = exp.train_experimental(store, dataset_id=manifest["id"], run_id="fresh")
    first = torch.load(resumed["checkpoint"], weights_only=True)
    second = torch.load(uninterrupted["checkpoint"], weights_only=True)
    assert all(torch.equal(value, second["model"][key]) for key, value in first["model"].items())
    child = exp.train_experimental(store, dataset_id=manifest["id"], run_id="child", checkpoint=path)
    _, checkpoint = load_model(Path(child["checkpoint"]), allow_experimental=True)
    assert checkpoint["dataLineage"]["experimentalAncestry"] and child["parent"]["experimentId"] == "resumed"
    with pytest.raises(ValueError, match="quarantined"):
        load_model(Path(child["checkpoint"]))


def test_nested_bundle_preserves_tier_and_pinned_artifacts(tmp_path, observation):
    pytest.importorskip("torch")
    from ptcg_lab.bundles import export_bundle, import_bundle
    from ptcg_lab.features import FEATURE_VERSION
    from ptcg_lab.model_registry import resolve_model
    main, experimental = stores(tmp_path)
    game = experimental_game(observation, training_seed(observation))
    experimental.save_replay(game)
    manifest = exp.prepare_dataset(experimental, [game["id"]])
    result = exp.train_experimental(experimental, dataset_id=manifest["id"], run_id="portable")
    identities = {"engineBuildHash": "synthetic", "deckManifestHash": "synthetic", "featureVersion": FEATURE_VERSION}
    # Journals and audit provenance travel unchanged; importing itself starts no jobs.
    experimental.put("learning-runs", "archived-run", {"id": "archived-run", "status": "paused"})
    experimental.put("teaching-audits", "attestation", {"id": "attestation", "evidence": "synthetic"})
    from ptcg_lab.presentation import FrameStream
    stream = FrameStream(experimental)
    stream.append("archived-game", game["frames"][0], committed=-1)
    frozen = experimental.path / "private-models/frozen.pt"
    frozen.parent.mkdir()
    frozen.write_bytes(Path(result["checkpoint"]).read_bytes())
    exported = export_bundle(main, tmp_path / "bundle", identities, min_free=0)
    imported = import_bundle(Store(tmp_path / "destination"), Path(exported["path"]), identities, min_free=0)
    target = Store(Path(imported["dataRoot"]))
    path = resolve_model(target, "experimental-portable")
    with pytest.raises(ValueError, match="quarantined"):
        load_model(path)
    assert load_model(path, allow_experimental=True)[1]["manifest"] == manifest
    assert not (target.path / "models" / Path(result["checkpoint"]).name).exists()
    nested = Store(target.path / "experimental")
    assert nested.get("learning-runs", "archived-run")["status"] == "paused"
    assert nested.get("teaching-audits", "attestation")["evidence"] == "synthetic"
    assert (nested.path / "private-streams/archived-game.jsonl").read_bytes() == (experimental.path / "private-streams/archived-game.jsonl").read_bytes()
    assert (nested.path / "private-models/frozen.pt").read_bytes() == frozen.read_bytes()


def test_tactical_diagnostics_support_heuristic_and_identical_model(tmp_path, observation):
    pytest.importorskip("torch")
    main, store = stores(tmp_path)
    _, snapshot = pin_teaching(tmp_path, main, store, observation)
    report = exp.bootstrap_policy(store, snapshot_id=snapshot["id"], run_id="guide")
    path = Path(report["checkpoint"])
    identical = exp.tactical_regressions(store, snapshot_id=snapshot["id"], candidate=path, incumbent=path)
    heuristic = exp.tactical_regressions(store, snapshot_id=snapshot["id"], candidate=path, incumbent="heuristic")
    assert identical["passes"] and not identical["regressions"]
    assert identical["candidateAccuracy"] == heuristic["candidateAccuracy"]
    assert 0 <= heuristic["candidateAccuracy"] <= 1
    assert "not held-out" in heuristic["description"]


def test_empty_batch_requests_collection_and_changed_rows_fail_before_training(tmp_path, observation):
    pytest.importorskip("torch")
    _, store = stores(tmp_path)
    incomplete = experimental_game(observation, 1001)
    incomplete.update(status="truncated", outcome=None)
    store.save_replay(incomplete)
    with pytest.raises(exp.InsufficientExperimentalData):
        exp.prepare_dataset(store, [incomplete["id"]])
    game = experimental_game(observation, training_seed(observation))
    store.save_replay(game)
    manifest = exp.prepare_dataset(store, [game["id"]])
    rows = store.get("dataset-rows", manifest["id"])
    rows[0]["expectedResult"] = .5
    store.put("dataset-rows", manifest["id"], rows)
    with pytest.raises(ValueError, match="incompatible or changed"):
        exp.train_experimental(store, dataset_id=manifest["id"], run_id="changed")
    assert not (store.path / "models/changed.pt").exists()


def test_reanalysis_cannot_transfer_label_between_similar_positions(tmp_path, observation):
    _, store = stores(tmp_path)
    game = experimental_game(observation, training_seed(observation), frames=1)
    store.save_replay(game)
    store.put("search-targets", "mismatched", {"id": "mismatched", "replayId": game["id"],
        "decisionIndex": 0, "actor": 0, "observationHash": "another-observation", "probabilities": {"action-0": 1., "action-1": 0.},
        "dataTier": "experimental", "status": "supported", "source": "reanalysis", "independentSeeds": 2})
    manifest = exp.prepare_dataset(store, [game["id"]])
    assert manifest["searchRows"] == 0 and manifest["searchTargets"] == []
    assert store.get("dataset-rows", manifest["id"])[0]["policyDistribution"] is None


def test_teaching_snapshot_copies_only_referenced_evidence_and_prebounds_reads(tmp_path, observation, monkeypatch):
    main, store = stores(tmp_path)
    record = review(main, fixture_record(tmp_path, main, observation))
    main.put("teaching-reviews", "unrelated", {"id": "unrelated", "largeProse": "not needed"})
    snapshot = exp.snapshot_teaching(main, store, "bounded")
    assert set(snapshot["categories"]["teaching-reviews"]) == {record["reviewHash"]}
    assert set(snapshot["categories"]["fixture-receipts"]) == {record["id"]}
    monkeypatch.setattr(exp, "MAX_ROWS_BYTES", 10)
    with pytest.raises(ValueError, match="source history"):
        exp.snapshot_teaching(main, store, "too-large")
    assert not store.location("teaching-snapshots", "too-large").exists()
