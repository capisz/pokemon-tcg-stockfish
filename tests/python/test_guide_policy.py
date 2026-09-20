from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.dataset import demonstration_examples, demonstration_records
from ptcg_lab.fixtures import from_fixture
from ptcg_lab.features import FEATURE_VERSION, energy_covers_attack, resource_features, resource_coverage
from ptcg_lab.model_registry import list_models, resolve_model
from ptcg_lab.storage import Store, digest
from ptcg_lab import teaching, training
from conftest import make_replay


def fixture_record(root, store, observation, *, family="study-family", partition="train", variation=1,
                   audited_actions=None):
    directory = root / "research"
    directory.mkdir(exist_ok=True)
    path = directory / "curriculum.json"
    families = json.loads(path.read_text())["families"] if path.exists() else []
    if not any(item["id"] == family for item in families):
        families.append({"id": family, "partition": partition, "source": {"author": "Synthetic test", "pages": [4]},
                         "variations": [{"id": f"{family}-{number}", "condition": "Test position"} for number in range(1, 4)]})
    path.write_text(json.dumps({"families": families}))
    receipt = {"fixtureId": family, "variationId": f"{family}-{variation}", "observation": copy.deepcopy(observation),
               "engineVersion": "synthetic-test-only", "fixtureHash": digest({"family": family, "variation": variation}),
               "mechanicsAudit": {"status": "verified", "tests": ["synthetic-transition-test"], "scope": ["test-only"],
                                  "limitations": ["Not a real strategy review"],
                                  "validatedActionIds": audited_actions if audited_actions is not None else [a["id"] for a in observation["legalActions"]]}}
    return from_fixture(store, root, receipt)


def review(store, record, accepted=None):
    return teaching.review(store, record["id"], review_status="reviewed",
                           acceptable_action_ids=accepted or ["action-0"], reasoning="Synthetic test annotation only")


def test_fixture_audit_is_separate_from_game_eligibility_and_never_auto_reviews(tmp_path, observation):
    store = Store(tmp_path / "data")
    record = fixture_record(tmp_path, store, observation, audited_actions=["action-0"])
    assert record["reviewStatus"] == "draft" and not record["trainingEligible"]
    assert demonstration_records(store) == []
    # The deck has no certification or outcome record. This particular audited
    # transition can supply policy evidence after explicit review.
    reviewed = review(store, record)
    assert reviewed["trainingEligible"]
    assert len(demonstration_records(store)) == 1
    unaudited = review(store, record, ["action-1"])
    assert not unaudited["trainingEligible"]
    assert demonstration_records(store) == []


def test_review_receipt_changes_and_partition_relabeling_cannot_admit_demos(tmp_path, observation):
    store = Store(tmp_path / "data")
    record = fixture_record(tmp_path, store, observation)
    review(store, record)
    receipt = store.get("fixture-receipts", record["id"])
    receipt["observation"]["turn"] += 1
    store.put("fixture-receipts", record["id"], receipt)
    assert demonstration_records(store) == []


def test_conflicting_family_quarantine_survives_archiving_bad_copy(tmp_path, observation):
    store = Store(tmp_path / "data")
    record = review(store, fixture_record(tmp_path, store, observation))
    bad_copy = {**record, "id": "wrong-partition-copy", "partition": "test"}
    store.put("teaching", bad_copy["id"], bad_copy)
    assert demonstration_records(store) == []
    store.location("teaching", bad_copy["id"]).unlink()
    assert demonstration_records(store) == []
    with pytest.raises(ValueError, match="immutable"):
        teaching.bind_family(store, "study-family", "test")
    heldout = fixture_record(tmp_path, store, observation, family="test-family", partition="test")
    reviewed = review(store, heldout)
    assert not reviewed["trainingEligible"]
    assert demonstration_records(store) == []
    assert len(demonstration_records(store, partition="test")) == 1
    reviewed["partition"] = "train"
    store.put("teaching", reviewed["id"], reviewed)
    assert demonstration_records(store) == []


def test_policy_only_rejects_guides_and_drafts_without_creating_model(tmp_path, observation):
    store = Store(tmp_path / "data")
    fixture_record(tmp_path, store, observation)
    store.put("guides", "guide", {"id": "guide", "reviewStatus": "reviewed", "text": "Attack to win"})
    with pytest.raises(ValueError, match="No eligible reviewed"):
        training.train_policy(store, epochs=1)
    assert list_models(store) == []
    assert not (store.path / "models").exists()


def test_multi_acceptable_action_loss_rewards_total_mass_without_outcome_labels():
    torch = pytest.importorskip("torch")
    rows = [{"actions": [[], [], []], "acceptableIndices": [0, 1]}]
    logits = torch.tensor([[1., 2., 0.]], requires_grad=True)
    loss = training.policy_loss(logits, torch.ones_like(logits, dtype=torch.bool), rows)
    probability = logits.softmax(-1)
    assert float(loss.detach()) == pytest.approx(float(-(probability[0, :2].sum()).log().detach()))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] < 0 and logits.grad[0, 2] > 0
    assert "expectedResult" not in rows[0] and "outcomeClass" not in rows[0]


def test_reviewed_policy_training_registry_and_unavailable_value(tmp_path, observation):
    torch = pytest.importorskip("torch")
    store = Store(tmp_path / "data")
    record = fixture_record(tmp_path, store, observation)
    review(store, record)
    heldout = fixture_record(tmp_path, store, observation, family="heldout-family", partition="test")
    review(store, heldout)
    rows = demonstration_examples(demonstration_records(store))
    assert "expectedResult" not in rows[0] and "outcomeClass" not in rows[0]
    result = training.train_policy(store, epochs=3)
    path = resolve_model(store, result["id"])
    loaded = training.load_model(path)
    checkpoint = loaded[1]
    assert result["modelKind"] == "policy-only" and not result["valueTrained"]
    assert checkpoint["dataLineage"]["seenSeeds"] == []
    assert checkpoint["dataLineage"]["teachingFamilyIds"] == ["study-family"]
    assert checkpoint["teachingManifest"][0]["id"] != heldout["id"]
    assert result["metrics"]["test"]["status"] == "measured"
    evaluation, scores = training.predict(path, observation, loaded)
    assert evaluation["status"] == "unavailable" and evaluation["score"] is None
    assert evaluation["components"] == [] and evaluation["winProbability"] is None
    assert len(scores) == 2 and all(np.isfinite(score) for score in scores)
    assert training.export_portable_value(path, loaded) is None
    assert list_models(store)[0]["id"] == result["id"]
    assert not list_models(store)[0]["valueTrained"]
    for invalid in ("../outside", "/tmp/model", "missing"):
        with pytest.raises(ValueError):
            resolve_model(store, invalid)
    # Value heads stayed exactly at initialization; they have not seen labels.
    torch.manual_seed(42)
    from ptcg_lab.model import PolicyResourceModel
    initial = PolicyResourceModel().state_dict()
    for name, value in checkpoint["model"].items():
        if name.startswith(("resource_terms.", "baseline", "interaction.", "outcomes.")):
            assert torch.equal(value, initial[name])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        resolve_model(store, result["id"])
    assert list_models(store) == []


def test_policy_resume_keeps_optimizer_and_changed_review_blocks_continuation(tmp_path, observation):
    torch = pytest.importorskip("torch")
    store = Store(tmp_path / "data")
    record = fixture_record(tmp_path, store, observation)
    review(store, record)
    first = training.train_policy(store, epochs=1)
    checkpoint = Path(first["checkpoint"])
    resumed = training.train_policy(store, epochs=2, resume=checkpoint)
    assert resumed["id"] == first["id"] and len(resumed["history"]) == 2
    assert resumed["teachingDiagnostic"]["baselineHash"] == first["teachingDiagnostic"]["baselineHash"]
    assert resumed["teachingDiagnostic"]["positions"][0]["before"] == first["teachingDiagnostic"]["positions"][0]["before"]
    continued = torch.load(checkpoint, weights_only=True)
    fresh = training.train_policy(store, epochs=2)
    complete = torch.load(fresh["checkpoint"], weights_only=True)
    assert all(torch.equal(value, complete["model"][key]) for key, value in continued["model"].items())
    assert fresh["teachingDiagnostic"] == resumed["teachingDiagnostic"]
    review(store, record, ["action-1"])
    with pytest.raises(ValueError, match="identical reviewed data"):
        training.train_policy(store, epochs=3, resume=checkpoint)
    assert store.get("teaching-reviews", continued["teachingManifest"][0]["reviewHash"])["acceptableActionIds"] == ["action-0"]


def test_policy_interruption_and_linked_continuation_keep_a_durable_checkpoint(tmp_path, observation):
    pytest.importorskip("torch")
    store = Store(tmp_path / "data")
    review(store, fixture_record(tmp_path, store, observation))
    class Stop:
        def check(self, **kwargs):
            raise KeyboardInterrupt("Synthetic interruption")
    with pytest.raises(KeyboardInterrupt):
        training.train_policy(store, epochs=1, guard=Stop())
    paused = store.list("experiments")[0]
    assert paused["status"] == "paused" and list_models(store) == []
    parent = Path(paused["checkpoint"])
    report = training.train_policy(store, epochs=1, resume=parent, linked_resume=True)
    assert report["id"] != paused["id"] and report["parent"]["experimentId"] == paused["id"]
    assert parent.exists() and len(list_models(store)) == 1


def test_fixture_rebuild_supersedes_only_drafts_and_bounds_queue(tmp_path, observation):
    store = Store(tmp_path / "data")
    old = fixture_record(tmp_path, store, observation)
    receipt = store.get("fixture-receipts", old["id"])
    receipt["engineVersion"] = "changed-engine"
    newer = from_fixture(store, tmp_path, receipt)
    assert newer["id"] != old["id"]
    assert store.get("teaching", old["id"])["superseded"]
    assert [record["id"] for record in teaching.queue(store)] == [newer["id"]]
    assert store.get("fixture-receipts", old["id"])["engineVersion"] == "synthetic-test-only"
    review(store, newer)
    receipt["engineVersion"] = "another-engine"
    latest = from_fixture(store, tmp_path, receipt)
    assert not store.get("teaching", newer["id"]).get("superseded")
    assert [record["id"] for record in teaching.queue(store)] == [latest["id"]]
    assert not review(store, old)["trainingEligible"]


def test_policy_can_warm_start_outcome_training_with_demonstration_lineage(tmp_path, observation):
    pytest.importorskip("torch")
    store = Store(tmp_path / "data")
    record = fixture_record(tmp_path, store, observation)
    review(store, record)
    policy = training.train_policy(store, epochs=1)
    for index in range(12):
        store.save_replay(make_replay(observation, index))
    result = training.train(store, epochs=1, max_positions=100, warm_start=Path(policy["checkpoint"]))
    model, checkpoint = training.load_model(Path(result["checkpoint"]))
    assert checkpoint["valueTrained"] and checkpoint["modelKind"] == "policy-value"
    assert checkpoint["dataLineage"]["teachingFamilyIds"] == ["study-family"]
    assert result["demonstrationExamples"] == 1
    portable = training.export_portable_value(Path(result["checkpoint"]), (model, checkpoint))
    assert hashlib.sha256(portable["payload"].encode()).hexdigest() == portable["hash"]
    payload = json.loads(portable["payload"])
    assert payload["featureVersion"] == FEATURE_VERSION and payload["valueTrained"]
    assert "action_encoder.0.weight" not in payload["weights"]
    assert "outcomes.weight" not in payload["weights"]
    assert len(portable["payload"]) < 950000


def test_visible_energy_coverage_distinguishes_types_costs_and_unknown_cards(observation):
    board = observation["players"][0]["active"]
    board["card"]["attacks"][0]["cost"] = ["GRASS", "COLORLESS", "COLORLESS"]
    board["energy"] = ["Basic Grass Energy"]
    assert not energy_covers_attack(board)
    board["energy"] = ["Basic Fire Energy", "Mist Energy", "Spiky Energy"]
    assert not energy_covers_attack(board)
    board["energy"] = ["Growing [G] Energy", "Mist Energy", "Spiky Energy"]
    assert energy_covers_attack(board)
    board["energy"] = ["Unknown Special Energy"]
    assert not energy_covers_attack(board)
    assert resource_coverage(observation)["unknownEnergyCards"] == ["Unknown Special Energy"]
    versus_nonex = resource_features(observation)[13]
    observation["players"][1]["active"]["card"]["name"] = "Dragapult ex"
    assert resource_features(observation)[13] > versus_nonex


def test_teaching_bundle_retains_receipts_review_snapshots_and_linked_resume(tmp_path, observation):
    pytest.importorskip("torch")
    from ptcg_lab.bundles import export_bundle, import_bundle
    store = Store(tmp_path / "source")
    record = review(store, fixture_record(tmp_path, store, observation))
    original = training.train_policy(store, epochs=1)
    identities = {"engineBuildHash": "synthetic", "deckManifestHash": "synthetic", "featureVersion": FEATURE_VERSION}
    exported = export_bundle(store, tmp_path / "bundle", identities, min_free=0)
    imported = import_bundle(Store(tmp_path / "destination"), Path(exported["path"]), identities, min_free=0)
    target = Store(Path(imported["dataRoot"]))
    assert len(demonstration_records(target)) == 1
    assert target.get("teaching-reviews", record["reviewHash"])["reviewHash"] == record["reviewHash"]
    assert digest(target.get("fixture-receipts", record["id"])) == record["fixtureReceiptHash"]
    Path(original["checkpoint"]).unlink()  # Original machine path is unavailable.
    imported_checkpoint = resolve_model(target, original["id"])
    continued = training.train_policy(target, epochs=2, resume=imported_checkpoint, linked_resume=True)
    assert continued["parent"]["experimentId"] == original["id"]
    assert continued["id"] != original["id"] and len(continued["history"]) == 2
    assert continued["teachingDiagnostic"]["baselineHash"] == original["teachingDiagnostic"]["baselineHash"]
    assert continued["teachingDiagnostic"]["positions"][0]["before"] == original["teachingDiagnostic"]["positions"][0]["before"]


def test_teaching_diagnostic_records_an_actual_policy_change_without_strength_claim(tmp_path, observation):
    torch = pytest.importorskip("torch")
    from ptcg_lab.model import PolicyResourceModel
    store = Store(tmp_path / "data")
    record = fixture_record(tmp_path, store, observation)
    torch.manual_seed(42)
    original_model = PolicyResourceModel()
    # Deliberately teach the opposite of this random initial preference, using
    # a synthetic annotation rather than claiming a user's real strategic review.
    rows = demonstration_examples([{**record, "acceptableActionIds": ["action-0"]}])
    resources, cards, actions, *_ = training._batch(rows, "cpu")
    initial_index = int(original_model(resources, cards, actions)["policy"][0].argmax())
    accepted = observation["legalActions"][1 - initial_index]["id"]
    reviewed = review(store, record, [accepted])
    result = training.train_policy(store, epochs=20)
    diagnostic = result["teachingDiagnostic"]
    assert diagnostic["kind"] == "training-teaching-diagnostic"
    assert diagnostic["comparedPositions"] == 1 and diagnostic["selectedActionsChanged"] == 1
    item = diagnostic["positions"][0]
    assert item["observationHash"] == reviewed["positionHash"] and item["reviewHash"] == reviewed["reviewHash"]
    assert item["before"]["selectedActionId"] == observation["legalActions"][initial_index]["id"]
    assert item["after"]["selectedActionId"] == accepted
    assert item["acceptableMassChange"] > 0 and item["after"]["acceptableActionProbabilityMass"] > .5
    assert "does not measure" in diagnostic["description"]
    assert training.predict(Path(result["checkpoint"]), observation)[0]["score"] is None


def test_missing_or_changed_fixed_baseline_blocks_resume(tmp_path, observation):
    torch = pytest.importorskip("torch")
    store = Store(tmp_path / "data")
    review(store, fixture_record(tmp_path, store, observation))
    result = training.train_policy(store, epochs=1)
    original = torch.load(result["checkpoint"], weights_only=True)
    missing = copy.deepcopy(original)
    missing.pop("teachingBaseline")
    tampered = copy.deepcopy(original)
    tampered["teachingBaseline"]["positions"][0]["acceptableActionProbabilityMass"] = .99
    for index, changed in enumerate((missing, tampered)):
        path = tmp_path / f"invalid-baseline-{index}.pt"
        torch.save(changed, path)
        with pytest.raises(ValueError, match="intact fixed initial teaching baseline"):
            training.train_policy(store, epochs=2, resume=path)


def test_teaching_diagnostic_is_bounded_without_dropping_training_examples(tmp_path, observation, monkeypatch):
    pytest.importorskip("torch")
    store = Store(tmp_path / "data")
    for variation in range(1, 4):
        review(store, fixture_record(tmp_path, store, observation, variation=variation))
    monkeypatch.setattr(training, "TEACHING_DIAGNOSTIC_LIMIT", 2)
    result = training.train_policy(store, epochs=1)
    diagnostic = result["teachingDiagnostic"]
    assert result["demonstrationExamples"] == diagnostic["admittedTrainingPositions"] == 3
    assert diagnostic["comparedPositions"] == 2 and diagnostic["omittedPositions"] == 1
    assert [item["teachingId"] for item in diagnostic["positions"]] == sorted(record["id"] for record in demonstration_records(store))[:2]
