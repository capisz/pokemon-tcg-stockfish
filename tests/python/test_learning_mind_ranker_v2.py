from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.learning_mind.ranker_features import (MACRO_FEATURE_SCHEMA_HASH,
    MACRO_FEATURE_SCHEMA, candidate_features_v2)
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.ranker_portable import inference_implementation_sha256
from ptcg_lab.learning_mind import ranker_v2
from ptcg_lab.learning_mind.macro import MacroCandidateV1, rollout_seed
from ptcg_lab.learning_mind.ranker import XGBoostMacroRanker
from ptcg_lab.learning_mind.ranker_v2 import (validate_ranker_v2_report,
    _bootstrap_mean, fit_macro_ranker_v2, predict_macro_ranker_v2,
    verify_macro_ranker_v2_artifact, _validated_macro_labels, _validate_source_game_units)
from ptcg_lab.learning_mind.schema import identity_hash
from test_learning_mind_representation import observation
from ptcg_lab.storage import digest as observation_digest


def _measured_report_metrics(position_hash="fixture-position"):
    detail = {"positionHash": position_hash, "sourceGameId": "fixture-game",
        "opponentArchetype": "crustle", "opponentPolicyFamily": "python-heuristic",
        "candidates": 2, "top1RelativeRegret": 0.0, "top3Recall": True,
        "pairwiseCorrect": 1, "pairwiseComparisons": 1}
    overall = _bootstrap_mean([0.0], seed_material=f"macro-ranker-v2|all|{position_hash}",
                              group_ids=["fixture-game"])
    groups = {}
    for field in ("opponentArchetype", "opponentPolicyFamily"):
        value = detail[field]
        result = _bootstrap_mean([0.0], seed_material=f"macro-ranker-v2|{field}|{value}|{position_hash}",
                                 group_ids=["fixture-game"])
        groups[field] = {value: {"positions": 1, "meanTop1RelativeRegret": 0.0,
            "meanTop1RelativeRegretCI95": result["interval95"], "bootstrapSeed": result["seed"],
            "independentSourceGames": result["independentUnits"]}}
    return {"status": "measured", "requestedPositions": 1, "positions": 1,
        "insufficientPositionHashes": [], "meanTop1RelativeRegret": 0.0,
        "meanTop1RelativeRegretCI95": overall["interval95"],
        "bootstrap": {"method": overall["method"], "replicates": overall["replicates"],
            "seed": overall["seed"], "independentSourceGames": overall["independentUnits"]},
        "top3Recall": 1.0, "pairwiseComparisons": 1, "pairwiseOrderingAccuracy": 1.0,
        "byOpponentArchetype": groups["opponentArchetype"],
        "byOpponentPolicyFamily": groups["opponentPolicyFamily"], "details": [detail]}


def test_ranker_v2_features_are_deterministic_state_and_plan_sensitive():
    obs = observation()
    candidate = {"turn_intent": "attack", "intended_attack": "Eon Blade",
        "attack_target": "active", "action_sequence": [{"type": "attack", "cardId": "EON-1",
        "targetRef": {"playerId": 1, "zone": "active"}}]}
    original = candidate_features_v2(obs, candidate)
    assert MACRO_FEATURE_SCHEMA_HASH
    assert original.shape == (640,) and np.isfinite(original).all()
    np.testing.assert_array_equal(original, candidate_features_v2(obs, candidate))

    hidden_change = json.loads(json.dumps(obs))
    hidden_change["players"][1]["hand"] = [{"id": "SECRET", "name": "Hidden card"}]
    np.testing.assert_array_equal(original, candidate_features_v2(hidden_change, candidate))

    visible_change = json.loads(json.dumps(obs))
    visible_change["players"][1]["active"]["card"]["id"] = "DIFFERENT-PUBLIC-POKEMON"
    assert not np.array_equal(original, candidate_features_v2(visible_change, candidate))
    changed_plan = {**candidate, "intended_attack": "Different attack"}
    assert not np.array_equal(original, candidate_features_v2(obs, changed_plan))


def test_ranker_v2_encodes_attack_target_slot_not_simulation_action_ids():
    obs = observation()
    candidate = {"turn_intent": "attack", "intended_attack": "Eon Blade",
        "action_ids": ["deterministic-id-a"], "action_sequence": [{"id": "deterministic-id-a",
        "type": "attack", "targetRef": {"playerId": 1, "zone": "active"}}]}
    first = candidate_features_v2(obs, candidate)
    changed_id = json.loads(json.dumps(candidate))
    changed_id["action_ids"] = ["different-id"]
    changed_id["action_sequence"][0]["id"] = "different-id"
    np.testing.assert_array_equal(first, candidate_features_v2(obs, changed_id))
    changed_target = json.loads(json.dumps(candidate))
    changed_target["action_sequence"][0]["targetRef"] = {"playerId": 1, "zone": "bench", "index": 0}
    assert not np.array_equal(first, candidate_features_v2(obs, changed_target))


def test_ranker_v2_report_rejects_feature_or_source_identity_drift():
    from pathlib import Path
    from ptcg_lab.learning_mind import ranker_features
    report = {"kind": "xgboost-macro-ranker-v2", "featureSchema": MACRO_FEATURE_SCHEMA,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": file_sha256(Path(ranker_features.__file__)),
        "inferenceImplementationSha256": inference_implementation_sha256(),
        "evaluationImplementationSha256": file_sha256(Path(ranker_v2.__file__)),
        "trainingImplementationSha256": file_sha256(Path(ranker_v2.__file__).with_name("ranker.py")),
        "trainingLibrary": ranker_v2._training_library_identity(),
        "iteration": {"iteration": 1, "teacher_hash": "teacher", "opponent_policy_hash": "opponents",
            "input_hash": "fixture-input", "position_hashes": ["fixture-position"]},
        "inputManifestHash": "fixture-input", "trainingPositions": 1, "trainingCandidates": 2,
        "training": _measured_report_metrics(), "development": _measured_report_metrics(),
        "holdouts": [{"kind": kind, "status": "measured", "metrics": _measured_report_metrics()}
            for kind in ("leave-one-opponent-archetype-out", "frozen-policy-family")],
        "acceptance": "review-required", "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    validate_ranker_v2_report(report)
    changed = {**report, "featureSchemaHash": "wrong"}
    with pytest.raises(ValueError, match="feature schema"):
        validate_ranker_v2_report(changed)
    changed_source = {**report, "featureImplementationSha256": "different"}
    changed_source["reportHash"] = identity_hash({key: value for key, value in changed_source.items()
                                                   if key != "reportHash"})
    with pytest.raises(ValueError, match="implementation"):
        validate_ranker_v2_report(changed_source)
    changed_inference = {**report, "inferenceImplementationSha256": "different"}
    changed_inference["reportHash"] = identity_hash({key: value for key, value in changed_inference.items()
                                                      if key != "reportHash"})
    with pytest.raises(ValueError, match="inference implementation"):
        validate_ranker_v2_report(changed_inference)
    changed_evaluation = {**report, "evaluationImplementationSha256": "different"}
    changed_evaluation["reportHash"] = identity_hash({key: value for key, value in changed_evaluation.items()
                                                       if key != "reportHash"})
    with pytest.raises(ValueError, match="evaluation implementation"):
        validate_ranker_v2_report(changed_evaluation)
    changed_training = {**report, "trainingImplementationSha256": "different"}
    changed_training["reportHash"] = identity_hash({key: value for key, value in changed_training.items()
                                                      if key != "reportHash"})
    with pytest.raises(ValueError, match="training implementation"):
        validate_ranker_v2_report(changed_training)
    changed_library = {**report, "trainingLibrary": {"name": "xgboost", "version": ""}}
    changed_library["reportHash"] = identity_hash({key: value for key, value in changed_library.items()
                                                    if key != "reportHash"})
    with pytest.raises(ValueError, match="training library identity"):
        validate_ranker_v2_report(changed_library)


def test_ranker_v2_position_bootstrap_is_reproducible_and_reports_empty_samples():
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    first = _bootstrap_mean(values, seed_material="frozen-position-set")
    second = _bootstrap_mean(values, seed_material="frozen-position-set")
    assert first == second
    assert first["interval95"]["low"] <= first["mean"] <= first["interval95"]["high"]
    assert first["replicates"] == 2000
    assert first["method"] == "source-game-cluster-bootstrap-percentile-v1"
    assert first["independentUnits"] == len(values)
    assert _bootstrap_mean([], seed_material="empty")["interval95"] == {"low": None, "high": None}
    clustered = _bootstrap_mean([0.0, 1.0, 0.5], seed_material="clustered",
                                group_ids=["same-game", "same-game", "other-game"])
    assert clustered["independentUnits"] == 2


def test_ranker_v2_portable_dump_matches_xgboost_scores_and_verifies_hash(tmp_path):
    from pathlib import Path
    from ptcg_lab.learning_mind import ranker_features
    features, labels, groups = [], [], []
    for index in range(16):
        features.extend([[float(index % 2)] + [0.] * 639,
                         [float(1 - index % 2)] + [1.] * 639])
        labels.extend([1., 0.] if index % 2 == 0 else [0., 1.])
        groups.append(2)
    ranker = XGBoostMacroRanker(n_estimators=8, max_depth=2).fit(features, labels, groups)
    scores = ranker.predict(features)
    artifact = {"schemaVersion": 1, "kind": "xgboost-macro-ranker-v2-portable",
        "objective": "rank:pairwise", "featureCount": 640,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": file_sha256(Path(ranker_features.__file__)),
        "inferenceImplementationSha256": inference_implementation_sha256(),
        "trees": [json.loads(tree) for tree in ranker.model.get_dump(dump_format="json")]}
    portable_scores = predict_macro_ranker_v2(artifact, features)
    np.testing.assert_allclose(portable_scores, scores, rtol=1e-6, atol=1e-6)

    model_path = tmp_path / "ranker.json"
    model_path.write_text(json.dumps(artifact, sort_keys=True))
    report = {"kind": "xgboost-macro-ranker-v2", "featureSchema": MACRO_FEATURE_SCHEMA,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": file_sha256(Path(ranker_features.__file__)),
        "inferenceImplementationSha256": inference_implementation_sha256(),
        "evaluationImplementationSha256": file_sha256(Path(ranker_v2.__file__)),
        "trainingImplementationSha256": file_sha256(Path(ranker_v2.__file__).with_name("ranker.py")),
        "trainingLibrary": ranker_v2._training_library_identity(),
        "iteration": {"iteration": 1, "teacher_hash": "teacher", "opponent_policy_hash": "opponents",
            "input_hash": "fixture-input", "position_hashes": ["fixture-position"]},
        "inputManifestHash": "fixture-input", "trainingPositions": 1, "trainingCandidates": 2,
        "modelSha256": file_sha256(model_path), "modelFeatureCount": 640,
        "training": _measured_report_metrics(), "development": _measured_report_metrics(),
        "holdouts": [{"kind": kind, "status": "measured", "metrics": _measured_report_metrics()}
            for kind in ("leave-one-opponent-archetype-out", "frozen-policy-family")],
        "acceptance": "review-required", "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    report_path = tmp_path / "ranker.manifest.json"
    report_path.write_text(json.dumps(report))
    verified = verify_macro_ranker_v2_artifact(model_path, report_path)
    assert verified["report"]["modelFeatureCount"] == 640
    np.testing.assert_allclose(predict_macro_ranker_v2(verified["artifact"], features), scores,
                               rtol=1e-6, atol=1e-6)
    changed_artifact = {**artifact, "inferenceImplementationSha256": "different"}
    with pytest.raises(ValueError, match="inference identity"):
        predict_macro_ranker_v2(changed_artifact, features)

    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        verify_macro_ranker_v2_artifact(model_path, report_path)


def _write_ranker_v2_fit_fixture(root):
    families = ("python-heuristic", "typescript-heuristic")
    splits = ("train", "development")
    archetypes = ("crustle", "dragapult")
    identity = {"fixtureIdentity": "synthetic-ranker-v2"}
    selection_entries = []
    records = []
    position_index = 0
    rollout_identities = {family: f"synthetic-rollout-{family}" for family in families}
    for family in families:
        for split in splits:
            hashes = []
            selected_positions = []
            for archetype in archetypes:
                position_hash = f"{family}-{split}-{archetype}"
                hashes.append(position_hash)
                obs = observation()
                obs["players"][1]["active"]["card"]["id"] = f"PUBLIC-{archetype}"
                obs["turn"] += position_index
                position_index += 1
                position_hash = observation_digest(obs)
                hashes[-1] = position_hash
                source_game_id = f"source-{position_index}"
                selected_positions.append({"positionHash": position_hash,
                    "sourceGameId": source_game_id})
                labels = []
                for choice in (0, 1):
                    action = obs["legalActions"][2 + choice]
                    candidate_value = MacroCandidateV1(turn_intent="attack",
                        intended_attack=str(action.get("label")), action_ids=(str(action["id"]),),
                        action_sequence=(action,))
                    candidate = asdict(candidate_value)
                    result = float(1 - choice)
                    uncertainty = (0.25 / 2) ** 0.5
                    labels.append({"candidate": candidate, "candidateHash": candidate_value.key(),
                        "attemptedRollouts": 2, "completedRollouts": 2,
                        "outcomes": {"finished": 2, "truncated": 0, "error": 0},
                        "outcomeReasons": {}, "expectedResult": result,
                        "relativeResult": result - 1.0, "uncertainty": uncertainty,
                        "weight": 2 / (1 + uncertainty)})
                namespace = "training" if split == "train" else "development"
                records.append({"positionHash": position_hash, "identity": identity,
                    "opponentPolicyFamily": family, "opponentArchetype": archetype,
                    "split": split, "status": "collected", "observation": obs,
                    "candidateCount": len(labels), "sourceGameId": source_game_id,
                    "seedNamespace": namespace, "rolloutIdentity": rollout_identities[family],
                    "rolloutSeeds": [rollout_seed(namespace, position_hash, index,
                        rollout_identities[family]) for index in range(2)],
                    "labels": labels})
            selection_entries.append({"policyFamily": family, "split": split,
                "sourceGames": len(hashes), "positionHashes": hashes,
                "positions": selected_positions})

    selection = {"schemaVersion": 1, "identity": identity, "splits": selection_entries}
    selection["selectionHash"] = identity_hash(selection)
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection))
    labels_dir = root / "combined"
    files = []
    for row in records:
        relative = Path(row["opponentPolicyFamily"]) / f"{row['positionHash']}.json"
        destination = labels_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")))
        files.append({"path": relative.as_posix(), "sha256": file_sha256(destination)})
    manifest = {"schemaVersion": 1, "kind": "combined-macro-label-runs-v1", "identity": identity,
        "selectionHash": selection["selectionHash"],
        "selectionManifestSha256": file_sha256(selection_path),
        "selectedPositionHashes": sorted(row["positionHash"] for row in records),
        "positionsByFamilySplit": {family: {split: len(archetypes) for split in splits} for family in families},
        "positionsBySplit": {split: sum(row["split"] == split for row in records) for split in splits},
        "policyFamilies": list(families), "positions": len(records), "files": files,
        "sourceRuns": [{"policyFamilies": [family], "rolloutIdentity": rollout_identities[family]}
                       for family in families]}
    manifest["manifestHash"] = identity_hash(manifest)
    (labels_dir / "manifest.json").write_text(json.dumps(manifest))
    return labels_dir, selection_path


def test_ranker_v2_end_to_end_fit_holdouts_and_portable_artifact(tmp_path, monkeypatch):
    from ptcg_lab.learning_mind import ranker_v2
    from ptcg_lab.learning_mind.ranker import XGBoostMacroRanker
    monkeypatch.setattr(ranker_v2, "XGBoostMacroRanker",
        lambda: XGBoostMacroRanker(n_estimators=4, max_depth=2))
    labels, selection = _write_ranker_v2_fit_fixture(tmp_path)
    model_path = tmp_path / "ranker.json"
    report = fit_macro_ranker_v2(labels, model_path, selection_path=selection,
        teacher_hash="frozen-teacher", opponent_policy_hash="frozen-opponent-set")
    assert report["kind"] == "xgboost-macro-ranker-v2"
    assert report["acceptance"] == "review-required"
    assert report["development"]["positions"] == 4
    assert report["development"]["meanTop1RelativeRegretCI95"]["low"] <= \
        report["development"]["meanTop1RelativeRegret"] <= \
        report["development"]["meanTop1RelativeRegretCI95"]["high"]
    assert len(report["holdouts"]) == 4
    verified = verify_macro_ranker_v2_artifact(model_path, model_path.with_suffix(".manifest.json"))
    assert verified["report"]["reportHash"] == report["reportHash"]
    assert len(predict_macro_ranker_v2(verified["artifact"], np.zeros((2, 640)))) == 2

    forged_report = json.loads(model_path.with_suffix(".manifest.json").read_text())
    forged_report["development"]["positions"] = 0
    forged_report["reportHash"] = identity_hash({key: value for key, value in forged_report.items()
                                                  if key != "reportHash"})
    with pytest.raises(ValueError, match="empty coverage does not reconcile"):
        validate_ranker_v2_report(forged_report)

    forged_summary = json.loads(model_path.with_suffix(".manifest.json").read_text())
    forged_summary["development"]["top3Recall"] = 0.123
    forged_summary["reportHash"] = identity_hash({key: value for key, value in forged_summary.items()
                                                    if key != "reportHash"})
    with pytest.raises(ValueError, match="aggregate metrics do not recompute"):
        validate_ranker_v2_report(forged_summary)


@pytest.mark.parametrize("incomplete_split", ("train", "development"))
def test_ranker_v2_does_not_report_partial_label_coverage_as_measured(
        tmp_path, monkeypatch, incomplete_split):
    from ptcg_lab.learning_mind import ranker_v2
    monkeypatch.setattr(ranker_v2, "XGBoostMacroRanker",
        lambda: XGBoostMacroRanker(n_estimators=4, max_depth=2))
    labels_dir, selection_path = _write_ranker_v2_fit_fixture(tmp_path)
    manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = None
    for item in manifest["files"]:
        candidate_path = labels_dir / item["path"]
        candidate_record = json.loads(candidate_path.read_text())
        if (candidate_record["opponentPolicyFamily"] == "python-heuristic"
                and candidate_record["split"] == incomplete_split):
            entry = item
            record_path = candidate_path
            record = candidate_record
            break
    assert entry is not None
    missing = record["labels"][1]
    missing.update(completedRollouts=0,
        outcomes={"finished": 0, "truncated": 2, "error": 0},
        expectedResult=None, relativeResult=None, uncertainty=None, weight=0)
    record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    entry["sha256"] = file_sha256(record_path)
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                               if key != "manifestHash"})
    manifest_path.write_text(json.dumps(manifest))

    report = fit_macro_ranker_v2(labels_dir, tmp_path / "ranker.json",
        selection_path=selection_path, teacher_hash="frozen-teacher",
        opponent_policy_hash="frozen-opponent-set")
    if incomplete_split == "development":
        assert report["development"]["status"] == "insufficient"
        assert report["development"]["requestedPositions"] == 4
        assert report["development"]["positions"] == 3
        assert len(report["development"]["insufficientPositionHashes"]) == 1
    else:
        assert report["development"]["status"] == "measured"
        assert report["training"]["status"] == "insufficient"
        assert report["training"]["requestedPositions"] == 4
        assert report["training"]["positions"] == 3
        assert len(report["training"]["insufficientPositionHashes"]) == 1
    assert report["acceptance"] == "insufficient"


def test_ranker_v2_tied_candidate_outcomes_are_not_measured_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(ranker_v2, "XGBoostMacroRanker",
        lambda: XGBoostMacroRanker(n_estimators=4, max_depth=2))
    labels_dir, selection_path = _write_ranker_v2_fit_fixture(tmp_path)
    manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        record_path = labels_dir / item["path"]
        record = json.loads(record_path.read_text())
        for label in record["labels"]:
            label["expectedResult"] = 0.5
            label["relativeResult"] = 0.0
        record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
        item["sha256"] = file_sha256(record_path)
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                               if key != "manifestHash"})
    manifest_path.write_text(json.dumps(manifest))

    report = fit_macro_ranker_v2(labels_dir, tmp_path / "ranker.json",
        selection_path=selection_path, teacher_hash="frozen-teacher",
        opponent_policy_hash="frozen-opponent-set")
    assert report["training"]["status"] == "insufficient"
    assert report["training"]["insufficientReason"] == "no-nontied-candidate-comparisons"
    assert report["development"]["status"] == "insufficient"
    assert all(row["status"] == "insufficient" for row in report["holdouts"])
    assert report["acceptance"] == "insufficient"


def test_ranker_v2_rejects_relative_targets_not_centered_on_best_completed_plan(tmp_path):
    labels_dir, _selection = _write_ranker_v2_fit_fixture(tmp_path)
    record_path = next((labels_dir / "python-heuristic").glob("*.json"))
    record = json.loads(record_path.read_text())
    record["labels"][0]["relativeResult"] = 0.25
    with pytest.raises(ValueError, match="best completed result"):
        _validated_macro_labels(record)


@pytest.mark.parametrize("tamper", ("promotion-namespace", "seed", "rollout-identity"))
def test_ranker_v2_rejects_seed_namespace_or_identity_drift(tmp_path, tamper):
    labels_dir, _selection = _write_ranker_v2_fit_fixture(tmp_path)
    record_path = next((labels_dir / "python-heuristic").glob("*.json"))
    record = json.loads(record_path.read_text())
    if tamper == "promotion-namespace":
        record["seedNamespace"] = "promotion"
    elif tamper == "seed":
        record["rolloutSeeds"][0] += 1
    else:
        with pytest.raises(ValueError, match="rollout identity"):
            _validated_macro_labels(record, expected_rollout_identity="wrong-family-run")
        return
    with pytest.raises(ValueError, match="seed namespace or rollout identity|seed list"):
        _validated_macro_labels(record)


def test_ranker_v2_bootstrap_units_must_match_frozen_unique_source_games(tmp_path):
    labels_dir, selection_path = _write_ranker_v2_fit_fixture(tmp_path)
    records = [json.loads(path.read_text()) for family in ("python-heuristic", "typescript-heuristic")
               for path in (labels_dir / family).glob("*.json")]
    _validate_source_game_units(selection_path, records)

    selection = json.loads(selection_path.read_text())
    first, second = selection["splits"][0]["positions"]
    second["sourceGameId"] = first["sourceGameId"]
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="reuses or omits a source game"):
        _validate_source_game_units(selection_path, records)


def test_ranker_v2_train_and_development_games_must_be_disjoint_across_families(tmp_path):
    labels_dir, selection_path = _write_ranker_v2_fit_fixture(tmp_path)
    records = [json.loads(path.read_text()) for family in ("python-heuristic", "typescript-heuristic")
               for path in (labels_dir / family).glob("*.json")]
    selection = json.loads(selection_path.read_text())
    python_train = next(row for row in selection["splits"]
                        if row["policyFamily"] == "python-heuristic" and row["split"] == "train")
    typescript_development = next(row for row in selection["splits"]
                                  if row["policyFamily"] == "typescript-heuristic"
                                  and row["split"] == "development")
    shared_game = python_train["positions"][0]["sourceGameId"]
    leaked_position = typescript_development["positions"][0]["positionHash"]
    typescript_development["positions"][0]["sourceGameId"] = shared_game
    next(row for row in records if row["positionHash"] == leaked_position)["sourceGameId"] = shared_game
    selection["selectionHash"] = identity_hash({key: value for key, value in selection.items()
                                                  if key != "selectionHash"})
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="across policy families"):
        _validate_source_game_units(selection_path, records)
