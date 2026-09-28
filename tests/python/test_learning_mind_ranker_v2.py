from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.learning_mind.ranker_features import (MACRO_FEATURE_SCHEMA_HASH,
    MACRO_FEATURE_SCHEMA, candidate_features_v2)
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.macro import MacroCandidateV1
from ptcg_lab.learning_mind.ranker import XGBoostMacroRanker
from ptcg_lab.learning_mind.ranker_v2 import (validate_ranker_v2_report,
    _bootstrap_mean, fit_macro_ranker_v2, predict_macro_ranker_v2,
    verify_macro_ranker_v2_artifact, _validated_macro_labels)
from ptcg_lab.learning_mind.schema import identity_hash
from test_learning_mind_representation import observation
from ptcg_lab.storage import digest as observation_digest


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
        "featureImplementationSha256": file_sha256(Path(ranker_features.__file__))}
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


def test_ranker_v2_position_bootstrap_is_reproducible_and_reports_empty_samples():
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    first = _bootstrap_mean(values, seed_material="frozen-position-set")
    second = _bootstrap_mean(values, seed_material="frozen-position-set")
    assert first == second
    assert first["interval95"]["low"] <= first["mean"] <= first["interval95"]["high"]
    assert first["replicates"] == 2000 and first["method"] == "position-bootstrap-percentile-v1"
    assert _bootstrap_mean([], seed_material="empty")["interval95"] == {"low": None, "high": None}


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
        "trees": [json.loads(tree) for tree in ranker.model.get_dump(dump_format="json")]}
    portable_scores = predict_macro_ranker_v2(artifact, features)
    np.testing.assert_allclose(portable_scores, scores, rtol=1e-6, atol=1e-6)

    model_path = tmp_path / "ranker.json"
    model_path.write_text(json.dumps(artifact, sort_keys=True))
    report = {"kind": "xgboost-macro-ranker-v2", "featureSchema": MACRO_FEATURE_SCHEMA,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": file_sha256(Path(ranker_features.__file__)),
        "modelSha256": file_sha256(model_path), "modelFeatureCount": 640}
    report["reportHash"] = identity_hash(report)
    report_path = tmp_path / "ranker.manifest.json"
    report_path.write_text(json.dumps(report))
    verified = verify_macro_ranker_v2_artifact(model_path, report_path)
    assert verified["report"]["modelFeatureCount"] == 640
    np.testing.assert_allclose(predict_macro_ranker_v2(verified["artifact"], features), scores,
                               rtol=1e-6, atol=1e-6)

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
    for family in families:
        for split in splits:
            hashes = []
            for archetype in archetypes:
                position_hash = f"{family}-{split}-{archetype}"
                hashes.append(position_hash)
                obs = observation()
                obs["players"][1]["active"]["card"]["id"] = f"PUBLIC-{archetype}"
                obs["turn"] += position_index
                position_index += 1
                position_hash = observation_digest(obs)
                hashes[-1] = position_hash
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
                records.append({"positionHash": position_hash, "identity": identity,
                    "opponentPolicyFamily": family, "opponentArchetype": archetype,
                    "split": split, "status": "collected", "observation": obs,
                    "candidateCount": len(labels), "sourceGameId": f"source-{position_index}",
                    "labels": labels})
            selection_entries.append({"policyFamily": family, "split": split,
                "sourceGames": len(hashes), "positionHashes": hashes})

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
        "policyFamilies": list(families), "positions": len(records), "files": files}
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


def test_ranker_v2_rejects_relative_targets_not_centered_on_best_completed_plan(tmp_path):
    labels_dir, _selection = _write_ranker_v2_fit_fixture(tmp_path)
    record_path = next((labels_dir / "python-heuristic").glob("*.json"))
    record = json.loads(record_path.read_text())
    record["labels"][0]["relativeResult"] = 0.25
    with pytest.raises(ValueError, match="best completed result"):
        _validated_macro_labels(record)
