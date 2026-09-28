from __future__ import annotations

import json

import numpy as np
import pytest

from ptcg_lab.learning_mind.ranker_features import (MACRO_FEATURE_SCHEMA_HASH,
    MACRO_FEATURE_SCHEMA, candidate_features_v2)
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.ranker import XGBoostMacroRanker
from ptcg_lab.learning_mind.ranker_v2 import (validate_ranker_v2_report,
    predict_macro_ranker_v2, verify_macro_ranker_v2_artifact)
from ptcg_lab.learning_mind.schema import identity_hash
from test_learning_mind_representation import observation


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
