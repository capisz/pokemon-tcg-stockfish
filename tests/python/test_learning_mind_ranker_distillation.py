from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.learning_mind import dataset_v1, experiment, ranker_distillation
from ptcg_lab.learning_mind.dataset_v1 import file_sha256, load_dataset
from ptcg_lab.learning_mind.encoding import encode_decision
from ptcg_lab.learning_mind.macro import MacroCandidateV1, rollout_seed
from ptcg_lab.learning_mind import ranker_features
from ptcg_lab.learning_mind.ranker_features import MACRO_FEATURE_SCHEMA, MACRO_FEATURE_SCHEMA_HASH
from ptcg_lab.learning_mind.ranker_portable import inference_implementation_sha256
from ptcg_lab.learning_mind.schema import IdentityManifest, identity_hash
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
from ptcg_lab.storage import digest as legacy_digest
from test_learning_mind_representation import observation


def _ranker_distillation_fixture(tmp_path, monkeypatch):
    identity_manifest = IdentityManifest.create(engine_build_hash="engine-v1", deck_manifests={},
                                                card_metadata={})
    identity = identity_manifest.record()
    obs = observation()
    attack_actions = [action for action in obs["legalActions"] if action["type"] == "attack"]
    tracker = ObservableHistoryTracker(0).update(obs)
    encoded = encode_decision(obs, tracker)
    position_hash = legacy_digest(obs)
    pool_row = {"positionHash": position_hash, "familyId": "family-1", "sourceGameId": "game-1",
        "sourceDecisionIndex": 0, "actor": 0, "deckHash": "deck-hash",
        "opponentArchetype": "crustle", "opponentPolicyFamily": "python-heuristic",
        "featureIdentityHash": encoded.identity, "split": "train", "observation": obs, "tracker": tracker}
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool_rows = pool_dir / "rows.jsonl"
    pool_rows.write_text(json.dumps(pool_row) + "\n")
    pool_manifest = {"id": "learning-mind-macro-position-pool-v1", "identity": identity,
        "rows": 1, "rowsSha256": file_sha256(pool_rows)}
    pool_manifest["manifestHash"] = identity_hash(pool_manifest)
    (pool_dir / "manifest.json").write_text(json.dumps(pool_manifest))

    rollout_id = "rollout-python-heuristic"
    seeds = [rollout_seed("training", position_hash, 0, rollout_id)]
    labels = []
    for action, result in zip(attack_actions, (.75, .25)):
        candidate = MacroCandidateV1(turn_intent="attack", intended_attack=action["label"],
                                     action_ids=(action["id"],), action_sequence=(action,))
        uncertainty = .5
        labels.append({"candidate": json.loads(json.dumps(asdict(candidate))), "candidateHash": candidate.key(),
            "attemptedRollouts": 1, "completedRollouts": 1,
            "outcomes": {"finished": 1, "truncated": 0, "error": 0}, "outcomeReasons": {},
            "expectedResult": result, "relativeResult": result - .75,
            "uncertainty": uncertainty, "weight": 1 / (1 + uncertainty)})
    record = {"positionHash": position_hash, "split": "train", "sourceGameId": "game-1",
        "familyId": "family-1", "opponentPolicyFamily": "python-heuristic",
        "opponentArchetype": "crustle", "observation": obs, "candidateCount": len(labels),
        "labels": labels, "seedNamespace": "training", "rolloutIdentity": rollout_id,
        "rolloutSeeds": seeds}

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    labels_manifest = {"identity": identity, "selectionHash": "frozen-selection-hash",
        "sourceRuns": [{"policyFamilies": [family], "rolloutIdentity": f"rollout-{family}"}
                       for family in ("python-heuristic", "typescript-heuristic")]}
    (labels_dir / "manifest.json").write_text(json.dumps(labels_manifest))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"selectionHash": "frozen-selection-hash"}))
    monkeypatch.setattr(experiment, "_load_ranker_input", lambda *_args: (labels_manifest, [record]))
    monkeypatch.setattr(ranker_distillation, "_validate_source_game_units", lambda *_args: None)

    model_path = tmp_path / "ranker.json"
    artifact = {"schemaVersion": 1, "kind": "xgboost-macro-ranker-v2-portable",
        "objective": "rank:pairwise", "featureCount": 640,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": file_sha256(Path(ranker_features.__file__)),
        "inferenceImplementationSha256": inference_implementation_sha256(),
        "trees": [{"nodeid": 0, "leaf": 0.0}]}
    model_path.write_text(json.dumps(artifact))
    report = {"kind": "xgboost-macro-ranker-v2", "featureSchema": MACRO_FEATURE_SCHEMA,
        "featureSchemaHash": MACRO_FEATURE_SCHEMA_HASH,
        "featureImplementationSha256": artifact["featureImplementationSha256"],
        "inferenceImplementationSha256": inference_implementation_sha256(),
        "identity": identity, "selectionHash": labels_manifest["selectionHash"],
        "inputManifestSha256": file_sha256(labels_dir / "manifest.json"),
        "selectionManifestSha256": file_sha256(selection_path),
        "modelSha256": file_sha256(model_path), "modelFeatureCount": 640,
        "development": {"status": "measured"},
        "holdouts": [{"kind": kind, "status": "measured"} for kind in
            ("leave-one-opponent-archetype-out", "frozen-policy-family")],
        "acceptance": "review-required", "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    report_path = tmp_path / "ranker.manifest.json"
    report_path.write_text(json.dumps(report))
    return identity_manifest, identity, pool_dir, labels_dir, selection_path, model_path, report_path


def test_ranker_distillation_is_verified_train_only_and_integrates_into_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_v1, "EXACT_REVIEWS", ())
    identity_manifest, identity, pool, labels, selection, model, report = _ranker_distillation_fixture(tmp_path, monkeypatch)
    distillation_dir = tmp_path / "distillation"
    manifest = ranker_distillation.build_macro_ranker_distillation(output=distillation_dir,
        macro_position_pool=pool, labels_dir=labels, selection_path=selection,
        model_path=model, report_path=report, identity=identity)
    assert manifest["rows"] == 1 and manifest["policySplit"] == "train-only"
    distill_manifest, distill_rows = ranker_distillation.load_macro_ranker_distillation(
        distillation_dir, identity=identity)
    assert len(distill_rows) == 1
    distribution = distill_rows[0]["policyDistribution"]
    assert len(distribution) == len(encode_decision(
        distill_rows[0]["observation"], distill_rows[0]["tracker"]).action_classes) + 1
    np.testing.assert_allclose(sorted(value for value in distribution if value > 0), [.5, .5], atol=1e-7)
    assert distribution[-1] == 0.0
    assert distill_rows[0]["rankerDistillation"]["candidateCount"] == 2

    source = {"settings": {"identity": identity}, "replays": []}
    source["manifestHash"] = identity_hash(source)
    source_path = tmp_path / "empty-source.json"
    source_path.write_text(json.dumps(source))
    supervised = tmp_path / "supervised"
    combined = dataset_v1.build_dataset(root=tmp_path, output=supervised,
        review_root=tmp_path / "reviews", experimental_root=tmp_path / "experimental",
        source_dataset_manifest=source_path,
        identity=identity_manifest, ranker_distillation_dir=distillation_dir)
    _loaded_manifest, rows = load_dataset(supervised, identity=identity)
    assert combined["policyLabelSources"] == ["macro-ranker-distillation"]
    assert rows[0]["rankerDistillation"]["reportHash"] == distill_manifest["rankerReportHash"]


def test_ranker_distillation_rejects_teacher_report_with_unmeasured_holdout(tmp_path, monkeypatch):
    _identity_manifest, identity, pool, labels, selection, model, report_path = _ranker_distillation_fixture(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text())
    report["holdouts"][0]["status"] = "insufficient"
    report["reportHash"] = identity_hash({key: value for key, value in report.items()
                                          if key != "reportHash"})
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="measured archetype and policy-family"):
        ranker_distillation.build_macro_ranker_distillation(output=tmp_path / "distill",
            macro_position_pool=pool, labels_dir=labels, selection_path=selection,
            model_path=model, report_path=report_path, identity=identity)
