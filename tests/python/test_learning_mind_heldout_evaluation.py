from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.learning_mind import heldout_evaluation as heldout
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.heldout_seed import HELDOUT_SEED_VERSION
from ptcg_lab.learning_mind.macro import MacroCandidateV1, rollout_seed
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.storage import digest as observation_digest


FAMILIES = heldout.FAMILIES


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _selections(tmp_path: Path, *, overlap: bool = False) -> tuple[Path, Path, dict]:
    identity = {"identityHash": "feature-identity"}
    heldout_splits, training_splits, selected = [], [], {}
    for family_index, family in enumerate(FAMILIES):
        position_hash = f"heldout-position-{family_index}"
        source_game = f"heldout-game-{family_index}"
        position = {"positionHash": position_hash, "sourceGameId": source_game,
                    "targetDeck": "deck-a", "opponentArchetype": "archetype-a",
                    "positionStage": "late"}
        selected[family] = {position_hash: position}
        heldout_splits.append({"policyFamily": family, "split": "heldout", "sourceGames": 1,
            "sourceGameIds": [source_game], "positionHashes": [position_hash], "positions": [position]})
        for split in ("train", "development"):
            game = source_game if overlap and family_index == 0 and split == "train" else f"{family}-{split}-game"
            training_position_hash = f"{family}-{split}-position"
            row = {"positionHash": training_position_hash, "sourceGameId": game}
            training_splits.append({"policyFamily": family, "split": split, "sourceGames": 1,
                "positionHashes": [training_position_hash], "positions": [row]})
    heldout_selection = {"schemaVersion": 1, "kind": "macro-ranker-heldout-selection-v1",
        "trainingEligible": False, "interpretation": "heldout evaluator selection only; no labels",
        "minimumCompleteCandidatesPerPosition": 2, "identity": identity,
        "sourcePools": [], "splits": heldout_splits}
    heldout_selection["selectionHash"] = identity_hash(heldout_selection)
    training_selection = {"schemaVersion": 1, "selection": "train-dev-only", "identity": identity,
        "splits": training_splits}
    training_selection["selectionHash"] = identity_hash(training_selection)
    return (_write(tmp_path / "heldout-selection.json", heldout_selection),
            _write(tmp_path / "training-selection.json", training_selection), selected)


def test_heldout_selection_requires_non_training_selection(tmp_path):
    path, _, _ = _selections(tmp_path)
    selection, by_family = heldout._read_frozen_heldout_selection(path)
    assert selection["trainingEligible"] is False
    assert set(by_family) == set(FAMILIES)
    selection["trainingEligible"] = True
    _write(path, selection)
    with pytest.raises(ValueError, match="training-eligible"):
        heldout._read_frozen_heldout_selection(path)


def test_heldout_seed_namespace_is_deterministic_and_disjoint():
    seed = heldout._heldout_rollout_seed("position", 7, "rollout-identity")
    assert seed == heldout._heldout_rollout_seed("position", 7, "rollout-identity")
    assert seed != rollout_seed("training", "position", 7, "rollout-identity")
    assert seed != rollout_seed("development", "position", 7, "rollout-identity")
    assert seed != rollout_seed("promotion", "position", 7, "rollout-identity")


def _valid_heldout_run(tmp_path: Path, *, seed_namespace: str = "heldout",
                       illegal_root: bool = False):
    family = FAMILIES[0]
    observation = {"playerId": 0, "legalActions": [
        {"id": "pass-1", "type": "pass"}, {"id": "pass-2", "type": "pass"}]}
    position_hash = observation_digest(observation)
    source_position = {"positionHash": position_hash, "sourceGameId": "game-heldout",
        "targetDeck": "deck-a", "opponentArchetype": "archetype-a", "positionStage": "late"}
    selection = {"identity": {"identityHash": "feature"}, "selectionHash": "d" * 64}
    selected = {position_hash: source_position}
    generator = {"version": "generator-v1"}
    labels = []
    for index, (action_id, score) in enumerate((("pass-1", .75), ("pass-2", .25))):
        if illegal_root and index == 1:
            action_id = "not-legal"
        candidate = MacroCandidateV1(turn_intent="no-attack", action_ids=(action_id,),
            action_sequence=({"id": action_id, "type": "pass"},))
        uncertainty = math.sqrt(.25 / 64)
        labels.append({"candidate": asdict(candidate), "candidateHash": candidate.key(),
            "completedRollouts": 64, "attemptedRollouts": 64,
            "outcomes": {"finished": 64, "truncated": 0, "error": 0},
            "outcomeReasons": {}, "decisionCountDistribution": {"1": 64},
            "outcomesBySeedIndex": {str(seed_index): {"status": "finished", "score": score,
                "decisionCount": 1} for seed_index in range(64)},
            "expectedResult": score, "relativeResult": score - .75,
            "uncertainty": uncertainty, "weight": 64 / (1 + uncertainty)})
    rollout_identity = "frozen-rollout-identity"
    record = {**source_position, "positionHash": position_hash, "split": "heldout", "identity": selection["identity"],
        "selectionHash": selection["selectionHash"],
        "datasetManifestHash": "frozen-dataset", "opponentPolicyFamily": family,
        "observation": observation, "labels": labels, "candidateCount": len(labels),
        "status": "collected", "seedNamespace": seed_namespace,
        "rolloutIdentity": rollout_identity,
        "rolloutSeeds": [heldout._heldout_rollout_seed(position_hash, index, rollout_identity)
                         for index in range(64)], "highConfidencePolicyEligible": False}
    if seed_namespace != "heldout":
        record["rolloutSeeds"] = [rollout_seed(seed_namespace, position_hash, index, rollout_identity)
                                  for index in range(64)]
    labels_dir = tmp_path / "heldout-run"
    labels_dir.mkdir(parents=True, exist_ok=True)
    record_path = _write(labels_dir / f"{position_hash}.json", record)
    manifest = {"identity": selection["identity"], "datasetManifestHash": "frozen-dataset",
        "splitFilter": "heldout", "selectedPositionHashes": [position_hash], "positions": 1,
        "candidateGeneratorVersion": generator["version"],
        "candidateGeneratorIdentity": generator, "highConfidencePolicyLabels": 0,
        **heldout.FROZEN_HELDOUT_SETTINGS,
        "rolloutSeedVersion": HELDOUT_SEED_VERSION,
        "rolloutSeedImplementationSha256": file_sha256(
            Path(heldout.__file__).with_name("heldout_seed.py")),
        "labelCollectorVersion": "heldout-macro-rollout-labeler-v1",
        "labelCollectorSha256": file_sha256(
            Path(heldout.__file__).with_name("heldout_collection.py")),
        "rolloutIdentity": rollout_identity,
        "maximumRollouts": 64, "files": [{"path": record_path.name, "sha256": file_sha256(record_path)}]}
    manifest["manifestHash"] = identity_hash(manifest)
    manifest_path = _write(labels_dir / "manifest.json", manifest)
    support = {"candidateGeneratorIdentity": generator}
    return labels_dir, manifest, selection, selected, support


def test_heldout_run_loader_checks_actor_legal_root_and_seed_namespace(tmp_path):
    labels_dir, manifest, selection, selected, support = _valid_heldout_run(tmp_path)
    loaded_manifest, records = heldout._load_heldout_run(family=FAMILIES[0], labels_dir=labels_dir,
        selection=selection, selected_positions=selected,
        dataset_manifest={"manifestHash": "frozen-dataset"}, support=support,
        expected_generator=support["candidateGeneratorIdentity"])
    assert loaded_manifest["manifestHash"] == manifest["manifestHash"]
    assert len(records) == 1
    assert records[0]["positionHash"] in selected

    wrong_seed_dir, _, selection, selected, support = _valid_heldout_run(
        tmp_path / "wrong-seed", seed_namespace="development")
    with pytest.raises(ValueError, match="heldout label record provenance"):
        heldout._load_heldout_run(family=FAMILIES[0], labels_dir=wrong_seed_dir,
            selection=selection, selected_positions=selected,
            dataset_manifest={"manifestHash": "frozen-dataset"}, support=support,
            expected_generator=support["candidateGeneratorIdentity"])


def test_heldout_run_loader_rejects_generator_version_or_nonfrozen_settings(tmp_path):
    labels_dir, manifest, selection, selected, support = _valid_heldout_run(tmp_path)
    manifest["candidateGeneratorVersion"] = "different-generator-version"
    manifest.pop("manifestHash")
    manifest["manifestHash"] = identity_hash(manifest)
    _write(labels_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest does not match"):
        heldout._load_heldout_run(family=FAMILIES[0], labels_dir=labels_dir,
            selection=selection, selected_positions=selected,
            dataset_manifest={"manifestHash": "frozen-dataset"}, support=support,
            expected_generator=support["candidateGeneratorIdentity"])

    altered_dir, manifest, selection, selected, support = _valid_heldout_run(tmp_path / "altered")
    manifest["rolloutWorkers"] = 2
    manifest.pop("manifestHash")
    manifest["manifestHash"] = identity_hash(manifest)
    _write(altered_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest does not match"):
        heldout._load_heldout_run(family=FAMILIES[0], labels_dir=altered_dir,
            selection=selection, selected_positions=selected,
            dataset_manifest={"manifestHash": "frozen-dataset"}, support=support,
            expected_generator=support["candidateGeneratorIdentity"])


def test_heldout_run_loader_requires_per_seed_outcome_reconciliation(tmp_path):
    labels_dir, _, selection, selected, support = _valid_heldout_run(tmp_path)
    position_hash = next(iter(selected))
    record_path = labels_dir / f"{position_hash}.json"
    record = json.loads(record_path.read_text())
    record["labels"][0]["outcomesBySeedIndex"].pop("63")
    _write(record_path, record)
    manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["sha256"] = file_sha256(record_path)
    manifest.pop("manifestHash")
    manifest["manifestHash"] = identity_hash(manifest)
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="per-seed outcomes do not match"):
        heldout._load_heldout_run(family=FAMILIES[0], labels_dir=labels_dir,
            selection=selection, selected_positions=selected,
            dataset_manifest={"manifestHash": "frozen-dataset"}, support=support,
            expected_generator=support["candidateGeneratorIdentity"])

    illegal_dir, _, selection, selected, support = _valid_heldout_run(
        tmp_path / "illegal", illegal_root=True)
    with pytest.raises(ValueError, match="root is not actor-visible and legal"):
        heldout._load_heldout_run(family=FAMILIES[0], labels_dir=illegal_dir,
            selection=selection, selected_positions=selected,
            dataset_manifest={"manifestHash": "frozen-dataset"}, support=support,
            expected_generator=support["candidateGeneratorIdentity"])


def test_training_selection_rejects_heldout_game_or_split_leakage(tmp_path):
    heldout_path, training_path, by_family = _selections(tmp_path, overlap=True)
    heldout_selection = json.loads(heldout_path.read_text())
    training_selection = json.loads(training_path.read_text())
    with pytest.raises(ValueError, match="overlap"):
        heldout._validate_training_selection(training_selection,
            identity=heldout_selection["identity"], heldout_by_family=by_family)

    _, clean_path, clean_by_family = _selections(tmp_path / "clean")
    clean_selection = json.loads(clean_path.read_text())
    clean_selection["splits"][1]["positions"][0]["sourceGameId"] = "python-heuristic-train-game"
    clean_selection["selectionHash"] = identity_hash({k: v for k, v in clean_selection.items()
        if k != "selectionHash"})
    with pytest.raises(ValueError, match="train/development"):
        heldout._validate_training_selection(clean_selection, identity=heldout_selection["identity"],
            heldout_by_family=clean_by_family)


def test_heldout_summary_weights_independent_source_games_equally():
    details = []
    for index, regret in enumerate((0.0, 0.0, 0.0)):
        details.append({"sourceGameId": "game-many-positions", "opponentPolicyFamily": FAMILIES[0],
            "positionHash": f"position-{index}", "top1RelativeRegret": regret,
            "top3Recall": True, "pairwiseComparisons": 1, "pairwiseCorrect": 1})
    details.append({"sourceGameId": "game-one-position", "opponentPolicyFamily": FAMILIES[0],
        "positionHash": "position-3", "top1RelativeRegret": 1.0,
        "top3Recall": False, "pairwiseComparisons": 1, "pairwiseCorrect": 0})

    summary = heldout._summarize(details, seed_material="uneven-game-clusters")

    assert summary["meanTop1RelativeRegret"] == pytest.approx(.5)
    assert summary["meanTop1RelativeRegretWeighting"] == "equal-source-game"
    assert summary["independentSourceGames"] == 2
    assert summary["positionWeightedMeanTop1RelativeRegret"] == pytest.approx(.25)


def test_heldout_ranker_evaluation_is_descriptive_and_immutable(tmp_path, monkeypatch):
    heldout_path, training_path, selected = _selections(tmp_path)
    heldout_selection = json.loads(heldout_path.read_text())
    ranker_selection = json.loads(training_path.read_text())
    model_path = _write(tmp_path / "ranker.json", {"model": "portable"})
    report_path = _write(tmp_path / "ranker-report.json", {"report": "frozen"})
    label_runs = {}
    run_manifests = {}
    for family in FAMILIES:
        run_dir = tmp_path / family / "labels"
        run_dir.mkdir(parents=True)
        manifest_path = _write(run_dir / "manifest.json", {"immutable": family})
        label_runs[family] = run_dir
        settings = dict(heldout.FROZEN_HELDOUT_SETTINGS)
        run_manifests[family] = {**settings,
            "manifestHash": ("d" if family == FAMILIES[0] else "e") * 64,
            "datasetManifestHash": ("a" if family == FAMILIES[0] else "b") * 64,
            "rolloutIdentity": ("f" if family == FAMILIES[0] else "9") * 64,
            "labelCollectorVersion": "macro-rollout-labeler-v6",
            "labelCollectorSha256": "a" * 64, "manifestPath": manifest_path}

    ranker_report = {"identity": heldout_selection["identity"],
        "selectionHash": ranker_selection["selectionHash"],
        "selectionManifestSha256": heldout.file_sha256(training_path),
        "modelSha256": "b" * 64, "reportHash": "c" * 64}
    monkeypatch.setattr(heldout, "verify_macro_ranker_v2_artifact",
        lambda *_: {"report": ranker_report, "artifact": {"portable": True}})
    generator = {"version": "candidate-generator-v1"}
    monkeypatch.setattr(heldout, "_verify_source_pool",
        lambda *, family, **_: ({"manifestHash": f"{family}-dataset"},
            {"candidateGeneratorIdentity": generator}))

    def load_run(*, family, **kwargs):
        position_hash, position = next(iter(selected[family].items()))
        labels = [
            {"candidate": {"plan": 1}, "candidateHash": f"{family}-low",
             "expectedResult": .25, "outcomes": {"finished": 20, "truncated": 0, "error": 0}},
            {"candidate": {"plan": 2}, "candidateHash": f"{family}-high",
             "expectedResult": .75, "outcomes": {"finished": 20, "truncated": 0, "error": 0}},
        ]
        record = {**position, "positionHash": position_hash, "labels": labels,
                  "observation": {"actorOnly": True}}
        return run_manifests[family], [record]

    monkeypatch.setattr(heldout, "_load_heldout_run", load_run)
    monkeypatch.setattr(heldout, "candidate_features_v2", lambda *_: np.zeros(3))
    monkeypatch.setattr(heldout, "predict_macro_ranker_v2", lambda *_: np.asarray([2.0, 1.0]))

    output_path = tmp_path / "heldout-report.json"
    report = heldout.evaluate_macro_ranker_v2_heldout(
        selection_path=heldout_path, ranker_selection_path=training_path,
        model_path=model_path, ranker_report_path=report_path,
        datasets={family: tmp_path / f"{family}-dataset" for family in FAMILIES},
        support_reports={family: tmp_path / f"{family}-support.json" for family in FAMILIES},
        label_runs=label_runs, output=output_path)
    assert report["status"] == "measured"
    assert report["positions"] == 2
    assert report["independentSourceGames"] == 2
    assert report["metrics"]["overall"]["meanTop1RelativeRegret"] == pytest.approx(.5)
    assert report["evaluatorImplementationSha256"] == file_sha256(Path(heldout.__file__))
    assert report["trainingEligible"] is False
    assert report["automaticPromotion"] is False
    assert all(item["engineErrors"] == item["truncatedRollouts"] == 0 for item in report["details"])
    verification = heldout.verify_macro_ranker_v2_heldout_report(
        report_path=output_path, selection_path=heldout_path,
        ranker_selection_path=training_path, model_path=model_path,
        ranker_report_path=report_path, datasets={
            family: tmp_path / f"{family}-dataset" for family in FAMILIES},
        support_reports={family: tmp_path / f"{family}-support.json" for family in FAMILIES},
        label_runs=label_runs)
    assert verification["verified"] is True
    assert verification["reportHash"] == report["reportHash"]

    tampered = json.loads(output_path.read_text())
    tampered["metrics"]["overall"]["meanTop1RelativeRegret"] = .1
    tampered.pop("reportHash")
    tampered["reportHash"] = identity_hash(tampered)
    _write(output_path, tampered)
    with pytest.raises(ValueError, match="aggregate metrics do not recompute"):
        heldout.verify_macro_ranker_v2_heldout_report(
            report_path=output_path, selection_path=heldout_path,
            ranker_selection_path=training_path, model_path=model_path,
            ranker_report_path=report_path, datasets={
                family: tmp_path / f"{family}-dataset" for family in FAMILIES},
            support_reports={family: tmp_path / f"{family}-support.json" for family in FAMILIES},
            label_runs=label_runs)
    with pytest.raises(ValueError, match="immutable"):
        heldout.evaluate_macro_ranker_v2_heldout(
            selection_path=heldout_path, ranker_selection_path=training_path,
            model_path=model_path, ranker_report_path=report_path,
            datasets={family: tmp_path / f"{family}-dataset" for family in FAMILIES},
            support_reports={family: tmp_path / f"{family}-support.json" for family in FAMILIES},
            label_runs=label_runs, output=tmp_path / "heldout-report.json")
