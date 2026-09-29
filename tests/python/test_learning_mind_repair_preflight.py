import json

import pytest

from ptcg_lab.learning_mind import repair_preflight
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.schema import identity_hash


def _fixture(tmp_path, monkeypatch):
    family = "python-heuristic"
    identity = {"identityHash": "fixture-identity"}
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}\n")
    (dataset / "rows.jsonl").write_text("fixture\n")
    support_path = tmp_path / "support.json"
    position_hash = "a" * 64
    generator_identity = {"version": "fixture-generator"}
    support = {"identity": identity, "datasetManifestHash": "dataset-hash",
        "datasetRowsSha256": file_sha256(dataset / "rows.jsonl"),
        "candidateGeneratorIdentity": generator_identity,
        "positions": [{"positionHash": position_hash, "status": "supported",
                       "completeCandidateCount": 3}]}
    support["reportHash"] = identity_hash(support)
    support_path.write_text(json.dumps(support))
    parent = {"schemaVersion": 1, "identity": identity, "selectionHash": "parent-hash",
        "splits": [
            {"policyFamily": family, "split": "train", "positionHashes": ["b" * 64],
             "positions": [{"positionHash": "b" * 64, "sourceGameId": "game-1"}]},
            {"policyFamily": family, "split": "development", "positionHashes": ["c" * 64],
             "positions": [{"positionHash": "c" * 64, "sourceGameId": "dev-game"}]},
            {"policyFamily": "typescript-heuristic", "split": "train", "positionHashes": ["d" * 64],
             "positions": [{"positionHash": "d" * 64, "sourceGameId": "ts-game"}]},
            {"policyFamily": "typescript-heuristic", "split": "development", "positionHashes": ["e" * 64],
             "positions": [{"positionHash": "e" * 64, "sourceGameId": "ts-dev-game"}]},
        ]}
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    monkeypatch.setattr(repair_preflight, "load_frozen_selection", lambda *_args, **_kwargs: (
        parent, {family: {"train": {"b" * 64}, "development": {"c" * 64}},
                 "typescript-heuristic": {"train": {"d" * 64}, "development": {"e" * 64}}}))
    selection = {"schemaVersion": 1,
        "selection": "candidate-supported-source-game-preserving-train-repair-v1",
        "status": "draft-unlabeled-lineage-merge-required", "identity": identity,
        "sourcePools": [{"policyFamily": family, "datasetManifestHash": "dataset-hash",
            "datasetManifestSha256": file_sha256(dataset / "manifest.json"),
            "datasetRowsSha256": file_sha256(dataset / "rows.jsonl"),
            "supportReportHash": support["reportHash"],
            "supportReportSha256": file_sha256(support_path)}],
        "splits": [
            {"policyFamily": family, "split": "train", "sourceGames": 1,
             "positionHashes": [position_hash],
             "positions": [{"positionHash": position_hash, "sourceGameId": "game-1"}]},
            {"policyFamily": family, "split": "development", "sourceGames": 1,
             "positionHashes": ["c" * 64],
             "positions": [{"positionHash": "c" * 64, "sourceGameId": "dev-game"}]},
            {"policyFamily": "typescript-heuristic", "split": "train", "sourceGames": 1,
             "positionHashes": ["f" * 64],
             "positions": [{"positionHash": "f" * 64, "sourceGameId": "ts-game"}]},
            {"policyFamily": "typescript-heuristic", "split": "development", "sourceGames": 1,
             "positionHashes": ["e" * 64],
             "positions": [{"positionHash": "e" * 64, "sourceGameId": "ts-dev-game"}]},
        ],
        "lineage": {"kind": "same-source-game-train-root-repair-v1",
            "parentSelectionHash": parent["selectionHash"],
            "parentSelectionManifestSha256": file_sha256(parent_path),
            "developmentPositionHashesUnchanged": True,
            "heldoutPositionsSelected": False, "rawParentLabelsMustRemainImmutable": True,
            "replacements": [
                {"policyFamily": family, "split": "train", "sourceGameId": "game-1",
                 "supersededPositionHash": "b" * 64, "replacementPositionHash": position_hash,
                 "replacementCompleteCandidateCount": 3},
                {"policyFamily": "typescript-heuristic", "split": "train", "sourceGameId": "ts-game",
                 "supersededPositionHash": "d" * 64, "replacementPositionHash": "f" * 64,
                 "replacementCompleteCandidateCount": 4}]}}
    selection["selectionHash"] = identity_hash(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(repair_preflight, "load_dataset", lambda *_args, **_kwargs: (
        {"manifestHash": "dataset-hash"}, [{"positionHash": position_hash,
            "split": "train", "sourceGameId": "game-1"}]))
    monkeypatch.setattr(repair_preflight, "transition_generator_identity",
                        lambda _root: generator_identity)
    from ptcg_lab.learning_mind import experiment
    monkeypatch.setattr(experiment, "runtime_identity", lambda _root: type(
        "RuntimeIdentity", (), {"record": lambda self: identity})())
    return family, dataset, support_path, selection_path, parent_path, position_hash


def test_training_repair_preflight_reports_exact_supported_work_without_writes(tmp_path, monkeypatch):
    family, dataset, support, selection, parent, position_hash = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "not-created"
    report = repair_preflight.preflight_training_repair_collection(root=tmp_path,
        family=family, dataset_dir=dataset, support_path=support,
        selection_path=selection, parent_selection_path=parent, output=output)
    assert report["selectionHash"] == json.loads(selection.read_text())["selectionHash"]
    assert report["settings"]["selectedPositionHashes"] == [position_hash]
    assert report["settings"]["selectionMethod"] == "position-hash-list"
    assert report["settings"]["splitFilter"] is None
    assert report["candidatePlans"] == 3
    assert report["initialCandidateSeedRollouts"] == 48
    assert report["maximumCandidateSeedRollouts"] == 192
    assert report["writesArtifacts"] is False
    assert report["startsEngine"] is False
    assert not output.exists()


def test_training_repair_preflight_rejects_modified_selection_or_nonempty_output(tmp_path, monkeypatch):
    family, dataset, support, selection, parent, _ = _fixture(tmp_path, monkeypatch)
    modified = json.loads(selection.read_text())
    modified["lineage"]["replacements"][0]["sourceGameId"] = "other-game"
    selection.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="valid frozen draft"):
        repair_preflight.preflight_training_repair_collection(root=tmp_path,
            family=family, dataset_dir=dataset, support_path=support,
            selection_path=selection, parent_selection_path=parent, output=tmp_path / "fresh")

    second_root = tmp_path / "second"
    second_root.mkdir()
    _, dataset, support, selection, parent, _ = _fixture(second_root, monkeypatch)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty output directory"):
        repair_preflight.preflight_training_repair_collection(root=second_root,
            family=family, dataset_dir=dataset, support_path=support,
            selection_path=selection, parent_selection_path=parent, output=occupied)


def test_training_repair_preflight_rejects_rehashed_wrong_source_game_lineage(tmp_path, monkeypatch):
    family, dataset, support, selection, parent, _ = _fixture(tmp_path, monkeypatch)
    modified = json.loads(selection.read_text())
    modified["lineage"]["replacements"][0]["sourceGameId"] = "different-game"
    modified["selectionHash"] = identity_hash({
        key: value for key, value in modified.items() if key != "selectionHash"})
    selection.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="preserve its parent source game"):
        repair_preflight.preflight_training_repair_collection(root=tmp_path,
            family=family, dataset_dir=dataset, support_path=support,
            selection_path=selection, parent_selection_path=parent, output=tmp_path / "fresh")


def test_training_repair_preflight_rejects_symlinked_evidence(tmp_path, monkeypatch):
    family, dataset, support, selection, parent, _ = _fixture(tmp_path, monkeypatch)
    selection_link = tmp_path / "selection-link.json"
    selection_link.symlink_to(selection)
    with pytest.raises(ValueError, match="regular non-symlink files"):
        repair_preflight.preflight_training_repair_collection(root=tmp_path,
            family=family, dataset_dir=dataset, support_path=support,
            selection_path=selection_link, parent_selection_path=parent,
            output=tmp_path / "fresh")
