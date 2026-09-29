import json

import pytest

from ptcg_lab.learning_mind import __main__
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.progress import macro_label_progress
from ptcg_lab.learning_mind.schema import identity_hash


FAMILIES = ("python-heuristic", "typescript-heuristic")
SPLITS = ("train", "development")


def _write_inputs(root):
    identity = {"identityHash": "frozen-identity"}
    selection = {
        "schemaVersion": 1,
        "identity": identity,
        "sourcePools": [{"policyFamily": family, "datasetManifestHash": f"dataset-{family}"}
                        for family in FAMILIES],
        "splits": [{"policyFamily": family, "split": split, "sourceGames": 1,
                    "positionHashes": [f"{family}-{split}"]}
                   for family in FAMILIES for split in SPLITS],
    }
    selection["selectionHash"] = identity_hash(selection)
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection))

    runs = {family: root / family for family in FAMILIES}
    for family, run_dir in runs.items():
        run_dir.mkdir()
        completed_splits = SPLITS if family == "python-heuristic" else ("train",)
        files = []
        for split in completed_splits:
            record = {
                "positionHash": f"{family}-{split}",
                "split": split,
                "identity": identity,
                "datasetManifestHash": f"dataset-{family}",
                "opponentPolicyFamily": family,
                "status": "collected",
                "highConfidencePolicyEligible": False,
            }
            record_path = run_dir / f"{record['positionHash']}.json"
            record_path.write_text(json.dumps(record))
            files.append({"path": record_path.name, "sha256": file_sha256(record_path)})
        if family == "python-heuristic":
            manifest = {
                "identity": identity,
                "datasetManifestHash": f"dataset-{family}",
                "positions": len(files),
                "selectedPositionHashes": [f"{family}-{split}" for split in SPLITS],
                "files": files,
            }
            manifest["manifestHash"] = identity_hash(manifest)
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

    frozen = {
        "positionHash": "typescript-heuristic-development",
        "identity": identity,
        "datasetManifestHash": "dataset-typescript-heuristic",
        "candidateHashes": ["candidate-a", "candidate-b"],
        "initialRollouts": 4,
        "maximumRollouts": 8,
    }
    state = {
        "allocation": {"initial": 4, "maximum": 8, "closeMargin": 0.1,
                       "extensionBatchSize": 2},
        "candidateHashes": frozen["candidateHashes"],
        "records": {
            "candidate-a": {"finished": 3, "truncated": 0, "error": 0},
            "candidate-b": {"finished": 2, "truncated": 0, "error": 0},
        },
        "completedInitialIndices": [0, 1],
        "completedExtensionIndices": [4],
        "closeCandidateHashes": ["candidate-a"],
    }
    checkpoint = {
        "schemaVersion": 1,
        "identityHash": identity_hash(frozen),
        "identity": frozen,
        "state": state,
    }
    checkpoint["progressHash"] = identity_hash(checkpoint)
    checkpoint_path = runs["typescript-heuristic"] / ".typescript-heuristic-development.progress"
    checkpoint_path.write_text(json.dumps(checkpoint))
    return selection_path, runs, checkpoint_path, checkpoint


def test_macro_label_progress_reports_frozen_coverage_and_checkpoint_without_writes(tmp_path):
    selection_path, runs, _, _ = _write_inputs(tmp_path)
    before = {path.relative_to(tmp_path): file_sha256(path)
              for path in tmp_path.rglob("*") if path.is_file()}

    report = macro_label_progress(selection_path=selection_path, runs=runs)
    after = {path.relative_to(tmp_path): file_sha256(path)
             for path in tmp_path.rglob("*") if path.is_file()}

    assert report["positionCoverage"] == {
        "selected": 4,
        "completedResultFiles": 3,
        "percent": 75.0,
        "positionsInProgress": 1,
        "policyEligiblePositions": 0,
    }
    checkpoint = report["families"]["typescript-heuristic"]["checkpoints"][0]
    assert checkpoint["completedInitialSeeds"] == 2
    assert checkpoint["initialSeedProgressPercent"] == 50.0
    assert checkpoint["completedExtensionSeeds"] == 1
    assert checkpoint["highestExtensionIndex"] == 4
    assert checkpoint["activeCandidateCount"] == 1
    assert checkpoint["sampleCountRangePerCandidate"] == [2, 3]
    assert checkpoint["outcomes"] == {"finished": 5, "truncated": 0, "error": 0}
    assert report["processStatus"].startswith("unknown")
    assert report["writesArtifacts"] is False
    assert report["ppoEnabled"] is False
    assert report["continuousOperationEnabled"] is False
    assert report["trustedPromotion"] is False
    assert after == before


def test_progress_cli_accepts_both_frozen_runs_and_prints_report(tmp_path, capsys):
    selection_path, runs, _, _ = _write_inputs(tmp_path)

    assert __main__.main([
        "progress-macro-labels",
        "--selection", str(selection_path),
        "--python-run", str(runs["python-heuristic"]),
        "--typescript-run", str(runs["typescript-heuristic"]),
    ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["positionCoverage"]["percent"] == 75.0
    assert report["processStatus"].startswith("unknown")


def test_progress_rejects_symlink_checkpoint(tmp_path):
    selection_path, runs, checkpoint_path, _ = _write_inputs(tmp_path)
    target = tmp_path / "checkpoint-target.json"
    target.write_bytes(checkpoint_path.read_bytes())
    checkpoint_path.unlink()
    checkpoint_path.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        macro_label_progress(selection_path=selection_path, runs=runs)


def test_progress_rejects_extension_indices_outside_frozen_rollout_cap(tmp_path):
    selection_path, runs, checkpoint_path, checkpoint = _write_inputs(tmp_path)
    checkpoint["state"]["completedExtensionIndices"] = [8]
    checkpoint["progressHash"] = identity_hash({key: value for key, value in checkpoint.items()
                                                 if key != "progressHash"})
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="seed progress is malformed"):
        macro_label_progress(selection_path=selection_path, runs=runs)
