import json

import pytest

from ptcg_lab.learning_mind.aggregation import combine_macro_label_runs
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.schema import identity_hash


def _write_run(root, family, position_hash, split, identity):
    root.mkdir()
    dataset_hash = f"dataset-{family}"
    rollout_hash = f"rollout-{family}"
    record = {
        "positionHash": position_hash,
        "split": split,
        "identity": identity,
        "datasetManifestHash": dataset_hash,
        "rolloutIdentity": rollout_hash,
        "opponentPolicyFamily": family,
        "status": "collected",
        "highConfidencePolicyEligible": False,
        "labels": [{"candidateHash": f"{position_hash}-a"}, {"candidateHash": f"{position_hash}-b"}],
    }
    record_path = root / f"{position_hash}.json"
    record_path.write_text(json.dumps(record))
    manifest = {
        "identity": identity,
        "datasetManifestHash": dataset_hash,
        "rolloutIdentity": rollout_hash,
        "candidateGeneratorIdentity": {"version": "fixture-v1"},
        "labelCollectorVersion": "fixture-collector-v1",
        "labelCollectorSha256": "collector-hash",
        "positions": 1,
        "selectedPositionHashes": [position_hash],
        "files": [{"path": record_path.name, "sha256": file_sha256(record_path)}],
    }
    manifest["manifestHash"] = identity_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return record_path


def test_combiner_verifies_and_merges_distinct_families(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = tmp_path / "python-run"
    typescript_dir = tmp_path / "typescript-run"
    _write_run(python_dir, "python-heuristic", "python-position", "train", identity)
    _write_run(typescript_dir, "typescript-heuristic", "typescript-position", "development", identity)

    output = tmp_path / "combined"
    result = combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=output, identity=identity)

    assert result["positions"] == 2
    assert result["positionsBySplit"] == {"development": 1, "train": 1}
    assert result["policyFamilies"] == ["python-heuristic", "typescript-heuristic"]
    assert all(file_sha256(output / item["path"]) == item["sha256"] for item in result["files"])
    recorded = result.pop("manifestHash")
    assert identity_hash(result) == recorded


def test_combiner_rejects_corrupt_inputs_without_publishing_partial_output(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = tmp_path / "python-run"
    typescript_dir = tmp_path / "typescript-run"
    record_path = _write_run(python_dir, "python-heuristic", "python-position", "train", identity)
    _write_run(typescript_dir, "typescript-heuristic", "typescript-position", "development", identity)
    record_path.write_text("corrupt after manifest freeze")
    output = tmp_path / "combined"

    with pytest.raises(ValueError, match="record hash mismatch"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=output, identity=identity)
    assert not output.exists()


def test_combiner_rejects_a_run_from_a_different_frozen_identity(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = tmp_path / "python-run"
    typescript_dir = tmp_path / "typescript-run"
    _write_run(python_dir, "python-heuristic", "python-position", "train", identity)
    _write_run(typescript_dir, "typescript-heuristic", "typescript-position", "development",
               {"identityHash": "different-identity"})

    with pytest.raises(ValueError, match="experiment identity mismatch"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=tmp_path / "combined",
                                 identity=identity)


@pytest.mark.parametrize("split", ["heldout", "unknown"])
def test_combiner_rejects_heldout_or_unknown_splits(tmp_path, split):
    identity = {"identityHash": "frozen-identity"}
    python_dir = tmp_path / "python-run"
    typescript_dir = tmp_path / "typescript-run"
    _write_run(python_dir, "python-heuristic", "python-position", split, identity)
    _write_run(typescript_dir, "typescript-heuristic", "typescript-position", "development", identity)
    output = tmp_path / "combined"

    with pytest.raises(ValueError, match="only train/development"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=output, identity=identity)
    assert not output.exists()
