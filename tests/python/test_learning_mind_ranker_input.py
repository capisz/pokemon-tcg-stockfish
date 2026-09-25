import json

import pytest

from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.experiment import _load_ranker_input
from ptcg_lab.learning_mind.schema import identity_hash


FAMILIES = ["python-heuristic", "typescript-heuristic"]


def _write_combined_input(root, *, python_split="train", typescript_split="development"):
    root.mkdir()
    identity = {"identityHash": "frozen-identity"}
    files = []
    for family, split in zip(FAMILIES, (python_split, typescript_split)):
        position_hash = f"{family}-position"
        relative = f"{family}/{position_hash}.json"
        target = root / relative
        target.parent.mkdir()
        target.write_text(json.dumps({
            "positionHash": position_hash,
            "identity": identity,
            "opponentPolicyFamily": family,
            "split": split,
            "status": "collected",
            "labels": [],
        }))
        files.append({"path": relative, "sha256": file_sha256(target)})
    manifest = {
        "kind": "combined-macro-label-runs-v1",
        "identity": identity,
        "policyFamilies": FAMILIES,
        "positions": len(files),
        "positionsBySplit": {python_split: 1, typescript_split: 1},
        "files": files,
    }
    manifest["manifestHash"] = identity_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, files


def test_ranker_input_verifies_and_loads_both_families(tmp_path):
    root, _files = _write_combined_input(tmp_path / "combined")

    manifest, records = _load_ranker_input(root)

    assert manifest["positions"] == 2
    assert {record["opponentPolicyFamily"] for record in records} == set(FAMILIES)


def test_ranker_input_rejects_heldout_and_unknown_splits(tmp_path):
    for split in ("heldout", "unknown"):
        root, _files = _write_combined_input(tmp_path / split, python_split=split)

        with pytest.raises(ValueError, match="only train/development"):
            _load_ranker_input(root)


def test_ranker_input_rejects_record_corruption(tmp_path):
    root, files = _write_combined_input(tmp_path / "combined")
    (root / files[0]["path"]).write_text("corrupt after manifest freeze")

    with pytest.raises(ValueError, match="record hash mismatch"):
        _load_ranker_input(root)


def test_ranker_input_rejects_manifest_tampering(tmp_path):
    root, _files = _write_combined_input(tmp_path / "combined")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["positionsBySplit"] = {"train": 2}
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        _load_ranker_input(root)
