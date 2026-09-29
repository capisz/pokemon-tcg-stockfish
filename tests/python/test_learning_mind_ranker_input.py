import json

import pytest

from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.experiment import _load_ranker_input, fit_ranker
from ptcg_lab.learning_mind.schema import identity_hash

FAMILIES = ["python-heuristic", "typescript-heuristic"]
SPLITS = ["train", "development"]


def _write_combined_input(root, *, omit=None):
    root.mkdir()
    identity = {"identityHash": "frozen-identity"}
    position_map = {family: {split: f"{family}-{split}" for split in SPLITS} for family in FAMILIES}
    selection = {"schemaVersion": 1, "identity": identity, "splits": [
        {"policyFamily": family, "split": split, "sourceGames": 1,
         "positionHashes": [position_map[family][split]]} for family in FAMILIES for split in SPLITS]}
    selection["selectionHash"] = identity_hash(selection)
    selection_path = root.parent / "selection.json"
    selection_path.write_text(json.dumps(selection))
    files = []
    for family in FAMILIES:
        for split in SPLITS:
            if (family, split) == omit:
                continue
            position_hash = position_map[family][split]
            relative = f"{family}/{position_hash}.json"
            target = root / relative
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps({"positionHash": position_hash, "identity": identity,
                "opponentPolicyFamily": family, "split": split, "status": "collected", "labels": []}))
            files.append({"path": relative, "sha256": file_sha256(target)})
    manifest = {"kind": "combined-macro-label-runs-v1", "identity": identity,
        "selectionHash": selection["selectionHash"], "selectionManifestSha256": file_sha256(selection_path),
        "selectedPositionHashes": sorted(position for family in FAMILIES for split in SPLITS
                                          for position in selection["splits"][FAMILIES.index(family)*2+SPLITS.index(split)]["positionHashes"]),
        "positionsByFamilySplit": {family: {split: 1 for split in SPLITS} for family in FAMILIES},
        "policyFamilies": FAMILIES, "positions": len(files),
        "positionsBySplit": {"train": sum(1 for item in files if "/" in item["path"] and "-train" in item["path"]),
                             "development": sum(1 for item in files if "-development" in item["path"])},
        "files": files}
    manifest["manifestHash"] = identity_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, selection_path, files


def test_ranker_input_rejects_incomplete_frozen_position_universe(tmp_path):
    root, selection, _ = _write_combined_input(tmp_path / "combined", omit=(FAMILIES[0], "train"))
    with pytest.raises(ValueError, match="exactly cover frozen"):
        _load_ranker_input(root, selection)


def test_ranker_input_rejects_unlisted_json_record(tmp_path):
    root, selection, _ = _write_combined_input(tmp_path / "combined")
    (root / "python-heuristic" / "unlisted.json").write_text("{}")
    with pytest.raises(ValueError, match="unlisted JSON records"):
        _load_ranker_input(root, selection)


def test_ranker_input_rejects_manifest_record_symlink(tmp_path):
    root, selection, files = _write_combined_input(tmp_path / "combined")
    item = files[0]
    target = root / item["path"]
    real = root.parent / "record-source.json"
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _load_ranker_input(root, selection)


@pytest.mark.parametrize("existing_output", ["model", "manifest"])
def test_ranker_fit_never_overwrites_existing_artifacts(tmp_path, existing_output):
    output = tmp_path / "ranker.json"
    existing_path = output if existing_output == "model" else output.with_suffix(".manifest.json")
    existing_path.write_text("previous immutable artifact")
    with pytest.raises(ValueError, match="outputs are immutable"):
        fit_ranker(tmp_path / "labels-not-read", output, selection_path=tmp_path / "selection-not-read.json",
                   teacher_hash="teacher", opponent_policy_hash="opponent")
    assert existing_path.read_text() == "previous immutable artifact"
