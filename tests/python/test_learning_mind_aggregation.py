import json

import pytest

from ptcg_lab.learning_mind.aggregation import combine_macro_label_runs
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.schema import identity_hash

FAMILIES = ["python-heuristic", "typescript-heuristic"]
SPLITS = ["train", "development"]


def _write_selection(path, identity, positions=None):
    positions = positions or {family: {split: [f"{family}-{split}"] for split in SPLITS} for family in FAMILIES}
    value = {"schemaVersion": 1, "identity": identity, "splits": [
        {"policyFamily": family, "split": split, "sourceGames": len(positions[family][split]),
         "positionHashes": positions[family][split]} for family in FAMILIES for split in SPLITS]}
    value["selectionHash"] = identity_hash(value)
    path.write_text(json.dumps(value))
    return path


def _write_run(root, family, identity, *, splits=SPLITS):
    root.mkdir()
    files, hashes = [], []
    for split in splits:
        position_hash = f"{family}-{split}"
        record = {"positionHash": position_hash, "split": split, "identity": identity,
            "datasetManifestHash": f"dataset-{family}", "rolloutIdentity": f"rollout-{family}",
            "opponentPolicyFamily": family, "status": "collected", "highConfidencePolicyEligible": False,
            "labels": [{"candidateHash": f"{position_hash}-a"}, {"candidateHash": f"{position_hash}-b"}]}
        record_path = root / f"{position_hash}.json"
        record_path.write_text(json.dumps(record))
        files.append({"path": record_path.name, "sha256": file_sha256(record_path)})
        hashes.append(position_hash)
    manifest = {"identity": identity, "datasetManifestHash": f"dataset-{family}",
        "rolloutIdentity": f"rollout-{family}", "candidateGeneratorIdentity": {"version": "fixture-v1"},
        "labelCollectorVersion": "fixture-collector-v1", "labelCollectorSha256": "collector-hash",
        "positions": len(files), "selectedPositionHashes": hashes, "files": files}
    manifest["manifestHash"] = identity_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_combiner_verifies_and_merges_frozen_positions(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    inputs = [_write_run(tmp_path / family, family, identity) for family in FAMILIES]
    selection = _write_selection(tmp_path / "selection.json", identity)
    output = tmp_path / "combined"
    result = combine_macro_label_runs(inputs=inputs, output=output, identity=identity, selection_path=selection)
    assert result["positions"] == 4
    assert result["positionsBySplit"] == {"development": 2, "train": 2}
    assert result["policyFamilies"] == FAMILIES
    assert all(file_sha256(output / item["path"]) == item["sha256"] for item in result["files"])
    recorded = result.pop("manifestHash")
    assert identity_hash(result) == recorded


def test_combiner_rejects_incomplete_frozen_position_set(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = _write_run(tmp_path / "python", FAMILIES[0], identity, splits=["train"])
    typescript_dir = _write_run(tmp_path / "typescript", FAMILIES[1], identity)
    selection = _write_selection(tmp_path / "selection.json", identity)
    with pytest.raises(ValueError, match="does not exactly cover frozen"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=tmp_path / "combined",
                                 identity=identity, selection_path=selection)
    assert not (tmp_path / "combined").exists()


def test_combiner_rejects_corrupt_input_without_partial_output(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = _write_run(tmp_path / "python", FAMILIES[0], identity)
    typescript_dir = _write_run(tmp_path / "typescript", FAMILIES[1], identity)
    (python_dir / f"{FAMILIES[0]}-train.json").write_text("corrupt")
    with pytest.raises(ValueError, match="record hash mismatch"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=tmp_path / "combined",
            identity=identity, selection_path=_write_selection(tmp_path / "selection.json", identity))
    assert not (tmp_path / "combined").exists()


def test_combiner_rejects_unlisted_json_record_without_partial_output(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = _write_run(tmp_path / "python", FAMILIES[0], identity)
    typescript_dir = _write_run(tmp_path / "typescript", FAMILIES[1], identity)
    (python_dir / "unlisted.json").write_text("{}")
    selection = _write_selection(tmp_path / "selection.json", identity)
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="unlisted JSON records"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=output,
            identity=identity, selection_path=selection)
    assert not output.exists()


def test_combiner_rejects_symlinked_manifest_record_without_partial_output(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = _write_run(tmp_path / "python", FAMILIES[0], identity)
    typescript_dir = _write_run(tmp_path / "typescript", FAMILIES[1], identity)
    item = json.loads((python_dir / "manifest.json").read_text())["files"][0]
    target = python_dir / item["path"]
    real = tmp_path / "external-record.json"
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)
    selection = _write_selection(tmp_path / "selection.json", identity)
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="must not be a symlink"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=output,
            identity=identity, selection_path=selection)
    assert not output.exists()


def test_combiner_rejects_wrong_frozen_identity(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = _write_run(tmp_path / "python", FAMILIES[0], identity)
    typescript_dir = _write_run(tmp_path / "typescript", FAMILIES[1], {"identityHash": "different"})
    with pytest.raises(ValueError, match="experiment identity mismatch"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=tmp_path / "combined",
            identity=identity, selection_path=_write_selection(tmp_path / "selection.json", identity))


def test_finalizer_combines_and_recomputes_confidence_without_training(tmp_path, monkeypatch):
    from ptcg_lab.learning_mind import aggregation, confidence_audit

    calls = []
    combined_output = tmp_path / "combined"
    confidence_output = tmp_path / "confidence.json"
    monkeypatch.setattr(aggregation, "combine_macro_label_runs", lambda **kwargs: (
        calls.append(("combine", kwargs)) or {"manifestHash": "combined-hash", "positions": 4}))
    report = {"reportHash": "report-hash", "status": "analysis-only"}
    monkeypatch.setattr(confidence_audit, "audit_macro_label_confidence", lambda **kwargs: (
        calls.append(("audit", kwargs)) or report))
    monkeypatch.setattr(confidence_audit, "verify_macro_label_confidence_audit", lambda **kwargs: (
        calls.append(("verify", kwargs)) or report))

    result = aggregation.finalize_macro_label_runs(inputs=[tmp_path / "py", tmp_path / "ts"],
        combined_output=combined_output, confidence_output=confidence_output,
        identity={"identityHash": "identity"}, selection_path=tmp_path / "selection.json")
    assert [call[0] for call in calls] == ["combine", "audit", "verify"]
    assert calls[1][1]["labels_dir"] == combined_output.resolve()
    assert calls[2][1]["report_path"] == confidence_output.resolve()
    assert result == {"status": "verified-analysis-only", "combinedManifestHash": "combined-hash",
        "combinedPositions": 4, "confidenceReportHash": "report-hash",
        "confidenceReportPath": str(confidence_output.resolve()),
        "policyLabelEligibilityChanged": False, "ppoEnablement": False,
        "promotionAuthority": "none"}


def test_finalizer_rejects_confidence_report_inside_combined_labels(tmp_path, monkeypatch):
    from ptcg_lab.learning_mind import aggregation

    called = False
    def unexpected(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must reject output collision before combining")
    monkeypatch.setattr(aggregation, "combine_macro_label_runs", unexpected)
    with pytest.raises(ValueError, match="outside the combined-label directory"):
        aggregation.finalize_macro_label_runs(inputs=[tmp_path / "py", tmp_path / "ts"],
            combined_output=tmp_path / "combined", confidence_output=tmp_path / "combined" / "confidence.json",
            identity={}, selection_path=tmp_path / "selection.json")
    assert called is False
