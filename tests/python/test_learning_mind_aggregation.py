import json
import math
from dataclasses import asdict

import pytest

from ptcg_lab.learning_mind.aggregation import (SHARED_ROLLOUT_SETTINGS,
    combine_macro_label_runs, combine_macro_label_training_repair_runs)
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.macro import MacroCandidateV1, rollout_seed
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.storage import digest as observation_digest
from test_learning_mind_representation import observation

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


def _write_run(root, family, identity, *, splits=SPLITS,
               collector_version="fixture-collector-v1", collector_sha="c" * 64):
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
        "labelCollectorVersion": collector_version, "labelCollectorSha256": collector_sha,
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


def test_lineage_repair_combiner_keeps_parent_immutable_and_binds_each_shard(tmp_path):
    from ptcg_lab.learning_mind.experiment import _load_ranker_input
    from ptcg_lab.learning_mind.selection import freeze_macro_label_training_repair

    identity = {"identityHash": "repair-merge-identity"}
    families = ["python-heuristic", "typescript-heuristic"]
    datasets, supports, parent_splits = {}, {}, []
    rows_by_family = {}
    generator = {"version": "repair-fixture-generator-v1"}
    for family in families:
        folder = tmp_path / f"{family}-dataset"
        folder.mkdir()
        rows = [
            {"positionHash": f"{family}-old", "sourceGameId": f"{family}-game-train",
             "split": "train", "targetDeck": "crustle", "opponentArchetype": "dragapult",
             "opponentPolicyFamily": family, "positionStage": "midgame"},
            {"positionHash": f"{family}-new", "sourceGameId": f"{family}-game-train",
             "split": "train", "targetDeck": "crustle", "opponentArchetype": "dragapult",
             "opponentPolicyFamily": family, "positionStage": "late"},
            {"positionHash": f"{family}-dev", "sourceGameId": f"{family}-game-dev",
             "split": "development", "targetDeck": "dragapult", "opponentArchetype": "raging-bolt",
             "opponentPolicyFamily": family, "positionStage": "opening"},
        ]
        rows_by_family[family] = rows
        rows_path = folder / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        dataset_manifest = {"id": "learning-mind-macro-position-pool-v1", "identity": identity,
            "rows": len(rows), "rowsSha256": file_sha256(rows_path)}
        dataset_manifest["manifestHash"] = identity_hash(dataset_manifest)
        (folder / "manifest.json").write_text(json.dumps(dataset_manifest))
        support = {"schemaVersion": 1, "audit": "actor-visible-macro-candidate-support-only",
            "status": "no-rollouts-no-labels", "identity": identity,
            "datasetManifestHash": dataset_manifest["manifestHash"],
            "datasetRowsSha256": file_sha256(rows_path), "candidateGeneratorIdentity": generator,
            "positions": [
                {"positionHash": f"{family}-old", "status": "supported", "completeCandidateCount": 1},
                {"positionHash": f"{family}-new", "status": "supported", "completeCandidateCount": 3},
                {"positionHash": f"{family}-dev", "status": "supported", "completeCandidateCount": 2},
            ]}
        support["reportHash"] = identity_hash(support)
        support_path = tmp_path / f"{family}-support.json"
        support_path.write_text(json.dumps(support))
        datasets[family], supports[family] = folder, support_path
        for split, row in (("train", rows[0]), ("development", rows[2])):
            metadata = {key: row[key] for key in (
                "positionHash", "sourceGameId", "targetDeck", "opponentArchetype", "positionStage")}
            parent_splits.append({"policyFamily": family, "split": split, "sourceGames": 1,
                "positionHashes": [row["positionHash"]], "positions": [metadata]})

    parent = {"schemaVersion": 1, "selection": "repair-parent-fixture-v1",
        "identity": identity, "splits": parent_splits}
    parent["selectionHash"] = identity_hash(parent)
    parent_path = tmp_path / "parent-selection.json"
    parent_path.write_text(json.dumps(parent))
    selection_path = tmp_path / "repair-selection.json"
    repair_selection = freeze_macro_label_training_repair(datasets=datasets,
        support_reports=supports, parent_selection_path=parent_path,
        output=selection_path, identity=identity)

    settings = {
        "initialRollouts": 16, "maximumRollouts": 64, "extensionBatchSize": 8,
        "horizon": 500, "rolloutBudgetMs": 60000, "rolloutWorkers": 8,
        "rolloutSeedVersion": "configuration-bound-v1",
        "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
        "selectionMethod": "position-hash-list", "splitFilter": None,
    }
    def write_run(path, family, role, selected_rows):
        path.mkdir()
        run_id = f"{family}-{role}-rollout"
        files = []
        for row in selected_rows:
            record = {"positionHash": row["positionHash"], "split": row["split"],
                "identity": identity, "datasetManifestHash": json.loads(
                    (datasets[family] / "manifest.json").read_text())["manifestHash"],
                "rolloutIdentity": run_id, "opponentPolicyFamily": family,
                "status": "collected", "highConfidencePolicyEligible": False}
            record_path = path / f"{row['positionHash']}.json"
            record_path.write_text(json.dumps(record))
            files.append({"path": record_path.name, "sha256": file_sha256(record_path)})
        manifest = {"identity": identity, "datasetManifestHash": json.loads(
            (datasets[family] / "manifest.json").read_text())["manifestHash"],
            "rolloutIdentity": run_id, "candidateGeneratorVersion": generator["version"],
            "candidateGeneratorIdentity": generator,
            "labelCollectorVersion": "fixture-collector-v1", "labelCollectorSha256": "c" * 64,
            **settings, "positions": len(files),
            "selectedPositionHashes": [item["path"][:-5] for item in files], "files": files}
        manifest["manifestHash"] = identity_hash(manifest)
        (path / "manifest.json").write_text(json.dumps(manifest))
        return path

    base_runs, repair_runs = {}, {}
    for family in families:
        rows = rows_by_family[family]
        base_runs[family] = write_run(tmp_path / f"{family}-base-run", family,
            "parent", [rows[0], rows[2]])
        repair_runs[family] = write_run(tmp_path / f"{family}-repair-run", family,
            "supplement", [rows[1]])
    output = tmp_path / "combined-repair"
    combined = combine_macro_label_training_repair_runs(base_runs=base_runs,
        repair_runs=repair_runs, datasets=datasets, support_reports=supports,
        output=output, identity=identity, selection_path=selection_path,
        parent_selection_path=parent_path)
    loaded_manifest, records = _load_ranker_input(output, selection_path)
    assert combined["positions"] == 4
    assert combined["positionsBySplit"] == {"development": 2, "train": 2}
    assert loaded_manifest["manifestHash"] == combined["manifestHash"]
    assert {record["positionHash"] for record in records} == {
        f"{family}-{split}" for family in families for split in ("new", "dev")}
    assert len(combined["sourceRuns"]) == 4
    assert all(run["positions"] == 1 for run in combined["sourceRuns"])
    assert all(run["sourceRunPositions"] == 2 for run in combined["sourceRuns"]
               if run["sourceRole"] == "parent-retained")
    assert repair_selection["lineage"]["parentSelectionHash"] == parent["selectionHash"]
    assert all((run / "manifest.json").is_file() for run in [*base_runs.values(), *repair_runs.values()])


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


def test_combiner_rejects_unapproved_collector_source_mixture(tmp_path):
    identity = {"identityHash": "frozen-identity"}
    python_dir = _write_run(tmp_path / "python", FAMILIES[0], identity,
        collector_version="macro-rollout-labeler-v6",
        collector_sha="1b7e8df4a69cba6dcbb74d23ba310a9ed18c8ca5859ae5b9c6efbf2f8e4e646a")
    typescript_dir = _write_run(tmp_path / "typescript", FAMILIES[1], identity,
        collector_version="macro-rollout-labeler-v6", collector_sha="f" * 64)
    selection = _write_selection(tmp_path / "selection.json", identity)
    output = tmp_path / "combined"
    with pytest.raises(ValueError, match="unaudited collector source hashes"):
        combine_macro_label_runs(inputs=[python_dir, typescript_dir], output=output,
            identity=identity, selection_path=selection)
    assert not output.exists()


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


def test_finalizer_end_to_end_on_frozen_offline_fixture(tmp_path):
    from ptcg_lab.learning_mind.aggregation import finalize_macro_label_runs
    from ptcg_lab.learning_mind.confidence_audit import verify_macro_label_confidence_audit

    identity = {"identityHash": "frozen-identity"}
    positions = {family: {split: [] for split in SPLITS} for family in FAMILIES}
    observations = {}
    for family_index, family in enumerate(FAMILIES):
        for split_index, split in enumerate(SPLITS):
            obs = observation()
            obs["turn"] += family_index * 10 + split_index
            position_hash = observation_digest(obs)
            positions[family][split].append(position_hash)
            observations[(family, split)] = obs
    selection = _write_selection(tmp_path / "selection.json", identity, positions)
    inputs = []
    for family_index, family in enumerate(FAMILIES):
        run_dir = tmp_path / family
        run_dir.mkdir()
        records = []
        for split_index, split in enumerate(SPLITS):
            position_hash = positions[family][split][0]
            obs = observations[(family, split)]
            action = obs["legalActions"][3]
            candidate = MacroCandidateV1(turn_intent="attack", intended_attack=action["label"],
                action_ids=(action["id"],), action_sequence=(action,))
            finished = 20
            expected = .6
            uncertainty = math.sqrt(max(expected * (1 - expected), .25) / finished)
            rollout_identity = f"rollout-{family}"
            namespace = "training" if split == "train" else "development"
            seeds = [rollout_seed(namespace, position_hash, index, rollout_identity) for index in range(20)]
            label = {"candidate": asdict(candidate), "candidateHash": candidate.key(),
                "completedRollouts": finished, "attemptedRollouts": finished,
                "outcomes": {"finished": finished, "truncated": 0, "error": 0},
                "outcomeReasons": {}, "decisionCountDistribution": {"12": finished},
                "expectedResult": expected, "relativeResult": 0.0,
                "uncertainty": uncertainty, "weight": finished / (1 + uncertainty)}
            record = {"positionHash": position_hash, "split": split,
                "seedNamespace": namespace, "rolloutIdentity": rollout_identity,
                "rolloutSeeds": seeds, "observation": obs,
                "opponentPolicyFamily": family, "identity": identity,
                "datasetManifestHash": f"dataset-{family}", "status": "collected",
                "candidateCount": 1, "labels": [label]}
            record_path = run_dir / f"{position_hash}.json"
            record_path.write_text(json.dumps(record))
            records.append({"path": record_path.name, "sha256": file_sha256(record_path)})
        collector_hash = ("1b7e8df4a69cba6dcbb74d23ba310a9ed18c8ca5859ae5b9c6efbf2f8e4e646a"
                          if family == "python-heuristic" else
                          "822500b4113aea6c69abeb7fe069ce2647d561d742977f5453097c0bae6882a4")
        manifest = {"identity": identity, "datasetManifestHash": f"dataset-{family}",
            "rolloutIdentity": f"rollout-{family}", "candidateGeneratorVersion": "fixture-v1",
            "candidateGeneratorIdentity": {"version": "fixture-v1"},
            "labelCollectorVersion": "macro-rollout-labeler-v6", "labelCollectorSha256": collector_hash,
            "initialRollouts": 16, "maximumRollouts": 64, "extensionBatchSize": 8,
            "horizon": 500, "rolloutBudgetMs": 60000, "rolloutWorkers": 8,
            "rolloutSeedVersion": "configuration-bound-v1",
            "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
            "selectionMethod": "position-hash-list", "splitFilter": None,
            "positions": len(records), "selectedPositionHashes": [positions[family][split][0] for split in SPLITS],
            "files": records}
        manifest["manifestHash"] = identity_hash(manifest)
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        inputs.append(run_dir)

    combined, confidence = tmp_path / "combined", tmp_path / "confidence.json"
    result = finalize_macro_label_runs(inputs=inputs, combined_output=combined,
        confidence_output=confidence, identity=identity, selection_path=selection)
    verified = verify_macro_label_confidence_audit(labels_dir=combined,
        selection_path=selection, report_path=confidence)
    assert result["status"] == "verified-analysis-only"
    assert verified["summaryByFamilyAndSplit"]["python-heuristic/train"]["positions"] == 1
    assert verified["summaryByFamilyAndSplit"]["typescript-heuristic/development"]["positions"] == 1
    assert result["confidenceReportHash"] == verified["reportHash"]
    assert verified["collectorIdentity"]["labelCollectorCompatibility"]["kind"] == "audited-same-collection-surface"
    assert verified["collectorIdentity"]["labelCollectorSourceSha256s"] == sorted({
        "1b7e8df4a69cba6dcbb74d23ba310a9ed18c8ca5859ae5b9c6efbf2f8e4e646a",
        "822500b4113aea6c69abeb7fe069ce2647d561d742977f5453097c0bae6882a4"})
