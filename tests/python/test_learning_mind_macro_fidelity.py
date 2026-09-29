from __future__ import annotations

from dataclasses import asdict
import hashlib
import copy
import json
from pathlib import Path

import pytest

from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.macro import CANDIDATE_GENERATOR_VERSION, MacroCandidateV1, rollout_seed
from ptcg_lab.learning_mind.macro_fidelity import audit_raging_bolt_macro_fidelity
from ptcg_lab.learning_mind.schema import IdentityManifest, identity_hash
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker
from ptcg_lab.storage import digest as legacy_digest
from test_learning_mind_representation import observation


ROOT = Path(__file__).parents[2]
FAMILIES = ("python-heuristic", "typescript-heuristic")


def _generator_identity():
    paths = {
        "plannerSha256": ROOT / "research/learning_mind/transition_macro_planner.ts",
        "actionKeySha256": ROOT / "packages/engine/src/action-key.ts",
        "plannerWorkerSha256": ROOT / "research/learning_mind/planner_worker.ts",
        "plannerBundleSha256": ROOT / "packages/engine/dist/learning-mind-planner.cjs",
        "adapterSha256": ROOT / "src/ptcg_lab/learning_mind/macro.py",
    }
    return {"version": CANDIDATE_GENERATOR_VERSION,
            **{key: file_sha256(path) for key, path in paths.items()}}


def _fixture(tmp_path, *, successful=True, wrong_root=False):
    identity = IdentityManifest.create(engine_build_hash="engine", deck_manifests={}, card_metadata={}).record()
    selection_entries = []
    family_inputs = {}
    for family_index, family in enumerate(FAMILIES):
        dataset = tmp_path / f"{family}-dataset"
        labels_dir = tmp_path / f"{family}-labels"
        dataset.mkdir(); labels_dir.mkdir()
        rows, files, selected = [], [], []
        for split_index, split in enumerate(("train", "development")):
            obs = copy.deepcopy(observation())
            obs["turn"] += 100 * family_index + split_index
            pass_action = {"id": f"pass-{family_index}-{split_index}", "type": "pass", "label": "End turn"}
            obs["legalActions"] = [pass_action]
            tracker = ObservableHistoryTracker(0).update(obs)
            position_hash = legacy_digest(obs)
            rows.append({"positionHash": position_hash, "familyId": f"{family}-{split}",
                "sourceGameId": f"{family}-{split}-game", "sourceDecisionIndex": 0, "actor": 0,
                "targetDeck": "raging-bolt", "deckHash": "frozen-deck", "opponentArchetype": "crustle",
                "opponentPolicyFamily": family, "positionStage": "opening", "split": split,
                "featureIdentityHash": "feature-hash", "policyLabelSource": None,
                "acceptableActionIndices": None, "policyDistribution": None,
                "observation": obs, "tracker": tracker})
            selected.append(position_hash)
            selection_entries.append({"policyFamily": family, "split": split,
                "positionHashes": [position_hash], "sourceGames": 1})

            candidate = MacroCandidateV1(turn_intent="no-attack", action_ids=(pass_action["id"],),
                action_sequence=(pass_action,))
            declared = asdict(candidate)
            if wrong_root and family == FAMILIES[0] and split == "train":
                declared["action_sequence"][0]["id"] = "not-the-root"
            if successful:
                outcomes = {"finished": 1, "truncated": 0, "error": 0}
                reasons = {}
            else:
                outcomes = {"finished": 0, "truncated": 0, "error": 1}
                reasons = {"MACRO_PLAN_UNEXECUTABLE:0:fixture": 1}
            label = {"candidate": declared, "candidateHash": candidate.key(),
                "attemptedRollouts": 1, "completedRollouts": outcomes["finished"],
                "outcomes": outcomes, "outcomeReasons": reasons}
            record = {"positionHash": position_hash, "split": split, "identity": identity,
                "datasetManifestHash": None, "opponentPolicyFamily": family,
                "observation": obs, "status": "collected", "rolloutIdentity": None,
                "seedNamespace": "training" if split == "train" else "development",
                "generatorSeed": int.from_bytes(hashlib.sha256(
                    f"learning-mind-v1|macro-generator|{position_hash}".encode()).digest()[:4], "big"),
                "candidateCount": 1, "labels": [label], "highConfidencePolicyEligible": False}
            files.append((position_hash, record))

        rows_path = dataset / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                    for row in rows))
        dataset_manifest = {"schemaVersion": 1, "identity": identity, "rows": len(rows),
            "rowsSha256": file_sha256(rows_path), "targetDeck": "raging-bolt"}
        dataset_manifest["manifestHash"] = identity_hash(dataset_manifest)
        (dataset / "manifest.json").write_text(json.dumps(dataset_manifest))

        settings = {"identity": identity, "datasetManifestHash": dataset_manifest["manifestHash"],
            "requestedPositions": len(selected), "initialRollouts": 1, "maximumRollouts": 1,
            "extensionBatchSize": 1, "horizon": 16, "rolloutBudgetMs": 1000, "rolloutWorkers": 1,
            "splitFilter": None, "selectionMethod": "position-hash-list",
            "selectedPositionHashes": selected, "candidateGeneratorVersion": CANDIDATE_GENERATOR_VERSION,
            "candidateGeneratorIdentity": _generator_identity(), "rolloutSeedVersion": "configuration-bound-v1",
            "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
            "labelCollectorVersion": "macro-rollout-labeler-v6",
            "labelCollectorSha256": file_sha256(ROOT / "src/ptcg_lab/learning_mind/experiment.py")}
        rollout_identity = identity_hash(settings)
        manifest_files = []
        for position_hash, record in files:
            record["datasetManifestHash"] = dataset_manifest["manifestHash"]
            record["rolloutIdentity"] = rollout_identity
            record["rolloutSeeds"] = [rollout_seed(record["seedNamespace"], position_hash, index,
                                                    rollout_identity)
                                      for index in range(settings["maximumRollouts"])]
            path = labels_dir / f"{position_hash}.json"
            path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
            manifest_files.append({"path": path.name, "sha256": file_sha256(path)})
        label_manifest = {"schemaVersion": 1, **settings, "rolloutIdentity": rollout_identity,
            "positions": len(manifest_files), "files": manifest_files,
            "supportedPositions": len(manifest_files), "unsupportedPositions": 0,
            "highConfidencePolicyLabels": 0, "resumePolicy": "verified position files are immutable"}
        label_manifest["manifestHash"] = identity_hash(label_manifest)
        (labels_dir / "manifest.json").write_text(json.dumps(label_manifest))
        family_inputs[family] = (dataset, labels_dir)

    selection = {"schemaVersion": 1, "identity": identity, "splits": selection_entries}
    selection["selectionHash"] = identity_hash(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    return identity, selection_path, family_inputs


def test_macro_fidelity_audit_requires_successful_plan_execution_per_raging_bolt_position(tmp_path):
    _identity, selection, inputs = _fixture(tmp_path)
    report = audit_raging_bolt_macro_fidelity(root=ROOT, selection_path=selection,
        python_dataset=inputs[FAMILIES[0]][0], typescript_dataset=inputs[FAMILIES[1]][0],
        python_labels=inputs[FAMILIES[0]][1], typescript_labels=inputs[FAMILIES[1]][1],
        output=tmp_path / "fidelity.json")
    assert report["status"] == "passed"
    assert report["ragingBoltMacroPlanFidelity"] == "passed"
    assert report["ragingBoltPositions"] == 4
    assert report["successfulCandidates"] == 4
    assert report["automaticPromotion"] is False


def test_macro_fidelity_audit_does_not_call_typed_failures_success(tmp_path):
    _identity, selection, inputs = _fixture(tmp_path, successful=False)
    report = audit_raging_bolt_macro_fidelity(root=ROOT, selection_path=selection,
        python_dataset=inputs[FAMILIES[0]][0], typescript_dataset=inputs[FAMILIES[1]][0],
        python_labels=inputs[FAMILIES[0]][1], typescript_labels=inputs[FAMILIES[1]][1],
        output=tmp_path / "fidelity.json")
    assert report["status"] == "insufficient"
    assert report["typedFailureCandidates"] == 4
    assert report["successfulCandidates"] == 0


def test_macro_fidelity_audit_rejects_root_action_not_in_frozen_actor_view(tmp_path):
    _identity, selection, inputs = _fixture(tmp_path, wrong_root=True)
    with pytest.raises(ValueError, match="action IDs differ"):
        audit_raging_bolt_macro_fidelity(root=ROOT, selection_path=selection,
            python_dataset=inputs[FAMILIES[0]][0], typescript_dataset=inputs[FAMILIES[1]][0],
            python_labels=inputs[FAMILIES[0]][1], typescript_labels=inputs[FAMILIES[1]][1],
            output=tmp_path / "fidelity.json")


def test_macro_fidelity_audit_rejects_duplicate_manifest_file_entry(tmp_path):
    _identity, selection, inputs = _fixture(tmp_path)
    labels_dir = inputs[FAMILIES[0]][1]
    manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][1] = copy.deepcopy(manifest["files"][0])
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                               if key != "manifestHash"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every frozen position exactly once"):
        audit_raging_bolt_macro_fidelity(root=ROOT, selection_path=selection,
            python_dataset=inputs[FAMILIES[0]][0], typescript_dataset=inputs[FAMILIES[1]][0],
            python_labels=inputs[FAMILIES[0]][1], typescript_labels=inputs[FAMILIES[1]][1],
            output=tmp_path / "fidelity.json")


@pytest.mark.parametrize("tamper", ("rollout-seed", "generator-seed"))
def test_macro_fidelity_audit_rejects_seed_identity_drift(tmp_path, tamper):
    _identity, selection, inputs = _fixture(tmp_path)
    labels_dir = inputs[FAMILIES[0]][1]
    labels_manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(labels_manifest_path.read_text())
    entry = manifest["files"][0]
    record_path = labels_dir / entry["path"]
    record = json.loads(record_path.read_text())
    if tamper == "rollout-seed":
        record["rolloutSeeds"][0] += 1
    else:
        record["generatorSeed"] += 1
    record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    entry["sha256"] = file_sha256(record_path)
    manifest["manifestHash"] = identity_hash({key: value for key, value in manifest.items()
                                               if key != "manifestHash"})
    labels_manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seeds differ from frozen position/split identity"):
        audit_raging_bolt_macro_fidelity(root=ROOT, selection_path=selection,
            python_dataset=inputs[FAMILIES[0]][0], typescript_dataset=inputs[FAMILIES[1]][0],
            python_labels=inputs[FAMILIES[0]][1], typescript_labels=inputs[FAMILIES[1]][1],
            output=tmp_path / "fidelity.json")
