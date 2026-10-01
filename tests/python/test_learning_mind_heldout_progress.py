from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ptcg_lab.learning_mind.heldout_collection import label_heldout_candidates
from ptcg_lab.learning_mind.heldout_progress import heldout_macro_label_progress
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.macro import MacroCandidateV1
from ptcg_lab.learning_mind.schema import identity_hash


FAMILIES = ("python-heuristic", "typescript-heuristic")


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _selection(path: Path, position_hash="position-python"):
    identity = {"identityHash": "frozen-identity"}
    splits = []
    for family, suffix in zip(FAMILIES, ("python", "typescript")):
        position = {"positionHash": position_hash if suffix == "python" else "position-typescript",
            "sourceGameId": f"game-{suffix}", "targetDeck": "deck-a",
            "opponentArchetype": "archetype-a", "positionStage": "late"}
        splits.append({"policyFamily": family, "split": "heldout", "sourceGames": 1,
            "sourceGameIds": [position["sourceGameId"]],
            "positionHashes": [position["positionHash"]], "positions": [position]})
    selection = {"schemaVersion": 1, "kind": "macro-ranker-heldout-selection-v1",
        "trainingEligible": False,
        "interpretation": "heldout evaluator selection only; not for training",
        "minimumCompleteCandidatesPerPosition": 2, "identity": identity,
        "sourcePools": [], "splits": splits}
    selection["selectionHash"] = identity_hash(selection)
    return _write(path, selection), selection


def _candidates():
    return [MacroCandidateV1(turn_intent="no-attack", action_ids=(action_id,),
        action_sequence=({"id": action_id, "type": "pass"},))
        for action_id in ("pass-a", "pass-b")]


def test_heldout_progress_reports_zero_for_a_fresh_missing_run_directory(tmp_path):
    selection_path, selection = _selection(tmp_path / "selection.json")
    run_dir = tmp_path / "not-started" / "python"
    report = heldout_macro_label_progress(family=FAMILIES[0],
        selection_path=selection_path, run_dir=run_dir)
    assert report["selectionHash"] == selection["selectionHash"]
    assert report["selectedPositions"] == 1
    assert report["finalizedResultFiles"] == 0
    assert report["finalizedPositionPercent"] == 0.0
    assert report["positionsInProgress"] == 0
    assert report["positionsNotStarted"] == 1
    assert report["checkpoints"] == []
    assert report["manifestStatus"] == "not-yet-published"
    assert report["processStatus"].startswith("unknown")
    assert report["writesArtifacts"] is False
    assert not run_dir.exists()


def test_heldout_progress_reports_verified_seed_checkpoint_without_polling(tmp_path):
    selection_path, selection = _selection(tmp_path / "selection.json")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    position_hash = "position-python"
    candidates = _candidates()
    saved = []

    def checkpoint(state):
        saved.append(copy.deepcopy(state))
        if len(saved) == 3:
            raise RuntimeError("stop after initial batch and confidence check")

    def rollout(_candidate, _seed):
        return {"status": "finished", "score": .5, "decisionCount": 4}

    with pytest.raises(RuntimeError, match="initial batch"):
        label_heldout_candidates(candidates, position_hash, rollout,
            rollout_identity="frozen-rollout", initial=2, maximum=4,
            extension_batch_size=2, checkpoint=checkpoint)

    progress_identity = {"positionHash": position_hash,
        "rolloutIdentity": "frozen-rollout",
        "candidateHashes": [candidate.key() for candidate in candidates],
        "generatorSeed": 17, "generatorHypothesisId": "public-hypothesis",
        "candidateGeneratorIdentity": {"version": "generator-v1"},
        "allocation": {"initial": 2, "maximum": 4,
            "extensionBatchSize": 2, "closeMargin": .10}}
    checkpoint_record = {"schemaVersion": 1, "progressIdentity": progress_identity,
        "progressIdentityHash": identity_hash(progress_identity), "state": saved[-1]}
    checkpoint_record["progressHash"] = identity_hash(checkpoint_record)
    _write(run_dir / f".{position_hash}.progress", checkpoint_record)

    report = heldout_macro_label_progress(family=FAMILIES[0],
        selection_path=selection_path, run_dir=run_dir)
    checkpoint_report = report["checkpoints"][0]
    assert report["selectionHash"] == selection["selectionHash"]
    assert report["positionsInProgress"] == 1 and report["positionsNotStarted"] == 0
    assert checkpoint_report["completedSeedIndices"] == [0, 1]
    assert checkpoint_report["initialSeedsCompleted"] == checkpoint_report["initialSeedsPlanned"] == 2
    assert checkpoint_report["completedExtensionSeeds"] == 0
    assert checkpoint_report["highestExtensionIndex"] is None
    assert checkpoint_report["activeCandidates"] == 2
    assert checkpoint_report["samplesPerActiveCandidateMinMax"] == [2, 2]
    assert checkpoint_report["outcomesAcrossActiveCandidates"] == {
        "finished": 4, "truncated": 0, "error": 0}
    assert "next completed extension batch" in checkpoint_report["nextStoppingGate"]
    assert report["processStatus"].startswith("unknown")
    assert report["writesArtifacts"] is False


def test_heldout_progress_rejects_seed_receipt_checkpoint_corruption(tmp_path):
    selection_path, _ = _selection(tmp_path / "selection.json")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate = _candidates()[0]
    state = {"allocation": {"initial": 1, "maximum": 1,
            "extensionBatchSize": 1, "closeMargin": .10},
        "candidateHashes": [candidate.key()], "rolloutIdentity": "frozen",
        "records": {candidate.key(): {"finished": 1, "truncated": 0, "error": 0,
            "outcomesByIndex": {}}}, "completedInitialIndices": [0],
        "completedExtensionIndices": [], "closeCandidateHashes": [candidate.key()],
        "confidenceEvaluatedThroughIndex": 0}
    progress_identity = {"positionHash": "position-python", "rolloutIdentity": "frozen",
        "candidateHashes": [candidate.key()], "allocation": state["allocation"]}
    checkpoint_record = {"schemaVersion": 1, "progressIdentity": progress_identity,
        "progressIdentityHash": identity_hash(progress_identity), "state": state}
    checkpoint_record["progressHash"] = identity_hash(checkpoint_record)
    _write(run_dir / ".position-python.progress", checkpoint_record)
    with pytest.raises(ValueError, match="receipts do not reconcile"):
        heldout_macro_label_progress(family=FAMILIES[0],
            selection_path=selection_path, run_dir=run_dir)


def test_heldout_progress_waits_for_extension_batch_before_saying_confidence_is_due(tmp_path):
    selection_path, _ = _selection(tmp_path / "selection.json")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    position_hash = "position-python"
    candidates = _candidates()
    saved = []

    def checkpoint(state):
        saved.append(copy.deepcopy(state))
        if len(saved) == 4:
            raise RuntimeError("stop after first extension seed")

    def rollout(_candidate, _seed):
        return {"status": "finished", "score": .5, "decisionCount": 4}

    with pytest.raises(RuntimeError, match="first extension seed"):
        label_heldout_candidates(candidates, position_hash, rollout,
            rollout_identity="frozen-rollout", initial=2, maximum=5,
            extension_batch_size=3, checkpoint=checkpoint)

    allocation = {"initial": 2, "maximum": 5,
        "extensionBatchSize": 3, "closeMargin": .10}
    progress_identity = {"positionHash": position_hash,
        "rolloutIdentity": "frozen-rollout",
        "candidateHashes": [candidate.key() for candidate in candidates],
        "generatorSeed": 17, "generatorHypothesisId": "public-hypothesis",
        "candidateGeneratorIdentity": {"version": "generator-v1"},
        "allocation": allocation}
    checkpoint_record = {"schemaVersion": 1,
        "progressIdentity": progress_identity,
        "progressIdentityHash": identity_hash(progress_identity),
        "state": saved[-1]}
    checkpoint_record["progressHash"] = identity_hash(checkpoint_record)
    _write(run_dir / f".{position_hash}.progress", checkpoint_record)

    report = heldout_macro_label_progress(family=FAMILIES[0],
        selection_path=selection_path, run_dir=run_dir)
    status = report["checkpoints"][0]
    assert status["completedSeedIndices"] == [0, 1, 2]
    assert status["confidenceEvaluatedThroughIndex"] == 1
    assert status["nextStoppingGate"] == "complete extension batch through seed index 4 before confidence reevaluation"


def test_heldout_progress_verifies_completed_manifest_and_position_counts(tmp_path):
    selection_path, selection = _selection(tmp_path / "selection.json")
    position_hash = "position-python"
    run_dir = tmp_path / "run"
    record = {"positionHash": position_hash, "identity": selection["identity"],
        "selectionHash": selection["selectionHash"], "split": "heldout",
        "opponentPolicyFamily": FAMILIES[0], "status": "unsupported",
        "unsupportedReason": "no complete candidates", "labels": [], "candidateCount": 0}
    record_path = _write(run_dir / f"{position_hash}.json", record)
    manifest = {"identity": selection["identity"], "selectionHash": selection["selectionHash"],
        "selectedPositionHashes": [position_hash], "positions": 1,
        "supportedPositions": 0, "unsupportedPositions": 1,
        "highConfidencePolicyLabels": 0,
        "files": [{"path": record_path.name, "sha256": file_sha256(record_path)}]}
    manifest["manifestHash"] = identity_hash(manifest)
    _write(run_dir / "manifest.json", manifest)

    report = heldout_macro_label_progress(family=FAMILIES[0],
        selection_path=selection_path, run_dir=run_dir)
    assert report["manifestStatus"] == "verified-complete"
    assert report["finalizedResultFiles"] == report["selectedPositions"] == 1
    assert report["unsupportedPositions"] == 1
    assert report["positionsNotStarted"] == 0

    manifest["supportedPositions"] = 1
    manifest.pop("manifestHash")
    manifest["manifestHash"] = identity_hash(manifest)
    _write(run_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="position counts differ"):
        heldout_macro_label_progress(family=FAMILIES[0],
            selection_path=selection_path, run_dir=run_dir)
