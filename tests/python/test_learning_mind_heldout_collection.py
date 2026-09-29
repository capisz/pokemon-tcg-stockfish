from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ptcg_lab.learning_mind.heldout_collection import label_heldout_candidates
from ptcg_lab.learning_mind import heldout_collection as collector
from ptcg_lab.learning_mind import heldout_evaluation as evaluator
from ptcg_lab.learning_mind.macro import MacroCandidateV1
from ptcg_lab.learning_mind.macro_fidelity import _validate_rollout_label
from ptcg_lab.learning_mind.macro import CANDIDATE_GENERATOR_VERSION
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.storage import digest as observation_digest


def _candidates():
    return [MacroCandidateV1(turn_intent="no-attack", action_ids=(action_id,),
        action_sequence=({"id": action_id, "type": "pass"},))
        for action_id in ("pass-a", "pass-b")]


def test_heldout_collector_lock_rejects_a_concurrent_writer(tmp_path):
    output = tmp_path / "python-run"
    with collector._heldout_output_lock(output):
        repo = Path(__file__).resolve().parents[2]
        code = """
from pathlib import Path
import sys
from ptcg_lab.learning_mind.heldout_collection import _heldout_output_lock
try:
    with _heldout_output_lock(Path(sys.argv[1])):
        print("acquired")
except RuntimeError:
    print("blocked")
"""
        env = dict(os.environ, PYTHONPATH=str(repo / "src"))
        result = subprocess.run([sys.executable, "-c", code, str(output)],
            check=True, capture_output=True, text=True, env=env)
        assert result.stdout.strip() == "blocked"
    assert not output.exists()


def test_heldout_adaptive_sampler_uses_common_seeds_and_valid_label_schema():
    candidates = _candidates()
    calls = {candidate.key(): [] for candidate in candidates}

    def rollout(candidate, seed):
        calls[candidate.key()].append(seed)
        score = 1.0 if candidate.action_ids[0] == "pass-b" else 0.0
        return {"status": "finished", "score": score, "decisionCount": 4}

    labels = label_heldout_candidates(candidates, "position-hash", rollout,
        rollout_identity="fixed-heldout-identity")
    assert [len(calls[candidate.key()]) for candidate in candidates] == [16, 64]
    assert calls[candidates[0].key()] == calls[candidates[1].key()][:16]
    assert labels[0]["outcomes"] == {"finished": 16, "truncated": 0, "error": 0}
    assert labels[1]["outcomes"] == {"finished": 64, "truncated": 0, "error": 0}
    for label in labels:
        evidence = _validate_rollout_label(label)
        assert evidence["candidate"].key() == label["candidateHash"]
        assert label["attemptedRollouts"] == label["completedRollouts"]
        assert set(label["outcomesBySeedIndex"]) == {
            str(index) for index in range(label["attemptedRollouts"])}


def test_heldout_sampler_resume_is_bit_equivalent_and_rejects_identity_drift():
    candidates = _candidates()

    def rollout(candidate, seed):
        return {"status": "finished", "score": .5 if seed % 2 else 1.0,
                "decisionCount": 7}

    uninterrupted = label_heldout_candidates(candidates, "position-hash", rollout,
        rollout_identity="resume-identity")
    saved = []

    def interrupt(state):
        saved.append(copy.deepcopy(state))
        if len(saved) == 5:
            raise RuntimeError("simulated interruption at durable seed boundary")

    with pytest.raises(RuntimeError, match="durable seed boundary"):
        label_heldout_candidates(candidates, "position-hash", rollout,
            rollout_identity="resume-identity", checkpoint=interrupt)
    resumed = label_heldout_candidates(candidates, "position-hash", rollout,
        rollout_identity="resume-identity", resume_state=saved[-1])
    assert resumed == uninterrupted
    with pytest.raises(ValueError, match="identity/allocation"):
        label_heldout_candidates(candidates, "position-hash", rollout,
            rollout_identity="different-identity", resume_state=saved[-1])


def test_heldout_sampler_keeps_truncations_and_errors_unscored():
    candidates = _candidates()

    def rollout(candidate, _seed):
        if candidate.action_ids[0] == "pass-a":
            return {"status": "truncated", "reason": "horizon", "decisionCount": 500}
        return {"status": "error", "reason": "engine-failure"}

    labels = label_heldout_candidates(candidates, "position-hash", rollout,
        rollout_identity="unfinished-outcomes", initial=2, maximum=2)
    assert labels[0]["outcomes"] == {"finished": 0, "truncated": 2, "error": 0}
    assert labels[1]["outcomes"] == {"finished": 0, "truncated": 0, "error": 2}
    assert all(label["expectedResult"] is None and label["weight"] == 0 for label in labels)
    assert all(sum(label["outcomes"].values()) == label["attemptedRollouts"] for label in labels)
    assert all(len(label["outcomesBySeedIndex"]) == 2 for label in labels)


def test_heldout_collection_manifest_loads_through_real_evaluator_path(tmp_path, monkeypatch):
    family = "python-heuristic"
    identity = {"identityHash": "frozen-identity"}
    observation = {"playerId": 0, "legalActions": [
        {"id": "pass-a", "type": "pass"}, {"id": "pass-b", "type": "pass"}]}
    position_hash = observation_digest(observation)
    metadata = {"positionHash": position_hash, "sourceGameId": "game-heldout",
        "targetDeck": "deck-a", "opponentArchetype": "archetype-a", "positionStage": "late"}
    row = {**metadata, "split": "heldout", "identity": identity,
        "familyId": "family-a", "opponentPolicyFamily": family, "observation": observation}
    selection = {"schemaVersion": 1, "kind": "macro-ranker-heldout-selection-v1",
        "identity": identity, "selectionHash": "e" * 64}
    selection_path = tmp_path / "heldout-selection.json"
    selection_path.write_text(json.dumps(selection))
    support_path = tmp_path / "support.json"
    generator = {"version": CANDIDATE_GENERATOR_VERSION}
    support = {"candidateGeneratorIdentity": generator, "reportHash": "frozen-support-hash",
        "positions": [{"positionHash": position_hash, "status": "supported",
            "completeCandidateCount": len(_candidates())}]}
    support_path.write_text(json.dumps(support))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_manifest = {"manifestHash": "frozen-dataset-hash"}
    candidates = _candidates()
    monkeypatch.setattr(collector, "runtime_identity",
        lambda _root: SimpleNamespace(record=lambda: identity))
    monkeypatch.setattr(collector, "_read_frozen_heldout_selection",
        lambda _path: (selection, {family: {position_hash: metadata}}))
    monkeypatch.setattr(collector, "_verify_source_pool",
        lambda **_: (dataset_manifest, support))
    monkeypatch.setattr(collector, "load_dataset", lambda *_args, **_kwargs:
        (dataset_manifest, [row]))
    monkeypatch.setattr(collector, "transition_generator_identity", lambda _root: generator)
    monkeypatch.setattr(collector, "generate_transition_candidates",
        lambda *_: (candidates, {"hypothesisId": "public-hypothesis"}))

    class FakeEnginePool:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @contextmanager
        def lease(self):
            yield self

        def request(self, _command, request):
            action_id = request["macroPlanActions"][0]["id"]
            winner = 0 if action_id == "pass-a" else None
            return {"status": "complete", "alternatives": [
                {"actionId": action_id, "visits": 1, "score": .5,
                 "continuation": {"end": "terminal",
                    "outcome": {"winner": winner}, "decisionCount": 8}}],
                "macroPlanExecution": {"requested": True, "completed": True},
                "warnings": []}

    monkeypatch.setattr(collector, "EnginePool", FakeEnginePool)
    output = tmp_path / "heldout-run"
    preflight = collector.preflight_heldout_macro_labels(root=tmp_path,
        family=family, dataset_dir=dataset_dir, support_path=support_path,
        selection_path=selection_path, output=output, initial=2, maximum=2,
        rollout_workers=1)
    assert preflight["candidatePlans"] == len(candidates)
    assert preflight["initialCandidateSeedRollouts"] == 4
    assert preflight["maximumCandidateSeedRollouts"] == 4
    assert preflight["startsEngine"] is False and preflight["writesArtifacts"] is False
    assert not output.exists()
    manifest = collector.collect_heldout_macro_labels(root=tmp_path,
        family=family, dataset_dir=dataset_dir, support_path=support_path,
        selection_path=selection_path, output=output, initial=2, maximum=2,
        rollout_workers=1)
    assert manifest["positions"] == manifest["supportedPositions"] == 1
    assert manifest["rolloutIdentity"] == preflight["rolloutIdentity"]
    loaded_manifest, records = evaluator._load_heldout_run(family=family,
        labels_dir=output, selection={**selection,
            "splits": [{"policyFamily": family, "positions": [metadata]}]},
        selected_positions={position_hash: metadata}, dataset_manifest=dataset_manifest,
        support=support, expected_generator=generator,
        expected_settings={key: manifest[key] for key in evaluator.FROZEN_HELDOUT_SETTINGS})
    assert loaded_manifest["manifestHash"] == manifest["manifestHash"]
    assert len(records) == 1
    assert records[0]["status"] == "collected"
