from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from ptcg_lab.learning_mind import confidence_audit
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.macro import MacroCandidateV1, rollout_seed
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.storage import digest as observation_digest


def _label(candidate, *, finished, truncated=0, errors=0, mean=None, center=0.0):
    raw = json.loads(json.dumps(asdict(candidate)))
    attempted = finished + truncated + errors
    uncertainty = (math.sqrt(max(mean * (1 - mean), .25) / finished)
                   if finished else None)
    return {
        "candidate": raw,
        "candidateHash": candidate.key(),
        "completedRollouts": finished,
        "attemptedRollouts": attempted,
        "outcomes": {"finished": finished, "truncated": truncated, "error": errors},
        "outcomeReasons": {
            **({"search-rollout-horizon-cutoff": truncated} if truncated else {}),
            **({"engine-error": errors} if errors else {}),
        },
        "decisionCountDistribution": {"12": attempted} if attempted else {},
        "expectedResult": mean,
        "relativeResult": mean - center if mean is not None else None,
        "uncertainty": uncertainty,
        "weight": finished / (1 + uncertainty) if finished else 0,
    }


def _record(means, *, samples=64, truncations=None, errors=None):
    actions = [{"id": f"pass-{index}", "type": "pass"} for index in range(len(means))]
    candidates = [MacroCandidateV1(turn_intent="no-attack", action_ids=(action["id"],),
        action_sequence=(action,)) for action in actions]
    observation = {"playerId": 0, "legalActions": actions}
    position_hash = observation_digest(observation)
    truncations = truncations or [0] * len(means)
    errors = errors or [0] * len(means)
    center = max((mean for mean in means if mean is not None), default=0.0)
    labels = [_label(candidate, finished=(samples if mean is not None else 0),
        truncated=truncations[index], errors=errors[index], mean=mean, center=center)
        for index, (candidate, mean) in enumerate(zip(candidates, means))]
    max_attempted = max((label["attemptedRollouts"] for label in labels), default=0)
    return {"positionHash": position_hash, "opponentPolicyFamily": "python-heuristic",
        "split": "train", "seedNamespace": "training", "rolloutIdentity": "fixture-run",
        "rolloutSeeds": [rollout_seed("training", position_hash, index, "fixture-run")
                         for index in range(max_attempted)],
        "observation": observation, "candidateCount": len(labels), "labels": labels}


def test_position_confidence_uses_simultaneous_bounded_intervals_and_can_separate_leader():
    report = confidence_audit._position_confidence(_record([1.0, 0.0]))
    assert report["confidenceStatus"] == "uniquely-separated"
    assert report["plausibleBestCandidateHashes"] == [report["candidates"][0]["candidateHash"]]
    for candidate in report["candidates"]:
        assert 0 <= candidate["simultaneousInterval"]["low"] <= candidate["expectedResultAmongFinished"]
        assert candidate["expectedResultAmongFinished"] <= candidate["simultaneousInterval"]["high"] <= 1


def test_position_confidence_keeps_low_samples_and_unfinished_outcomes_explicit():
    record = _record([0.5, None], samples=16, truncations=[4, 16], errors=[1, 1])
    report = confidence_audit._position_confidence(record)
    assert report["confidenceStatus"] == "ambiguous"
    assert report["completed"] == 16
    assert report["truncated"] == 20
    assert report["errors"] == 2
    assert report["candidates"][0]["sampleStatus"] == "insufficient"
    assert report["candidates"][0]["completionRate"] == 16 / 21
    assert report["candidates"][1]["simultaneousInterval"] == {"low": 0.0, "high": 1.0}


def test_position_confidence_rejects_invalid_alpha_and_duplicate_candidates():
    record = _record([0.5])
    with pytest.raises(ValueError, match="alpha"):
        confidence_audit._position_confidence(record, familywise_alpha=1.0)
    duplicated = {**record, "labels": record["labels"] * 2, "candidateCount": 2}
    with pytest.raises(ValueError, match="duplicate candidate hashes"):
        confidence_audit._position_confidence(duplicated)


def test_custom_confidence_level_is_not_mislabeled_as_95_percent():
    report = confidence_audit._position_confidence(_record([1.0, 0.0]), familywise_alpha=0.10)
    assert report["candidates"][0]["confidenceLevel"] == pytest.approx(0.90)
    assert "simultaneousInterval" in report["candidates"][0]
    assert "interval95" not in report["candidates"][0]


def test_confidence_audit_is_immutable_and_cannot_change_learning_gates(tmp_path, monkeypatch):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    manifest = {"manifestHash": "verified-manifest", "selectionHash": "frozen-selection",
        "labelCollectorVersion": "collector-v1", "labelCollectorSha256": "collector-sha",
        "candidateGeneratorIdentity": {"version": "planner-v1"}, "sourceRuns": [],
        "files": [{"path": "python-heuristic/position.json", "sha256": "source-sha"}]}
    (labels_dir / "manifest.json").write_text(json.dumps(manifest))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"selectionHash": "frozen-selection"}))
    record = _record([1.0, 0.0])
    monkeypatch.setattr(confidence_audit, "_load_ranker_input",
        lambda _labels, _selection: (manifest, [record]))
    output = tmp_path / "confidence.json"
    report = confidence_audit.audit_macro_label_confidence(labels_dir=labels_dir,
        selection_path=selection_path, output=output)
    assert report["status"] == "analysis-only"
    assert report["policyLabelEligibilityChanged"] is False
    assert report["ppoEnablement"] is False
    assert report["promotionAuthority"] == "none"
    assert report["inputManifestSha256"] == file_sha256(labels_dir / "manifest.json")
    assert report["selectionManifestSha256"] == file_sha256(selection_path)
    assert report["reportHash"] == identity_hash({key: value for key, value in report.items()
                                                    if key != "reportHash"})
    assert confidence_audit.verify_macro_label_confidence_audit(labels_dir=labels_dir,
        selection_path=selection_path, report_path=output) == report
    tampered = tmp_path / "tampered-confidence.json"
    corrupt = {**report, "policyLabelEligibilityChanged": True}
    corrupt["reportHash"] = identity_hash({key: value for key, value in corrupt.items()
                                           if key != "reportHash"})
    tampered.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="fresh input and implementation recomputation"):
        confidence_audit.verify_macro_label_confidence_audit(labels_dir=labels_dir,
            selection_path=selection_path, report_path=tampered)
    with pytest.raises(ValueError, match="immutable"):
        confidence_audit.audit_macro_label_confidence(labels_dir=labels_dir,
            selection_path=selection_path, output=output)
