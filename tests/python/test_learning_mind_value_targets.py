from __future__ import annotations

import copy
import json

import pytest
import torch

from ptcg_lab.storage import Store
from ptcg_lab.learning_mind.dataset_v1 import stable_split
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.learning_mind.training import train_supervised
from ptcg_lab.learning_mind.value_evaluation import evaluate_value_head
from ptcg_lab.learning_mind.value_targets import (build_value_target_dataset,
    load_value_target_dataset, training_records)
from test_learning_mind_representation import observation


def _source(tmp_path, *, training_eligible=True, draw=False):
    experimental = tmp_path / "experimental"
    store = Store(experimental)
    first_view = observation()
    second_view = copy.deepcopy(first_view)
    second_view["playerId"] = 1
    second_view["decisionPlayer"] = 1
    second_view["players"][0]["hand"] = []
    second_view["players"][1]["hand"] = first_view["players"][0]["hand"]
    private_view = copy.deepcopy(second_view)
    private_view["players"][1]["hand"] = [{"id": "SECRET", "name": "Private Card", "kind": "trainer"}]
    private_view["players"][1]["handCount"] = 1
    replay = {"id": "game-1", "status": "finished", "dataTier": "experimental",
        "experimentalLearning": True, "trainingEligible": training_eligible,
        "outcome": {"winner": None, "reason": "rules-draw"} if draw else
                   {"winner": 0, "reason": "rules-prizes"},
        "frames": [
            {"actor": 0, "decisionIndex": 0, "action": {"id": "a"},
             "observations": [first_view, private_view]},
            {"actor": 1, "decisionIndex": 1, "action": {"id": "b"},
             "observations": [first_view, second_view]},
        ]}
    store.put("replays", "game-1", replay)
    identity = {"identityHash": "frozen-test"}
    source = {"settings": {"identity": identity},
        "replays": [{"id": "game-1", "familyId": "family-1", "status": "finished"}]}
    source["manifestHash"] = identity_hash(source)
    source_path = tmp_path / "source-manifest.json"
    source_path.write_text(json.dumps(source))
    return experimental, identity, source_path


def test_value_targets_are_terminal_outcomes_actor_view_only_and_game_disjoint(tmp_path):
    experimental, identity, source = _source(tmp_path)
    output = tmp_path / "value-targets"
    manifest = build_value_target_dataset(output=output, experimental_root=experimental,
        source_manifest_path=source, identity=identity)
    loaded_manifest, rows = load_value_target_dataset(output, identity=identity)
    assert loaded_manifest == manifest
    assert [row["valueTarget"] for row in rows] == [1.0, -1.0]
    assert all(row["valueLabelSource"] == "completed-self-play-outcome" for row in rows)
    assert all("SECRET" not in json.dumps(row["observation"]) for row in rows)
    records = training_records(rows, split=rows[0]["split"])
    assert len(records) == 2 and {row["valueTarget"] for row in records} == {-1.0, 1.0}
    assert not any("policyLabelSource" in row for row in records)
    manifest_path = output / "manifest.json"
    altered = json.loads(manifest_path.read_text())
    altered["games"][0]["decisionRows"] += 1
    altered["manifestHash"] = identity_hash({key: value for key, value in altered.items()
                                               if key != "manifestHash"})
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="game summary differs"):
        load_value_target_dataset(output, identity=identity)
    with pytest.raises(ValueError, match="immutable"):
        build_value_target_dataset(output=output, experimental_root=experimental,
            source_manifest_path=source, identity=identity)


def test_draws_map_to_zero_and_ineligible_games_are_excluded(tmp_path):
    experimental, identity, source = _source(tmp_path, draw=True)
    output = tmp_path / "draw-value-targets"
    manifest = build_value_target_dataset(output=output, experimental_root=experimental,
        source_manifest_path=source, identity=identity)
    _, rows = load_value_target_dataset(output, identity=identity)
    assert [row["valueTarget"] for row in rows] == [0.0, 0.0]
    experimental, identity, source = _source(tmp_path / "excluded", training_eligible=False)
    output = tmp_path / "excluded" / "ineligible-value-targets"
    manifest = build_value_target_dataset(output=output, experimental_root=experimental,
        source_manifest_path=source, identity=identity)
    assert manifest["rows"] == 0
    assert manifest["excludedGames"] == [{"replayId": "game-1", "reason": "not-explicitly-training-eligible"}]
    assert load_value_target_dataset(output, identity=identity)[1] == []


def test_value_targets_reject_manifest_drift_and_nonrules_draw(tmp_path):
    experimental, identity, source = _source(tmp_path, draw=True)
    source_record = json.loads(source.read_text())
    source_record["settings"]["identity"] = {"identityHash": "other"}
    source_record["manifestHash"] = identity_hash({key: value for key, value in source_record.items()
                                                    if key != "manifestHash"})
    source.write_text(json.dumps(source_record))
    with pytest.raises(ValueError, match="runtime identity"):
        build_value_target_dataset(output=tmp_path / "identity-failure", experimental_root=experimental,
            source_manifest_path=source, identity=identity)

    experimental, identity, source = _source(tmp_path / "bad-draw", draw=True)
    store = Store(experimental)
    replay = store.get("replays", "game-1")
    replay["outcome"]["reason"] = "unknown-terminal"
    store.put("replays", "game-1", replay)
    output = tmp_path / "bad-draw" / "bad-value-targets"
    with pytest.raises(ValueError, match="genuine rules draw"):
        build_value_target_dataset(output=output, experimental_root=experimental,
            source_manifest_path=source, identity=identity)
    assert not output.exists()


def test_value_head_evaluation_is_hash_bound_and_heldout_only(tmp_path):
    experimental, identity, source = _source(tmp_path)
    source_record = json.loads(source.read_text())
    heldout_family = next(f"heldout-{index}" for index in range(1000)
                          if stable_split(f"heldout-{index}") == "heldout")
    train_family = next(f"train-{index}" for index in range(1000)
                        if stable_split(f"train-{index}") == "train")
    source_record["replays"][0]["familyId"] = heldout_family
    store = Store(experimental)
    training_replay = store.get("replays", "game-1")
    training_replay.update({"id": "game-2", "outcome": {"winner": 1, "reason": "rules-prizes"}})
    store.put("replays", "game-2", training_replay)
    source_record["replays"].append({"id": "game-2", "familyId": train_family, "status": "finished"})
    source_record["manifestHash"] = identity_hash({key: value for key, value in source_record.items()
                                                    if key != "manifestHash"})
    source.write_text(json.dumps(source_record))

    dataset = tmp_path / "train-and-heldout-values"
    manifest = build_value_target_dataset(output=dataset, experimental_root=experimental,
        source_manifest_path=source, identity=identity)
    _, rows = load_value_target_dataset(dataset, identity=identity)
    training_rows = training_records(rows, "train")
    assert training_rows
    checkpoint = tmp_path / "value-head.pt"
    train_supervised(training_rows, checkpoint, identity, batch_size=2,
        dataset_manifest_hash="d" * 64, value_dataset_manifest_hash=manifest["manifestHash"])

    report = evaluate_value_head(dataset_dir=dataset, checkpoint_path=checkpoint,
        output=tmp_path / "value-evaluation.json", split="heldout")
    assert report["split"] == "heldout" and report["perRecord"]["records"] == 2
    assert report["perUniquePosition"]["records"] >= 1
    assert report["calibration"]["binCount"] == 10
    assert len(report["evaluatorSha256"]) == 64
    assert report["checkpointSha256"] and report["reportHash"]
    mismatched_checkpoint = tmp_path / "mismatched-value-checkpoint.pt"
    checkpoint_record = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_record["trainingConfig"]["valueDatasetManifestHash"] = "0" * 64
    torch.save(checkpoint_record, mismatched_checkpoint)
    with pytest.raises(ValueError, match="value-dataset hash mismatch"):
        evaluate_value_head(dataset_dir=dataset, checkpoint_path=mismatched_checkpoint,
            output=tmp_path / "mismatched-value-evaluation.json", split="heldout")
    with pytest.raises(ValueError, match="never training"):
        evaluate_value_head(dataset_dir=dataset, checkpoint_path=checkpoint,
            output=tmp_path / "forbidden-training-evaluation.json", split="train")
