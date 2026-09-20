from __future__ import annotations

import copy
import json

import pytest

from ptcg_lab import teaching
from ptcg_lab.guides import import_guide, retrieve
from ptcg_lab.storage import Store, digest
from conftest import make_replay


def position_and_curriculum(tmp_path, observation, *, partition="train", evaluation=False, audited=True):
    store = Store(tmp_path / "data")
    replay = make_replay(observation)
    replay["trainingEligible"] = audited
    if evaluation: replay["evaluationExperiment"] = "held-out"
    store.save_replay(replay)
    store.put("positions", "real", {"id": "real", "title": "Preserve the answer", "playerId": 0,
              "sourceReplayId": replay["id"], "decisionIndex": 0, "engineVersion": replay["engineVersion"],
              "observation": observation})
    directory = tmp_path / "research"
    directory.mkdir()
    (directory / "curriculum.json").write_text(json.dumps({"families": [{"id": "crustle-resources",
        "partition": partition, "source": {"author": "Expert", "page": 5, "sourceHash": "hash"}}]}))
    return store


def test_review_needs_verified_source_concrete_position_and_legal_actions(tmp_path, observation):
    store = position_and_curriculum(tmp_path, observation)
    draft = teaching.from_position(store, tmp_path, "real", "crustle-resources")
    assert draft["reviewStatus"] == "draft" and not draft["trainingEligible"]
    assert draft["source"]["page"] == 5
    reviewed = teaching.review(store, draft["id"], review_status="reviewed", acceptable_action_ids=["action-0", "action-1"],
                              reasoning="Either works unless their last recovery remains.", critical_resources="Last recovery")
    assert reviewed["trainingEligible"]
    altered = copy.deepcopy(reviewed)
    altered["observation"]["turn"] += 1
    assert not teaching.eligible(altered)
    for arguments in ({"acceptable_action_ids": ["nonlegal"], "reasoning": "An invented move"},
                      {"acceptable_action_ids": [], "reasoning": "No move"},
                      {"acceptable_action_ids": ["action-0"], "reasoning": " "},
                      {"acceptable_action_ids": ["action-0"], "rejected_action_ids": ["action-0"], "reasoning": "Contradictory"}):
        with pytest.raises(ValueError):
            teaching.review(store, draft["id"], review_status="reviewed", **arguments)


@pytest.mark.parametrize("partition,evaluation,audited", [("test", False, True), ("validation", False, True),
                                                           ("train", True, True), ("train", False, False)])
def test_partitions_benchmarks_and_unaudited_rules_cannot_be_training(tmp_path, observation, partition, evaluation, audited):
    store = position_and_curriculum(tmp_path, observation, partition=partition, evaluation=evaluation, audited=audited)
    draft = teaching.from_position(store, tmp_path, "real", "crustle-resources")
    reviewed = teaching.review(store, draft["id"], review_status="reviewed", acceptable_action_ids=["action-0"], reasoning="A sound line")
    assert not reviewed["trainingEligible"]
    assert reviewed["partition"] == ("test" if evaluation else partition)


def test_prose_and_modified_positions_do_not_become_legal_demonstrations(tmp_path, observation):
    store = position_and_curriculum(tmp_path, observation)
    with pytest.raises(ValueError, match="existing curriculum"):
        teaching.from_position(store, tmp_path, "real", "invented")
    position = store.get("positions", "real")
    position["observation"]["legalActions"].append({"id": "invented", "label": "Win"})
    store.put("positions", "real", position)
    with pytest.raises(ValueError, match="original engine frame"):
        teaching.from_position(store, tmp_path, "real", "crustle-resources")
    assert not teaching.eligible({"reviewStatus": "reviewed", "partition": "train", "text": "Always attack"})


def test_review_queue_is_bounded_and_hides_full_observations(tmp_path):
    store = Store(tmp_path)
    for index in range(15):
        identifier = f"item-{index:02d}"
        store.put("teaching", identifier, {"id": identifier, "reviewStatus": "draft", "reason": "human-bookmark", "observation": {"private": "own-hand"}})
    store.put("teaching", "priority", {"id": "priority", "reviewStatus": "draft", "reason": "rules-issue", "observation": {}})
    result = teaching.queue(store, limit=100)
    assert len(result) == 10 and result[0]["id"] == "priority"
    assert all("observation" not in item for item in result)


def test_video_transcript_attribution_is_retained_without_training_labels(tmp_path):
    text = tmp_path / "transcript.txt"
    text.write_text("Preserve the last recovery card against Crustle.")
    store = Store(tmp_path / "data")
    guide = import_guide(store, text, title="Expert matchup", author="Expert", source="user-supplied transcript",
                         matchup="Crustle", deck_version="frozen-list", media_type="video-transcript", timestamp="12:30")
    found = retrieve(store, "Crustle recovery")
    assert not guide["trainingEligible"]
    assert guide["mediaType"] == "video-transcript"
    assert found[0]["timestamp"] == "12:30" and found[0]["author"] == "Expert"
    assert found[0]["deckVersion"] == "frozen-list" and found[0]["contentHash"]
