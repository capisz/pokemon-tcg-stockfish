from __future__ import annotations

import copy
import json
from ptcg_lab.storage import Store
from ptcg_lab.learning_mind.dataset_v1 import (build_strategy_probe_dataset,
    load_strategy_probe_dataset, stable_split)
from ptcg_lab.learning_mind.experiment import (evaluate_strategy_probes,
    summarize_strategy_probe_decisions)
from ptcg_lab.learning_mind.model import StrategyTransformerV1
from ptcg_lab.learning_mind.schema import IdentityManifest, identity_hash
from test_learning_mind_representation import observation
from research.strategy_baseline.probes import PROBES


def _source_fixture(tmp_path):
    experimental = tmp_path / "experimental"
    replay_id = "heldout-replay"
    family = next(f"family-{index}" for index in range(100)
                  if stable_split(f"family-{index}") == "heldout")
    own_zero = observation()
    own_one = copy.deepcopy(own_zero)
    own_one["playerId"] = 1
    own_one["decisionPlayer"] = 1
    opposite_secret = copy.deepcopy(own_zero)
    opposite_secret["players"][1]["hand"] = [{"id": "SHH", "name": "HIDDEN FROM ACTOR", "kind": "pokemon"}]
    replay = {"id": replay_id, "dataTier": "experimental", "status": "finished",
        "frames": [
            {"decisionIndex": 0, "actor": 0, "observations": [own_zero, opposite_secret]},
            {"decisionIndex": 1, "actor": 1, "observations": [opposite_secret, own_one]},
        ]}
    Store(experimental).put("replays", replay_id, replay)
    identity = IdentityManifest.create(engine_build_hash="engine", deck_manifests={}, card_metadata={})
    source_value = {"settings": {"identity": identity.record()}, "replays": [
        {"id": replay_id, "familyId": family},
        {"id": "training-replay", "familyId": next(f"train-{index}" for index in range(100)
          if stable_split(f"train-{index}") == "train")},
    ]}
    source_value["manifestHash"] = identity_hash(source_value)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(json.dumps(source_value))
    return experimental, source_manifest, identity


def test_probe_dataset_keeps_every_heldout_actor_view_and_excludes_actions(tmp_path):
    experimental, source, identity = _source_fixture(tmp_path)
    output = tmp_path / "probe-corpus"
    manifest = build_strategy_probe_dataset(output=output, experimental_root=experimental,
        source_dataset_manifest=source, identity=identity)
    loaded_manifest, rows = load_strategy_probe_dataset(output, identity=identity.record())
    assert loaded_manifest == manifest
    assert manifest["sourceGameCount"] == 1
    assert manifest["actorDecisionRows"] == 2
    assert {row["sourceDecisionIndex"] for row in rows} == {0, 1}
    assert {row["gameSideKey"] for row in rows} == {"heldout-replay:0", "heldout-replay:1"}
    assert all(row["split"] == "heldout" for row in rows)
    assert all("policyAction" not in row and "chosenAction" not in row for row in rows)
    assert "HIDDEN FROM ACTOR" not in (output / "rows.jsonl").read_text()


def test_probe_corpus_loader_rejects_sparse_supervised_dataset(tmp_path):
    directory = tmp_path / "sparse"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"kind": "learning-mind-supervised-v1"}))
    identity = IdentityManifest.create(engine_build_hash="engine", deck_manifests={}, card_metadata={})
    try:
        load_strategy_probe_dataset(directory, identity=identity.record())
    except ValueError as error:
        assert "dedicated full-decision corpus" in str(error)
    else:
        raise AssertionError("sparse policy-label dataset was accepted as a probe corpus")


def test_probe_builder_rejects_source_identity_drift(tmp_path):
    experimental, source, identity = _source_fixture(tmp_path)
    value = json.loads(source.read_text())
    value["settings"]["identity"]["engine_build_hash"] = "different-engine"
    value["manifestHash"] = identity_hash({key: item for key, item in value.items()
                                           if key != "manifestHash"})
    source.write_text(json.dumps(value))
    try:
        build_strategy_probe_dataset(output=tmp_path / "probe-corpus", experimental_root=experimental,
            source_dataset_manifest=source, identity=identity)
    except ValueError as error:
        assert "source replay identity mismatch" in str(error)
    else:
        raise AssertionError("probe builder accepted a source manifest from a different engine identity")


def test_probe_summary_uses_first_qualifying_decision_per_side_and_marks_n_below_20():
    decisions = [{"probeId": "crustle-fan-active-kangaskhan", "gameSideKey": f"g{i}:0",
        "positionHash": f"p{i}", "decisionIndex": 1, "modelAction": {}, "heuristicAction": {},
        "modelAdherent": True, "heuristicAdherent": False} for i in range(19)]
    decisions.append({**decisions[0], "decisionIndex": 0, "modelAdherent": False,
                      "heuristicAdherent": True, "positionHash": "earliest"})
    summary = summarize_strategy_probe_decisions(decisions)
    by_id = {row["probeId"]: row for row in summary["probes"]}
    target = by_id["crustle-fan-active-kangaskhan"]
    assert target["headline"]["eligibleGameSides"] == 19
    assert target["headline"]["modelAdherent"] == 18
    assert target["headline"]["heuristicAdherent"] == 1
    assert target["headline"]["status"] == "insufficient"
    assert target["headline"]["modelWilson95"]["low"] is not None
    assert target["allQualifying"]["decisions"] == 20
    assert summary["targetProbeWin"] is False


def test_probe_evaluator_reports_registered_coverage_and_stays_closed_without_triggers(tmp_path):
    experimental, source, identity = _source_fixture(tmp_path)
    corpus = tmp_path / "probe-corpus"
    manifest = build_strategy_probe_dataset(output=corpus, experimental_root=experimental,
        source_dataset_manifest=source, identity=identity)
    model = StrategyTransformerV1()
    model.eval()
    result = evaluate_strategy_probes(corpus, model, identity.record(),
                                      manifest["sourceDatasetManifestSha256"])
    assert result["evaluationSplit"] == "heldout"
    assert result["actorViewOnly"] is True
    assert result["actorDecisionRows"] == 2
    assert result["probeCount"] == len(result["probes"]) == len(PROBES)
    assert all(probe["headline"]["status"] == "insufficient" for probe in result["probes"])
    assert result["targetProbeWin"] is False
    assert result["severityThreeCoverage"] == "insufficient"
    try:
        evaluate_strategy_probes(corpus, model, identity.record(), "different-source-manifest")
    except ValueError as error:
        assert "different frozen source games" in str(error)
    else:
        raise AssertionError("probe evaluator accepted a corpus from a different source manifest")
