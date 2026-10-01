from __future__ import annotations

import json
import gzip
import hashlib

import pytest
import torch

import ptcg_lab.learning_mind.promotion_evidence as promotion_evidence
from ptcg_lab.learning_mind.dataset_v1 import file_sha256
from ptcg_lab.learning_mind.encoding import encode_decision
from ptcg_lab.learning_mind.evaluation import PROMOTION_MATCHUPS, promotion_seed
from ptcg_lab.learning_mind.model import StrategyTransformerV1
from ptcg_lab.learning_mind.notifications import AtomicRollbackRegistry
from ptcg_lab.learning_mind.policy_evaluation import _single_action
from ptcg_lab.learning_mind.promotion_evidence import (_reproduce_replay_actions,
    issue_promotion_evidence,
    reissue_promotion_evidence_report, verify_promotion_evidence_report)
from ptcg_lab.learning_mind.schema import identity_hash
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker

ENGINE_BUILD_HASH = None
ENGINE_FIXTURES_BY_SEED = {}


class _ReplayEngine:
    """Test double: serves saved source traces without starting the engine or simulating games."""
    def __init__(self, _root, timeout=300):
        self.cursors = {}
        self.expected = None
        self.frame_index = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def request(self, method, params=None):
        params = params or {}
        if method == "health":
            return {"engineBuildHash": ENGINE_BUILD_HASH}
        if method == "reset":
            seed = params["seed"]
            index = self.cursors.get(seed, 0)
            self.expected = ENGINE_FIXTURES_BY_SEED[seed][index]
            self.cursors[seed] = index + 1
            self.frame_index = 0
            return {}
        if method == "observe":
            return self.expected["frames"][self.frame_index]["observations"][params["playerId"]]
        if method == "step":
            frame = self.expected["frames"][self.frame_index]
            assert frame["action"]["id"] == params["actionId"]
            self.frame_index += 1
            return {}
        if method == "replay":
            return self.expected
        raise AssertionError(f"unexpected engine method {method}")


@pytest.fixture(autouse=True)
def use_saved_replay_engine(monkeypatch):
    monkeypatch.setattr(promotion_evidence, "EngineClient", _ReplayEngine)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


def _promotion_sources(root, observation, *, complete=True):
    global ENGINE_BUILD_HASH
    ENGINE_FIXTURES_BY_SEED.clear()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / "candidate.pt"
    control = root / "control.pt"
    torch.manual_seed(19)
    model = StrategyTransformerV1()
    model.eval()
    policy_identity = {"schemaVersion": 1, "featureSchemaHash": identity_hash("features"),
                       "deckManifestHash": identity_hash("decks")}
    torch.save({"kind": "StrategyTransformerV1-supervised",
                "identity": policy_identity, "model": model.state_dict(),
                "fixtureTag": "candidate"}, candidate)
    torch.save({"kind": "StrategyTransformerV1-supervised",
                "identity": policy_identity, "model": model.state_dict(),
                "fixtureTag": "control"}, control)
    opponent = root / "blind-opponent.pt"
    torch.save({"kind": "StrategyTransformerV1-supervised",
                "identity": policy_identity, "model": model.state_dict(),
                "fixtureTag": "blind-opponent"}, opponent)
    candidate_hash, control_hash = file_sha256(candidate), file_sha256(control)
    opponent_hash = file_sha256(opponent)
    opponent_policy_id = "blind-policy-v1"
    policy_set_body = {"schemaVersion": 1, "kind": "learning-mind-opponent-policy-set-v1",
        "policies": [{"policyId": opponent_policy_id, "family": "blind-family",
            "checkpointPath": str(opponent.resolve()), "checkpointSha256": opponent_hash}]}
    policy_set = {**policy_set_body, "setHash": identity_hash(policy_set_body)}
    identity = {key: identity_hash(key) for key in (
        "deckManifestHash", "featureSchemaHash", "trackerRulesHash",
        "cardMetadataHash", "actionEquivalenceHash", "schedulerIdentity")}
    identity["opponentPolicySetHash"] = policy_set["setHash"]
    identity["evaluationRunnerSha256"] = promotion_evidence.promotion_runner_sha256()
    identity["engineBuildHash"] = identity_hash("test-engine-build")
    ENGINE_BUILD_HASH = identity["engineBuildHash"]
    identity["policyIdentityHash"] = identity_hash(policy_identity)
    identity_path = _write_json(root / "promotion-identity.json", {
        "schemaVersion": 1, "kind": "learning-mind-promotion-identity-v1",
        "identity": identity, "candidateCheckpointSha256": candidate_hash,
        "controlCheckpointSha256": control_hash,
        "identityHash": identity_hash(identity)})

    training_source = _write_json(root / "training-sources.json",
        {"trainingOpponentPolicyFamilies": ["training-family"]})
    family_body = {"schemaVersion": 1, "kind": "learning-mind-training-policy-families-v1",
        "trainingOpponentPolicyFamilies": ["training-family"],
        "sourceArtifacts": [{"path": str(training_source.resolve()),
                              "sha256": file_sha256(training_source)}]}
    families_path = _write_json(root / "training-families.json",
        {**family_body, "manifestHash": identity_hash(family_body)})

    identity_hash_value = identity_hash(identity)
    action_by_seat = {}
    actor_view_by_seat = {}
    for seat in (0, 1):
        actor_view = dict(observation, playerId=seat, decisionPlayer=seat)
        tracker = ObservableHistoryTracker(seat)
        encoded = encode_decision(actor_view, tracker.update(actor_view))
        chosen_class, _ = _single_action(model, encoded)
        action_by_seat[seat] = encoded.action_classes[chosen_class].actions[0]
        actor_view_by_seat[seat] = actor_view

    candidate_rows, control_rows = [], []
    for own, opponent in sorted(PROMOTION_MATCHUPS):
        if not complete and (own, opponent) == ("crustle", "crustle"):
            continue
        for index in range(100):
            pair_id = f"{own}|{opponent}|{index}"
            seat, first = index % 2, (index + 1) % 2
            position = f"promotion-matrix-v1|{own}|{opponent}|seat-{seat}|first-{first}"
            seed = promotion_seed(position, index)
            actor_view = actor_view_by_seat[seat]
            decks = [None, None]
            decks[seat], decks[1 - seat] = own, opponent
            chosen_action = action_by_seat[seat]
            opponent_seat = 1 - seat
            opponent_view = actor_view_by_seat[opponent_seat]
            opponent_action = action_by_seat[opponent_seat]
            for checkpoint_hash, score, target in (
                    (candidate_hash, 1, candidate_rows), (control_hash, 0, control_rows)):
                game_id = f"{checkpoint_hash[:8]}-{pair_id}"
                learner_policy_id = f"{target is candidate_rows and 'candidate' or 'control'}:{checkpoint_hash}"
                policy_ids = [None, None]
                policy_hashes = [None, None]
                policy_ids[seat], policy_ids[opponent_seat] = learner_policy_id, opponent_policy_id
                policy_hashes[seat], policy_hashes[opponent_seat] = checkpoint_hash, opponent_hash
                replay = {"id": game_id, "status": "finished", "seed": seed,
                    "firstPlayer": first, "engineBuildHash": identity["engineBuildHash"],
                    "decks": decks,
                    "policyIdsBySeat": policy_ids, "policyHashesBySeat": policy_hashes,
                    "outcome": {"winner": seat if score == 1 else 1 - seat, "reason": "rules-terminal"},
                    "frames": [{"actor": seat, "action": chosen_action,
                        "decisionIndex": 0,
                        "observations": [actor_view if view_seat == seat else None for view_seat in (0, 1)]},
                        {"actor": opponent_seat, "action": opponent_action, "decisionIndex": 1,
                        "observations": [opponent_view if view_seat == opponent_seat else None
                                         for view_seat in (0, 1)]}]}
                replay_path = root / "replays" / f"{game_id}.json.gz"
                replay_path.parent.mkdir(parents=True, exist_ok=True)
                replay_bytes = (json.dumps(replay, separators=(",", ":")) + "\n").encode()
                with replay_path.open("wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
                        archive.write(replay_bytes)
                target.append({"pairId": pair_id, "gameId": game_id,
                    "status": "finished", "score": score, "seed": seed,
                    "seedNamespace": "promotion", "scheduleIndex": index,
                    "ownArchetype": own, "opponentArchetype": opponent,
                    "learnerSeat": seat, "firstPlayer": first,
                    "decks": decks,
                    "opponentPolicyFamily": "blind-family",
                    "opponentPolicyId": opponent_policy_id,
                    "schedulerIdentity": identity["schedulerIdentity"],
                    "checkpointSha256": checkpoint_hash,
                    "evaluationIdentityHash": identity_hash_value,
                    "replay": {"path": str(replay_path.resolve()),
                        "sha256": file_sha256(replay_path),
                        "decodedSha256": hashlib.sha256(replay_bytes).hexdigest()}})
                ENGINE_FIXTURES_BY_SEED.setdefault(seed, []).append(replay)
    results_body = {"schemaVersion": 1,
        "kind": "learning-mind-matched-promotion-results-v1",
        "candidateCheckpointSha256": candidate_hash,
        "controlCheckpointSha256": control_hash,
        "evaluationIdentityHash": identity_hash_value,
        "runnerImplementationSha256": identity["evaluationRunnerSha256"],
        "opponentPolicySet": policy_set,
        "candidateRecords": candidate_rows, "controlRecords": control_rows}
    results_path = _write_json(root / "matched-results.json",
        {**results_body, "reportHash": identity_hash(results_body)})

    strategy_path = _write_json(root / "strategy-evaluation.json", {
        "checkpointSha256": candidate_hash,
        "evaluationIdentityHash": identity_hash_value,
        "targetProbeWin": True, "severityThreeRegression": False,
        "severityThreeProbeCoverage": "sufficient", "automaticPromotion": False})
    return candidate, control, identity_path, results_path, families_path, strategy_path


def test_promotion_receipt_recomputes_replays_and_requires_explicit_registry_approval(tmp_path, observation):
    candidate, control, identity, results, families, strategy = _promotion_sources(tmp_path, observation)
    report_path = tmp_path / "receipts" / "promotion.json"
    report, evidence = issue_promotion_evidence(candidate_checkpoint=candidate,
        control_checkpoint=control, identity_path=identity, results_path=results,
        training_families_path=families, strategy_report_path=strategy,
        output=report_path, human_approved=True)
    assert report["matrixCriteriaPassed"] is True
    assert report["promotionCriteriaPassed"] is False
    assert report["actionReproductionVerified"] is True
    assert report["engineReproductionVerified"] is True
    assert report["actionReproduction"]["candidate"] == 2500
    assert report["actionReproduction"]["control"] == 2500
    assert report["engineReproductionVerified"] is True
    assert report["engineReproduction"]["candidate"] == 2500
    assert report["engineReproduction"]["control"] == 2500
    assert report["opponentFamilyProvenanceVerified"] is True
    assert report["opponentPolicyReproduction"] == {"passed": True, "policies": 1}
    assert report["promotionStrategyProbes"]["targetProbeWin"] is False
    assert report["notReadyReasons"] == ["promotion-seed strategy probes did not pass their evidence gate"]
    assert evidence is None
    assert verify_promotion_evidence_report(report_path) == report
    reissued_report, reissued_evidence = reissue_promotion_evidence_report(report_path)
    assert reissued_report == report
    assert reissued_evidence is None

    registry = AtomicRollbackRegistry(str(control.resolve()))
    registry.queue(str(candidate.resolve()))
    with pytest.raises(PermissionError, match="verifier-issued evidence"):
        registry.promote(human_approved=True, evidence=evidence)
    assert registry.trusted_checkpoint == str(control.resolve())


def test_promotion_receipt_is_not_issued_for_incomplete_matrix_or_without_approval(tmp_path, observation):
    candidate, control, identity, results, families, strategy = _promotion_sources(tmp_path, observation, complete=False)
    report, evidence = issue_promotion_evidence(candidate_checkpoint=candidate,
        control_checkpoint=control, identity_path=identity, results_path=results,
        training_families_path=families, strategy_report_path=strategy,
        output=tmp_path / "incomplete.json", human_approved=True)
    assert report["promotionCriteriaPassed"] is False
    assert evidence is None

    candidate, control, identity, results, families, strategy = _promotion_sources(tmp_path / "unapproved", observation)
    report, evidence = issue_promotion_evidence(candidate_checkpoint=candidate,
        control_checkpoint=control, identity_path=identity, results_path=results,
        training_families_path=families, strategy_report_path=strategy,
        output=tmp_path / "unapproved.json", human_approved=False)
    assert report["promotionCriteriaPassed"] is False
    assert evidence is None


def test_promotion_receipt_rejects_tampered_sources_and_wrong_candidate(tmp_path, observation):
    candidate, control, identity, results, families, strategy = _promotion_sources(tmp_path, observation)
    with pytest.raises(ValueError, match="different artifacts"):
        issue_promotion_evidence(candidate_checkpoint=candidate, control_checkpoint=candidate,
            identity_path=identity, results_path=results, training_families_path=families,
            strategy_report_path=strategy, output=tmp_path / "wrong.json", human_approved=True)

    report, evidence = issue_promotion_evidence(candidate_checkpoint=candidate,
        control_checkpoint=control, identity_path=identity, results_path=results,
        training_families_path=families, strategy_report_path=strategy,
        output=tmp_path / "valid.json", human_approved=True)
    assert evidence is None
    strategy.write_text(strategy.read_text().replace('"targetProbeWin": true', '"targetProbeWin": false'))
    with pytest.raises(ValueError, match="source artifact changed"):
        verify_promotion_evidence_report(tmp_path / "valid.json")
    with pytest.raises(ValueError, match="source artifact changed"):
        reissue_promotion_evidence_report(tmp_path / "valid.json")


def test_policy_action_reproduction_uses_actor_view_and_rejects_wrong_checkpoint_action(tmp_path, observation):
    torch.manual_seed(7)
    model = StrategyTransformerV1().eval()
    tracker = ObservableHistoryTracker(0)
    encoded = encode_decision(observation, tracker.update(observation))
    chosen, _ = _single_action(model, encoded)
    selected_action = encoded.action_classes[chosen].actions[0]
    other_action = encoded.action_classes[(chosen + 1) % len(encoded.action_classes)].actions[0]
    replay = {"frames": [{"actor": 0, "action": selected_action,
        "observations": [observation, {"hidden": "must-not-be-read"}]}]}
    path = tmp_path / "policy-replay.json.gz"

    def save():
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
                archive.write(json.dumps(replay).encode())

    save()
    assert _reproduce_replay_actions(path, model, "candidate", 0, {})[0] == 1
    replay["frames"][0]["action"] = other_action
    save()
    with pytest.raises(ValueError, match="does not reproduce"):
        _reproduce_replay_actions(path, model, "candidate", 0, {})
