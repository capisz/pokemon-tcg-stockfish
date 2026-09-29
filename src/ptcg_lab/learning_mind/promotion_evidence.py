"""Source-bound verifier for human-approved, matched promotion evidence.

This module only issues an in-process capability after recomputing the matrix
from immutable inputs. It does not change a trusted checkpoint or run games.
"""

from __future__ import annotations

import json
import os
import gzip
import inspect
from pathlib import Path
import tempfile

import torch

from .dataset_v1 import file_sha256
from .audit import audit_replay
from ptcg_lab.engine import EngineClient, EngineError
from .evaluation import matched_promotion_matrix_decision
from .encoding import collate, encode_decision
from .model import StrategyTransformerV1, greedy_single_action_class
from .notifications import VerifiedPromotionEvidence, _VERIFIED_PROMOTION_TOKEN
from .schema import canonical_json, identity_hash
from .tracker import ObservableHistoryTracker


def promotion_runner_sha256() -> str:
    """Hash every local implementation that determines promotion evidence."""
    from . import audit, evaluation
    paths = {"promotionVerifier": Path(__file__),
        "replayAuditor": Path(audit.__file__),
        "matrixEvaluator": Path(evaluation.__file__),
        "engineClient": Path(inspect.getsourcefile(EngineClient))}
    return identity_hash({name: file_sha256(path) for name, path in sorted(paths.items())})

IDENTITY_FIELDS = frozenset({"engineBuildHash", "deckManifestHash", "featureSchemaHash",
    "trackerRulesHash", "cardMetadataHash", "actionEquivalenceHash", "schedulerIdentity",
    "opponentPolicySetHash", "evaluationRunnerSha256", "policyIdentityHash"})


def _read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} source artifact is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} source artifact must be a JSON object")
    return value


def _sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _source(path: Path, label: str) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} source artifact is missing")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _load_policy_checkpoint(path: Path, label: str) -> tuple[dict, StrategyTransformerV1]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"{label} checkpoint metadata is unreadable") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{label} checkpoint metadata is malformed")
    if checkpoint.get("kind") == "StrategyTransformerV1-supervised":
        identity = checkpoint.get("identity")
    elif checkpoint.get("kind") == "learning-mind-ppo-checkpoint-v1":
        experiment_identity = checkpoint.get("experimentIdentity")
        identity = (experiment_identity.get("learningMindIdentity")
                    if isinstance(experiment_identity, dict) else None)
    else:
        identity = None
    if not isinstance(identity, dict) or not identity:
        raise ValueError(f"{label} checkpoint does not bind a learning-mind policy identity")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError(f"{label} checkpoint has no StrategyTransformerV1 model state")
    model = StrategyTransformerV1()
    try:
        model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"{label} checkpoint model state is incompatible") from error
    model.eval()
    return identity, model


def _reproduce_replay_actions(path: Path, model: StrategyTransformerV1,
                              label: str, learner_seat: int,
                              prediction_cache: dict) -> tuple[int, list[dict]]:
    """Reproduce learner turns from that seat's observation history only."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            replay = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} replay cannot be loaded for policy reproduction") from error
    trackers = {0: ObservableHistoryTracker(0), 1: ObservableHistoryTracker(1)}
    reproduced = 0
    probe_decisions = []
    for frame in replay.get("frames", []):
        action = frame.get("action")
        if action is None:
            continue
        actor = frame.get("actor")
        if actor != learner_seat:
            continue
        observations = frame.get("observations")
        if type(actor) is not int or actor not in (0, 1) or not isinstance(observations, list):
            raise ValueError(f"{label} replay decision lacks an actor-visible observation")
        observation = observations[actor]
        if not isinstance(observation, dict) or observation.get("playerId") != actor:
            raise ValueError(f"{label} replay decision is not the actor's private view")
        snapshot = trackers[actor].update(observation)
        encoded = encode_decision(observation, snapshot)
        observed_action_id = action.get("id")
        observed_classes = [group for group in encoded.action_classes
            if any(candidate.get("id") == observed_action_id for candidate in group.actions)]
        if len(observed_classes) != 1:
            raise ValueError(f"{label} replay action is not represented exactly once")
        predicted_key = prediction_cache.get(encoded.identity)
        if predicted_key is None:
            tensors = {key: torch.as_tensor(value) for key, value in collate([encoded]).items()}

            def logits_for(chosen):
                selected_mask = torch.zeros_like(tensors["option_mask"])
                for index in chosen:
                    selected_mask[0, index] = True
                with torch.no_grad():
                    return model.policy_forward(**tensors, selected_mask=selected_mask)[0]

            selected_class = greedy_single_action_class(logits_for,
                action_count=len(encoded.action_classes),
                legality=lambda _chosen, index: 0 <= index < len(encoded.action_classes))
            predicted_key = encoded.action_classes[selected_class].semantic_key
            prediction_cache[encoded.identity] = predicted_key
        if observed_classes[0].semantic_key != predicted_key:
            raise ValueError(f"{label} checkpoint does not reproduce a recorded legal action")
        reproduced += 1
        from ptcg_lab.features import heuristic_action_score
        from research.strategy_baseline.probes import PROBES
        legal_actions = observation.get("legalActions") or []
        heuristic_action = max(legal_actions,
            key=lambda option: heuristic_action_score(option, observation))
        game_side = f"{replay.get('id')}:{actor}"
        decision_index = frame.get("decisionIndex", reproduced - 1)
        for probe in PROBES:
            model_adherent = probe.evaluate(observation, action)
            heuristic_adherent = probe.evaluate(observation, heuristic_action)
            if model_adherent is not None:
                if heuristic_adherent is None:
                    raise ValueError("probe eligibility differs between model and heuristic actions")
                probe_decisions.append({"probeId": probe.id, "gameSideKey": game_side,
                    "positionHash": identity_hash(observation), "decisionIndex": decision_index,
                    "modelAction": action, "heuristicAction": heuristic_action,
                    "modelAdherent": bool(model_adherent),
                    "heuristicAdherent": bool(heuristic_adherent)})
    if reproduced == 0:
        raise ValueError(f"{label} replay has no actions to reproduce")
    return reproduced, probe_decisions


def _reproduce_engine_game(engine: EngineClient, replay_path: Path, row: dict,
                           expected_engine_hash: str) -> int:
    """Replay the source action trace through the frozen engine and compare actor views."""
    try:
        with gzip.open(replay_path, "rt", encoding="utf-8") as source:
            expected = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("promotion replay cannot be loaded for engine reproduction") from error
    if (expected.get("seed") != row["seed"]
            or expected.get("firstPlayer") != row["firstPlayer"]
            or expected.get("decks") != row.get("decks")
            or expected.get("engineBuildHash") != expected_engine_hash):
        raise ValueError("promotion replay schedule does not match its frozen row")
    frames = expected.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("promotion replay has no engine action trace")
    try:
        engine.request("reset", {"seed": row["seed"], "decks": row["decks"],
                                  "firstPlayer": row["firstPlayer"]})
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("promotion replay contains a malformed frame")
            action = frame.get("action")
            if action is None:
                continue
            actor = frame.get("actor")
            observations = frame.get("observations")
            if type(actor) is not int or actor not in (0, 1) or not isinstance(observations, list):
                raise ValueError("promotion replay action lacks its actor observation")
            expected_view = observations[actor]
            actual_view = engine.request("observe", {"playerId": actor})
            if (not isinstance(expected_view, dict)
                    or canonical_json(actual_view) != canonical_json(expected_view)):
                raise ValueError("frozen engine actor observation differs from the recorded replay")
            engine.request("step", {"actionId": action.get("id")})
        actual = engine.request("replay")
    except EngineError as error:
        raise ValueError("frozen engine could not reproduce a recorded promotion game") from error
    if (actual.get("seed") != row["seed"] or actual.get("firstPlayer") != row["firstPlayer"]
            or actual.get("decks") != row.get("decks")
            or actual.get("engineBuildHash") != expected_engine_hash):
        raise ValueError("reproduced engine game has a different frozen identity")
    expected_actions = [frame["action"].get("id") for frame in frames
                        if isinstance(frame, dict) and frame.get("action") is not None]
    actual_frames = actual.get("frames")
    actual_actions = [frame.get("action", {}).get("id") for frame in actual_frames or []
                      if isinstance(frame, dict) and frame.get("action") is not None]
    if actual_actions != expected_actions:
        raise ValueError("frozen engine action trace differs from the source replay")
    if row["status"] == "finished":
        if actual.get("status") != "finished" or actual.get("outcome") != expected.get("outcome"):
            raise ValueError("frozen engine terminal outcome differs from the source replay")
    elif row["status"] == "truncated":
        if actual.get("status") != "truncated" or actual.get("outcome") is not None:
            raise ValueError("truncated promotion replay does not reproduce as an unfinished engine result")
    else:
        raise ValueError("engine reproduction is unavailable for errored promotion rows")
    return len(expected_actions)


def _verify_identity(path: Path, *, candidate_hash: str, control_hash: str,
                     candidate_policy_identity: dict, control_policy_identity: dict) -> tuple[dict, dict]:
    source = _source(path, "promotion identity")
    artifact = _read_object(path, "promotion identity")
    identity = artifact.get("identity")
    if (artifact.get("schemaVersion") != 1
            or artifact.get("kind") != "learning-mind-promotion-identity-v1"
            or not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS
            or any(not _sha256(identity.get(key)) for key in IDENTITY_FIELDS)
            or identity["policyIdentityHash"] != identity_hash(candidate_policy_identity)
            or identity["evaluationRunnerSha256"] != promotion_runner_sha256()
            or candidate_policy_identity != control_policy_identity
            or artifact.get("candidateCheckpointSha256") != candidate_hash
            or artifact.get("controlCheckpointSha256") != control_hash
            or artifact.get("identityHash") != identity_hash(identity)):
        raise ValueError("promotion identity source is malformed or has a bad hash")
    return identity, source


def _verify_training_families(path: Path) -> tuple[list[str], list[dict], dict]:
    source = _source(path, "training-family manifest")
    artifact = _read_object(path, "training-family manifest")
    families = artifact.get("trainingOpponentPolicyFamilies")
    sources = artifact.get("sourceArtifacts")
    body = {key: value for key, value in artifact.items() if key != "manifestHash"}
    if (artifact.get("schemaVersion") != 1
            or artifact.get("kind") != "learning-mind-training-policy-families-v1"
            or not isinstance(families, list) or not families
            or any(not isinstance(item, str) or not item for item in families)
            or len(set(families)) != len(families)
            or not isinstance(sources, list) or not sources
            or artifact.get("manifestHash") != identity_hash(body)):
        raise ValueError("training-family evidence manifest is malformed or has a bad hash")
    verified_sources = []
    seen_paths = set()
    observed_families = set()

    def collect_families(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"opponentPolicyFamily", "trainingOpponentPolicyFamilies"}:
                    if isinstance(child, str) and child:
                        observed_families.add(child)
                    elif isinstance(child, list):
                        observed_families.update(item for item in child
                            if isinstance(item, str) and item)
                collect_families(child)
        elif isinstance(value, list):
            for child in value:
                collect_families(child)

    for item in sources:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not isinstance(item.get("path"), str) or not _sha256(item.get("sha256"))):
            raise ValueError("training-family source list is malformed")
        item_path = Path(item["path"]).resolve()
        actual = _source(item_path, "training-family dependency")
        if (actual["sha256"] != item["sha256"] or actual["path"] in seen_paths):
            raise ValueError("training-family source identity mismatch or duplicate")
        seen_paths.add(actual["path"])
        try:
            source_value = json.loads(item_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("training-family dependency must be a readable JSON artifact") from error
        collect_families(source_value)
        verified_sources.append(actual)
    if not observed_families or observed_families != set(families):
        raise ValueError("training-family labels do not match the bound training artifacts")
    return sorted(families), verified_sources, source


def _verify_strategy_report(path: Path, *, candidate_hash: str,
                            identity_hash_value: str) -> tuple[dict, dict]:
    source = _source(path, "strategy evaluation")
    report = _read_object(path, "strategy evaluation")
    if (report.get("checkpointSha256") != candidate_hash
            or report.get("evaluationIdentityHash") != identity_hash_value
            or report.get("targetProbeWin") is not True
            or report.get("severityThreeRegression") is not False
            or report.get("severityThreeProbeCoverage") != "sufficient"
            or report.get("automaticPromotion") is not False):
        raise ValueError("strategy evaluation does not meet the frozen candidate promotion criteria")
    return report, source


def _verify_results(path: Path, *, candidate_hash: str, control_hash: str,
                    identity_hash_value: str, identity: dict,
                    candidate_model: StrategyTransformerV1,
                    control_model: StrategyTransformerV1, engine: EngineClient
                    ) -> tuple[dict, dict, list[dict], dict, dict]:
    source = _source(path, "matched promotion results")
    bundle = _read_object(path, "matched promotion results")
    body = {key: value for key, value in bundle.items() if key != "reportHash"}
    if (bundle.get("schemaVersion") != 1
            or bundle.get("kind") != "learning-mind-matched-promotion-results-v1"
            or bundle.get("candidateCheckpointSha256") != candidate_hash
            or bundle.get("controlCheckpointSha256") != control_hash
            or bundle.get("evaluationIdentityHash") != identity_hash_value
            or bundle.get("runnerImplementationSha256") != identity["evaluationRunnerSha256"]
            or bundle.get("reportHash") != identity_hash(body)
            or not isinstance(bundle.get("candidateRecords"), list)
            or not isinstance(bundle.get("controlRecords"), list)):
        raise ValueError("matched promotion results are malformed or bound to different artifacts")
    replay_paths = set()
    replay_sources = []
    prediction_caches = {"candidate": {}, "control": {}}
    reproduced_decisions = {"candidate": 0, "control": 0}
    reproduced_games = {"candidate": 0, "control": 0}
    promotion_probe_decisions = []
    for field, expected_checkpoint, rows in (
            ("candidate", candidate_hash, bundle["candidateRecords"]),
            ("control", control_hash, bundle["controlRecords"])):
        model = candidate_model if field == "candidate" else control_model
        if not rows:
            raise ValueError(f"matched promotion {field} results are empty")
        for row in rows:
            if (not isinstance(row, dict)
                    or row.get("checkpointSha256") != expected_checkpoint
                    or row.get("evaluationIdentityHash") != identity_hash_value
                    or row.get("schedulerIdentity") != identity["schedulerIdentity"]
                    or row.get("seedNamespace") != "promotion"
                    or not isinstance(row.get("decks"), list) or len(row["decks"]) != 2):
                raise ValueError(f"matched promotion {field} row identity does not match frozen sources")
            seat = row.get("learnerSeat")
            if (type(seat) is not int or seat not in (0, 1)
                    or row["decks"][seat] != row.get("ownArchetype")
                    or row["decks"][1 - seat] != row.get("opponentArchetype")):
                raise ValueError("matched promotion row deck assignment does not match its learner seat")
            replay = row.get("replay")
            if (not isinstance(replay, dict)
                    or not isinstance(replay.get("path"), str)
                    or not _sha256(replay.get("sha256"))
                    or not _sha256(replay.get("decodedSha256"))):
                raise ValueError(f"matched promotion {field} row lacks a checksummed raw replay")
            replay_path = Path(replay["path"]).resolve()
            replay_source = _source(replay_path, "matched promotion replay")
            if replay_source["sha256"] != replay["sha256"] or replay_source["path"] in replay_paths:
                raise ValueError("matched promotion replay hash mismatch or replay reused across games")
            replay_paths.add(replay_source["path"])
            replay_sources.append(replay_source)
            audited = audit_replay(replay_path, replay)
            count, probe_decisions = _reproduce_replay_actions(replay_path, model, field,
                row["learnerSeat"], prediction_caches[field])
            engine_count = _reproduce_engine_game(engine, replay_path, row,
                                                  identity["engineBuildHash"])
            winner = audited["outcome"].get("winner") if isinstance(audited["outcome"], dict) else None
            score = (.5 if winner is None else 1. if winner == row["learnerSeat"] else 0.)
            if (audited["replayId"] != row["gameId"]
                    or audited["status"] != row["status"]
                    or audited["seed"] != row["seed"]
                    or audited["firstPlayer"] != row["firstPlayer"]
                    or audited["engineBuildHash"] != identity["engineBuildHash"]
                    or audited["unsupportedPositions"] != 0
                    or audited["actionDecisionsByActor"][row["learnerSeat"]] != count
                    or engine_count != sum(audited["actionDecisionsByActor"].values())
                    or (row["status"] == "finished" and row.get("score") != score)
                    or (row["status"] != "finished" and row.get("score") is not None)):
                raise ValueError("promotion result row differs from its actor-view audited replay")
            reproduced_decisions[field] += count
            reproduced_games[field] += 1
            if field == "candidate":
                promotion_probe_decisions.extend(probe_decisions)
    from .experiment import summarize_strategy_probe_decisions
    promotion_probe_summary = summarize_strategy_probe_decisions(promotion_probe_decisions)
    return (bundle, source, replay_sources, {"passed": True, **reproduced_decisions},
            {"passed": True, **reproduced_games}, promotion_probe_summary)


def _write_immutable(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError("promotion evidence receipts are immutable; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def issue_promotion_evidence(*, candidate_checkpoint: Path, control_checkpoint: Path,
        identity_path: Path, results_path: Path, training_families_path: Path,
        strategy_report_path: Path, output: Path, human_approved: bool) -> tuple[dict, VerifiedPromotionEvidence | None]:
    """Recompute promotion gates from hashed artifacts; never promote automatically."""
    if type(human_approved) is not bool:
        raise ValueError("promotion approval must be an explicit boolean")
    candidate_source = _source(candidate_checkpoint, "candidate checkpoint")
    control_source = _source(control_checkpoint, "control checkpoint")
    if candidate_source["sha256"] == control_source["sha256"]:
        raise ValueError("candidate and control checkpoints must be different artifacts")
    candidate_policy_identity, candidate_model = _load_policy_checkpoint(
        Path(candidate_source["path"]), "candidate")
    control_policy_identity, control_model = _load_policy_checkpoint(
        Path(control_source["path"]), "control")
    identity, identity_source = _verify_identity(identity_path,
        candidate_hash=candidate_source["sha256"], control_hash=control_source["sha256"],
        candidate_policy_identity=candidate_policy_identity,
        control_policy_identity=control_policy_identity)
    families, family_dependencies, families_source = _verify_training_families(training_families_path)
    strategy, strategy_source = _verify_strategy_report(strategy_report_path,
        candidate_hash=candidate_source["sha256"], identity_hash_value=identity_hash(identity))
    engine_root = Path(__file__).resolve().parents[3]
    with EngineClient(engine_root) as engine:
        health = engine.request("health")
        if health.get("engineBuildHash") != identity["engineBuildHash"]:
            raise ValueError("promotion engine binary differs from the frozen identity")
        (results, results_source, replay_sources, action_reproduction,
         engine_reproduction, promotion_probe_summary) = _verify_results(
            results_path, candidate_hash=candidate_source["sha256"],
            control_hash=control_source["sha256"], identity_hash_value=identity_hash(identity),
            identity=identity, candidate_model=candidate_model, control_model=control_model,
            engine=engine)

    severity_three_regressions = [item["probeId"]
        for item in promotion_probe_summary["probes"]
        if item["severity"] == 3 and item["headline"]["status"] == "measured"
        and item["headline"]["modelRate"] < item["headline"]["heuristicRate"]]
    strategy_summary = {"severityThreeRegressions": severity_three_regressions}
    decision = matched_promotion_matrix_decision(results["candidateRecords"], results["controlRecords"],
        training_policy_families=families, strategy=strategy_summary,
        identities_match=True, human_approved=human_approved)
    sources = {"candidateCheckpoint": candidate_source, "controlCheckpoint": control_source,
        "promotionIdentity": identity_source, "matchedResults": results_source,
        "matchedReplays": replay_sources,
        "trainingFamilies": families_source, "strategyEvaluation": strategy_source,
        "trainingFamilyDependencies": family_dependencies}
    matrix_passed = decision["promotionGate"]["promotable"] is True
    action_reproduction_verified = action_reproduction["passed"] is True
    engine_reproduction_verified = engine_reproduction["passed"] is True
    promotion_probe_passed = (promotion_probe_summary["targetProbeWin"] is True
        and promotion_probe_summary["severityThreeRegression"] is False
        and promotion_probe_summary["severityThreeCoverage"] == "sufficient")
    # Rows currently bind family names, but the frozen evaluator has no source
    # roster that proves which opponent implementation produced each response.
    opponent_family_provenance_verified = False
    not_ready = []
    if not action_reproduction_verified:
        not_ready.append("checkpoint-to-action replay reproduction failed")
    if not engine_reproduction_verified:
        not_ready.append("frozen-engine replay reproduction failed")
    if not promotion_probe_passed:
        not_ready.append("promotion-seed strategy probes did not pass their evidence gate")
    if not opponent_family_provenance_verified:
        not_ready.append("opponent family runtime provenance is not verified")
    report = {"schemaVersion": 1, "kind": "verified-learning-mind-promotion-evidence-v1",
        "sources": sources, "identityHash": identity_hash(identity),
        "candidateCheckpoint": candidate_source["path"],
        "candidateCheckpointSha256": candidate_source["sha256"],
        "controlCheckpoint": control_source["path"],
        "controlCheckpointSha256": control_source["sha256"],
        "decision": decision, "humanApproved": human_approved,
        "actionReproduction": action_reproduction,
        "engineReproduction": engine_reproduction,
        "promotionStrategyProbes": promotion_probe_summary,
        "opponentFamilyProvenanceVerified": opponent_family_provenance_verified,
        "matrixCriteriaPassed": matrix_passed,
        "actionReproductionVerified": action_reproduction_verified,
        "engineReproductionVerified": engine_reproduction_verified,
        "notReadyReasons": not_ready,
        "promotionCriteriaPassed": (matrix_passed and action_reproduction_verified
                                    and engine_reproduction_verified and promotion_probe_passed
                                    and opponent_family_provenance_verified),
        "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    _write_immutable(output, report)
    if report["promotionCriteriaPassed"] is not True:
        return report, None
    evidence_values = {"candidateCheckpoint": candidate_source["path"],
        "candidateCheckpointSha256": candidate_source["sha256"],
        "controlCheckpointSha256": control_source["sha256"],
        "promotionCriteriaPassed": True, "humanApproved": human_approved,
        "sourceReportHash": report["reportHash"], "automaticPromotion": False}
    return report, VerifiedPromotionEvidence(evidence_values,
        _verification_token=_VERIFIED_PROMOTION_TOKEN)


def verify_promotion_evidence_report(path: Path) -> dict:
    """Verify the immutable summary checksum and every recorded source checksum."""
    report = _read_object(path, "promotion receipt")
    if (report.get("schemaVersion") != 1
            or report.get("kind") != "verified-learning-mind-promotion-evidence-v1"
            or report.get("reportHash") != identity_hash(
                {key: value for key, value in report.items() if key != "reportHash"})
            or report.get("automaticPromotion") is not False):
        raise ValueError("promotion receipt is malformed or has a bad hash")
    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("promotion receipt source list is missing")
    flattened = []
    for key, value in sources.items():
        values = value if key in {"trainingFamilyDependencies", "matchedReplays"} else [value]
        if not isinstance(values, list):
            raise ValueError("promotion receipt source entry is malformed")
        flattened.extend(values)
    for source in flattened:
        if (not isinstance(source, dict) or set(source) != {"path", "sha256"}
                or not isinstance(source.get("path"), str) or not _sha256(source.get("sha256"))
                or file_sha256(Path(source["path"])) != source["sha256"]):
            raise ValueError("promotion receipt source artifact changed after verification")
    return report


def reissue_promotion_evidence_report(
        path: Path) -> tuple[dict, VerifiedPromotionEvidence | None]:
    """Recompute a receipt and reissue its in-process capability if still valid.

    A checksum-verified JSON receipt alone is not a promotion capability: this
    function reruns all source checks and the matched-matrix decision before
    returning the opaque capability required by the rollback registry.
    """
    prior = verify_promotion_evidence_report(path)
    sources = prior["sources"]
    with tempfile.TemporaryDirectory(prefix="learning-mind-reissue-") as temporary:
        report, evidence = issue_promotion_evidence(
            candidate_checkpoint=Path(prior["candidateCheckpoint"]),
            control_checkpoint=Path(prior["controlCheckpoint"]),
            identity_path=Path(sources["promotionIdentity"]["path"]),
            results_path=Path(sources["matchedResults"]["path"]),
            training_families_path=Path(sources["trainingFamilies"]["path"]),
            strategy_report_path=Path(sources["strategyEvaluation"]["path"]),
            output=Path(temporary) / "recomputed.json",
            human_approved=prior["humanApproved"])
    if report != prior:
        raise ValueError("promotion receipt no longer reproduces from its frozen sources")
    return report, evidence
