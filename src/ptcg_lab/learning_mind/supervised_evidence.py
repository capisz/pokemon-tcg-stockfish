from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import torch

from ptcg_lab.features import heuristic_action_score
from ptcg_lab.storage import digest as legacy_digest

from .dataset_v1 import file_sha256, load_dataset
from .encoding import collate, encode_decision
from .evaluation import wilson
from .model import StrategyTransformerV1
from .schema import identity_hash
from .training import require_checkpoint_implementation

MINIMUM_INDEPENDENT_GAME_SIDES = 20
NON_POLICY_SOURCE_FAMILIES = frozenset({"human-review", "frozen-search-source"})


def summarize_paired_game_sides(side_records: list[dict], *, minimum_game_sides: int =
                                MINIMUM_INDEPENDENT_GAME_SIDES) -> dict:
    if type(minimum_game_sides) is not int or minimum_game_sides < 1:
        raise ValueError("minimum independent game-side count must be a positive integer")
    wins = sum(item["modelAccuracy"] > item["heuristicAccuracy"] for item in side_records)
    losses = sum(item["modelAccuracy"] < item["heuristicAccuracy"] for item in side_records)
    ties = len(side_records) - wins - losses
    decisive = wins + losses
    interval = wilson(wins, decisive)
    enough = len(side_records) >= minimum_game_sides and decisive >= minimum_game_sides
    status = ("insufficient" if not enough else
              "supported-improvement" if interval["low"] > .5 else
              "supported-regression" if interval["high"] < .5 else "inconclusive")
    return {"minimumIndependentGameSides": minimum_game_sides,
            "independentGameSides": len(side_records), "pairedWins": wins,
            "pairedLosses": losses, "pairedTies": ties, "decisiveGameSides": decisive,
            "pairedWinRate": wins / decisive if decisive else None,
            "pairedWilson95": interval, "status": status,
            "heldOutLabelWin": status == "supported-improvement"}


def summarize_blind_policy_families(side_records: list[dict], training_families: set[str], *,
                                    minimum_game_sides: int = MINIMUM_INDEPENDENT_GAME_SIDES) -> dict:
    eligible = [side for side in side_records if side.get("blindFamilyEligible") is True]
    unseen = sorted({side["opponentPolicyFamily"] for side in eligible
                     if side["opponentPolicyFamily"] not in training_families})
    results = {family: summarize_paired_game_sides(
        [side for side in eligible if side["opponentPolicyFamily"] == family],
        minimum_game_sides=minimum_game_sides) for family in unseen}
    statuses = {result["status"] for result in results.values()}
    status = ("missing" if not results else
              "supported-improvement" if "supported-improvement" in statuses else
              "supported-regression" if "supported-regression" in statuses else
              "inconclusive" if "inconclusive" in statuses else "insufficient")
    return {"status": status, "families": results,
            "unseenOpponentPolicyFamilies": unseen}


def audit_supervised_evaluation(*, dataset_dir: Path, checkpoint: Path,
                                evaluation_path: Path, output: Path,
                                minimum_game_sides: int = MINIMUM_INDEPENDENT_GAME_SIDES) -> dict:
    """Verify held-out predictions against frozen labels and require independent paired evidence."""
    if type(minimum_game_sides) is not int or minimum_game_sides < MINIMUM_INDEPENDENT_GAME_SIDES:
        raise ValueError(f"audit threshold cannot be lower than the frozen {MINIMUM_INDEPENDENT_GAME_SIDES} game-sides")
    output = output.resolve()
    if output.exists():
        raise ValueError("supervised evidence audit is immutable; choose a new output path")
    manifest, rows = load_dataset(dataset_dir)
    if any(row.get("split") not in {"train", "development", "heldout"} for row in rows):
        raise ValueError("supervised dataset contains an unknown split")
    checkpoint = checkpoint.resolve()
    evaluation_path = evaluation_path.resolve()
    checkpoint_record = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_record.get("identity") != manifest.get("identity"):
        raise ValueError("supervised checkpoint and dataset identities differ")
    if checkpoint_record.get("datasetManifestHash") != manifest.get("manifestHash"):
        raise ValueError("supervised checkpoint was not trained from this frozen dataset manifest")
    if checkpoint_record.get("kind") != "StrategyTransformerV1-supervised":
        raise ValueError("checkpoint is not a supervised StrategyTransformerV1 artifact")
    require_checkpoint_implementation(checkpoint_record)
    model = StrategyTransformerV1()
    model.load_state_dict(checkpoint_record["model"])
    model.eval()

    evaluation = json.loads(evaluation_path.read_text())
    checkpoint_hash = file_sha256(checkpoint)
    if evaluation.get("checkpointSha256") != checkpoint_hash:
        raise ValueError("evaluation checkpoint hash differs from the supplied checkpoint")
    if evaluation.get("datasetManifestHash") != manifest.get("manifestHash"):
        raise ValueError("evaluation dataset manifest differs from the frozen dataset")
    position_hashes = [row.get("positionHash") for row in rows]
    if any(not isinstance(value, str) or not value for value in position_hashes) or len(set(position_hashes)) != len(position_hashes):
        raise ValueError("supervised dataset position hashes must be present and globally unique")
    game_splits: dict[str, str] = {}
    for row in rows:
        game_id = row.get("sourceGameId")
        if game_id is None:
            continue
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("supervised source-game identity is invalid")
        prior_split = game_splits.setdefault(game_id, row.get("split"))
        if prior_split != row.get("split"):
            raise ValueError("source game crosses supervised dataset splits")
    expected = {row["positionHash"]: row for row in rows if row.get("split") != "train"}
    evaluated = evaluation.get("positions")
    if not isinstance(evaluated, list):
        raise ValueError("evaluation does not contain its per-position decisions")
    observed = {}
    for item in evaluated:
        if not isinstance(item, dict):
            raise ValueError("evaluation has an invalid position record")
        position_hash = item.get("positionHash")
        source = expected.get(position_hash)
        if source is None or position_hash in observed or item.get("split") != source.get("split"):
            raise ValueError("evaluation positions do not exactly match the frozen non-training dataset")
        observation = source.get("observation")
        tracker = source.get("tracker")
        if (not isinstance(observation, dict) or observation.get("playerId") != source.get("actor")
                or not isinstance(tracker, dict) or legacy_digest(observation) != position_hash):
            raise ValueError("frozen held-out row lacks its actor-visible observation and tracker")
        encoded = encode_decision(observation, tracker)
        if encoded.identity != source.get("featureIdentityHash"):
            raise ValueError("frozen held-out feature identity mismatch")
        model_class, heuristic_class = item.get("modelClass"), item.get("heuristicClass")
        if (type(model_class) is not int or not 0 <= model_class <= len(encoded.action_classes)
                or type(heuristic_class) is not int or not 0 <= heuristic_class < len(encoded.action_classes)):
            raise ValueError("evaluation contains an out-of-range policy or heuristic action class")
        batch = {key: torch.as_tensor(value) for key, value in collate([encoded]).items()}
        with torch.no_grad():
            actual_model_class = int(torch.argmax(model.policy_forward(**batch)[0]).item())
        if model_class != actual_model_class:
            raise ValueError("evaluation model action differs from the frozen checkpoint logits")
        if source.get("acceptableActionIndices"):
            acceptable_values = source["acceptableActionIndices"]
            if (not isinstance(acceptable_values, list)
                    or any(type(value) is not int or not 0 <= value < len(encoded.action_classes)
                           for value in acceptable_values)):
                raise ValueError("frozen acceptable-action indices are invalid")
            acceptable = set(acceptable_values)
            expected_model_hit = model_class in acceptable
            expected_heuristic_hit = heuristic_class in acceptable
            expected_kind = "acceptable-set"
        else:
            distribution = source.get("policyDistribution")
            if (not isinstance(distribution, list) or len(distribution) != len(encoded.action_classes) + 1
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in distribution)
                    or not math.isclose(sum(distribution), 1., rel_tol=0., abs_tol=1e-5)):
                raise ValueError("frozen exact-search distribution does not cover action classes and STOP")
            best = max(range(len(distribution)), key=distribution.__getitem__)
            expected_model_hit = model_class == best
            expected_heuristic_hit = heuristic_class == best
            expected_kind = "distribution"
        if (item.get("labelKind") != expected_kind
                or type(item.get("modelHit")) is not bool or item["modelHit"] != expected_model_hit
                or type(item.get("heuristicHit")) is not bool
                or item["heuristicHit"] != expected_heuristic_hit):
            raise ValueError("evaluation action hits do not match the frozen acceptable labels")
        observation_actions = observation["legalActions"]
        heuristic_id = max(observation_actions,
            key=lambda action: heuristic_action_score(action, observation))["id"]
        expected_heuristic_class = next((index for index, group in enumerate(encoded.action_classes)
            if any(action.get("id") == heuristic_id for action in group.actions)), None)
        if heuristic_class != expected_heuristic_class:
            raise ValueError("evaluation heuristic action differs from the frozen heuristic")
        observed[position_hash] = (source, item)
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise ValueError(f"evaluation omitted frozen non-training positions: {missing[:5]}")

    training_rows = [row for row in rows if row.get("split") == "train"]
    if not training_rows or any(not isinstance(row.get("opponentPolicyFamily"), str)
                                or not row["opponentPolicyFamily"] for row in training_rows):
        raise ValueError("training rows lack frozen opponent-policy family metadata")
    training_policy_families = {row["opponentPolicyFamily"] for row in training_rows
                                if row.get("sourceGameId") is not None
                                and row["opponentPolicyFamily"] not in NON_POLICY_SOURCE_FAMILIES}
    heldout_sides: dict[tuple[str, str, int], list[tuple[bool, bool]]] = {}
    side_families: dict[tuple[str, str, int], str] = {}
    for source, item in observed.values():
        if source.get("split") != "heldout":
            continue
        source_game = source.get("sourceGameId")
        unit_kind = "game" if source_game else "family"
        unit = source_game or source.get("familyId")
        actor = source.get("actor")
        policy_family = source.get("opponentPolicyFamily")
        if not isinstance(unit, str) or not unit or type(actor) is not int or actor not in (0, 1):
            raise ValueError("held-out decision lacks an independent source-game/review-family perspective")
        if not isinstance(policy_family, str) or not policy_family:
            raise ValueError("held-out decision lacks its frozen opponent-policy family")
        key = (unit_kind, unit, actor)
        if key in side_families and side_families[key] != policy_family:
            raise ValueError("one held-out game-side mixes opponent-policy families")
        side_families[key] = policy_family
        heldout_sides.setdefault(key, []).append((item["modelHit"], item["heuristicHit"]))
    side_records = []
    for (unit_kind, unit, actor), decisions in sorted(heldout_sides.items()):
        model_rate = sum(model for model, _heuristic in decisions) / len(decisions)
        heuristic_rate = sum(heuristic for _model, heuristic in decisions) / len(decisions)
        side_records.append({"gameSide": f"{unit_kind}:{unit}:{actor}", "positions": len(decisions),
                             "opponentPolicyFamily": side_families[(unit_kind, unit, actor)],
                             "blindFamilyEligible": (unit_kind == "game" and
                                 side_families[(unit_kind, unit, actor)] not in NON_POLICY_SOURCE_FAMILIES),
                             "modelAccuracy": model_rate, "heuristicAccuracy": heuristic_rate,
                             "pairedResult": "win" if model_rate > heuristic_rate else
                                 "loss" if model_rate < heuristic_rate else "tie"})
    paired = summarize_paired_game_sides(side_records, minimum_game_sides=minimum_game_sides)
    blind = summarize_blind_policy_families(side_records, training_policy_families,
                                            minimum_game_sides=minimum_game_sides)
    report = {"schemaVersion": 1, "kind": "supervised-heldout-evidence-v1",
        "datasetManifestHash": manifest["manifestHash"],
        "datasetManifestSha256": file_sha256(dataset_dir / "manifest.json"),
        "checkpointSha256": checkpoint_hash,
        "evaluationSha256": file_sha256(evaluation_path),
        "evaluationIdentityStatus": "matched",
        "trainingOpponentPolicyFamilies": sorted(training_policy_families),
        "blindOpponentPolicyFamilyStatus": blind["status"],
        "blindOpponentPolicyFamilies": blind["families"],
        "unseenOpponentPolicyFamilies": blind["unseenOpponentPolicyFamilies"],
        "heldoutPositions": sum(source.get("split") == "heldout" for source, _item in observed.values()),
        **paired, "gameSides": side_records,
        "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", delete=False) as temporary:
        temporary.write(json.dumps(report, indent=2) + "\n")
        temporary.flush(); os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report


def verify_supervised_audit_report(*, dataset_dir: Path, checkpoint: Path,
                                   evaluation_path: Path, audit_path: Path) -> dict:
    """Recompute a prior audit and reject reports whose self-hash hides forged conclusions."""
    claimed = json.loads(audit_path.read_text())
    recorded_hash = claimed.get("reportHash")
    if recorded_hash != identity_hash({key: value for key, value in claimed.items()
                                       if key != "reportHash"}):
        raise ValueError("supervised evidence audit report hash mismatch")
    minimum = claimed.get("minimumIndependentGameSides")
    if type(minimum) is not int or minimum < MINIMUM_INDEPENDENT_GAME_SIDES:
        raise ValueError("supervised evidence audit does not use the frozen independent-side threshold")
    with tempfile.TemporaryDirectory(prefix="learning-mind-audit-verify-") as directory:
        recomputed = audit_supervised_evaluation(dataset_dir=dataset_dir, checkpoint=checkpoint,
            evaluation_path=evaluation_path, output=Path(directory) / "verified.json",
            minimum_game_sides=minimum)
    if claimed != recomputed:
        raise ValueError("supervised evidence audit conclusions do not match a fresh recomputation")
    return recomputed
