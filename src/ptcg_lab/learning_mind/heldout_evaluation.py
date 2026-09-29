"""Leakage-safe, descriptive evaluation of macro-ranker artifacts on heldout labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import os

import numpy as np

from .collector_compatibility import collector_compatibility
from .dataset_v1 import file_sha256, load_dataset
from .experiment import SHARED_ROLLOUT_SETTINGS
from .macro_fidelity import _validate_rollout_label
from .heldout_seed import HELDOUT_SEED_VERSION, heldout_rollout_seed
from .ranker_features import candidate_features_v2
from .ranker_portable import predict_macro_ranker_v2
from .ranker_v2 import _bootstrap_mean, verify_macro_ranker_v2_artifact
from .schema import identity_hash
from ptcg_lab.storage import digest as observation_digest


FAMILIES = ("python-heuristic", "typescript-heuristic")

# These are the values used by the reviewed v2 heldout preflights. A run with
# different values is a different experiment, not a continuation of this gate.
FROZEN_HELDOUT_SETTINGS = {
    "initialRollouts": 16,
    "maximumRollouts": 64,
    "extensionBatchSize": 8,
    "horizon": 500,
    "rolloutBudgetMs": 60_000,
    "rolloutWorkers": 8,
    "rolloutSeedVersion": HELDOUT_SEED_VERSION,
    "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
    "selectionMethod": "frozen-heldout-selection-v1",
    "splitFilter": "heldout",
}


_heldout_rollout_seed = heldout_rollout_seed


def _publish_json(path: Path, value: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError("heldout evaluation reports are immutable; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as temporary:
        json.dump(value, temporary, sort_keys=True, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_frozen_heldout_selection(path: Path) -> tuple[dict, dict[str, dict]]:
    selection = json.loads(path.read_text())
    if (not isinstance(selection, dict)
            or selection.get("schemaVersion") != 1
            or selection.get("kind") != "macro-ranker-heldout-selection-v1"
            or selection.get("trainingEligible") is not False
                or not isinstance(selection.get("interpretation"), str)
                or not selection["interpretation"].startswith("heldout evaluator selection only")
            or selection.get("selectionHash") != identity_hash(
                {key: value for key, value in selection.items() if key != "selectionHash"})):
        raise ValueError("heldout position selection is malformed or training-eligible")
    if selection.get("minimumCompleteCandidatesPerPosition") != 2:
        raise ValueError("heldout selection does not require distinct complete candidates")
    if not isinstance(selection.get("identity"), dict):
        raise ValueError("heldout selection lacks its frozen feature identity")
    splits = selection.get("splits")
    if not isinstance(splits, list) or len(splits) != len(FAMILIES):
        raise ValueError("heldout selection must contain exactly one split per policy family")
    by_family = {}
    all_positions = set()
    all_source_games = set()
    for split in splits:
        if not isinstance(split, dict):
            raise ValueError("heldout selection split record is malformed")
        family = split.get("policyFamily")
        positions = split.get("positions")
        if (family not in FAMILIES or family in by_family or split.get("split") != "heldout"
                or not isinstance(positions, list) or not positions
                or type(split.get("sourceGames")) is not int or split["sourceGames"] < 1):
            raise ValueError("heldout selection family/split record is invalid")
        hashes, games = set(), set()
        for position in positions:
            if (not isinstance(position, dict)
                    or not isinstance(position.get("positionHash"), str)
                    or not isinstance(position.get("sourceGameId"), str)
                    or not position.get("sourceGameId")
                    or position.get("positionHash") in hashes):
                raise ValueError("heldout selection contains malformed or duplicate position metadata")
            hashes.add(position["positionHash"])
            games.add(position["sourceGameId"])
        if (split.get("positionHashes") != [row["positionHash"] for row in positions]
                or split.get("sourceGameIds") != sorted(games)
                or split.get("sourceGames") != len(games)):
            raise ValueError("heldout selection position/game counts do not reconcile")
        if all_positions.intersection(hashes):
            raise ValueError("heldout selection reuses a position across policy families")
        # Keep policy-family provenance in the cluster key so potentially
        # matched seeds across families are not treated as one unpaired unit.
        if all_source_games.intersection((family, game) for game in games):
            raise ValueError("heldout selection reuses a family/source-game cluster")
        all_positions.update(hashes)
        all_source_games.update((family, game) for game in games)
        by_family[family] = {row["positionHash"]: row for row in positions}
    if set(by_family) != set(FAMILIES):
        raise ValueError("heldout selection does not include both policy families")
    return selection, by_family


def _verify_source_pool(*, family: str, dataset_dir: Path, support_path: Path,
                        selection: dict) -> tuple[dict, dict]:
    source_pools = selection.get("sourcePools")
    if not isinstance(source_pools, list):
        raise ValueError("heldout selection has no frozen source-pool records")
    source = next((item for item in source_pools
                   if isinstance(item, dict) and item.get("policyFamily") == family), None)
    if not isinstance(source, dict):
        raise ValueError(f"heldout selection omits {family} source provenance")
    dataset_dir, support_path = dataset_dir.resolve(), support_path.resolve()
    manifest, rows = load_dataset(dataset_dir, identity=selection["identity"])
    support = json.loads(support_path.read_text())
    if (manifest.get("manifestHash") != source.get("datasetManifestHash")
            or file_sha256(dataset_dir / "manifest.json") != source.get("datasetManifestSha256")
            or file_sha256(dataset_dir / "rows.jsonl") != source.get("datasetRowsSha256")
            or support.get("identity") != selection["identity"]
            or support.get("datasetManifestHash") != manifest.get("manifestHash")
            or support.get("datasetRowsSha256") != source.get("datasetRowsSha256")
            or support.get("reportHash") != source.get("supportReportHash")
            or file_sha256(support_path) != source.get("supportReportSha256")
            or support.get("reportHash") != identity_hash(
                {key: value for key, value in support.items() if key != "reportHash"})):
        raise ValueError(f"{family} heldout source dataset/support identity mismatch")
    rows_by_hash = {row.get("positionHash"): row for row in rows}
    if len(rows_by_hash) != len(rows):
        raise ValueError(f"{family} heldout source dataset repeats position hashes")
    support_positions = support.get("positions")
    if not isinstance(support_positions, list):
        raise ValueError(f"{family} heldout support report has no position rows")
    support_by_hash = {row.get("positionHash"): row for row in support_positions
                       if isinstance(row, dict)}
    if len(support_by_hash) != len(support_positions):
        raise ValueError(f"{family} heldout support report repeats or malforms position hashes")
    for item in selection["splits"]:
        if item["policyFamily"] != family:
            continue
        for position_hash, selected in ((row["positionHash"], row) for row in item["positions"]):
            row = rows_by_hash.get(position_hash)
            support_row = support_by_hash.get(position_hash)
            if (row is None or row.get("split") != "heldout"
                    or row.get("sourceGameId") != selected.get("sourceGameId")
                    or row.get("opponentPolicyFamily") != family
                    or support_row is None or support_row.get("status") != "supported"
                    or type(support_row.get("completeCandidateCount")) is not int
                    or support_row["completeCandidateCount"] < 2):
                raise ValueError(f"{family} heldout selection differs from its frozen dataset")
    return manifest, support


def _validate_training_selection(ranker_selection: dict, *, identity: dict,
                                 heldout_by_family: dict[str, dict]) -> tuple[set, set]:
    if (not isinstance(ranker_selection, dict)
            or ranker_selection.get("selectionHash") != identity_hash(
                {key: value for key, value in ranker_selection.items() if key != "selectionHash"})
            or ranker_selection.get("identity") != identity):
        raise ValueError("ranker training selection is invalid or uses a different model identity")
    training_games, training_positions = set(), set()
    games_by_split: dict[tuple[str, str], set[str]] = {}
    training_splits = ranker_selection.get("splits")
    if not isinstance(training_splits, list) or len(training_splits) != len(FAMILIES) * 2:
        raise ValueError("ranker training selection must contain both families and train/development splits")
    seen_splits = set()
    for split in training_splits:
        if (not isinstance(split, dict) or split.get("policyFamily") not in FAMILIES
                or split.get("split") not in {"train", "development"}):
            raise ValueError("ranker training selection contains heldout or unknown split rows")
        split_key = (split["policyFamily"], split["split"])
        if split_key in seen_splits:
            raise ValueError("ranker training selection repeats a family/split record")
        seen_splits.add(split_key)
        positions = split.get("positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError("ranker training selection contains an empty split")
        split_games, split_positions = set(), set()
        for position in positions:
            if (not isinstance(position, dict) or not isinstance(position.get("sourceGameId"), str)
                    or not position.get("sourceGameId") or not isinstance(position.get("positionHash"), str)
                    or not position.get("positionHash")):
                raise ValueError("ranker training selection has malformed position provenance")
            split_games.add(position["sourceGameId"])
            split_positions.add(position["positionHash"])
        if (len(split_games) != len(positions) or len(split_positions) != len(positions)
                or split.get("sourceGames") != len(split_games)
                or split.get("positionHashes") != [position["positionHash"] for position in positions]):
            raise ValueError("ranker training selection positions/games do not reconcile")
        if training_positions.intersection(split_positions):
            raise ValueError("ranker training selection reuses a position across family/split rows")
        training_positions.update(split_positions)
        games_by_split[split_key] = split_games
        training_games.update((split["policyFamily"], game) for game in split_games)
    if seen_splits != {(family, split) for family in FAMILIES for split in ("train", "development")}:
        raise ValueError("ranker training selection omits a family/split")
    for family in FAMILIES:
        if games_by_split[(family, "train")].intersection(games_by_split[(family, "development")]):
            raise ValueError("ranker training selection reuses source games across train/development")
    heldout_games = {(family, position["sourceGameId"])
                     for family in FAMILIES for position in heldout_by_family[family].values()}
    heldout_positions = {position_hash for family in FAMILIES for position_hash in heldout_by_family[family]}
    if training_games.intersection(heldout_games) or training_positions.intersection(heldout_positions):
        raise ValueError("ranker training/development data overlap the frozen heldout games or positions")
    return training_games, training_positions


def _load_heldout_run(*, family: str, labels_dir: Path, selection: dict,
                      selected_positions: dict[str, dict], dataset_manifest: dict,
                      support: dict, expected_generator: dict,
                      expected_settings: dict | None = None) -> tuple[dict, list[dict]]:
    labels_dir = labels_dir.resolve()
    manifest_path = labels_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_hashes = set(selected_positions)
    settings = FROZEN_HELDOUT_SETTINGS if expected_settings is None else expected_settings
    if (manifest.get("manifestHash") != identity_hash(
            {key: value for key, value in manifest.items() if key != "manifestHash"})
            or manifest.get("identity") != selection["identity"]
            or manifest.get("datasetManifestHash") != dataset_manifest.get("manifestHash")
            or manifest.get("splitFilter") != "heldout"
            or manifest.get("selectedPositionHashes") != sorted(expected_hashes)
            or manifest.get("positions") != len(expected_hashes)
            or manifest.get("candidateGeneratorIdentity") != expected_generator
            or manifest.get("candidateGeneratorIdentity") != support.get("candidateGeneratorIdentity")
            or manifest.get("candidateGeneratorVersion") != expected_generator.get("version")
            or any(manifest.get(key) != value for key, value in settings.items())
            or manifest.get("highConfidencePolicyLabels") != 0
            or manifest.get("rolloutSeedVersion") != HELDOUT_SEED_VERSION
            or manifest.get("rolloutSeedImplementationSha256") != file_sha256(
                Path(__file__).with_name("heldout_seed.py"))
            or manifest.get("labelCollectorVersion") != "heldout-macro-rollout-labeler-v1"
            or manifest.get("labelCollectorSha256") != file_sha256(
                Path(__file__).with_name("heldout_collection.py"))):
        raise ValueError(f"{family} heldout label run manifest does not match the frozen evaluation selection")
    files = manifest.get("files")
    expected_names = {f"{position_hash}.json" for position_hash in expected_hashes}
    listed_names = [item.get("path") if isinstance(item, dict) else None for item in files] if isinstance(files, list) else []
    if (not isinstance(files, list) or len(files) != len(expected_names)
            or any(not isinstance(name, str) for name in listed_names)
            or set(listed_names) != expected_names or len(set(listed_names)) != len(listed_names)):
        raise ValueError(f"{family} heldout label run file list differs from selection")
    actual_names = {path.name for path in labels_dir.glob("*.json") if path.name != "manifest.json"}
    if actual_names != expected_names:
        raise ValueError(f"{family} heldout label run contains missing or unlisted result files")
    records = []
    for item in files:
        relative = Path(item["path"])
        if (relative.is_absolute() or ".." in relative.parts or relative.name != item["path"]
                or relative.name == "manifest.json"):
            raise ValueError("unsafe heldout label record path")
        path = labels_dir / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise ValueError(f"{family} heldout label record hash/path mismatch: {relative}")
        record = json.loads(path.read_text())
        position_hash = relative.stem
        position = selected_positions.get(position_hash)
        observation = record.get("observation")
        if (position is None or record.get("positionHash") != position_hash
                or record.get("identity") != selection["identity"]
                or record.get("selectionHash") != selection["selectionHash"]
                or record.get("datasetManifestHash") != dataset_manifest.get("manifestHash")
                or record.get("split") != "heldout"
                or record.get("opponentPolicyFamily") != family
                or record.get("sourceGameId") != position["sourceGameId"]
                or record.get("seedNamespace") != "heldout"
                or record.get("highConfidencePolicyEligible") is not False
                or observation_digest(observation) != position_hash):
            raise ValueError(f"{family} heldout label record provenance/actor-view hash mismatch")
        rollout_identity = manifest.get("rolloutIdentity")
        if record.get("rolloutIdentity") != rollout_identity:
            raise ValueError("heldout label record rollout identity differs from its run manifest")
        if record.get("status") == "unsupported":
            if (not isinstance(record.get("unsupportedReason"), str)
                    or not record["unsupportedReason"] or record.get("labels") != []
                    or type(record.get("candidateCount")) is not int or record["candidateCount"] < 0
                    or "rolloutSeeds" in record):
                raise ValueError("unsupported heldout position carries labels or malformed failure evidence")
            records.append({**record, "sourceGameId": position["sourceGameId"],
                "targetDeck": position["targetDeck"],
                "opponentArchetype": position["opponentArchetype"],
                "positionStage": position["positionStage"]})
            continue
        if record.get("status") != "collected":
            raise ValueError(f"{family} heldout label record has an unknown status")
        seeds = record.get("rolloutSeeds")
        maximum_rollouts = manifest.get("maximumRollouts")
        if (type(maximum_rollouts) is not int or maximum_rollouts < 1
                or not isinstance(seeds, list) or len(seeds) != maximum_rollouts
                or any(type(seed) is not int for seed in seeds)
                or len(set(seeds)) != len(seeds)
                or seeds != [_heldout_rollout_seed(position_hash, index, rollout_identity)
                             for index in range(len(seeds))]):
            raise ValueError("heldout label seed list does not match its frozen namespace")
        labels = record.get("labels")
        if (not isinstance(labels, list) or len(labels) < 2
                or record.get("candidateCount") != len(labels)):
            raise ValueError("heldout label record lacks at least two complete candidates")
        candidates, completed = set(), []
        legal_list = observation.get("legalActions") if isinstance(observation, dict) else None
        if not isinstance(legal_list, list):
            raise ValueError("heldout label observation lacks actor-visible legal actions")
        legal_actions = {str(action.get("id")): action
                         for action in legal_list if isinstance(action, dict)}
        for label in labels:
            evidence = _validate_rollout_label(label)
            outcomes_by_index = label.get("outcomesBySeedIndex")
            if not isinstance(outcomes_by_index, dict):
                raise ValueError("heldout candidate lacks auditable per-seed rollout outcomes")
            attempted = label["attemptedRollouts"]
            if set(outcomes_by_index) != {str(index) for index in range(attempted)}:
                raise ValueError("heldout per-seed outcomes do not match the candidate's sampled prefix")
            reconciled = {"finished": 0, "truncated": 0, "error": 0}
            reconciled_reasons, reconciled_decisions, finished_scores = {}, {}, []
            for index_text, seed_outcome in outcomes_by_index.items():
                index = int(index_text)
                if index >= len(seeds) or not isinstance(seed_outcome, dict):
                    raise ValueError("heldout per-seed outcome index/value is malformed")
                status = seed_outcome.get("status")
                if status == "finished":
                    score = seed_outcome.get("score")
                    if (set(seed_outcome) - {"status", "score", "decisionCount"}
                            or isinstance(score, bool) or not isinstance(score, (int, float))
                            or not math.isfinite(score) or not 0 <= score <= 1):
                        raise ValueError("heldout finished seed outcome has invalid score evidence")
                    finished_scores.append(float(score))
                elif status in {"truncated", "error"}:
                    reason = seed_outcome.get("reason")
                    if (set(seed_outcome) - {"status", "reason", "decisionCount"}
                            or not isinstance(reason, str) or not reason):
                        raise ValueError("heldout unfinished seed outcome lacks a typed reason")
                    reconciled_reasons[reason] = reconciled_reasons.get(reason, 0) + 1
                else:
                    raise ValueError("heldout per-seed outcome has an unknown status")
                reconciled[status] += 1
                decision_count = seed_outcome.get("decisionCount")
                if decision_count is not None:
                    if type(decision_count) is not int or decision_count < 0:
                        raise ValueError("heldout per-seed decision count is malformed")
                    text = str(decision_count)
                    reconciled_decisions[text] = reconciled_decisions.get(text, 0) + 1
            if (reconciled != label["outcomes"]
                    or reconciled_reasons != label["outcomeReasons"]
                    or reconciled_decisions != label["decisionCountDistribution"]
                    or (finished_scores and not math.isclose(
                        sum(finished_scores) / len(finished_scores),
                        float(label.get("expectedResult")), rel_tol=1e-6, abs_tol=1e-8))
                    or (not finished_scores and label.get("expectedResult") is not None)):
                raise ValueError("heldout per-seed evidence does not reconcile with candidate aggregates")
            candidate = evidence["candidate"]
            candidate_hash = candidate.key()
            if candidate_hash in candidates:
                raise ValueError("heldout label record repeats a candidate")
            candidates.add(candidate_hash)
            first_action = candidate.action_sequence[0]
            if legal_actions.get(str(first_action.get("id"))) != first_action:
                raise ValueError("heldout candidate root is not actor-visible and legal")
            if label.get("attemptedRollouts") > len(seeds):
                raise ValueError("heldout candidate used more rollouts than the frozen seed list")
            expected_result = label.get("expectedResult")
            finished = label["outcomes"]["finished"]
            if expected_result is None:
                if finished or label.get("relativeResult") is not None or label.get("weight") != 0:
                    raise ValueError("unscored heldout candidate carries fabricated outcomes")
                continue
            if (isinstance(expected_result, bool) or not isinstance(expected_result, (int, float))
                    or not math.isfinite(expected_result) or not 0 <= expected_result <= 1
                    or finished < 1):
                raise ValueError("heldout candidate expected result is invalid")
            uncertainty = label.get("uncertainty")
            expected_uncertainty = math.sqrt(max(expected_result * (1 - expected_result), .25) / finished)
            expected_weight = finished / (1 + expected_uncertainty)
            relative = label.get("relativeResult")
            if (isinstance(relative, bool) or not isinstance(relative, (int, float))
                    or not math.isfinite(relative)
                    or not math.isclose(float(uncertainty), expected_uncertainty, rel_tol=1e-6, abs_tol=1e-8)
                    or not math.isclose(float(label.get("weight")), expected_weight, rel_tol=1e-6, abs_tol=1e-8)):
                raise ValueError("heldout candidate target/weight does not reconcile with completed outcomes")
            completed.append(label)
        if len(completed) < 2:
            raise ValueError("heldout position has fewer than two scored candidates")
        center = max(float(label["expectedResult"]) for label in completed)
        if any(not math.isclose(float(label["relativeResult"]),
                                float(label["expectedResult"]) - center, rel_tol=1e-6, abs_tol=1e-8)
               for label in completed):
            raise ValueError("heldout relative results do not reconcile within the position")
        records.append({**record, "sourceGameId": position["sourceGameId"],
                        "targetDeck": position["targetDeck"],
                        "opponentArchetype": position["opponentArchetype"],
                        "positionStage": position["positionStage"]})
    if {record["positionHash"] for record in records} != expected_hashes:
        raise ValueError(f"{family} heldout label records do not exactly cover the frozen selection")
    return manifest, records


def _summarize(details: list[dict], *, seed_material: str) -> dict:
    if not details:
        return {"status": "insufficient", "positions": 0, "independentSourceGames": 0,
                "meanTop1RelativeRegret": None, "meanTop1RelativeRegretCI95": {"low": None, "high": None},
                "positionWeightedMeanTop1RelativeRegret": None,
                "positionWeightedMeanTop1RelativeRegretCI95": {"low": None, "high": None},
                "top3Recall": None, "pairwiseComparisons": 0, "pairwiseOrderingAccuracy": None}
    regret = [float(item["top1RelativeRegret"]) for item in details]
    clusters = [f"{item['opponentPolicyFamily']}|{item['sourceGameId']}" for item in details]
    position_weighted = _bootstrap_mean(regret, group_ids=clusters,
        seed_material=seed_material + "|" + "|".join(sorted(item["positionHash"] for item in details)))
    regrets_by_game = {}
    for cluster, value in zip(clusters, regret):
        regrets_by_game.setdefault(cluster, []).append(value)
    source_game_keys = sorted(regrets_by_game)
    source_game_means = [float(np.mean(regrets_by_game[key])) for key in source_game_keys]
    game_balanced = _bootstrap_mean(source_game_means,
        group_ids=source_game_keys,
        seed_material=seed_material + "|equal-source-game-means|"
            + "|".join(sorted(item["positionHash"] for item in details)))
    comparisons = sum(item["pairwiseComparisons"] for item in details)
    correct = sum(item["pairwiseCorrect"] for item in details)
    return {"status": "measured" if comparisons else "insufficient",
        "positions": len(details), "independentSourceGames": game_balanced["independentUnits"],
        "meanTop1RelativeRegret": game_balanced["mean"],
        "meanTop1RelativeRegretCI95": game_balanced["interval95"],
        "meanTop1RelativeRegretWeighting": "equal-source-game",
        "bootstrapMethod": game_balanced["method"],
        "bootstrapReplicates": game_balanced["replicates"],
        "bootstrapSeed": game_balanced["seed"],
        "positionWeightedMeanTop1RelativeRegret": position_weighted["mean"],
        "positionWeightedMeanTop1RelativeRegretCI95": position_weighted["interval95"],
        "positionWeightedBootstrapMethod": position_weighted["method"],
        "positionWeightedBootstrapSeed": position_weighted["seed"],
        "top3Recall": sum(item["top3Recall"] for item in details) / len(details),
        "pairwiseComparisons": comparisons,
        "pairwiseOrderingAccuracy": correct / comparisons if comparisons else None}


def evaluate_macro_ranker_v2_heldout(*, selection_path: Path, ranker_selection_path: Path,
                                     model_path: Path, ranker_report_path: Path,
                                     datasets: dict[str, Path], support_reports: dict[str, Path],
                                     label_runs: dict[str, Path], output: Path) -> dict:
    """Score a frozen ranker on heldout labels; never fits or changes the model."""
    output = output.resolve()
    if output.exists():
        raise ValueError("heldout evaluation reports are immutable; choose a new output path")
    if set(datasets) != set(FAMILIES) or set(support_reports) != set(FAMILIES) or set(label_runs) != set(FAMILIES):
        raise ValueError("heldout evaluation requires both frozen policy families")
    selection_path, ranker_selection_path = selection_path.resolve(), ranker_selection_path.resolve()
    selection, selected_by_family = _read_frozen_heldout_selection(selection_path)
    ranker_selection = json.loads(ranker_selection_path.read_text())
    _validate_training_selection(ranker_selection, identity=selection["identity"],
                                 heldout_by_family=selected_by_family)

    verified = verify_macro_ranker_v2_artifact(model_path.resolve(), ranker_report_path.resolve())
    ranker_report, model = verified["report"], verified["artifact"]
    if (ranker_report.get("identity") != selection["identity"]
            or ranker_report.get("selectionHash") != ranker_selection.get("selectionHash")
            or ranker_report.get("selectionManifestSha256") != file_sha256(ranker_selection_path)):
        raise ValueError("ranker artifact is not bound to the supplied frozen training selection")
    expected_generator = None
    pool_manifests, pool_supports = {}, {}
    for family in FAMILIES:
        dataset_manifest, support = _verify_source_pool(family=family, dataset_dir=datasets[family],
            support_path=support_reports[family], selection=selection)
        generator = support.get("candidateGeneratorIdentity")
        if not isinstance(generator, dict):
            raise ValueError(f"{family} support report lacks candidate generator provenance")
        if expected_generator is None:
            expected_generator = generator
        elif generator != expected_generator:
            raise ValueError("heldout source pools use different candidate generators")
        pool_manifests[family], pool_supports[family] = dataset_manifest, support

    loaded_runs = {}
    for family in FAMILIES:
        loaded_runs[family] = _load_heldout_run(family=family, labels_dir=label_runs[family],
            selection=selection, selected_positions=selected_by_family[family],
            dataset_manifest=pool_manifests[family], support=pool_supports[family],
            expected_generator=expected_generator)
    compatibility = collector_compatibility(
        versions=[loaded_runs[family][0]["labelCollectorVersion"] for family in FAMILIES],
        source_hashes=[loaded_runs[family][0]["labelCollectorSha256"] for family in FAMILIES])
    shared_settings = [{key: loaded_runs[family][0].get(key) for key in SHARED_ROLLOUT_SETTINGS}
                       for family in FAMILIES]
    if (shared_settings[0] != shared_settings[1]
            or any(settings.get("splitFilter") != "heldout" for settings in shared_settings)):
        raise ValueError("heldout label runs do not share frozen rollout settings/split")

    position_details = []
    unsupported_positions = []
    for family in FAMILIES:
        for record in loaded_runs[family][1]:
            if record.get("status") == "unsupported":
                unsupported_positions.append({"policyFamily": family,
                    "positionHash": record["positionHash"],
                    "sourceGameId": record["sourceGameId"],
                    "reason": record["unsupportedReason"]})
                continue
            labels = [label for label in record["labels"] if label.get("expectedResult") is not None]
            features = [candidate_features_v2(record["observation"], label["candidate"]) for label in labels]
            scores = predict_macro_ranker_v2(model, features)
            truth = np.asarray([label["expectedResult"] for label in labels], dtype=np.float64)
            winner = int(np.argmax(truth))
            chosen = int(np.argmax(scores))
            comparisons = correct = 0
            for left in range(len(labels)):
                for right in range(left + 1, len(labels)):
                    if truth[left] == truth[right]:
                        continue
                    comparisons += 1
                    correct += int((scores[left] - scores[right]) * (truth[left] - truth[right]) > 0)
            position_details.append({
                "positionHash": record["positionHash"], "sourceGameId": record["sourceGameId"],
                "opponentPolicyFamily": family, "targetDeck": record["targetDeck"],
                "opponentArchetype": record["opponentArchetype"], "positionStage": record["positionStage"],
                "candidates": len(labels), "chosenCandidateHash": labels[chosen]["candidateHash"],
                "bestCandidateHash": labels[winner]["candidateHash"],
                "top1RelativeRegret": float(truth[winner] - truth[chosen]),
                "top3Recall": bool(winner in np.argsort(scores)[-min(3, len(labels)):]),
                "pairwiseCorrect": correct, "pairwiseComparisons": comparisons,
                "finishedRollouts": sum(label["outcomes"]["finished"] for label in labels),
                "truncatedRollouts": sum(label["outcomes"]["truncated"] for label in labels),
                "engineErrors": sum(label["outcomes"]["error"] for label in labels),
            })
    summaries = {"overall": _summarize(position_details,
                    seed_material="macro-ranker-v2-heldout|" + selection["selectionHash"]),
        "byPolicyFamily": {}, "byTargetDeck": {}, "byOpponentArchetype": {}}
    for field, destination in (("opponentPolicyFamily", summaries["byPolicyFamily"]),
                               ("targetDeck", summaries["byTargetDeck"]),
                               ("opponentArchetype", summaries["byOpponentArchetype"])):
        for value in sorted({item[field] for item in position_details}):
            subset = [item for item in position_details if item[field] == value]
            destination[value] = _summarize(subset,
                seed_material=f"macro-ranker-v2-heldout|{field}|{value}|{selection['selectionHash']}")

    selected_positions = sum(len(selected_by_family[family]) for family in FAMILIES)
    result = {
        "schemaVersion": 1,
        "kind": "macro-ranker-v2-heldout-evaluation-v1",
        "evaluatorImplementationSha256": file_sha256(Path(__file__)),
        "identity": selection["identity"],
        "selectionHash": selection["selectionHash"],
        "selectionManifestSha256": file_sha256(selection_path),
        "rankerModelSha256": ranker_report["modelSha256"],
        "rankerReportHash": ranker_report["reportHash"],
        "rankerReportSha256": file_sha256(ranker_report_path.resolve()),
        "trainingSelectionHash": ranker_selection["selectionHash"],
        "trainingSelectionSha256": file_sha256(ranker_selection_path),
        "heldoutSourceRuns": [{"policyFamily": family,
            "runManifestHash": loaded_runs[family][0]["manifestHash"],
            "runManifestSha256": file_sha256(label_runs[family].resolve() / "manifest.json"),
            "datasetManifestHash": loaded_runs[family][0]["datasetManifestHash"],
            "rolloutIdentity": loaded_runs[family][0]["rolloutIdentity"],
            "candidateGeneratorIdentity": expected_generator,
            "labelCollectorVersion": loaded_runs[family][0]["labelCollectorVersion"],
            "labelCollectorSha256": loaded_runs[family][0]["labelCollectorSha256"]}
            for family in FAMILIES],
        "collectorCompatibility": compatibility,
        "sharedRolloutSettings": shared_settings[0],
        "selectedPositions": selected_positions,
        "positions": len(position_details),
        "unsupportedPositions": len(unsupported_positions),
        "unsupportedDetails": unsupported_positions,
        "independentSourceGames": summaries["overall"]["independentSourceGames"],
        "metrics": summaries,
        "details": position_details,
        "status": (summaries["overall"]["status"] if len(position_details) == selected_positions
                   and not unsupported_positions else "insufficient"),
        "interpretation": "descriptive heldout macro-candidate ranking only; source-game clustered uncertainty; not a policy win-rate, supervised-policy, PPO, or promotion result",
        "trainingEligible": False,
        "automaticPromotion": False,
    }
    result["reportHash"] = identity_hash(result)
    _publish_json(output, result)
    return result


def verify_macro_ranker_v2_heldout_report(*, report_path: Path, selection_path: Path,
        ranker_selection_path: Path, model_path: Path, ranker_report_path: Path,
        datasets: dict[str, Path], support_reports: dict[str, Path],
        label_runs: dict[str, Path]) -> dict:
    """Rebuild a heldout report from frozen source artifacts and compare exactly."""
    report_path = Path(report_path)
    if report_path.is_symlink():
        raise ValueError("heldout evaluation report must not be a symlink")
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise ValueError("heldout evaluation report must be a regular non-symlink file")
    report = json.loads(report_path.read_text())
    if (not isinstance(report, dict)
            or report.get("schemaVersion") != 1
            or report.get("kind") != "macro-ranker-v2-heldout-evaluation-v1"
            or report.get("reportHash") != identity_hash(
                {key: value for key, value in report.items() if key != "reportHash"})):
        raise ValueError("heldout evaluation report schema/hash is invalid")
    if report.get("evaluatorImplementationSha256") != file_sha256(Path(__file__)):
        raise ValueError("heldout evaluation report was produced by a different evaluator implementation")
    if (report.get("trainingEligible") is not False
            or report.get("automaticPromotion") is not False
            or not isinstance(report.get("identity"), dict)
            or not isinstance(report.get("selectionHash"), str)
            or len(report["selectionHash"]) != 64
            or not isinstance(report.get("selectionManifestSha256"), str)
            or len(report["selectionManifestSha256"]) != 64
            or not isinstance(report.get("trainingSelectionHash"), str)
            or len(report["trainingSelectionHash"]) != 64
            or not isinstance(report.get("trainingSelectionSha256"), str)
            or len(report["trainingSelectionSha256"]) != 64
            or not isinstance(report.get("rankerModelSha256"), str)
            or len(report["rankerModelSha256"]) != 64
            or not isinstance(report.get("rankerReportHash"), str)
            or len(report["rankerReportHash"]) != 64
            or not isinstance(report.get("rankerReportSha256"), str)
            or len(report["rankerReportSha256"]) != 64):
        raise ValueError("heldout evaluation report is missing frozen provenance or research-only guards")

    source_runs = report.get("heldoutSourceRuns")
    if not isinstance(source_runs, list) or len(source_runs) != len(FAMILIES):
        raise ValueError("heldout evaluation report must preserve both family source runs")
    source_by_family = {}
    generator_identity = None
    versions, hashes = [], []
    rollout_ids = set()
    for source in source_runs:
        if not isinstance(source, dict):
            raise ValueError("heldout source-run receipt is malformed")
        family = source.get("policyFamily")
        collector_version = source.get("labelCollectorVersion")
        collector_hash = source.get("labelCollectorSha256")
        run_id = source.get("rolloutIdentity")
        generator = source.get("candidateGeneratorIdentity")
        for field in ("runManifestHash", "runManifestSha256", "datasetManifestHash"):
            value = source.get(field)
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("heldout source-run receipt has malformed content hashes")
        if (family not in FAMILIES or family in source_by_family
                or not isinstance(collector_version, str) or not collector_version
                or not isinstance(collector_hash, str) or len(collector_hash) != 64
                or not isinstance(run_id, str) or not run_id or run_id in rollout_ids
                or not isinstance(generator, dict)
                or not isinstance(generator.get("version"), str) or not generator["version"]):
            raise ValueError("heldout source-run identity is malformed or duplicated")
        if generator_identity is None:
            generator_identity = generator
        elif generator != generator_identity:
            raise ValueError("heldout source-run candidate generator identities differ")
        rollout_ids.add(run_id)
        source_by_family[family] = source
        versions.append(collector_version)
        hashes.append(collector_hash)
    if set(source_by_family) != set(FAMILIES):
        raise ValueError("heldout evaluation report omits a policy-family run")
    try:
        compatibility = collector_compatibility(versions=versions, source_hashes=hashes)
    except ValueError as error:
        raise ValueError("heldout report collector sources are not compatible") from error
    if report.get("collectorCompatibility") != compatibility:
        raise ValueError("heldout report collector compatibility receipt does not reconcile")
    settings = report.get("sharedRolloutSettings")
    if settings != FROZEN_HELDOUT_SETTINGS:
        raise ValueError("heldout report settings differ from the frozen preflight protocol")

    details = report.get("details")
    unsupported = report.get("unsupportedDetails")
    selected = report.get("selectedPositions")
    positions = report.get("positions")
    unsupported_count = report.get("unsupportedPositions")
    if (not isinstance(details, list) or not isinstance(unsupported, list)
            or type(selected) is not int or selected < 1
            or type(positions) is not int or positions != len(details)
            or type(unsupported_count) is not int or unsupported_count != len(unsupported)
            or positions + unsupported_count != selected):
        raise ValueError("heldout report position coverage counts do not reconcile")
    seen_positions = set()
    for item in details:
        if not isinstance(item, dict):
            raise ValueError("heldout detail row is malformed")
        position_hash = item.get("positionHash")
        family = item.get("opponentPolicyFamily")
        if (not isinstance(position_hash, str) or not position_hash or position_hash in seen_positions
                or family not in FAMILIES or not isinstance(item.get("sourceGameId"), str)
                or not item["sourceGameId"]
                or any(not isinstance(item.get(key), str) or not item[key]
                       for key in ("targetDeck", "opponentArchetype", "positionStage"))
                or type(item.get("candidates")) is not int or item["candidates"] < 2
                or type(item.get("top3Recall")) is not bool):
            raise ValueError("heldout detail position/candidate metadata is malformed")
        regret = item.get("top1RelativeRegret")
        if (isinstance(regret, bool) or not isinstance(regret, (int, float))
                or not math.isfinite(regret) or not 0 <= regret <= 1):
            raise ValueError("heldout detail regret is invalid")
        for key in ("pairwiseCorrect", "pairwiseComparisons", "finishedRollouts",
                    "truncatedRollouts", "engineErrors"):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError("heldout detail outcome/comparison count is invalid")
        if item["pairwiseCorrect"] > item["pairwiseComparisons"]:
            raise ValueError("heldout detail pairwise correct count exceeds comparisons")
        seen_positions.add(position_hash)
    for item in unsupported:
        if (not isinstance(item, dict) or item.get("policyFamily") not in FAMILIES
                or not isinstance(item.get("positionHash"), str) or not item["positionHash"]
                or item["positionHash"] in seen_positions
                or not isinstance(item.get("sourceGameId"), str) or not item["sourceGameId"]
                or not isinstance(item.get("reason"), str) or not item["reason"]):
            raise ValueError("heldout unsupported-position receipt is malformed or duplicated")
        seen_positions.add(item["positionHash"])
    if len(seen_positions) != selected:
        raise ValueError("heldout report repeats or omits selected position hashes")

    expected_metrics = {
        "overall": _summarize(details, seed_material="macro-ranker-v2-heldout|" + report["selectionHash"]),
        "byPolicyFamily": {}, "byTargetDeck": {}, "byOpponentArchetype": {},
    }
    for field, destination in (("opponentPolicyFamily", expected_metrics["byPolicyFamily"]),
                               ("targetDeck", expected_metrics["byTargetDeck"]),
                               ("opponentArchetype", expected_metrics["byOpponentArchetype"])):
        for value in sorted({item[field] for item in details}):
            subset = [item for item in details if item[field] == value]
            destination[value] = _summarize(subset,
                seed_material=f"macro-ranker-v2-heldout|{field}|{value}|{report['selectionHash']}")
    if report.get("metrics") != expected_metrics:
        raise ValueError("heldout aggregate metrics do not recompute from per-position details")
    expected_status = (expected_metrics["overall"]["status"]
                       if positions == selected and unsupported_count == 0 else "insufficient")
    if report.get("status") != expected_status:
        raise ValueError("heldout report status does not follow coverage and metric evidence")
    with tempfile.TemporaryDirectory(prefix="ptcg-heldout-report-audit-") as temporary:
        recomputed_path = Path(temporary) / "recomputed.json"
        recomputed = evaluate_macro_ranker_v2_heldout(
            selection_path=selection_path, ranker_selection_path=ranker_selection_path,
            model_path=model_path, ranker_report_path=ranker_report_path,
            datasets=datasets, support_reports=support_reports,
            label_runs=label_runs, output=recomputed_path)
        if recomputed != report:
            raise ValueError("heldout evaluation report does not reproduce from supplied frozen inputs")
    return {"verified": True, "reportHash": report["reportHash"],
            "selectionHash": report["selectionHash"], "rankerModelSha256": report["rankerModelSha256"],
            "selectedPositions": selected, "positions": positions,
            "unsupportedPositions": unsupported_count, "status": expected_status,
            "trainingEligible": False, "automaticPromotion": False}
