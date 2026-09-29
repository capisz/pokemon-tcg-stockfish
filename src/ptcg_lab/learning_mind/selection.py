from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from .dataset_v1 import file_sha256, load_dataset
from .schema import identity_hash


MIN_COMPLETE_CANDIDATES_PER_POSITION = 2


def eligible_multi_candidate_hashes(support_positions: list[dict]) -> set[str]:
    """Keep only roots with at least two distinct complete executable plans."""
    if not isinstance(support_positions, list):
        raise ValueError("candidate support positions must be a list")
    seen = set()
    eligible = set()
    for item in support_positions:
        if not isinstance(item, dict):
            raise ValueError("candidate support position is malformed")
        position_hash = item.get("positionHash")
        if not isinstance(position_hash, str) or not position_hash or position_hash in seen:
            raise ValueError("candidate support position hashes must be unique and nonempty")
        seen.add(position_hash)
        if item.get("status") == "supported":
            count = item.get("completeCandidateCount")
            if type(count) is not int or count < 1:
                raise ValueError("supported candidate root has an invalid complete-candidate count")
            if count >= MIN_COMPLETE_CANDIDATES_PER_POSITION:
                eligible.add(position_hash)
    return eligible


def _select_supported_source_game_positions(rows: list[dict], supported_hashes: set[str], *,
                                           split: str, pinned_hashes: list[str] | None = None) -> list[dict]:
    """Select one supported position per source game, balancing deck/opponent/stage."""
    if split not in {"train", "development"}:
        raise ValueError("label selection only accepts train or development; heldout is intentionally excluded")
    if not isinstance(supported_hashes, set) or any(not isinstance(item, str) or not item for item in supported_hashes):
        raise ValueError("supported position hashes must be a set of nonempty strings")
    by_hash = {row.get("positionHash"): row for row in rows}
    if len(by_hash) != len(rows) or None in by_hash:
        raise ValueError("dataset rows must have unique position hashes")
    eligible = {key: row for key, row in by_hash.items()
                if key in supported_hashes and row.get("split") == split}
    if not eligible:
        raise ValueError(f"no supported dataset rows for {split}")
    pinned_hashes = pinned_hashes or []
    if len(pinned_hashes) != len(set(pinned_hashes)):
        raise ValueError("pinned position hashes must be unique")
    selected = []
    used_games = set()
    counts = {field: Counter() for field in ("targetDeck", "opponentArchetype", "positionStage")}
    for key in pinned_hashes:
        row = eligible.get(key)
        if row is None:
            raise ValueError(f"pinned position is not supported in the requested split: {key}")
        game = row.get("sourceGameId")
        if not isinstance(game, str) or not game or game in used_games:
            raise ValueError("pinned positions must come from distinct source games")
        used_games.add(game)
        selected.append(row)
        for field in counts:
            counts[field][str(row.get(field, "unknown"))] += 1
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in eligible.values():
        game = row.get("sourceGameId")
        if not isinstance(game, str) or not game:
            raise ValueError("eligible dataset row lacks a source game ID")
        if game not in used_games:
            groups[game].append(row)
    while groups:
        choices = [(row, game) for game, group in groups.items() for row in group]
        row, game = min(choices, key=lambda pair: (
            counts["targetDeck"][str(pair[0].get("targetDeck", "unknown"))],
            counts["opponentArchetype"][str(pair[0].get("opponentArchetype", "unknown"))],
            counts["positionStage"][str(pair[0].get("positionStage", "unknown"))],
            pair[0]["positionHash"], game,
        ))
        selected.append(row)
        used_games.add(game)
        del groups[game]
        for field in counts:
            counts[field][str(row.get(field, "unknown"))] += 1
    return selected


def select_supported_source_game_positions(rows: list[dict], supported_hashes: set[str], *,
                                          split: str, pinned_hashes: list[str] | None = None) -> list[dict]:
    """Select train/development rows only; heldout selection has a separate API."""
    return _select_supported_source_game_positions(
        rows, supported_hashes, split=split, pinned_hashes=pinned_hashes)


def freeze_macro_label_selection(*, datasets: dict[str, Path], support_reports: dict[str, Path],
                                 output: Path, identity: dict,
                                 pinned_train_hashes: dict[str, list[str]] | None = None) -> dict:
    """Freeze source-game-disjoint train/development representatives; never select heldout rows."""
    expected = {"python-heuristic", "typescript-heuristic"}
    if set(datasets) != expected or set(support_reports) != expected:
        raise ValueError("selection requires both frozen policy-family pools and support audits")
    output = output.resolve()
    if output.exists():
        raise ValueError("selection manifests are immutable; choose a new output path")
    source_records, selections = [], []
    for family in sorted(expected):
        directory = datasets[family].resolve()
        manifest, rows = load_dataset(directory, identity=identity)
        support_path = support_reports[family].resolve()
        support = json.loads(support_path.read_text())
        if support.get("identity") != identity or support.get("datasetManifestHash") != manifest.get("manifestHash"):
            raise ValueError(f"support report identity/dataset mismatch for {family}")
        if support.get("datasetRowsSha256") != file_sha256(directory / "rows.jsonl"):
            raise ValueError(f"support report rows hash mismatch for {family}")
        if support.get("reportHash") != identity_hash({key: value for key, value in support.items()
                                                       if key != "reportHash"}):
            raise ValueError(f"support report hash mismatch for {family}")
        if any(row.get("opponentPolicyFamily") != family for row in rows):
            raise ValueError(f"dataset contains a different policy family than {family}")
        supported = eligible_multi_candidate_hashes(support.get("positions"))
        for split in ("train", "development"):
            selected = select_supported_source_game_positions(
                rows, supported, split=split,
                pinned_hashes=(pinned_train_hashes or {}).get(family, []) if split == "train" else None)
            selections.append({
                "policyFamily": family, "split": split, "sourceGames": len(selected),
                "pinnedExistingLabels": len((pinned_train_hashes or {}).get(family, [])) if split == "train" else 0,
                "positionHashes": [row["positionHash"] for row in selected],
                "positions": [{key: row.get(key) for key in (
                    "positionHash", "sourceGameId", "targetDeck", "opponentArchetype", "positionStage")}
                    for row in selected],
                "countsByTargetDeck": dict(sorted(Counter(row["targetDeck"] for row in selected).items())),
                "countsByOpponent": dict(sorted(Counter(row["opponentArchetype"] for row in selected).items())),
                "countsByStage": dict(sorted(Counter(row["positionStage"] for row in selected).items())),
                "completeCandidates": sum(item.get("completeCandidateCount", 0)
                    for item in support["positions"]
                    if item.get("positionHash") in {row["positionHash"] for row in selected}),
            })
        source_records.append({
            "policyFamily": family,
            "datasetManifestHash": manifest["manifestHash"],
            "datasetManifestSha256": file_sha256(directory / "manifest.json"),
            "datasetRowsSha256": file_sha256(directory / "rows.jsonl"),
            "supportReportHash": support["reportHash"],
            "supportReportSha256": file_sha256(support_path),
        })
    result = {
        "schemaVersion": 1,
        "selection": "supported-multi-candidate-one-position-per-source-game-greedy-deck-opponent-stage-v2",
        "minimumCompleteCandidatesPerPosition": MIN_COMPLETE_CANDIDATES_PER_POSITION,
        "identity": identity,
        "sourcePools": source_records,
        "splits": selections,
        "heldoutPolicy": "not selected; all heldout rows remain reserved for later frozen evaluation",
        "interpretation": "selection only; no rollouts, labels, ranker fitting, or policy updates",
    }
    result["selectionHash"] = identity_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(output)
    return result


def freeze_macro_label_training_repair(*, datasets: dict[str, Path],
                                       support_reports: dict[str, Path],
                                       parent_selection_path: Path, output: Path,
                                       identity: dict) -> dict:
    """Draft a same-game training-root repair without changing development roots.

    Existing train positions with at least two supported candidates are pinned.
    A train position that fails that criterion is replaced only by a supported
    position from the exact same source game. The parent selection is never
    modified, and this draft is not itself label or ranker evidence.
    """
    expected = {"python-heuristic", "typescript-heuristic"}
    if set(datasets) != expected or set(support_reports) != expected:
        raise ValueError("training repair requires both frozen policy-family pools and support audits")
    from .aggregation import load_frozen_selection

    parent_path = Path(parent_selection_path).resolve()
    parent, _ = load_frozen_selection(parent_path, identity=identity)
    if output.exists() or output.is_symlink():
        raise ValueError("training repair selections are immutable; choose a new output path")
    output = output.resolve()
    source_records, repaired_splits, replacements = [], [], []
    for family in sorted(expected):
        directory = Path(datasets[family]).resolve()
        manifest, rows = load_dataset(directory, identity=identity)
        support_path = Path(support_reports[family]).resolve()
        support = json.loads(support_path.read_text())
        if (support.get("identity") != identity
                or support.get("datasetManifestHash") != manifest.get("manifestHash")
                or support.get("datasetRowsSha256") != file_sha256(directory / "rows.jsonl")
                or support.get("reportHash") != identity_hash(
                    {key: value for key, value in support.items() if key != "reportHash"})):
            raise ValueError(f"training repair support audit identity mismatch for {family}")
        if any(row.get("opponentPolicyFamily") != family for row in rows):
            raise ValueError(f"training repair dataset contains another policy family: {family}")
        eligible_hashes = eligible_multi_candidate_hashes(support.get("positions"))
        support_by_hash = {row["positionHash"]: row for row in support["positions"]}
        row_by_hash = {row["positionHash"]: row for row in rows}
        if len(row_by_hash) != len(rows):
            raise ValueError(f"training repair dataset repeats positions for {family}")
        family_splits = [item for item in parent["splits"] if item["policyFamily"] == family]
        if {item["split"] for item in family_splits} != {"train", "development"}:
            raise ValueError(f"parent selection lacks train/development splits for {family}")
        source_games_by_split = {}
        for split in ("train", "development"):
            source = next(item for item in family_splits if item["split"] == split)
            parent_rows = source.get("positions")
            if (not isinstance(parent_rows, list)
                    or [row.get("positionHash") for row in parent_rows] != source["positionHashes"]
                    or len({row.get("sourceGameId") for row in parent_rows}) != len(parent_rows)):
                raise ValueError(f"parent selection position provenance is invalid for {family}/{split}")
            if split == "development":
                for position in parent_rows:
                    row = row_by_hash.get(position["positionHash"])
                    if (row is None or row.get("split") != "development"
                            or any(row.get(key) != position.get(key) for key in (
                                "sourceGameId", "targetDeck", "opponentArchetype", "positionStage"))
                            or position["positionHash"] not in eligible_hashes):
                        raise ValueError("training repair cannot alter or retain unsupported development roots")
                repaired = parent_rows
            else:
                retained, missing = [], []
                used_games = set()
                counts = {field: Counter() for field in
                          ("targetDeck", "opponentArchetype", "positionStage")}
                for position in parent_rows:
                    row = row_by_hash.get(position["positionHash"])
                    if (row is None or row.get("split") != "train"
                            or row.get("sourceGameId") != position.get("sourceGameId")
                            or any(row.get(key) != position.get(key) for key in (
                                "targetDeck", "opponentArchetype", "positionStage"))):
                        raise ValueError("parent training root differs from its frozen dataset row")
                    game_id = row.get("sourceGameId")
                    if game_id in used_games:
                        raise ValueError("parent training selection repeats a source game")
                    used_games.add(game_id)
                    if position["positionHash"] in eligible_hashes:
                        retained.append(row)
                        for field in counts:
                            counts[field][str(row.get(field, "unknown"))] += 1
                    else:
                        missing.append(row)
                repaired = list(retained)
                used_positions = {row["positionHash"] for row in retained}
                for old_row in missing:
                    alternatives = [row for row in rows
                        if row.get("split") == "train"
                        and row.get("sourceGameId") == old_row.get("sourceGameId")
                        and row["positionHash"] in eligible_hashes
                        and row["positionHash"] not in used_positions]
                    if not alternatives:
                        raise ValueError(
                            "unsupported training root has no multi-candidate alternative in the same source game")
                    replacement = min(alternatives, key=lambda row: (
                        counts["targetDeck"][str(row.get("targetDeck", "unknown"))],
                        counts["opponentArchetype"][str(row.get("opponentArchetype", "unknown"))],
                        counts["positionStage"][str(row.get("positionStage", "unknown"))],
                        row["positionHash"]))
                    used_positions.add(replacement["positionHash"])
                    repaired.append(replacement)
                    for field in counts:
                        counts[field][str(replacement.get(field, "unknown"))] += 1
                    replacements.append({
                        "policyFamily": family, "split": "train",
                        "sourceGameId": old_row["sourceGameId"],
                        "supersededPositionHash": old_row["positionHash"],
                        "replacementPositionHash": replacement["positionHash"],
                        "replacementCompleteCandidateCount": support_by_hash[
                            replacement["positionHash"]]["completeCandidateCount"],
                        "reason": "parent root has fewer than two complete executable candidates",
                    })
                if len(repaired) != len(parent_rows) or {row["sourceGameId"] for row in repaired} != used_games:
                    raise ValueError("training repair changed source-game coverage")
            source_games_by_split[split] = {row["sourceGameId"] for row in repaired}

            hashes = [row["positionHash"] for row in repaired]
            repaired_splits.append({
                "policyFamily": family, "split": split, "sourceGames": len(repaired),
                "pinnedExistingLabels": len(repaired) - len(missing) if split == "train" else 0,
                "positionHashes": hashes,
                "positions": [{key: row.get(key) for key in (
                    "positionHash", "sourceGameId", "targetDeck", "opponentArchetype", "positionStage")}
                    for row in repaired],
                "countsByTargetDeck": dict(sorted(Counter(row["targetDeck"] for row in repaired).items())),
                "countsByOpponent": dict(sorted(Counter(row["opponentArchetype"] for row in repaired).items())),
                "countsByStage": dict(sorted(Counter(row["positionStage"] for row in repaired).items())),
                "completeCandidates": sum(support_by_hash[row["positionHash"]]["completeCandidateCount"]
                    for row in repaired),
            })
        if source_games_by_split["train"].intersection(source_games_by_split["development"]):
            raise ValueError("training repair source games overlap development games")
        source_records.append({
            "policyFamily": family,
            "datasetManifestHash": manifest["manifestHash"],
            "datasetManifestSha256": file_sha256(directory / "manifest.json"),
            "datasetRowsSha256": file_sha256(directory / "rows.jsonl"),
            "supportReportHash": support["reportHash"],
            "supportReportSha256": file_sha256(support_path),
        })

    lineage = {
        "kind": "same-source-game-train-root-repair-v1",
        "parentSelectionHash": parent["selectionHash"],
        "parentSelectionManifestSha256": file_sha256(parent_path),
        "replacements": replacements,
        "developmentPositionHashesUnchanged": True,
        "heldoutPositionsSelected": False,
        "rawParentLabelsMustRemainImmutable": True,
    }
    result = {
        "schemaVersion": 1,
        "selection": "candidate-supported-source-game-preserving-train-repair-v1",
        "minimumCompleteCandidatesPerPosition": MIN_COMPLETE_CANDIDATES_PER_POSITION,
        "identity": identity,
        "sourcePools": source_records,
        "splits": repaired_splits,
        "lineage": lineage,
        "status": "draft-unlabeled-lineage-merge-required",
        "interpretation": "selection proposal only; no labels, fitting, or policy updates",
    }
    result["selectionHash"] = identity_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(output)
    return result


def freeze_macro_heldout_selection(*, datasets: dict[str, Path], support_reports: dict[str, Path],
                                   output: Path, identity: dict) -> dict:
    """Freeze heldout-only evaluator positions without making them train labels."""
    expected = {"python-heuristic", "typescript-heuristic"}
    if set(datasets) != expected or set(support_reports) != expected:
        raise ValueError("heldout selection requires both frozen policy-family pools and support audits")
    output = output.resolve()
    if output.exists():
        raise ValueError("heldout selection manifests are immutable; choose a new output path")

    source_records, selections = [], []
    for family in sorted(expected):
        directory = datasets[family].resolve()
        manifest, rows = load_dataset(directory, identity=identity)
        support_path = support_reports[family].resolve()
        support = json.loads(support_path.read_text())
        if support.get("identity") != identity or support.get("datasetManifestHash") != manifest.get("manifestHash"):
            raise ValueError(f"heldout support report identity/dataset mismatch for {family}")
        if support.get("datasetRowsSha256") != file_sha256(directory / "rows.jsonl"):
            raise ValueError(f"heldout support report rows hash mismatch for {family}")
        if support.get("reportHash") != identity_hash({key: value for key, value in support.items()
                                                       if key != "reportHash"}):
            raise ValueError(f"heldout support report hash mismatch for {family}")
        if any(row.get("opponentPolicyFamily") != family for row in rows):
            raise ValueError(f"heldout dataset contains a different policy family than {family}")

        supported = eligible_multi_candidate_hashes(support.get("positions"))
        selected = sorted((row for row in rows
                           if row.get("split") == "heldout"
                           and row.get("positionHash") in supported),
                          key=lambda row: row["positionHash"])
        if not selected:
            raise ValueError(f"no multi-candidate heldout rows for {family}")
        source_game_ids = sorted({row.get("sourceGameId") for row in selected})
        if any(not isinstance(game_id, str) or not game_id for game_id in source_game_ids):
            raise ValueError(f"heldout rows lack source-game provenance for {family}")
        selections.append({
            "policyFamily": family,
            "split": "heldout",
            "sourceGames": len(source_game_ids),
            "sourceGameIds": source_game_ids,
            "positionHashes": [row["positionHash"] for row in selected],
            "positions": [{key: row.get(key) for key in (
                "positionHash", "sourceGameId", "targetDeck", "opponentArchetype", "positionStage")}
                for row in selected],
            "countsByTargetDeck": dict(sorted(Counter(row["targetDeck"] for row in selected).items())),
            "countsByOpponent": dict(sorted(Counter(row["opponentArchetype"] for row in selected).items())),
            "countsByStage": dict(sorted(Counter(row["positionStage"] for row in selected).items())),
            "completeCandidates": sum(item.get("completeCandidateCount", 0)
                for item in support["positions"]
                if item.get("positionHash") in {row["positionHash"] for row in selected}),
        })
        source_records.append({
            "policyFamily": family,
            "datasetManifestHash": manifest["manifestHash"],
            "datasetManifestSha256": file_sha256(directory / "manifest.json"),
            "datasetRowsSha256": file_sha256(directory / "rows.jsonl"),
            "supportReportHash": support["reportHash"],
            "supportReportSha256": file_sha256(support_path),
        })

    for item in selections:
        if len({position["sourceGameId"] for position in item["positions"]}) != item["sourceGames"]:
            raise ValueError("heldout selection contains duplicate source games")
    result = {
        "schemaVersion": 1,
        "kind": "macro-ranker-heldout-selection-v1",
        "selection": "all-supported-multi-candidate-heldout-positions-source-game-clustered-v1",
        "minimumCompleteCandidatesPerPosition": MIN_COMPLETE_CANDIDATES_PER_POSITION,
        "identity": identity,
        "sourcePools": source_records,
        "splits": selections,
        "trainingEligible": False,
        "interpretation": "heldout evaluator selection only; no rollouts, labels, ranker fitting, or policy updates",
    }
    result["selectionHash"] = identity_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(output)
    return result
