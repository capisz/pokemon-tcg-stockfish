from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from .dataset_v1 import file_sha256, load_dataset
from .schema import identity_hash


def select_supported_source_game_positions(rows: list[dict], supported_hashes: set[str], *,
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
        supported = {item["positionHash"] for item in support.get("positions", [])
                     if item.get("status") == "supported"}
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
        "selection": "supported-one-position-per-source-game-greedy-deck-opponent-stage-v1",
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
