from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

from .dataset_v1 import file_sha256
from .encoding import collate, encode_decision
from .model import StrategyTransformerV1
from .schema import identity_hash
from .training import require_checkpoint_implementation
from .value_targets import load_value_target_dataset


def _metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0, "mse": None, "mae": None, "meanPrediction": None, "meanTarget": None}
    errors = [row["prediction"] - row["target"] for row in rows]
    return {"records": len(rows), "mse": sum(error * error for error in errors) / len(errors),
            "mae": sum(abs(error) for error in errors) / len(errors),
            "meanPrediction": sum(row["prediction"] for row in rows) / len(rows),
            "meanTarget": sum(row["target"] for row in rows) / len(rows)}


def _calibration(rows: list[dict], bins: int = 10) -> dict:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        bounded = min(1.0, max(-1.0, row["prediction"]))
        index = min(bins - 1, int((bounded + 1.0) * .5 * bins))
        groups[index].append({**row, "boundedPrediction": bounded})
    records = []
    weighted_gap = 0.0
    for index, selected in sorted(groups.items()):
        predicted = sum(row["boundedPrediction"] for row in selected) / len(selected)
        observed = sum(row["target"] for row in selected) / len(selected)
        gap = abs(predicted - observed)
        weighted_gap += len(selected) / len(rows) * gap
        records.append({"bin": index, "records": len(selected), "meanPrediction": predicted,
                        "meanObservedOutcome": observed, "absoluteGap": gap})
    return {"bins": records, "expectedCalibrationError": weighted_gap,
            "predictionRange": [-1.0, 1.0], "binCount": bins}


def evaluate_value_head(*, dataset_dir: Path, checkpoint_path: Path, output: Path,
                        split: str = "heldout") -> dict:
    """Evaluate blind value predictions on a frozen non-training source-game split."""
    if split not in {"development", "heldout"}:
        raise ValueError("value evaluation must use development or heldout, never training")
    output = output.resolve()
    if output.exists():
        raise ValueError("value evaluation reports are immutable; choose a new output path")
    manifest, rows = load_value_target_dataset(dataset_dir)
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("value checkpoint artifact is malformed")
    training_config = checkpoint.get("trainingConfig")
    if (checkpoint.get("kind") != "StrategyTransformerV1-supervised"
            or checkpoint.get("identity") != manifest.get("identity")
            or not isinstance(training_config, dict)
            or training_config.get("valueDatasetManifestHash") != manifest.get("manifestHash")):
        raise ValueError("value checkpoint identity or value-dataset hash mismatch")
    require_checkpoint_implementation(checkpoint)
    model = StrategyTransformerV1().eval()
    model.load_state_dict(checkpoint["model"])

    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"value dataset has no {split} rows")
    decisions = []
    for row in selected:
        encoded = encode_decision(row["observation"], row["tracker"])
        if encoded.identity != row["featureIdentityHash"]:
            raise ValueError("value evaluation feature identity drift")
        tensors = {key: torch.as_tensor(value) for key, value in collate([encoded]).items()}
        inputs = {key: tensors[key] for key in
                  ("state_card_ids", "state_features", "state_type_ids", "state_mask")}
        with torch.no_grad():
            prediction = float(model.evaluation_forward(**inputs)[0].item())
        if not math.isfinite(prediction):
            raise ValueError("value head produced a non-finite prediction")
        decisions.append({"positionHash": row["positionHash"], "sourceGameId": row["sourceGameId"],
            "sourceDecisionIndex": row["sourceDecisionIndex"], "actor": row["actor"],
            "target": float(row["valueTarget"]), "prediction": prediction,
            "clippedPrediction": min(1.0, max(-1.0, prediction))})

    per_position: dict[str, list[dict]] = defaultdict(list)
    per_side: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in decisions:
        per_position[row["positionHash"]].append(row)
        per_side[(row["sourceGameId"], row["actor"])].append(row)
    unique_positions = [{"positionHash": key,
        "prediction": sum(row["prediction"] for row in group) / len(group),
        "target": sum(row["target"] for row in group) / len(group), "records": len(group)}
        for key, group in sorted(per_position.items())]
    game_sides = []
    for (game_id, actor), group in sorted(per_side.items()):
        metrics = _metric_rows(group)
        game_sides.append({"sourceGameId": game_id, "actor": actor, **metrics})
    report = {"schemaVersion": 1, "kind": "learning-mind-value-head-evaluation-v1",
        "split": split, "datasetManifestHash": manifest["manifestHash"],
        "datasetManifestSha256": file_sha256(dataset_dir / "manifest.json"),
        "evaluatorSha256": file_sha256(Path(__file__).resolve()),
        "checkpointSha256": file_sha256(checkpoint_path),
        "identity": manifest["identity"], "automaticPromotion": False,
        "perRecord": _metric_rows(decisions),
        "perUniquePosition": _metric_rows(unique_positions),
        "calibration": _calibration(decisions), "gameSides": game_sides,
        "decisions": decisions, "uniquePositions": unique_positions}
    report["reportHash"] = identity_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", delete=False) as temporary:
        temporary.write(json.dumps(report, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report
