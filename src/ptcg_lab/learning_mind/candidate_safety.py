from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import torch

from .dataset_v1 import file_sha256, load_dataset
from .disagreement_review import _read_verified_audit
from .encoding import collate, encode_decision
from .model import StrategyTransformerV1, autoregressive_select
from .schema import LIMITS, identity_hash
from .training import require_checkpoint_implementation


def _atomic_new_json(output: Path, value: dict) -> None:
    output = output.resolve()
    if output.exists():
        raise ValueError("candidate safety reports are immutable; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", delete=False) as temporary:
        temporary.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_legal_action_coverage(legal_actions: list[dict], action_classes: list) -> int:
    legal_ids = [action.get("id") for action in legal_actions if isinstance(action, dict)]
    if len(legal_ids) != len(legal_actions) or any(not isinstance(value, str) or not value for value in legal_ids):
        raise ValueError("legal engine actions have invalid or missing IDs")
    if len(legal_ids) != len(set(legal_ids)):
        raise ValueError("legal engine action IDs are not unique")
    represented_ids = [action.get("id") for group in action_classes for action in group.actions]
    if Counter(represented_ids) != Counter(legal_ids):
        raise ValueError("encoded action classes omit or duplicate a legal engine action")
    return len(legal_ids)


def audit_candidate_safety(*, dataset_dir: Path, checkpoint: Path,
                           evaluation_path: Path, audit_path: Path,
                           output: Path) -> dict:
    """Recompute held-out decisions and audit legal-option and one-step decoder safety."""
    audit, evaluation, ordered_rows = _read_verified_audit(dataset_dir=dataset_dir,
        checkpoint=checkpoint, evaluation_path=evaluation_path, audit_path=audit_path)
    manifest, _rows = load_dataset(dataset_dir)
    checkpoint_record = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint_record.get("identity") != manifest.get("identity")
            or checkpoint_record.get("datasetManifestHash") != manifest.get("manifestHash")
            or checkpoint_record.get("kind") != "StrategyTransformerV1-supervised"):
        raise ValueError("candidate checkpoint identity differs from the audited supervised artifact")
    require_checkpoint_implementation(checkpoint_record)
    model = StrategyTransformerV1().eval()
    model.load_state_dict(checkpoint_record["model"])
    max_action_classes = 0
    evaluated = {item["positionHash"]: item for item in evaluation["positions"]}
    action_class_counts = Counter()
    legal_actions_checked = 0
    for source in ordered_rows:
        if source.get("split") == "train":
            raise ValueError("audited evaluation unexpectedly includes training data")
        observation, tracker = source.get("observation"), source.get("tracker")
        if (not isinstance(observation, dict) or observation.get("playerId") != source.get("actor")
                or not isinstance(tracker, dict)):
            raise ValueError("candidate safety audit requires the actor-visible observation and tracker")
        legal_actions = observation.get("legalActions")
        if not isinstance(legal_actions, list) or not legal_actions:
            raise ValueError("evaluated decision has no legal engine actions")
        encoded = encode_decision(observation, tracker)
        count = len(encoded.action_classes)
        if not 0 < count <= LIMITS["legalActionClasses"]:
            raise ValueError("encoded legal-action class count is outside the frozen cap")
        legal_count = _verify_legal_action_coverage(legal_actions, encoded.action_classes)
        max_action_classes = max(max_action_classes, count)
        legal_actions_checked += legal_count

        arrays = collate([encoded])
        tensors = {key: torch.as_tensor(value) for key, value in arrays.items()}
        with torch.no_grad():
            logits = model.policy_forward(**tensors)[0]
            model_class = int(torch.argmax(logits).item())
        record = evaluated[source["positionHash"]]
        if model_class != record.get("modelClass") or not 0 <= model_class <= count:
            raise ValueError("candidate output differs from frozen checkpoint or leaves legal options")

        def step_logits(chosen: tuple[int, ...]) -> torch.Tensor:
            selected = torch.zeros_like(tensors["option_mask"])
            for index in chosen:
                selected[0, index] = True
            with torch.no_grad():
                return model.policy_forward(**tensors, selected_mask=selected)[0]

        decoded = autoregressive_select(step_logits, action_count=count, minimum=1, maximum=1,
                                        legality=lambda _chosen, index: 0 <= index < count)
        if (not decoded.stopped or len(decoded.indices) != 1
                or not 0 <= decoded.indices[0] < count):
            raise ValueError("autoregressive decoder emitted an illegal or incomplete one-action choice")
        action_class_counts[str(count)] += 1

    report = {"schemaVersion": 1, "kind": "supervised-candidate-safety-v1",
        "datasetManifestHash": audit["datasetManifestHash"],
        "checkpointSha256": audit["checkpointSha256"],
        "evaluationSha256": audit["evaluationSha256"],
        "supervisedAuditHash": audit["reportHash"],
        "evaluationIdentityStatus": "matched",
        "positionsChecked": len(ordered_rows), "legalActionsChecked": legal_actions_checked,
        "actionClassCountDistribution": dict(sorted(action_class_counts.items(), key=lambda item: int(item[0]))),
        "maximumActionClasses": max_action_classes,
        "decoderPolicy": "one engine action; autoregressive bounds minimum=1 maximum=1",
        "legalActionOmission": False,
        "illegalAutoregressiveSelection": False,
        "capOverflow": False,
        "automaticPromotion": False}
    report["reportHash"] = identity_hash(report)
    _atomic_new_json(output, report)
    return report
