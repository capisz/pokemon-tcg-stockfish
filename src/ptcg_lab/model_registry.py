"""Inference admission for completed, checksummed local experiments.

Admitted is not promoted: a reviewed-demonstration policy can be selected for
experimental play while its value estimate and strength remain unavailable.
"""
from __future__ import annotations

import re
from pathlib import Path

from .storage import file_digest


def _admitted(store, model_id: str):
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", model_id):
        raise ValueError("Invalid local model identifier")
    path = store.path / "models" / f"{model_id}.pt"
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("Requested model is not an admitted local checkpoint")
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Checkpoint exceeds the local model size bound")
    try:
        report = store.get("experiments", model_id)
    except FileNotFoundError as exc:
        raise ValueError("Checkpoint has no completed training experiment") from exc
    if (report.get("id") != model_id or report.get("status") != "completed"
            or report.get("type") not in {"policy-value-training", "policy-only-training"}):
        raise ValueError("Checkpoint training has not completed")
    expected_hash = report.get("modelHash")
    if not expected_hash or file_digest(path) != expected_hash:
        raise ValueError("Checkpoint does not match its completed experiment checksum")
    from .training import load_model
    _, checkpoint = load_model(path)
    if (checkpoint.get("experimentId") != model_id or checkpoint.get("config") != report.get("config")
            or checkpoint["dataLineage"]["hash"] != report.get("dataLineageHash")
            or file_digest(path) != expected_hash):
        raise ValueError("Checkpoint identity, provenance or bytes changed during admission")
    return path, report, checkpoint


def resolve_model(store, model_id: str) -> Path:
    """Resolve an opaque ID. Callers then freeze and recheck bytes for each run."""
    return _admitted(store, model_id)[0]


def list_models(store) -> list[dict]:
    result = []
    for report in store.iter_records("experiments"):
        if report.get("type") not in {"policy-value-training", "policy-only-training"} or report.get("status") != "completed":
            continue
        try:
            _, report, checkpoint = _admitted(store, report["id"])
        except (ValueError, OSError, RuntimeError, KeyError, EOFError):
            continue
        kind = checkpoint.get("modelKind", "policy-value")
        result.append({"id": report["id"], "name": f"{kind} · {report['id'][:8]}", "modelKind": kind,
                       "modelHash": report["modelHash"], "featureVersion": checkpoint["config"]["featureVersion"],
                       "parameters": checkpoint.get("parameters"), "valueTrained": kind == "policy-value",
                       "calibrated": checkpoint.get("calibration") is not None,
                       "demonstrationExamples": report.get("demonstrationExamples", 0),
                       "policyTarget": checkpoint.get("policyTarget"), "status": "experimental",
                       "description": "Admitted for local inference; strength requires held-out evaluation."})
    return sorted(result, key=lambda item: item["id"])
