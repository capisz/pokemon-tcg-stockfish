"""Inference admission for completed, checksummed local experiments.

Admitted is not promoted: a reviewed-demonstration policy can be selected for
experimental play while its value estimate and strength remain unavailable.
"""
from __future__ import annotations

import re
from pathlib import Path

from .storage import Store, file_digest


def _admitted(store, model_id: str):
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", model_id):
        raise ValueError("Invalid local model identifier")
    experimental = model_id.startswith("experimental-")
    if experimental:
        model_id = model_id[len("experimental-"):]
        if not model_id:
            raise ValueError("Experimental model identifier is missing")
        nested = store.path / "experimental"
        if nested.is_symlink():
            raise ValueError("Experimental data root cannot be a symlink")
        store = Store(nested, store.max_bytes, store.min_free_bytes)
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
            or report.get("type") not in {"policy-value-training", "policy-only-training", "experimental-training"}
            or (report.get("dataTier") == "experimental") != experimental):
        raise ValueError("Checkpoint training has not completed")
    expected_hash = report.get("modelHash")
    if not expected_hash or file_digest(path) != expected_hash:
        raise ValueError("Checkpoint does not match its completed experiment checksum")
    from .training import load_model
    _, checkpoint = load_model(path, allow_experimental=experimental)
    if (checkpoint.get("experimentId") != model_id or checkpoint.get("config") != report.get("config")
            or checkpoint["dataLineage"]["hash"] != report.get("dataLineageHash")
            or file_digest(path) != expected_hash or (checkpoint.get("dataTier") == "experimental") != experimental):
        raise ValueError("Checkpoint identity, provenance or bytes changed during admission")
    return path, report, checkpoint


def resolve_model(store, model_id: str) -> Path:
    """Resolve an opaque ID. Callers then freeze and recheck bytes for each run."""
    return _admitted(store, model_id)[0]


def list_models(store) -> list[dict]:
    result = []
    nested = Store(store.path / "experimental", store.max_bytes, store.min_free_bytes)
    sources = [(False, report) for report in store.iter_records("experiments")]
    if not nested.path.is_symlink():
        sources.extend((True, report) for report in nested.iter_records("experiments"))
    for experimental, report in sources:
        if report.get("type") not in {"policy-value-training", "policy-only-training", "experimental-training"} or report.get("status") != "completed":
            continue
        identifier = f"experimental-{report['id']}" if experimental else report["id"]
        try:
            _, report, checkpoint = _admitted(store, identifier)
        except (ValueError, OSError, RuntimeError, KeyError, EOFError):
            continue
        kind = checkpoint.get("modelKind", "policy-value")
        result.append({"id": identifier, "name": f"{'Experimental ' if experimental else ''}{kind} · {report['id'][:8]}", "modelKind": kind,
                       "dataTier": "experimental" if experimental else "verified",
                       "modelHash": report["modelHash"], "featureVersion": checkpoint["config"]["featureVersion"],
                       "parameters": checkpoint.get("parameters"), "valueTrained": kind == "policy-value",
                       "calibrated": checkpoint.get("calibration") is not None,
                       "demonstrationExamples": report.get("demonstrationExamples", 0),
                       "policyTarget": checkpoint.get("policyTarget"), "status": "experimental",
                       "description": ("Quarantined experimental learning; unverified rules, no calibrated probabilities or trusted champion status."
                                       if experimental else "Admitted for local inference; strength requires held-out evaluation.")})
    return sorted(result, key=lambda item: item["id"])
