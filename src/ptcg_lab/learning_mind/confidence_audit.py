from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .dataset_v1 import file_sha256
from .experiment import _load_ranker_input
from .ranker_v2 import _validated_macro_labels
from .schema import identity_hash


def _position_confidence(record: dict, *, familywise_alpha: float = 0.05) -> dict:
    """Re-evaluate the plausible-best candidate set from verified finished rollouts.

    The Bonferroni-adjusted Hoeffding intervals cover every candidate in this
    position simultaneously. They are conditional on finished rollouts; the
    report keeps truncations and errors visible and does not impute outcomes.
    """
    if (not isinstance(familywise_alpha, (int, float)) or isinstance(familywise_alpha, bool)
            or not math.isfinite(familywise_alpha) or not 0 < familywise_alpha < 1):
        raise ValueError("family-wise alpha must be finite and between zero and one")
    labels = record.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("confidence reevaluation requires a non-empty candidate set")
    candidate_hashes = [label.get("candidateHash") for label in labels]
    if (any(not isinstance(value, str) or not value for value in candidate_hashes)
            or len(candidate_hashes) != len(set(candidate_hashes))):
        raise ValueError("confidence reevaluation found missing or duplicate candidate hashes")
    completed_labels = _validated_macro_labels(record)
    completed_by_hash = {label["candidateHash"]: label for label in completed_labels}

    per_candidate_alpha = familywise_alpha / len(labels)
    log_term = math.log(2.0 / per_candidate_alpha)
    candidates = []
    for label in labels:
        candidate_hash = label["candidateHash"]
        outcomes = label["outcomes"]
        n = outcomes["finished"]
        mean = label["expectedResult"]
        if n == 0:
            lower, upper = 0.0, 1.0
        else:
            if candidate_hash not in completed_by_hash or mean is None:
                raise ValueError("completed rollout summary is missing its verified candidate")
            radius = math.sqrt(log_term / (2 * n))
            lower, upper = max(0.0, float(mean) - radius), min(1.0, float(mean) + radius)
        attempted = label["attemptedRollouts"]
        candidates.append({
            "candidateHash": candidate_hash,
            "completed": n,
            "attempted": attempted,
            "truncated": outcomes["truncated"],
            "errors": outcomes["error"],
            "completionRate": n / attempted,
            "expectedResultAmongFinished": mean,
            "interval95": {"low": lower, "high": upper},
            "sampleStatus": "insufficient" if n < 20 else "measured",
        })

    observed = [item for item in candidates if item["completed"] > 0]
    if not observed:
        plausible_best = []
        status = "no-completed-rollouts"
    else:
        highest_lower = max(item["interval95"]["low"] for item in candidates)
        plausible_best = sorted(item["candidateHash"] for item in candidates
                                if item["interval95"]["high"] >= highest_lower)
        status = "uniquely-separated" if len(plausible_best) == 1 else "ambiguous"
    return {
        "positionHash": record["positionHash"],
        "policyFamily": record["opponentPolicyFamily"],
        "split": record["split"],
        "candidateCount": len(candidates),
        "completed": sum(item["completed"] for item in candidates),
        "attempted": sum(item["attempted"] for item in candidates),
        "truncated": sum(item["truncated"] for item in candidates),
        "errors": sum(item["errors"] for item in candidates),
        "confidenceStatus": status,
        "plausibleBestCandidateHashes": plausible_best,
        "candidates": candidates,
    }


def _publish_immutable(output: Path, report: dict) -> None:
    output = output.resolve()
    if output.exists():
        raise ValueError("confidence audit reports are immutable; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=f".{output.name}.", suffix=".tmp", delete=False) as temporary:
        json.dump(report, temporary, sort_keys=True, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def audit_macro_label_confidence(*, labels_dir: Path, selection_path: Path,
                                 output: Path, familywise_alpha: float = 0.05) -> dict:
    """Verify a finalized combined label bundle and publish an analysis-only audit."""
    output = output.resolve()
    if output.exists():
        raise ValueError("confidence audit reports are immutable; choose a new output path")
    manifest, records = _load_ranker_input(labels_dir, selection_path)
    report = _build_report(labels_dir=labels_dir, selection_path=selection_path,
        manifest=manifest, records=records, familywise_alpha=familywise_alpha)
    report["reportHash"] = identity_hash(report)
    _publish_immutable(output, report)
    return report


def _build_report(*, labels_dir: Path, selection_path: Path, manifest: dict,
                  records: list[dict], familywise_alpha: float) -> dict:
    positions = [_position_confidence(record, familywise_alpha=familywise_alpha)
                 for record in records]
    positions.sort(key=lambda item: (item["policyFamily"], item["split"], item["positionHash"]))
    grouped = {}
    for family in ("python-heuristic", "typescript-heuristic"):
        for split in ("train", "development"):
            selected = [item for item in positions
                        if item["policyFamily"] == family and item["split"] == split]
            grouped[f"{family}/{split}"] = {
                "positions": len(selected),
                "uniquelySeparated": sum(item["confidenceStatus"] == "uniquely-separated" for item in selected),
                "ambiguous": sum(item["confidenceStatus"] == "ambiguous" for item in selected),
                "noCompletedRollouts": sum(item["confidenceStatus"] == "no-completed-rollouts" for item in selected),
                "finished": sum(item["completed"] for item in selected),
                "truncated": sum(item["truncated"] for item in selected),
                "errors": sum(item["errors"] for item in selected),
            }
    return {
        "schemaVersion": 1,
        "kind": "macro-label-confidence-audit-v1",
        "status": "analysis-only",
        "intervalMethod": "bonferroni-simultaneous-hoeffding-bounded-finished-outcomes-v1",
        "familywiseAlpha": familywise_alpha,
        "familywiseConfidence": 1.0 - familywise_alpha,
        "confidenceScope": "candidate set within each position; not simultaneous across positions",
        "interpretation": "Intervals are conditional on finished rollouts; truncations and errors are reported separately, not imputed.",
        "policyLabelEligibilityChanged": False,
        "ppoEnablement": False,
        "promotionAuthority": "none",
        "inputManifestHash": manifest["manifestHash"],
        "inputManifestSha256": file_sha256(labels_dir.resolve() / "manifest.json"),
        "selectionHash": manifest["selectionHash"],
        "selectionManifestSha256": file_sha256(selection_path.resolve()),
        "confidenceAuditImplementationSha256": file_sha256(Path(__file__)),
        "collectorIdentity": {
            "labelCollectorVersion": manifest["labelCollectorVersion"],
            "labelCollectorSha256": manifest["labelCollectorSha256"],
            "candidateGeneratorIdentity": manifest["candidateGeneratorIdentity"],
            "sourceRuns": manifest["sourceRuns"],
        },
        "sourceRecords": manifest["files"],
        "summaryByFamilyAndSplit": grouped,
        "positions": positions,
    }


def verify_macro_label_confidence_audit(*, labels_dir: Path, selection_path: Path,
                                        report_path: Path) -> dict:
    claimed = json.loads(report_path.read_text())
    if (not isinstance(claimed, dict) or claimed.get("reportHash") != identity_hash(
            {key: value for key, value in claimed.items() if key != "reportHash"})):
        raise ValueError("confidence audit report hash mismatch")
    if claimed.get("kind") != "macro-label-confidence-audit-v1":
        raise ValueError("unsupported confidence audit report")
    manifest, records = _load_ranker_input(labels_dir, selection_path)
    alpha = claimed.get("familywiseAlpha")
    if (type(alpha) not in (int, float) or isinstance(alpha, bool)
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("confidence audit report has an invalid family-wise alpha")
    fresh = _build_report(labels_dir=labels_dir, selection_path=selection_path,
        manifest=manifest, records=records, familywise_alpha=alpha)
    fresh["reportHash"] = identity_hash(fresh)
    if claimed != fresh:
        raise ValueError("confidence audit differs from fresh input and implementation recomputation")
    return fresh


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only confidence audit for finalized macro labels")
    parser.add_argument("--labels", type=Path, required=True,
                        help="verified combined train/development macro-label directory")
    parser.add_argument("--selection", type=Path, required=True,
                        help="exact frozen macro-label selection manifest")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path,
                      help="new immutable JSON report path")
    mode.add_argument("--verify-report", type=Path,
                      help="recompute and verify an existing report")
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.verify_report:
        report = verify_macro_label_confidence_audit(labels_dir=args.labels,
            selection_path=args.selection, report_path=args.verify_report)
    else:
        report = audit_macro_label_confidence(labels_dir=args.labels,
            selection_path=args.selection, output=args.output,
            familywise_alpha=args.familywise_alpha)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
