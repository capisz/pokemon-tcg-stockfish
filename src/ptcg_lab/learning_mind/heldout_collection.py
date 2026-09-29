"""Resumable, heldout-only macro-label collection with an isolated seed domain.

This collector is deliberately separate from experiment.collect_macro_labels:
the latter is byte-audited against historical train/development runs and must
not be changed to add a new seed namespace.
"""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .dataset_v1 import file_sha256, load_dataset
from ptcg_lab.engine import EnginePool

from .experiment import (generate_transition_candidates,
                         runtime_identity, transition_generator_identity)
from .heldout_evaluation import (FAMILIES, _heldout_rollout_seed,
                                 _read_frozen_heldout_selection,
                                 _verify_source_pool)
from .heldout_seed import HELDOUT_SEED_VERSION
from .macro import CANDIDATE_GENERATOR_VERSION
from .schema import UnsupportedPosition, identity_hash


LABEL_COLLECTOR_VERSION = "heldout-macro-rollout-labeler-v1"
MAX_ROLLOUTS = 64


def _atomic_json(path: Path, value: dict, *, immutable: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
        json.dump(value, temporary, sort_keys=True, indent=2, allow_nan=False)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if immutable:
            os.link(temporary_path, path)
        else:
            temporary_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _heldout_output_lock(output: Path):
    """Prevent two processes from sampling the same heldout output at once."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.heldout-collector.lock"
    with lock_path.open("a+b") as lock_file:
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise RuntimeError("a heldout collector already owns this output") from error
                unlock = lambda: msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    raise RuntimeError("a heldout collector already owns this output") from error
                unlock = lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            locked = True
            yield
        finally:
            if locked:
                unlock()


def _new_candidate_state(candidate_hashes: list[str]) -> dict:
    return {key: {"scores": [], "scoresByIndex": {}, "finished": 0,
                  "truncated": 0, "error": 0, "reasons": {}, "decisionCounts": {},
                  "outcomesByIndex": {}}
            for key in candidate_hashes}


def _close_candidates(records: dict, active_hashes: list[str], *, candidate_count: int,
                      initial: int, maximum: int, extension_batch_size: int,
                      close_margin: float) -> list[str]:
    if not active_hashes:
        return []
    stage_count = 1 + math.ceil((maximum - initial) / extension_batch_size)
    alpha = .05 / (candidate_count * stage_count)
    log_term = math.log(2 / alpha)
    intervals = {}
    for key in active_hashes:
        scores = records[key]["scores"]
        if not scores:
            intervals[key] = (0.0, 1.0)
            continue
        mean = sum(scores) / len(scores)
        radius = math.sqrt(log_term / (2 * len(scores)))
        intervals[key] = (max(0.0, mean - radius), min(1.0, mean + radius))
    best_lower = max(low for low, _high in intervals.values())
    return [key for key in active_hashes if intervals[key][1] >= best_lower - close_margin]


def label_heldout_candidates(candidates, position_hash: str, rollout, *, rollout_identity: str,
                             initial: int = 16, maximum: int = 64,
                             extension_batch_size: int = 8, close_margin: float = .10,
                             rollout_workers: int = 1, resume_state: dict | None = None,
                             checkpoint=None) -> list[dict]:
    """Common-random-number adaptive labels with durable per-seed checkpoints."""
    if (not isinstance(candidates, list) or not candidates
            or not isinstance(position_hash, str) or not position_hash
            or not isinstance(rollout_identity, str) or not rollout_identity
            or type(initial) is not int or type(maximum) is not int
            or not 1 <= initial <= maximum <= MAX_ROLLOUTS
            or type(extension_batch_size) is not int or not 1 <= extension_batch_size <= MAX_ROLLOUTS
            or isinstance(close_margin, bool) or not isinstance(close_margin, (int, float))
            or not math.isfinite(close_margin) or not 0 <= close_margin <= 1
            or type(rollout_workers) is not int or not 1 <= rollout_workers <= 8):
        raise ValueError("invalid heldout rollout allocation or candidate set")
    candidate_hashes = [candidate.key() for candidate in candidates]
    if len(candidate_hashes) != len(set(candidate_hashes)):
        raise ValueError("heldout macro candidates must have unique hashes")
    allocation = {"initial": initial, "maximum": maximum,
        "extensionBatchSize": extension_batch_size, "closeMargin": float(close_margin)}
    records = _new_candidate_state(candidate_hashes)
    completed_initial: set[int] = set()
    completed_extension: set[int] = set()
    close_hashes: list[str] | None = None
    confidence_evaluated_through_index = -1
    if resume_state is not None:
        if (not isinstance(resume_state, dict) or resume_state.get("allocation") != allocation
                or resume_state.get("candidateHashes") != candidate_hashes
                or resume_state.get("rolloutIdentity") != rollout_identity):
            raise ValueError("heldout rollout checkpoint identity/allocation mismatch")
        prior_records = resume_state.get("records")
        if not isinstance(prior_records, dict) or set(prior_records) != set(candidate_hashes):
            raise ValueError("heldout rollout checkpoint candidate records mismatch")
        completed_initial = set(resume_state.get("completedInitialIndices", []))
        completed_extension = set(resume_state.get("completedExtensionIndices", []))
        if (any(type(index) is not int or not 0 <= index < initial for index in completed_initial)
                or any(type(index) is not int or not initial <= index < maximum for index in completed_extension)):
            raise ValueError("heldout rollout checkpoint contains invalid seed indices")
        close_hashes = resume_state.get("closeCandidateHashes")
        if close_hashes is not None and (not isinstance(close_hashes, list)
                or any(key not in records for key in close_hashes)):
            raise ValueError("heldout rollout checkpoint active-candidate set is invalid")
        confidence_evaluated_through_index = resume_state.get("confidenceEvaluatedThroughIndex", -1)
        if (type(confidence_evaluated_through_index) is not int
                or not -1 <= confidence_evaluated_through_index < maximum):
            raise ValueError("heldout checkpoint confidence watermark is invalid")
        records = prior_records

    def save():
        if checkpoint is not None:
            checkpoint({"allocation": allocation, "candidateHashes": candidate_hashes,
                "rolloutIdentity": rollout_identity,
                "records": records, "completedInitialIndices": sorted(completed_initial),
                "completedExtensionIndices": sorted(completed_extension),
                "closeCandidateHashes": close_hashes,
                "confidenceEvaluatedThroughIndex": confidence_evaluated_through_index})

    executor = ThreadPoolExecutor(max_workers=rollout_workers) if rollout_workers > 1 else None

    def run_indices(indices, selected, *, extension: bool):
        completed = completed_extension if extension else completed_initial
        for index in indices:
            if index in completed:
                continue
            seed = _heldout_rollout_seed(position_hash, index, rollout_identity)
            outcomes = ([rollout(candidate, seed) for candidate in selected] if executor is None
                else list(executor.map(lambda candidate: rollout(candidate, seed), selected)))
            for candidate, outcome in zip(selected, outcomes):
                key = candidate.key()
                if not isinstance(outcome, dict):
                    raise ValueError("heldout rollout returned a malformed outcome")
                status, score = outcome.get("status"), outcome.get("score")
                record = records[key]
                outcome_index = str(index)
                if outcome_index in record["outcomesByIndex"]:
                    raise ValueError("heldout rollout checkpoint repeats a candidate/seed outcome")
                if (status == "finished" and isinstance(score, (int, float))
                        and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1):
                    record["scores"].append(float(score))
                    record["scoresByIndex"][str(index)] = float(score)
                    record["finished"] += 1
                elif status in {"truncated", "error"}:
                    record[status] += 1
                    reason = outcome.get("reason")
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("heldout unfinished rollout lacks a typed reason")
                    record["reasons"][reason] = record["reasons"].get(reason, 0) + 1
                else:
                    raise ValueError("heldout rollout returned an invalid status or score")
                decision_count = outcome.get("decisionCount")
                seed_outcome = {"status": status}
                if status == "finished":
                    seed_outcome["score"] = float(score)
                else:
                    seed_outcome["reason"] = reason
                if type(decision_count) is int and decision_count >= 0:
                    counts = records[key]["decisionCounts"]
                    text = str(decision_count)
                    counts[text] = counts.get(text, 0) + 1
                    seed_outcome["decisionCount"] = decision_count
                record["outcomesByIndex"][outcome_index] = seed_outcome
            completed.add(index)
            save()

    try:
        run_indices(range(initial), candidates, extension=False)
        if close_hashes is None:
            close_hashes = _close_candidates(records, candidate_hashes,
                candidate_count=len(candidate_hashes), initial=initial, maximum=maximum,
                extension_batch_size=extension_batch_size, close_margin=close_margin)
            confidence_evaluated_through_index = initial - 1
            save()
        elif completed_extension:
            last = max(completed_extension)
            stage_start = initial + ((last - initial) // extension_batch_size) * extension_batch_size
            stage_end = min(maximum, stage_start + extension_batch_size)
            if all(index in completed_extension for index in range(stage_start, stage_end)):
                close_hashes = _close_candidates(records, close_hashes,
                    candidate_count=len(candidate_hashes), initial=initial, maximum=maximum,
                    extension_batch_size=extension_batch_size, close_margin=close_margin)
                confidence_evaluated_through_index = stage_end - 1
                save()
        while maximum > initial and close_hashes:
            missing = [index for index in range(initial, maximum) if index not in completed_extension]
            if not missing:
                close_hashes = _close_candidates(records, close_hashes,
                    candidate_count=len(candidate_hashes), initial=initial, maximum=maximum,
                    extension_batch_size=extension_batch_size, close_margin=close_margin)
                confidence_evaluated_through_index = maximum - 1
                save()
                break
            start = missing[0]
            batch_end = min(maximum, initial +
                (((start - initial) // extension_batch_size) + 1) * extension_batch_size)
            active = set(close_hashes)
            selected = [candidate for candidate in candidates if candidate.key() in active]
            run_indices(range(start, batch_end), selected, extension=True)
            close_hashes = _close_candidates(records, close_hashes,
                candidate_count=len(candidate_hashes), initial=initial, maximum=maximum,
                extension_batch_size=extension_batch_size, close_margin=close_margin)
            confidence_evaluated_through_index = batch_end - 1
            save()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    finite_means = [sum(record["scores"]) / len(record["scores"])
                    for record in records.values() if record["scores"]]
    center = max(finite_means) if finite_means else 0.0
    leader_key = max((key for key, record in records.items() if record["scores"]),
        key=lambda key: sum(records[key]["scores"]) / len(records[key]["scores"]), default=None)
    output = []
    for candidate in candidates:
        key, record = candidate.key(), records[candidate.key()]
        scores = record["scores"]
        mean = sum(scores) / len(scores) if scores else None
        uncertainty = math.sqrt(max(mean * (1 - mean), .25) / len(scores)) if scores else None
        paired = None
        if leader_key is not None:
            leader_scores = records[leader_key]["scoresByIndex"]
            common = sorted(set(record["scoresByIndex"]) & set(leader_scores), key=int)
            deltas = [record["scoresByIndex"][index] - leader_scores[index] for index in common]
            delta = sum(deltas) / len(deltas) if deltas else None
            if delta is not None and len(deltas) > 1:
                variance = sum((value - delta) ** 2 for value in deltas) / (len(deltas) - 1)
                standard_error = math.sqrt(variance / len(deltas))
            else:
                standard_error = None
            paired = {"leaderCandidateHash": leader_key, "commonFinishedRollouts": len(common),
                "meanScoreDifference": delta, "standardError": standard_error,
                "sampleStatus": "insufficient" if len(common) < 20 else "descriptive",
                "interpretation": "descriptive-selected-leader-comparison-not-confidence-bound"}
        attempted = record["finished"] + record["truncated"] + record["error"]
        raw_candidate = asdict(candidate)
        raw_candidate["action_ids"] = list(candidate.action_ids)
        raw_candidate["action_sequence"] = list(candidate.action_sequence)
        output.append({"candidate": raw_candidate,
            "candidateHash": key, "completedRollouts": len(scores), "attemptedRollouts": attempted,
            "outcomes": {name: record[name] for name in ("finished", "truncated", "error")},
            "outcomeReasons": dict(sorted(record["reasons"].items())),
            "decisionCountDistribution": dict(sorted(record["decisionCounts"].items(),
                                                       key=lambda item: int(item[0]))),
            "outcomesBySeedIndex": dict(sorted(record["outcomesByIndex"].items(),
                                               key=lambda item: int(item[0]))),
            "expectedResult": mean, "relativeResult": mean - center if mean is not None else None,
            "uncertainty": uncertainty, "weight": 0 if not scores else len(scores) / (1 + uncertainty),
            "pairedComparisonToObservedLeader": paired})
    return output


def _validate_heldout_collection_inputs(*, root: Path, family: str, dataset_dir: Path,
                                        support_path: Path, selection_path: Path) -> dict:
    if family not in FAMILIES:
        raise ValueError("heldout labels require one approved policy family")
    root, dataset_dir, support_path, selection_path = (
        Path(path).resolve() for path in (root, dataset_dir, support_path, selection_path))
    selection, selected_by_family = _read_frozen_heldout_selection(selection_path)
    identity = runtime_identity(root).record()
    if identity != selection["identity"]:
        raise ValueError("heldout runtime identity differs from frozen selection")
    source_manifest, source_support = _verify_source_pool(family=family,
        dataset_dir=dataset_dir, support_path=support_path, selection=selection)
    if transition_generator_identity(root) != source_support.get("candidateGeneratorIdentity"):
        raise ValueError("heldout candidate generator differs from frozen support audit")
    dataset_manifest, dataset_rows = load_dataset(dataset_dir, identity=identity)
    if dataset_manifest.get("manifestHash") != source_manifest.get("manifestHash"):
        raise ValueError("heldout dataset identity changed between verification and load")
    rows_by_hash = {row["positionHash"]: row for row in dataset_rows}
    selected = selected_by_family[family]
    selected_rows = []
    for position_hash, metadata in selected.items():
        row = rows_by_hash.get(position_hash)
        if (row is None or row.get("split") != "heldout"
                or row.get("sourceGameId") != metadata["sourceGameId"]
                or row.get("opponentPolicyFamily") != family):
            raise ValueError("frozen heldout row differs from its dataset source")
        selected_rows.append(row)
    support_by_hash = {row.get("positionHash"): row for row in source_support.get("positions", [])
                       if isinstance(row, dict)}
    if len(support_by_hash) != len(source_support.get("positions", [])):
        raise ValueError("heldout support report repeats or malforms position rows")
    support_counts = {}
    for position_hash in selected:
        support_row = support_by_hash.get(position_hash)
        count = support_row.get("completeCandidateCount") if support_row else None
        if support_row is None or support_row.get("status") != "supported" or type(count) is not int or count < 2:
            raise ValueError("frozen heldout position lacks two complete plans in its support audit")
        support_counts[position_hash] = count
    return {"root": root, "datasetDir": dataset_dir, "supportPath": support_path,
        "selectionPath": selection_path, "selection": selection, "identity": identity,
        "sourceManifest": source_manifest, "sourceSupport": source_support,
        "datasetManifest": dataset_manifest, "selected": selected,
        "selectedRows": selected_rows, "supportCounts": support_counts}


def _heldout_run_settings(inputs: dict, *, initial: int, maximum: int,
                          extension_batch_size: int, horizon: int,
                          rollout_budget_ms: int, rollout_workers: int) -> dict:
    selection, identity = inputs["selection"], inputs["identity"]
    selected_rows, source_manifest, source_support = (
        inputs["selectedRows"], inputs["sourceManifest"], inputs["sourceSupport"])
    return {"schemaVersion": 1, "kind": "heldout-macro-label-run-v1",
        "identity": identity, "selectionHash": selection["selectionHash"],
        "selectionManifestSha256": file_sha256(inputs["selectionPath"]),
        "datasetManifestHash": source_manifest["manifestHash"],
        "supportReportHash": source_support["reportHash"],
        "supportReportSha256": file_sha256(inputs["supportPath"]),
        "requestedPositions": len(selected_rows), "initialRollouts": initial,
        "maximumRollouts": maximum, "extensionBatchSize": extension_batch_size,
        "horizon": horizon, "rolloutBudgetMs": rollout_budget_ms,
        "rolloutWorkers": rollout_workers, "splitFilter": "heldout",
        "selectionMethod": "frozen-heldout-selection-v1",
        "selectedPositionHashes": sorted(inputs["selected"]),
        "candidateGeneratorVersion": CANDIDATE_GENERATOR_VERSION,
        "candidateGeneratorIdentity": source_support["candidateGeneratorIdentity"],
        "rolloutSeedVersion": HELDOUT_SEED_VERSION,
        "rolloutSeedImplementationSha256": file_sha256(Path(__file__).with_name("heldout_seed.py")),
        "adaptiveAllocationVersion": "staged-monotone-simultaneous-hoeffding-v3",
        "labelCollectorVersion": LABEL_COLLECTOR_VERSION,
        "labelCollectorSha256": file_sha256(Path(__file__))}


def _validate_heldout_allocation(*, initial: int, maximum: int,
                                 extension_batch_size: int, horizon: int,
                                 rollout_budget_ms: int, rollout_workers: int) -> None:
    if (type(initial) is not int or type(maximum) is not int or not 1 <= initial <= maximum <= MAX_ROLLOUTS
            or type(extension_batch_size) is not int or not 1 <= extension_batch_size <= MAX_ROLLOUTS
            or type(horizon) is not int or not 1 <= horizon <= 500
            or type(rollout_budget_ms) is not int or not 1 <= rollout_budget_ms <= 120_000
            or type(rollout_workers) is not int or not 1 <= rollout_workers <= 8):
        raise ValueError("invalid heldout collection allocation or runtime limits")


def preflight_heldout_macro_labels(*, root: Path, family: str, dataset_dir: Path,
                                   support_path: Path, selection_path: Path, output: Path,
                                   initial: int = 16, maximum: int = 64,
                                   extension_batch_size: int = 8, horizon: int = 500,
                                   rollout_budget_ms: int = 60_000,
                                   rollout_workers: int = 8) -> dict:
    """Validate a heldout collection plan without creating files or running the engine."""
    _validate_heldout_allocation(initial=initial, maximum=maximum,
        extension_batch_size=extension_batch_size, horizon=horizon,
        rollout_budget_ms=rollout_budget_ms, rollout_workers=rollout_workers)
    original_output = Path(output)
    if original_output.is_symlink():
        raise ValueError("heldout label output directory must not be a symlink")
    inputs = _validate_heldout_collection_inputs(root=root, family=family,
        dataset_dir=dataset_dir, support_path=support_path, selection_path=selection_path)
    output = original_output.resolve()
    selection, selected = inputs["selection"], inputs["selected"]
    settings = _heldout_run_settings(inputs, initial=initial, maximum=maximum,
        extension_batch_size=extension_batch_size, horizon=horizon,
        rollout_budget_ms=rollout_budget_ms, rollout_workers=rollout_workers)
    settings["rolloutIdentity"] = identity_hash(settings)
    if output.exists() and not output.is_dir():
        raise ValueError("heldout label output must be a directory")
    plan_state = "fresh"
    if output.exists():
        if (output / "manifest.json").exists():
            raise ValueError("heldout output is already finalized; select a new output directory")
        allowed = {f"{key}.json" for key in selected} | {f".{key}.progress" for key in selected}
        if any(path.is_dir() or path.name not in allowed for path in output.iterdir()):
            raise ValueError("heldout output contains files outside the frozen selection")
        plan_state = "existing-partial-requires-collector-identity-validation"
    total_candidates = sum(inputs["supportCounts"].values())
    current_results = sum((output / f"{key}.json").is_file() for key in selected) if output.exists() else 0
    active_checkpoints = sum((output / f".{key}.progress").is_file() for key in selected) if output.exists() else 0
    return {"schemaVersion": 1, "kind": "heldout-macro-label-preflight-v1",
        "family": family, "identity": inputs["identity"],
        "selectionHash": selection["selectionHash"], "rolloutIdentity": settings["rolloutIdentity"],
        "settings": settings, "outputState": plan_state,
        "selectedPositions": len(selected), "verifiedSupportedPositions": len(inputs["supportCounts"]),
        "candidatePlans": total_candidates,
        "initialCandidateSeedRollouts": total_candidates * initial,
        "maximumCandidateSeedRollouts": total_candidates * maximum,
        "adaptivePruningMayReduceAttempts": True,
        "existingResultFiles": current_results, "existingProgressCheckpoints": active_checkpoints,
        "elapsedTimeEstimateSeconds": None,
        "elapsedTimeEstimateStatus": "unknown; no comparable heldout runtime benchmark",
        "writesArtifacts": False, "startsEngine": False, "processStatus": "not-started-by-preflight"}


def _collect_heldout_macro_labels(*, root: Path, family: str, dataset_dir: Path,
                                  support_path: Path, selection_path: Path, output: Path,
                                  initial: int = 16, maximum: int = 64,
                                  extension_batch_size: int = 8, horizon: int = 500,
                                  rollout_budget_ms: int = 60_000,
                                  rollout_workers: int = 8) -> dict:
    """Collect one approved policy family's frozen heldout positions.

    This function is an explicit collection entry point. Merely importing it or
    running its CLI help performs no rollouts.
    """
    _validate_heldout_allocation(initial=initial, maximum=maximum,
        extension_batch_size=extension_batch_size, horizon=horizon,
        rollout_budget_ms=rollout_budget_ms, rollout_workers=rollout_workers)
    original_output = Path(output)
    if original_output.is_symlink():
        raise ValueError("heldout label output directory must not be a symlink")
    inputs = _validate_heldout_collection_inputs(root=root, family=family,
        dataset_dir=dataset_dir, support_path=support_path, selection_path=selection_path)
    root, dataset_dir, support_path, selection_path = (
        inputs[key] for key in ("root", "datasetDir", "supportPath", "selectionPath"))
    output = original_output.resolve()
    selection, identity = inputs["selection"], inputs["identity"]
    source_manifest, source_support = inputs["sourceManifest"], inputs["sourceSupport"]
    selected, selected_rows = inputs["selected"], inputs["selectedRows"]
    generator_identity = source_support["candidateGeneratorIdentity"]

    settings = _heldout_run_settings(inputs, initial=initial, maximum=maximum,
        extension_batch_size=extension_batch_size, horizon=horizon,
        rollout_budget_ms=rollout_budget_ms, rollout_workers=rollout_workers)
    rollout_identity = identity_hash(settings)
    settings["rolloutIdentity"] = rollout_identity
    if output.exists() and not output.is_dir():
        raise ValueError("heldout label output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("completed heldout label runs are immutable; choose a new output path")
    expected_results = {f"{position_hash}.json" for position_hash in selected}
    for path in output.iterdir():
        if path.is_dir() or path.name == "manifest.json":
            continue
        if path.name in {f".{position_hash}.progress" for position_hash in selected}:
            continue
        if path.name not in expected_results:
            raise ValueError(f"heldout checkpoint contains an unexpected file: {path.name}")

    completed_records = {}
    for row in selected_rows:
        result_path = output / f"{row['positionHash']}.json"
        if result_path.exists():
            if result_path.is_symlink():
                raise ValueError("heldout result files cannot be symlinks")
            record = json.loads(result_path.read_text())
            if (record.get("rolloutIdentity") != rollout_identity
                    or record.get("selectionHash") != selection["selectionHash"]
                    or record.get("identity") != identity
                    or record.get("datasetManifestHash") != source_manifest["manifestHash"]):
                raise ValueError(f"heldout result identity drift: {result_path.name}")
            completed_records[result_path.name] = record

    with EnginePool(root, size=rollout_workers, timeout=300) as engine_pool:
        for row in selected_rows:
            key = row["positionHash"]
            result_path, progress_path = output / f"{key}.json", output / f".{key}.progress"
            if result_path.exists():
                continue
            observation = row["observation"]
            legal = {str(action["id"]): action for action in observation.get("legalActions", [])
                     if isinstance(action, dict) and isinstance(action.get("id"), str)}
            generator_seed = int.from_bytes(hashlib.sha256(
                f"learning-mind-v1|macro-generator|{key}".encode()).digest()[:4], "big")
            try:
                candidates, generation = generate_transition_candidates(root, observation, generator_seed)
            except UnsupportedPosition as error:
                record = {"schemaVersion": 1, "kind": "heldout-macro-label-position-v1",
                    "positionHash": key, "split": "heldout", "identity": identity,
                    "selectionHash": selection["selectionHash"],
                    "datasetManifestHash": source_manifest["manifestHash"],
                    "sourceGameId": row.get("sourceGameId"), "familyId": row["familyId"],
                    "targetDeck": row["targetDeck"], "opponentArchetype": row["opponentArchetype"],
                    "opponentPolicyFamily": family, "positionStage": row["positionStage"],
                    "observation": observation, "labels": [], "candidateCount": 0,
                    "status": "unsupported", "unsupportedReason": str(error),
                    "seedNamespace": "heldout", "rolloutIdentity": rollout_identity,
                    "highConfidencePolicyEligible": False}
                _atomic_json(result_path, record, immutable=True)
                completed_records[result_path.name] = record
                continue
            executable = [candidate for candidate in candidates
                if candidate.action_sequence and candidate.turn_intent in {"attack", "no-attack"}]
            if not executable or len(executable) < 2:
                record = {"schemaVersion": 1, "kind": "heldout-macro-label-position-v1",
                    "positionHash": key, "split": "heldout", "identity": identity,
                    "selectionHash": selection["selectionHash"],
                    "datasetManifestHash": source_manifest["manifestHash"],
                    "sourceGameId": row.get("sourceGameId"), "familyId": row["familyId"],
                    "targetDeck": row["targetDeck"], "opponentArchetype": row["opponentArchetype"],
                    "opponentPolicyFamily": family, "positionStage": row["positionStage"],
                    "observation": observation, "labels": [], "candidateCount": len(executable),
                    "status": "unsupported",
                    "unsupportedReason": "fewer than two executable complete macro candidates",
                    "seedNamespace": "heldout", "rolloutIdentity": rollout_identity,
                    "highConfidencePolicyEligible": False}
                _atomic_json(result_path, record, immutable=True)
                completed_records[result_path.name] = record
                continue

            progress_identity = {"positionHash": key, "rolloutIdentity": rollout_identity,
                "candidateHashes": [candidate.key() for candidate in executable],
                "generatorSeed": generator_seed, "generatorHypothesisId": generation.get("hypothesisId"),
                "candidateGeneratorIdentity": generator_identity, "allocation": {
                    "initial": initial, "maximum": maximum,
                    "extensionBatchSize": extension_batch_size, "closeMargin": .10}}
            progress_hash = identity_hash(progress_identity)
            resume_state = None
            if progress_path.exists():
                if progress_path.is_symlink():
                    raise ValueError("heldout progress checkpoints cannot be symlinks")
                checkpoint_record = json.loads(progress_path.read_text())
                supplied_hash = checkpoint_record.pop("progressHash", None)
                if (supplied_hash != identity_hash(checkpoint_record)
                        or checkpoint_record.get("progressIdentity") != progress_identity
                        or checkpoint_record.get("progressIdentityHash") != progress_hash):
                    raise ValueError(f"heldout progress checkpoint identity/hash mismatch: {progress_path.name}")
                resume_state = checkpoint_record.get("state")

            def save_progress(state):
                value = {"schemaVersion": 1, "progressIdentity": progress_identity,
                    "progressIdentityHash": progress_hash, "state": state}
                value["progressHash"] = identity_hash(value)
                _atomic_json(progress_path, value)

            def rollout(candidate, seed):
                root_action = legal.get(candidate.action_ids[0]) if candidate.action_ids else None
                if root_action is None:
                    return {"status": "error", "reason": "heldout-macro-root-no-longer-legal"}
                narrowed = copy.deepcopy(observation)
                narrowed["legalActions"] = [root_action]
                plan_actions = list(candidate.action_sequence)
                if not plan_actions or plan_actions[0].get("id") != root_action["id"]:
                    return {"status": "error", "reason": "heldout-macro-plan-root-mismatch"}
                with engine_pool.lease() as engine:
                    result = engine.request("search", {"observation": narrowed,
                        "seed": int(seed) & 0xffffffff, "budgetMs": rollout_budget_ms,
                        "method": "rollout", "iterations": 1,
                        "maxRolloutDecisions": horizon, "macroPlanActions": plan_actions,
                        "researchHypothesisId": generation.get("hypothesisId"),
                        "researchDeterminizationSeed": generator_seed,
                        "researchMaxRolloutDecisions": horizon})
                alternative = next((item for item in result.get("alternatives", [])
                    if item.get("actionId") == root_action["id"] and item.get("visits", 0) > 0), None)
                execution = result.get("macroPlanExecution") or {}
                if execution.get("requested") and not execution.get("completed"):
                    reason = (execution.get("failures") or [{"reason": "macro-plan-unexecuted"}])[0]["reason"]
                    return {"status": "error", "reason": str(reason)}
                if result.get("status") != "complete":
                    warnings = result.get("warnings") or []
                    detail = warnings[0] if warnings and isinstance(warnings[0], str) else "no search warning supplied"
                    return {"status": "error", "reason": f"search-result-{result.get('status', 'missing')}: {detail}"}
                if alternative is None or alternative.get("score") is None:
                    return {"status": "error", "reason": "search-action-unvisited-or-score-missing"}
                continuation = alternative.get("continuation") or {}
                if continuation.get("end") != "terminal":
                    cutoff = continuation.get("cutoffReason")
                    reason = "search-rollout-budget-cutoff" if cutoff == "budget" else "search-rollout-horizon-cutoff"
                    return {"status": "truncated", "reason": reason,
                            "decisionCount": continuation.get("decisionCount")}
                outcome = continuation.get("outcome")
                winner = outcome.get("winner") if isinstance(outcome, dict) else "invalid"
                if winner is not None and (type(winner) is not int or winner not in (0, 1)):
                    return {"status": "error", "reason": "terminal-rollout-missing-valid-outcome"}
                score = .5 if winner is None else 1. if winner == observation["playerId"] else 0.
                return {"status": "finished", "score": score,
                        "decisionCount": continuation.get("decisionCount")}

            labels = label_heldout_candidates(executable, key, rollout,
                rollout_identity=rollout_identity, initial=initial, maximum=maximum,
                extension_batch_size=extension_batch_size, rollout_workers=rollout_workers,
                resume_state=resume_state, checkpoint=save_progress)
            record = {"schemaVersion": 1, "kind": "heldout-macro-label-position-v1",
                "positionHash": key, "split": "heldout", "identity": identity,
                "selectionHash": selection["selectionHash"],
                "datasetManifestHash": source_manifest["manifestHash"],
                "sourceGameId": row.get("sourceGameId"), "familyId": row["familyId"],
                "targetDeck": row["targetDeck"], "opponentArchetype": row["opponentArchetype"],
                "opponentPolicyFamily": family, "positionStage": row["positionStage"],
                "observation": observation, "labels": labels, "status": "collected",
                "generatorSeed": generator_seed, "generatorHypothesisId": generation.get("hypothesisId"),
                "candidateCount": len(executable), "rolloutBudgetMs": rollout_budget_ms,
                "seedNamespace": "heldout", "rolloutIdentity": rollout_identity,
                "rolloutSeeds": [_heldout_rollout_seed(key, index, rollout_identity)
                    for index in range(maximum)],
                "candidateGeneratorIdentity": generator_identity,
                "labelCollectorVersion": LABEL_COLLECTOR_VERSION,
                "labelCollectorSha256": settings["labelCollectorSha256"],
                "highConfidencePolicyEligible": False,
                "semantics": "heldout-only transition-aware macro labels; no training or policy updates"}
            _atomic_json(result_path, record, immutable=True)
            completed_records[result_path.name] = record
            progress_path.unlink(missing_ok=True)

    files = []
    for name in sorted(completed_records):
        path = output / name
        files.append({"path": name, "sha256": file_sha256(path)})
    final = {**settings, "positions": len(files),
        "supportedPositions": sum(record.get("status") == "collected" for record in completed_records.values()),
        "unsupportedPositions": sum(record.get("status") == "unsupported" for record in completed_records.values()),
        "highConfidencePolicyLabels": 0, "files": files,
        "resumePolicy": "verified position results are immutable; interrupted seed batches replay the same seed"}
    final["manifestHash"] = identity_hash(final)
    _atomic_json(manifest_path, final, immutable=True)
    return final


def collect_heldout_macro_labels(*, root: Path, family: str, dataset_dir: Path,
                                 support_path: Path, selection_path: Path, output: Path,
                                 initial: int = 16, maximum: int = 64,
                                 extension_batch_size: int = 8, horizon: int = 500,
                                 rollout_budget_ms: int = 60_000,
                                 rollout_workers: int = 8) -> dict:
    """Collect a frozen heldout family while excluding concurrent duplicate runs."""
    original_output = Path(output)
    if original_output.is_symlink():
        raise ValueError("heldout label output directory must not be a symlink")
    output = original_output.resolve()
    with _heldout_output_lock(output):
        return _collect_heldout_macro_labels(root=root, family=family,
            dataset_dir=dataset_dir, support_path=support_path,
            selection_path=selection_path, output=output, initial=initial,
            maximum=maximum, extension_batch_size=extension_batch_size,
            horizon=horizon, rollout_budget_ms=rollout_budget_ms,
            rollout_workers=rollout_workers)
