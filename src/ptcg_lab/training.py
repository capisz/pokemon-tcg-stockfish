from __future__ import annotations

import math
import os
import random
import time
import uuid
import platform
from pathlib import Path

import numpy as np

from .dataset import (complete_games, examples, game_split, family_key, demonstration_records,
                      demonstration_examples, demonstration_manifest)
from .features import ACTION_DIM, FEATURE_NAMES, FEATURE_VERSION, action_features, card_tokens, resource_features, resource_coverage
from .storage import Store, digest, file_digest


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("Training needs the local extra: pip install -e '.[training]'") from exc


def _batch(rows, device):
    torch = _torch()
    width = max(len(row["actions"]) for row in rows)
    action_array = np.zeros((len(rows), width, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        action_array[i, :len(row["actions"])] = row["actions"]
        mask[i, :len(row["actions"])] = True
    return (torch.tensor([row["resources"] for row in rows], dtype=torch.float32, device=device),
            torch.tensor([row["cards"] for row in rows], dtype=torch.long, device=device),
            torch.tensor(action_array, device=device), torch.tensor(mask, device=device),
            torch.tensor([row.get("expectedResult", 0) for row in rows], dtype=torch.float32, device=device),
            torch.tensor([row.get("outcomeClass", 0) for row in rows], dtype=torch.long, device=device),
            torch.tensor([row.get("selected", 0) for row in rows], dtype=torch.long, device=device))


def policy_loss(policy_logits, mask, rows):
    """A demonstration rewards total mass on acceptable moves, not one arbitrary move.

    Outcome labels never enter this function. Search distributions and recorded
    legal moves keep their existing targets in mixed outcome-training batches.
    """
    torch = _torch()
    log_probs = torch.nn.functional.log_softmax(policy_logits.masked_fill(~mask, -1e9), dim=-1)
    losses = []
    for index, row in enumerate(rows):
        accepted = row.get("acceptableIndices")
        if accepted is not None:
            if not accepted or any(not 0 <= action < len(row["actions"]) for action in accepted):
                raise ValueError("Demonstration has no valid acceptable action set")
            losses.append(-torch.logsumexp(log_probs[index, sorted(set(accepted))], dim=0))
        elif row.get("policyDistribution") is not None:
            weights = torch.tensor(row["policyDistribution"], device=policy_logits.device)
            losses.append(-(weights * log_probs[index, :len(weights)]).sum())
        else:
            losses.append(-log_probs[index, row["selected"]])
    return torch.stack(losses).mean()


def _raw_outputs(model, rows, device):
    torch = _torch()
    values, outcomes, targets, classes = [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            resources, cards, actions, mask, target, labels, _ = _batch(rows[start:start + 32], device)
            output = model(resources, cards)
            values.extend(output["logit"].cpu().tolist())
            outcomes.extend(output["outcome_logits"].cpu().tolist())
            targets.extend(target.cpu().tolist())
            classes.extend(labels.cpu().tolist())
    return np.array(values), np.array(outcomes), np.array(targets), np.array(classes)


def _calibrate(model, rows, device):
    logits, outcomes, targets, classes = _raw_outputs(model, rows, device)
    # At least twenty independent games and examples of every actual outcome are
    # required. Missing draws must not be presented as calibrated zero probability.
    game_count = len({row["gameId"] for row in rows})
    if game_count < 20 or len(np.unique(classes)) < 3:
        return None, {"status": "insufficient_data", "games": game_count,
                      "reason": "Calibration requires 20 held-out games and all three outcome classes."}
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(C=1.0, max_iter=500).fit(outcomes, classes)
    return {"coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
            "classes": classifier.classes_.tolist()}, {"status": "fitted", "games": game_count}


def _probabilities(logits: np.ndarray, calibration: dict | None) -> np.ndarray:
    if calibration is not None:
        logits = logits @ np.array(calibration["coef"]).T + np.array(calibration["intercept"])
    stable = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(stable)
    return probs / probs.sum(axis=-1, keepdims=True)


def train(store: Store, *, epochs: int = 3, seed: int = 42, device: str = "cpu", resume: Path | None = None,
          max_positions: int = 10000, mlflow: bool = False, linked_resume: bool = False,
          guard=None, warm_start: Path | None = None, run_id: str | None = None) -> dict:
    torch = _torch()
    from .model import PolicyResourceModel
    if not 1 <= epochs <= 100 or not 100 <= max_positions <= 100000:
        raise ValueError("Bound training to 1..100 epochs and 100..100000 positions")
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("Requested local device is unavailable; choose cpu or an available mps backend")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    games = complete_games(store)
    if len(games) < 10:
        raise ValueError("Need at least 10 completed games; truncated/error games cannot become labels")
    splits = game_split(games)
    data = {name: examples(selected, max_positions) for name, selected in splits.items()}
    if any(not rows for rows in data.values()):
        raise ValueError("Every game split needs at least one valid recorded decision")
    manifest = {name: [{"id": game["id"], "hash": digest(game)} for game in selected] for name, selected in splits.items()}
    demonstrations = demonstration_records(store, limit=max_positions)
    teaching_manifest = demonstration_manifest(demonstrations)
    data["train"].extend(demonstration_examples(demonstrations))
    config = {"seed": seed, "device": device, "maxPositions": max_positions, "batchSize": 32,
              "learningRate": .001, "features": list(FEATURE_NAMES), "actionDimensions": ACTION_DIM,
              "featureVersion": FEATURE_VERSION, "splitPolicy": "stable-family-hash-v1",
              "datasetHash": digest(manifest), "teachingHash": digest(teaching_manifest),
              "eligibilityPolicy": "explicit-verified-roles-v1"}
    model = PolicyResourceModel().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 2_000_000:
        raise RuntimeError("Model exceeds the two-million-parameter local budget")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    start_epoch = 0
    history = []
    experiment_id = run_id or uuid.uuid4().hex
    parent = None
    seen_seeds = {game["seed"] for game in games}
    seen_families = {family_key(game) for game in games}
    teaching_families = {record["familyId"] for record in demonstrations}
    teaching_reviews = {record["reviewHash"] for record in demonstrations}
    start_batch = 0
    if resume and warm_start:
        raise ValueError("Choose resume or warm-start, not both")
    if warm_start:
        previous_model, previous_checkpoint = load_model(warm_start)
        model.load_state_dict(previous_model.state_dict())
        previous_lineage = checkpoint_lineage(previous_checkpoint)
        seen_seeds.update(previous_lineage["seenSeeds"])
        seen_families.update(previous_lineage["familyIds"])
        teaching_families.update(previous_lineage.get("teachingFamilyIds", []))
        teaching_reviews.update(previous_lineage.get("teachingReviewHashes", []))
        parent = {"experimentId": previous_checkpoint["experimentId"], "checkpointHash": file_digest(warm_start),
                  "mode": "warm-start-new-data", "optimizer": "fresh"}
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        previous_lineage = checkpoint_lineage(checkpoint)
        seen_seeds.update(previous_lineage["seenSeeds"])
        seen_families.update(previous_lineage["familyIds"])
        teaching_families.update(previous_lineage.get("teachingFamilyIds", []))
        teaching_reviews.update(previous_lineage.get("teachingReviewHashes", []))
        previous_config = dict(checkpoint["config"])
        comparable = dict(config)
        if linked_resume:
            previous_config.pop("device", None)
            comparable.pop("device", None)
        if previous_config != comparable:
            raise ValueError("Resume config/data hash differs from checkpoint; start a new experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        if linked_resume:
            parent = {"experimentId": checkpoint["experimentId"], "checkpointHash": file_digest(resume),
                      "platform": checkpoint.get("platform"), "device": checkpoint["config"].get("device")}
        else:
            parent = checkpoint.get("parent")
            experiment_id = checkpoint["experimentId"]
        start_batch = checkpoint.get("nextBatch", 0)
        history = checkpoint.get("history", [])
        if start_epoch > epochs:
            raise ValueError("Requested epochs must exceed the completed checkpoint epoch")
    model_dir = store.path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{experiment_id}.pt"
    search_examples = sum(row["policyTargetSource"] == "search-distillation" for row in data["train"])
    policy_target = "mixed search distillation and recorded legal decisions" if search_examples else "behavior cloning of recorded legal decisions; not search distillation"

    seen_families.update(digest({"teachingFamily": family}) for family in teaching_families)
    lineage = {"schemaVersion": 2, "seenSeeds": sorted(seen_seeds), "familyIds": sorted(seen_families),
               "teachingFamilyIds": sorted(teaching_families), "teachingReviewHashes": sorted(teaching_reviews)}
    lineage["hash"] = digest(lineage)

    def checkpoint_at(epoch, next_batch=0):
        return {"schemaVersion": 2, "experimentId": experiment_id, "parent": parent, "platform": platform.platform(),
                "epoch": epoch, "nextBatch": next_batch, "config": config, "manifest": manifest, "parameters": parameters,
                "dataLineage": lineage, "modelKind": "policy-value", "valueTrained": True,
                "teachingManifest": teaching_manifest,
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "calibration": None, "history": history,
                "searchTargetExamples": search_examples, "policyTarget": policy_target,
                "rngPolicy": "deterministic epoch shuffle; floating point equivalence across devices is not promised"}
    checkpoint = checkpoint_at(start_epoch, start_batch)
    if not resume or linked_resume:
        _save_checkpoint(store, model_path, checkpoint)
    store.put("experiments", experiment_id, {"id": experiment_id, "status": "training", "config": config,
              "epoch": start_epoch, "checkpoint": str(model_path), "parent": parent})
    tracking = None
    if mlflow:
        import mlflow as tracking
        tracking.set_tracking_uri(f"sqlite:///{store.path / 'mlflow.db'}")
        tracking.set_experiment("ptcg-local-policy-value")
        tracking.start_run(run_name=experiment_id)
        tracking.log_params({**config, "parameters": parameters})
    try:
        for epoch in range(start_epoch, epochs):
            # Epoch-local shuffle permits reproducible resume without pickle RNG state.
            order = np.random.default_rng(seed + epoch).permutation(len(data["train"]))
            model.train()
            losses = []
            last_saved = time.monotonic()
            for start in range(start_batch if epoch == start_epoch else 0, len(order), 32):
                if guard:
                    guard.check(storage=start % 3200 == 0)
                batch = [data["train"][int(index)] for index in order[start:start + 32]]
                resources, cards, actions, mask, targets, labels, selected = _batch(batch, device)
                output = model(resources, cards, actions)
                outcome_mask = torch.tensor(["expectedResult" in row for row in batch], device=device)
                loss = .25 * policy_loss(output["policy"], mask, batch)
                if outcome_mask.any():
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(output["logit"][outcome_mask], targets[outcome_mask])
                    loss = loss + .5 * torch.nn.functional.cross_entropy(output["outcome_logits"][outcome_mask], labels[outcome_mask])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if (start // 32 + 1) % 100 == 0 or time.monotonic() - last_saved >= 60:
                    _save_checkpoint(store, model_path, checkpoint_at(epoch, start + 32))
                    last_saved = time.monotonic()
            metric = {"epoch": epoch + 1, "trainingLoss": float(np.mean(losses)) if losses else None}
            history.append(metric)
            if tracking and metric["trainingLoss"] is not None:
                tracking.log_metric("training_loss", metric["trainingLoss"], step=epoch + 1)
            checkpoint = checkpoint_at(epoch + 1)
            _save_checkpoint(store, model_path, checkpoint)
            store.put("experiments", experiment_id, {"id": experiment_id, "status": "training", "config": config,
                      "epoch": epoch + 1, "parameters": parameters, "checkpoint": str(model_path), "history": history})
        calibration, calibration_status = _calibrate(model, data["calibration"], device)
        logits, outcomes, targets, classes = _raw_outputs(model, data["test"], device)
        probabilities = _probabilities(outcomes, calibration)
        expected = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        metrics = {"expectedResultMSE": float(np.mean((expected - targets) ** 2)),
                   "outcomeBrier": float(np.mean(np.sum((probabilities - np.eye(3)[classes.astype(int)]) ** 2, axis=1))),
                   "testGames": len(splits["test"]), "testPositions": len(data["test"])}
        checkpoint["calibration"] = calibration
        checkpoint["calibrationStatus"] = calibration_status
        checkpoint["testMetrics"] = metrics
        _save_checkpoint(store, model_path, checkpoint)
        report = {"id": experiment_id, "status": "completed", "type": "policy-value-training", "config": config,
                  "parameters": parameters, "checkpoint": str(model_path), "modelHash": file_digest(model_path),
                  "metrics": metrics, "calibration": calibration_status, "history": history,
                  "parent": parent, "dataLineageHash": lineage["hash"],
                  "searchTargetExamples": search_examples, "policyTarget": policy_target,
                  "demonstrationExamples": len(demonstrations), "modelKind": "policy-value", "valueTrained": True,
                  "splits": {name: [game["id"] for game in selected] for name, selected in splits.items()},
                  "promotion": "experimental; no champion promotion without held-out playing-strength evaluation"}
        store.put("experiments", experiment_id, report)
        return report
    except (KeyboardInterrupt, RuntimeError, OSError) as exc:
        try:
            store.put("experiments", experiment_id, {"id": experiment_id, "status": "paused",
                      "config": config, "checkpoint": str(model_path), "parent": parent,
                      "pauseReason": str(exc) or "Interrupted by user",
                      "note": "Continue from the last durably saved optimizer and batch checkpoint."})
        except (RuntimeError, OSError):
            pass
        raise
    finally:
        if tracking:
            tracking.end_run()


def _save_checkpoint(store: Store, path: Path, checkpoint: dict) -> None:
    torch = _torch()
    from .resources import directory_bytes
    used = directory_bytes(store.path)
    if used > store.max_bytes - 32 * 1024**2:
        raise RuntimeError("Insufficient space under local data cap for training checkpoint")
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(checkpoint, temporary)
        from .resources import check_storage, volume_free, ResourceLimit
        check_storage(store.path, store.max_bytes, 0)
        if volume_free(path) < store.min_free_bytes:
            raise ResourceLimit("Checkpoint destination free-space reserve reached")
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


TEACHING_DIAGNOSTIC_LIMIT = 100


def _policy_state_hash(model) -> str:
    """Hash tensor bytes, independent of checkpoint timestamps and optimizer state."""
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def _teaching_policy_snapshot(model, records, device) -> list[dict]:
    torch = _torch()
    rows = demonstration_examples(records)
    output_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            resources, cards, actions, mask, *_ = _batch(batch, device)
            scores = model(resources, cards, actions)["policy"].masked_fill(~mask, -1e9)
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite policy cannot supply a teaching diagnostic")
            probabilities = scores.softmax(-1)
            for index, (row, record) in enumerate(zip(batch, records[start:start + 32])):
                selected = int(scores[index].argmax())
                output_rows.append({"teachingId": record["id"], "familyId": record["familyId"],
                                    "observationHash": record["positionHash"], "reviewHash": record["reviewHash"],
                                    "selectedActionId": record["observation"]["legalActions"][selected]["id"],
                                    "selectedIsAcceptable": selected in row["acceptableIndices"],
                                    "acceptableActionProbabilityMass": min(1., max(0., float(probabilities[index, row["acceptableIndices"]].sum())))})
    return output_rows


def _validate_teaching_baseline(baseline, records, dataset_hash):
    if (not isinstance(baseline, dict) or baseline.get("schemaVersion") != 1
            or baseline.get("datasetHash") != dataset_hash
            or digest({key: value for key, value in baseline.items() if key != "hash"}) != baseline.get("hash")
            or not isinstance(baseline.get("initialPolicyHash"), str) or len(baseline["initialPolicyHash"]) != 64
            or not isinstance(baseline.get("positions"), list) or len(baseline["positions"]) != len(records)):
        raise ValueError("Policy resume lacks an intact fixed initial teaching baseline; start a new experiment")
    for item, record in zip(baseline["positions"], records):
        legal = {action["id"] for action in record["observation"]["legalActions"]}
        if not isinstance(item, dict):
            raise ValueError("Fixed teaching baseline contains an invalid position")
        mass = item.get("acceptableActionProbabilityMass")
        if (item.get("teachingId") != record["id"] or item.get("familyId") != record["familyId"]
                or item.get("observationHash") != record["positionHash"] or item.get("reviewHash") != record["reviewHash"]
                or item.get("selectedActionId") not in legal or type(item.get("selectedIsAcceptable")) is not bool
                or item["selectedIsAcceptable"] != (item["selectedActionId"] in record["acceptableActionIds"])
                or type(mass) not in (float, int)
                or not math.isfinite(mass) or not 0 <= mass <= 1):
            raise ValueError("Fixed teaching baseline differs from its reviewed legal positions")


def _teaching_comparison(model, records, baseline, device, total_positions):
    after = _teaching_policy_snapshot(model, records, device)
    positions = []
    for before, final in zip(baseline["positions"], after):
        keys = ("selectedActionId", "selectedIsAcceptable", "acceptableActionProbabilityMass")
        positions.append({key: before[key] for key in ("teachingId", "familyId", "observationHash", "reviewHash")} | {
            "before": {key: before[key] for key in keys}, "after": {key: final[key] for key in keys},
            "selectedActionChanged": before["selectedActionId"] != final["selectedActionId"],
            "acceptableMassChange": final["acceptableActionProbabilityMass"] - before["acceptableActionProbabilityMass"]})
    return {"schemaVersion": 1, "kind": "training-teaching-diagnostic", "baselineHash": baseline["hash"],
            "initialPolicyHash": baseline["initialPolicyHash"], "finalPolicyHash": _policy_state_hash(model),
            "selectionRule": "Highest policy logit; the first legal action breaks ties.",
            "positionSelection": f"First {TEACHING_DIAGNOSTIC_LIMIT} admitted training records sorted by immutable teaching ID.",
            "admittedTrainingPositions": total_positions, "comparedPositions": len(positions),
            "omittedPositions": total_positions - len(positions),
            "selectedActionsChanged": sum(item["selectedActionChanged"] for item in positions),
            "positions": positions,
            "description": "Before/after preferences on reviewed training positions. This teaching diagnostic does not measure held-out generalization, playing strength, or game-result probability."}


def train_policy(store: Store, *, epochs: int = 3, seed: int = 42, device: str = "cpu",
                 max_positions: int = 10000, resume: Path | None = None, guard=None, linked_resume: bool = False) -> dict:
    """Imitation warm-start from reviewed tactical decisions; no outcome targets.

    The result is an experimental policy. It cannot produce a learned evaluation
    bar or a leaf value, even though it shares the eventual policy/value network.
    """
    torch = _torch()
    from .model import PolicyResourceModel
    if not 1 <= epochs <= 100 or not 1 <= max_positions <= 100000:
        raise ValueError("Bound policy training to 1..100 epochs and 1..100000 positions")
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError("Requested local device is unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    records = demonstration_records(store, limit=max_positions)
    if not records:
        raise ValueError("No eligible reviewed demonstrations. Imported guides and draft or unaudited positions cannot train a policy.")
    manifest = demonstration_manifest(records)
    rows = demonstration_examples(records)
    config = {"seed": seed, "device": device, "maxPositions": max_positions, "batchSize": 32,
              "learningRate": .001, "features": list(FEATURE_NAMES), "actionDimensions": ACTION_DIM,
              "featureVersion": FEATURE_VERSION, "splitPolicy": "immutable-teaching-families-v1",
              "datasetHash": digest(manifest), "eligibilityPolicy": "reviewed-audited-demonstrations-v1"}
    model = PolicyResourceModel().to(device)
    # These heads have no target and remain frozen. The shared context can learn
    # policy preferences; no resulting random value head is exposed to consumers.
    for name, parameter in model.named_parameters():
        if name.startswith(("resource_terms.", "baseline", "interaction.", "outcomes.")):
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.001)
    identifier, epoch_start, batch_start, history = uuid.uuid4().hex, 0, 0, []
    parent, teaching_baseline = None, None
    if resume:
        previous_model, previous = load_model(resume)
        previous_config, comparable = dict(previous["config"]), dict(config)
        if linked_resume:
            previous_config.pop("device", None)
            comparable.pop("device", None)
        if previous.get("modelKind") != "policy-only" or previous_config != comparable:
            raise ValueError("Policy resume requires identical reviewed data and configuration")
        model.load_state_dict(previous_model.state_dict())
        optimizer.load_state_dict(previous["optimizer"])
        epoch_start, batch_start = previous["epoch"], previous.get("nextBatch", 0)
        if linked_resume:
            parent = {"experimentId": previous["experimentId"], "checkpointHash": file_digest(resume),
                      "platform": previous.get("platform"), "device": previous["config"].get("device"),
                      "mode": "linked-device-continuation"}
        else:
            identifier, parent = previous["experimentId"], previous.get("parent")
        history = previous.get("history", [])
        if epoch_start > epochs:
            raise ValueError("Requested epochs precede the saved checkpoint")
        teaching_baseline = previous.get("teachingBaseline")
    diagnostic_records = sorted(records, key=lambda record: record["id"])[:TEACHING_DIAGNOSTIC_LIMIT]
    if resume:
        # Never reset the comparison to the partially trained resume model.
        _validate_teaching_baseline(teaching_baseline, diagnostic_records, config["datasetHash"])
    else:
        teaching_baseline = {"schemaVersion": 1, "datasetHash": config["datasetHash"],
                             "initialPolicyHash": _policy_state_hash(model),
                             "positions": _teaching_policy_snapshot(model, diagnostic_records, device)}
        teaching_baseline["hash"] = digest(teaching_baseline)
    families = sorted({record["familyId"] for record in records})
    lineage = {"schemaVersion": 2, "seenSeeds": [],
               "familyIds": sorted(digest({"teachingFamily": family}) for family in families),
               "teachingFamilyIds": families, "teachingReviewHashes": sorted({record["reviewHash"] for record in records})}
    lineage["hash"] = digest(lineage)
    path = store.path / "models" / f"{identifier}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    def checkpoint_at(epoch, next_batch=0):
        return {"schemaVersion": 3, "experimentId": identifier, "parent": parent, "platform": platform.platform(),
                "modelKind": "policy-only", "valueTrained": False, "epoch": epoch, "nextBatch": next_batch,
                "config": config, "manifest": {}, "teachingManifest": manifest, "parameters": parameters,
                "teachingBaseline": teaching_baseline,
                "dataLineage": lineage, "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "history": history, "calibration": None,
                "policyTarget": "reviewed acceptable-action sets; no game-result or generated-prose labels"}

    checkpoint = checkpoint_at(epoch_start, batch_start)
    _save_checkpoint(store, path, checkpoint)
    store.put("experiments", identifier, {"id": identifier, "status": "training", "type": "policy-only-training",
                                         "config": config, "checkpoint": str(path)})
    try:
        for epoch in range(epoch_start, epochs):
            order = np.random.default_rng(seed + epoch).permutation(len(rows))
            model.train()
            losses, last_saved = [], time.monotonic()
            for start in range(batch_start if epoch == epoch_start else 0, len(order), 32):
                if guard:
                    guard.check(storage=start % 3200 == 0)
                batch = [rows[int(index)] for index in order[start:start + 32]]
                resources, cards, actions, mask, *_ = _batch(batch, device)
                output = model(resources, cards, actions)
                loss = policy_loss(output["policy"], mask, batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if (start // 32 + 1) % 100 == 0 or time.monotonic() - last_saved >= 60:
                    _save_checkpoint(store, path, checkpoint_at(epoch, start + 32))
                    last_saved = time.monotonic()
            history.append({"epoch": epoch + 1, "policyLoss": float(np.mean(losses)) if losses else None})
            checkpoint = checkpoint_at(epoch + 1)
            _save_checkpoint(store, path, checkpoint)
        # Whole-family validation/test is read only. Report unavailable instead of
        # quietly reusing training positions to claim generalization.
        metrics = {}
        model.eval()
        for partition in ("validation", "test"):
            heldout_records = demonstration_records(store, partition=partition, limit=max_positions)
            heldout_rows = demonstration_examples(heldout_records)
            if not heldout_rows:
                metrics[partition] = {"status": "unavailable", "reason": "No reviewed audited held-out teaching families"}
                continue
            losses, correct = [], 0
            with torch.no_grad():
                for start in range(0, len(heldout_rows), 32):
                    batch = heldout_rows[start:start + 32]
                    resources, cards, actions, mask, *_ = _batch(batch, device)
                    policy = model(resources, cards, actions)["policy"].masked_fill(~mask, -1e9)
                    losses.append(float(policy_loss(policy, mask, batch)) * len(batch))
                    correct += sum(int(selected) in row["acceptableIndices"] for selected, row in zip(policy.argmax(-1), batch))
            metrics[partition] = {"status": "measured", "families": len({record["familyId"] for record in heldout_records}),
                                  "positions": len(heldout_rows), "policyLoss": sum(losses) / len(heldout_rows),
                                  "acceptableActionAccuracy": correct / len(heldout_rows)}
        diagnostic = _teaching_comparison(model, diagnostic_records, teaching_baseline, device, len(records))
        checkpoint["teachingDiagnostic"] = diagnostic
        _save_checkpoint(store, path, checkpoint)
        report = {"id": identifier, "status": "completed", "type": "policy-only-training", "modelKind": "policy-only",
                  "valueTrained": False, "config": config, "parameters": parameters, "checkpoint": str(path),
                  "modelHash": file_digest(path), "dataLineageHash": lineage["hash"], "demonstrationExamples": len(records),
                  "demonstrationFamilies": families, "history": history, "metrics": metrics,
                  "parent": parent, "teachingDiagnostic": diagnostic,
                  "calibration": {"status": "unavailable", "reason": "No outcome targets"},
                  "promotion": "experimental policy; playing strength has not been established"}
        store.put("experiments", identifier, report)
        return report
    except (KeyboardInterrupt, RuntimeError, OSError) as exc:
        try:
            store.put("experiments", identifier, {"id": identifier, "status": "paused", "type": "policy-only-training",
                                                 "config": config, "checkpoint": str(path), "pauseReason": str(exc)})
        except (RuntimeError, OSError):
            pass
        raise


def checkpoint_lineage(checkpoint: dict) -> dict:
    lineage = checkpoint.get("dataLineage")
    if not isinstance(lineage, dict) or lineage.get("schemaVersion") not in {1, 2}:
        raise ValueError("Checkpoint lacks cumulative data lineage; held-out evaluation and continuation are blocked")
    if digest({key: value for key, value in lineage.items() if key != "hash"}) != lineage.get("hash"):
        raise ValueError("Checkpoint cumulative data lineage checksum is invalid")
    if (not isinstance(lineage.get("seenSeeds"), list) or not isinstance(lineage.get("familyIds"), list)
            or not lineage["familyIds"]
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in lineage["seenSeeds"])
            or any(not isinstance(key, str) or len(key) != 64 for key in lineage["familyIds"])):
        raise ValueError("Checkpoint cumulative data lineage is malformed")
    if not lineage["seenSeeds"] and (lineage.get("schemaVersion") != 2 or not lineage.get("teachingFamilyIds")
                                     or not lineage.get("teachingReviewHashes")):
        raise ValueError("Checkpoint has neither verified game seeds nor reviewed teaching lineage")
    if lineage.get("schemaVersion") == 2:
        if (not isinstance(lineage.get("teachingFamilyIds"), list) or not isinstance(lineage.get("teachingReviewHashes"), list)
                or any(not isinstance(value, str) or not value for value in lineage["teachingFamilyIds"])
                or any(not isinstance(value, str) or len(value) != 64 for value in lineage["teachingReviewHashes"])
                or not {digest({"teachingFamily": family}) for family in lineage["teachingFamilyIds"]} <= set(lineage["familyIds"])):
            raise ValueError("Checkpoint teaching lineage is malformed")
    return lineage


def load_model(path: Path):
    torch = _torch()
    from .model import PolicyResourceModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = checkpoint.get("config", {}).get("eligibilityPolicy")
    if policy not in {"explicit-verified-roles-v1", "reviewed-audited-demonstrations-v1"}:
        raise ValueError("Historical checkpoint lacks verified training provenance; preserve it as evidence and train a new eligible model")
    if (checkpoint.get("config", {}).get("featureVersion") != FEATURE_VERSION
            or checkpoint.get("config", {}).get("actionDimensions") != ACTION_DIM):
        raise ValueError("Checkpoint feature schema is incompatible with this engine lab version; train a fresh checkpoint. Historical artifacts are preserved.")
    checkpoint_lineage(checkpoint)
    if policy == "reviewed-audited-demonstrations-v1":
        manifest = checkpoint.get("teachingManifest")
        if (checkpoint.get("modelKind") != "policy-only" or checkpoint.get("valueTrained") is not False
                or not isinstance(manifest, list) or not manifest or digest(manifest) != checkpoint["config"].get("datasetHash")
                or any(item.get("partition") != "train" or not item.get("reviewHash") for item in manifest)
                or checkpoint.get("calibration") is not None):
            raise ValueError("Policy-only checkpoint has invalid teaching evidence or claims an untrained value")
    model = PolicyResourceModel()
    model.load_state_dict(checkpoint["model"])
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model parameters")
    model.eval()
    return model, checkpoint


def predict(path: Path, observation: dict, loaded=None):
    torch = _torch()
    model, checkpoint = loaded or load_model(path)
    legal = observation.get("legalActions", [])
    with torch.no_grad():
        resources = torch.tensor(resource_features(observation)[None, :])
        cards = torch.tensor(card_tokens(observation)[None, :])
        actions = torch.tensor(np.array([action_features(action) for action in legal])[None, :]) if legal else None
        output = model(resources, cards, actions)
    if checkpoint.get("modelKind") == "policy-only":
        evaluation = {"status": "unavailable", "score": None, "expectedResult": None,
                      "winProbability": None, "drawProbability": None, "lossProbability": None,
                      "modelVersion": checkpoint["experimentId"], "components": [], "calibrated": False,
                      "valueTrained": False, "resourceCoverage": resource_coverage(observation),
                      "description": "Policy trained on reviewed tactical demonstrations. No outcome target was used; evaluation scores and probabilities are unavailable."}
        return evaluation, output["policy"][0].tolist() if legal else []
    scale = math.log(2)
    components = [{"name": name, "value": float(value) / scale} for name, value in zip(FEATURE_NAMES, output["components"][0])]
    components.extend([{"name": "baseline", "value": float(output["baseline"][0].detach()) / scale},
                       {"name": "interaction", "value": float(output["interaction"][0]) / scale}])
    calibrated = checkpoint.get("calibration") is not None
    probabilities = _probabilities(output["outcome_logits"].numpy(), checkpoint.get("calibration"))[0] if calibrated else None
    evaluation = {"status": "trained", "score": float(output["logit"][0]) / scale,
                  "expectedResult": float(probabilities[2] + .5 * probabilities[1]) if calibrated else None,
                  "winProbability": float(probabilities[2]) if calibrated else None,
                  "drawProbability": float(probabilities[1]) if calibrated else None,
                  "lossProbability": float(probabilities[0]) if calibrated else None,
                  "modelVersion": checkpoint["experimentId"], "components": components, "calibrated": calibrated,
                  "valueTrained": True, "resourceCoverage": resource_coverage(observation),
                  "description": "Learned outcome logit / ln(2); one unit doubles raw expected-result odds. Resource contributions are model terms, not causal card values."
                                 + (" Probabilities use held-out calibration." if calibrated else " Insufficient calibration data; probabilities are withheld.")}
    return evaluation, output["policy"][0].tolist() if legal else []


def export_portable_value(path: Path, loaded=None) -> dict | None:
    """Verified raw expected-result leaf evaluator, never a policy-only value head.

    Hash exact JSON bytes rather than relying on cross-language float formatting.
    This is the uncalibrated value logit; it is not a calibrated W/D/L predictor.
    """
    import hashlib
    import json
    from .features import CARD_BUCKETS, MAX_VISIBLE_CARDS
    model, checkpoint = loaded or load_model(path)
    if checkpoint.get("modelKind") == "policy-only" or checkpoint.get("valueTrained") is False:
        return None
    names = ("card_embedding.", "resource_terms.", "baseline", "context.", "interaction.")
    weights = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items() if key.startswith(names)}
    payload = json.dumps({"schemaVersion": 1, "featureVersion": FEATURE_VERSION,
                          "modelVersion": checkpoint["experimentId"], "checkpointHash": file_digest(path),
                          "featureNames": list(FEATURE_NAMES), "cardBuckets": CARD_BUCKETS,
                          "maxVisibleCards": MAX_VISIBLE_CARDS, "weights": weights, "valueTrained": True,
                          "valueSemantics": "uncalibrated-expected-result-logit"},
                         separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return {"schemaVersion": 1, "payload": payload, "hash": hashlib.sha256(payload.encode()).hexdigest()}
