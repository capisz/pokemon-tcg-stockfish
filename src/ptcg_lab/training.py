from __future__ import annotations

import math
import os
import random
import time
import uuid
import platform
from pathlib import Path

import numpy as np

from .dataset import complete_games, examples, game_split
from .features import ACTION_DIM, FEATURE_NAMES, action_features, card_tokens, resource_features
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
            torch.tensor([row["expectedResult"] for row in rows], dtype=torch.float32, device=device),
            torch.tensor([row["outcomeClass"] for row in rows], dtype=torch.long, device=device),
            torch.tensor([row["selected"] for row in rows], dtype=torch.long, device=device))


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
    config = {"seed": seed, "device": device, "maxPositions": max_positions, "batchSize": 32,
              "learningRate": .001, "features": list(FEATURE_NAMES), "actionDimensions": ACTION_DIM,
              "featureVersion": "owned-zone-semantic-actions-v2", "splitPolicy": "stable-family-hash-v1",
              "datasetHash": digest(manifest)}
    model = PolicyResourceModel().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 2_000_000:
        raise RuntimeError("Model exceeds the two-million-parameter local budget")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    start_epoch = 0
    history = []
    experiment_id = run_id or uuid.uuid4().hex
    parent = None
    start_batch = 0
    if resume and warm_start:
        raise ValueError("Choose resume or warm-start, not both")
    if warm_start:
        previous_model, previous_checkpoint = load_model(warm_start)
        model.load_state_dict(previous_model.state_dict())
        parent = {"experimentId": previous_checkpoint["experimentId"], "checkpointHash": file_digest(warm_start),
                  "mode": "warm-start-new-data", "optimizer": "fresh"}
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
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

    def checkpoint_at(epoch, next_batch=0):
        return {"schemaVersion": 2, "experimentId": experiment_id, "parent": parent, "platform": platform.platform(),
                "epoch": epoch, "nextBatch": next_batch, "config": config, "manifest": manifest, "parameters": parameters,
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
                policy_log = torch.nn.functional.log_softmax(output["policy"].masked_fill(~mask, -1e9), dim=-1)
                distribution = torch.zeros_like(policy_log)
                for index, row in enumerate(batch):
                    if row["policyDistribution"] is None:
                        distribution[index, row["selected"]] = 1
                    else:
                        distribution[index, :len(row["policyDistribution"])]=torch.tensor(row["policyDistribution"], device=device)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(output["logit"], targets)
                        + .5 * torch.nn.functional.cross_entropy(output["outcome_logits"], labels)
                        - .25 * (distribution * policy_log).sum(dim=-1).mean())
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
                  "parent": parent, "searchTargetExamples": search_examples, "policyTarget": policy_target,
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


def load_model(path: Path):
    torch = _torch()
    from .model import PolicyResourceModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("config", {}).get("featureVersion") != "owned-zone-semantic-actions-v2"
            or checkpoint.get("config", {}).get("actionDimensions") != ACTION_DIM):
        raise ValueError("Checkpoint feature schema is incompatible with this engine lab version; train a fresh checkpoint. Historical artifacts are preserved.")
    model = PolicyResourceModel()
    model.load_state_dict(checkpoint["model"])
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
                  "description": "Learned outcome logit / ln(2); one unit doubles raw expected-result odds. Resource contributions are model terms, not causal card values."
                                 + (" Probabilities use held-out calibration." if calibrated else " Insufficient calibration data; probabilities are withheld.")}
    return evaluation, output["policy"][0].tolist() if legal else []
