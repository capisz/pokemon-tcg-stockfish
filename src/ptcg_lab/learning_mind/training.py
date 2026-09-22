from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .encoding import EncodedDecision, collate
from .model import StrategyTransformerV1

POLICY_LABEL_SOURCES = frozenset({"exact-search-distribution", "compatible-reviewed-acceptable-set",
                                  "high-confidence-macro-plan", "macro-ranker-distillation"})


def supervised_policy_rows(rows: Iterable[dict]) -> list[dict]:
    """Ordinary self-play actions are value evidence, never presumed policy truth."""
    accepted = []
    for row in rows:
        source = row.get("policyLabelSource")
        if source is None:
            continue
        if source not in POLICY_LABEL_SOURCES:
            raise ValueError(f"unsupported policy label source: {source}")
        if not (row.get("acceptableActionIndices") or row.get("policyDistribution")):
            raise ValueError("policy-labelled row has no target")
        accepted.append(row)
    return accepted


def acceptable_set_loss(logits: torch.Tensor, mask: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    losses = []
    for index, row in enumerate(rows):
        if row.get("policyDistribution") is not None:
            target = torch.as_tensor(row["policyDistribution"], dtype=logits.dtype, device=logits.device)
            if target.numel() != logits.shape[1] or not torch.isclose(target.sum(), torch.tensor(1., device=logits.device)):
                raise ValueError("invalid policy distribution")
            losses.append(-(target * log_probs[index]).sum())
        else:
            actions = sorted(set(int(value) for value in row["acceptableActionIndices"]))
            if not actions or any(value < 0 or value >= logits.shape[1] or not mask[index, value] for value in actions):
                raise ValueError("acceptable action is not represented")
            losses.append(-torch.logsumexp(log_probs[index, actions], dim=0))
    if not losses:
        raise ValueError("no supervised policy rows")
    return torch.stack(losses).mean()


def atomic_checkpoint(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_checkpoint_identity(checkpoint: dict, identity: dict) -> None:
    if checkpoint.get("identity") != identity:
        raise ValueError("checkpoint identity does not match the frozen experiment")


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .01
    value_coefficient: float = .5
    optimization_epochs: int = 1
    actor_decisions: int = 8192
    minibatch: int = 512
    current_self_play: float = .8
    historical_play: float = .2
    mirror_fraction: float = .05

    def validate(self) -> None:
        if self != PPOConfig():
            raise ValueError("Autonomous Learning Mind v1 PPO hyperparameters are frozen")


def generalized_advantages(records: list[dict], values: list[float], bootstrap: float = 0.0,
                           config: PPOConfig = PPOConfig()) -> list[float]:
    if len(records) != len(values):
        raise ValueError("record/value count mismatch")
    result = [0.0] * len(records)
    carry = 0.0
    for index in reversed(range(len(records))):
        row = records[index]
        status = row.get("status")
        boundary = bool(row.get("episodeEnd")) or status in {"truncated", "error"}
        if status in {"truncated", "error"}:
            reward, next_value = 0.0, 0.0
        else:
            reward = float(row.get("reward", 0.0))
            next_value = 0.0 if boundary else (values[index + 1] if index + 1 < len(values) else bootstrap)
        delta = reward + config.gamma * next_value - values[index]
        carry = delta if boundary else delta + config.gamma * config.gae_lambda * carry
        result[index] = carry
    return result


def update_guard(*, approximate_kl: float, value_loss: float, finite: bool = True) -> tuple[bool, str | None]:
    if not finite or not all(math.isfinite(value) for value in (approximate_kl, value_loss)):
        return False, "non-finite-tensor"
    if approximate_kl > .05:
        return False, "approximate-kl-exceeded"
    if value_loss > .5:
        return False, "value-loss-exceeded"
    return True, None


def ppo_enablement(stage_record: dict) -> dict:
    passed = bool(stage_record.get("representationParity") and stage_record.get("heldOutLabelWin")
                  and stage_record.get("targetProbeWin") and not stage_record.get("severityThreeRegression"))
    return {"enabled": passed and bool(stage_record.get("humanEnablePPO")),
            "prerequisitesPassed": passed,
            "reason": None if passed else "supervised milestone has not passed",
            "humanEnableRequired": True}


def eligible_ppo_records(records: Iterable[dict]) -> list[dict]:
    result = []
    for row in records:
        if row.get("status") in {"truncated", "error"}:
            continue
        required = {"encoded", "selectedAction", "oldLogProb", "return", "advantage"}
        if not required <= set(row):
            raise ValueError(f"PPO record is missing: {sorted(required - set(row))}")
        result.append(row)
    return result


def ppo_update(model: StrategyTransformerV1, optimizer, records: list[dict], *,
               config: PPOConfig = PPOConfig(), seed: int = 7543298) -> dict:
    """One frozen PPO epoch; rejected minibatches do not mutate the model."""
    config.validate(); records = eligible_ppo_records(records)
    if not records: raise ValueError("no completed PPO traces")
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed)).tolist()
    accepted = rejected = 0; reasons: dict[str, int] = {}; metrics = []
    for start in range(0, len(order), config.minibatch):
        selected = [records[index] for index in order[start:start + config.minibatch]]
        arrays = collate([row["encoded"] for row in selected])
        tensors = {key: torch.as_tensor(value) for key, value in arrays.items()}
        local_actions = []
        for row in selected:
            action = int(row["selectedAction"])
            count = len(row["encoded"].action_classes)
            if not 0 <= action <= count: raise ValueError("selected PPO action is not represented")
            local_actions.append(arrays["option_mask"].shape[1] - 1 if action == count else action)
        chosen = torch.tensor(local_actions, dtype=torch.long)
        old = torch.tensor([row["oldLogProb"] for row in selected], dtype=torch.float32)
        returns = torch.tensor([row["return"] for row in selected], dtype=torch.float32)
        advantages = torch.tensor([row["advantage"] for row in selected], dtype=torch.float32)
        logits = model.policy_forward(**tensors)
        distribution = torch.distributions.Categorical(logits=logits)
        new = distribution.log_prob(chosen)
        values = model.evaluation_forward(**{key: tensors[key] for key in
                                             ("state_card_ids", "state_features", "state_type_ids", "state_mask")})
        ratio = torch.exp(new - old)
        clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
        policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        approximate_kl = (old - new).mean()
        finite = all(torch.isfinite(item).all() for item in (policy_loss, value_loss, entropy, approximate_kl))
        allowed, reason = update_guard(approximate_kl=float(approximate_kl.detach()),
                                       value_loss=float(value_loss.detach()), finite=finite)
        metric = {"approximateKL": float(approximate_kl.detach()), "valueLoss": float(value_loss.detach()),
                  "policyLoss": float(policy_loss.detach()), "entropy": float(entropy.detach())}
        metrics.append(metric)
        if not allowed:
            rejected += 1; reasons[reason] = reasons.get(reason, 0) + 1
            continue
        loss = policy_loss + config.value_coefficient * value_loss - config.entropy * entropy
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        accepted += 1
    return {"optimizationEpochs": 1, "acceptedMinibatches": accepted, "rejectedMinibatches": rejected,
            "rejectionReasons": reasons, "metrics": metrics,
            "pauseRequired": rejected >= 3}


def manifest_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor_batch(records: list[dict], indices: list[int]):
    selected = [records[index] for index in indices]
    arrays = collate([row["encoded"] for row in selected])
    tensors = {key: torch.as_tensor(value) for key, value in arrays.items()}
    rows = []
    for row, decision in zip(selected, (row["encoded"] for row in selected)):
        target = {key: row[key] for key in ("policyLabelSource", "acceptableActionIndices", "policyDistribution") if key in row}
        # Distributions are padded to the batch action width; STOP remains the final valid index per row.
        if target.get("policyDistribution") is not None:
            distribution = list(target["policyDistribution"])
            expected = len(decision.action_classes) + 1
            if len(distribution) != expected: raise ValueError("distribution does not cover actions plus STOP")
            stop = distribution[-1]
            target["policyDistribution"] = distribution[:-1] + [0.] * (arrays["option_mask"].shape[1] - expected) + [stop]
        rows.append(target)
    return tensors, rows


def train_supervised(records: list[dict], output: Path, identity: dict, *, epochs: int = 1,
                     batch_size: int = 32, seed: int = 7543298, resume: Path | None = None,
                     stop_after_batches: int | None = None) -> dict:
    """Deterministic same-device bootstrap with durable optimizer/RNG cursor."""
    records = supervised_policy_rows(records)
    if not records: raise ValueError("supervised bootstrap has no approved labels")
    torch.manual_seed(seed)
    model = StrategyTransformerV1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start_epoch = start_batch = 0; history = []
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        require_checkpoint_identity(checkpoint, identity)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rngState"])
        start_epoch, start_batch = checkpoint["epoch"], checkpoint["nextBatch"]
        history = checkpoint.get("history", [])

    def save(epoch, next_batch):
        payload = {"schemaVersion": 1, "kind": "StrategyTransformerV1-supervised", "identity": identity,
                   "epoch": epoch, "nextBatch": next_batch, "seed": seed, "batchSize": batch_size,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rngState": torch.get_rng_state(), "history": history,
                   "policyLabelSources": sorted(POLICY_LABEL_SOURCES)}
        return atomic_checkpoint(output, payload)

    completed_batches = 0
    for epoch in range(start_epoch, epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(records), generator=generator).tolist()
        model.train(); losses = []
        offset = start_batch if epoch == start_epoch else 0
        for start in range(offset, len(order), batch_size):
            tensors, labels = _tensor_batch(records, order[start:start + batch_size])
            logits = model.policy_forward(**tensors)
            loss = acceptable_set_loss(logits, tensors["option_mask"], labels)
            if not torch.isfinite(loss): raise RuntimeError("non-finite supervised loss")
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            losses.append(float(loss.detach())); save(epoch, start + batch_size)
            completed_batches += 1
            if stop_after_batches is not None and completed_batches >= stop_after_batches:
                return {"status": "paused", "checkpoint": str(output), "epoch": epoch,
                        "nextBatch": start + batch_size, "examples": len(records)}
        history.append({"epoch": epoch + 1, "policyLoss": sum(losses) / len(losses)})
        save(epoch + 1, 0); start_batch = 0
    digest = save(epochs, 0)
    return {"status": "completed", "checkpoint": str(output), "sha256": digest, "epochs": epochs,
            "examples": len(records), "parameters": model.parameter_count(), "history": history}
