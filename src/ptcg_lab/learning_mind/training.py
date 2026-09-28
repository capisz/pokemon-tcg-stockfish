from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable

import torch

from .encoding import EncodedDecision, collate
from .model import StrategyTransformerV1
from . import dataset_v1 as dataset_module, encoding as encoding_module, model as model_module
from . import tracker as tracker_module, value_targets as value_target_module

POLICY_LABEL_SOURCES = frozenset({"exact-search-distribution", "compatible-reviewed-acceptable-set",
                                  "high-confidence-macro-plan", "macro-ranker-distillation"})


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supervised_implementation_identity() -> dict:
    """Pin the executable policy/training stack and CPU runtime in checkpoints."""
    return {"sources": {
        "trainerSha256": _source_sha256(Path(__file__)),
        "modelSha256": _source_sha256(Path(model_module.__file__)),
        "encodingSha256": _source_sha256(Path(encoding_module.__file__)),
        "trackerSha256": _source_sha256(Path(tracker_module.__file__)),
        "policyDatasetBuilderSha256": _source_sha256(Path(dataset_module.__file__)),
        "valueTargetBuilderSha256": _source_sha256(Path(value_target_module.__file__)),
        "trainingCliSha256": _source_sha256(Path(__file__).with_name("__main__.py")),
    }, "runtime": {"device": "cpu", "pythonVersion": platform.python_version(),
        "torchVersion": str(torch.__version__), "torchThreads": torch.get_num_threads(),
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled()}}


def require_checkpoint_implementation(checkpoint: dict) -> None:
    expected = supervised_implementation_identity()
    parent_hash = checkpoint.get("parentCheckpointSha256")
    teacher_hashes = checkpoint.get("teacherHashes")
    training_config = checkpoint.get("trainingConfig")
    valid_hash = lambda value: (isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))
    valid_teachers = (isinstance(teacher_hashes, list)
        and all(valid_hash(value) for value in teacher_hashes)
        and teacher_hashes == sorted(set(teacher_hashes)))
    policy_sources = checkpoint.get("policyLabelSources")
    value_sources = checkpoint.get("valueLabelSources")
    valid_policy_sources = (isinstance(policy_sources, list)
        and all(isinstance(value, str) and value in POLICY_LABEL_SOURCES for value in policy_sources)
        and policy_sources == sorted(set(policy_sources)))
    valid_value_sources = (isinstance(value_sources, list)
        and all(isinstance(value, str) and value == "completed-self-play-outcome" for value in value_sources)
        and value_sources == sorted(set(value_sources)))
    value_manifest_hash = (training_config.get("valueDatasetManifestHash")
        if isinstance(training_config, dict) else None)
    if (checkpoint.get("schemaVersion") != 2
            or not isinstance(training_config, dict)
            or checkpoint.get("implementationIdentity") != expected
            or training_config.get("implementationIdentity") != expected
            or (parent_hash is not None and not valid_hash(parent_hash))
            or not valid_teachers
            or training_config.get("teacherHashes") != teacher_hashes
            or not valid_policy_sources or not valid_value_sources
            or training_config.get("policyLabelSources") != policy_sources
            or training_config.get("valueLabelSources") != value_sources
            or training_config.get("valueLossCoefficient") != .5
            or training_config.get("lossContract") != "approved-policy-plus-terminal-outcome-mse-v1"
            or (bool(value_sources) != valid_hash(value_manifest_hash))):
        raise ValueError("supervised checkpoint implementation identity mismatch")


def supervised_policy_rows(rows: Iterable[dict]) -> list[dict]:
    """Ordinary self-play actions are value evidence, never presumed policy truth."""
    accepted = []
    for row in rows:
        source = row.get("policyLabelSource")
        if source is None:
            continue
        if source not in POLICY_LABEL_SOURCES:
            raise ValueError(f"unsupported policy label source: {source}")
        acceptable = row.get("acceptableActionIndices")
        distribution = row.get("policyDistribution")
        has_acceptable = isinstance(acceptable, list) and bool(acceptable)
        has_distribution = distribution is not None
        if has_acceptable == has_distribution:
            raise ValueError("policy-labelled row must have exactly one nonempty target form")
        teacher_hashes = row.get("teacherHashes", [])
        if source == "macro-ranker-distillation":
            valid_teacher_list = (isinstance(teacher_hashes, list) and len(teacher_hashes) == 2
                and all(isinstance(value, str) and len(value) == 64
                        and all(character in "0123456789abcdef" for character in value)
                        for value in teacher_hashes))
            if not valid_teacher_list or len(set(teacher_hashes)) != 2:
                raise ValueError("ranker-distilled policy rows require exact model and report teacher hashes")
        elif teacher_hashes not in ([], None):
            raise ValueError("non-distilled policy rows cannot claim ranker teacher hashes")
        accepted.append(row)
    return accepted


def _validate_value_target(row: dict) -> None:
    target = row.get("valueTarget")
    if (row.get("valueLabelSource") != "completed-self-play-outcome"
            or type(target) not in {int, float} or not math.isfinite(target)
            or target not in {-1, 0, 1}):
        raise ValueError("value-only targets must be terminal win/draw/loss outcomes in {-1, 0, 1}")


def supervised_training_rows(rows: Iterable[dict]) -> list[dict]:
    """Keep approved policy labels and terminal value labels; discard unlabeled moves."""
    accepted = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("supervised training rows must be objects")
        source = row.get("policyLabelSource")
        has_value = "valueTarget" in row
        if source is None:
            if not has_value:
                if row.get("acceptableActionIndices") is not None or row.get("policyDistribution") is not None:
                    raise ValueError("policy targets cannot be used without an approved policy-label source")
                continue
            if row.get("acceptableActionIndices") is not None or row.get("policyDistribution") is not None:
                raise ValueError("value-only rows cannot contain policy targets")
            _validate_value_target(row)
            accepted.append(row)
            continue
        supervised_policy_rows([row])
        if has_value:
            _validate_value_target(row)
        accepted.append(row)
    return accepted


def acceptable_set_loss(logits: torch.Tensor, mask: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    losses = []
    for index, row in enumerate(rows):
        if row.get("policyDistribution") is not None:
            raw_target = row["policyDistribution"]
            if (not isinstance(raw_target, list)
                    or any(type(value) not in {int, float} for value in raw_target)):
                raise ValueError("invalid policy distribution")
            target = torch.as_tensor(raw_target, dtype=logits.dtype, device=logits.device)
            total = target.sum()
            if (target.ndim != 1 or target.numel() != logits.shape[1]
                    or not torch.isfinite(target).all() or (target < 0).any()
                    or not torch.isclose(total, torch.tensor(1., dtype=logits.dtype,
                                                              device=logits.device), rtol=1e-6, atol=1e-6)):
                raise ValueError("invalid policy distribution")
            positive = target > 0
            if not positive.any() or not mask[index, positive].all():
                raise ValueError("policy distribution assigns mass to an unavailable action")
            # Avoid the undefined 0 * -inf produced by padded, masked options.
            losses.append(-(target[positive] * log_probs[index, positive]).sum())
        else:
            raw_actions = row["acceptableActionIndices"]
            if (not isinstance(raw_actions, list)
                    or any(type(value) is not int for value in raw_actions)):
                raise ValueError("acceptable actions must be integer indices")
            actions = sorted(set(raw_actions))
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
    require_checkpoint_implementation(checkpoint)


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


def _ppo_episode_groups(records: list[dict]) -> list[tuple[str, str, list[int]]]:
    groups: list[tuple[str, str, list[int]]] = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        episode_id, status = row.get("episodeId"), row.get("episodeStatus")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("PPO experience requires a nonempty episodeId")
        if status not in {"finished", "truncated", "error"}:
            raise ValueError("PPO experience episodeStatus must be finished, truncated, or error")
        if not groups or groups[-1][0] != episode_id:
            if episode_id in seen:
                raise ValueError("PPO episodes must be contiguous and appear only once")
            seen.add(episode_id)
            groups.append((episode_id, status, []))
        elif groups[-1][1] != status:
            raise ValueError("one PPO episode cannot mix terminal statuses")
        groups[-1][2].append(index)

    for episode_id, status, indices in groups:
        rows = [records[index] for index in indices]
        if any(type(row.get("episodeEnd")) is not bool for row in rows):
            raise ValueError(f"PPO episode {episode_id} must explicitly mark every episodeEnd")
        if any(row["episodeEnd"] for row in rows[:-1]) or rows[-1]["episodeEnd"] is not True:
            raise ValueError(f"PPO episode {episode_id} must end exactly once on its final record")
        rewards = [row.get("reward") for row in rows]
        if any(type(reward) not in (int, float) or not math.isfinite(reward) for reward in rewards):
            raise ValueError(f"PPO episode {episode_id} has a non-finite reward")
        if any(reward != 0 for reward in rewards[:-1]):
            raise ValueError("PPO v1 uses terminal-only rewards; shaping is forbidden")
        if status == "finished" and rewards[-1] not in {-1, 0, 1}:
            raise ValueError("completed PPO episodes require a terminal win/draw/loss reward")
        if status != "finished" and any(reward != 0 for reward in rewards):
            raise ValueError("truncated/error PPO episodes cannot carry a reward")
    return groups


def generalized_advantages(records: list[dict], values: list[float], bootstrap: float = 0.0,
                           config: PPOConfig = PPOConfig()) -> list[float]:
    if len(records) != len(values):
        raise ValueError("record/value count mismatch")
    if not math.isfinite(bootstrap) or bootstrap != 0.0:
        raise ValueError("PPO v1 never bootstraps across a game or rollout boundary")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("PPO value estimates must be finite")
    config.validate()
    result = [0.0] * len(records)
    for _episode_id, status, indices in _ppo_episode_groups(records):
        if status != "finished":
            # Exclude every decision in a truncated/error episode, not just its last row.
            continue
        carry = 0.0
        for offset in reversed(range(len(indices))):
            index = indices[offset]
            row = records[index]
            reward = float(row["reward"])
            next_value = 0.0 if offset == len(indices) - 1 else float(values[indices[offset + 1]])
            delta = reward + config.gamma * next_value - float(values[index])
            carry = delta if row["episodeEnd"] else delta + config.gamma * config.gae_lambda * carry
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


def ppo_policy_fingerprint(model: StrategyTransformerV1) -> str:
    """Stable hash of the exact actor/value parameters that generated a PPO batch."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0"); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _verify_ppo_behavior_policy(model: StrategyTransformerV1, records: list[dict], *,
                                minibatch: int) -> None:
    expected_hash = ppo_policy_fingerprint(model)
    if any(row.get("behaviorPolicyHash") != expected_hash for row in records):
        raise ValueError("PPO experience was not generated by the exact current behavior policy")
    identities = {row["encoded"].identity for row in records}
    if len(identities) != 1:
        raise ValueError("PPO experience mixes feature identities")
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(records), minibatch):
                selected = records[start:start + minibatch]
                arrays = collate([row["encoded"] for row in selected])
                tensors = {key: torch.as_tensor(value) for key, value in arrays.items()}
                local_actions = []
                for row in selected:
                    action = row["selectedAction"]
                    count = len(row["encoded"].action_classes)
                    if not 0 <= action <= count:
                        raise ValueError("PPO behavior action is not represented by the encoded legal options")
                    local_actions.append(arrays["option_mask"].shape[1] - 1 if action == count else action)
                logits = model.policy_forward(**tensors)
                actual = torch.distributions.Categorical(logits=logits).log_prob(
                    torch.tensor(local_actions, dtype=torch.long, device=logits.device))
                recorded = torch.tensor([row["oldLogProb"] for row in selected],
                                        dtype=actual.dtype, device=actual.device)
                if not torch.isfinite(actual).all() or not torch.allclose(actual, recorded,
                        rtol=0.0, atol=1e-6):
                    raise ValueError("PPO oldLogProb does not match the frozen behavior policy")
    finally:
        model.train(previous_mode)


class VerifiedPPOStageRecord:
    """In-process capability issued only by the stage-evidence verifier.

    Persisted JSON is data, not proof: callers cannot enable PPO by supplying a
    dictionary of optimistic booleans. The verifier which issues this object is
    intentionally separate from the optimizer API.
    """
    __slots__ = ("_values",)

    def __init__(self, values: dict, *, _verification_token: object):
        if _verification_token is not _VERIFIED_PPO_STAGE_TOKEN:
            raise TypeError("VerifiedPPOStageRecord must be issued by the stage-evidence verifier")
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, _name, _value):
        raise AttributeError("verified stage evidence is immutable")


_VERIFIED_PPO_STAGE_TOKEN = object()


def ppo_enablement(stage_record: VerifiedPPOStageRecord | None) -> dict:
    if not isinstance(stage_record, VerifiedPPOStageRecord):
        return {"enabled": False, "prerequisitesPassed": False,
                "reason": "stage record must be issued from verified generated evidence",
                "humanEnableRequired": True}
    stage_record = stage_record._values
    passed = (stage_record.get("representationParity") is True
              and stage_record.get("heldOutLabelWin") is True
              and stage_record.get("heldOutLabelEvidenceStatus") == "supported-improvement"
              and stage_record.get("blindOpponentPolicyFamilyStatus") == "supported-improvement"
              and stage_record.get("targetProbeWin") is True
              and stage_record.get("ragingBoltMacroPlanFidelity") == "passed"
              and stage_record.get("severityThreeProbeCoverage") == "sufficient"
              and stage_record.get("severityThreeRegression") is False
              and stage_record.get("legalActionOmission") is False
              and stage_record.get("illegalAutoregressiveSelection") is False
              and stage_record.get("capOverflow") is False
              and stage_record.get("evaluationIdentityStatus") == "matched"
              and stage_record.get("representativeDisagreementsReviewed") is True)
    return {"enabled": passed and stage_record.get("humanEnablePPO") is True,
            "prerequisitesPassed": passed,
            "reason": None if passed else "supervised milestone evidence is incomplete or not improved",
            "humanEnableRequired": True}


def eligible_ppo_records(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    _ppo_episode_groups(rows)
    required = {"encoded", "selectedAction", "oldLogProb", "return", "advantage",
                "behaviorPolicyHash"}
    result = []
    for _episode_id, status, indices in _ppo_episode_groups(rows):
        if status != "finished":
            continue
        for index in indices:
            row = rows[index]
            if not required <= set(row):
                raise ValueError(f"PPO record is missing: {sorted(required - set(row))}")
            if (type(row["selectedAction"]) is not int
                    or any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
                           for key in ("oldLogProb", "return", "advantage"))):
                raise ValueError("PPO record action and numeric targets must be finite values")
            result.append(row)
    return result


def ppo_update(model: StrategyTransformerV1, optimizer, records: list[dict], *,
               config: PPOConfig = PPOConfig(), seed: int = 7543298,
               stage_record: VerifiedPPOStageRecord | None = None) -> dict:
    """One frozen PPO epoch; rejected minibatches do not mutate the model."""
    if stage_record is None or not ppo_enablement(stage_record)["enabled"]:
        raise PermissionError("PPO update requires passed supervised/macro evidence and explicit human enablement")
    config.validate(); records = eligible_ppo_records(records)
    if not records: raise ValueError("no completed PPO traces")
    _verify_ppo_behavior_policy(model, records, minibatch=config.minibatch)
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
        target = {key: row[key] for key in ("policyLabelSource", "acceptableActionIndices", "policyDistribution",
                                             "valueLabelSource", "valueTarget") if key in row}
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
                     stop_after_batches: int | None = None,
                     dataset_manifest_hash: str | None = None,
                     teacher_hashes: list[str] | None = None,
                     value_dataset_manifest_hash: str | None = None) -> dict:
    """Deterministic same-device bootstrap with durable optimizer/RNG cursor."""
    records = supervised_training_rows(records)
    if not records: raise ValueError("supervised bootstrap has no approved policy or value labels")
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("supervised epochs and batch size must be positive integers")
    if type(seed) is not int:
        raise ValueError("supervised seed must be an integer")
    has_value_rows = any("valueTarget" in row for row in records)
    if has_value_rows != (value_dataset_manifest_hash is not None):
        raise ValueError("terminal value rows and their dataset manifest hash must be supplied together")
    if (value_dataset_manifest_hash is not None
            and (not isinstance(value_dataset_manifest_hash, str) or len(value_dataset_manifest_hash) != 64
                 or any(character not in "0123456789abcdef" for character in value_dataset_manifest_hash))):
        raise ValueError("value-target dataset manifest hash must be lowercase SHA-256")
    row_teacher_hashes = sorted({value for row in records for value in (row.get("teacherHashes") or [])})
    if teacher_hashes is None:
        teacher_hashes = row_teacher_hashes
    if (not isinstance(teacher_hashes, list)
            or any(not isinstance(value, str) or len(value) != 64
           or any(character not in "0123456789abcdef" for character in value)
           for value in teacher_hashes)):
        raise ValueError("supervised teacher hashes must be lowercase SHA-256 values")
    teacher_hashes = sorted(set(teacher_hashes))
    if teacher_hashes != row_teacher_hashes:
        raise ValueError("explicit teacher hashes do not match supervised row provenance")
    policy_label_sources = sorted({row["policyLabelSource"] for row in records
                                   if row.get("policyLabelSource") is not None})
    value_label_sources = sorted({row["valueLabelSource"] for row in records
                                  if row.get("valueTarget") is not None})
    output = output.resolve()
    resume = resume.resolve() if resume is not None else None
    if output.exists() and (resume is None or output != resume):
        raise ValueError("supervised checkpoints are immutable; resume at the same path or choose a new output")
    if resume is not None and not resume.is_file():
        raise ValueError("supervised resume checkpoint does not exist")
    implementation = supervised_implementation_identity()
    training_config = {"seed": seed, "batchSize": batch_size, "epochs": epochs,
                       "optimizer": "AdamW", "learningRate": 1e-4, "weightDecay": 1e-4,
                       "model": "StrategyTransformerV1",
                       "datasetManifestHash": dataset_manifest_hash,
                       "valueDatasetManifestHash": value_dataset_manifest_hash,
                       "valueLossCoefficient": .5,
                       "lossContract": "approved-policy-plus-terminal-outcome-mse-v1",
                       "policyLabelSources": policy_label_sources,
                       "valueLabelSources": value_label_sources,
                       "implementationIdentity": implementation,
                       "teacherHashes": teacher_hashes}
    torch.manual_seed(seed)
    model = StrategyTransformerV1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start_epoch = start_batch = 0; history = []
    current_epoch_loss_sum = 0.0
    current_epoch_policy_loss_sum = current_epoch_value_loss_sum = 0.0
    current_epoch_policy_batches = current_epoch_value_batches = 0
    current_epoch_batches = 0
    parent_checkpoint_sha256 = None
    if resume:
        resume_bytes = resume.read_bytes()
        parent_checkpoint_sha256 = hashlib.sha256(resume_bytes).hexdigest()
        checkpoint = torch.load(io.BytesIO(resume_bytes), map_location="cpu", weights_only=False)
        require_checkpoint_identity(checkpoint, identity)
        if checkpoint.get("kind") != "StrategyTransformerV1-supervised" or checkpoint.get("trainingConfig") != training_config:
            raise ValueError("supervised resume training configuration mismatch")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rngState"])
        start_epoch, start_batch = checkpoint["epoch"], checkpoint["nextBatch"]
        history = checkpoint.get("history", [])
        current_epoch_loss_sum = checkpoint.get("currentEpochLossSum", 0.0)
        current_epoch_policy_loss_sum = checkpoint.get("currentEpochPolicyLossSum", 0.0)
        current_epoch_value_loss_sum = checkpoint.get("currentEpochValueLossSum", 0.0)
        current_epoch_policy_batches = checkpoint.get("currentEpochPolicyBatches", 0)
        current_epoch_value_batches = checkpoint.get("currentEpochValueBatches", 0)
        current_epoch_batches = checkpoint.get("currentEpochBatches", 0)
        if (type(start_epoch) is not int or not 0 <= start_epoch <= epochs
                or type(start_batch) is not int or start_batch < 0 or start_batch > len(records)
                or (start_epoch < epochs and start_batch not in {len(records), *range(0, len(records), batch_size)})
                or type(current_epoch_batches) is not int or current_epoch_batches < 0
                or not isinstance(current_epoch_loss_sum, (int, float))
                or not math.isfinite(current_epoch_loss_sum) or current_epoch_loss_sum < 0
                or not isinstance(current_epoch_policy_loss_sum, (int, float))
                or not math.isfinite(current_epoch_policy_loss_sum) or current_epoch_policy_loss_sum < 0
                or not isinstance(current_epoch_value_loss_sum, (int, float))
                or not math.isfinite(current_epoch_value_loss_sum) or current_epoch_value_loss_sum < 0
                or type(current_epoch_policy_batches) is not int
                or not 0 <= current_epoch_policy_batches <= current_epoch_batches
                or type(current_epoch_value_batches) is not int
                or not 0 <= current_epoch_value_batches <= current_epoch_batches
                or (current_epoch_batches == 0 and current_epoch_loss_sum != 0.0)
                or (current_epoch_policy_batches == 0 and current_epoch_policy_loss_sum != 0.0)
                or (current_epoch_value_batches == 0 and current_epoch_value_loss_sum != 0.0)
                or len(history) != start_epoch):
            raise ValueError("supervised checkpoint cursor or epoch metrics are invalid")
        if (start_epoch < epochs and start_batch == 0 and current_epoch_batches != 0
                or start_batch > 0 and current_epoch_batches != math.ceil(start_batch / batch_size)):
            raise ValueError("supervised checkpoint batch metrics do not match its cursor")

    def save(epoch, next_batch):
        nonlocal parent_checkpoint_sha256
        payload = {"schemaVersion": 2, "kind": "StrategyTransformerV1-supervised", "identity": identity,
                   "trainingConfig": training_config, "datasetManifestHash": dataset_manifest_hash,
                   "implementationIdentity": implementation, "teacherHashes": teacher_hashes,
                   "parentCheckpointSha256": parent_checkpoint_sha256,
                   "epoch": epoch, "nextBatch": next_batch,
                   "seed": seed, "batchSize": batch_size,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rngState": torch.get_rng_state(), "history": history,
                   "currentEpochLossSum": current_epoch_loss_sum,
                   "currentEpochPolicyLossSum": current_epoch_policy_loss_sum,
                   "currentEpochValueLossSum": current_epoch_value_loss_sum,
                   "currentEpochPolicyBatches": current_epoch_policy_batches,
                   "currentEpochValueBatches": current_epoch_value_batches,
                   "currentEpochBatches": current_epoch_batches,
                   "policyLabelSources": policy_label_sources,
                   "valueLabelSources": value_label_sources}
        checkpoint_hash = atomic_checkpoint(output, payload)
        parent_checkpoint_sha256 = checkpoint_hash
        return checkpoint_hash

    completed_batches = 0
    for epoch in range(start_epoch, epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(records), generator=generator).tolist()
        model.train()
        offset = start_batch if epoch == start_epoch else 0
        for start in range(offset, len(order), batch_size):
            tensors, labels = _tensor_batch(records, order[start:start + batch_size])
            policy_indices = [index for index, row in enumerate(labels)
                              if row.get("policyLabelSource") is not None]
            value_indices = [index for index, row in enumerate(labels) if "valueTarget" in row]
            policy_loss = value_loss = None
            loss = None
            if policy_indices:
                indices = torch.tensor(policy_indices, dtype=torch.long)
                policy_inputs = {key: value[indices] for key, value in tensors.items()}
                logits = model.policy_forward(**policy_inputs)
                policy_loss = acceptable_set_loss(logits, tensors["option_mask"][indices],
                    [labels[index] for index in policy_indices])
                loss = policy_loss
            if value_indices:
                indices = torch.tensor(value_indices, dtype=torch.long)
                value_inputs = {key: tensors[key][indices] for key in
                    ("state_card_ids", "state_features", "state_type_ids", "state_mask")}
                values = model.evaluation_forward(**value_inputs)
                targets = torch.tensor([labels[index]["valueTarget"] for index in value_indices],
                                       dtype=values.dtype)
                value_loss = torch.nn.functional.mse_loss(values, targets)
                loss = .5 * value_loss if loss is None else loss + .5 * value_loss
            if loss is None:
                raise ValueError("training batch contains neither policy nor value targets")
            if not torch.isfinite(loss): raise RuntimeError("non-finite supervised loss")
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            current_epoch_loss_sum += float(loss.detach())
            if policy_loss is not None:
                current_epoch_policy_loss_sum += float(policy_loss.detach())
                current_epoch_policy_batches += 1
            if value_loss is not None:
                current_epoch_value_loss_sum += float(value_loss.detach())
                current_epoch_value_batches += 1
            current_epoch_batches += 1
            next_batch = min(start + batch_size, len(order))
            save(epoch, next_batch)
            completed_batches += 1
            if stop_after_batches is not None and completed_batches >= stop_after_batches:
                return {"status": "paused", "checkpoint": str(output), "epoch": epoch,
                        "nextBatch": next_batch, "examples": len(records)}
        if current_epoch_batches == 0:
            raise ValueError("supervised resume cursor skipped an epoch without saved batch metrics")
        history.append({"epoch": epoch + 1,
            "compositeLoss": current_epoch_loss_sum / current_epoch_batches,
            "policyLoss": (current_epoch_policy_loss_sum / current_epoch_policy_batches
                           if current_epoch_policy_batches else None),
            "policyBatches": current_epoch_policy_batches,
            "terminalValueMSE": (current_epoch_value_loss_sum / current_epoch_value_batches
                                 if current_epoch_value_batches else None),
            "valueBatches": current_epoch_value_batches})
        current_epoch_loss_sum = 0.0
        current_epoch_policy_loss_sum = current_epoch_value_loss_sum = 0.0
        current_epoch_policy_batches = current_epoch_value_batches = 0
        current_epoch_batches = 0
        save(epoch + 1, 0); start_batch = 0
    digest = save(epochs, 0)
    return {"status": "completed", "checkpoint": str(output), "sha256": digest, "epochs": epochs,
            "examples": len(records), "parameters": model.parameter_count(), "history": history}
