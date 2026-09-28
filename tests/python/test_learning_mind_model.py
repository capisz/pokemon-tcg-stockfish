from __future__ import annotations

import numpy as np
import hashlib
import pytest

torch = pytest.importorskip("torch")

from ptcg_lab.learning_mind.encoding import collate
from ptcg_lab.learning_mind.model import StrategyTransformerV1, autoregressive_select
from ptcg_lab.learning_mind.training import (PPOConfig, acceptable_set_loss, ppo_update,
    supervised_implementation_identity, train_supervised)
from test_learning_mind_representation import encoded, observation


def tensors(batch):
    return {key: torch.as_tensor(value) for key, value in batch.items()}


def test_parameter_count_and_separate_interfaces():
    model = StrategyTransformerV1()
    assert 1_400_000 <= model.parameter_count() <= 1_900_000
    with pytest.raises(RuntimeError, match="policy_forward"):
        model()


def test_actor_visible_logits_are_deterministic_and_value_has_no_critic_input():
    torch.manual_seed(1); model = StrategyTransformerV1().eval()
    batch = tensors(collate([encoded()]))
    with torch.no_grad():
        first = model.policy_forward(**batch)
        second = model.policy_forward(**batch)
        value = model.evaluation_forward(**{key: batch[key] for key in ("state_card_ids", "state_features", "state_type_ids", "state_mask")})
    assert torch.equal(first, second)
    assert value.shape == (1,)
    assert list(model.evaluation_forward.__code__.co_varnames[:1]) == ["self"]
    assert "critic" not in model.evaluation_forward.__code__.co_varnames


def test_autoregressive_min_max_stop_and_legality():
    steps = [torch.tensor([4., 3., 2., 10.]), torch.tensor([4., 3., 2., 10.]), torch.tensor([1., 1., 1., 9.])]
    result = autoregressive_select(lambda chosen: steps[len(chosen)], action_count=3, minimum=2, maximum=3,
                                   legality=lambda chosen, item: item != 1)
    assert result.indices == (0, 2) and result.stopped


def test_selected_flags_change_policy_context_without_changing_value():
    torch.manual_seed(2); model = StrategyTransformerV1().eval(); batch = tensors(collate([encoded()]))
    selected = torch.zeros_like(batch["option_mask"])
    with torch.no_grad():
        base = model.policy_forward(**batch)
        selected[:, 0] = True
        changed = model.policy_forward(**batch, selected_mask=selected)
    assert not torch.equal(base, changed)


def test_supervised_resume_is_bit_equivalent_on_same_device(tmp_path):
    decisions = [encoded(), encoded(observation())]
    records = [{"encoded": item, "policyLabelSource": "compatible-reviewed-acceptable-set",
                "acceptableActionIndices": [0]} for item in decisions]
    identity = {"identityHash": "fixed"}
    interrupted = tmp_path / "interrupted.pt"
    assert train_supervised(records, interrupted, identity, batch_size=1, stop_after_batches=1)["status"] == "paused"
    resume_parent_hash = hashlib.sha256(interrupted.read_bytes()).hexdigest()
    resumed = train_supervised(records, interrupted, identity, batch_size=1, resume=interrupted)
    complete_path = tmp_path / "complete.pt"
    train_supervised(records, complete_path, identity, batch_size=1)
    left = torch.load(resumed["checkpoint"], weights_only=False)["model"]
    right = torch.load(complete_path, weights_only=False)["model"]
    assert all(torch.equal(left[key], right[key]) for key in left)
    resumed_checkpoint = torch.load(resumed["checkpoint"], weights_only=False)
    complete_checkpoint = torch.load(complete_path, weights_only=False)
    assert complete_checkpoint["parentCheckpointSha256"] is not None
    assert len(resume_parent_hash) == 64
    assert resumed_checkpoint["implementationIdentity"] == supervised_implementation_identity()


def test_supervised_resume_after_final_batch_completes_epoch_bit_equivalently(tmp_path):
    decisions = [encoded(), encoded(observation())]
    records = [{"encoded": item, "policyLabelSource": "compatible-reviewed-acceptable-set",
                "acceptableActionIndices": [0]} for item in decisions]
    identity = {"identityHash": "fixed"}
    interrupted = tmp_path / "final-batch.pt"
    paused = train_supervised(records, interrupted, identity, batch_size=1, stop_after_batches=2)
    assert paused["status"] == "paused" and paused["nextBatch"] == len(records)
    resume_parent_hash = hashlib.sha256(interrupted.read_bytes()).hexdigest()
    resumed = train_supervised(records, interrupted, identity, batch_size=1, resume=interrupted)
    complete_path = tmp_path / "complete.pt"
    complete = train_supervised(records, complete_path, identity, batch_size=1)
    left = torch.load(resumed["checkpoint"], weights_only=False)
    right = torch.load(complete_path, weights_only=False)
    assert resumed["history"] == complete["history"]
    assert len(resume_parent_hash) == 64
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])


@pytest.mark.parametrize("changed", [{"seed": 7}, {"batch_size": 2}, {"epochs": 2},
                                      {"dataset_manifest_hash": "data-b"}])
def test_supervised_resume_rejects_training_configuration_drift(tmp_path, changed):
    records = [{"encoded": encoded(), "policyLabelSource": "compatible-reviewed-acceptable-set",
                "acceptableActionIndices": [0]}]
    identity = {"identityHash": "fixed"}
    checkpoint = tmp_path / "config-drift.pt"
    train_supervised(records, checkpoint, identity, batch_size=1, stop_after_batches=1,
                     dataset_manifest_hash="data-a")
    options = {"seed": 7543298, "batch_size": 1, "epochs": 1,
               "dataset_manifest_hash": "data-a"}
    options.update(changed)
    with pytest.raises(ValueError, match="training configuration mismatch"):
        train_supervised(records, checkpoint, identity, resume=checkpoint, **options)


def test_supervised_training_refuses_to_overwrite_checkpoint_without_resume(tmp_path):
    records = [{"encoded": encoded(), "policyLabelSource": "compatible-reviewed-acceptable-set",
                "acceptableActionIndices": [0]}]
    output = tmp_path / "frozen.pt"
    train_supervised(records, output, {"identityHash": "fixed"}, batch_size=1,
                     stop_after_batches=1)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="checkpoints are immutable"):
        train_supervised(records, output, {"identityHash": "fixed"}, batch_size=1)
    assert output.read_bytes() == original


def test_supervised_resume_links_each_checkpoint_to_exact_parent(tmp_path, monkeypatch):
    from ptcg_lab.learning_mind import training as training_module
    records = [{"encoded": encoded(), "policyLabelSource": "compatible-reviewed-acceptable-set",
                "acceptableActionIndices": [0]}]
    output = tmp_path / "lineage.pt"
    train_supervised(records, output, {"identityHash": "fixed"}, batch_size=1,
                     stop_after_batches=1)
    resume_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    real_save = training_module.atomic_checkpoint
    links = []

    def save_and_record(path, payload):
        digest = real_save(path, payload)
        links.append((payload["parentCheckpointSha256"], digest))
        return digest

    monkeypatch.setattr(training_module, "atomic_checkpoint", save_and_record)
    train_supervised(records, output, {"identityHash": "fixed"}, batch_size=1, resume=output)
    assert links and links[0][0] == resume_hash
    assert all(links[index][0] == links[index - 1][1] for index in range(1, len(links)))


def test_supervised_checkpoint_records_teacher_hashes_from_rows(tmp_path):
    teacher_hashes = ["1" * 64, "2" * 64]
    records = [{"encoded": encoded(), "policyLabelSource": "macro-ranker-distillation",
        "policyDistribution": [1.] + [0.] * len(encoded().action_classes),
        "teacherHashes": teacher_hashes}]
    checkpoint_path = tmp_path / "distilled.pt"
    train_supervised(records, checkpoint_path, {"identityHash": "fixed"}, batch_size=1,
                     stop_after_batches=1, dataset_manifest_hash="a" * 64)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["teacherHashes"] == sorted(teacher_hashes)
    assert checkpoint["trainingConfig"]["teacherHashes"] == checkpoint["teacherHashes"]


def test_ranker_distillation_training_requires_matching_teacher_provenance(tmp_path):
    from ptcg_lab.learning_mind.training import supervised_policy_rows
    teacher_hashes = ["1" * 64, "2" * 64]
    row = {"encoded": encoded(), "policyLabelSource": "macro-ranker-distillation",
        "policyDistribution": [1.] + [0.] * len(encoded().action_classes)}
    with pytest.raises(ValueError, match="model and report teacher hashes"):
        supervised_policy_rows([row])
    row["teacherHashes"] = teacher_hashes
    with pytest.raises(ValueError, match="do not match supervised row provenance"):
        train_supervised([row], tmp_path / "mismatched-teacher.pt", {"identityHash": "fixed"},
                         batch_size=1, teacher_hashes=["3" * 64])
    assert not (tmp_path / "mismatched-teacher.pt").exists()


def test_search_distribution_loss_ignores_zero_mass_padded_options():
    mask = torch.tensor([[True, True, False, True]])
    logits = torch.tensor([[1., 0., -torch.inf, -1.]])
    loss = acceptable_set_loss(logits, mask, [{"policyDistribution": [.75, .25, 0., 0.]}])
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="unavailable action"):
        acceptable_set_loss(logits, mask, [{"policyDistribution": [.5, 0., .5, 0.]}])


@pytest.mark.parametrize("target", [
    [1.1, -.1, 0., 0.],
    [float("nan"), 0., 0., 0.],
    [float("inf"), 0., 0., 0.],
    [1., 0., 0.],
    ["1", 0., 0., 0.],
    [True, 0., 0., 0.],
])
def test_search_distribution_loss_rejects_malformed_targets(target):
    logits = torch.tensor([[1., 2., 3., 0.]])
    mask = torch.tensor([[True, True, True, True]])
    with pytest.raises(ValueError, match="invalid policy distribution"):
        acceptable_set_loss(logits, mask, [{"policyDistribution": target}])


def test_acceptable_set_loss_rejects_noninteger_acceptable_actions():
    logits = torch.tensor([[1., 2.]])
    mask = torch.tensor([[True, True]])
    with pytest.raises(ValueError, match="integer indices"):
        acceptable_set_loss(logits, mask, [{"acceptableActionIndices": [0.5]}])


def test_ppo_one_epoch_updates_completed_trace_and_rejects_high_kl():
    torch.manual_seed(4); model = StrategyTransformerV1(); decision = encoded()
    batch = tensors(collate([decision]))
    with torch.no_grad():
        logits = model.policy_forward(**batch)
        old = torch.log_softmax(logits, -1)[0, 0].item()
        value = model.evaluation_forward(**{key: batch[key] for key in
                                            ("state_card_ids", "state_features", "state_type_ids", "state_mask")})[0].item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = PPOConfig()
    row = {"status": "finished", "encoded": decision, "selectedAction": 0,
           "oldLogProb": old, "return": value, "advantage": 1.0}
    with pytest.raises(PermissionError, match="requires passed supervised/macro evidence"):
        ppo_update(model, optimizer, [row], config=config)
    approved_stage = {"representationParity": True, "heldOutLabelWin": True,
        "heldOutLabelEvidenceStatus": "supported-improvement",
        "blindOpponentPolicyFamilyStatus": "supported-improvement",
        "targetProbeWin": True, "ragingBoltMacroPlanFidelity": "passed",
        "severityThreeProbeCoverage": "sufficient", "severityThreeRegression": False,
        "legalActionOmission": False, "illegalAutoregressiveSelection": False,
        "capOverflow": False, "evaluationIdentityStatus": "matched",
        "representativeDisagreementsReviewed": True,
        "humanEnablePPO": True}
    result = ppo_update(model, optimizer, [row], config=config, stage_record=approved_stage)
    assert result["acceptedMinibatches"] == 1 and result["optimizationEpochs"] == 1
    rejected = ppo_update(model, optimizer, [{**row, "oldLogProb": old + 1}], config=config,
                          stage_record=approved_stage)
    assert rejected["rejectedMinibatches"] == 1
