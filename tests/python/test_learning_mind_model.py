from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ptcg_lab.learning_mind.encoding import collate
from ptcg_lab.learning_mind.model import StrategyTransformerV1, autoregressive_select
from ptcg_lab.learning_mind.training import PPOConfig, ppo_update, train_supervised
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
    resumed = train_supervised(records, interrupted, identity, batch_size=1, resume=interrupted)
    complete_path = tmp_path / "complete.pt"
    train_supervised(records, complete_path, identity, batch_size=1)
    left = torch.load(resumed["checkpoint"], weights_only=False)["model"]
    right = torch.load(complete_path, weights_only=False)["model"]
    assert all(torch.equal(left[key], right[key]) for key in left)


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
    result = ppo_update(model, optimizer, [row], config=config)
    assert result["acceptedMinibatches"] == 1 and result["optimizationEpochs"] == 1
    rejected = ppo_update(model, optimizer, [{**row, "oldLogProb": old + 1}], config=config)
    assert rejected["rejectedMinibatches"] == 1
