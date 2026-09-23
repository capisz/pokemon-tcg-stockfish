from __future__ import annotations

import json

import pytest

from ptcg_lab.learning_mind.evaluation import promotion_gate, sequential_decision
from ptcg_lab.learning_mind.macro import (MacroExecutionFailure, UnsupportedPosition,
    candidates_from_transition_plans,
    execute_candidate, generate_candidates, label_candidates, rollout_seed)
from ptcg_lab.learning_mind.ranker import FrozenIteration, XGBoostMacroRanker, holdout_splits
from ptcg_lab.learning_mind.training import PPOConfig, generalized_advantages, ppo_enablement, supervised_policy_rows, update_guard
from ptcg_lab.learning_mind.curriculum import assignment, promotion_seed_namespace_disjoint, specialist_for_deck
from ptcg_lab.learning_mind.notifications import AtomicRollbackRegistry, NotificationRouter
from test_learning_mind_representation import observation


def test_macro_candidates_are_deterministic_executable_and_capped():
    obs = observation()
    first, second = generate_candidates(obs), generate_candidates(obs)
    assert [item.key() for item in first] == [item.key() for item in second]
    for candidate in first:
        assert [row["id"] for row in execute_candidate(candidate, obs["legalActions"])] == list(candidate.action_ids)
    selected = next(item for item in first if item.action_ids)
    with pytest.raises(MacroExecutionFailure): execute_candidate(selected, [])


def test_macro_abstraction_preserves_every_legal_root_action():
    obs = observation()
    candidates = generate_candidates(obs)
    represented = {identifier for candidate in candidates for identifier in candidate.action_ids}
    assert represented == {str(action["id"]) for action in obs["legalActions"]}


def test_candidate_generator_never_combines_root_actions_into_unverified_sequences():
    obs = observation()
    candidates = generate_candidates(obs)
    assert all(len(candidate.action_ids) == 1 for candidate in candidates)
    assert len({candidate.action_ids[0] for candidate in candidates}) == len(obs["legalActions"])
    with pytest.raises(UnsupportedPosition, match="candidate cap"):
        generate_candidates(obs, cap=len(obs["legalActions"]) - 1)


def test_transition_plans_preserve_ordered_actions_and_semantic_roles():
    root = {"id": "1:0", "type": "play-trainer", "label": "Play Judge", "cardId": "MEG-116"}
    attack = {"id": "2:1", "type": "attack", "label": "Attack: Eon Blade", "target": "active"}
    candidate, = candidates_from_transition_plans([{"actions": [root, attack], "completion": "attack"}])
    assert candidate.action_ids == ("1:0", "2:1")
    assert candidate.action_sequence == (root, attack)
    assert candidate.turn_intent == "attack"
    assert candidate.supporter == "MEG-116" and candidate.intended_attack == "Attack: Eon Blade"
    passed = {"id": "2:2", "type": "pass", "label": "End turn"}
    no_attack, = candidates_from_transition_plans([{"actions": [root, passed], "completion": "no-attack"}])
    assert no_attack.turn_intent == "no-attack" and no_attack.intended_attack is None
    incomplete, = candidates_from_transition_plans([{"actions": [root]}])
    assert incomplete.turn_intent == "incomplete"
    with pytest.raises(MacroExecutionFailure, match="does not match"):
        candidates_from_transition_plans([{"actions": [root, passed], "completion": "attack"}])
    with pytest.raises(UnsupportedPosition, match="cap"):
        candidates_from_transition_plans([{"actions": [root]},
                                          {"actions": [attack], "completion": "attack"}], cap=1)


def test_common_random_numbers_adaptive_rollouts_and_namespace_isolation():
    candidates = generate_candidates(observation())[:2]; calls = []
    def rollout(candidate, seed):
        calls.append((candidate.key(), seed)); return {"status": "finished", "score": .5}
    rollout_identity = "frozen-config-a"
    rows = label_candidates(candidates, "position", rollout, initial=2, maximum=4,
                            rollout_identity=rollout_identity)
    assert all(row["completedRollouts"] == 4 for row in rows)
    for index in range(4):
        expected = rollout_seed("training", "position", index, rollout_identity)
        assert sum(seed == expected for _, seed in calls) == 2
    assert rollout_seed("training", "position", 0) != rollout_seed("development", "position", 0)
    assert rollout_seed("training", "position", 0, "config-a") != rollout_seed("training", "position", 0, "config-b")
    with pytest.raises(ValueError, match="promotion"): label_candidates(candidates, "p", rollout, namespace="promotion")


def test_adaptive_rollouts_extend_when_small_sample_intervals_overlap():
    candidates = generate_candidates(observation())[:2]
    rows = label_candidates(candidates, "uncertain-position",
        lambda candidate, _seed: {"status": "finished", "score": 1.0 if candidate == candidates[0] else 0.0},
        initial=2, maximum=4, close_margin=0.0)
    assert [row["completedRollouts"] for row in rows] == [4, 4]


def test_staged_rollouts_prune_distant_candidates_and_report_actual_attempts():
    candidates = generate_candidates(observation())[:2]
    seeds = {candidate.key(): [] for candidate in candidates}

    def rollout(candidate, seed):
        seeds[candidate.key()].append(seed)
        return {"status": "finished", "score": 1.0 if candidate == candidates[0] else 0.0}

    rows = label_candidates(candidates, "staged-position", rollout, initial=4, maximum=24,
                            extension_batch_size=4, close_margin=.1)
    assert rows[0]["attemptedRollouts"] == 24
    assert rows[1]["attemptedRollouts"] == 16
    assert seeds[candidates[0].key()] == [rollout_seed("training", "staged-position", i) for i in range(24)]
    assert seeds[candidates[1].key()] == [rollout_seed("training", "staged-position", i) for i in range(16)]


def test_staged_rollouts_resume_without_repeating_completed_seed_batches():
    candidates = generate_candidates(observation())[:2]
    checkpoints = []
    seen = {candidate.key(): [] for candidate in candidates}

    def rollout(candidate, seed):
        seen[candidate.key()].append(seed)
        return {"status": "finished", "score": 1.0 if candidate == candidates[0] else 0.0}

    def interrupt_after_extension_seed(state):
        checkpoints.append(json.loads(json.dumps(state)))
        if state["completedExtensionIndices"] == [4]:
            raise InterruptedError("simulated mid-stage interruption")

    with pytest.raises(InterruptedError):
        label_candidates(candidates, "staged-resume-position", rollout, initial=4, maximum=24,
                         extension_batch_size=4, checkpoint=interrupt_after_extension_seed)
    seen = {candidate.key(): [] for candidate in candidates}
    resumed = label_candidates(candidates, "staged-resume-position", rollout, initial=4, maximum=24,
                               extension_batch_size=4, resume_state=checkpoints[-1])
    clean = label_candidates(candidates, "staged-resume-position",
                             lambda candidate, _seed: {"status": "finished",
                                 "score": 1.0 if candidate == candidates[0] else 0.0},
                             initial=4, maximum=24, extension_batch_size=4)
    assert resumed == clean
    assert seen[candidates[0].key()] == [rollout_seed("training", "staged-resume-position", i)
                                         for i in range(5, 24)]
    assert seen[candidates[1].key()] == [rollout_seed("training", "staged-resume-position", i)
                                         for i in range(5, 16)]


def test_rollout_errors_are_not_fabricated_scores():
    candidate = generate_candidates(observation())[:1]
    rows = label_candidates(candidate, "position", lambda c, s: {"status": "truncated", "reason": "fixed-horizon"}, initial=2, maximum=2)
    assert rows[0]["completedRollouts"] == 0 and rows[0]["expectedResult"] is None
    assert rows[0]["outcomeReasons"] == {"fixed-horizon": 2}


def test_rollout_accepts_bounded_horizon_values_but_rejects_invalid_scores():
    candidate = generate_candidates(observation())[:1]
    rows = label_candidates(candidate, "position", lambda c, s: {"status": "finished", "score": .362},
                            initial=2, maximum=2)
    assert rows[0]["completedRollouts"] == 2
    with pytest.raises(ValueError, match="invalid status"):
        label_candidates(candidate, "position", lambda c, s: {"status": "finished", "score": 1.1},
                         initial=1, maximum=1)


def test_holdouts_and_refit_limit():
    rows = [{"opponentArchetype": "a", "opponentPolicyFamily": "old"},
            {"opponentArchetype": "b", "opponentPolicyFamily": "new"}]
    assert len(holdout_splits(rows)) == 4
    with pytest.raises(ValueError, match="six"): FrozenIteration(7, "t", "o", "i", ("p",))


def test_xgboost_ranker_accepts_rollout_group_weights():
    ranker = XGBoostMacroRanker(n_estimators=2, max_depth=2).fit(
        [[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
        [0., 1., 0., 1.], [2, 2], [1., 2., 3., 4.])
    assert len(ranker.predict([[0., 0.], [1., 1.]])) == 2
    with pytest.raises(ValueError, match="cover every"):
        XGBoostMacroRanker(n_estimators=1).fit([[0.], [1.]], [0., 1.], [1], [1., 1.])


def test_only_approved_policy_labels_and_ppo_remains_human_gated():
    rows = [{"policyLabelSource": "exact-search-distribution", "policyDistribution": [1.]},
            {"playedAction": 1}]
    assert supervised_policy_rows(rows) == rows[:1]
    with pytest.raises(ValueError): supervised_policy_rows([{"policyLabelSource": "ordinary-self-play", "acceptableActionIndices": [0]}])
    stage = {"representationParity": True, "heldOutLabelWin": True, "targetProbeWin": True,
             "severityThreeRegression": False}
    assert not ppo_enablement(stage)["enabled"]
    assert ppo_enablement({**stage, "humanEnablePPO": True})["enabled"]


def test_truncation_ends_advantage_trace_and_guards_skip_updates():
    records = [{"status": "running", "reward": 0}, {"status": "truncated", "reward": 1, "episodeEnd": True},
               {"status": "finished", "reward": 1, "episodeEnd": True}]
    values = [0., .5, .25]
    advantages = generalized_advantages(records, values)
    assert advantages[1] == -.5  # no terminal reward and no bootstrap
    assert advantages[2] == .75
    assert update_guard(approximate_kl=.051, value_loss=.1) == (False, "approximate-kl-exceeded")
    assert update_guard(approximate_kl=.01, value_loss=.51) == (False, "value-loss-exceeded")


def test_sequential_evaluation_and_manual_promotion_gate():
    records = [{"status": "finished", "score": 1}] * 100
    assert sequential_decision(records)["status"] == "supported-improvement"
    gate = promotion_gate(aggregate=sequential_decision(records), matchups=[],
                          strategy={"severityThreeRegressions": []}, blind_family_passed=True,
                          identities_match=True, human_approved=False)
    assert not gate["promotable"] and gate["automaticPromotion"] is False


def test_curriculum_ratios_specialist_hash_routing_and_seed_namespaces():
    rows = [assignment(index, ["h1", "h2"]) for index in range(100)]
    assert sum(row["mirror"] for row in rows) == 5
    assert sum(row["policyFamily"] == "historical" for row in rows) == 20
    assert {row["ownArchetype"] for row in rows} == {"crustle", "dragapult", "raging-bolt", "grimmsnarl", "mega-lucario"}
    assert specialist_for_deck("exact", {"exact": {"checkpoint": "special", "approved": True}}, "general") == "special"
    assert specialist_for_deck("modified", {}, "general") == "general"
    seeds = {promotion_seed_namespace_disjoint(1, kind, 0) for kind in ("training", "development", "promotion")}
    assert len(seeds) == 3


def test_notification_scope_and_atomic_rollback():
    events = []; router = NotificationRouter(local=events.append)
    router({"kind": "progress"}); router({"kind": "review-ready"})
    assert events == [{"kind": "review-ready"}]
    registry = AtomicRollbackRegistry("trusted"); registry.queue("candidate")
    with pytest.raises(PermissionError): registry.promote(human_approved=False, evidence_passed=True)
    prior = registry.promote(human_approved=True, evidence_passed=True)
    assert prior == "trusted" and registry.trusted_checkpoint == "candidate"
    registry.rollback(prior); assert registry.trusted_checkpoint == "trusted"
