from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ptcg_lab.learning_mind.encoding import LIMITS, UnsupportedPosition, collate, encode_decision
from ptcg_lab.learning_mind.schema import IdentityError, IdentityManifest
from ptcg_lab.learning_mind.tracker import ObservableHistoryTracker

ROOT = Path(__file__).parents[2]


def card(identifier="A", name="Card", kind="trainer", **extra):
    return {"id": identifier, "name": name, "kind": kind, **extra}


def pokemon(identifier, name):
    return {"card": card(identifier, name, "pokemon", hp=100, attacks=[]), "damage": 0,
            "energy": [], "tools": [], "conditions": []}


def observation():
    return {"schemaVersion": 1, "playerId": 0, "decisionPlayer": 0, "turn": 2, "phase": "player-turn",
            "status": "running", "players": [
                {"id": 0, "active": pokemon("OWN", "Own"), "bench": [], "hand": [card("X", "Judge"), card("X", "Judge")],
                 "handCount": 2, "deckCount": 40, "prizesRemaining": 6, "discard": []},
                {"id": 1, "active": pokemon("OPP", "Opponent"), "bench": [pokemon("B", "Bench")], "hand": [],
                 "handCount": 7, "deckCount": 39, "prizesRemaining": 6, "discard": []}],
            "stadium": None, "knowledge": [], "ownDeck": [], "history": ["P2: drew a card"], "warnings": [],
            "legalActions": [
                {"id": "1", "type": "play-trainer", "label": "Play Judge", "cardId": "X",
                 "sourceRef": {"playerId": 0, "zone": "hand", "index": 0}},
                {"id": "2", "type": "play-trainer", "label": "Play Judge", "cardId": "X",
                 "sourceRef": {"playerId": 0, "zone": "hand", "index": 1}},
                {"id": "3", "type": "attack", "label": "Attack Bench 1", "targetRef": {"playerId": 1, "zone": "bench", "index": 0}},
                {"id": "4", "type": "attack", "label": "Attack Active", "targetRef": {"playerId": 1, "zone": "active"}}],
            "prompt": None}


def encoded(obs=None):
    obs = obs or observation(); tracker = ObservableHistoryTracker(0); snap = tracker.update(obs)
    return encode_decision(obs, snap)


def test_tracker_rejects_opposite_private_view_and_tracks_structured_facts():
    obs = observation(); obs["knowledge"] = [
        {"type": "hand-reveal", "playerId": 1, "cards": [card("R", "Rare Candy")]},
        {"type": "known-deck-order", "playerId": 0, "cards": [card("T", "Top")]},
        {"type": "prize-deduction", "playerId": 0, "cards": [card("P", "Prize")]},
        {"type": "once-used", "playerId": 0, "name": "Ability"}]
    tracker = ObservableHistoryTracker(0); snap = tracker.update(obs)
    assert snap["seats"]["1"]["revealedHand"] == {"R": 1}
    assert snap["seats"]["0"]["knownDeckOrder"] == ["T"]
    assert snap["seats"]["0"]["exactPrizeDeductions"] == {"P": 1}
    assert snap["seats"]["0"]["oncePerTurn"] == ["Ability"]
    other = copy.deepcopy(obs); other["playerId"] = 1
    with pytest.raises(ValueError, match="another seat"):
        tracker.update(other)


def test_equivalence_merges_only_interchangeable_hand_copies_and_preserves_targets():
    result = encoded()
    assert len(result.action_classes) == 3
    assert sorted(item.multiplicity for item in result.action_classes) == [1, 1, 2]
    ids = [action["id"] for group in result.action_classes for action in group.actions]
    assert sorted(ids) == ["1", "2", "3", "4"]
    assert len(ids) == len(set(ids))


def test_dynamic_padding_and_fail_closed_action_cap():
    first = encoded(); obs = observation(); obs["legalActions"] = obs["legalActions"][:1]
    batch = collate([first, encoded(obs)])
    assert batch["option_features"].shape[1] == len(first.action_classes) + 1
    assert batch["option_mask"][1].sum() == 2  # one action plus STOP
    obs = observation(); obs["legalActions"] = [{"id": str(i), "type": "other", "label": f"Unique {i}"}
                                                  for i in range(LIMITS["legalActionClasses"] + 1)]
    with pytest.raises(UnsupportedPosition, match="action cap"):
        encoded(obs)


def test_same_public_actor_view_has_same_encoding_despite_opposite_private_view():
    obs = observation(); first = encoded(obs)
    hidden = copy.deepcopy(obs)
    hidden["players"][1]["hand"] = [card("SECRET", "Impossible private card")]
    # Opponent hand is not actor-visible and should already be redacted by the engine;
    # defensive encoding ignores it even if a malformed fixture includes it.
    second = encoded(hidden)
    np.testing.assert_array_equal(first.state_card_ids, second.state_card_ids)
    np.testing.assert_array_equal(first.state_features, second.state_features)
    assert first.identity == second.identity


def test_identity_manifest_rejects_drift():
    manifest = IdentityManifest.create(engine_build_hash="e", deck_manifests={"a": 1}, card_metadata={"x": 1})
    manifest.require_match(manifest.record())
    changed = manifest.record(); changed["engine_build_hash"] = "other"
    with pytest.raises(IdentityError, match="engine_build_hash"):
        manifest.require_match(changed)


def test_typescript_and_python_structural_feature_parity():
    tsx = Path("/Users/admin/Documents/ChatGPT/Pokemon Ai project/node_modules/.bin/tsx")
    if not tsx.exists(): pytest.skip("configured TypeScript runtime unavailable")
    obs = observation(); tracker = ObservableHistoryTracker(0); snap = tracker.update(obs)
    expected = encode_decision(obs, snap)
    loader = tsx.parents[1] / "tsx/dist/loader.mjs"
    completed = subprocess.run(["node", "--import", str(loader),
                                str(ROOT / "research/learning_mind/encoding_parity.ts")],
                               input=json.dumps({"observation": obs, "tracker": snap}), text=True,
                               capture_output=True, check=True)
    actual = json.loads(completed.stdout)
    assert actual == {"tokenKeys": expected.token_keys,
                      "actionKeys": [item.semantic_key for item in expected.action_classes],
                      "identity": expected.identity}
