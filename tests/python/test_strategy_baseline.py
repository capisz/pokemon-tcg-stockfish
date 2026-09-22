from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probes = load_module("strategy_baseline_probes", ROOT / "research/strategy_baseline/probes.py")
baseline = load_module("strategy_baseline_main", ROOT / "research/strategy_baseline/baseline.py")


def pokemon(name: str, *, hp: int = 200, damage: int = 0, energy=()):
    return {"card": {"name": name, "hp": hp, "attacks": []}, "damage": damage,
            "energy": list(energy), "tools": [], "conditions": []}


def base_observation():
    return {"schemaVersion": 1, "playerId": 0, "decisionPlayer": 0, "turn": 3,
            "phase": "player-turn", "status": "running", "prompt": None,
            "players": [
                {"id": 0, "active": pokemon("Crustle"), "bench": [], "hand": [], "handCount": 4,
                 "deckCount": 30, "prizesRemaining": 6, "discard": []},
                {"id": 1, "active": pokemon("Dragapult ex"), "bench": [], "hand": [], "handCount": 4,
                 "deckCount": 30, "prizesRemaining": 6, "discard": []},
            ], "ownDeck": [], "legalActions": [], "history": [], "warnings": []}


def action(identifier: str, label: str, kind: str = "play-trainer", **extra):
    return {"id": identifier, "label": label, "type": kind, **extra}


def probe_case(identifier: str):
    obs = base_observation()
    good = bad = None
    if identifier == "crustle-opening-kangaskhan":
        obs["prompt"] = {"message": "CHOOSE_STARTING"}
        good, bad = action("g", "Mega Kangaskhan ex", "prompt"), action("b", "Dwebble", "prompt")
    elif identifier == "crustle-one-fortress-restraint":
        good, bad = action("g", "End turn", "pass"), action("b", "Bench Dwebble", "bench")
    elif identifier in {"crustle-energy-jobs", "crustle-dragapult-energy-priority"}:
        good = action("g", "Attach Mist Energy to Active", "attach-energy", target="active")
        bad = action("b", "Attach Growing Grass Energy to Active", "attach-energy", target="active")
    elif identifier == "crustle-safe-first-prize-delay":
        obs["players"][0]["active"] = pokemon("Crustle", hp=150, energy=["Grass", "Grass"])
        obs["players"][0]["active"]["card"]["attacks"] = [{"name": "Scissors", "damage": 120}]
        obs["players"][1]["active"] = pokemon("Dreepy", hp=70)
        obs["players"][1]["active"]["card"]["attacks"] = [{"name": "Peck", "damage": 10}]
        good, bad = action("g", "End turn", "pass"), action("b", "Attack: Scissors", "attack")
    elif identifier == "crustle-target-public-answer":
        good, bad = action("g", "Choose Bench 1: Drakloak", "choice"), action("b", "Choose Bench 2: Budew", "choice")
    elif identifier == "crustle-kangaskhan-pivot":
        obs["players"][0]["bench"] = [pokemon("Mega Kangaskhan ex", energy=["Grass"])]
        obs["players"][0]["active"]["energy"] = ["Grass"]
        good = action("g", "Attach Grass Energy to Bench 1: Mega Kangaskhan ex", "attach-energy", target="bench 1")
        bad = action("b", "Attach Grass Energy to Active Crustle", "attach-energy", target="active")
    elif identifier == "crustle-fan-active-kangaskhan":
        obs["players"][0]["active"] = pokemon("Mega Kangaskhan ex")
        good = action("g", "Attach Handheld Fan to Active", "attach-tool", target="active", cardId="TWM-150")
        bad = action("b", "Attach Handheld Fan to Bench 1", "attach-tool", target="bench 1", cardId="TWM-150")
    elif identifier == "crustle-eri-critical-item":
        obs["prompt"] = {"message": "Eri: choose Items"}
        good, bad = action("g", "Choose Unfair Stamp", "choice"), action("b", "Finish", "choice", choiceOperation="finish")
    elif identifier == "crustle-special-red-card-window":
        obs["players"][1].update(prizesRemaining=3, handCount=7)
        good, bad = action("g", "Play Special Red Card", cardId="CRI-82"), action("b", "End turn", "pass")
    elif identifier == "crustle-lumiose-last":
        obs["players"][1]["bench"] = [pokemon("Budew")]
        good = action("g", "Attach Grass Energy to Active", "attach-energy", target="active")
        bad = action("b", "Use Lumiose City", "stadium", cardId="POR-77")
    elif identifier == "crustle-healing-third-energy":
        obs["players"][0]["active"] = pokemon("Crustle", damage=80, energy=["Grass", "Mist"])
        good = action("g", "Attach Grass Energy to Active", "attach-energy", target="active")
        bad = action("b", "Attach Grass Energy to Bench 1", "attach-energy", target="bench 1")
    elif identifier == "crustle-fan-stack-destination":
        obs["prompt"] = {"message": "Move Energy to opponent Bench"}
        obs["players"][1]["bench"] = [pokemon("Drakloak", energy=["Psychic"]), pokemon("Budew")]
        good, bad = action("g", "Move Energy to Bench 1", "choice"), action("b", "Move Energy to Bench 2", "choice")
    elif identifier == "dragapult-information-poke-pad-first":
        obs["players"][0]["active"] = pokemon("Dragapult ex")
        good, bad = action("g", "Play Poké Pad", cardId="POR-81"), action("b", "Use Drakloak: Recon Directive", "ability")
    elif identifier == "dragapult-backup-attachment":
        obs["players"][0]["active"] = pokemon("Dragapult ex", energy=["Psychic", "Fire"])
        obs["players"][0]["bench"] = [pokemon("Drakloak", energy=["Psychic"])]
        good = action("g", "Attach Fire Energy to Bench 1: Drakloak", "attach-energy", target="bench 1")
        bad = action("b", "Attach Fire Energy to Active", "attach-energy", target="active")
    elif identifier == "dragapult-conditional-budew-lock":
        obs["players"][0]["active"] = pokemon("Budew")
        obs["players"][1]["active"] = pokemon("Crustle")
        good, bad = action("g", "Attack: Itchy Pollen", "attack"), action("b", "End turn", "pass")
    elif identifier == "dragapult-initiate-with-backup":
        obs["players"][0]["active"] = pokemon("Dragapult ex")
        obs["players"][0]["bench"] = [pokemon("Drakloak", energy=["Psychic"])]
        good, bad = action("g", "Attack: Phantom Dive", "attack"), action("b", "End turn", "pass")
    elif identifier in {"dragapult-counters-unprotected-crustle", "dragapult-pressure-unmist-crustle"}:
        obs["prompt"] = {"message": "Place damage counters"}
        obs["players"][1]["bench"] = [pokemon("Crustle"), pokemon("Dudunsparce")]
        good, bad = action("g", "Choose Bench 1: Crustle", "choice"), action("b", "Choose Bench 2: Dudunsparce", "choice")
    elif identifier == "dragapult-large-hand-judge":
        obs["players"][1]["handCount"] = 8
        good, bad = action("g", "Play Judge", cardId="POR-76"), action("b", "End turn", "pass")
    elif identifier == "dragapult-hammer-commitment":
        obs["players"][1]["active"] = pokemon("Crustle", energy=["Grass", "Mist Energy"])
        good, bad = action("g", "Play Crushing Hammer", cardId="POR-71"), action("b", "End turn", "pass")
    elif identifier == "dragapult-judge-before-kangaskhan-ko":
        obs["players"][1]["active"] = pokemon("Mega Kangaskhan ex", hp=300, damage=120)
        obs["players"][1]["handCount"] = 8
        good, bad = action("g", "Play Judge", cardId="POR-76"), action("b", "Attack: Phantom Dive", "attack")
    elif identifier == "dragapult-dudunsparce-avoid-spiky":
        obs["players"][0]["active"] = pokemon("Dudunsparce")
        obs["players"][1]["active"] = pokemon("Crustle", energy=["Spiky Energy"])
        good, bad = action("g", "Use Run Away Draw", "ability"), action("b", "Attack: Land Crash", "attack")
    elif identifier == "dragapult-dudunsparce-run-away":
        obs["players"][0]["active"] = pokemon("Dudunsparce")
        obs["players"][0]["bench"] = [pokemon("Drakloak")]
        good, bad = action("g", "Use Run Away Draw", "ability"), action("b", "End turn", "pass")
    else:
        raise AssertionError(identifier)
    obs["legalActions"] = [good, bad]
    return obs, good, bad


@pytest.mark.parametrize("probe", probes.PROBES, ids=lambda item: item.id)
def test_every_probe_has_adherent_nonadherent_and_nonqualifying_fixed_positions(probe):
    observation, good, bad = probe_case(probe.id)
    assert probe.evaluate(copy.deepcopy(observation), copy.deepcopy(good)) is True
    assert probe.evaluate(copy.deepcopy(observation), copy.deepcopy(bad)) is False
    observation["legalActions"] = []
    assert probe.evaluate(observation, good) is None


def test_registry_covers_every_effective_principle_once_to_three_times():
    effective = json.loads((ROOT / "docs/validation/strategy-baseline-v1-guarded-p3-pilot-2026-09-21/effective-contract.json").read_text())
    principle_ids = {item["id"] for playbook in effective["playbooks"].values() for item in playbook["principles"]}
    probes.assert_contract_coverage(principle_ids)
    assert len(principle_ids) == 23


def test_wilson_seed_plan_and_matched_schedule_are_deterministic():
    interval = probes.wilson(5, 10)
    assert interval["low"] == pytest.approx(0.2366, abs=1e-4)
    assert interval["high"] == pytest.approx(0.7634, abs=1e-4)
    first, second = baseline.schedule(), baseline.schedule()
    assert first == second and len(first) == 48
    assert len({item["seed"] for item in first}) == 48
    assert not ({item["seed"] for item in first} & set(baseline.PILOT_SEEDS))


def test_replay_checkpoint_hash_resume_and_private_actor_view(tmp_path):
    actor = base_observation()
    actor["players"][0]["active"] = pokemon("Dragapult ex")
    actor["legalActions"] = [action("g", "Play Poké Pad", cardId="POR-81"),
                              action("b", "Use Drakloak: Recon Directive", "ability")]
    other = copy.deepcopy(actor)
    other["playerId"] = 1
    other["legalActions"] = [action("x", "Choose Bench 1: Crustle", "choice")]
    replay = {"id": "private-view-test", "status": "finished", "outcome": {"winner": 0, "reason": "rules-terminal"},
              "frames": [{"decisionIndex": 0, "actor": 0, "action": actor["legalActions"][0],
                          "observations": [actor, other]}]}
    artifact = baseline.gzip_replay(tmp_path / "replay.json.gz", replay)
    row = {"key": "P1:test:0:1", "policy": "P1", "cell": {"id": "test", "decks": ["dragapult", "crustle"], "firstPlayer": 0},
           "gameIndex": 0, "seed": 1, "status": "finished", "outcome": replay["outcome"],
           "details": {"decisions": 1, "guard": None, "search": None}, "replay": artifact}
    rows = tmp_path / "games.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    assert baseline.load_rows(rows)[row["key"]]["replay"]["sha256"] == artifact["sha256"]
    results, gaps, _ = baseline.analyze_probes([row])
    assert results["firstPlayerPreference"]["status"] == "structurally-unmeasurable"
    target = next(item for item in results["probes"] if item["policy"] == "P1" and item["id"] == "dragapult-information-poke-pad-first")
    assert target["headline"] == {"k": 1, "n": 1, "rate": 1.0,
                                  "wilson95": probes.wilson(1, 1), "sufficient": False}
    assert all(item["probeId"] != "dragapult-counters-unprotected-crustle" for item in gaps)


def test_gap_scenario_selection_respects_rank_and_deduplication():
    gaps = [{"policy": "P1", "probeId": "a"}, {"policy": "P2", "probeId": "b"}]
    failures = [
        {"policy": "P2", "probeId": "b", "principleId": "p2", "gameKey": "2", "replayPath": "r2", "replayId": "r2",
         "decisionIndex": 2, "actor": 0, "observationHash": "same", "action": {}, "observation": {}},
        {"policy": "P1", "probeId": "a", "principleId": "p1", "gameKey": "1", "replayPath": "r1", "replayId": "r1",
         "decisionIndex": 1, "actor": 0, "observationHash": "same", "action": {}, "observation": {}},
    ]
    selected = baseline.draft_scenarios(failures, gaps)
    assert len(selected) == 1 and selected[0]["probeId"] == "a"
    assert selected[0]["status"] == "unlabeled-awaiting-human-review"


def test_typescript_saved_observation_evaluator_matches_worker(tmp_path):
    from ptcg_lab.engine import EngineClient

    tsx = Path("/Users/admin/Documents/ChatGPT/Pokemon Ai project/node_modules/.bin/tsx")
    if not tsx.exists():
        pytest.skip("Configured local tsx runtime is unavailable")
    with EngineClient(ROOT) as engine:
        state = engine.request("reset", {"seed": 9876, "decks": ["crustle", "dragapult"], "firstPlayer": 0})
        observation = state["observation"]
        worker = engine.request("choose", {"policy": "heuristic", "seed": 123456})["id"]
    record = {"observation": observation, "positionHash": "fixed"}
    assert baseline.typescript_choices(tsx, ROOT, [record]) == [worker]


def test_typescript_loader_is_resolved_beside_node_modules_bin(tmp_path, monkeypatch):
    node_modules = tmp_path / "node_modules"
    tsx = node_modules / ".bin" / "tsx"
    loader = node_modules / "tsx" / "dist" / "loader.mjs"
    tsx.parent.mkdir(parents=True)
    loader.parent.mkdir(parents=True)
    tsx.touch()
    loader.touch()

    captured = {}

    class Result:
        stdout = '[{"actionId":"chosen"}]'

    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr(baseline.subprocess, "run", fake_run)
    records = [{"observation": {}, "positionHash": "fixed"}]
    assert baseline.typescript_choices(tsx, ROOT, records) == ["chosen"]
    assert captured["command"][2] == str(loader)
