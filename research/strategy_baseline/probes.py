"""Conservative acting-view probes for Strategy Baseline v1.

Every evaluator receives only the acting player's observation and the action
accepted at that decision.  ``None`` means that the narrow observable trigger
did not apply; otherwise the boolean is adherence to the predeclared behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable


Evaluator = Callable[[dict, dict], bool | None]


@dataclass(frozen=True)
class Probe:
    id: str
    principle_id: str
    deck: str
    severity: int
    coverage: str
    rules_tests: tuple[str, ...]
    trigger: str
    preferred: str
    evaluate: Evaluator

    def record(self) -> dict:
        return {key: getattr(self, key) for key in (
            "id", "principle_id", "deck", "severity", "coverage",
            "rules_tests", "trigger", "preferred",
        )}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"low": None, "high": None}
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"low": max(0.0, centre - margin), "high": min(1.0, centre + margin)}


def _text(action: dict) -> str:
    return " ".join(str(action.get(key, "")) for key in ("label", "cardId", "type", "target")).lower()


def _legal(observation: dict, *needles: str) -> list[dict]:
    lowered = tuple(needle.lower() for needle in needles)
    return [action for action in observation.get("legalActions", [])
            if any(needle in _text(action) for needle in lowered)]


def _chosen(action: dict, *needles: str) -> bool:
    value = _text(action)
    return any(needle.lower() in value for needle in needles)


def _pokemon(player: dict) -> list[dict]:
    return [item for item in [player.get("active"), *player.get("bench", [])] if item]


def _has_pokemon(player: dict, *names: str) -> bool:
    return any(any(name.lower() in item.get("card", {}).get("name", "").lower() for name in names)
               for item in _pokemon(player))


def _active(observation: dict, own: bool = True) -> dict:
    actor = observation["playerId"]
    return observation["players"][actor if own else 1 - actor].get("active") or {}


def _bench(observation: dict, own: bool = True) -> list[dict]:
    actor = observation["playerId"]
    return observation["players"][actor if own else 1 - actor].get("bench", [])


def _hand(observation: dict, own: bool = True) -> list[dict]:
    actor = observation["playerId"]
    return observation["players"][actor if own else 1 - actor].get("hand", [])


def _card_names(cards: list[dict]) -> set[str]:
    return {card.get("name", "").lower() for card in cards}


def _energy(item: dict) -> list[str]:
    return [str(value).lower() for value in item.get("energy", [])]


def _opening_shell(obs: dict, action: dict) -> bool | None:
    prompt = str(obs.get("prompt", {}).get("message", "")).lower()
    labels = " ".join(_text(item) for item in obs.get("legalActions", []))
    if "choose_starting" not in prompt and "starting" not in prompt:
        return None
    if "kangaskhan" not in labels or "dwebble" not in labels:
        return None
    return _chosen(action, "kangaskhan")


def _one_fortress(obs: dict, action: dict) -> bool | None:
    own, opponent = obs["players"][obs["playerId"]], obs["players"][1 - obs["playerId"]]
    if not _has_pokemon(own, "crustle") or not _has_pokemon(opponent, "dragapult", "drakloak", "dreepy", "budew"):
        return None
    candidates = [item for item in _legal(obs, "dwebble", "crustle") if item.get("type") == "bench"]
    if not candidates:
        return None
    return action.get("id") not in {item["id"] for item in candidates}


def _protection_energy(obs: dict, action: dict) -> bool | None:
    if "crustle" not in _active(obs).get("card", {}).get("name", "").lower():
        return None
    protection = _legal(obs, "mist energy", "spiky energy", "tef-161", "jtg-159")
    growing = _legal(obs, "growing grass", "por-86")
    if not protection or not growing:
        return None
    return action.get("id") in {item["id"] for item in protection}


def _safe_prize_delay(obs: dict, action: dict) -> bool | None:
    active, target = _active(obs), _active(obs, False)
    if "crustle" not in active.get("card", {}).get("name", "").lower() or len(_energy(active)) >= 3:
        return None
    attacks, passes = _legal(obs, "attack:"), [item for item in obs.get("legalActions", []) if item.get("type") == "pass"]
    if not attacks or not passes or obs["players"][1 - obs["playerId"]].get("prizesRemaining") != 6:
        return None
    hp = target.get("card", {}).get("hp", 10**9) - target.get("damage", 0)
    own_damage = max((attack.get("damage", 0) for attack in active.get("card", {}).get("attacks", [])), default=0)
    opposing_damage = max((attack.get("damage", 0) for attack in target.get("card", {}).get("attacks", [])), default=10**9)
    remaining = active.get("card", {}).get("hp", 0) - active.get("damage", 0)
    if own_damage < hp or opposing_damage >= remaining:
        return None
    return action.get("id") in {item["id"] for item in passes}


def _target_answer(obs: dict, action: dict) -> bool | None:
    choices = _legal(obs, "drakloak", "dragapult")
    if not choices or len(obs.get("legalActions", [])) < 2:
        return None
    return action.get("id") in {item["id"] for item in choices}


def _kangaskhan_pivot(obs: dict, action: dict) -> bool | None:
    active = _active(obs)
    if "crustle" not in active.get("card", {}).get("name", "").lower():
        return None
    if not any("kangaskhan" in item.get("card", {}).get("name", "").lower() for item in _bench(obs)):
        return None
    kangaskhan = _legal(obs, "kangaskhan")
    active_attach = [item for item in obs.get("legalActions", []) if item.get("type") == "attach-energy" and item.get("target") == "active"]
    if not kangaskhan or not active_attach or len(_energy(active)) >= 2:
        return None
    return action.get("id") in {item["id"] for item in kangaskhan}


def _fan_kangaskhan(obs: dict, action: dict) -> bool | None:
    active = _active(obs)
    if "kangaskhan" not in active.get("card", {}).get("name", "").lower():
        return None
    fan = [item for item in _legal(obs, "handheld fan", "twm-150") if item.get("type") == "attach-tool"]
    if not fan:
        return None
    active_choices = [item for item in fan if item.get("target") == "active" or "active" in _text(item)]
    return action.get("id") in {item["id"] for item in active_choices or fan}


def _eri_response(obs: dict, action: dict) -> bool | None:
    if "eri" not in str((obs.get("prompt") or {}).get("message", "")).lower():
        return None
    valuable = _legal(obs, "unfair stamp", "crushing hammer", "night stretcher", "poké pad", "poke pad")
    if not valuable:
        return None
    return action.get("id") in {item["id"] for item in valuable}


def _special_red_card(obs: dict, action: dict) -> bool | None:
    cards = _legal(obs, "special red card", "cri-82")
    opponent = obs["players"][1 - obs["playerId"]]
    if not cards or opponent.get("prizesRemaining", 7) > 3 or opponent.get("handCount", 0) < 5:
        return None
    return action.get("id") in {item["id"] for item in cards}


def _lumiose(obs: dict, action: dict) -> bool | None:
    lumiose = _legal(obs, "lumiose", "por-77")
    if not lumiose or any("dwebble" in item.get("card", {}).get("name", "").lower() for item in _pokemon(obs["players"][obs["playerId"]])):
        return None
    if not _has_pokemon(obs["players"][1 - obs["playerId"]], "budew"):
        return None
    prep = [item for item in obs.get("legalActions", []) if item.get("type") in {"attach-energy", "play-supporter"}
            or (item.get("type") == "play-trainer" and "lumiose" not in _text(item))]
    selected_lumiose = action.get("id") in {item["id"] for item in lumiose}
    return not selected_lumiose if prep else selected_lumiose


def _healing_threshold(obs: dict, action: dict) -> bool | None:
    active = _active(obs)
    if ("crustle" not in active.get("card", {}).get("name", "").lower()
            or active.get("damage", 0) <= 0 or len(_energy(active)) != 2):
        return None
    active_attach = [item for item in obs.get("legalActions", [])
                     if item.get("type") == "attach-energy" and (item.get("target") == "active" or "active" in _text(item))]
    other_attach = [item for item in obs.get("legalActions", []) if item.get("type") == "attach-energy" and item not in active_attach]
    if not active_attach or not other_attach:
        return None
    return action.get("id") in {item["id"] for item in active_attach}


def _fan_destination(obs: dict, action: dict) -> bool | None:
    message = str((obs.get("prompt") or {}).get("message", "")).lower()
    if "energy" not in message or "bench" not in message:
        return None
    energized = [index for index, item in enumerate(_bench(obs, False)) if _energy(item)]
    if not energized:
        return None
    preferred = [item for item in obs.get("legalActions", [])
                 if any(f"bench {index + 1}" in _text(item) for index in energized)]
    if not preferred:
        return None
    return action.get("id") in {item["id"] for item in preferred}


def _information_order(obs: dict, action: dict) -> bool | None:
    pad, recon = _legal(obs, "poké pad", "poke pad", "por-81"), _legal(obs, "recon directive")
    if not pad or not recon:
        return None
    return action.get("id") in {item["id"] for item in pad}


def _backup_attachment(obs: dict, action: dict) -> bool | None:
    active = _active(obs)
    if "dragapult" not in active.get("card", {}).get("name", "").lower() or len(_energy(active)) < 2:
        return None
    backup_indices = [index for index, item in enumerate(_bench(obs))
                      if any(name in item.get("card", {}).get("name", "").lower() for name in ("drakloak", "dragapult"))]
    if not backup_indices:
        return None
    backup = [item for item in obs.get("legalActions", []) if item.get("type") == "attach-energy"
              and any(f"bench {index + 1}" in _text(item) for index in backup_indices)]
    active_actions = [item for item in obs.get("legalActions", []) if item.get("type") == "attach-energy" and item.get("target") == "active"]
    if not backup or not active_actions:
        return None
    return action.get("id") in {item["id"] for item in backup}


def _conditional_budew(obs: dict, action: dict) -> bool | None:
    if "budew" not in _active(obs).get("card", {}).get("name", "").lower():
        return None
    if not _has_pokemon(obs["players"][1 - obs["playerId"]], "dwebble", "crustle", "kangaskhan"):
        return None
    attacks = _legal(obs, "itchy pollen")
    if not attacks:
        return None
    return action.get("id") in {item["id"] for item in attacks}


def _initiate_chain(obs: dict, action: dict) -> bool | None:
    phantom = _legal(obs, "phantom dive")
    backup = [item for item in _bench(obs) if any(name in item.get("card", {}).get("name", "").lower()
                                                for name in ("drakloak", "dragapult")) and _energy(item)]
    if not phantom or not backup:
        return None
    return action.get("id") in {item["id"] for item in phantom}


def _counter_target(obs: dict, action: dict) -> bool | None:
    message = str((obs.get("prompt") or {}).get("message", "")).lower()
    if "counter" not in message:
        return None
    targets = []
    for index, item in enumerate(_bench(obs, False)):
        if "crustle" in item.get("card", {}).get("name", "").lower() and not any("mist" in energy for energy in _energy(item)):
            targets.extend(candidate for candidate in obs.get("legalActions", []) if f"bench {index + 1}" in _text(candidate))
    if not targets:
        return None
    return action.get("id") in {item["id"] for item in targets}


def _combined_disruption(obs: dict, action: dict) -> bool | None:
    judge = _legal(obs, "judge", "por-76")
    opponent = obs["players"][1 - obs["playerId"]]
    if not judge or opponent.get("handCount", 0) < 7:
        return None
    return action.get("id") in {item["id"] for item in judge}


def _hammer(obs: dict, action: dict) -> bool | None:
    hammer = _legal(obs, "crushing hammer", "por-71")
    target = _active(obs, False)
    if not hammer or "crustle" not in target.get("card", {}).get("name", "").lower() or not _energy(target):
        return None
    should_hammer = len(_energy(target)) >= 2 or any(any(kind in energy for kind in ("mist", "spiky")) for energy in _energy(target))
    selected = action.get("id") in {item["id"] for item in hammer}
    return selected == should_hammer


def _race_kangaskhan(obs: dict, action: dict) -> bool | None:
    judge, phantom = _legal(obs, "judge", "por-76"), _legal(obs, "phantom dive")
    target = _active(obs, False)
    if (not judge or not phantom or "kangaskhan" not in target.get("card", {}).get("name", "").lower()
            or obs["players"][1 - obs["playerId"]].get("handCount", 0) < 7):
        return None
    remaining = target.get("card", {}).get("hp", 10**9) - target.get("damage", 0)
    if remaining > 200:
        return None
    return action.get("id") in {item["id"] for item in judge}


def _dudunsparce_avoid(obs: dict, action: dict) -> bool | None:
    if "dudunsparce" not in _active(obs).get("card", {}).get("name", "").lower():
        return None
    target = _active(obs, False)
    if "crustle" not in target.get("card", {}).get("name", "").lower() or not any("spiky" in item for item in _energy(target)):
        return None
    attacks = [item for item in obs.get("legalActions", []) if item.get("type") == "attack"]
    alternatives = _legal(obs, "run away draw", "crushing hammer", "boss's orders")
    if not attacks or not alternatives:
        return None
    return action.get("id") not in {item["id"] for item in attacks}


def _dudunsparce_escape(obs: dict, action: dict) -> bool | None:
    if "dudunsparce" not in _active(obs).get("card", {}).get("name", "").lower():
        return None
    escape = _legal(obs, "run away draw")
    if not escape or not _bench(obs):
        return None
    return action.get("id") in {item["id"] for item in escape}


def _probe(identifier: str, principle: str, deck: str, severity: int, trigger: str, preferred: str,
           evaluator: Evaluator, *rules: str, coverage: str = "partial") -> Probe:
    return Probe(identifier, principle, deck, severity, coverage, tuple(rules), trigger, preferred, evaluator)


PROBES = (
    _probe("crustle-opening-kangaskhan", "crustle-opening-shell", "crustle", 2,
           "Opening selection contains both Mega Kangaskhan ex and Dwebble.", "Select Mega Kangaskhan ex.", _opening_shell),
    _probe("crustle-one-fortress-restraint", "crustle-one-fortress", "crustle", 3,
           "A live Crustle exists versus a publicly visible Dragapult line and another Dwebble can be benched.", "Do not bench the extra Dwebble.", _one_fortress),
    _probe("crustle-energy-jobs", "crustle-energy-has-jobs", "crustle", 2,
           "Active Crustle can receive either Mist/Spiky or Growing Grass Energy.", "Choose Mist or Spiky.", _protection_energy),
    _probe("crustle-safe-first-prize-delay", "crustle-delay-first-prize", "crustle", 3,
           "A safe first-prize knockout is optional while Crustle remains below the three-Energy survival threshold.", "End the turn instead of taking the knockout.", _safe_prize_delay),
    _probe("crustle-target-public-answer", "crustle-target-answers", "crustle", 3,
           "A target prompt includes Drakloak/Dragapult and another target.", "Choose Drakloak/Dragapult.", _target_answer),
    _probe("crustle-kangaskhan-pivot", "crustle-pivot-after-commitment", "crustle", 2,
           "Active Crustle is underdeveloped and an attachment can develop either it or a benched Kangaskhan.", "Attach to Kangaskhan.", _kangaskhan_pivot),
    _probe("crustle-fan-active-kangaskhan", "crustle-fan-kangaskhan", "crustle", 3,
           "Handheld Fan can be attached while Mega Kangaskhan ex is Active.", "Attach Fan to the Active Kangaskhan.", _fan_kangaskhan,
           "prebaseline-rules:handheld-fan"),
    _probe("crustle-eri-critical-item", "crustle-eri-response-window", "crustle", 2,
           "Eri selection reveals a listed critical Item.", "Select a critical Item instead of finishing empty.", _eri_response,
           "prebaseline-rules:eri"),
    _probe("crustle-special-red-card-window", "crustle-special-red-card-window", "crustle", 2,
           "Special Red Card is legal at three or fewer opposing Prizes against a hand of at least five.", "Play Special Red Card.", _special_red_card,
           "prebaseline-rules:special-red-card"),
    _probe("crustle-lumiose-last", "crustle-lumiose-under-item-lock", "crustle", 3,
           "Budew is public, no Dwebble is in play, and Lumiose City is legal.", "Take remaining Supporter/attachment actions before Lumiose, then use Lumiose.", _lumiose,
           "prebaseline-rules:lumiose-city"),
    _probe("crustle-healing-third-energy", "crustle-preserve-healing-threshold", "crustle", 3,
           "A damaged Active Crustle has two Energy and attachment targets include Active and elsewhere.", "Attach the third Energy to Active Crustle.", _healing_threshold,
           "prebaseline-rules:jumbo-ice-cream"),
    _probe("crustle-dragapult-energy-priority", "crustle-dragapult-energy-priority", "crustle", 3,
           "Active Crustle can receive Mist/Spiky or Growing Grass Energy.", "Choose Mist or Spiky.", _protection_energy,
           "prebaseline-rules:mist-energy", "prebaseline-rules:spiky-energy"),
    _probe("crustle-fan-stack-destination", "crustle-fan-destination", "crustle", 2,
           "A Fan Energy-move prompt has a legal opposing Bench target that already holds Energy.", "Move Energy to the already energized Bench target.", _fan_destination,
           "prebaseline-rules:handheld-fan"),
    _probe("dragapult-information-poke-pad-first", "dragapult-information-order", "dragapult", 1,
           "Poké Pad and Recon Directive are simultaneously legal.", "Use Poké Pad first.", _information_order),
    _probe("dragapult-backup-attachment", "dragapult-stay-ahead-on-energy", "dragapult", 2,
           "Active Dragapult is ready and Energy can attach to it or a backup Drakloak/Dragapult.", "Attach to the backup.", _backup_attachment),
    _probe("dragapult-conditional-budew-lock", "dragapult-conditional-budew", "dragapult", 2,
           "Budew is Active against a publicly visible Crustle line and Itchy Pollen is legal.", "Use Itchy Pollen.", _conditional_budew,
           "prebaseline-rules:itchy-pollen"),
    _probe("dragapult-initiate-with-backup", "dragapult-initiate-with-a-chain", "dragapult", 3,
           "Phantom Dive is legal and an energized backup Drakloak/Dragapult exists.", "Use Phantom Dive.", _initiate_chain),
    _probe("dragapult-counters-unprotected-crustle", "dragapult-place-damage-for-prizes", "dragapult", 3,
           "A counter-placement prompt can target a benched Crustle without Mist.", "Target the unprotected Crustle.", _counter_target,
           "prebaseline-rules:mist-energy"),
    _probe("dragapult-large-hand-judge", "dragapult-combine-disruption", "dragapult", 2,
           "Judge is legal against an opposing hand of at least seven.", "Play Judge.", _combined_disruption),
    _probe("dragapult-hammer-commitment", "dragapult-hammer-committed-crustle", "dragapult", 3,
           "Crushing Hammer is legal against an Active Crustle with attached Energy.", "Use it at two Energy or for Mist/Spiky; otherwise preserve it.", _hammer,
           "prebaseline-rules:jumbo-ice-cream", "prebaseline-rules:mist-energy", "prebaseline-rules:spiky-energy"),
    _probe("dragapult-judge-before-kangaskhan-ko", "dragapult-race-kangaskhan-opening", "dragapult", 3,
           "Judge and Phantom Dive are legal against a KO-range Kangaskhan with a large hand.", "Play Judge before attacking.", _race_kangaskhan),
    _probe("dragapult-pressure-unmist-crustle", "dragapult-pressure-underprotected-crustle", "dragapult", 3,
           "A counter-placement prompt can target a benched Crustle without Mist.", "Target the unprotected Crustle.", _counter_target,
           "prebaseline-rules:mist-energy"),
    _probe("dragapult-dudunsparce-avoid-spiky", "dragapult-dudunsparce-crustle-role", "dragapult", 3,
           "Active Dudunsparce faces Spiky Crustle and a non-attack alternative exists.", "Do not attack into Spiky.", _dudunsparce_avoid,
           "prebaseline-rules:spiky-energy"),
    _probe("dragapult-dudunsparce-run-away", "dragapult-dudunsparce-crustle-role", "dragapult", 2,
           "Active Dudunsparce can use Run Away Draw and has a replacement Active.", "Use Run Away Draw.", _dudunsparce_escape,
           "prebaseline-rules:run-away-draw"),
)


PROBES_BY_ID = {probe.id: probe for probe in PROBES}


def assert_contract_coverage(principle_ids: set[str]) -> None:
    mapped = {probe.principle_id for probe in PROBES}
    if mapped != principle_ids:
        raise ValueError(f"Probe registry differs from effective principles: missing={sorted(principle_ids - mapped)}, extra={sorted(mapped - principle_ids)}")
    if not all(1 <= sum(item.principle_id == principle for item in PROBES) <= 3 for principle in principle_ids):
        raise ValueError("Every principle must have one to three probes")
    if not all(probe.severity in {1, 2, 3} for probe in PROBES):
        raise ValueError("Probe severity must be predeclared on the 1..3 scale")
