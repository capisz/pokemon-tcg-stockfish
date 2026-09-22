"""Resumable, measurement-only Strategy Baseline v1 runner."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pilot import (  # noqa: E402
    CELLS,
    PILOT_SEEDS,
    RecordingEngine,
    effective_contract,
    file_hash,
    run_python_game,
)
from probes import PROBES, assert_contract_coverage, canonical_hash, wilson  # noqa: E402
from ptcg_lab.decision_guard import VERSION as GUARD_VERSION  # noqa: E402
from ptcg_lab.engine import EngineClient  # noqa: E402
from ptcg_lab.selfplay import Agent, search_choice  # noqa: E402
from ptcg_lab.strategy_contract import validate_strategy_revision_v12  # noqa: E402


POLICIES = ("P1", "P2", "P3", "P4")
EXACT_REVIEWS = (
    "530af43192745703ac2acf2dc7ea4a4ac1520f226b4cba3a4636eac617f954e4",
    "832239559647d57880eba66a7f28718c1bebb0ebd82f5f66040bcd9f7a4aab32",
    "b6a6c6526c4fb448e856f471edb25d94c048cb9c34c0d3ab9456502657ac0f0c",
    "da82cf9119099fbff943b254a499f579c8c8512020f5cbdcaf767512ab7ee23b",
)
STALE_REVIEWS = (
    "1e803b1a30c55cf8f61bbe7ee1cc2ec6aca2b5dbaa7f20f416560aeaf84a0510",
    "49f4021acf629bce7b69352b8eee611f490665f771b222c1aa0b042fae6fa36d",
    "c0d1cfe46ceda16810fc05b4e2290083685000e9186962350cc33c5e0875ea82",
    "f34b1c1bc4182ba857136585dcb3c683b195e59487830308444a1eab3be51260",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def game_count(cell: dict) -> int:
    return 4 if cell["id"].startswith("cross-") else 8


def seed_for(cell_id: str, game_index: int) -> int:
    token = f"strategy-baseline-v1-main|{cell_id}|{game_index}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def schedule() -> list[dict]:
    entries, seen = [], set(PILOT_SEEDS)
    for cell in CELLS:
        for game_index in range(game_count(cell)):
            seed = seed_for(cell["id"], game_index)
            if seed in seen:
                raise ValueError("Main seed collides with a pilot or another main seed")
            seen.add(seed)
            entries.append({"cell": cell, "gameIndex": game_index, "seed": seed})
    if len(entries) != 48:
        raise ValueError("Frozen plan must contain 48 matched games per policy")
    return entries


def metadata(root: Path, checkpoint: Path) -> dict:
    validated = validate_strategy_revision_v12(root)
    if validated["activeRevision"] != "v1.2":
        raise RuntimeError("Strategy v1.2 is not active")
    with EngineClient(root) as engine:
        health = engine.request("health")
        decks = engine.request("decks")
    deck_hashes = {item["id"]: item["listHash"] for item in decks if item["id"] in {"crustle", "dragapult"}}
    effective = effective_contract(root, health["engineVersion"], health["engineBuildHash"])
    effective_hash = hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":"),
                                                  ensure_ascii=False).encode()).hexdigest()
    principles = {item["id"] for playbook in effective["playbooks"].values() for item in playbook["principles"]}
    assert_contract_coverage(principles)
    commit = git(root, "rev-parse", "HEAD")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("Baseline must start from a clean committed harness")
    return {
        "schemaVersion": 1,
        "id": "strategy-baseline-v1-frozen-inputs",
        "harnessCommit": commit,
        "engineFingerprint": health["engineVersion"],
        "engineBuildHash": health["engineBuildHash"],
        "deckListHashes": deck_hashes,
        "effectiveContractHash": effective_hash,
        "effectiveContract": effective,
        "checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint)},
        "guardVersion": GUARD_VERSION,
        "policies": {
            "P1": "TypeScript choose heuristic",
            "P2": "Python heuristic_action_score",
            "P3": "frozen guide-policy checkpoint through visible-repetition-v1",
            "P4": "ISMCTS 200 ms with Python heuristic_action_score fallback",
        },
        "searchBudgetMs": 200,
        "maxDecisions": 1000,
        "seedDerivation": "uint32 big-endian first four SHA-256 bytes of strategy-baseline-v1-main|cell-id|game-index",
        "schedule": schedule(),
        "probeRegistryHash": canonical_hash([probe.record() for probe in PROBES]),
        "exactTeachingReviews": list(EXACT_REVIEWS),
        "staleTeachingReviews": list(STALE_REVIEWS),
    }


def comparable_frozen(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "effectiveContract"}


def verify_runtime_identity(root: Path, frozen: dict) -> None:
    with EngineClient(root) as engine:
        health = engine.request("health")
        decks = engine.request("decks")
    current = {item["id"]: item["listHash"] for item in decks if item["id"] in {"crustle", "dragapult"}}
    if (health["engineVersion"] != frozen["engineFingerprint"]
            or health["engineBuildHash"] != frozen["engineBuildHash"]
            or current != frozen["deckListHashes"]):
        raise RuntimeError("Frozen engine or deck identity drifted during the baseline")


def game_key(policy: str, item: dict) -> str:
    return f"{policy}:{item['cell']['id']}:{item['gameIndex']}:{item['seed']}"


def gzip_replay(path: Path, replay: dict) -> dict:
    decoded = (json.dumps(replay, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(decoded)
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)
    encoded = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "decodedSha256": hashlib.sha256(decoded).hexdigest(),
        "decodedBytes": len(decoded),
    }


def read_replay(entry: dict) -> dict:
    path = Path(entry["path"])
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != entry["sha256"]:
        raise ValueError(f"Replay hash mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        replay = json.load(source)
    decoded = (json.dumps(replay, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if hashlib.sha256(decoded).hexdigest() != entry["decodedSha256"]:
        raise ValueError(f"Decoded replay hash mismatch: {path}")
    return replay


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["key"] in rows:
            raise ValueError(f"Duplicate completed game: {row['key']}")
        read_replay(row["replay"])
        rows[row["key"]] = row
    return rows


def run_game(root: Path, checkpoint: Path, policy: str, item: dict, max_decisions: int) -> tuple[dict, dict]:
    cell, seed = item["cell"], item["seed"]
    if policy == "P1":
        with EngineClient(root, timeout=300) as engine:
            replay = engine.request("run", {"seed": seed, "decks": cell["decks"],
                                             "firstPlayer": cell["firstPlayer"], "policy": "heuristic",
                                             "maxDecisions": max_decisions})
        details = {"decisions": max(0, len(replay.get("frames", [])) - 1), "failure": None,
                   "guard": None, "search": None}
        if replay.get("status") != "finished":
            replay["status"], replay["outcome"] = "truncated", None
    else:
        replay, details = run_python_game(root, cell, seed, policy, max_decisions, checkpoint)
    return replay, details


def action_summary(action: dict) -> dict:
    return {key: action[key] for key in ("id", "type", "label", "cardId", "target", "choiceOperation",
                                         "sourceRef", "targetRef") if key in action}


def observation_summary(observation: dict) -> dict:
    def pokemon(item: dict | None):
        if not item:
            return None
        return {"name": item.get("card", {}).get("name"), "damage": item.get("damage"),
                "energy": item.get("energy", []), "tools": item.get("tools", [])}
    return {
        "playerId": observation.get("playerId"), "turn": observation.get("turn"),
        "phase": observation.get("phase"), "prompt": observation.get("prompt"),
        "players": [{"active": pokemon(player.get("active")), "bench": [pokemon(item) for item in player.get("bench", [])],
                     "handCount": player.get("handCount"), "deckCount": player.get("deckCount"),
                     "prizesRemaining": player.get("prizesRemaining")} for player in observation.get("players", [])],
        "legalActions": [action_summary(item) for item in observation.get("legalActions", [])],
        "historyTail": observation.get("history", [])[-8:],
    }


def policy_tag(row: dict, decision_index: int) -> str | None:
    search = row["details"].get("search")
    if not search:
        return None
    tag = next((item for item in search.get("tags", []) if item["decisionIndex"] == decision_index), None)
    return tag.get("kind") if tag else None


def analyze_probes(rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    buckets = {(policy, probe.id): {"headline": [], "secondary": [], "examples": [], "failures": []}
               for policy in POLICIES for probe in PROBES}
    eligible_sides = Counter()
    for row in rows:
        replay = read_replay(row["replay"])
        seen = set()
        for actor, deck in enumerate(row["cell"]["decks"]):
            eligible_sides[(row["policy"], deck)] += 1
        for frame in replay.get("frames", []):
            action = frame.get("action")
            actor = frame.get("actor")
            observations = frame.get("observations", [])
            if not action or actor not in (0, 1) or len(observations) != 2:
                continue
            observation = observations[actor]
            if observation.get("playerId") != actor:
                raise ValueError("Probe analysis received a non-acting observation")
            deck = row["cell"]["decks"][actor]
            for probe in (item for item in PROBES if item.deck == deck):
                adherence = probe.evaluate(observation, action)
                if adherence is None:
                    continue
                key = (row["policy"], probe.id)
                item = {
                    "gameKey": row["key"], "replayPath": row["replay"]["path"],
                    "replayId": replay.get("id"), "decisionIndex": frame["decisionIndex"], "actor": actor,
                    "observationHash": canonical_hash(observation), "action": action_summary(action),
                    "adherent": adherence, "p4DecisionKind": policy_tag(row, frame["decisionIndex"]),
                    "observationSummary": observation_summary(observation),
                }
                buckets[key]["secondary"].append(item)
                side_key = (row["key"], actor, probe.id)
                if side_key not in seen:
                    buckets[key]["headline"].append(item)
                    seen.add(side_key)
                    if not adherence:
                        buckets[key]["failures"].append({**item, "observation": observation})
                if len(buckets[key]["examples"]) < 3:
                    buckets[key]["examples"].append(item)
    results, all_failures, gaps = [], [], []
    for policy in POLICIES:
        for probe in PROBES:
            bucket = buckets[(policy, probe.id)]
            headline, secondary = bucket["headline"], bucket["secondary"]
            k, n = sum(item["adherent"] for item in headline), len(headline)
            secondary_k, secondary_n = sum(item["adherent"] for item in secondary), len(secondary)
            split = {}
            if policy == "P4":
                for kind in ("searched", "fallback"):
                    selected = [item for item in secondary if item["p4DecisionKind"] == kind]
                    split[kind] = {"k": sum(item["adherent"] for item in selected), "n": len(selected)}
            result = {**probe.record(), "policy": policy,
                      "headline": {"k": k, "n": n, "rate": k / n if n else None,
                                   "wilson95": wilson(k, n), "sufficient": n >= 20},
                      "allQualifying": {"k": secondary_k, "n": secondary_n,
                                        "rate": secondary_k / secondary_n if secondary_n else None},
                      "p4DecisionSplit": split or None,
                      "examples": [{key: value for key, value in item.items() if key != "observation"}
                                   for item in bucket["examples"]]}
            results.append(result)
            denominator = eligible_sides[(policy, probe.deck)]
            failures = n - k
            gaps.append({"policy": policy, "probeId": probe.id, "principleId": probe.principle_id,
                         "severity": probe.severity, "headlineFailures": failures, "headlineN": n,
                         "eligibleGameSides": denominator,
                         "gapScore": failures / denominator * probe.severity if denominator else 0,
                         "insufficient": n < 20})
            all_failures.extend({**item, "policy": policy, "probeId": probe.id,
                                 "principleId": probe.principle_id, "severity": probe.severity}
                                for item in bucket["failures"])
    gaps.sort(key=lambda item: (-item["gapScore"], -item["headlineFailures"], item["policy"], item["probeId"]))
    return {"schemaVersion": 1, "probes": results}, gaps[:5], all_failures


def outcome_metrics(rows: list[dict]) -> dict:
    groups = defaultdict(Counter)
    lengths, deckouts = [], 0
    for row in rows:
        lengths.append(row["details"]["decisions"])
        outcome, status = row.get("outcome"), row["status"]
        if outcome and "deck" in str(outcome.get("reason", "")).lower():
            deckouts += 1
        for actor, deck in enumerate(row["cell"]["decks"]):
            opponent = row["cell"]["decks"][1 - actor]
            perspective = f"{deck}-mirror" if deck == opponent else f"{deck}-to-{opponent}"
            dimensions = {
                "perspective": perspective,
                "seat": f"seat{actor}",
                "firstPlayer": str(actor == row["cell"]["firstPlayer"]).lower(),
            }
            if status != "finished":
                category = "unfinished"
            elif not outcome or outcome.get("winner") is None:
                category = "draw"
            else:
                category = "win" if outcome["winner"] == actor else "loss"
            for dimension, value in dimensions.items():
                groups[(row["policy"], dimension, value)][category] += 1
                groups[(row["policy"], dimension, value)]["scheduled"] += 1
    output = []
    for (policy, dimension, value), counts in sorted(groups.items()):
        n = counts["scheduled"]
        categories = {category: {"k": counts[category], "n": n, "rate": counts[category] / n,
                                 "wilson95": wilson(counts[category], n)}
                      for category in ("win", "draw", "loss", "unfinished")}
        output.append({"policy": policy, "dimension": dimension, "value": value, "categories": categories})
    guard = {policy: sum((row["details"].get("guard") or {}).get("filteredDecisions", 0)
                         for row in rows if row["policy"] == policy) for policy in POLICIES}
    search = {}
    for policy in POLICIES:
        items = [row["details"].get("search") for row in rows if row["policy"] == policy and row["details"].get("search")]
        attempts = sum(item.get("searchAttempts", 0) for item in items)
        search[policy] = {"searchedDecisions": sum(item.get("searchedDecisions", 0) for item in items),
                          "fallbackDecisions": sum(item.get("fallbackDecisions", 0) for item in items),
                          "searchAttempts": attempts,
                          "totalIterations": sum(item.get("totalIterations", 0) for item in items)}
        search[policy]["iterationsPerAttempt"] = search[policy]["totalIterations"] / attempts if attempts else None
    return {"schemaVersion": 1, "outcomes": output, "gameLengthDecisions": {
                "mean": statistics.mean(lengths), "median": statistics.median(lengths),
                "min": min(lengths), "max": max(lengths)},
            "deckOuts": deckouts, "guardFilteredDecisions": guard, "search": search,
            "claimBoundary": "Context metrics only; this baseline does not establish playing strength."}


def teaching_seed(policy: str, position_hash: str) -> int:
    return int.from_bytes(hashlib.sha256(f"teaching|{policy}|{position_hash}".encode()).digest()[:4], "big")


def typescript_choices(tsx: Path, root: Path, records: list[dict]) -> list[str]:
    payload = [{"observation": record["observation"], "seed": teaching_seed("P1", record["positionHash"])}
               for record in records]
    # `.bin/tsx` resolves from `<node_modules>/.bin`, so the package loader is
    # a sibling of `.bin`, not nested below a second `tsx` directory.
    loader = tsx.parent.parent / "tsx" / "dist" / "loader.mjs"
    if not loader.exists():
        raise FileNotFoundError(f"tsx loader is unavailable: {loader}")
    command = ["node", "--import", str(loader), str(root / "research/strategy_baseline/policy_eval.ts")]
    result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, check=True, cwd=root)
    return [item["actionId"] for item in json.loads(result.stdout)]


def grade_teaching(root: Path, checkpoint: Path, review_root: Path, tsx: Path) -> dict:
    records = [json.loads((review_root / f"{identifier}.json").read_text(encoding="utf-8")) for identifier in EXACT_REVIEWS]
    if any(record.get("reviewHash") != identifier for record, identifier in zip(records, EXACT_REVIEWS)):
        raise ValueError("Exact teaching review identity mismatch")
    p1 = typescript_choices(tsx, root, records)
    choices = {"P1": p1, "P2": [], "P3": [], "P4": []}
    heuristic_agents, guide_agents = {}, {}
    engine = RecordingEngine(EngineClient(root, timeout=300))
    try:
        for record in records:
            observation, position = record["observation"], record["positionHash"]
            for policy, cache in (("P2", heuristic_agents), ("P3", guide_agents)):
                seed = teaching_seed(policy, position)
                agent = cache.setdefault((policy, seed), Agent(str(checkpoint) if policy == "P3" else "heuristic", seed,
                                                               allow_experimental=policy == "P3"))
                if policy == "P3":
                    from pilot import select_policy_action
                    selected, _ = select_policy_action(policy, agent, observation, {})
                    choices[policy].append(selected["id"])
                else:
                    choices[policy].append(agent.choose(observation))
            before = len(engine.searches)
            action, _ = search_choice(engine, observation, Agent("heuristic", teaching_seed("P4", position)),
                                      seed=teaching_seed("P4", position), budget_ms=200, method="ismcts")
            choices["P4"].append(action)
            if len(engine.searches) == before:
                raise ValueError("Teaching P4 search did not attempt the available search position")
    finally:
        engine.client.close()
    rows = []
    for policy in POLICIES:
        for record, selected in zip(records, choices[policy]):
            rows.append({"policy": policy, "reviewHash": record["reviewHash"], "teachingId": record["id"],
                         "positionHash": record["positionHash"], "selectedActionId": selected,
                         "acceptableActionIds": record["acceptableActionIds"],
                         "acceptableHit": selected in record["acceptableActionIds"],
                         "preferredActionId": None, "preferredHit": None})
    per_record = [{"policy": item["policy"], "reviewHash": item["reviewHash"], "k": int(item["acceptableHit"]), "n": 1}
                  for item in rows]
    unique = []
    for policy in POLICIES:
        values = {}
        for item in (row for row in rows if row["policy"] == policy):
            values.setdefault(item["positionHash"], item["acceptableHit"])
        unique.append({"policy": policy, "k": sum(values.values()), "n": len(values),
                       "rate": sum(values.values()) / len(values) if values else None})
    return {"schemaVersion": 1, "records": rows, "perRecord": per_record, "perUniquePosition": unique,
            "preferredActionRate": None,
            "preferredActionNote": "Unavailable: the exact-compatible reviews have no preferred-action annotation.",
            "abstentions": {policy: 0 for policy in POLICIES},
            "staleReviewsForHumanRereview": list(STALE_REVIEWS)}


def draft_scenarios(failures: list[dict], gaps: list[dict]) -> list[dict]:
    rank = {(item["policy"], item["probeId"]): index for index, item in enumerate(gaps)}
    ordered = sorted((item for item in failures if (item["policy"], item["probeId"]) in rank),
                     key=lambda item: (rank[(item["policy"], item["probeId"])], item["gameKey"], item["decisionIndex"]))
    selected, seen, per_principle = [], set(), Counter()
    for item in ordered:
        if item["observationHash"] in seen or per_principle[item["principleId"]] >= 2:
            continue
        selected.append({"id": f"draft-{len(selected) + 1}", "status": "unlabeled-awaiting-human-review",
                         **{key: item[key] for key in ("policy", "probeId", "principleId", "gameKey", "replayPath",
                                                          "replayId", "decisionIndex", "actor", "observationHash", "action")},
                         "observation": item["observation"]})
        seen.add(item["observationHash"])
        per_principle[item["principleId"]] += 1
        if len(selected) == 10:
            break
    return selected


def report_markdown(frozen: dict, rows: list[dict], probe_results: dict, context: dict,
                    teaching: dict, gaps: list[dict], scenarios: list[dict]) -> str:
    finished = sum(row["status"] == "finished" for row in rows)
    truncated = sum(row["status"] == "truncated" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    lines = ["# Strategy Baseline v1", "", "Status: completed measurement; no policy improvement was made.", "",
             "This is a fixed 192-game simulator baseline against the approved v1.2 playbooks. Outcomes are context, not established playing strength.", "",
             "## Frozen execution", "",
             f"- Harness commit: `{frozen['harnessCommit']}`", f"- Engine: `{frozen['engineFingerprint']}` / `{frozen['engineBuildHash']}`",
             f"- Effective contract: `{frozen['effectiveContractHash']}`", f"- Guide checkpoint: `{frozen['checkpoint']['sha256']}`",
             f"- Games: {len(rows)} scheduled; {finished} finished; {truncated} truncated; {errors} errors.",
             "- Plan: 48 matched seeds per policy; four cross games and eight mirror games per cell.", "",
             "## Measurement summary", "",
             f"- Mean/median game length: {context['gameLengthDecisions']['mean']:.1f} / {context['gameLengthDecisions']['median']:.1f} decisions.",
             f"- Deck-outs: {context['deckOuts']}.",
             f"- P3 guard-filtered decisions: {context['guardFilteredDecisions']['P3']}.",
             f"- P4 searched/fallback decisions: {context['search']['P4']['searchedDecisions']} / {context['search']['P4']['fallbackDecisions']}.",
             "- W/D/L/unfinished Wilson intervals by perspective, seat, and first player are in `context-metrics.json`.", "",
             "## Largest measured strategic gaps", "",
             "| Rank | Policy | Probe | Failures / headline n | Severity | Gap score |",
             "|---:|---|---|---:|---:|---:|"]
    for index, item in enumerate(gaps, 1):
        lines.append(f"| {index} | {item['policy']} | {item['probeId']} | {item['headlineFailures']} / {item['headlineN']} | {item['severity']} | {item['gapScore']:.3f} |")
    lines.extend(["", "Rates with fewer than 20 qualifying headline cases are marked insufficient in the machine-readable results.", "",
                  "## Teaching and review queue", "",
                  f"- Graded four exact-compatible review records covering {teaching['perUniquePosition'][0]['n']} unique positions.",
                  "- Preferred-action rate is unavailable because those reviews contain acceptable sets but no preferred-action annotation.",
                  "- Four stale Crustle records remain listed for human re-review and were not graded or changed.",
                  f"- Proposed {len(scenarios)} unlabeled draft scenarios from the highest-ranked observed failures.", "",
                  "## Evidence", "",
                  "Tracked machine-readable files are under `docs/validation/strategy-baseline-v1-2026-09-21/`. Every private raw replay remains in ignored local artifacts and is checksummed by `replay-manifest.json`.", "",
                  "No engine, heuristic, search, feature, checkpoint, training data, teaching review, or file under `data/competitive` was modified. The next step requires a separate human choice among the measured gaps.", ""])
    return "\n".join(lines)


def finalize(root: Path, output: Path, replay_root: Path, rows: list[dict], frozen: dict,
             checkpoint: Path, review_root: Path, tsx: Path) -> None:
    if len(rows) != 192 or {row["key"] for row in rows} != {
        game_key(policy, item) for policy in POLICIES for item in frozen["schedule"]}:
        raise ValueError("Exactly 192 unique frozen games are required")
    probe_results, gaps, failures = analyze_probes(rows)
    context = outcome_metrics(rows)
    teaching = grade_teaching(root, checkpoint, review_root, tsx)
    scenarios = draft_scenarios(failures, gaps)
    manifest = {"schemaVersion": 1, "root": str(replay_root), "replays": [
        {"gameKey": row["key"], **row["replay"]} for row in rows]}
    write_json(output / "probe-results.json", probe_results)
    write_json(output / "context-metrics.json", context)
    write_json(output / "teaching-results.json", teaching)
    write_json(output / "ranked-gaps.json", {"schemaVersion": 1, "rankingFormula": "headline failures / eligible game-sides * severity", "gaps": gaps})
    write_json(output / "draft-scenarios.json", {"schemaVersion": 1, "scenarios": scenarios})
    write_json(output / "replay-manifest.json", manifest)
    write_json(output / "results.json", {"schemaVersion": 1, "id": "strategy-baseline-v1-results",
        "frozenInputsHash": canonical_hash(comparable_frozen(frozen)), "scheduledGames": len(rows),
        "finished": sum(row["status"] == "finished" for row in rows),
        "truncated": sum(row["status"] == "truncated" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "files": ["probe-results.json", "context-metrics.json", "teaching-results.json", "ranked-gaps.json",
                  "draft-scenarios.json", "replay-manifest.json"]})
    (root / "docs/BASELINE-STRATEGY-V1.md").write_text(
        report_markdown(frozen, rows, probe_results, context, teaching, gaps, scenarios), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-artifacts", type=Path, required=True)
    parser.add_argument("--guide-checkpoint", type=Path, required=True)
    parser.add_argument("--teaching-review-root", type=Path, required=True)
    parser.add_argument("--tsx", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    checkpoint, review_root, tsx = args.guide_checkpoint.resolve(), args.teaching_review_root.resolve(), args.tsx.resolve()
    replay_root = args.replay_artifacts.resolve()
    fresh = metadata(root, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "frozen-inputs.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if comparable_frozen(frozen) != comparable_frozen(fresh):
            raise RuntimeError("Resume inputs differ from the frozen baseline")
    else:
        frozen = fresh
        effective = frozen.pop("effectiveContract")
        write_json(output / "effective-contract.json", {**effective, "effectiveContractHash": frozen["effectiveContractHash"]})
        write_json(output / "probe-registry.json", {"schemaVersion": 1, "hash": frozen["probeRegistryHash"],
                                                      "probes": [probe.record() for probe in PROBES]})
        write_json(frozen_path, frozen)
    rows_path = output / "games.jsonl"
    completed = load_rows(rows_path)
    consecutive_errors = Counter()
    for policy in POLICIES:
        verify_runtime_identity(root, frozen)
        for item in frozen["schedule"]:
            key = game_key(policy, item)
            if key in completed:
                continue
            replay, details = run_game(root, checkpoint, policy, item, frozen["maxDecisions"])
            artifact_path = replay_root / policy / item["cell"]["id"] / f"{item['gameIndex']:02d}-{item['seed']}.json.gz"
            artifact = gzip_replay(artifact_path, replay)
            row = {"key": key, "policy": policy, "cell": item["cell"], "gameIndex": item["gameIndex"],
                   "seed": item["seed"], "status": replay.get("status"), "outcome": replay.get("outcome"),
                   "warnings": replay.get("warnings", []), "details": details, "replay": artifact}
            with rows_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
            completed[key] = row
            print(json.dumps({"completed": len(completed), "key": key, "status": row["status"],
                              "decisions": details["decisions"]}), flush=True)
            if row["status"] == "error":
                signature = details.get("failure") or "engine-error"
                consecutive_errors[signature] += 1
                if consecutive_errors[signature] >= 3:
                    raise RuntimeError(f"Systemic repeated game failure: {signature}")
            else:
                consecutive_errors.clear()
    rows = [completed[game_key(policy, item)] for policy in POLICIES for item in frozen["schedule"]]
    finalize(root, output, replay_root, rows, frozen, checkpoint, review_root, tsx)
    print(json.dumps({"status": "complete", "games": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
