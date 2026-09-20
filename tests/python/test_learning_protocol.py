from __future__ import annotations

import copy
from collections import defaultdict

import pytest

from ptcg_lab.learning_protocol import (collection_assignment, comparison_assignment, comparison_summary,
                                        configuration, game_seed)

ARCHETYPES = ["dragapult", "raging-bolt", "grimmsnarl", "mega-lucario", "crustle"]
DECKS = [f"{archetype}-{variant}" for archetype in ARCHETYPES for variant in ("main", "training")]
MAPPING = {deck: archetype for archetype in ARCHETYPES for deck in DECKS if deck.startswith(archetype+"-")}
TACTICAL = {"status": "measured", "positions": 5, "regressions": [], "passes": True}


def comparisons(seeds=2):
    records = []
    for index in range(len(DECKS)**2 * seeds * 2):
        assignment = comparison_assignment(index, DECKS, seeds)
        ordered = assignment["decks"] if assignment["candidateSeat"] == 0 else assignment["decks"][::-1]
        records.append({**assignment, "gameId": f"comparison-{index}", "status": "finished", "score": 1,
                        "seed": game_seed(42, "comparison", f"run:cycle:{assignment['pairNumber']}"),
                        "archetypePair": " / ".join(MAPPING[deck] for deck in ordered)})
    return records


def summarize(records, *, seeds=2, tactical=None, **kwargs):
    return comparison_summary(records, len(DECKS)**2 * seeds * 2, TACTICAL if tactical is None else tactical,
                              deck_ids=DECKS, seeds=seeds, archetypes=MAPPING, **kwargs)


def test_persistent_collection_cursor_covers_all_hundred_pairs_across_fifty_game_batches():
    cursor, batches = 0, []
    for _ in range(8):
        # This models saving/reloading the global cursor at every batch boundary.
        batch = [collection_assignment(index, DECKS, ["heuristic", "guide", "incumbent"])
                 for index in range(cursor, cursor+50)]
        batches.append(batch)
        cursor += len(batch)
    assert {a["pairKey"] for a in batches[0]}.isdisjoint(a["pairKey"] for a in batches[1])
    expected = {f"{left} / {right}" for left in DECKS for right in DECKS}
    assert {a["pairKey"] for batch in batches[:2] for a in batch} == expected
    cells = defaultdict(set)
    for batch in batches:
        for assignment in batch:
            cells[assignment["pairKey"]].add((assignment["candidateSeat"], assignment["firstPlayer"]))
    assert set(cells) == expected
    assert all(value == {(0, 0), (0, 1), (1, 0), (1, 1)} for value in cells.values())
    assert [a for batch in batches for a in batch] == [collection_assignment(i, DECKS, ["heuristic", "guide", "incumbent"]) for i in range(400)]


def test_collection_cycles_through_frozen_opponents_for_each_pair_and_cell():
    cells = defaultdict(set)
    for index in range(1200):
        assignment = collection_assignment(index, DECKS, ["heuristic", "guide", "incumbent"])
        cells[assignment["pairKey"]].add((assignment["opponent"], assignment["candidateSeat"], assignment["firstPlayer"]))
    assert len(cells) == 100 and all(len(value) == 12 for value in cells.values())


def test_comparison_default_seeds_cover_exact_ids_and_all_four_start_cells():
    records = comparisons()
    assert len(records) == 400
    cells, pairs = defaultdict(set), defaultdict(list)
    for record in records:
        cells[record["pairKey"]].add((record["candidateSeat"], record["firstPlayer"]))
        pairs[record["pairNumber"]].append(record)
    assert len(cells) == 100 and set(pairs) == set(range(200))
    assert all(value == {(0, 0), (0, 1), (1, 0), (1, 1)} for value in cells.values())
    assert all(len(value) == 2 and len({g["seed"] for g in value}) == 1 for value in pairs.values())
    result = summarize(records)
    assert result["adopted"] and result["completedGames"] == 400 and result["pairs"] == 200
    assert len(result["matchups"]) == 25 and all(value["pairs"] == 8 for value in result["matchups"].values())
    assert "not trusted" in result["scope"]
    with pytest.raises(ValueError, match="exhausted"):
        comparison_assignment(400, DECKS, 2)


def test_one_seed_diagnostic_remains_inconclusive():
    result = summarize(comparisons(1), seeds=1)
    assert not result["adopted"]
    assert any("four seat/start" in note for note in result["notes"])


def test_seed_ranges_are_deterministic_and_disjoint_for_every_phase():
    for purpose, offset in [("collection", 0), ("comparison", 1_000_000_000), ("investigation", 2_000_000_000)]:
        seeds = [game_seed(42, purpose, f"run:cycle:{index}") for index in range(200)]
        assert all(offset <= seed < offset+900_000_000 for seed in seeds)
        assert seeds == [game_seed(42, purpose, f"run:cycle:{index}") for index in range(200)]
    assert game_seed(42, "collection", "run:1") != game_seed(42, "collection", "other-run:1")
    for values in [(42, "unknown", 0), (-1, "collection", 0), (True, "collection", 0), (42, "collection", -1)]:
        with pytest.raises(ValueError):
            game_seed(*values)


@pytest.mark.parametrize("field,value", [("status", "truncated"), ("status", "error"), ("score", None),
    ("score", float("nan")), ("score", float("inf")), ("score", -1), ("score", 2), ("score", True),
    ("score", .3), ("pairNumber", 200), ("candidateSeat", 1), ("firstPlayer", 1),
    ("pairKey", "invented / pair"), ("pairIndex", 99), ("seedOffset", 1),
    ("decks", ["crustle-main", "dragapult-main"]), ("archetypePair", "unknown / unknown"),
    ("seed", 1), ("pairNumber", -1), ("pairNumber", True)])
def test_invalid_or_mismatched_game_cannot_adopt(field, value):
    records = comparisons()
    records[0][field] = value
    result = summarize(records)
    assert not result["adopted"] and result["notes"]


def test_missing_pair_and_same_size_duplicate_pair_or_game_id_cannot_pass():
    for change in (lambda records: records.pop(),
                   lambda records: records.__setitem__(slice(0, 2), copy.deepcopy(records[2:4])),
                   lambda records: records[0].update(gameId=records[2]["gameId"]),
                   lambda records: [record.update(pairNumber=1000) for record in records[:2]]):
        records = comparisons()
        change(records)
        assert not summarize(records)["adopted"]


def test_same_seed_reused_across_different_comparison_pairs_cannot_pass():
    records = comparisons()
    for record in records[2:4]:
        record["seed"] = records[0]["seed"]
    assert not summarize(records)["adopted"]


def test_aggregate_uncertainty_and_supported_matchup_regression_block_adoption():
    records = comparisons()
    for record in records:
        record["score"] = .5
    assert any("inconclusive" in note for note in summarize(records)["notes"])
    records = comparisons()
    for record in records:
        if record["archetypePair"] == "crustle / dragapult":
            record["score"] = 0
    result = summarize(records)
    assert result["lower"] > .5 and not result["adopted"]
    assert any("supported regression" in note for note in result["notes"])


@pytest.mark.parametrize("tactical", [{}, {"status": "unavailable"},
    {"status": "measured", "positions": 0, "regressions": []},
    {"status": "measured", "positions": 5},
    {"status": "measured", "positions": 5, "regressions": [{"teachingId": "regressed"}]},
    {"status": "measured", "positions": 5, "regressions": 1},
    {"status": "measured", "positions": 5, "regressions": [], "passes": False}])
def test_missing_or_regressed_tactical_evidence_blocks_adoption(tactical):
    result = summarize(comparisons(), tactical=tactical)
    assert not result["adopted"] and any("tactical" in note for note in result["notes"])


def test_legacy_count_only_summary_is_diagnostic_without_exact_schedule():
    result = comparison_summary(comparisons(), 400, TACTICAL)
    assert not result["adopted"] and any("schedule metadata" in note for note in result["notes"])


def test_bad_schedule_inputs_and_boolean_configuration_are_rejected():
    for index, decks in [(-1, DECKS), (True, DECKS), (0, []), (0, ["duplicate", "duplicate"]), (0, [""])]:
        with pytest.raises(ValueError):
            collection_assignment(index, decks, ["heuristic"])
    with pytest.raises(ValueError):
        comparison_assignment(0, DECKS, 0)
    for values in [{"seed": True}, {"comparisonSeeds": 0}, {"keepAwake": 1}, {"invented": 1}]:
        with pytest.raises(ValueError):
            configuration(values)
