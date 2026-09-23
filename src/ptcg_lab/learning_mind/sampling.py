from __future__ import annotations

from collections import Counter


def select_stratified_rows(rows: list[dict], limit: int) -> list[dict]:
    """Deterministically sample across target deck, opponent, stage, split, and source game."""
    remaining = list(rows)
    selected: list[dict] = []
    counts: dict[str, Counter] = {
        name: Counter() for name in (
            "targetDeck", "opponentArchetype", "positionStage", "split", "sourceGameId",
        )
    }
    while remaining and len(selected) < limit:
        row = min(remaining, key=lambda item: (
            counts["targetDeck"][str(item.get("targetDeck", "unknown"))],
            counts["opponentArchetype"][str(item.get("opponentArchetype", "unknown"))],
            counts["positionStage"][str(item.get("positionStage", "unknown"))],
            counts["split"][str(item.get("split", "unknown"))],
            counts["sourceGameId"][str(item.get("sourceGameId", "unknown"))],
            str(item.get("positionHash", "")),
        ))
        selected.append(row)
        remaining.remove(row)
        for name, counter in counts.items():
            counter[str(row.get(name, "unknown"))] += 1
    return selected
