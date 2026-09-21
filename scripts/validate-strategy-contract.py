#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_lab.strategy_contract import validate_strategy_contract, validate_strategy_revision  # noqa: E402


if __name__ == "__main__":
    result = validate_strategy_contract(ROOT)
    revision = validate_strategy_revision(ROOT)
    print(
        "Validated strategy-contract-v1: "
        f"{len(result['playbooks'])} specialists, "
        f"{sum(len(item['principles']) for item in result['playbooks'].values())} principles, "
        f"{sum(len(item['matchups']) for item in result['playbooks'].values())} matchup perspectives."
    )
    print(
        "Validated approved strategy-contract-v1.1: "
        f"{len(revision['playbooks'])} specialist overlays, "
        f"{sum(len(item['principlePatches']) for item in revision['playbooks'].values())} principle patches, "
        f"{sum(len(item['matchupPatches']) for item in revision['playbooks'].values())} matchup patches."
    )
